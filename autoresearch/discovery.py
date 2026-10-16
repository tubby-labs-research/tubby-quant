"""Bounded retrieval of article abstracts and explicit offline source fixtures."""
from __future__ import annotations

import hashlib
import json
import os
import re
import ssl
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


class DiscoveryError(ValueError):
    """A source could not be retrieved or validated; retry or repair configuration."""


def _clean(value):
    return " ".join(str(value or "").split())


def _canonical_arxiv(url):
    match = re.search(r"arxiv\.org/(?:abs|pdf)/(.+?)(?:\.pdf)?$", url)
    if not match:
        raise DiscoveryError("arXiv entry has no canonical article URL")
    article = re.sub(r"v\d+$", "", match.group(1))
    return "https://arxiv.org/abs/" + article


def _source(title, url, text, level, timestamp, source_id=None):
    if not all(isinstance(v, str) and v.strip() for v in (title, url, text)):
        raise DiscoveryError("source title, URL and text must be nonempty strings")
    content = {"title": _clean(title), "url": url, "text": _clean(text), "evidence_level": level}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:24]
    return {"id": source_id or "source_" + digest, **content, "retrieved_at": timestamp}


def parse_arxiv(data, retrieved_at=None):
    """Parse Atom abstracts. IDs ignore version suffixes but bind the actual text."""
    timestamp = retrieved_at or datetime.now(timezone.utc).isoformat()
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise DiscoveryError("arXiv returned invalid Atom XML") from exc
    ns = {"a": "http://www.w3.org/2005/Atom"}
    if root.tag != "{http://www.w3.org/2005/Atom}feed":
        raise DiscoveryError("arXiv response is not an Atom feed")
    records = {}
    for entry in root.findall("a:entry", ns):
        url = _canonical_arxiv(_clean(entry.findtext("a:id", "", ns)))
        source = _source(_clean(entry.findtext("a:title", "", ns)), url,
                         _clean(entry.findtext("a:summary", "", ns)),
                         "abstract", timestamp)
        # One current article per fetch, even if a feed repeats versioned entries.
        records[url] = source
    return list(records.values())


def _fetch(url, timeout, max_bytes):
    request = Request(url, headers={"User-Agent": "Quant-autoresearch-draft/1", "Accept": "application/atom+xml"})
    try:
        certificate_file = os.environ.get("SSL_CERT_FILE")
        if certificate_file:
            context = ssl.create_default_context(cafile=certificate_file)
        else:
            try:
                import certifi
            except ImportError:
                context = ssl.create_default_context()
            else:
                context = ssl.create_default_context(cafile=certifi.where())
        with urlopen(request, timeout=timeout, context=context) as response:
            if urlparse(response.geturl()).hostname not in {"export.arxiv.org", "arxiv.org"}:
                raise DiscoveryError("unexpected arXiv redirect")
            data = response.read(max_bytes + 1)
    except DiscoveryError:
        raise
    except Exception as exc:
        # Avoid exception URLs or headers containing accidental secret material.
        raise DiscoveryError("arXiv retrieval failed") from exc
    if len(data) > max_bytes:
        raise DiscoveryError("arXiv response exceeds configured byte limit")
    return data


def discover(config):
    config = config.get("discovery", config)
    limit = config.get("max_results", 10)
    timeout = config.get("timeout_seconds", 10)
    max_bytes = config.get("max_response_bytes", 1_000_000)
    if type(limit) is not int or not 1 <= limit <= 100:
        raise DiscoveryError("max_results must be between 1 and 100")
    if not isinstance(timeout, (int, float)) or not 0 < timeout <= 120:
        raise DiscoveryError("timeout_seconds must be between 0 and 120")
    if type(max_bytes) is not int or not 1 <= max_bytes <= 10_000_000:
        raise DiscoveryError("max_response_bytes must be between 1 and 10000000")
    timestamp = datetime.now(timezone.utc).isoformat()
    evidence_level = config.get("snapshot_evidence_level", "fixture")
    if evidence_level not in {"fixture", "excerpt", "abstract"}:
        raise DiscoveryError("invalid snapshot evidence level")
    fixtures = config.get("fixture_sources", [])
    if "fixture_path" in config:
        path = Path(config["fixture_path"])
        try:
            if path.stat().st_size > max_bytes:
                raise DiscoveryError("source fixture exceeds configured byte limit")
            fixtures = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DiscoveryError("source fixture cannot be read") from exc
    if not isinstance(fixtures, list):
        raise DiscoveryError("fixture sources must be a JSON array")
    records = {}
    for item in fixtures[:limit]:
        if not isinstance(item, dict):
            raise DiscoveryError("source fixture entries must be objects")
        supplied_id = item.get("id")
        if supplied_id is not None and (not isinstance(supplied_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", supplied_id)):
            raise DiscoveryError("fixture source ID must be safe")
        record = _source(item.get("title"), item.get("url"), item.get("text"),
                         evidence_level, item.get("retrieved_at", timestamp), supplied_id)
        if record["id"] in records and records[record["id"]] != record:
            raise DiscoveryError("fixture source ID conflict")
        records[record["id"]] = record
    queries = config.get("arxiv_queries", [])
    if not isinstance(queries, list) or len(queries) > 10 or any(not isinstance(q, str) or not q.strip() or len(q) > 2000 for q in queries):
        raise DiscoveryError("arxiv_queries must contain at most ten nonempty query strings")
    if queries and config.get("network_enabled") is not True:
        raise DiscoveryError("arXiv retrieval requires network_enabled=true")
    interval = config.get("query_interval_seconds", 3)
    if not isinstance(interval, (int, float)) or isinstance(interval, bool) or not 3 <= interval <= 60:
        raise DiscoveryError("query_interval_seconds must be between 3 and 60")
    for query_index, query in enumerate(queries):
        if query_index:
            # arXiv asks clients to leave at least three seconds between requests.
            # Waiting after the previous response is conservative and easily mockable.
            time.sleep(interval)
        params = urlencode({"search_query": query, "start": 0, "max_results": limit,
                            "sortBy": "submittedDate", "sortOrder": "descending"})
        sources = parse_arxiv(_fetch("https://export.arxiv.org/api/query?" + params, timeout, max_bytes), timestamp)
        for source in sources:
            records[source["id"]] = source
    # Dynamic overlapping searches retain one article identity, regardless of version.
    by_url = {}
    for source in records.values():
        by_url[source["url"]] = source
    return list(by_url.values())[:limit]
