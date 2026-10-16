"""Pure descriptive SEC context aliases; no catalog access or identity rewrite.

Standard aliases extend sec_names' unchanged grammar with sorted context labels:
``sec_<base>[_q|_<N>q][__unit][__taxonomy]__<context>...``. LegalEntity and
Scenario use ``entity_`` and ``scenario_``; other dimensions retain their full
axis/member words under ``context_``. Co-registrants use ``coreg_``. Only a
byte-for-byte identical LegalEntity member/coreg pair shares a display label.
Printable ASCII punctuation becomes word separators; there is no truncation,
transliteration, invented economic synonym, checksum, or opaque fallback.

An exact accession taxonomy can instead use externally supplied source SUB
metadata: ``[__unit]__issuer_<name>__form_<form>__filed_<yyyy_mm_dd>`` followed
by contexts. Original tag words are mandatory for these filing-scoped aliases.
The metadata's adsh must equal the exact taxonomy. This is a display hook, not
proof of classification, source integrity, PIT eligibility or unique identity.
Additional readable units use ``__unit_<words>``; additional meaningful
taxonomies use ``__taxonomy_<words>_family`` or ``_edition``. These labels are
lookup hints, not guessed exact selectors. Opaque contexts, opaque taxonomies
and absent accession filing metadata fail closed.

Integration contract (the registry/resolver owns this work):
* candidate_for_name supplies hints, NEVER an exact six-key recipe. Restore tag
  case only through a bounded indexed tag lookup, including case collisions.
* Probe recipes by actual tag + qtrs, adding exact unit/taxonomy when provided.
  An explicit unit_hint/taxonomy_hint needs candidate comparison, never guessed
  raw spelling. Cache bounded selector groups within a request for page reuse.
  Filing aliases omit taxonomy: bound the tag-indexed versions, join each exact
  accession to filings_adsh, then compare the supplied name/form/date metadata.
  No JSON issuer search, full-catalog scan or new dictionary is required here.
* Recompute aliases with accepts_name for each stored exact candidate. Reject
  multiple distinct six-key specs, even when only case, punctuation, segment
  order or redundant coreg differs. A truncated/budget-exhausted candidate set
  cannot certify uniqueness. Do this before displaying AND resolving aliases.
* Preserve the selected exact spec and original storage name through sealed
  execution verification. Context labels and filing dates do not select facts
  or change the selected-field adapter's historical availability rules.
"""
from __future__ import annotations

from datetime import date
import re

from . import sec_names


_KEYS = frozenset(('tag', 'taxonomy', 'unit', 'qtrs', 'segments', 'coreg'))
_WORDS = re.compile(r'[a-z0-9]+(?:_[a-z0-9]+)*\Z', re.ASCII)
_ACCESSION = re.compile(r'[0-9]{10}-[0-9]{2}-[0-9]{6}', re.ASCII)
_UUID = re.compile(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z')
_CONTEXT_PREFIXES = ('entity_', 'scenario_', 'context_', 'coreg_')


def _words(value):
    """Preserve complete readable source words; decline unsupported/opaque text."""
    if (not isinstance(value, str) or not value
            or any(ord(char) < 32 or ord(char) > 126 for char in value)
            or any(char in '<>\\' for char in value)
            or _ACCESSION.search(value) or _UUID.fullmatch(value)
            or re.fullmatch(r'[0-9a-fA-F]{20,}', value)):
        return None
    value = re.sub(r'([A-Z])([A-Z][a-z])', r'\1_\2', value)
    value = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', value)
    value = re.sub(r'([A-Za-z])([0-9])', r'\1_\2', value)
    value = re.sub(r'([0-9])([A-Za-z])', r'\1_\2', value)
    result = re.sub(r'[^a-z0-9]+', '_', value.lower()).strip('_')
    return result if result and re.search(r'[a-z]', result) else None


def _valid_spec(spec):
    return (isinstance(spec, dict) and set(spec) == _KEYS
            and all(isinstance(spec[key], str) for key in _KEYS - {'qtrs'})
            and type(spec['qtrs']) is int and spec['qtrs'] >= 0)


def _context_labels(spec):
    """Accept explicit Axis=Member pairs separated by semicolons, not raw blobs.

    A single trailing semicolon is supported. Pair order is display-only. Neither
    Axis/Member suffixes nor namespace words are stripped. Duplicate axes (even
    normalized case/punctuation variants) are deliberately unsupported.
    """
    segments, coreg = spec['segments'], spec['coreg']
    labels, axes, legal_entity = [], set(), None
    if segments:
        pairs = segments.split(';')
        if pairs[-1] == '':
            pairs.pop()
        for pair in pairs:
            if pair.count('=') != 1:
                return None
            axis, member = pair.split('=')
            axis_words, member_words = _words(axis), _words(member)
            if axis_words is None or member_words is None or axis_words in axes:
                return None
            axes.add(axis_words)
            if axis == 'LegalEntity':
                label, legal_entity = 'entity_' + member_words, member
            elif axis == 'Scenario':
                label = 'scenario_' + member_words
            else:
                label = 'context_' + axis_words + '_' + member_words
            labels.append(label)
    if coreg and coreg != legal_entity:
        coreg_words = _words(coreg)
        if coreg_words is None:
            return None
        labels.append('coreg_' + coreg_words)
    # Equal normalized labels are not silently deduplicated.
    if len(set(labels)) != len(labels):
        return None
    return tuple(sorted(labels))


def _filed_date(value):
    if not isinstance(value, str):
        return None
    if re.fullmatch(r'[0-9]{8}', value, re.ASCII):
        value = value[:4] + '-' + value[4:6] + '-' + value[6:]
    if not re.fullmatch(r'[1-9][0-9]{3}-[0-9]{2}-[0-9]{2}', value, re.ASCII):
        return None
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        return None


def _filing_labels(spec, filing):
    if (not isinstance(filing, dict) or filing.get('adsh') != spec['taxonomy']
            or not _ACCESSION.fullmatch(spec['taxonomy'])):
        return None
    issuer = _words(filing.get('name'))
    form, filed = filing.get('form'), _filed_date(filing.get('filed'))
    if (issuer is None or not isinstance(form, str)
            or not re.fullmatch(r'[A-Z0-9]+(?:[-/][A-Z0-9]+)*', form, re.ASCII)
            or filed is None):
        return None
    form = form.lower().replace('-', '_').replace('/', '_')
    return ('issuer_' + issuer, 'form_' + form, 'filed_' + filed.replace('-', '_'))


def _unit_label(unit):
    # Retain operator meaning and all source words for non-vocabulary units.
    # Punctuation/case can still collide: the registry confirms exact selectors.
    parts = re.split(r'([/*])', unit)
    words = []
    for part in parts:
        if part in ('/', '*'):
            words.append('per' if part == '/' else 'times')
        else:
            label = _words(part)
            if label is None:
                return None
            words.append(label)
    return 'unit_' + '_'.join(words)


def _taxonomy_label(taxonomy):
    words = _words(taxonomy)
    if words is None:
        return None
    scope = 'family' if taxonomy.endswith('/') else 'edition'
    return 'taxonomy_' + words + '_' + scope


def _alias_for_spec(spec, *, curated, kind, filing):
    if not _valid_spec(spec) or kind not in ('standard', 'custom'):
        return None
    labels = _context_labels(spec)
    if labels is None:
        return None
    blank = dict(spec, segments='', coreg='')
    unit_probe = sec_names.generic_alias_for_spec(dict(blank, tag='Assets', taxonomy='us-gaap/'))
    if unit_probe is None:
        unit = _unit_label(spec['unit'])
        if unit is None:
            return None
        unit_labels = (unit,)
    else:
        unit_labels = tuple(unit_probe.split('__')[1:])
    taxonomy_probe = sec_names.generic_alias_for_spec(dict(blank, tag='Assets', unit='USD'))
    filing_scoped = bool(_ACCESSION.fullmatch(spec['taxonomy']))
    if filing_scoped:
        filing_labels = _filing_labels(spec, filing)
        if filing_labels is None:
            return None
        taxonomy_labels = filing_labels
    elif taxonomy_probe is None:
        taxonomy = _taxonomy_label(spec['taxonomy'])
        if taxonomy is None:
            return None
        taxonomy_labels = (taxonomy,)
    else:
        if kind != 'standard':
            return None
        taxonomy_labels = tuple(taxonomy_probe.split('__')[1:])
    generator = sec_names.preferred_name if curated and not filing_scoped else sec_names.generic_alias_for_spec
    # Reuse the tag/duration grammar. Unknown scopes use a generic surrogate to
    # avoid curated US-GAAP meanings; the surrogate never escapes as a selector.
    head = generator(dict(blank, unit='USD', taxonomy=spec['taxonomy'] if taxonomy_probe else 'ifrs-full/'))
    if head is None:
        return None
    body = head.split('__')[0]
    name = body + ''.join('__' + label for label in unit_labels + taxonomy_labels + labels)
    return name if candidate_for_name(name) is not None else None


def preferred_name(spec, *, kind='standard', filing=None) -> str | None:
    """Return a descriptive candidate; caller certifies source kind/uniqueness.

    filing may be a source SUB payload (extra provenance keys are allowed).
    It must describe the taxonomy's defining accession, not a later fact filing.
    Ordinary supported standard recipes retain sec_names' preferred spelling.
    """
    return _alias_for_spec(spec, curated=True, kind=kind, filing=filing)


def generic_alias_for_spec(spec, *, kind='standard', filing=None) -> str | None:
    """Complete original-tag alternate, with all the same identity safeguards."""
    return _alias_for_spec(spec, curated=False, kind=kind, filing=filing)


def _is_context_label(label):
    for prefix in _CONTEXT_PREFIXES:
        if label.startswith(prefix):
            words = label[len(prefix):]
            return bool(_WORDS.fullmatch(words) and re.search(r'[a-z]', words))
    return False


def candidate_for_name(name) -> dict | None:
    """Parse lookup hints, without reconstructing raw context/taxonomy identity.

    Standard vocabulary: tag (casing hint), taxonomy, unit, qtrs, context_labels.
    Explicit unit_hint/taxonomy_hint REPLACE the corresponding exact key.
    Filing: tag (casing hint), unit or unit_hint, qtrs, context_labels, filing_hint containing
    issuer_slug, form_slug and ISO filed date. No taxonomy key in that case.
    Neither shape has segments/coreg: normalization cannot invert exact strings.
    A plausible tag may parse without existing; this is not an availability test.
    """
    if (not isinstance(name, str) or not name.startswith('sec_')
            or name != name.lower()):
        return None
    parts = name.split('__')
    filing_start = next((i for i, part in enumerate(parts) if part.startswith('issuer_')), None)
    context_start = next((i for i, part in enumerate(parts) if _is_context_label(part)), len(parts))
    if filing_start is not None:
        trio = parts[filing_start:filing_start + 3]
        if (len(trio) != 3 or not trio[1].startswith('form_')
                or not trio[2].startswith('filed_') or context_start < filing_start):
            return None
        issuer, form = trio[0][7:], trio[1][5:]
        filed = _filed_date(trio[2][6:].replace('_', '-'))
        if (not _WORDS.fullmatch(issuer) or not re.search(r'[a-z]', issuer)
                or not _WORDS.fullmatch(form) or filed is None):
            return None
        # Parsing with IFRS preserves generic tag words, including curated-word
        # coincidences such as custom Inventory. No case reconstruction is exact.
        head = parts[:filing_start]
        labels = parts[filing_start + 3:]
        filing_hint = dict(issuer_slug=issuer, form_slug=form, filed=filed)
    else:
        head = parts[:context_start]
        labels = parts[context_start:]
        filing_hint = None
    unit_hint = taxonomy_hint = None
    if len(head) > 1 and head[1].startswith('unit_'):
        unit_hint = head[1][5:]
        if not _WORDS.fullmatch(unit_hint) or not re.search(r'[a-z]', unit_hint):
            return None
        head = head[:1] + head[2:]
    if len(head) > 1 and head[-1].startswith('taxonomy_'):
        taxonomy_hint = head[-1][9:]
        if (filing_hint is not None or not _WORDS.fullmatch(taxonomy_hint)
                or not re.fullmatch(r'.+_(family|edition)', taxonomy_hint)
                or not re.search(r'[a-z]', taxonomy_hint.rsplit('_', 1)[0])):
            return None
        head = head[:-1]
    surrogate = filing_hint is not None or taxonomy_hint is not None
    base = sec_names.spec_for_name('__'.join(head) + ('__ifrs_full' if surrogate else ''))
    if base is None:
        return None
    result = {key: base[key] for key in ('tag', 'qtrs')}
    if unit_hint is not None:
        # An explicit unit hint cannot appear alongside a second real unit.
        if base['unit'] != 'USD':
            return None
        result['unit_hint'] = unit_hint
    else:
        result['unit'] = base['unit']
    if filing_hint is not None:
        result['filing_hint'] = filing_hint
    elif taxonomy_hint is not None:
        result['taxonomy_hint'] = taxonomy_hint
    else:
        result['taxonomy'] = base['taxonomy']
    if (any(not _is_context_label(label) for label in labels)
            or labels != sorted(set(labels))):
        return None
    # These roles occur at most once in a supported exact context.
    for prefix in ('entity_', 'scenario_', 'coreg_'):
        if sum(label.startswith(prefix) for label in labels) > 1:
            return None
    result['context_labels'] = tuple(labels)
    return result


def accepts_name(name, spec, *, kind='standard', filing=None) -> bool:
    """Filter one exact stored candidate; NOT a uniqueness or integrity proof."""
    if candidate_for_name(name) is None:
        return False
    return (name == preferred_name(spec, kind=kind, filing=filing)
            or name == generic_alias_for_spec(spec, kind=kind, filing=filing))
