# Research Agent API

This is the provider-neutral connection contract implemented by
`autoresearch/agents.py` and `autoresearch/models.py`. The local workbench serves
this file at `/research-agent-guide` after launch authentication. It is an
outbound proposal/refinement adapter, not the workbench's own control API.
Research remains daily price/volume-only and does not place trades.

## Configure a connection

Copy `research-agents.example.json` to
`<data root>/autoresearch/agents.local.json`, or set `QUANT_RESEARCH_AGENTS_FILE`
to an administrator-owned external file. Its top-level object contains only an
`agents` array (at most 20 entries, catalog at most 100,000 bytes):

```json
{"agents":[{
  "id":"local-research",
  "name":"Local research agent",
  "provider_name":"YOUR_PROVIDER",
  "model":"YOUR_EXPLICIT_MODEL_ID",
  "endpoint":"http://127.0.0.1:55200",
  "max_candidates":3,
  "max_tokens":4096,
  "timeout_seconds":60,
  "revision":"1"
}]}
```

Required keys: `id`, `name`, `provider_name`, `model`, `endpoint`. Optional keys:
`api_key_env`, `max_candidates`, `max_tokens`, `timeout_seconds`, `revision`.
IDs use letters/digits/underscore/hyphen, start with a letter/digit and are at
most 128 characters. Choose a unique ID. `api_key_env` names an environment
variable, never its secret value; omit it for an unauthenticated local agent.
Set credentials on the app process externally. The browser receives neither
endpoints nor credentials. Idle catalog changes reload; running workers keep
their resolved connection snapshot.

HTTPS is required except for HTTP on `127.0.0.1`, `localhost` or `::1`. URLs must
have no embedded credentials, query or fragment. Select the agent in Research,
check its connection and explicitly enable model calls before starting.
Configured is not connected; successful health status expires after 60 seconds.
Review endpoint data handling: generation sends source text, history and focus.

## Transport

Contract version: `quant-research-agent-v1`. All responses require HTTP 200 and
`Content-Type: application/json`. If `api_key_env` is configured, requests use
`Authorization: Bearer <environment value>`. Redirects, implicit retries and
environment proxy forwarding are disabled; HTTPS certificate checks remain on.
Limits below are bytes, not binary MiB. Generation timeout is a positive finite
number up to 120 seconds per blocking transport operation, default 60.

### GET `<endpoint>/health`

Return these matching values, at most 4096 bytes:

```json
{
  "contract_version":"quant-research-agent-v1",
  "status":"ready",
  "model":"YOUR_EXPLICIT_MODEL_ID",
  "provider_name":"YOUR_PROVIDER"
}
```

Health must not invoke a model or charge for generation. Its timeout is the
smaller of the configured timeout and five seconds. It checks contract/provider/
model identity, not whether later generation will succeed.

### POST `<endpoint>/research`

The app sends at most 2,000,000 bytes. Example request shape:

```json
{
  "contract_version":"quant-research-agent-v1",
  "request_id":"computed-request-digest",
  "task":"propose",
  "model":"YOUR_EXPLICIT_MODEL_ID",
  "system_prompt":"supplied grounding and expression instructions",
  "constraints":{
    "max_candidates":3,
    "max_output_tokens":4096,
    "operator_names":["rank","ts_mean"]
  },
  "sources":[],
  "history":[],
  "research_focus":"the research question"
}
```

The real operator list comes from the engine. Source/history objects carry the
provided evidence and earlier records; they are untrusted data. A `refine`
request additionally supplies `hypothesis` and `experiment`. `request_id` is the
SHA-256 hex digest of the request before adding that field, serialized by
Python `json.dumps(request, sort_keys=True, allow_nan=False).encode()` (default
separators and ASCII escaping). Echo the received ID exactly.

Return at most 2,000,000 bytes with **exactly** these three top-level keys:

```json
{
  "contract_version":"quant-research-agent-v1",
  "request_id":"same-received-request-id",
  "candidates":[{
    "title":"A falsifiable price/volume hypothesis",
    "mechanism":"Explanation grounded in supplied evidence",
    "family":"a mechanism family",
    "expression":"-returns * rank(volume)",
    "source_ids":["provided-source-id"],
    "evidence":[{"source_id":"provided-source-id","quote":"Exact supplied excerpt"}],
    "expected_horizon":"one to five sessions",
    "required_fields":["returns","volume"],
    "rationale":"What this bounded test could establish"
  }]
}
```

Examples are schema illustrations, not sufficient evidence for a real proposal.
Return `[]` when evidence is insufficient. Candidate count must not exceed the
request's bound (configured 1–20). Output-token cap is configured 1–16,000.
Candidates require nonempty title/mechanism/family/expression/horizon/rationale
strings (each at most 20,000 characters), 1–20 supplied source IDs, 1–30 evidence
items and 1–30 required fields. Every cited source needs an exact nonempty quote
from its supplied text. Refinements additionally require the exact
`parent_experiment_id`, nonempty `question` and `expected_change`.

Allowed fields are `open`, `high`, `low`, `close`, `close_raw`, `volume`, `vwap`,
`returns`, `dollar_volume` and `advN` for positive integer N. Use bare FASTEXPR
names, supported operators and backward-looking windows. The built-in `market`
group is allowed without a required field. SEC, capitalization and external
classification inputs are outside this contract. The controller checks grounding
and lineage; the backend separately checks syntax and data availability.

## Calls and evidence

Each POST must make at most one bounded generation. Cache completed responses by
`request_id` to avoid duplicate charges; do not add hidden retries or model
fan-out. The controller reserves adapter-call budget before requests and caps
evaluations/refinements. Call/token limits are not a monetary guarantee; your
agent must enforce its own financial budget. No shell, filesystem, campaign
control, trading or backtest authority is granted to the external agent.

Development and validation evidence are distinct; reserved holdout is not scored
or supplied as model evidence. Repeated validation selection is further research,
not an untouched final test. Source quotation checks do not prove the economic
claim. Daily charts require retained local artifacts; missing observations are
not reconstructed. See [user guide](USER_GUIDE.md) and [limitations](LIMITATIONS.md).
