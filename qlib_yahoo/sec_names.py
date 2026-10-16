"""Pure readable SEC aliases, independent of immutable v2 storage names.

Callers must verify stored ``kind == 'standard'`` and recipe existence by
indexed lookup: syntactic recognition is neither taxonomy membership nor data
availability. Generic parsing guesses PascalCase and can lose acronym casing.
Resolve actual tags with the catalog's indexed case-insensitive compact-word
lookup, filter with ``accepts_name``, and fail closed on multiple matching tags
before resolving OR advertising an alias. Nothing here changes storage policy.
Curated names are Brain-style inspired, not official Brain identifiers/parity.

Grammar: sec_<base>[_q|_<N>q][__<unit>][__<taxonomy>]. Instant has no
duration suffix; N >= 2 denotes the reported duration, never calculated TTM.
USD and us-gaap/ are implicit. Other supported units/taxonomies are explicit;
suffix order is fixed. Every alias is lowercase. Only us-gaap/ uses curated
bases; generic bases split the complete original tag into snake_case words.
Original-tag snake_case alternates are accepted for curated tags too. Contexts
are never discarded. No exact generic-tag roundtrip is claimed before lookup.
"""
from __future__ import annotations

import re
from types import MappingProxyType


_KEYS = frozenset(('tag', 'taxonomy', 'unit', 'qtrs', 'segments', 'coreg'))
_TAG = re.compile(r'[A-Z][A-Za-z0-9]*\Z', re.ASCII)
_BASE = re.compile(r'[a-z]+(?:_[a-z]+)*\Z', re.ASCII)
_GENERIC_BASE = re.compile(r'[a-z][a-z0-9]*(?:_[a-z0-9]+)*\Z', re.ASCII)
_DURATION = re.compile(r'_(q|[0-9]+q)\Z', re.ASCII)
_YEAR = r'[1-9][0-9]{3}'

# Each base identifies exactly one tag. In particular, these do not include
# restricted cash/investments, gross inventory or noncontrolling-interest equity.
_CURATED_ROWS = (
    # Exact tag, reviewed base, default reported qtrs, default unit.
    ('Assets', 'assets', 0, 'USD'),
    ('AssetsCurrent', 'assets_curr', 0, 'USD'),
    ('InventoryNet', 'inventory', 0, 'USD'),
    ('Liabilities', 'liabilities', 0, 'USD'),
    ('LiabilitiesCurrent', 'liabilities_curr', 0, 'USD'),
    ('CashAndCashEquivalentsAtCarryingValue', 'cash_equivalents', 0, 'USD'),
    ('StockholdersEquity', 'equity', 0, 'USD'),
    ('NetIncomeLoss', 'net_income', 1, 'USD'),
    ('NetCashProvidedByUsedInOperatingActivities', 'cashflow_op', 1, 'USD'),
    ('NetCashProvidedByUsedInInvestingActivities', 'cashflow_investing', 1, 'USD'),
    ('NetCashProvidedByUsedInFinancingActivities', 'cashflow_financing', 1, 'USD'),
    ('OperatingIncomeLoss', 'operating_income', 1, 'USD'),
    ('GrossProfit', 'gross_profit', 1, 'USD'),
    ('Revenues', 'revenues', 1, 'USD'),
    ('ResearchAndDevelopmentExpense', 'rd_expense', 1, 'USD'),
    ('SellingGeneralAndAdministrativeExpense', 'sga_expense', 1, 'USD'),
    ('IncomeTaxExpenseBenefit', 'income_tax', 1, 'USD'),
    ('EarningsPerShareBasic', 'eps_basic', 1, 'USD/shares'),
    ('EarningsPerShareDiluted', 'eps_diluted', 1, 'USD/shares'),
    ('RetainedEarningsAccumulatedDeficit', 'retained_earnings', 0, 'USD'),
    ('AccountsPayableCurrent', 'accounts_payable_curr', 0, 'USD'),
    ('LongTermDebtCurrent', 'debt_lt_current', 0, 'USD'),
    ('LongTermDebtNoncurrent', 'debt_lt_noncurrent', 0, 'USD'),
)


def _curated_maps(pairs):
    """Fail closed on collisions or bases ambiguous with the suffix grammar."""
    by_tag, by_base = {}, {}
    for tag, base in pairs:
        if (not isinstance(tag, str) or not _TAG.fullmatch(tag)
                or not isinstance(base, str) or not _BASE.fullmatch(base)
                or _DURATION.search('_' + base)):
            raise ValueError('Invalid or ambiguous curated SEC name')
        if tag in by_tag or base in by_base:
            raise ValueError('Curated SEC name collision')
        by_tag[tag], by_base[base] = base, tag
    return MappingProxyType(by_tag), MappingProxyType(by_base)


CURATED_NAMES, _CURATED_TAGS = _curated_maps((row[:2] for row in _CURATED_ROWS))

# Explicit bounded unit vocabulary, not a heuristic accepting nonsense codes.
# Unsupported units retain their storage name. No currency conversion is implied.
_CURRENCIES = ('EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'CNY', 'HKD',
               'NZD', 'SEK', 'NOK', 'DKK', 'SGD', 'INR', 'KRW', 'BRL',
               'MXN', 'ZAR', 'TWD')
_UNIT_SUFFIX = MappingProxyType({
    'USD': '', 'shares': 'shares', 'USD/shares': 'usd_per_share',
    **{currency: currency.lower() for currency in _CURRENCIES},
})
_SUFFIX_UNIT = MappingProxyType({suffix: unit for unit, suffix in _UNIT_SUFFIX.items()
                                if suffix})
if len(_SUFFIX_UNIT) != len(_UNIT_SUFFIX) - 1:
    raise ValueError('SEC unit suffix collision')


def _taxonomy_suffix(taxonomy):
    if taxonomy == 'us-gaap/':
        return ''
    if taxonomy == 'ifrs-full/':
        return 'ifrs_full'
    if taxonomy == 'ifrs/':
        return 'ifrs'
    if re.fullmatch(r'us-gaap/' + _YEAR, taxonomy, re.ASCII):
        return 'us_gaap_' + taxonomy.rsplit('/', 1)[1]
    if re.fullmatch(r'ifrs-full/' + _YEAR, taxonomy, re.ASCII):
        return 'ifrs_full_' + taxonomy.rsplit('/', 1)[1]
    if re.fullmatch(r'ifrs/' + _YEAR, taxonomy, re.ASCII):
        return 'ifrs_' + taxonomy.rsplit('/', 1)[1]
    return None


def _taxonomy_for_suffix(suffix):
    if suffix == 'ifrs_full':
        return 'ifrs-full/'
    if suffix == 'ifrs':
        return 'ifrs/'
    if re.fullmatch(r'us_gaap_' + _YEAR, suffix, re.ASCII):
        return 'us-gaap/' + suffix.rsplit('_', 1)[1]
    if re.fullmatch(r'ifrs_full_' + _YEAR, suffix, re.ASCII):
        return 'ifrs-full/' + suffix.rsplit('_', 1)[1]
    if re.fullmatch(r'ifrs_' + _YEAR, suffix, re.ASCII):
        return 'ifrs/' + suffix.rsplit('_', 1)[1]
    return None


def _snake_tag(tag):
    return display_label(tag).lower().replace(' ', '_')


def _alias_for_spec(spec, *, curated):
    if not isinstance(spec, dict) or set(spec) != _KEYS:
        return None
    if any(not isinstance(spec[key], str) for key in _KEYS - {'qtrs'}):
        return None
    if (not _TAG.fullmatch(spec['tag']) or spec['segments'] != '' or spec['coreg'] != ''
            or type(spec['qtrs']) is not int or spec['qtrs'] < 0):
        return None
    taxonomy = _taxonomy_suffix(spec['taxonomy'])
    unit = _UNIT_SUFFIX.get(spec['unit'])
    if taxonomy is None or unit is None:
        return None
    base = _snake_tag(spec['tag'])
    if curated and spec['taxonomy'] == 'us-gaap/':
        base = CURATED_NAMES.get(spec['tag'], base)
    # A final duration-shaped word cannot also be part of a generic tag.
    if not _GENERIC_BASE.fullmatch(base) or _DURATION.search(base):
        return None
    # Curated bases are reserved only in the taxonomy where they are used.
    # E.g. Inventory must not be silently reinterpreted as InventoryNet.
    if (spec['taxonomy'] == 'us-gaap/' and base in _CURATED_TAGS
            and _CURATED_TAGS[base] != spec['tag']):
        return None
    try:
        duration = '' if spec['qtrs'] == 0 else ('_q' if spec['qtrs'] == 1 else f"_{spec['qtrs']}q")
    except ValueError:  # Python's oversized integer-to-string protection.
        return None
    name = 'sec_' + base + duration
    if unit:
        name += '__' + unit
    if taxonomy:
        name += '__' + taxonomy
    # Guard suffix identity, without treating the generic PascalCase guess as an
    # exact tag. Canonical tag identity/uniqueness must be established by lookup.
    candidate = spec_for_name(name)
    if candidate is None or any(candidate[key] != spec[key] for key in _KEYS - {'tag'}):
        return None
    return name


def preferred_name(spec) -> str | None:
    """Return a lowercase alias candidate for a supported six-key recipe.

    Callers establish standard kind and canonical-tag uniqueness using indexed
    lookup before displaying this alias. Missing/null/advanced contexts, extra
    metadata, and collisions with another reviewed base fail closed.
    """
    return _alias_for_spec(spec, curated=True)


def generic_alias_for_spec(spec) -> str | None:
    """Original-tag snake_case alternate, with the same selector/collision guards."""
    return _alias_for_spec(spec, curated=False)


def accepts_name(name, spec) -> bool:
    """Match a name against a canonically resolved exact recipe.

    Matches either its preferred alias or original-tag snake_case alternate,
    including exact suffix selectors. Multiple actual tags may still match an
    acronym alias: the caller must reject that ambiguity, not choose one.
    """
    if spec_for_name(name) is None:
        return False
    return name == preferred_name(spec) or name == generic_alias_for_spec(spec)


def spec_for_name(name) -> dict | None:
    """Parse a lowercase alias into a reviewed exact spec or a guessed selector.

    Generic PascalCase is ONLY a lookup candidate; acronym capitalization is
    lost. Caller resolves canonical tags, filters with accepts_name, and checks
    uniqueness/standard kind/existence. Syntactically valid unknown tags can
    parse successfully; this function cannot establish their existence.
    """
    if (not isinstance(name, str) or not name.startswith('sec_')
            or name != name.lower()):
        return None
    parts = name[4:].split('__')
    if len(parts) > 3 or any(not part for part in parts):
        return None
    body, unit, taxonomy = parts[0], 'USD', 'us-gaap/'
    suffixes = parts[1:]
    if suffixes and suffixes[0] in _SUFFIX_UNIT:
        unit = _SUFFIX_UNIT[suffixes.pop(0)]
    if suffixes:
        if len(suffixes) != 1:
            return None
        taxonomy = _taxonomy_for_suffix(suffixes[0])
        if taxonomy is None:
            return None
    qtrs = 0
    duration = _DURATION.search(body)
    if duration:
        token = duration.group(1)
        if token == 'q':
            qtrs = 1
        else:
            digits = token[:-1]
            if digits.startswith('0'):
                return None
            try:
                qtrs = int(digits)
            except ValueError:
                return None
            if qtrs < 2:
                return None
        body = body[:duration.start()]
    if not _GENERIC_BASE.fullmatch(body) or _DURATION.search(body):
        return None
    if taxonomy == 'us-gaap/' and body in _CURATED_TAGS:
        tag = _CURATED_TAGS[body]
    else:
        tag = ''.join(word[:1].upper() + word[1:] for word in body.split('_'))
        if _snake_tag(tag) != body:
            return None
    return dict(tag=tag, taxonomy=taxonomy, unit=unit, qtrs=qtrs, segments='', coreg='')


def recognized_alias(name) -> bool:
    """Syntactic recognition only, not a claim that the tag exists."""
    return spec_for_name(name) is not None


def display_label(tag) -> str:
    """Split the full original tag into English words, preserving acronyms.

    Does not use abbreviated vendor labels (e.g. 'Basic') or semantic synonyms.
    Malformed tags return an empty label rather than normalized invented words.
    """
    if not isinstance(tag, str) or not _TAG.fullmatch(tag):
        return ''
    label = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', tag)
    label = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', label)
    label = re.sub(r'([A-Za-z])([0-9])', r'\1 \2', label)
    return re.sub(r'([0-9])([A-Za-z])', r'\1 \2', label)


def curated_defs() -> dict[str, dict]:
    """Fresh explicit default recipes, preserving five immutable common aliases.

    Balance-sheet tags default to instant; flows default to one quarter, with
    USD except explicit USD/shares for EPS. NetIncomeLoss also exposes the
    existing four-quarter alias. This finite list is for discovery/known-
    unavailable errors, not a restriction on accepted durations.
    """
    result = {}
    for tag, base, qtrs, unit in _CURATED_ROWS:
        spec = dict(tag=tag, taxonomy='us-gaap/', unit=unit, qtrs=qtrs, segments='', coreg='')
        name = preferred_name(spec)
        if name is None or spec_for_name(name) != spec:
            raise ValueError('Curated SEC definition does not roundtrip')
        result[name] = spec
    result['sec_net_income_4q'] = spec_for_name('sec_net_income_4q')
    return result
