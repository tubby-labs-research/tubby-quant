# SEC fields

A **catalog** is your prepared local collection of SEC tables. The **dictionary**
maps each official `sec_` name to one observed numeric recipe: tag, taxonomy,
unit, duration, segment and co-registrant. A name is not a promise of coverage.
Prices and SEC inputs are selected separately. Research campaigns remain
price/volume-only; use fundamentals in Alpha Lab or Qlib Validation.

## First field

1. Open **SEC fields** and choose a prepared **Source catalog**.
2. If prompted, run **Prepare shared dictionary** once. If only execution
   preparation is missing, use **Prepare run index**. These jobs read the local
   catalog; they do not download data or materialize every field's daily arrays.
3. In **Company fields**, search an accounting term, original tag or official
   name. Search `inventory` or `sec_inventory`, for example.
4. Review the exact definition. A single observed unit/period is fixed text;
   multiple variants require **Choose unit and period…** before insertion.
5. Use **Insert field** and review the formula's resolved inputs before running.

`rank(sec_inventory / sec_assets)` is Lab syntax;
`$sec_inventory / $sec_assets` is Qlib syntax. These are definition/syntax
examples, not recommended strategies. They require the matching recipes in your
catalog. Type verified names directly for automatic dependency resolution.
Unknown or unavailable names fail rather than silently selecting substitutes.

## Read the definition

Company fields group standard concepts with observed blank segment/co-registrant
contexts. **Advanced definitions** includes exact editions, special dimensions,
entities and custom/unconfirmed definitions. Some candidates need **Verify &
insert**. Disabled insertion and partial search results are explained in the UI.
An empty Company fields query shows starters, not every field.

| Choice | Meaning |
| --- | --- |
| Unit, such as USD or shares | Exact reported unit; no automatic conversion. |
| `qtrs=0` | Instant balance at a reporting date. |
| `qtrs=1`, `2`, `3`, `4` | Reported rounded quarter duration; no automatic YTD subtraction or rolling TTM. |
| Taxonomy family, such as `us-gaap/` | Same exact tag across available editions; not proof of unchanged accounting meaning. |
| Exact edition, such as `us-gaap/2024` | Dictionary edition, not financial year. |
| Segments / co-registrant | A distinct reporting context/entity, not necessarily a whole-company value. |

Duration suffixes `_q`, `_2q`, `_3q`, `_4q` describe reported periods. Custom,
currency and contextual variants have distinct verified names. Expand **Recipe
details** rather than constructing a name by guesswork. Structural null metadata
is not an observed empty context; unsupported recipes remain raw-queryable.

## Inspect original rows

Expand **Advanced: inspect raw source tables and notes**, select a table, submit
**Filters (JSON)** and press **Read rows**. Face-statement catalogs expose SUB
(filings), TAG (definitions), NUM (values) and PRE (layout). Additional tables are
available only when supplied by your catalog; text/notes are not daily numeric inputs.

Example NUM filter:

```json
{"tag":"InventoryNet","uom":"USD","qtrs":"0","segments":"","coreg":""}
```

Use string values and actual column names. Start with `{}` to inspect columns.
Raw filters match exact strings: `us-gaap/` is not a family wildcard here.
Inspect TAG by tag and SUB by a copied `adsh` accession; views do not auto-join.

## Missing values and saved inputs

The numeric convention waits two calendar days after the later of accepted/filed
dates, then uses the first eligible price session. It is inferred availability,
not an observed dissemination clock. Latest eligible period end wins, then its
latest eligible acceptance. Default expiry is 15 calendar months after the period
end. Conflicting/invalid winners remain missing without reviving older values.
Current ticker mapping is a research proxy; daily float32 panels can round values.

Search/formula checks can work before the run index exists; **Run never builds
or repairs it**. Missing indexes, changed source bytes or saved provenance block
reuse. Restore the original artifacts or explicitly choose inputs for a new run.
Earlier saved bindings remain read-only with their original meanings. **Switch
to dictionary** or **Clear saved input pins** changes current editor settings,
not historical results. Preparation and syntax checks do not certify coverage
or data quality. See [data](DATA.md) and [limitations](LIMITATIONS.md).
