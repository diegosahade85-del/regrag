# RegRAG

Retrieval-augmented assistant over Latin American product certification
regulation (Argentina, Chile, Peru, Colombia, Paraguay, Uruguay).

Built to answer compliance questions like "what marking obligations does
Res. SIC 16/2025 impose on imported power supplies?" with citations to the
specific article, not a plausible-sounding paraphrase.

**Status:** in development.
**Stack:** Python, PostgreSQL 17 + pgvector, Voyage AI embeddings, Claude Sonnet 5,
FastAPI, ragas.

## Running it

```bash
uv sync
docker compose up -d                         # Postgres 17 + pgvector
uv run pytest                                # 137 tests
uv run python scripts/ingest_corpus.py       # corpus -> data/processed/*.jsonl
uv run python scripts/index_corpus.py        # embed + load into Postgres
uv run python scripts/search.py "¿qué exige el artículo 4 sobre marcado?"
uv run python scripts/search.py --mode dense "IEC 60364"   # or lexical / hybrid
uv run python scripts/diagnose_retrieval.py  # dense vs lexical vs hybrid
uv run python scripts/ask.py "¿qué obligaciones tienen los importadores?"
```

Corpus documents are not committed (see `data/sources.md` for provenance and
download URLs). Secrets live in `.env`, which is gitignored.

## Design decisions

### Chunking: article boundaries, windowed only where they overflow

Three strategies are implemented and measured against the real corpus rather
than chosen by intuition. Numbers below are from 13 source documents across six
jurisdictions:

| Strategy | Chunks | Median chars | p95 | Max | Carry an article label |
|---|---|---|---|---|---|
| `article` | 289 | 3,042 | 27,423 | **168,351** | 96% |
| `fixed` (1200/200) | 2,081 | 1,197 | 1,200 | 1,200 | **0%** |
| `hybrid` (1200/200) | 2,170 | 1,196 | 1,200 | 1,200 | 91% |

**`article` alone fails on size.** Splitting on article boundaries is
semantically right — a regulatory obligation is scoped to its article — but
real norms contain articles that are enormous. Article 20 of Colombia's RETIE
(`CO_MINENERGIA_res-9703`) is 168k characters on its own: it is the entire
product-requirements chapter, covering insulators through transformers, with
its tables inline. That is not a parsing bug, that is how the norm is written.
A chunk that size cannot be embedded and would swamp any context window.

**`fixed` alone fails on citability.** Uniform windows are well-behaved
dimensionally and completely anonymous: not one chunk knows which article it
came from. For a compliance assistant that is disqualifying. "Imported power
supplies must carry the marking" is worthless without "per art. 4 of
Res. SIC 16/2025" attached — an answer a compliance officer cannot trace to a
clause is an answer they cannot act on.

**`hybrid` is what ships.** Split on article boundaries first, then window only
the articles that exceed the budget, propagating the article label onto every
piece. Bounded like `fixed` (max 1,200 chars), citable like `article` (91% of
chunks carry a label; the remainder are preambles and recitals, which genuinely
have no article number).

Whether 1200/200 is the right budget is an open question — that gets settled
against the eval set, not by argument.

### Deduplication: same law, indexed twice

Several norms are published in more than one place, and both renderings are in
the corpus: Res. SIC 16/2025 as the Boletín Oficial edition *and* the *texto
original* page, ENACOM Res. 57/2026 as a PDF *and* its HTML. Their articles came
out **byte-identical** — measured at cosine similarity 1.0000 with matching
character counts across 50 cross-variant pairs — so every such clause was
embedded twice, cost twice, and competed against itself for retrieval slots.

Deduplication is on normalized text within a single norm, which drops 55 chunks
(2,225 → 2,170). Two details matter:

- **Chunk-level, not document-level.** Dropping a whole rendering would discard
  the content only that rendering has — the *texto original* page carries annexes
  the Boletín edition does not (12,874 vs 25,876 characters for the same 13
  articles). Collapsing identical chunks keeps everything unique from both.
- **Scoped per norm.** Closing formulas like "Comuníquese, publíquese y
  archívese" recur across unrelated norms. Each is a real, separately-citable
  clause of its own norm, not a duplicate, so the key is `(norm_id, text)`.

Exact cross-variant pairs fell from 50 to 2. The residue is windowing, not
duplication: the two renderings differ slightly in length, so window boundaries
inside long articles land in different places and produce overlapping — not
repeated — chunks.

**This did not move recall@10** (identical before and after). Deduplication is
corpus hygiene: it halves the embedding cost of the affected norms and stops one
clause occupying two result slots, but the retrieval failures below are failures
of matching, not of slot competition. Worth recording as a negative result — and
it means the Day-4 hybrid comparison runs against a clean corpus, so any
improvement is attributable to the retrieval change alone.

### Retrieval baseline: dense search, and where it breaks

2,170 chunks embedded with `voyage-3` (1024 dimensions), cosine similarity.
The corpus is its own ground truth here: for each probe, SQL finds by exact
string match which chunks actually contain it, then each retriever is asked the
same thing. No annotation required, and the measurement is reproducible with
`scripts/diagnose_retrieval.py`. Per-retriever numbers are in the hybrid table
below; dense alone reaches **17/22**.

**Roughly a quarter of questions whose answer sits verbatim in the corpus do not
retrieve it.** Two distinct failure modes, and both argue for lexical search
alongside the vectors:

**Standard and document numbers are invisible to embeddings.** `IEC 60364` and
`NTC 4552` both fail; `NTC 2050` succeeds only because it appears in 174 chunks
and frequency carries it. An embedding of "IEC 60364" is not meaningfully
distinct from one of "IEC 60335" — the model has no representation for the
difference between two standards whose numbers differ by one digit, and that
difference is the entire question. This is exactly what a compliance officer
types.

**Short phrases retrieve documents *about* a topic, not documents that
*impose* it.** Querying `declaración jurada de conformidad` returns
fill-in-the-blank declaration forms from Colombia's RETIE — templates that are
wholly about declaring conformity, so their chunk embedding sits close to a
short noun phrase. Articles 3 and 4 of Res. SIC 16/2025, which actually create
the sworn-declaration obligation, are nowhere in the top 10: there the phrase
is one clause inside dense legal text. In compliance the form template is
useless and the article is the answer, so this ranking is backwards in the way
that matters most.

### Hybrid retrieval: dense and lexical, fused by rank

Postgres full-text search over the same table (a `tsvector` column with Spanish
stemming, GIN-indexed), fused with the vector results by Reciprocal Rank Fusion.

| Probe type | Dense | Lexical | **Hybrid** |
|---|---|---|---|
| Identifiers (`IEC 60364`, `RESOL-2025-16-APN-SIYC#MEC`) | 6/9 | 9/9 | **9/9** |
| Verbatim phrases (`declaración jurada de conformidad`) | 6/8 | 8/8 | **8/8** |
| Paraphrases (`penalidades` → *sanciones*) | 5/5 | 2/5 | **5/5** |
| **Total recall@10** | **17/22** | **19/22** | **22/22** |

**Neither retriever alone is sufficient, and they fail in places the other
cannot see.** Dense search misses `IEC 60364` entirely — not one of its top five
results contains the string — because an embedding of a standard number is not
meaningfully distinct from an embedding of the adjacent standard, and the digits
are the whole query. Lexical search misses "penalidades por incumplimiento"
because the corpus says *sanciones*, and no amount of stemming bridges two
different words.

**The paraphrase row is there to stop the comparison being rigged.** Every probe
in the first two categories is a string lifted verbatim from the corpus, which
is exactly what full-text search is built for: measured on those alone, lexical
ties hybrid at 17/17 and the vectors look like dead weight. The paraphrase probes
use wording that appears **nowhere** in the corpus, paired with the term the
corpus actually uses, so lexical must fail by construction. That is the only
category in this harness that can show what dense retrieval contributes.

**Why RRF rather than blending the scores.** Cosine similarity and `ts_rank` are
not on a common scale — 0.5 from one means nothing in terms of the other — so
they cannot be averaged, normalised, or thresholded against each other in any
principled way. Rank position is the one output both produce that *is*
comparable. RRF scores a result as Σ 1/(k + rank) across rankings, with k = 60:
large enough that the gap between rank 1 and rank 5 is small, so a result both
retrievers found outranks one that placed first in a single list. No
per-retriever weight to tune, and adding a third retriever later needs no
recalibration.

Two implementation details that matter:

- **The candidate pool is wider than the requested limit** (5× by default).
  Fusing only the top-`limit` of each ranker discards precisely the results that
  win on combined support rather than on either ranking alone.
- **Ties break deterministically, on chunk id.** With two rankers, ties are
  common — both rank-1 results score exactly 1/61 — and an arbitrary order would
  make identical runs return different answers.

### Answering: refusal is a state, and citations are verified

Answers come back as a schema, not prose to be parsed:

```python
class Answer(BaseModel):
    answerable: bool
    answer: str
    citations: list[Citation]        # country, norm_id, article
    confidence: Literal["alta", "media", "baja"]
```

**Refusing is structural.** `answerable: false` is a field the caller branches
on, not a sentence in the output that has to be pattern-matched. In compliance
the two error directions are not equivalent: an invented answer costs a client
an import held at customs, a "not in the corpus" costs them one more search. The
system prompt says so in those terms, because a rule with its reason attached
survives paraphrase better than a rule without one.

Asked about a product model absent from the corpus, the assistant declines *and*
says what it would need — "faltaría identificar el tipo de producto (eléctrico
de baja tensión, equipo de telecomunicaciones) para vincularlo con el
procedimiento correspondiente." Asked what sanctions Ley 24.240 sets out — a law
the corpus repeatedly *references* but does not *contain*, and one the model
certainly knows from training — it reports what the corpus does say (that
infractions under Res. 16/2025 art. 6 are subject to that law), states that the
sanctions articles themselves are not in the corpus, and declines to enumerate
them.

**Every citation is checked against what was actually retrieved.** A
hallucinated citation is indistinguishable from a real one by reading it: same
shape, plausible norm, plausible article. `unsupported_citations()` compares
each returned `(country, norm_id, article)` against the triples actually placed
in the context window, and the CLI marks any that fail and exits non-zero. The
prompt asks the model not to invent citations; the check is what makes that
claim verifiable rather than hoped-for.

**Confidence is ordinal, not a float.** `alta` / `media` / `baja` rather than
`0.87`, because a two-decimal number implies a calibration the model does not
have. Three levels the prompt defines explicitly — *does the context answer this
directly, partially, or only tangentially* — are a question the model can
actually answer.

### No ANN index at this corpus size

pgvector warned `ivfflat index created with little data`, so the index was
measured rather than trusted (on the 2,225-chunk corpus, before deduplication):

| Configuration | recall@10 vs exact | Latency |
|---|---|---|
| Exact scan (no index) | 10/10 by definition | 16.5 ms |
| IVFFlat `lists=100, probes=1` | 4/10 | 1.2 ms |
| IVFFlat `lists=2, probes=1` | 10/10 | 20.0 ms |
| IVFFlat `lists=10, probes=10` | 10/10 | 16.1 ms |

Every configuration that reaches full recall is **as slow as or slower than**
scanning the whole table exactly. Approximate search is a trade, and at this
size there is nothing to buy, so the indexer drops the index below a 50,000-row
threshold and says so. The threshold and the tuning belong together: `lists`
must scale with row count (pgvector's guidance is `rows/1000`), and a default
of 100 on a 2k-row table puts ~22 rows in each list, so `probes=1` scans one
percent of the corpus.

## Two classes of bug worth knowing about

### Parsing traps

**Cross-references look exactly like article headers.** Regulatory prose is
dense with them: "las infracciones previstas en el artículo 4° de la presente
resolución". A naive `artículo \d+` split fires on every one and shreds the
article that actually contains the obligation. The chunker requires a header to
start a line *and* be followed by a separator, which rejects mid-sentence
references.

**Tables of contents hijack article boundaries.** The RETIE PDF opens with a
TOC whose entries — `ARTÍCULO 20º. REQUERIMIENTOS ......... 83` — are
indistinguishable from real headers by shape alone. Left in, they claim the
article boundary at the top of the document, and the real article body 80 pages
later is absorbed into whatever span precedes it. In this corpus 31 of 63
detected headers were TOC entries. Lines carrying dot leaders (4+ dots, so a
prose ellipsis survives) are dropped during extraction.

**Site chrome outranks the regulation.** Before this was fixed, the top two
results for `Res. 16/2025` were the Argentine portal's navigation menu
("Presidencia de la Nación · Pasar al contenido principal · Buscar"), embedded
and indexed exactly like regulation. Filtering `<nav>`/`<header>`/`<footer>`
does not reach it: argentina.gob.ar is a Drupal build with **zero** semantic
layout elements — the chrome lives in anonymous `<div>`s. Content extraction by
layout (trafilatura) removes it, and as a side effect recovers content the
DOM-walking extractor was silently dropping (ENACOM Res. 57/2026 went from
20,609 to 40,327 characters). Semantic tags are still stripped first, as a
cheap pre-pass for pages that do use them.

### Silent-failure traps

None of these raise an error. Each returns a plausible wrong answer, which is
why they all carry tests.

| Trap | What happens |
|---|---|
| Stale IVFFlat index | An IVFFlat index stores centroids from the data present when it was built. Built before a load — or left in place across a `TRUNCATE` — it describes data that is no longer there. Measured here: **0/10** recall, no warning. `CREATE INDEX IF NOT EXISTS` is the wrong statement; the index must be dropped and rebuilt. |
| Wrong embedding `input_type` | Voyage embeds queries and documents asymmetrically. Sending a question with `input_type="document"` returns a perfectly valid, worse vector. The symptom is degraded recall, never an exception. |
| Per-request batch cap | Voyage's document cap only errors at scale, so a batching bug ships clean from a five-chunk development run. |
| `CREATE TABLE IF NOT EXISTS` on an evolved schema | A no-op against a table that already exists, so the `tsvector` column added for full-text search never reaches a database created before it. Lexical search then returns nothing, on a corpus sitting right there, with no error. `create_schema` adds the column explicitly for older databases. |
| A hand-maintained `tsvector` | Kept in sync by application code, it drifts the moment one write path forgets to update it, and the affected chunk silently stops being findable. It is a `GENERATED ALWAYS` column so Postgres owns it. |

## Known limitations

- Tables are flattened to text by the PDF extractor; column structure is lost.
  Several product requirements in RETIE and the Chilean SEC protocols are
  tabular, so this will cost recall on "what is the requirement for product X"
  questions.
- Article-number extraction on the RETIE PDF recovers 32 headers but skips a
  few (22, 25, 28–30, 33) and reports one out of sequence, most likely headers
  broken across page boundaries during PDF extraction. Not yet investigated.
- Deduplication is exact-match only. Two norms are published in two renderings
  each, and where windowing splits a long article at different offsets the
  resulting chunks overlap without being identical — 28 cross-variant pairs
  still sit above cosine 0.9. Near-duplicate collapsing is not implemented, and
  is a judgement call rather than an obvious win: those chunks are not the same
  text, so merging them would discard content.
- Everything is measured on 22 probes, and the harness answers "is the chunk
  containing this string in the top 10?" — not "is the retrieved article the one
  that answers the question?". That is enough to establish a direction and not
  enough to be precise about it; the golden set of real compliance questions,
  with human-verified answers, replaces it.
- `k = 60` and a 5× candidate pool are the conventional defaults, carried over
  untested. Both are tunable against the golden set once it exists.
- 13 source documents against a target of 30–60. Four downloads are blocked at
  the network level from the VPS (see `data/sources.md`).
