# RegRAG

Retrieval-augmented assistant over Latin American product certification
regulation (Argentina, Chile, Peru, Colombia, Paraguay, Uruguay).

Built to answer compliance questions like "what marking obligations does
Res. SIC 16/2025 impose on imported power supplies?" with citations to the
specific article, not a plausible-sounding paraphrase.

**Status:** in development.
**Stack:** Python, PostgreSQL 17 + pgvector, Voyage AI embeddings, Claude API,
FastAPI, ragas.

## Running it

```bash
uv sync
docker compose up -d                         # Postgres 17 + pgvector
uv run pytest                                # 77 tests
uv run python scripts/ingest_corpus.py       # corpus -> data/processed/*.jsonl
uv run python scripts/index_corpus.py        # embed + load into Postgres
uv run python scripts/search.py "¿qué exige el artículo 4 sobre marcado?"
uv run python scripts/diagnose_dense.py      # retrieval quality baseline
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
| `article` | 313 | 2,744 | 25,926 | **168,351** | 96% |
| `fixed` (1200/200) | 2,053 | 1,197 | 1,200 | 1,200 | **0%** |
| `hybrid` (1200/200) | 2,225 | 1,196 | 1,200 | 1,200 | 92% |

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
piece. Bounded like `fixed` (max 1,200 chars), citable like `article` (92% of
chunks carry a label; the remainder are preambles and recitals, which genuinely
have no article number).

Whether 1200/200 is the right budget is an open question — that gets settled
against the eval set, not by argument.

### Retrieval baseline: dense search, and where it breaks

2,225 chunks embedded with `voyage-3` (1024 dimensions), cosine similarity.
The corpus is its own ground truth here: for each probe, SQL finds by exact
string match which chunks actually contain it, then dense retrieval is asked
the same thing. No annotation required, and the measurement is reproducible
with `scripts/diagnose_dense.py`.

| Probe type | recall@10 |
|---|---|
| Identifiers (`IEC 60364`, `RESOL-2025-16-APN-SIYC#MEC`, `Ley N° 24.240`) | 6/9 |
| Verbatim phrases (`declaración jurada de conformidad`) | 6/8 |
| **Combined** | **12/17 (71%)** |

**Roughly a third of questions whose answer sits verbatim in the corpus do not
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

### No ANN index at this corpus size

pgvector warned `ivfflat index created with little data`, so the index was
measured rather than trusted:

| Configuration | recall@10 vs exact | Latency |
|---|---|---|
| Exact scan (no index) | 10/10 by definition | 16.5 ms |
| IVFFlat `lists=100, probes=1` | 4/10 | 1.2 ms |
| IVFFlat `lists=2, probes=1` | 10/10 | 20.0 ms |
| IVFFlat `lists=10, probes=10` | 10/10 | 16.1 ms |

Every configuration that reaches full recall is **as slow as or slower than**
scanning all 2,225 vectors exactly. Approximate search is a trade, and at this
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

## Known limitations

- Tables are flattened to text by the PDF extractor; column structure is lost.
  Several product requirements in RETIE and the Chilean SEC protocols are
  tabular, so this will cost recall on "what is the requirement for product X"
  questions.
- Article-number extraction on the RETIE PDF recovers 32 headers but skips a
  few (22, 25, 28–30, 33) and reports one out of sequence, most likely headers
  broken across page boundaries during PDF extraction. Not yet investigated.
- Two norms are indexed twice under different variants (Res. 16/2025 as both
  the Boletín Oficial and the *texto original* rendering; ENACOM Res. 57/2026 as
  both PDF and HTML). Near-duplicate chunks therefore compete for the same
  retrieval slots. Deduplication is not implemented.
- 13 source documents against a target of 30–60. Four downloads are blocked at
  the network level from the VPS (see `data/sources.md`).
