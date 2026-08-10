# RegRAG

Retrieval-augmented assistant over Latin American product certification
regulation (Argentina, Chile, Peru, Colombia, Paraguay, Uruguay).

Built to answer compliance questions like "what marking obligations does
Res. SIC 16/2025 impose on imported power supplies?" with citations to the
specific article, not a plausible-sounding paraphrase.

**Status:** in development.
**Stack:** Python, FastAPI, PostgreSQL + pgvector, Claude API, ragas.

## Running it

```bash
uv sync
uv run pytest                            # 50 tests
uv run python scripts/ingest_corpus.py   # corpus -> data/processed/*.jsonl
```

Corpus documents are not committed (see `data/sources.md` for provenance and
download URLs).

## Design decisions

### Chunking: article boundaries, windowed only where they overflow

Three strategies are implemented and measured against the real corpus rather
than chosen by intuition. Numbers below are from 13 source documents
(~1.4M characters) across six jurisdictions:

| Strategy | Chunks | Median chars | p95 | Max | Carry an article label |
|---|---|---|---|---|---|
| `article` | 313 | 2,768 | 25,926 | **168,351** | 96% |
| `fixed` (1200/200) | 2,058 | 1,197 | 1,200 | 1,200 | **0%** |
| `hybrid` (1200/200) | 2,181 | 1,196 | 1,200 | 1,200 | 91% |

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
chunks carry a label; the remaining 9% are preambles and recitals, which
genuinely have no article number). RETIE article 20 becomes ~140 windows that
all still cite article 20.

Whether 1200/200 is the right budget is an open question — that gets settled
against the eval set, not by argument.

### Two parsing traps worth knowing about

**Cross-references look exactly like article headers.** Regulatory prose is
dense with them: "las infracciones previstas en el artículo 4° de la presente
resolución". A naive `artículo \d+` split fires on every one of these and
shreds the article that actually contains the obligation. The chunker requires
a header to start a line *and* be followed by a separator, which rejects
mid-sentence references.

**Tables of contents hijack article boundaries.** The RETIE PDF opens with a
TOC whose entries — `ARTÍCULO 20º. REQUERIMIENTOS ......... 83` — are
indistinguishable from real headers by shape alone. Left in, they claim the
article boundary at the top of the document, and the real article body 80
pages later gets absorbed into whatever span precedes it. In this corpus 31 of
63 detected headers were TOC entries. Lines carrying dot leaders (4+ dots, so a
prose ellipsis survives) are dropped during extraction; that alone changed
`article` chunk counts from 503 to 313.

### Metadata lives in the filename

Corpus files follow `{COUNTRY}_{AGENCY}_{norm-id}[_{variant}].{ext}`. The raw
corpus is gitignored, so encoding provenance in the filename keeps it
self-describing across a re-download, and gives every chunk `country`,
`agency`, `norm_id`, `year` and `article` for metadata filtering at retrieval
time — which is what makes "how does Argentina differ from Chile on marking?"
answerable.

## Known limitations

- Article-number extraction on the RETIE PDF recovers 32 headers but skips a
  few (22, 25, 28–30, 33) and reports one out of sequence, most likely headers
  broken across page boundaries during PDF extraction. Not yet investigated.
- Tables are flattened to text by the PDF extractor; column structure is lost.
  Several product requirements in RETIE and the Chilean SEC protocols are
  tabular, so this will cost recall on "what is the requirement for product X"
  style questions.
- 13 source documents so far, against a target of 30–60. Four downloads are
  blocked at the network level from the VPS (see `data/sources.md`).
