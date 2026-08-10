"""Metadata carried by a source document, derived from its filename.

Corpus files follow `{COUNTRY}_{AGENCY}_{norm-id}[_{variant}].{ext}` — see
data/sources.md. Encoding provenance in the filename keeps the raw corpus
(which is gitignored) self-describing, so metadata survives a re-download.
"""

import re
from dataclasses import dataclass

# Years in this corpus are 19xx/20xx. Matching narrowly keeps norm numbers that
# merely happen to be four digits (e.g. "res-9703") from being read as dates.
_YEAR = re.compile(r"(?:19|20)\d{2}")


@dataclass(frozen=True)
class SourceMetadata:
    country: str
    agency: str
    norm_id: str
    variant: str | None = None
    year: int | None = None


def parse_source_filename(filename: str) -> SourceMetadata:
    stem = filename.rsplit(".", 1)[0]
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(
            f"{filename!r} breaks the naming convention "
            f"{{COUNTRY}}_{{AGENCY}}_{{norm-id}}[_{{variant}}].{{ext}}"
        )

    country, agency, norm_id, *rest = parts
    years = _YEAR.findall(norm_id)
    return SourceMetadata(
        country=country,
        agency=agency,
        norm_id=norm_id,
        variant="_".join(rest) or None,
        year=int(years[-1]) if years else None,
    )
