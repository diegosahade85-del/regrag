# Corpus sources

Documents live in `data/raw/` (gitignored — not committed, since some are large
binaries and this keeps the repo lean). This file is the reproducible record of
what was downloaded, from where, and what is still missing.

Naming convention: `{COUNTRY}_{AGENCY}_{norm-id}.{ext}`

## Downloaded (13)

| File | Country | Agency | Source URL |
|---|---|---|---|
| AR_SIC_res-16-2025_texto.html | AR | Secretaría de Industria y Comercio | https://www.argentina.gob.ar/normativa/nacional/resoluci%C3%B3n-16-2025-410052/texto |
| AR_SIC_res-16-2025_boletin.html | AR | Boletín Oficial | https://www.boletinoficial.gob.ar/detalleAviso/primera/321814/20250225 |
| AR_ENACOM_res-57-2026_texto.html | AR | ENACOM | https://www.argentina.gob.ar/normativa/nacional/norma-423392/texto |
| AR_ENACOM_res-57-2026.pdf | AR | ENACOM | https://www.cabase.org.ar/wp-content/uploads/2026/03/Resolucion-ENACOM-57-2026.pdf |
| AR_SC_res-92-1998.html | AR | Secretaría de Comercio (histórica, referenciada por normas vigentes) | https://servicios.infoleg.gob.ar/infolegInternet/anexos/45000-49999/49285/norma.htm |
| CL_SEC_res-exenta-37705-2026.pdf | CL | Superintendencia de Electricidad y Combustibles | https://www.diariooficial.interior.gob.cl/publicaciones/2026/03/05/44392/01/2775206.pdf |
| CL_SEC_rex-28201_protocolo-pe-8-11.pdf | CL | SEC | https://www.ingcer.cl/wp-content/uploads/2024/11/REX-28201-Modifica-protocolo-PE-8-11_2023.pdf |
| CO_MINENERGIA_res-40117-2024_retie.pdf | CO | Ministerio de Minas y Energía (RETIE) | https://www.minenergia.gov.co/documents/11563/Resoluci%C3%B3n_40117_de_2024.pdf |
| CO_MINENERGIA_res-9703.pdf | CO | Ministerio de Minas y Energía | https://www.minenergia.gov.co/documents/9024/9703.pdf |
| PE_INDECOPI_ntp-370-053-1999.pdf | PE | INDECOPI | https://prevencionlaboralrimac.com/cms_data/contents/rimacdatabase/media/legislaciones/leg-8588686583102193887.pdf |
| PE_INDECOPI_ntp-370-301-2002.pdf | PE | INDECOPI | https://www.alfacent.com/uploads/NTP%20INSTALACIONES%20ELECTRICAS%20EN%20EDIFICIOS.pdf |
| UY_URSEA_productos-electricos.html | UY | URSEA | https://www.gub.uy/unidad-reguladora-servicios-energia-agua/productos-electricos |
| PY_INTN_direccion-seguridad-electrica.html | PY | INTN | https://www.intn.gov.py/index.php/organismos/direccion-de-seguridad-electrica |

## Pending / blocked (4)

These hosts refused connections from the VPS network (common for LatAm sites
blocking hosting-provider IP ranges) — need manual download from a residential
connection and upload via `scp`.

| Intended file | Source URL | Notes |
|---|---|---|
| CO_RETIE_anexo-general.pdf | https://www.suin-juriscol.gov.co/imagenes/09/09/2021/1631201674506_Anexo%20General.pdf | RETIE full annex, important — high priority to fetch manually |
| PE_INDECOPI_ntp-370-306-2003.pdf | https://www.servilex.pe/documents/seguridad/r056-2003-1-indecopi.pdf | servilex.pe unreachable from VPS |
| PE_INDECOPI_ntp-iec-60364-4-42-2013.pdf | https://www.servilex.pe/documents/seguridad/iec60364-4-42.pdf | same host |
| PE_INDECOPI_ntp-370-310-2013.pdf | https://servilex.pe/documents/seguridad/370.310.pdf | same host |

## Still missing (Día 1 target was 30–60 docs; got 13)

Not yet covered: IRAM standard indexes, older repealed AR resolutions
(731/87, 524/98, 169/18) referenced by Res. 16/2025, Chile SEC protocol
catalog beyond the 2 fetched, Peru MTC (telecom homologation, separate from
INDECOPI product certs), Colombia CRC (telecom), Uruguay UNIT standard
catalog, Paraguay INTN resolutions (only got an org page, not a resolution
text). Fill these in as time allows — quality/relevance matters more than
hitting a document count.
