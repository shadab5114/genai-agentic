# Pre-downloaded docling models

This folder holds the model weights docling needs (layout detection, table
structure, RapidOCR, code/formula recognition, picture classification). It is
**gitignored** — do not commit it here; push it to your company's internal
artifact store (Artifactory/Nexus/S3/network share) instead and have each
machine pull it from there.

## Why this exists

By default `docling.DocumentConverter()` downloads its models from Hugging
Face Hub on first use. If HF downloads are blocked on your network, do the
download once from a machine that *does* have HF access, then point docling
at the resulting folder instead — no code changes required beyond what's
already wired up in `document_processor/file_handler.py`.

## How this folder was produced

From a machine with Hugging Face access, with `docling` installed:

```bash
docling-tools models download -o docling_models --all
```

(`--all` was used here to also grab the code-formula and picture-classifier
models for future use; drop `--all` for just the models the default pipeline
config in this project actually loads: layout, tableformer, rapidocr.)

## How it's used

`document_processor/file_handler.py` looks for this folder (or the path in
the `DOCLING_ARTIFACTS_PATH` env var / `.env` setting) and, if present, passes
it as `artifacts_path` to docling's `PdfPipelineOptions`. It also sets
`HF_HUB_OFFLINE=1` so any accidental cache-miss fails loudly instead of
silently reaching out to the internet.

## Restoring this folder on a new machine

1. Pull the archive from your internal artifact store.
2. Extract it to `multi_agent_rag/docling_models/` (this exact path, or set
   `DOCLING_ARTIFACTS_PATH` in `.env` to wherever you put it).
3. Run the app as normal — no Hugging Face access required.

## Verifying it's fully offline

```bash
HF_HUB_OFFLINE=1 python -c "
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

opts = PdfPipelineOptions(artifacts_path=str(Path('docling_models').resolve()))
conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
print(conv.convert('test/ocr_test.pdf').document.export_to_markdown()[:200])
"
```

If this succeeds with `HF_HUB_OFFLINE=1` set, no network call to Hugging Face
was made.
