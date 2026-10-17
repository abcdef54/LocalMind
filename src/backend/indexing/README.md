# Background indexing and scoped retrieval (BE-08)

Follow one accepted Send request through indexing. Parsing reads file formats; splitting prepares text for embedding; the store changes PostgreSQL state.

```text
User sends message + files
        ↓
Submission API saves files and database records
File: pending · Submission: accepted · Run: pending
        ↓
worker.py asks store.py → claim_next() to reserve a file
File: indexing
        ↓
parsing.py → parse_document() starts a separate parser process
        ↓
parser_process.py → extract_sections()
Reads PDF/DOCX/text and returns extracted text with page information
        ↓
vectors.py → split_extracted_sections_into_document_chunks()
Divides extracted text into smaller overlapping chunks
        ↓
vectors.py → generate_embeddings_and_upsert_chunk_vectors()
Generates embeddings and writes chunks to Chroma
        ↓
store.py → finish()
File: ready
        ↓
When every file in the submission is ready:
Submission: ready · Run: queued
```

A failed file blocks the run; retry resets failed files only. The executor runs queued AI jobs through GenerateReply (BE-10).

[Configuration, contracts and troubleshooting](REFERENCE.md).
