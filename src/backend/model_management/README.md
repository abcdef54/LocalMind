# Model download flow

```text
POST /api/v1/model-downloads → store.py saves/reuses task → HTTP 202 + task ID
        ↓
worker.py polls PostgreSQL and reserves shared model storage (one download at a time)
        ↓
Task: downloading → download_process.py pins a revision and downloads every GGUF part
        ↓
Validate complete files → move staging files into managed/<installation-id>/
        ↓
Save installed_models metadata → write llama-swap config → model: ready
        ↓
Task: completed → frontend reads GET /model-downloads/{id}

Failure/shutdown → failed/interrupted → POST the same source to retry
Refresh → GET /model-downloads → recover task IDs and status
DELETE /models/{id} → model: deleting → worker removes config + files + DB row
GET /capabilities → frontend discovers server options and limits
```
