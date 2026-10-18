# Hugging Face catalog and model files

```text
GET /api/v1/models/catalog or /models/quants
        ↓
huggingface.py → search GGUF metadata / group split quantizations

POST /api/v1/model-downloads
        ↓
model_management/ → persisted task → background worker → download subprocess
        ↓
cpp_models/managed/<installation-id>/ → registry + llama-swap config
        ↓
External llama-swap loads the selected GGUF for inference
```
