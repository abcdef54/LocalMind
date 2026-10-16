# Model files and llama-swap configuration

This directory manages local GGUF model files. Actual inference uses the model client configured by the graph and the external llama-swap server.

```text
Legacy /models route → services/model_services.py
        ↓
LocalMindModelManager (model_manager.py)
   ├─ Browse / list quants → huggingface.py → Hugging Face metadata
   ├─ Download → huggingface.py → local cpp_models/ GGUF files
   └─ Scan / delete → local filesystem
        ↓
ModelService updates installed_models records
        ↓
Manager writes llama_swap_config.yaml
        ↓
External llama-swap reads config and serves inference requests
```
