"""Search Hugging Face GGUF metadata; downloads are owned by model_management."""

import re
from huggingface_hub import HfApi

api = HfApi()

PIPELINE_NAMES = {
    "text-generation": "Text",
    "image-text-to-text": "Vision",
    "feature-extraction": "Embedding",
    "text-classification": "Classification",
    "text2text-generation": "Seq2Seq",
}


MODEL_FAMILIES = [
    "Qwen",
    "Llama",
    "DeepSeek",
    "Gemma",
    "GLM",
    "Phi",
    "Mistral",
    "Yi",
    "Command",
    "SmolLM",
]


def format_downloads(n: int | None) -> str:
    if not n:
        return "0"

    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"

    if n >= 1_000:
        return f"{n / 1_000:.1f}K"

    return str(n)


def extract_license(tags: list[str]) -> str | None:
    for tag in tags:
        if tag.startswith("license:"):
            return tag.removeprefix("license:").replace("-", " ").title()
    return None


def extract_family(name: str) -> str:
    lower = name.lower()

    for family in MODEL_FAMILIES:
        if family.lower() in lower:
            return family

    return "Other"


def extract_size(name: str) -> str | None:
    m = re.search(r"(\d+(?:\.\d+)?)([BM])", name, re.I)

    if not m:
        return None

    return f"{m.group(1)}{m.group(2).upper()}"


def prettify_name(repo_id: str):
    author, name = repo_id.split("/", 1)

    name = re.sub(r"-GGUF$", "", name, flags=re.I)
    name = name.replace("_", " ")

    return author, name


def model_to_dict(model):
    author, name = prettify_name(model.id)

    return {
        "id": model.id,
        "author": author,
        "name": name,
        "family": extract_family(name),
        "size": extract_size(name),
        "task": PIPELINE_NAMES.get(
            model.pipeline_tag,
            model.pipeline_tag or "Unknown",
        ),
        "downloads": model.downloads or 0,
        "downloads_text": format_downloads(model.downloads),
        "likes": model.likes or 0,
        "license": extract_license(model.tags or []),
        "created_at": (model.created_at.date().isoformat() if model.created_at else None),
    }


def is_secondary_split_part(filename: str) -> bool:
    """Check if the filename represents a secondary part of a split GGUF model."""
    m = re.search(r"[-.](?:gguf[-.])?(\d+)-of-(\d+)(?:\.gguf)?$", filename, re.IGNORECASE)
    if m:
        part_num = int(m.group(1))
        if part_num > 1:
            return True
    return False


def browse_models(
    query: str | None = None,
    limit: int = 20,
):
    kwargs = {
        "filter": "gguf",
        "sort": "downloads",
        "limit": limit,
    }

    if query and query.strip():
        kwargs["search"] = query.strip()

    models = api.list_models(**kwargs)

    return [model_to_dict(m) for m in models]


def list_quants(repo_id: str):
    info = api.model_info(repo_id, files_metadata=True)

    quants_dict = {}

    for file in info.siblings:
        filename = file.rfilename
        is_gguf = filename.endswith(".gguf") or re.search(
            r"\.gguf[-.]\d+-of-\d+$", filename, re.IGNORECASE
        )
        if not is_gguf:
            continue

        # Detect if it is a split part
        m = re.search(r"^(.*)[-.](?:gguf[-.])?(\d+)-of-(\d+)(?:\.gguf)?$", filename, re.IGNORECASE)
        if m:
            prefix = m.group(1)
            part_str = m.group(2)
            total_parts = m.group(3)
            first_part_str = "1".zfill(len(part_str))

            # Reconstruct base key using correct separator and extension location
            if ".gguf" in filename.lower():
                if "gguf-" in filename.lower() or "gguf." in filename.lower():
                    sep = "-" if "gguf-" in filename.lower() else "."
                    base_key = f"{prefix}.gguf{sep}{first_part_str}-of-{total_parts}"
                else:
                    sep = filename[m.start(2) - 1]
                    base_key = f"{prefix}{sep}{first_part_str}-of-{total_parts}.gguf"
            else:
                base_key = f"{prefix}-{first_part_str}-of-{total_parts}"

            if base_key not in quants_dict:
                quants_dict[base_key] = {
                    "filename": base_key,
                    "quant": extract_quant(base_key),
                    "size_bytes": 0,
                }
            quants_dict[base_key]["size_bytes"] += file.size
        else:
            quants_dict[filename] = {
                "filename": filename,
                "quant": extract_quant(filename),
                "size_bytes": file.size,
            }

    quants = []
    for k, v in quants_dict.items():
        v["size_gb"] = round(v["size_bytes"] / 1024**3, 2)
        quants.append(v)

    quants.sort(key=lambda x: x["size_bytes"])

    return quants


def extract_quant(filename: str):
    filename = filename.upper()

    patterns = [
        "IQ1_M",
        "IQ1_S",
        "IQ2_XXS",
        "IQ2_XS",
        "IQ2_S",
        "IQ3_XXS",
        "IQ3_XS",
        "IQ3_S",
        "IQ3_M",
        "IQ4_XS",
        "IQ4_NL",
        "IQ4_XL",
        "Q2_K",
        "Q3_K_S",
        "Q3_K_M",
        "Q3_K_L",
        "Q4_0",
        "Q4_1",
        "Q4_K_S",
        "Q4_K_M",
        "Q5_0",
        "Q5_1",
        "Q5_K_S",
        "Q5_K_M",
        "Q6_K",
        "Q8_0",
        "F16",
        "BF16",
    ]

    for p in patterns:
        if p in filename:
            return p

    return "Unknown"
