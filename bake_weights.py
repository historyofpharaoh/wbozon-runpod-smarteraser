# Build-time: качаем SE веса (с публичного HF repo) + CLIP-ViT-Large-Patch14.
# SE веса → /weights (структура SD pipeline + clip_mlp_weight.pth)
# CLIP → /opt/SmartEraser/Model_framework/ckpts/clip-vit-large-patch14/
import os
import shutil
import sys
import traceback

try:
    from huggingface_hub import snapshot_download

    print("Downloading SmartEraser weights...", flush=True)
    se_path = snapshot_download(
        repo_id="Nikita12312425345/smarteraser-weights",
        local_dir="/weights",
        ignore_patterns=["*.msgpack", "*.h5"],
        max_workers=4,
    )
    print(f"  → {se_path}", flush=True)

    print("Downloading CLIP-ViT-Large-Patch14 (for CLIPVisualPrompt)...", flush=True)
    clip_path = snapshot_download(
        repo_id="openai/clip-vit-large-patch14",
        local_dir="/opt/SmartEraser/Model_framework/ckpts/clip-vit-large-patch14",
        ignore_patterns=["*.msgpack", "*.h5", "tf_model.h5", "flax_model.msgpack"],
        max_workers=4,
    )
    print(f"  → {clip_path}", flush=True)

    import subprocess
    print("Sizes:")
    subprocess.run(["du", "-sh", "/weights"])
    subprocess.run(["du", "-sh", "/opt/SmartEraser/Model_framework/ckpts/clip-vit-large-patch14"])
except Exception:
    traceback.print_exc()
    sys.exit(1)
