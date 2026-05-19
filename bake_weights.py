# Build-time: качаем RORem (~6.5 GB) + sdxl-vae-fp16-fix (~335 MB).
# Обе модели публичные, без gated-доступа.
#
# RORem:           /weights/RORem/             (LetsThink/RORem)
# sdxl-vae:        /weights/sdxl-vae-fp16-fix/ (madebyollin/sdxl-vae-fp16-fix)
import subprocess
import sys
import traceback

try:
    from huggingface_hub import snapshot_download

    print("Downloading RORem weights (~6.5 GB)...", flush=True)
    rorem_path = snapshot_download(
        repo_id="LetsThink/RORem",
        local_dir="/weights/RORem",
        # Качаем только safetensors+конфиги; pickle .bin не нужны
        ignore_patterns=["*.msgpack", "*.h5", "*.bin", "tf_model.h5", "flax_model.msgpack"],
        max_workers=4,
    )
    print(f"  → {rorem_path}", flush=True)

    print("Downloading sdxl-vae-fp16-fix (~335 MB)...", flush=True)
    vae_path = snapshot_download(
        repo_id="madebyollin/sdxl-vae-fp16-fix",
        local_dir="/weights/sdxl-vae-fp16-fix",
        ignore_patterns=["*.msgpack", "*.h5", "*.bin"],
        max_workers=4,
    )
    print(f"  → {vae_path}", flush=True)

    print("Sizes:")
    subprocess.run(["du", "-sh", "/weights/RORem"])
    subprocess.run(["du", "-sh", "/weights/sdxl-vae-fp16-fix"])
except Exception:
    traceback.print_exc()
    sys.exit(1)
