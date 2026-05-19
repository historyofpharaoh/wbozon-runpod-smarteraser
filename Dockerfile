# Runpod Serverless image для RORem inpainting (object removal).
#
# Модель: LetsThink/RORem (SDXL-inpaint fine-tune, CVPR 2025).
# Pipeline: StableDiffusionXLInpaintPipeline + HD_CROP + Poisson blend.
# VAE: madebyollin/sdxl-vae-fp16-fix (стоковый SDXL VAE дрейфит в fp16).
#
# Веса baked в образ через bake_weights.py:
#   /weights/RORem/             — RORem (~6.5 GB)
#   /weights/sdxl-vae-fp16-fix/ — fp16-safe VAE (~335 MB)
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    WEIGHTS_DIR=/weights

# libgl1 + libglib2.0 — для opencv-python (Poisson blend)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip git libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 --version

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --upgrade pip \
    && pip3 install -r requirements.txt \
    && pip3 install hf_transfer==0.1.8

# Baked веса: качаем при сборке, чтобы первый запрос воркера не качал 7 GB с HF.
# FlashBoot потом снэпшотит уже прогруженную в RAM модель.
COPY bake_weights.py .
RUN mkdir -p /weights && python3 -u bake_weights.py

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
