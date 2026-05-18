# Runpod Serverless image для SmartEraser inpainting.
# Зеркало Modal-эндпоинта smarteraser: 512×512 padding-mode, 50 steps, guidance=7.5.
#
# Особенности:
# - Custom pipeline StableDiffusionInpaintRegionPipeline + CLIPVisualPrompt (git clone)
# - CLIP-ViT-Large-Patch14 нужен для visual prompts (~1.5GB)
# - SE веса с публичного HF repo Nikita12312425345/smarteraser-weights (~5GB)
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONPATH=/opt/SmartEraser/Model_framework

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip git libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 --version

# Клонируем upstream SmartEraser для custom modules (CLIPVisualPrompt + pipeline).
RUN git clone https://github.com/longtaojiang/SmartEraser.git /opt/SmartEraser

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --upgrade pip \
    && pip3 install -r requirements.txt \
    && pip3 install hf_transfer==0.1.8

# Baked весов:
# 1. SE main weights (UNet/VAE/text_encoder/etc + clip_mlp_weight.pth) → /weights
# 2. CLIP-ViT-Large-Patch14 для CLIPVisualPrompt → /opt/SmartEraser/Model_framework/ckpts/
COPY bake_weights.py .
RUN mkdir -p /weights /opt/SmartEraser/Model_framework/ckpts && \
    python3 -u bake_weights.py

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
