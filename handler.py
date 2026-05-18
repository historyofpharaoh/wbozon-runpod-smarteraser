# Runpod Serverless handler для SmartEraser inpainting.
# Зеркало Modal-эндпоинта smarteraser (apples-to-apples).
# 512×512 padding-mode, 50 steps, guidance=7.5.
# Composite уже встроен в pipeline через crop+resize+blend в конце.

import base64
import io
import os
import sys
import time

# SmartEraser custom modules (CLIPVisualPrompt + custom pipeline)
sys.path.insert(0, "/opt/SmartEraser/Model_framework")

import numpy as np
import requests
import runpod
import torch
from PIL import Image, ImageFilter, ImageOps

# Загрузим в _load_pipe()
_pipe = None
_clip_model = None
_clip_processor = None
_tokenizer = None


def _load_pipe():
    global _pipe, _clip_model, _clip_processor, _tokenizer
    if _pipe is not None:
        return
    t0 = time.time()
    from modules.clip_visual_token import CLIPVisualPrompt
    from modules.pipeline.pipeline_stable_diffusion_inpaint_region import StableDiffusionInpaintRegionPipeline
    from transformers import CLIPTokenizer, CLIPImageProcessor

    checkpoint_dir = "/weights"
    clip_dir = "/opt/SmartEraser/Model_framework/ckpts/clip-vit-large-patch14"
    weight_dtype = torch.float16

    _pipe = StableDiffusionInpaintRegionPipeline.from_pretrained(
        checkpoint_dir, torch_dtype=weight_dtype
    ).to("cuda")

    _clip_model = CLIPVisualPrompt(clip_dir)
    _clip_processor = CLIPImageProcessor.from_pretrained(clip_dir)
    _tokenizer = CLIPTokenizer.from_pretrained(clip_dir)
    _clip_model.load_mlp_weight(os.path.join(checkpoint_dir, "clip_mlp_weight.pth"))
    for m in [_clip_model.vision_model, _clip_model.text_model, _clip_model.clip_mlp]:
        m.eval()
        m.to("cuda")

    alloc_gb = torch.cuda.memory_allocated() / 1e9
    print(f"[SE] loaded in {time.time()-t0:.1f}s, VRAM={alloc_gb:.2f}GB", flush=True)


def handler(event):
    payload = event.get("input", {}) or {}
    image_url = payload.get("image_url")
    mask_url = payload.get("mask_url")
    if not image_url or not mask_url:
        return {"error": "image_url and mask_url required"}

    num_steps = int(payload.get("num_steps", 50))
    guidance_scale = float(payload.get("guidance_scale", 7.5))

    _load_pipe()

    try:
        image_pil = Image.open(io.BytesIO(requests.get(image_url, timeout=30).content)).convert("RGB")
        mask_pil = Image.open(io.BytesIO(requests.get(mask_url, timeout=30).content)).convert("L")
    except Exception as e:
        return {"error": f"Download failed: {e}"}

    if mask_pil.size != image_pil.size:
        mask_pil = mask_pil.resize(image_pil.size, Image.NEAREST)
    mask_np = np.array(mask_pil)
    mask_np = (mask_np > 127).astype(np.uint8) * 255
    if mask_np.sum() == 0:
        return {"error": "Mask is empty"}
    mask_pil_bin = Image.fromarray(mask_np, mode="L")

    # SmartEraser inference: padding mode до 512×512 (как в их app_remove.py)
    ori_image = image_pil
    ori_mask = mask_pil_bin
    original_size = ori_image.size

    W, H = ori_image.size
    if W > H:
        scale = 512 / W
        new_w, new_h = 512, int(H * scale)
    else:
        scale = 512 / H
        new_w, new_h = int(W * scale), 512
    image_resized = ori_image.resize((new_w, new_h), Image.BILINEAR)
    mask_resized = ori_mask.resize((new_w, new_h), Image.NEAREST)
    pad_w = (512 - new_w) // 2
    pad_h = (512 - new_h) // 2
    input_img = ImageOps.expand(image_resized, (pad_w, pad_h, 512 - new_w - pad_w, 512 - new_h - pad_h), (255, 255, 255))
    input_mask = ImageOps.expand(mask_resized, (pad_w, pad_h, 512 - new_w - pad_w, 512 - new_h - pad_h), 0)

    # CLIP visual prompt — вырезаем masked region на белом фоне
    img_arr = np.array(input_img)
    mask_arr = np.array(input_mask)
    white_bg = np.ones_like(img_arr) * 255
    paste = np.where(mask_arr[..., None] > 127, img_arr, white_bg).astype(np.uint8)
    paste_clip_image = Image.fromarray(paste)

    paste_clip_image = _clip_processor(images=paste_clip_image, return_tensors="pt")["pixel_values"]
    vtoken_prompt = _tokenizer("Remove the instance of", padding="max_length", max_length=7, truncation=True, return_tensors="pt")["input_ids"]
    uncond_vtoken_prompt = _tokenizer("", padding="max_length", max_length=7, truncation=True, return_tensors="pt")["input_ids"]

    prompt_emb, uncond_emb = _clip_model.inference_vtoken(
        vtoken_prompt.to(device=_pipe.device),
        uncond_vtoken_prompt.to(device=_pipe.device),
        paste_clip_image.to(device=_pipe.device),
        _pipe.text_encoder,
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = _pipe(
        prompt_embeds=prompt_emb,
        negative_prompt_embeds=uncond_emb,
        image=input_img,
        mask_image=input_mask,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
    ).images[0]
    infer_s = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"[SE] steps={num_steps} infer={infer_s:.2f}s peak_VRAM={peak_gb:.2f}GB", flush=True)

    # Crop back from padding → resize to original size → composite через blurred mask
    result_cropped = result.crop((pad_w, pad_h, pad_w + new_w, pad_h + new_h))
    result_final = result_cropped.resize(original_size, Image.BILINEAR)
    ori_mask_blur = ori_mask.filter(ImageFilter.GaussianBlur(radius=5))
    result_composite = Image.composite(result_final, ori_image, ori_mask_blur)

    buf = io.BytesIO()
    result_composite.save(buf, format="PNG", optimize=False)

    return {
        "result_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "infer_s": round(infer_s, 2),
        "peak_vram_gb": round(peak_gb, 2),
        "config": {
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
            "padding_mode": True,
        },
    }


# Прелоад
_load_pipe()

runpod.serverless.start({"handler": handler})
