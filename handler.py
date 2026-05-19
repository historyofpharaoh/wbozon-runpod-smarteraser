# Runpod Serverless handler для RORem inpainting (object removal).
#
# Модель: LetsThink/RORem (CVPR 2025) — SDXL-inpaint fine-tune, авторы дообучили
# на задачу убирания объектов с human-in-the-loop.
# Стратегия: HD_CROP (bbox маски + margin → diffuse → paste обратно) +
# Poisson blend (cv2.seamlessClone NORMAL_CLONE поверх TELEA pre-fill).
#
# Контракт API (совпадает с тем что шлёт apps/api/src/services/runpodInpaint.ts):
#   input:  { image_url, mask_url }
#   output: { result_b64 }  # PNG base64
#
# Параметры RORem (НЕ менять — рекомендации авторов):
#   guidance_scale = 1.0       (выше → галлюцинации)
#   strength       = 0.99      (ниже → объект остаётся призраком)
#   num_inference_steps = 20   (компромисс качество/скорость)
#   VAE = madebyollin/sdxl-vae-fp16-fix  (стоковый SDXL VAE в fp16 даёт color drift)
#   HD_CROP margin=96, max_fraction=0.5
#   Poisson: NORMAL_CLONE, ERODE=6, PAD=96, TELEA pre-fill, feather_px=0

import base64
import io
import os
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import requests
import runpod
import torch
from PIL import Image

# ─── Параметры RORem (из rorem_standalone — НЕ менять) ────────────────────────

DEFAULT_PROMPT = (
    "4K, high quality, masterpiece, Highly detailed, Sharp focus, "
    "Professional, photorealistic, realistic"
)
DEFAULT_NEG_PROMPT = (
    "low quality, worst, bad proportions, blurry, extra finger, "
    "Deformed, disfigured, unclear background"
)

SIZE_MULT = 8           # SDXL UNet требует image dims % 8 == 0
INFER_MAX_SIDE = 1024   # SDXL native resolution
MARGIN = 96             # HD_CROP — контекст вокруг bbox маски (px)
MAX_FRACTION = 0.5      # HD_CROP отключается если маска > 50% площади
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 1.0
STRENGTH = 0.99
SEED = 42
FEATHER_PX = 0          # 0 = hard composite

# ─── Загрузка модели (один раз при старте воркера) ────────────────────────────

WEIGHTS_DIR = Path(os.environ.get("WEIGHTS_DIR", "/weights"))
RUNPOD_DOWNLOAD_TIMEOUT = 30  # сек на скачивание image/mask с S3

_pipe = None  # глобальный кэш pipeline


def _load_pipe() -> None:
    global _pipe
    if _pipe is not None:
        return
    from diffusers import StableDiffusionXLInpaintPipeline, AutoencoderKL

    t0 = time.time()
    rorem_path = WEIGHTS_DIR / "RORem"
    vae_path = WEIGHTS_DIR / "sdxl-vae-fp16-fix"

    if not (rorem_path / "model_index.json").exists():
        raise FileNotFoundError(f"RORem weights not found at {rorem_path}")
    if not vae_path.exists():
        raise FileNotFoundError(f"sdxl-vae-fp16-fix not found at {vae_path}")

    dtype = torch.float16
    # sdxl-vae-fp16-fix — стоковый SDXL VAE дрейфит в fp16; этот дообучен быть стабильным
    vae = AutoencoderKL.from_pretrained(str(vae_path), torch_dtype=dtype)

    _pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        str(rorem_path),
        torch_dtype=dtype,
        use_safetensors=True,
        vae=vae,
    )
    _pipe.to("cuda")
    # VAE slicing — decode по тайлам, -1.5 GB peak. attention_slicing и cpu_offload
    # НЕ включаем (рекомендация авторов: на 24 GB GPU они только замедляют).
    try:
        _pipe.vae.enable_slicing()
    except Exception:
        pass

    alloc_gb = torch.cuda.memory_allocated() / 1e9
    print(f"[RORem] loaded in {time.time()-t0:.1f}s, VRAM={alloc_gb:.2f}GB", flush=True)


# ─── HD_CROP + Poisson helpers (1-в-1 из rorem_standalone/utils.py) ───────────

def resize_to_multiple(w: int, h: int, mult: int = 8, max_side: int | None = None):
    if max_side is not None and max(w, h) > max_side:
        scale = max_side / max(w, h)
        w = int(round(w * scale))
        h = int(round(h * scale))
    nw = max(mult, (w // mult) * mult)
    nh = max(mult, (h // mult) * mult)
    return nw, nh


def poisson_blend(
    result: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    feather_px: int = 0,
) -> np.ndarray:
    binary = (mask > 127).astype(np.uint8) * 255
    if binary.sum() == 0:
        return original.copy()

    src_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    dst_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)

    PAD = 96
    src_p = cv2.copyMakeBorder(src_bgr, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
    dst_p = cv2.copyMakeBorder(dst_bgr, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
    mask_p = cv2.copyMakeBorder(binary, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=0)

    # TELEA pre-fill: убирает объект из dst, чтобы Poisson на границе тянулся к фону
    # (а не к цвету самого объекта). NS-метод даёт тёмные дорожки — не использовать.
    clean_p = cv2.inpaint(dst_p, mask_p, 3, cv2.INPAINT_TELEA)

    ERODE = 6
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ERODE * 2 + 1, ERODE * 2 + 1))
    mask_eroded = cv2.erode(mask_p, k, iterations=1)

    # cv2.seamlessClone крашится если маска касается края padded-кадра — обнуляем 2 px по бордеру
    binary_safe = mask_eroded.copy()
    binary_safe[:2, :] = 0
    binary_safe[-2:, :] = 0
    binary_safe[:, :2] = 0
    binary_safe[:, -2:] = 0

    if binary_safe.sum() == 0:
        # Маска слишком маленькая после erode → чистый TELEA-fill вместо Poisson
        clean_bgr = clean_p[PAD:PAD + original.shape[0], PAD:PAD + original.shape[1]]
        clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
        m3 = np.stack([binary // 255] * 3, axis=-1)
        return (clean_rgb * m3 + original * (1 - m3)).astype(np.uint8)

    ys, xs = np.where(binary_safe > 0)
    cx = int((xs.min() + xs.max()) // 2)
    cy = int((ys.min() + ys.max()) // 2)
    # NORMAL_CLONE — переносит ТОЛЬКО градиенты source. MIXED_CLONE тянет
    # оригинальные пиксели сквозь маску — для object removal не годится.
    blended_p = cv2.seamlessClone(src_p, clean_p, binary_safe, (cx, cy), cv2.NORMAL_CLONE)

    h, w = original.shape[:2]
    blended_bgr = blended_p[PAD:PAD + h, PAD:PAD + w]
    blended_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)

    if feather_px > 0:
        sigma = max(0.5, feather_px / 3.0)
        alpha = cv2.GaussianBlur(
            binary.astype(np.float32) / 255.0, (0, 0), sigmaX=sigma, sigmaY=sigma,
        ).clip(0.0, 1.0)
        alpha3 = np.stack([alpha] * 3, axis=-1)
        out = blended_rgb.astype(np.float32) * alpha3 + original.astype(np.float32) * (1.0 - alpha3)
        return out.clip(0, 255).astype(np.uint8)

    m3 = np.stack([binary // 255] * 3, axis=-1)
    return (blended_rgb * m3 + original * (1 - m3)).astype(np.uint8)


def hd_crop_pipeline(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    diffuse_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    margin: int = MARGIN,
    max_fraction: float = MAX_FRACTION,
    feather_px: int = FEATHER_PX,
) -> np.ndarray:
    h0, w0 = image_rgb.shape[:2]
    binary = mask > 127
    if not binary.any():
        return image_rgb.copy()

    ys, xs = np.where(binary)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())

    y0 = max(0, y_min - margin)
    y1 = min(h0, y_max + margin + 1)
    x0 = max(0, x_min - margin)
    x1 = min(w0, x_max + margin + 1)

    crop_area = (y1 - y0) * (x1 - x0)
    if crop_area / (h0 * w0) > max_fraction:
        # Маска большая — кропить нет смысла, прогоняем целиком
        result = diffuse_fn(image_rgb, mask)
        return poisson_blend(result, image_rgb, mask, feather_px=feather_px)

    crop_img = image_rgb[y0:y1, x0:x1].copy()
    crop_msk = mask[y0:y1, x0:x1].copy()

    crop_result = diffuse_fn(crop_img, crop_msk)
    if crop_result.shape != crop_img.shape:
        raise ValueError(f"diffuse returned {crop_result.shape}, expected {crop_img.shape}")

    full_result = image_rgb.copy()
    full_result[y0:y1, x0:x1] = crop_result
    return poisson_blend(full_result, image_rgb, mask, feather_px=feather_px)


def _diffuse(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Чистая диффузия RORem: resize → SDXL → resize back."""
    h0, w0 = image_rgb.shape[:2]
    nw, nh = resize_to_multiple(w0, h0, mult=SIZE_MULT, max_side=INFER_MAX_SIDE)

    img_pil = Image.fromarray(image_rgb).resize((nw, nh), Image.LANCZOS)
    # NEAREST для маски обязательно — LANCZOS даёт мягкие края, SDXL inpaint
    # их некорректно интерпретирует.
    msk_pil = Image.fromarray(mask).resize((nw, nh), Image.NEAREST).convert("L")

    # Generator на CPU — детерминизм между запусками
    generator = torch.Generator(device="cpu").manual_seed(SEED)

    with torch.inference_mode():
        out = _pipe(
            prompt=DEFAULT_PROMPT,
            negative_prompt=DEFAULT_NEG_PROMPT,
            image=img_pil,
            mask_image=msk_pil,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            strength=STRENGTH,
            width=nw,
            height=nh,
            output_type="pil",
            generator=generator,
        ).images[0]

    if (nw, nh) != (w0, h0):
        out = out.resize((w0, h0), Image.LANCZOS)
    return np.array(out.convert("RGB"))


# ─── Runpod handler ───────────────────────────────────────────────────────────

def handler(event):
    payload = (event or {}).get("input", {}) or {}
    image_url = payload.get("image_url")
    mask_url = payload.get("mask_url")
    if not image_url or not mask_url:
        return {"error": "image_url and mask_url required"}

    _load_pipe()  # no-op если уже загружено

    # Качаем входы с presigned URL'ов нашего S3
    try:
        image_pil = Image.open(io.BytesIO(
            requests.get(image_url, timeout=RUNPOD_DOWNLOAD_TIMEOUT).content
        )).convert("RGB")
        mask_pil = Image.open(io.BytesIO(
            requests.get(mask_url, timeout=RUNPOD_DOWNLOAD_TIMEOUT).content
        )).convert("L")
    except Exception as e:
        return {"error": f"Download failed: {e}"}

    if mask_pil.size != image_pil.size:
        mask_pil = mask_pil.resize(image_pil.size, Image.NEAREST)

    image_rgb = np.array(image_pil)
    mask = np.array(mask_pil, dtype=np.uint8)
    # Бинаризуем (на случай антиалиасинга / серого PNG из браузера)
    mask = (mask > 127).astype(np.uint8) * 255
    if mask.sum() == 0:
        return {"error": "Mask is empty"}

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = hd_crop_pipeline(image_rgb, mask, _diffuse, margin=MARGIN, feather_px=FEATHER_PX)
    infer_s = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"[RORem] infer={infer_s:.2f}s peak_VRAM={peak_gb:.2f}GB", flush=True)

    buf = io.BytesIO()
    Image.fromarray(result).save(buf, format="PNG", optimize=False)

    return {
        "result_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "infer_s": round(infer_s, 2),
        "peak_vram_gb": round(peak_gb, 2),
        "config": {
            "model": "rorem",
            "num_steps": NUM_INFERENCE_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "strength": STRENGTH,
            "hd_crop_margin": MARGIN,
            "feather_px": FEATHER_PX,
        },
    }


# Прелоад модели при старте контейнера (cold start включает loading в этой строке).
_load_pipe()

runpod.serverless.start({"handler": handler})
