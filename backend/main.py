from pathlib import Path
import base64
import io
import time

import torch
from diffusers import StableDiffusionPipeline
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

app = FastAPI(title="8-bit Character Generator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
LORA_DIR = BASE_DIR / "lora"
HISTORY_DIR = BASE_DIR / "history"
LORA_DIR.mkdir(exist_ok=True)
HISTORY_DIR.mkdir(exist_ok=True)


def _resolve_lora_path(name: str) -> Path | None:
    candidates = [
        LORA_DIR / f"{name}_8bit_v2.safetensors",
        LORA_DIR / f"{name}_peft",
        LORA_DIR / f"{name}.safetensors",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

LORA_FILES = {
    "public": LORA_DIR / "public_pixel_art.safetensors",
    "custom": None,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32


def _load_pipeline() -> StableDiffusionPipeline:
    common = dict(torch_dtype=DTYPE, safety_checker=None)
    try:
        print(f"Loading model from cache... (device: {DEVICE})")
        p = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            local_files_only=True,
            **common,
        )
        print("Model loaded from cache.")
        return p
    except Exception:
        pass
    print("Cache not found. Downloading from HuggingFace...")
    p = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        **common,
    )
    print("Model downloaded and loaded.")
    return p


pipe = _load_pipeline().to(DEVICE)
pipe.enable_attention_slicing()
if DEVICE == "cpu":
    print("CPU mode: generation may take 3-8 minutes per image")

current_lora: str | None = None


def switch_lora(model_type: str) -> None:
    global current_lora
    if model_type == current_lora:
        return
    if current_lora is not None:
        pipe.unload_lora_weights()
        current_lora = None
    if model_type != "base":
        if model_type == "custom":
            lora_path = _resolve_lora_path("custom")
        else:
            lora_path = LORA_FILES.get(model_type)
        if lora_path is None or not lora_path.exists():
            raise FileNotFoundError(
                f"LoRA '{model_type}' not found in backend/lora/. "
                "Run train_lora.py or place custom_8bit_v2.safetensors there."
            )
        pipe.load_lora_weights(str(lora_path))
        current_lora = model_type


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = (
        "realistic, 3d, blurry, photographic, smooth, noise, "
        "low quality, ugly, deformed, text, watermark"
    )
    steps: int = 25
    model_type: str = "base"
    output_size: int = 128


@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        switch_lora(request.model_type)
        full_prompt = (
            f"pixel art sprite, 8-bit character, {request.prompt}, "
            "NES style, retro game, simple flat colors, pixelated, "
            "white background, sprite sheet character"
        )
        result = pipe(
            prompt=full_prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.steps,
            height=512,
            width=512,
        )
        image: Image.Image = result.images[0]
        size = request.output_size if request.output_size in (80, 128) else 128
        image = image.resize((size, size), Image.NEAREST)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode()
        filename = f"{int(time.time())}_{request.model_type}_{size}px.png"
        (HISTORY_DIR / filename).write_bytes(img_bytes)
        return {
            "image": img_base64,
            "prompt": full_prompt,
            "model": request.model_type,
            "size": size,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "current_lora": current_lora,
        "cuda_available": torch.cuda.is_available(),
        "lora_available": {
            "public": (LORA_DIR / "public_pixel_art.safetensors").exists(),
            "custom": _resolve_lora_path("custom") is not None,
        },
    }


@app.get("/history")
async def get_history():
    files = sorted(HISTORY_DIR.glob("*.png"), reverse=True)[:20]
    result = []
    for f in files:
        img_base64 = base64.b64encode(f.read_bytes()).decode()
        parts = f.stem.split("_")
        result.append({
            "filename": f.name,
            "image": img_base64,
            "model": parts[1] if len(parts) > 1 else "unknown",
        })
    return {"history": result}


frontend_dist = BASE_DIR.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="static")
