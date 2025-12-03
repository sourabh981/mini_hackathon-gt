"""
app.py
Local Auto-Creative Engine (Stable Diffusion + GPT4All)
- No API keys required (models are local)
- Endpoint: POST /generate
  - form fields:
      brand_name (str)
      style (str) optional: minimal|bold|lifestyle|luxury
      count (int) optional (default 4)
      logo (file) required
      product (file) required
  - returns: application/zip (images/ + captions.csv)
"""

import io
import os
import zipfile
import json
from typing import List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image, ImageOps
import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path

# try import gpt4all; if not installed, we'll fallback later
try:
    from gpt4all import GPT4All
    HAVE_GPT4ALL = True
except Exception:
    GPT4All = None
    HAVE_GPT4ALL = False

app = FastAPI(title="Auto-Creative Engine (Local)")

# -----------------------
# Config: local model paths
# -----------------------
SD_MODEL_DIR = Path("./models/stable-diffusion")   # place SD model files here
GPT4ALL_MODEL_PATH = Path("./models/ggml-gpt4all.bin")  # place GPT4All binary here (change name if needed)

# -----------------------
# Basic prompts & styles
# -----------------------
STYLE_MAP = {
    "minimal": "clean minimal aesthetic, lots of negative space",
    "bold": "vibrant colors, strong contrast, energetic",
    "lifestyle": "natural light, candid feeling, cozy",
    "luxury": "matte finish, premium feel, soft lighting",
}

PROMPT_TEMPLATES = [
    "A high-resolution product ad for {brand} featuring the product on a clean background, {style_desc}, centered composition, professional photography, 4k",
    "Lifestyle scene showcasing {brand} {product_desc} being used by a person, {style_desc}, cinematic lighting, 4k",
    "Minimalist ad for {brand} product, bold typography space, product floating, {style_desc}, 4k",
]

# -----------------------
# Load Stable Diffusion pipeline (local)
# -----------------------
def load_sd_pipeline(model_dir: Path):
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Stable Diffusion model directory not found at {model_dir}. "
            "Download or put model files (config + weights) there. See README for instructions."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # We attempt to use float16 when GPU is available to save memory
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        safety_checker=None,  # optional: you can enable safety checker if present
    )
    pipe = pipe.to(device)
    # reduce memory/VRAM when possible
    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    return pipe

# -----------------------
# Load GPT4All (optional)
# -----------------------
def load_gpt4all(model_path: Path):
    if not HAVE_GPT4ALL:
        return None
    if not model_path.exists():
        return None
    try:
        llm = GPT4All(model=str(model_path))
        return llm
    except Exception:
        return None

# Try to load models on startup (but allow app to run if they are missing — we provide useful errors)
SD_PIPELINE = None
GPT4ALL = None
try:
    SD_PIPELINE = load_sd_pipeline(SD_MODEL_DIR)
except Exception as e:
    # keep None and raise helpful message when endpoint called
    SD_PIPELINE = None
    SD_LOAD_ERROR = str(e)
else:
    SD_LOAD_ERROR = None

GPT4ALL = load_gpt4all(GPT4ALL_MODEL_PATH)

# -----------------------
# Utilities
# -----------------------
def overlay_logo(img: Image.Image, logo_bytes: bytes, scale: float = 0.12, corner: str = "bottom-right") -> Image.Image:
    """Overlay logo (bytes) onto img (PIL.Image) at chosen corner."""
    logo = Image.open(io.BytesIO(logo_bytes)).convert("RGBA")
    img = img.convert("RGBA")
    w, h = img.size

    # scale logo proportionally
    max_logo_w = int(w * scale)
    lw, lh = logo.size
    ratio = max_logo_w / float(lw)
    new_size = (max_logo_w, int(lh * ratio))
    logo = logo.resize(new_size, Image.LANCZOS)

    margin = int(w * 0.03)
    if corner == "bottom-right":
        pos = (w - logo.width - margin, h - logo.height - margin)
    elif corner == "top-left":
        pos = (margin, margin)
    elif corner == "top-right":
        pos = (w - logo.width - margin, margin)
    else:
        pos = (margin, h - logo.height - margin)

    # paste with alpha
    img.paste(logo, pos, logo)
    return img.convert("RGB")

def safe_image_bytes(pil_img: Image.Image, fmt: str = "PNG"):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt, dpi=(300,300))
    buf.seek(0)
    return buf.read()

# -----------------------
# Caption generation
# -----------------------
def generate_captions_with_gpt4all(llm, brand: str, product_desc: str, style: str, filenames: List[str]) -> dict:
    """
    Ask GPT4All to produce a JSON mapping of filename -> caption.
    If parsing fails, fallback to a line list.
    """
    # create a compact prompt that asks for JSON only
    filenames_str = ", ".join(filenames)
    prompt = (
        f"You are a concise marketing copywriter. For each filename in the list: {filenames_str}, "
        f"create one short ad caption (maximum 12 words). Brand: {brand}. Product: {product_desc}. Style: {style}.\n"
        "Respond only with a JSON object mapping filename to caption, e.g. {\"creative_1.png\": \"Caption 1\", ...}."
    )

    # GPT4All usage: generate text
    try:
        resp = llm.generate(prompt, max_tokens=200)
        text = resp if isinstance(resp, str) else str(resp)
        # Try to extract JSON substring
        # naive attempt: find first '{' and last '}' and parse
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end+1]
            data = json.loads(json_str)
            # ensure mapping includes our filenames (fallback if not)
            return {fn: data.get(fn, data.get(fn.replace(".png",""), f"{brand} — {product_desc}")) for fn in filenames}
    except Exception:
        pass

    # fallback: produce predictable captions
    return {fn: f"{brand} — {product_desc}" for fn in filenames}

def generate_simple_captions(brand: str, product_desc: str, style: str, filenames: List[str]) -> dict:
    return {fn: f"{brand} — {product_desc}" for fn in filenames}

# -----------------------
# API endpoint
# -----------------------
@app.post("/generate")
async def generate(
    logo: UploadFile = File(...),
    product: UploadFile = File(...),
    brand_name: str = Form(...),
    style: str = Form("minimal"),
    count: int = Form(4),
):
    """
    Generate `count` creatives (default 4) using local Stable Diffusion and produce captions using GPT4All (if available).
    Returns a ZIP file: images/* and captions.csv
    """
    # Validate model availability
    if SD_PIPELINE is None:
        msg = (
            "Stable Diffusion model not loaded. "
            "Ensure you placed model files under ./models/stable-diffusion/ and restart the app.\n"
            f"Load error: {SD_LOAD_ERROR}"
        )
        raise HTTPException(status_code=500, detail=msg)

    # Validate inputs
    if count < 1 or count > 20:
        raise HTTPException(status_code=400, detail="count must be between 1 and 20")

    try:
        logo_bytes = await logo.read()
        product_bytes = await product.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading uploaded files: {e}")

    # basic product description (could be improved by local image captioning)
    product_desc = "product"

    style_desc = STYLE_MAP.get(style, STYLE_MAP["minimal"])

    filenames = []
    images_bytes = []

    # Generate images
    for i in range(count):
        template = PROMPT_TEMPLATES[i % len(PROMPT_TEMPLATES)]
        prompt = template.format(brand=brand_name, product_desc=product_desc, style_desc=style_desc)
        prompt = f"{prompt}, variation {i+1}, photorealistic, product-focused, high detail"

        # run the pipeline
        # Avoid passing negative prompts or complex arguments to keep compatibility
        with torch.autocast("cuda") if torch.cuda.is_available() else torch.no_grad():
            out = SD_PIPELINE(prompt, guidance_scale=7.5, num_inference_steps=25)
            pil_img = out.images[0]

        # overlay logo
        pil_img = overlay_logo(pil_img, logo_bytes, scale=0.14)

        # store bytes
        name = f"creative_{i+1}.png"
        b = safe_image_bytes(pil_img, fmt="PNG")
        filenames.append(name)
        images_bytes.append((name, b))

    # Generate captions
    if GPT4ALL is not None:
        try:
            captions_map = generate_captions_with_gpt4all(GPT4ALL, brand_name, product_desc, style, filenames)
        except Exception:
            captions_map = generate_simple_captions(brand_name, product_desc, style, filenames)
    else:
        captions_map = generate_simple_captions(brand_name, product_desc, style, filenames)

    # Create zip in memory
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, b in images_bytes:
            zf.writestr(f"images/{name}", b)
        # captions CSV
        csv_buf = io.StringIO()
        csv_buf.write("filename,caption\n")
        for name in filenames:
            caption = captions_map.get(name, "")
            # escape commas and quotes
            safe_cap = caption.replace('"', '""')
            csv_buf.write(f'"{name}","{safe_cap}"\n')
        zf.writestr("captions.csv", csv_buf.getvalue())
    mem_zip.seek(0)

    return StreamingResponse(mem_zip, media_type="application/zip",
                             headers={"Content-Disposition": f"attachment; filename={brand_name}_creatives.zip"})