Auto-Creative Engine

A fully offline Auto-Creative Engine that generates multiple ad creatives for a brand/product using local models:

Stable Diffusion (local) → image generation

GPT4All (local) → caption generation

No API keys required

Generates a ZIP containing images and captions.csv.

Features

Upload product image + brand logo

Generate multiple image variations (default 4, adjustable 1–20)

Overlay brand logo on each image

Generate catchy captions using local GPT4All (or fallback simple captions)

Return a ZIP file with images + captions

Requirements

Python 3.10+

GPU recommended for Stable Diffusion; CPU works but slow

Disk space for local models

Installation

Clone or download repo.

Install dependencies:

pip install -r requirements.txt


Download Stable Diffusion model files and place under ./models/stable-diffusion/:

models/stable-diffusion/config.json
models/stable-diffusion/pytorch_model.bin or .safetensors
models/stable-diffusion/tokenizer files...


Download GPT4All binary and place under ./models/ggml-gpt4all.bin (optional):

https://gpt4all.io/models/

Running the App
uvicorn app:app --reload --port 8000


Server runs at http://127.0.0.1:8000.

Usage
API Endpoint: /generate (POST)

Form data:

Field	Type	Description
brand_name	str	Brand name
style	str	Optional: minimal, bold, lifestyle, luxury (default: minimal)
count	int	Optional: 1–20 (default: 4)
logo	file	Brand logo image (PNG/JPG)
product	file	Product image (PNG/JPG)

Returns: application/zip containing:

images/creative_1.png ... creative_N.png

captions.csv

Example curl
curl -X POST "http://127.0.0.1:8000/generate" \
  -F "brand_name=Acme" \
  -F "style=minimal" \
  -F "count=4" \
  -F "logo=@/path/to/logo.png" \
  -F "product=@/path/to/product.jpg" \
  --output acme_creatives.zip

Notes & Tips

Model Loading: If Stable Diffusion fails, the endpoint will return an error with guidance.

Memory: GPU recommended. CPU works but slow and may need reduced count.

Captions: GPT4All optional; fallback captions are simple defaults.

Prompt Customization: You can edit PROMPT_TEMPLATES in app.py to adjust style and description.

Project Structure
.
├── app.py                 # FastAPI backend
├── requirements.txt       # Python dependencies
├── models/                # Local model files
│   ├── stable-diffusion/
│   └── ggml-gpt4all.bin
└── README.md

License

MIT License (free for hackathon and personal use)