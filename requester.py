import requests

url = "http://localhost:8000/generate"
files = {
    "logo": open(r".\logo.jpg", "rb"),
    "product": open(r".\product.jpg", "rb"),
}
data = {"brand_name": "MyBrand", "style": "minimal", "count": "4"}

with requests.post(url, files=files, data=data, stream=True) as r:
    r.raise_for_status()
    with open("MyBrand_creatives.zip", "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)