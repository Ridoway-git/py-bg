from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
).to(device).eval()

transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def remove_background(image: Image.Image) -> Image.Image:
    original_size = image.size
    input_tensor = transform_image(image).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_mask = model(input_tensor)[-1].sigmoid().cpu()[0, 0]
    mask = transforms.ToPILImage()(pred_mask).resize(original_size).convert("L")
    image.putalpha(mask)
    return image

@app.post("/remove-bg")
async def remove_bg(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read())).convert("RGB")
    result = remove_background(image)
    buf = BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
