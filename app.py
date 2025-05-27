import gradio as gr
from loadimg import load_img
import spaces
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms

torch.set_float32_matmul_precision(["high", "highest"][0])

birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
birefnet.to("cuda")

transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

def fn(image):
    im = load_img(image, output_type="pil")
    im = im.convert("RGB")
    origin = im.copy()
    processed_image = process(im)
    return (processed_image, origin)

@spaces.GPU
def process(image):
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to("cuda")
    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)
    return image

def process_file(f):
    name_path = f.rsplit(".", 1)[0] + ".png"
    im = load_img(f, output_type="pil")
    im = im.convert("RGB")
    transparent = process(im)
    transparent.save(name_path)
    return name_path

slider1 = gr.ImageSlider(label="Processed Image", type="pil", format="png")
slider2 = gr.ImageSlider(label="Processed Image from URL", type="pil", format="png")
image_upload = gr.Image(label="Upload an image")
image_file_upload = gr.Image(label="Upload an image", type="filepath")
url_input = gr.Textbox(label="Paste an image URL")
output_file = gr.File(label="Output PNG File")

# Example images
chameleon = load_img("butterfly.jpg", output_type="pil")
url_example = "https://hips.hearstapps.com/hmg-prod/images/gettyimages-1229892983-square.jpg"

tab1 = gr.Interface(fn, inputs=image_upload, outputs=slider1, examples=[chameleon], api_name="image")
tab2 = gr.Interface(fn, inputs=url_input, outputs=slider2, examples=[url_example], api_name="text")
tab3 = gr.Interface(process_file, inputs=image_file_upload, outputs=output_file, examples=["butterfly.jpg"], api_name="png")

demo = gr.TabbedInterface(
    [tab1, tab2, tab3], ["Image Upload", "URL Input", "File Output"], title="Background Removal Tool"
)

if __name__ == "__main__":
    demo.launch(show_error=True)