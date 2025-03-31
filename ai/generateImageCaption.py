from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import io

# Load BLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

def generateImageCaption(file):
    try:
        # Read the image file
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Process image through BLIP to generate a caption
        inputs = processor(images=image, return_tensors="pt").to(device)
        output = blip_model.generate(**inputs)
        caption = processor.tokenizer.decode(output[0], skip_special_tokens=True)
        return {"caption": caption, "error": None}

    except Exception as e:
        return {"caption": None, "error": str(e)}

