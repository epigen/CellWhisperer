# TODO this is boilerplate code and needs to be adjusted to work with our geneformer-biogpt model
from transformers import TranscriptomeTextDualEncoderModel, AutoTokenizer


# # loading model and config from pretrained folder
model = TranscriptomeTextDualEncoderModel.from_pretrained("geneformer-biogpt")
tokenizer = AutoTokenizer.from_pretrained("biogpt")

inputs = tokenizer(
    ["una foto di un gatto", "una foto di un cane"], padding=True, return_tensors="pt"
)
model_out, text_features = model.get_text_features(**inputs)

from PIL import Image
import requests
from transformers import TranscriptomeTextDualEncoderModel, AutoImageProcessor

model = TranscriptomeTextDualEncoderModel.from_pretrained("clip-italian/clip-italian")
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = image_processor(images=image, return_tensors="pt")

image_features = model.get_image_features(**inputs)
