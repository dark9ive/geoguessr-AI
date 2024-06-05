from flask import Flask, request, jsonify, send_from_directory
from torchvision import datasets, transforms
from PIL import Image
import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTFeatureExtractor
import argparse

app = Flask(__name__)

parser = argparse.ArgumentParser(description="Demo site for image classification.")
parser.add_argument("--model", "-m", type=str, default="", help="Path to saved model")
args = parser.parse_args()
model_file = args.model

# Load the dataset to get class labels
train_data_path = 'data'
train_dataset = datasets.ImageFolder(root=train_data_path)
class_to_idx = train_dataset.class_to_idx
class_labels = list(class_to_idx.keys())

print(class_labels)

# 設定參數
img_height, img_width = 224, 224
num_classes = len(class_labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

# Load the pretrained ViT model and feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-large-patch16-224-in21k')
model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224-in21k', num_labels=num_classes)

model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    model.classifier,
    nn.Dropout(0.3)
)
model = model.to(device)

# Load model
model.load_state_dict(torch.load(f'{model_file}'))
model.eval()

# Define image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])


@app.route("/")
@app.route("/index.html")
def index():
    return send_from_directory("html", "index.html")

@app.route("/scripts.js")
def js():
    return send_from_directory("js", "scripts.js")

@app.route("/style.css")
def css():
    return send_from_directory("css", "style.css")

@app.route("/api/prediction", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    # Read the image file
    image_file = request.files["image"]
    
    # Open and preprocess the image
    image = Image.open(image_file).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    input_tensor = input_tensor.to(device)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_class_idx = torch.argmax(outputs.logits, dim=1).item()
        predicted_label = class_labels[predicted_class_idx]
    
    # Return the predicted label as JSON response
    return jsonify({"prediction": predicted_label})

if __name__ == "__main__":
    app.run(debug=True, port=56400)

