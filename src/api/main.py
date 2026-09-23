import io
import sys

import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from torchvision import transforms

# Allow Python to find src/model/model.py
sys.path.append("./src/model")

from model import MNISTCNN


MODEL_PATH = "./models/mnist_cnn.pth"

app = FastAPI(
    title="MNIST Model API",
    version="1.0.0"
)


# -----------------------------
# Device
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -----------------------------
# Load model once at startup
# -----------------------------

model = MNISTCNN().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# -----------------------------
# Preprocessing
# -----------------------------

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.1307,),
        (0.3081,)
    )
])


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device)
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # Read uploaded image
    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("L")

    # Apply preprocessing
    tensor = transform(image)

    # Add batch dimension
    tensor = tensor.unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = (
            probabilities.argmax(
                dim=1
            ).item()
        )

        confidence = (
            probabilities[
                0,
                predicted_class
            ].item()
        )

    return {
        "class": predicted_class,
        "confidence": confidence
    }