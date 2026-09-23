import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from model import MNISTCNN

MODEL_PATH = "models/mnist_cnn.pth"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Same preprocessing used during training/evaluation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST test dataset
    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    # Load model
    model = MNISTCNN().to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )
    model.eval()

    # Select one test image
    image, true_label = test_dataset[0]

    # Add batch dimension: [1, 1, 28, 28]
    input_image = image.unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(input_image)

        probabilities = torch.softmax(output, dim=1)
        predicted_label = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, predicted_label].item()

    print(f"True label:      {true_label}")
    print(f"Predicted label: {predicted_label}")
    print(f"Confidence:      {confidence * 100:.2f}%")

    # Display image
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(
        f"True: {true_label} | "
        f"Predicted: {predicted_label} | "
        f"Confidence: {confidence * 100:.2f}%"
    )
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()