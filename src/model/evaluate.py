import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MNISTCNN


MODEL_PATH = "models/mnist_cnn.pth"
BATCH_SIZE = 64


def main():
    # Select CPU or GPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Same preprocessing used during training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load test dataset
    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Create model
    model = MNISTCNN().to(device)

    # Load trained weights
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"Correct predictions: {correct}/{total}")
    print(f"Test accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()