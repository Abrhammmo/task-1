import os
import sys

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Allow Python to find src/model/model.py
sys.path.append("./src/model")

from model import MNISTCNN


MODEL_PATH = "./models/mnist_cnn.pth"
OUTPUT_DIR = "./adversarial_examples/fgsm"

# Start small. We will experiment with this later.
EPSILON = 0.8


def fgsm_attack(image, epsilon, gradient):
    """
    Generate an adversarial example using FGSM.

    image: original input image
    epsilon: attack strength
    gradient: gradient of the loss with respect to the image
    """

    # Take only the direction of the gradient
    gradient_sign = gradient.sign()

    # Move the image in that direction
    adversarial_image = image + epsilon * gradient_sign

    return adversarial_image


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Same preprocessing used when training the model
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

    # Use the same image we tested previously
    image, true_label = test_dataset[0]

    # Add batch dimension
    image = image.unsqueeze(0).to(device)

    # Make sure the input can receive gradients
    image.requires_grad = True

    # Load model
    model = MNISTCNN().to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )
    model.eval()

    # --------------------------------------------------
    # STEP 1: Make prediction on clean image
    # --------------------------------------------------

    output = model(image)

    clean_prediction = output.argmax(dim=1).item()

    clean_probabilities = torch.softmax(output, dim=1)
    clean_confidence = clean_probabilities[
        0, clean_prediction
    ].item()

    print("\n--- CLEAN IMAGE ---")
    print(f"True label:       {true_label}")
    print(f"Prediction:       {clean_prediction}")
    print(f"Confidence:       {clean_confidence * 100:.4f}%")

    # --------------------------------------------------
    # STEP 2: Calculate loss
    # --------------------------------------------------

    loss_function = nn.CrossEntropyLoss()

    label = torch.tensor(
        [true_label],
        device=device
    )

    loss = loss_function(output, label)

    # --------------------------------------------------
    # STEP 3: Calculate gradient
    # --------------------------------------------------

    model.zero_grad()

    loss.backward()

    gradient = image.grad.data

    # --------------------------------------------------
    # STEP 4: Generate adversarial image
    # --------------------------------------------------

    adversarial_image = fgsm_attack(
        image,
        EPSILON,
        gradient
    )

    # --------------------------------------------------
    # STEP 5: Test adversarial image
    # --------------------------------------------------

    adversarial_output = model(
        adversarial_image.detach()
    )

    adversarial_prediction = adversarial_output.argmax(
        dim=1
    ).item()

    adversarial_probabilities = torch.softmax(
        adversarial_output,
        dim=1
    )

    adversarial_confidence = adversarial_probabilities[
        0,
        adversarial_prediction
    ].item()

    print("\n--- ADVERSARIAL IMAGE ---")
    print(f"True label:       {true_label}")
    print(f"Prediction:       {adversarial_prediction}")
    print(
        f"Confidence:       "
        f"{adversarial_confidence * 100:.4f}%"
    )

    attack_success = (
        adversarial_prediction != true_label
    )

    print("\n--- ATTACK RESULT ---")
    print(f"Epsilon:          {EPSILON}")
    print(f"Attack successful: {attack_success}")

    # --------------------------------------------------
    # STEP 6: Save images
    # --------------------------------------------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    clean_path = os.path.join(
        OUTPUT_DIR,
        "clean.png"
    )

    adversarial_path = os.path.join(
        OUTPUT_DIR,
        "fgsm.png"
    )

    # Remove batch dimension for visualization
    clean_display = image.detach().cpu().squeeze()
    adversarial_display = (
        adversarial_image.detach().cpu().squeeze()
    )

    plt.figure()
    plt.imshow(clean_display, cmap="gray")
    plt.title(
        f"Clean | True: {true_label} | "
        f"Predicted: {clean_prediction}"
    )
    plt.axis("off")
    plt.savefig(clean_path)
    plt.close()

    plt.figure()
    plt.imshow(adversarial_display, cmap="gray")
    plt.title(
        f"FGSM | True: {true_label} | "
        f"Predicted: {adversarial_prediction}"
    )
    plt.axis("off")
    plt.savefig(adversarial_path)
    plt.close()

    print("\nImages saved:")
    print(clean_path)
    print(adversarial_path)


if __name__ == "__main__":
    main()