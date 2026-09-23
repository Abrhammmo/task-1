import os
import sys

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

sys.path.append("./src/model")

from model import MNISTCNN


MODEL_PATH = "./models/mnist_cnn.pth"

EXAMPLE_DIR = "./adversarial_examples/fgsm/epsilon_0.10"

CLEAN_PATH = os.path.join(
    EXAMPLE_DIR,
    "clean.pt"
)

ADVERSARIAL_PATH = os.path.join(
    EXAMPLE_DIR,
    "adversarial.pt"
)

OUTPUT_PATH = os.path.join(
    EXAMPLE_DIR,
    "fgsm_comparison.png"
)


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------
    # Load model
    # -----------------------------------------

    model = MNISTCNN().to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.eval()

    # -----------------------------------------
    # Load saved examples
    # -----------------------------------------

    clean_image = torch.load(
        CLEAN_PATH,
        map_location=device
    )

    adversarial_image = torch.load(
        ADVERSARIAL_PATH,
        map_location=device
    )

    # -----------------------------------------
    # Get predictions
    # -----------------------------------------

    with torch.no_grad():

        clean_output = model(clean_image)

        adversarial_output = model(
            adversarial_image
        )

        clean_probabilities = torch.softmax(
            clean_output,
            dim=1
        )

        adversarial_probabilities = torch.softmax(
            adversarial_output,
            dim=1
        )

        clean_prediction = (
            clean_output.argmax(dim=1).item()
        )

        adversarial_prediction = (
            adversarial_output.argmax(dim=1).item()
        )

        clean_confidence = (
            clean_probabilities[
                0,
                clean_prediction
            ].item()
        )

        adversarial_confidence = (
            adversarial_probabilities[
                0,
                adversarial_prediction
            ].item()
        )

    # -----------------------------------------
    # Determine true label
    # -----------------------------------------

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,),
            (0.3081,)
        )
    ])

    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    # Find the original image index by comparing
    # against the saved clean tensor.
    true_label = None

    for index in range(len(test_dataset)):

        image, label = test_dataset[index]

        if torch.allclose(
            image.unsqueeze(0),
            clean_image.cpu(),
            atol=1e-6
        ):
            true_label = label
            break

    if true_label is None:
        raise RuntimeError(
            "Could not identify the original MNIST label."
        )

    # -----------------------------------------
    # Calculate perturbation
    # -----------------------------------------

    perturbation = (
        adversarial_image - clean_image
    )

    perturbation_min = perturbation.min().item()
    perturbation_max = perturbation.max().item()

    # -----------------------------------------
    # Print results
    # -----------------------------------------

    print("\nFGSM ADVERSARIAL EXAMPLE")
    print("=" * 50)

    print(f"True label:              {true_label}")
    print(f"Clean prediction:        {clean_prediction}")
    print(
        f"Clean confidence:        "
        f"{clean_confidence * 100:.4f}%"
    )

    print(
        f"Adversarial prediction:  "
        f"{adversarial_prediction}"
    )

    print(
        f"Adversarial confidence:  "
        f"{adversarial_confidence * 100:.4f}%"
    )

    print(
        f"Attack successful:       "
        f"{adversarial_prediction != true_label}"
    )

    print(
        f"Perturbation range:      "
        f"[{perturbation_min:.4f}, "
        f"{perturbation_max:.4f}]"
    )

    # -----------------------------------------
    # Prepare images
    # -----------------------------------------

    clean_display = clean_image.cpu().squeeze()
    adversarial_display = (
        adversarial_image.cpu().squeeze()
    )
    perturbation_display = (
        perturbation.cpu().squeeze()
    )

    # -----------------------------------------
    # Create comparison figure
    # -----------------------------------------

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4)
    )

    axes[0].imshow(
        clean_display,
        cmap="gray"
    )

    axes[0].set_title(
        f"Clean\n"
        f"True: {true_label} | "
        f"Pred: {clean_prediction}\n"
        f"Confidence: "
        f"{clean_confidence * 100:.2f}%"
    )

    axes[0].axis("off")

    # Amplify perturbation visually
    axes[1].imshow(
        perturbation_display,
        cmap="gray"
    )

    axes[1].set_title(
        "FGSM Perturbation"
    )

    axes[1].axis("off")

    axes[2].imshow(
        adversarial_display,
        cmap="gray"
    )

    axes[2].set_title(
        f"Adversarial\n"
        f"True: {true_label} | "
        f"Pred: {adversarial_prediction}\n"
        f"Confidence: "
        f"{adversarial_confidence * 100:.2f}%"
    )

    axes[2].axis("off")

    figure.suptitle(
        "FGSM Adversarial Attack (ε = 0.10)"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"\nComparison saved to:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()