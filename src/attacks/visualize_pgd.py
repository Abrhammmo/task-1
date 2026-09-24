import os
import sys

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

sys.path.append("./src/model")
from model import MNISTCNN


MODEL_PATH = "./models/mnist_cnn.pth"

EPSILON = 0.10

EXAMPLE_DIR = (
    f"adversarial_examples/pgd/"
    f"epsilon_{EPSILON:.2f}"
)


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    clean_path = os.path.join(
        EXAMPLE_DIR,
        "clean.pt"
    )

    adversarial_path = os.path.join(
        EXAMPLE_DIR,
        "adversarial.pt"
    )

    metadata_path = os.path.join(
        EXAMPLE_DIR,
        "metadata.txt"
    )

    if not os.path.exists(
        adversarial_path
    ):

        print(
            "No successful PGD example "
            f"found for epsilon={EPSILON:.2f}"
        )

        return

    clean = torch.load(
        clean_path,
        map_location=device
    )

    adversarial = torch.load(
        adversarial_path,
        map_location=device
    )

    # Read metadata
    with open(metadata_path) as f:

        metadata = f.read()

    print(metadata)

    model = MNISTCNN().to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.eval()

    with torch.no_grad():

        clean_output = model(clean)

        adversarial_output = model(
            adversarial
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
            clean_output
            .argmax(dim=1)
            .item()
        )

        adversarial_prediction = (
            adversarial_output
            .argmax(dim=1)
            .item()
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

    perturbation = (
        adversarial - clean
    )

    print()
    print("PGD ADVERSARIAL EXAMPLE")
    print("=" * 50)

    print(
        f"Clean prediction:        "
        f"{clean_prediction}"
    )

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
        f"{clean_prediction != adversarial_prediction}"
    )

    print(
        f"Perturbation range:      "
        f"[{perturbation.min().item():.4f}, "
        f"{perturbation.max().item():.4f}]"
    )

    # Convert normalized image back into
    # displayable MNIST pixel values.

    clean_display = (
        clean.squeeze().cpu()
        * 0.3081
        + 0.1307
    )

    adversarial_display = (
        adversarial.squeeze().cpu()
        * 0.3081
        + 0.1307
    )

    perturbation_display = (
        perturbation.squeeze().cpu()
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(9, 3)
    )

    axes[0].imshow(
        clean_display,
        cmap="gray"
    )

    axes[0].set_title(
        f"Clean\nPrediction: {clean_prediction}"
    )

    axes[1].imshow(
        adversarial_display,
        cmap="gray"
    )

    axes[1].set_title(
        f"PGD\nPrediction: {adversarial_prediction}"
    )

    axes[2].imshow(
        perturbation_display,
        cmap="gray"
    )

    axes[2].set_title(
        "Perturbation"
    )

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()

    output_path = os.path.join(
        EXAMPLE_DIR,
        "pgd_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()

    print()
    print(
        f"Visualization saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()