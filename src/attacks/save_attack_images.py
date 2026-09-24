import os
import torch
from PIL import Image
from torchvision import datasets, transforms


NORMALIZE_MEAN = 0.1307
NORMALIZE_STD = 0.3081

BASE_DIR = "adversarial_examples"

FGSM_DIR = os.path.join(
    BASE_DIR,
    "fgsm",
    "epsilon_0.10"
)

PGD_DIR = os.path.join(
    BASE_DIR,
    "pgd",
    "epsilon_0.10"
)


def tensor_to_png(tensor, output_path):
    """
    Convert a normalized MNIST tensor back into
    a normal 0-255 grayscale PNG.
    """

    image = tensor.squeeze().detach().cpu()

    # Undo normalization
    image = image * NORMALIZE_STD + NORMALIZE_MEAN

    # Convert to valid image range
    image = image.clamp(0, 1)

    # Convert to uint8
    image = (image * 255).byte()

    pil_image = Image.fromarray(
        image.numpy(),
        mode="L"
    )

    pil_image.save(output_path)

    print(f"Saved: {output_path}")


def process_directory(directory):
    clean_path = os.path.join(
        directory,
        "clean.pt"
    )

    adversarial_path = os.path.join(
        directory,
        "adversarial.pt"
    )

    clean = torch.load(
        clean_path,
        map_location="cpu"
    )

    adversarial = torch.load(
        adversarial_path,
        map_location="cpu"
    )

    tensor_to_png(
        clean,
        os.path.join(directory, "clean.png")
    )

    tensor_to_png(
        adversarial,
        os.path.join(directory, "adversarial.png")
    )


if __name__ == "__main__":

    print("\nConverting FGSM example...")
    process_directory(FGSM_DIR)

    print("\nConverting PGD example...")
    process_directory(PGD_DIR)

    print("\nDone.")