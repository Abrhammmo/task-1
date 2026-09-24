import os
import torch
from PIL import Image

MEAN = 0.1307
STD = 0.3081

PGD_DIR = "adversarial_examples/pgd"

EPSILONS = [
    "0.05",
    "0.10",
    "0.15",
    "0.20",
    "0.25",
    "0.30"
]


def save_tensor_as_png(tensor, output_path):
    # Remove batch/channel dimensions
    image = tensor.squeeze().detach().cpu()

    # Undo MNIST normalization
    image = image * STD + MEAN

    # Keep pixel values between 0 and 1
    image = image.clamp(0, 1)

    # Convert to 8-bit grayscale
    image = (image * 255).byte()

    # Convert to PNG
    image = Image.fromarray(
        image.numpy(),
        mode="L"
    )

    image.save(output_path)

    print(f"Saved: {output_path}")


def main():

    for epsilon in EPSILONS:

        epsilon_dir = os.path.join(
            PGD_DIR,
            f"epsilon_{epsilon}"
        )

        clean_path = os.path.join(
            epsilon_dir,
            "clean.pt"
        )

        adversarial_path = os.path.join(
            epsilon_dir,
            "adversarial.pt"
        )

        # Check that the attack results exist
        if not os.path.exists(clean_path):
            print(
                f"Skipping epsilon {epsilon}: "
                f"clean.pt not found"
            )
            continue

        if not os.path.exists(adversarial_path):
            print(
                f"Skipping epsilon {epsilon}: "
                f"adversarial.pt not found"
            )
            continue

        # Load tensors
        clean = torch.load(
            clean_path,
            map_location="cpu"
        )

        adversarial = torch.load(
            adversarial_path,
            map_location="cpu"
        )

        # Output paths
        clean_png = os.path.join(
            epsilon_dir,
            "clean.png"
        )

        adversarial_png = os.path.join(
            epsilon_dir,
            "adversarial.png"
        )

        # Convert to PNG
        save_tensor_as_png(
            clean,
            clean_png
        )

        save_tensor_as_png(
            adversarial,
            adversarial_png
        )

    print("\nFinished converting all PGD examples.")


if __name__ == "__main__":
    main()