import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Allow Python to find src/model/model.py
sys.path.append("./src/model")

from model import MNISTCNN


MODEL_PATH = "./models/mnist_cnn.pth"
OUTPUT_DIR = "./adversarial_examples/fgsm"

BATCH_SIZE = 64

# We will test several perturbation strengths.
EPSILONS = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
]

# Number of test images to evaluate.
MAX_IMAGES = 1000


def fgsm_attack(image, epsilon, gradient):
    """
    Generate an FGSM adversarial example.
    """
    gradient_sign = gradient.sign()

    adversarial_image = image + epsilon * gradient_sign

    return adversarial_image


def evaluate_epsilon(
    model,
    dataset,
    epsilon,
    device,
):
    """
    Evaluate FGSM attack success for one epsilon value.
    """

    loss_function = nn.CrossEntropyLoss()

    total = 0
    initially_correct = 0
    successful_attacks = 0

    successful_example = None

    for index in range(min(MAX_IMAGES, len(dataset))):

        image, true_label = dataset[index]

        image = image.unsqueeze(0).to(device)
        image.requires_grad = True

        label = torch.tensor(
            [true_label],
            device=device
        )

        # -----------------------------
        # Clean prediction
        # -----------------------------

        output = model(image)

        clean_prediction = output.argmax(
            dim=1
        ).item()

        total += 1

        # We only consider images that
        # the model originally classified correctly.
        if clean_prediction != true_label:
            continue

        initially_correct += 1

        # -----------------------------
        # Calculate gradient
        # -----------------------------

        loss = loss_function(output, label)

        model.zero_grad()

        if image.grad is not None:
            image.grad.zero_()

        loss.backward()

        gradient = image.grad.data

        # -----------------------------
        # Generate adversarial example
        # -----------------------------

        adversarial_image = fgsm_attack(
            image,
            epsilon,
            gradient
        )

        # -----------------------------
        # Adversarial prediction
        # -----------------------------

        adversarial_output = model(
            adversarial_image.detach()
        )

        adversarial_prediction = (
            adversarial_output.argmax(
                dim=1
            ).item()
        )

        # -----------------------------
        # Check attack success
        # -----------------------------

        if adversarial_prediction != true_label:

            successful_attacks += 1

            # Save the first successful example
            if successful_example is None:
                successful_example = {
                    "index": index,
                    "true_label": true_label,
                    "clean_prediction": clean_prediction,
                    "adversarial_prediction": (
                        adversarial_prediction
                    ),
                    "clean_image": image.detach().cpu(),
                    "adversarial_image": (
                        adversarial_image.detach().cpu()
                    ),
                }

    # Avoid division by zero
    if initially_correct > 0:
        attack_success_rate = (
            successful_attacks
            / initially_correct
            * 100
        )
    else:
        attack_success_rate = 0.0

    return {
        "epsilon": epsilon,
        "total": total,
        "initially_correct": initially_correct,
        "successful_attacks": successful_attacks,
        "attack_success_rate": attack_success_rate,
        "successful_example": successful_example,
    }


def save_successful_example(example, epsilon):
    """
    Save the first successful adversarial example
    for an epsilon value.
    """

    if example is None:
        return

    output_dir = os.path.join(
        OUTPUT_DIR,
        f"epsilon_{epsilon:.2f}"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    clean_path = os.path.join(
        output_dir,
        "clean.pt"
    )

    adversarial_path = os.path.join(
        output_dir,
        "adversarial.pt"
    )

    torch.save(
        example["clean_image"],
        clean_path
    )

    torch.save(
        example["adversarial_image"],
        adversarial_path
    )


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # -----------------------------
    # Load dataset
    # -----------------------------

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

    # -----------------------------
    # Load model
    # -----------------------------

    model = MNISTCNN().to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.eval()

    print(
        f"Testing first "
        f"{MAX_IMAGES} test images..."
    )

    print("\nFGSM ROBUSTNESS EXPERIMENT")
    print("=" * 60)

    results = []

    for epsilon in EPSILONS:

        result = evaluate_epsilon(
            model,
            test_dataset,
            epsilon,
            device
        )

        results.append(result)

        print(
            f"\nEpsilon: "
            f"{result['epsilon']:.2f}"
        )

        print(
            f"Initially correct: "
            f"{result['initially_correct']}"
        )

        print(
            f"Successful attacks: "
            f"{result['successful_attacks']}"
        )

        print(
            f"Attack success rate: "
            f"{result['attack_success_rate']:.2f}%"
        )

        # Save one successful example
        save_successful_example(
            result["successful_example"],
            epsilon
        )

    # -----------------------------
    # Final summary
    # -----------------------------

    print("\n")
    print("=" * 60)
    print("FINAL FGSM RESULTS")
    print("=" * 60)

    print(
        "\nEpsilon | Successful | Success Rate"
    )
    print(
        "--------|-------------|-------------"
    )

    for result in results:

        print(
            f"{result['epsilon']:>7.2f} | "
            f"{result['successful_attacks']:>11} | "
            f"{result['attack_success_rate']:>10.2f}%"
        )


if __name__ == "__main__":
    main()