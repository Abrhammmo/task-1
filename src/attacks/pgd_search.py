import sys
import os

import torch
import torch.nn as nn
from torchvision import datasets, transforms

sys.path.append("./src/model")
from model import MNISTCNN


MODEL_PATH = "./models/mnist_cnn.pth"

NUM_IMAGES = 1000

EPSILONS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
]

ALPHA = 0.01
STEPS = 40

OUTPUT_DIR = "adversarial_examples/pgd"


def pgd_attack(model, image, label, epsilon, alpha, steps):

    original = image.detach().clone()

    adversarial = original.clone().detach()

    for _ in range(steps):

        adversarial.requires_grad = True

        output = model(adversarial)

        loss = nn.CrossEntropyLoss()(output, label)

        model.zero_grad()

        loss.backward()

        gradient = adversarial.grad.detach()

        # Gradient ascent: increase classification loss
        adversarial = (
            adversarial
            + alpha * gradient.sign()
        )

        # Keep perturbation within epsilon
        perturbation = torch.clamp(
            adversarial - original,
            min=-epsilon,
            max=epsilon
        )

        adversarial = (
            original + perturbation
        )

        adversarial = adversarial.detach()

    return adversarial


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")
    print()
    print(
        f"Testing first {NUM_IMAGES} "
        "MNIST test images..."
    )
    print()

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

    model = MNISTCNN().to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.eval()

    # ------------------------------------------------
    # First determine which images are initially
    # classified correctly.
    # ------------------------------------------------

    correctly_classified = []

    print("Finding correctly classified images...")

    for index in range(NUM_IMAGES):

        image, label = test_dataset[index]

        image = image.unsqueeze(0).to(device)

        label_tensor = torch.tensor(
            [label],
            device=device
        )

        with torch.no_grad():

            output = model(image)

            prediction = output.argmax(
                dim=1
            ).item()

        if prediction == label:

            correctly_classified.append(
                (index, image, label_tensor)
            )

    initial_correct = len(
        correctly_classified
    )

    print(
        f"Initially correct: "
        f"{initial_correct}"
    )
    print()

    # ------------------------------------------------
    # PGD experiment
    # ------------------------------------------------

    results = []

    print("PGD ROBUSTNESS EXPERIMENT")
    print("=" * 60)

    for epsilon in EPSILONS:

        successful = 0

        saved_example = False

        print()
        print(
            f"Epsilon: {epsilon:.2f}"
        )

        for index, image, label in correctly_classified:

            adversarial = pgd_attack(
                model=model,
                image=image,
                label=label,
                epsilon=epsilon,
                alpha=ALPHA,
                steps=STEPS
            )

            with torch.no_grad():

                output = model(
                    adversarial
                )

                prediction = output.argmax(
                    dim=1
                ).item()

            # Attack is successful when
            # the adversarial prediction
            # differs from the true label.

            if prediction != label.item():

                successful += 1

                # Save the first successful
                # adversarial example for this epsilon.

                if not saved_example:

                    epsilon_dir = os.path.join(
                        OUTPUT_DIR,
                        f"epsilon_{epsilon:.2f}"
                    )

                    os.makedirs(
                        epsilon_dir,
                        exist_ok=True
                    )

                    clean_path = os.path.join(
                        epsilon_dir,
                        "clean.pt"
                    )

                    adversarial_path = os.path.join(
                        epsilon_dir,
                        "adversarial.pt"
                    )

                    torch.save(
                        image.cpu(),
                        clean_path
                    )

                    torch.save(
                        adversarial.cpu(),
                        adversarial_path
                    )

                    # Get clean prediction and confidence
                    with torch.no_grad():
                        clean_output = model(image)
                        clean_probabilities = torch.softmax(
                            clean_output,
                            dim=1
                        )

                        clean_prediction = clean_probabilities.argmax(
                            dim=1
                        ).item()

                        clean_confidence = clean_probabilities[
                            0,
                            clean_prediction
                        ].item()

                        # Get adversarial prediction and confidence
                        adversarial_output = model(adversarial)

                        adversarial_probabilities = torch.softmax(
                            adversarial_output,
                            dim=1
                        )

                        adversarial_prediction = adversarial_probabilities.argmax(
                            dim=1
                        ).item()

                        adversarial_confidence = adversarial_probabilities[
                            0,
                            adversarial_prediction
                        ].item()

                    with open(
                        os.path.join(
                            epsilon_dir,
                            "metadata.txt"
                        ),
                        "w"
                    ) as f:

                        f.write(
                            f"Dataset index: {index}\n"
                        )

                        f.write(
                            f"True label: {label.item()}\n"
                        )

                        f.write(
                            f"Clean prediction: "
                            f"{clean_prediction}\n"
                        )

                        f.write(
                            f"Clean confidence: "
                            f"{clean_confidence:.6f}\n"
                        )

                        f.write(
                            f"Adversarial prediction: "
                            f"{adversarial_prediction}\n"
                        )

                        f.write(
                            f"Adversarial confidence: "
                            f"{adversarial_confidence:.6f}\n"
                        )

                        f.write(
                            f"Epsilon: {epsilon:.2f}\n"
                        )

                        f.write(
                            f"Alpha: {ALPHA}\n"
                        )

                        f.write(
                            f"Steps: {STEPS}\n"
                        )

                        f.write(
                            f"Attack successful: "
                            f"{adversarial_prediction != label.item()}\n"
                        )

                    print()
                    print("SUCCESSFUL PGD EXAMPLE FOUND")
                    print("-" * 50)
                    print(f"Dataset index:          {index}")
                    print(f"True label:             {label.item()}")
                    print(f"Clean prediction:       {clean_prediction}")
                    print(f"Clean confidence:       {clean_confidence:.4%}")
                    print(f"Adversarial prediction: {adversarial_prediction}")
                    print(f"Adversarial confidence: {adversarial_confidence:.4%}")
                    print(
                        f"Attack successful:      "
                        f"{adversarial_prediction != label.item()}"
                    )
                    print(
                        f"Saved to:               {epsilon_dir}"
                    )
                    print()

                    saved_example = True

        success_rate = (
            successful
            / initial_correct
            * 100
        )

        print(
            f"Successful attacks: "
            f"{successful}"
        )

        print(
            f"Attack success rate: "
            f"{success_rate:.2f}%"
        )

        results.append(
            (
                epsilon,
                successful,
                success_rate
            )
        )

    # ------------------------------------------------
    # Final results
    # ------------------------------------------------

    print()
    print("FINAL PGD RESULTS")
    print("=" * 60)

    print(
        "Epsilon | Successful | Success Rate"
    )

    print(
        "--------|-------------|-------------"
    )

    for epsilon, successful, success_rate in results:

        print(
            f"  {epsilon:>4.2f}  |"
            f" {successful:>10}  |"
            f" {success_rate:>10.2f}%"
        )


if __name__ == "__main__":
    main()