import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms

sys.path.append("./src/model")
from model import MNISTCNN


MODEL_PATH = "./models/mnist_cnn.pth"

EPSILON = 0.3
ALPHA = 0.01
STEPS = 40


def pgd_attack(model, image, label, epsilon, alpha, steps):
    """
    Projected Gradient Descent attack.

    image: normalized input tensor
    label: true class
    epsilon: maximum perturbation
    alpha: step size
    steps: number of attack iterations
    """

    original = image.detach().clone()

    adversarial = original.clone().detach()

    for _ in range(steps):

        adversarial.requires_grad = True

        output = model(adversarial)

        loss = nn.CrossEntropyLoss()(output, label)

        model.zero_grad()

        loss.backward()

        gradient = adversarial.grad.detach()

        # Move in the direction that maximizes the loss
        adversarial = adversarial + alpha * gradient.sign()

        # Project back into the epsilon-ball
        perturbation = torch.clamp(
            adversarial - original,
            min=-epsilon,
            max=epsilon
        )

        adversarial = original + perturbation

        adversarial = adversarial.detach()

    return adversarial


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

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

    image, label = test_dataset[0]

    image = image.unsqueeze(0).to(device)
    label = torch.tensor(
        [label],
        device=device
    )

    # Clean prediction
    with torch.no_grad():
        clean_output = model(image)
        clean_prediction = clean_output.argmax(
            dim=1
        ).item()

    print(f"True label:       {label.item()}")
    print(f"Clean prediction: {clean_prediction}")

    if clean_prediction != label.item():
        print("Image is already misclassified. Exiting.")
        return

    # Generate adversarial example
    adversarial = pgd_attack(
        model,
        image,
        label,
        EPSILON,
        ALPHA,
        STEPS
    )

    # Adversarial prediction
    with torch.no_grad():
        adversarial_output = model(adversarial)

        adversarial_prediction = (
            adversarial_output
            .argmax(dim=1)
            .item()
        )

        probabilities = torch.softmax(
            adversarial_output,
            dim=1
        )

        confidence = probabilities[
            0,
            adversarial_prediction
        ].item()

    perturbation = adversarial - image

    print()
    print("PGD RESULT")
    print("=" * 50)
    print(f"True label:              {label.item()}")
    print(f"Clean prediction:        {clean_prediction}")
    print(f"Adversarial prediction:  {adversarial_prediction}")
    print(f"Confidence:              {confidence:.4f}")
    print(
        f"Perturbation range:      "
        f"[{perturbation.min().item():.4f}, "
        f"{perturbation.max().item():.4f}]"
    )

    print(
        f"Attack successful:       "
        f"{adversarial_prediction != label.item()}"
    )


if __name__ == "__main__":
    main()