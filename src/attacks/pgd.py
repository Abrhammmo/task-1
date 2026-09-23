import torch


def pgd_attack(model, inputs, labels, epsilon=0.03, alpha=0.01, steps=10):
    adv_inputs = inputs.clone().detach()
    adv_inputs.requires_grad_(True)

    for _ in range(steps):
        adv_inputs.requires_grad_(True)
        outputs = model(adv_inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()

        grad = adv_inputs.grad
        adv_inputs = adv_inputs + alpha * grad.sign()
        adv_inputs = torch.min(torch.max(adv_inputs, inputs - epsilon), inputs + epsilon)
        adv_inputs = adv_inputs.detach()

    return adv_inputs
