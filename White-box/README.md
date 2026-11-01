# White-Box Adversarial Attacks

MNIST CNN (`WhiteBoxModel`) for white-box experiments (FGSM, L-BFGS, C&W, …).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train / load model

```bash
python Code/white_box_model.py          # trains & saves checkpoints/mnist_cnn.pt
python Code/attack_starter.py           # implement attacks here
```

Inputs are `(N, 1, 28, 28)` tensors in `[0, 1]`. Use `model.logits(x)` for differentiable white-box gradients.
