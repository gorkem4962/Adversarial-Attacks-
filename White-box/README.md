# White-Box Adversarial Attacks

MNIST CNN (`WhiteBoxModel`) plus white-box attacks for the AML course.

## Layout

```
White-box/
  Code/
    white_box_model.py   # CNN + train / load / logits API
    data_utils.py        # MNIST helpers
    visualize.py         # clean / adv / diff plots
    attacks/
      lbfgs.py           # Szegedy L-BFGS-B
      fgsm.py            # Goodfellow FGSM (targeted / untargeted)
      # cw.py, ...       (add later)
    run_lbfgs.py         # CLI for L-BFGS
    run_fgsm.py          # CLI for FGSM
  checkpoints/           # trained weights
  outputs/               # saved figures
  requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train model (once)

```bash
python Code/white_box_model.py
```

## Run L-BFGS

```bash
python Code/run_lbfgs.py
python Code/run_lbfgs.py --index 10 --target 7 --no-show
```

## Run FGSM

```bash
# untargeted (default): leave correct class
python Code/run_fgsm.py
python Code/run_fgsm.py --epsilon 0.2 --no-show

# targeted: force a specific class
python Code/run_fgsm.py --targeted --target 1 --no-show
```

Inputs are `(N, 1, 28, 28)` in `[0, 1]`. Use `model.logits(x)` for input gradients.

## Add another attack

1. Create `Code/attacks/<name>.py` with a function returning `(x_adv, meta)`.
2. Export it from `Code/attacks/__init__.py`.
3. Add `Code/run_<name>.py` (same pattern as `run_lbfgs.py`).
