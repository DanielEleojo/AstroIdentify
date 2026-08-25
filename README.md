# AstroIdentify — Constellation Classifier

A deep learning web app that identifies which of 12 zodiac constellations appears in a night-sky image. A ResNet18 model (transfer learning, PyTorch) reaches **86% test accuracy**, served through a Flask web app and JSON API with secure uploads and structured error handling.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet18-orange.svg)
![Flask](https://img.shields.io/badge/Flask-API%20%2B%20Web%20UI-green.svg)

## The problem

Star-field photos are hard to classify: constellations are sparse point patterns, not textured objects, and the same constellation varies with framing, rotation, and exposure. The goal was to train a classifier that handles that variance and then ship it as something a person can actually use — not leave it in a notebook.

## How it works

**Model** ([`astromodel.py`](astromodel.py))
- ResNet18 pretrained on ImageNet, final layer replaced for 12 constellation classes
- Fine-tuned end-to-end with Adam (lr=1e-4), 15 epochs
- Augmentation tuned for star fields: horizontal flips, small rotations (±5°), light brightness/contrast jitter — enough to generalize without destroying the point patterns that identify a constellation
- Best checkpoint selected on validation accuracy → **86% on the held-out test set** across 12 classes (Aquarius, Aries, Cancer, Capricornus, Gemini, Leo, Libra, Pisces, Sagittarius, Scorpius, Taurus, Virgo)

**Serving** ([`app.py`](app.py), [`constellation_classifier.py`](constellation_classifier.py))
- Flask web UI: upload an image, get the prediction with confidence scores and top-5 alternatives
- JSON API: `POST /api/predict` (image in, structured prediction out) and `GET /api/model-info`
- Secure file handling: extension allow-list, size limits, and explicit `413` / `404` / `500` error responses

**Tests** ([`test_astroidentify.py`](test_astroidentify.py))
- Endpoint and model-loading checks, plus a tiny-overfit sanity script ([`tiny_overfit_check.py`](tiny_overfit_check.py)) used during training to verify the pipeline could learn at all before full runs

## Run it

```bash
pip install -r requirements.txt
python app.py
# open http://localhost:5000
```

Requires the trained weights (`constellation_best_model.pt`) and `constellation_classes.json`, both included in the repo.

## What I'd do next

- Real-time detection from device camera frames
- Expand beyond the 12 zodiac constellations
- Confidence calibration — the model is overconfident on out-of-distribution images (e.g., non-sky photos)

---

Built by [Daniel Baba](https://linkedin.com/in/baba-daniel) — B.Sc. Computer Science (Math minor), Ontario Tech University.
