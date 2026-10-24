# Pet Breed Classification - End-to-End Computer Vision Project

Classify 37 cat/dog breeds from the Oxford-IIIT Pet dataset (~7.4k images) using
transfer learning (PyTorch / torchvision), with a systematic set of experiments,
error analysis, ONNX export and a Gradio demo.

## Pipeline
data (`src/data.py`) -> model (`src/model.py`) -> training (`src/train.py`) ->
test evaluation (`src/evaluate.py`) -> comparison (`src/compare_runs.py`) ->
optimization (`src/export_onnx.py`) -> demo (`app.py`)

## Run on Google Colab (free GPU: Runtime > Change runtime type > T4)
```bash
!git clone <YOUR_REPO_URL> && cd pet-vision
!pip install -r requirements.txt
!bash run_experiments.sh            # trains + evaluates all experiments
!python app.py --ckpt runs/resnet18_ft_basic/best.pt --share
```
Single run: `python -m src.train --name my_run --model resnet18 --augment basic --epochs 10 --lr 3e-4`

## Design decisions
- Official `trainval` split -> 85% train / 15% validation (fixed seed); the official `test` split
  is used once, for final numbers. Best epoch is chosen on validation, never on test.
- ImageNet normalization because backbones are ImageNet-pretrained.
- Metrics: accuracy, top-5 accuracy, macro-F1, confusion matrix, worst confident errors.
- Optimization: mixed precision (AMP) in training, lighter backbone (MobileNetV3), ONNX Runtime inference.

## Results
See `runs/comparison.md`, `runs/curves.png` and `REPORT_ka.md` (fill with your own numbers).

## Limitations
Single seed per experiment; no hyperparameter search; classes are not stratified in the val split.
