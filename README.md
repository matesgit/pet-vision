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
!git clone https://github.com/matesgit/pet-vision.git
%cd pet-vision
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

## Results (Oxford-IIIT Pet test set, 3,669 images)

| Model | Setup | Test acc | Macro-F1 | Size |
|---|---|---|---|---|
| SimpleCNN | from scratch | 35.3% | 0.342 | 4.5 MB |
| ResNet18 | frozen backbone | 88.9% | 0.888 | 42.7 MB |
| ResNet18 | fine-tune, no augmentation | 88.4% | 0.880 | 42.7 MB |
| ResNet18 | fine-tune, basic augmentation | 88.4% | 0.883 | 42.7 MB |
| ResNet18 | fine-tune, strong augmentation | 88.9% | 0.887 | 42.7 MB |
| MobileNetV3-Small | fine-tune, basic augmentation | 85.0% | 0.848 | 6.0 MB |

Final model: `resnet18_ft_basic` (chosen by validation accuracy, 92.6%).

ONNX Runtime on CPU, per image: ResNet18 59.4 ms -> 33.8 ms, MobileNetV3 9.4 ms -> 2.3 ms.

Most confused breeds: American Pit Bull Terrier / Staffordshire Bull Terrier, Ragdoll / Birman.

Full comparison table: `runs/comparison.md`. Learning curves and confusion matrices: `runs/`.

## Trained models
Model weights are not stored in the repository because of their size. To reproduce them,
run the Colab commands above: `run_experiments.sh` retrains all models in about 30-40 minutes
and saves them to `runs/<experiment>/best.pt`.

## Technical report
The technical report (in Georgian) is included in this repository as a separate file.

## Limitations
Single seed per experiment; no hyperparameter search; classes are not stratified in the val split.
The model knows only the 37 breeds of the dataset and has no "unknown" class.





