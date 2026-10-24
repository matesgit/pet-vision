#!/usr/bin/env bash
# Runs all experiments sequentially (about 20-30 min on a free Colab T4 GPU).
set -e
run() { python -m src.train "$@"; python -m src.evaluate --name "$2"; }

# 1) From scratch vs transfer learning
run --name resnet18_linear    --model resnet18 --freeze-backbone --augment basic --epochs 10 --lr 1e-3
# 2) Effect of augmentation (full fine-tuning)
run --name resnet18_ft_noaug  --model resnet18 --augment none   --epochs 10 --lr 3e-4
run --name resnet18_ft_basic  --model resnet18 --augment basic  --epochs 10 --lr 3e-4
run --name resnet18_ft_strong --model resnet18 --augment strong --epochs 10 --lr 3e-4
# 3) Smaller / faster architecture
run --name mobilenet_ft_basic --model mobilenet_v3_small --augment basic --epochs 10 --lr 1e-3

python -m src.compare_runs
python -m src.export_onnx --name resnet18_ft_basic
python -m src.export_onnx --name mobilenet_ft_basic
