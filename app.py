"""Gradio demo: upload a pet photo -> top-5 breed predictions.
python app.py --ckpt runs/resnet18_finetune_basic/best.pt [--share]
"""
import argparse

import gradio as gr
import torch

from src.data import get_transforms
from src.utils import get_device, load_model

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="runs/resnet18_finetune_basic/best.pt")
p.add_argument("--share", action="store_true")
args = p.parse_args()

# გამოთვლითი მოწყობილობის (GPU/CPU) განსაზღვრა და გაწვრთნილი მოდელის ჩატვირთვა
device = get_device()
model, classes, _ = load_model(args.ckpt, device)
_, tf = get_transforms("none")


@torch.no_grad()
def classify(img):
    """ღებულობს ატვირთულ სურათს, ამზადებს მოდელისთვის და აბრუნებს ტოპ-5 ჯიშს ალბათობებით."""
    x = tf(img.convert("RGB")).unsqueeze(0).to(device)
    probs = model(x).softmax(1)[0]
    return {classes[i]: float(probs[i]) for i in probs.topk(5).indices}


# Gradio ვებ-ინტერფეისის კონფიგურაცია და გაშვება
gr.Interface(fn=classify, inputs=gr.Image(type="pil"),
             outputs=gr.Label(num_top_classes=5),
             title="Pet breed classifier (37 breeds)",
             description="Fine-tuned CNN on Oxford-IIIT Pet.").launch(share=args.share)
