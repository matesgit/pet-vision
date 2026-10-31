"""Collect all runs into one markdown table + learning-curve plot."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows, fig_data = [], {}
# ვეძებთ და ვკითხულობთ ყველა ექსპერიმენტის მეტრიკების ფაილს runs/ საქაღალდიდან
for d in sorted(Path("runs").glob("*/test_metrics.json")):
    m = json.loads(d.read_text()); c = m["config"]
    # ვამზადებთ ცხრილის თითოეულ რიგს მიღებული მეტრიკებისა და კონფიგურაციის მიხედვით
    rows.append((m["test_acc"], f"| {m['name']} | {c['model']} | {c['pretrained']} | {c['freeze_backbone']} | "
                 f"{c['augment']} | {c['epochs']} | {m['best_val_acc']:.3f} | {m['test_acc']:.3f} | "
                 f"{m['test_macro_f1']:.3f} | {m['params_m']:.1f} | {m['size_mb']:.1f} | {m['latency_ms_bs1']:.1f} |"))
    # ვამოწმებთ და ვკითხულობთ ტრენინგის ისტორიის (history.json) ფაილს გრაფიკისთვის
    h = d.parent / "history.json"
    if h.exists():
        fig_data[m["name"]] = json.loads(h.read_text())

# ვადგენთ ცხრილის თავსართს (Header)
head = ("| run | model | pretrained | frozen | augment | epochs | val acc | test acc | macro F1 | "
        "params (M) | size (MB) | latency (ms) |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
# ვაწყობთ სრულ ცხრილს ტესტის სიზუსტის (test_acc) კლებადობის მიხედვით და ვინახავთ ფაილში
table = head + "\n" + "\n".join(r for _, r in sorted(rows, reverse=True))
Path("runs/comparison.md").write_text(table)
print(table)

# თუ ისტორიის მონაცემები არსებობს, ვხაზავთ სწავლის მრუდების გრაფიკს
if fig_data:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, h in fig_data.items():
        # ვხაზავთ ვალიდაციის სიზუსტეს და დანაკარგს (loss) ეპოქების მიხედვით
        a1.plot([e["val_acc"] for e in h], label=name)
        a2.plot([e["val_loss"] for e in h], label=name)
    a1.set_title("Validation accuracy"); a2.set_title("Validation loss")
    for a in (a1, a2): a.set_xlabel("epoch")
    a1.legend(fontsize=7); fig.tight_layout(); fig.savefig("runs/curves.png", dpi=130)
