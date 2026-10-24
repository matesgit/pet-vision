# გასაუბრების გზამკვლევი: ყველაფერი, რაც უნდა გესმოდეს

## 0. პირველ რიგში (ძალიან მნიშვნელოვანია)
დავალება პირდაპირ ამბობს: AI ნებადართულია, **მაგრამ გასაუბრებაზე კოდს უნდა ხსნიდე და ცვლილებას თავად შეიტან**.
ამიტომ: 1) პროექტი თავად გაუშვი Colab-ში და საკუთარი ციფრები მიიღე; 2) ყველა ფაილი ერთხელ ხელით წაიკითხე;
3) შეცვალე რამე (სხვა lr, სხვა მოდელი, ახალი augmentation) და შეხედე რა მოხდა — ეს საუკეთესო მომზადებაა;
4) ანგარიშში AI-ის გამოყენებაზე გულწრფელად დაწერე.

## 1. დიდი სურათი (30 წამში ასე ამოუხსენი)
„ავიღე Oxford-IIIT Pet (37 ჯიში). ვიცი, რომ ცოტა მონაცემზე from-scratch ტრენინგი სუსტია, ამიტომ გამოვიყენე transfer
learning: ImageNet-ზე გაწვრთნილი ResNet18, ბოლო ფენა შევცვალე 37 კლასზე და მთლიანი მოდელი დავატრენინგე მცირე
learning rate-ით. გავუშვი 6 კონტროლირებადი ექსპერიმენტი: scratch vs pretrained, frozen vs fine-tune, augmentation-ის
სამი დონე, დიდი vs პატარა არქიტექტურა. შევაფასე test-ზე accuracy-ით, macro-F1-ით და confusion matrix-ით, გავაანალიზე
შეცდომები, შემდეგ ONNX-ით დავაჩქარე ინფერენსი და გავაკეთე Gradio დემო."

## 2. ფაილი-ფაილ ახსნა
**src/data.py** — სურათების ჩატვირთვა. `Normalize(MEAN, STD)` ImageNet-ის სტატისტიკით, რადგან pretrained მოდელი ასეთ
შესასვლელს ელოდება. train-ზე random augmentation, val/test-ზე დეტერმინისტული (Resize 256 → CenterCrop 224).
trainval იყოფა 85/15-ად ფიქსირებული seed-ით; ორი dataset ობიექტი გვაქვს (ერთი train transform-ით, მეორე eval-ით),
indices ერთი და იგივეა, რომ val-ზე augmentation არ მოხვდეს. test მხოლოდ ბოლოს გამოიყენება.

**src/model.py** — `SimpleCNN`: 4 ბლოკი (Conv-BN-ReLU ×2 + MaxPool), Global Average Pooling, Dropout, Linear.
`build_model`: ResNet18/MobileNetV3-ს ბოლო ფენას ვცვლით `num_classes`-ზე. `freeze_backbone` ყინავს ყველა პარამეტრს
(`requires_grad=False`), გარდა თავისა → ე.წ. linear probe.

**src/train.py** — ციკლი: forward → loss → backward → optimizer step. AdamW (Adam + სწორი weight decay),
CosineAnnealing (lr ნელ-ნელა მცირდება), AMP (float16 გამოთვლები უფრო სწრაფი და ნაკლები მეხსიერება; GradScaler იცავს
underflow-სგან). საუკეთესო epoch ინახება **validation**-ის მიხედვით.

**src/evaluate.py** — test-ზე: accuracy, top-5, macro-F1, confusion matrix, ყველაზე ხშირი აღრევები, ყველაზე
თავდაჯერებული შეცდომების სურათები, ზომა/პარამეტრები/latency.

**src/export_onnx.py** — PyTorch → ONNX; ONNX Runtime-ით ვადარებთ გამოსავალს (max abs diff უნდა იყოს ~1e-5) და სიჩქარეს.

**app.py** — Gradio: სურათი → top-5 ალბათობა. **compare_runs.py** — ყველა run ერთ ცხრილში და გრაფიკზე.

## 3. ცნებები მარტივად
- **Transfer learning**: ImageNet-ზე მოდელმა უკვე ისწავლა კიდეები, ტექსტურები, ფორმები. ჩვენ მხოლოდ ვამუშავებთ ახალ ამოცანაზე.
  3 000 სურათზე from-scratch ძნელად დაეწევა.
- **Linear probe vs Fine-tuning**: პირველში მხოლოდ ბოლო ფენა ისწავლება (სწრაფი, უსაფრთხო); მეორეში მთელი ქსელი
  (ჩვეულებრივ უკეთესი, მაგრამ მცირე lr-ია საჭირო, რომ pretrained ცოდნა არ გავანადგუროთ).
- **Overfitting**: train accuracy მაღალია, val დაბალი. ნიშნები: val loss იზრდება. წამალი: augmentation, weight decay, dropout, მეტი მონაცემი, early stopping.
- **Augmentation**: სურათების შემთხვევითი ცვლილებით ხელოვნურად ვზრდით მრავალფეროვნებას.
- **Train/Val/Test**: train — სწავლა, val — არჩევანი (epoch, hyperparameters), test — საბოლოო პატიოსანი შეფასება. test-ზე არჩევანის გაკეთება = data leakage.
- **Accuracy vs Macro-F1**: accuracy საერთო სისწორეა; macro-F1 ყველა კლასს თანაბრად აფასებს (precision და recall-ის ჰარმონიული საშუალო), ამიტომ დისბალანსის დროს უფრო სანდოა.
- **Top-5**: სწორი ჯიში პირველ ხუთ პროგნოზში არის თუ არა — fine-grained ამოცანაში სასარგებლოა.
- **Confusion matrix**: რომელი კლასი რომელში ერევა.
- **Batch Normalization**: აქტივაციების ნორმალიზება, სწავლას ასტაბილურებს. **ReLU**: არაწრფივობა.
- **Cross-entropy**: სწორი კლასის −log(ალბათობას) სჯის; softmax logits-ს ალბათობებად გარდაქმნის.
- **ONNX**: მოდელის საერთო ფორმატი, ONNX Runtime ხშირად უფრო სწრაფია CPU-ზე და ფრეიმვორკისგან დამოუკიდებელია.
- **Latency vs throughput**: ერთი სურათის დრო vs სურათი წამში.

## 4. მოსალოდნელი კითხვები და პასუხების ლოგიკა
1. **რატომ ეს ამოცანა/ნაკრები?** ღია, საშუალო ზომის, Colab-ზე სწრაფი, fine-grained → საინტერესო ანალიზი.
2. **რატომ ResNet18 და არა უფრო დიდი?** რესურსის შეზღუდვა, სწრაფი ექსპერიმენტები, სანდო baseline; ახალი ფაქტი: MobileNet-ს შევადარე.
3. **რატომ pretrained?** მცირე მონაცემი; ექსპერიმენტით დავადასტურე scratch-თან შედარებით (შენი ციფრები).
4. **რატომ დაბალი lr fine-tuning-ზე?** დიდი lr გაანადგურებს pretrained წონებს (catastrophic forgetting).
5. **როგორ აირჩიე საუკეთესო მოდელი?** validation accuracy-ით, არა test-ით.
6. **რამ შეამცირა overfitting?** (შენი გრაფიკებით) augmentation, weight decay, cosine schedule.
7. **რატომ აგრესიულმა augmentation-მა არ/დაგეხმარა?** 10 epoch-ზე ძლიერი აუგმენტაცია შეიძლება underfit-ს იწვევდეს; ეს ჰიპოთეზაა და დამოკიდებულია შენს შედეგზე — პატიოსნად თქვი რასაც ხედავ.
8. **სად ცდება მოდელი?** `most_confused_pairs`-იდან: ვიზუალურად მსგავს ჯიშებს ერევა (მაგ. ბრიტანული/რუსული ცისფერი კატა — ნახე შენს შედეგში).
9. **რა გააუმჯობესებდი?** მეტი seed, hyperparameter search, უფრო ძლიერი backbone, TTA, Grad-CAM, INT8 quantization.
10. **ONNX-მა სიზუსტე შეცვალა?** max abs diff-ით დავადასტურე, რომ უმნიშვნელოა.
11. **რატომ არ გაქვს ერთი ექსპერიმენტი მრავალი seed-ით?** დროის/რესურსის შეზღუდვა; აღიარე, რომ სხვაობა 1% ცდომილების ფარგლებშია.
12. **რატომ AdamW?** ნაკლები tuning სჭირდება; weight decay სწორადაა გამოყოფილი.
13. **რა არის AMP და რატომ?** float16 → სწრაფი და ნაკლები მეხსიერება; GradScaler გრადიენტებს იცავს.
14. **როგორ გამოიყენე AI?** პატიოსნად: გენერაცია + სწავლა + შენი ცვლილებები.
15. **ცვლილება ლაივში**: მოემზადე, რომ შეგეძლოს: სხვა მოდელის დამატება `model.py`-ში, lr შეცვლა, ახალი augmentation `data.py`-ში, label smoothing (`nn.CrossEntropyLoss(label_smoothing=0.1)`).

## 5. გაშვების ნაბიჯები
1. შექმენი GitHub repo და ატვირთე ეს საქაღალდე (`git init`, `git add .`, `git commit`, `git push`).
2. Colab → Runtime → T4 GPU → `!git clone ...`, `!pip install -r requirements.txt`, `!bash run_experiments.sh` (~20–30 წთ).
3. `runs/comparison.md`-დან ციფრები გადაიტანე `REPORT_ka.md`-ში; ჩასვი `confusion_matrix.png`, `curves.png`.
4. `!python app.py --ckpt runs/resnet18_ft_basic/best.pt --share` → დემოს ბმული/ეკრანის ჩანაწერი.
5. `best.pt` ფაილები გაუზიარე Google Drive-ით (git-ში არ შედის), README-ში მიუთითე.
