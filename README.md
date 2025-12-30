# Inference

Follow the steps below to run inference.

---

## 1. Download model weights

Download the pretrained weights from the link below:

https://drive.google.com/file/d/1ciSWlgjcFDkJnJk8wccZekwk1cbGUgVh/view?usp=sharing

---

## 2. Create output directory

```bash
mkdir -p outputs_pix2pix/shiq_lr4e4_problora_msl1_perceptual
```

---

## 3. Place the checkpoint file

Place the downloaded file `model_84001.pth` under:

```text
outputs_pix2pix/shiq_lr4e4_problora_msl1_perceptual/
```

---

## 4. Run inference

```bash
bash inference_paired_shiq.sh
```

---
