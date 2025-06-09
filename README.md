Bu proje, güvenlik kameralarından elde edilen videolarda anormal olayları tespit etmek amacıyla geliştirilen bir video işleme sistemidir. UCF-Crime veri seti üzerinde çalışılmış ve Video Vision Transformer (ViViT) mimarisi kullanılarak bir derin öğrenme modeli oluşturulmuştur. Çevresel gürültüyü azaltmak için öncelikle Mask R-CNN ile insan ve araç gibi nesnelerin bulunduğu bölgeler tespit edilmiş, ardından bu bölgeler üzerinden TV-L1 yöntemiyle optik akış hesaplanmıştır. Elde edilen hareket bilgisi ViViT modeline giriş olarak verilmiş ve model anomalileri yüksek doğrulukla sınıflandırabilmiştir. Bu yaklaşım, klasik ViViT modellerine kıyasla %37 daha yüksek doğruluk ve %45 daha iyi F1 skoru elde etmiştir.

---

````markdown
# 🧠 ViViT-Flow: Anomaly Detection in Surveillance Videos

A deep learning-based video anomaly detection pipeline using **ViViT (Video Vision Transformer)**, enhanced with **instance segmentation** and **masked optical flow** preprocessing. Applied on the **UCF-Crime dataset** for detecting real-world anomalies in surveillance footage.

---

## 📂 Project Structure

```bash
vivit-anomaly-detection/
├── segmentation/             # Mask R-CNN instance-level segmentation
├── optical_flow/             # TV-L1 method for dense optical flow
├── vivit_model/              # ViViT-B/16x2 model and fine-tuning code
├── dataset/                  # Preprocessed UCF-Crime video clips
├── notebooks/                # Jupyter notebooks for experiments
├── utils/                    # Utility functions
├── requirements.txt
└── README.md
````

---

## 🔍 Overview

This project proposes a novel ViViT-based anomaly detection pipeline:

1. **Mask R-CNN** is used to extract human/object regions from frames.
2. **TV-L1 optical flow** is computed over segmentation masks to focus on motion-relevant areas.
3. (u, v, g) flow triplets are fed into a ViViT-B/16x2 model pre-trained on Kinetics-400.
4. **Focal Loss** is applied to address class imbalance between normal and anomalous events.

---

## 🧪 Dataset

* **UCF-Crime** dataset: 1900 surveillance videos across 13 anomaly types.
* This work uses a subset (290 videos): 145 normal, 145 anomalous.
* Frame size: **224x224**, Clip length: **32 frames**, Input shape: (3 × 32 × 224 × 224)

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/vivit-anomaly-detection.git
cd vivit-anomaly-detection
pip install -r requirements.txt
```

---

## 🚀 Run Pipeline

```bash
# 1. Instance Segmentation
python segmentation/run_segmentation.py

# 2. Compute Masked Optical Flow
python optical_flow/compute_flow.py

# 3. Train ViViT Model
python vivit_model/train.py
```

You can also follow the `notebooks/ViViT_Training.ipynb` for visual, step-by-step execution.

---

## 📊 Results

### 📌 Evaluation Metrics on UCF-Crime (Binary Classification)

| Model                 | Accuracy  | Precision | Recall    | F1-Score  | F1 (Normal) | F1 (Anomaly) |
| --------------------- | --------- | --------- | --------- | --------- | ----------- | ------------ |
| **ViViT-Flow (Ours)** | **88.3%** | **0.865** | **0.869** | **0.868** | 0.820       | 0.907        |
| ViViT (RGB-only)      | 51.3%     | 0.402     | 0.425     | 0.407     | 0.156       | 0.658        |

> ✅ Our ViViT-Flow model significantly outperforms the baseline (ViViT on raw RGB) by +37% in accuracy and +45% in F1-score.

---

### 📚 Comparison with State-of-the-Art

| Method                 | Accuracy  | Precision | Recall    | F1-Score  |
| ---------------------- | --------- | --------- | --------- | --------- |
| CNN + RNN \[7]         | 82.5%     | 0.760     | 0.870     | 0.813     |
| 1D Transformer \[8]    | 80.7%     | 0.784     | 0.856     | –         |
| Weak Supervision \[11] | 81.2%     | 0.745     | 0.820     | 0.782     |
| **ViViT-Flow (Ours)**  | **88.3%** | **0.865** | **0.869** | **0.868** |

---

## 🛠️ Model Details

* **Backbone**: ViViT-B/16x2 (Pretrained on Kinetics-400)
* **Loss Function**: Focal Loss (γ = 1.5)
* **Input**: Segmentation-masked TV-L1 optical flow (u, v, g)
* **Optimizer**: AdamW
* **Training**: 10 epochs on NVIDIA T4, batch size 8, gradient accumulation 8

---

## 💡 Future Work

* [ ] Add video augmentation using diffusion models (for rare anomaly simulation)
* [ ] Deploy ViViT-Flow on edge devices
* [ ] Extend to multi-class anomaly detection

---

## 📜 License

MIT License - see [`LICENSE`](./LICENSE) for more info.

---

## 🙏 Acknowledgements

Special thanks to:

* **Dr. Erkut Arıcan** for mentorship and guidance
* **PyTorchVideo**, **Detectron2**, and **Hugging Face** for open-source libraries

```

---

Bu dosyayı doğrudan `README.md` olarak yapıştırabilirsin. Eğer istersen sana `.md` formatında dosyayı da verebilirim. Projeyi GitHub'a yükledikten sonra görseller eklemeyi veya örnek bir inference videosu (GIF formatında) koymayı da düşünebilirsin.

Hazır mısın yoksa son bir güncelleme yapalım mı?
```
---

## 🧠 ViViT-Based Anomaly Detection on UCF-Crime Dataset

A deep learning-based video anomaly detection pipeline using **ViViT (Video Vision Transformer)**, enhanced with **instance segmentation** and **optical flow** preprocessing. Applied on the **UCF-Crime dataset** for detecting real-world anomalies in surveillance footage.

---

### 📂 Project Structure

```bash
vivit-anomaly-detection/
│
├── segmentation/               # Mask R-CNN for instance-level segmentation
├── optical_flow/               # TV-L1 method to compute flow (u, v, g)
├── vivit_model/                # ViViT model code and fine-tuning scripts
├── dataset/                    # Preprocessed UCF-Crime clips
├── utils/                      # Helper functions for I/O, plotting, etc.
├── notebooks/                  # Jupyter Notebooks for experiments
├── requirements.txt            # Python dependencies
└── README.md
```

---

### 🔍 Project Description

This project implements an anomaly detection pipeline consisting of:

1. **Instance Segmentation** using Mask R-CNN (to extract regions of interest like humans and vehicles),
2. **TV-L1 Optical Flow** calculation masked by segmentation output,
3. Feeding optical flow triplets `(u, v, g)` into **ViViT**, a transformer-based video classifier,
4. **Fine-tuning ViViT** on the `UCF-Crime` dataset for binary anomaly classification (normal vs abnormal).

---

### 🧪 Dataset

* **Dataset**: [UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/)
* **Classes**: Abuse, Arson, Assault, Burglary, Explosion, Robbery, etc.
* **Input Shape**: (T, H, W, C) = (32, 224, 224, 3)

---

### 📦 Requirements

```bash
pip install -r requirements.txt
```

Main libraries:

* PyTorch
* TorchVision
* PyTorchVideo
* OpenCV
* Detectron2 (for Mask R-CNN)
* scikit-learn
* numpy

---

### 🚀 How to Run

```bash
# Step 1: Segment frames using Mask R-CNN
python segmentation/run_segmentation.py

# Step 2: Compute Optical Flow (TV-L1)
python optical_flow/compute_flow.py

# Step 3: Train or Fine-Tune ViViT
python vivit_model/train.py
```

Or use `notebooks/ViViT_Training.ipynb` for step-by-step visual training.

---

### 📊 Evaluation Metrics

* Accuracy, Precision, Recall
* AUC-ROC
* Confusion Matrix


### 📜 License

MIT License - see [`LICENSE`](./LICENSE) for more information.

---

### 🤝 Acknowledgements

* [ViViT Paper (Google Research)](https://arxiv.org/abs/2103.15691)
* [UCF Crime Dataset](https://www.crcv.ucf.edu/projects/real-world/)
* [PyTorchVideo](https://pytorchvideo.org/)
* [Detectron2](https://github.com/facebookresearch/detectron2)

