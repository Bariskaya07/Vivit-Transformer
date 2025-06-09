"""## **B- Segmentation with Mask R-CNN**"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn

# 1️⃣ Modeli yükle ve eval moduna geçir
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 2️⃣ Videodan ilk kareyi oku (240×320 BGR)
cap = cv2.VideoCapture(
    "/content/drive/MyDrive/Crime/Anomaly-Videos-Part-1/Assault/Assault036_x264.mp4"
)
ret, bgr = cap.read(); cap.release()
frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # RGB uint8

# 3️⃣ Görüntüyü tensor’a dönüştür ve normalize et
img_tensor = F.to_tensor(frame)  # float32 [0–1], shape (3,H,W)

# 4️⃣ Inference
with torch.no_grad():
    outputs = model([img_tensor])[0]
    # outputs["masks"]: (N,1,H,W) float in [0,1]
    # outputs["labels"], outputs["scores"]

# 5️⃣ Yüksek güven skoruna sahip “person” maskelerini ayıkla
person_indices = [
    i for i,(lab,score) in enumerate(zip(outputs["labels"], outputs["scores"]))
    if lab==1 and score>0.5
]
if person_indices:
    masks = outputs["masks"][person_indices,0].cpu().numpy()  # (K,H,W)
    # Binarize
    masks = masks > 0.5
    # Tüm person objelerini birleştir
    combined_mask = np.any(masks, axis=0)
else:
    combined_mask = np.zeros(frame.shape[:2], dtype=bool)

# 6️⃣ Overlay görselleştirme
overlay = frame.copy()
overlay[combined_mask] = [255, 0, 0]  # kırmızı
blend = ((frame*0.5) + (overlay*0.5)).astype(np.uint8)

# 7️⃣ Sonuçları göster
fig, axes = plt.subplots(1,3,figsize=(15,5))
axes[0].imshow(frame);       axes[0].set_title("Orijinal"); axes[0].axis("off")
axes[1].imshow(combined_mask, cmap="gray"); axes[1].set_title("Person Mask"); axes[1].axis("off")
axes[2].imshow(blend);       axes[2].set_title("Overlay");  axes[2].axis("off")
plt.show()
