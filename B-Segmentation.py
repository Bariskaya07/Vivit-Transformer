# ===============================  C  ==================================
# Mask-tabanlı optik akış (GPU TV-L1 -> CPU TV-L1 -> Farnebäck sırası)
# Gray-Scale frames de eklendi (Strateji B için)
# .npy ÇIKTILARI ANA DATASET KLASÖRLERİNE KAYDEDİLECEK (ALT KATEGORİLER OLMADAN)
# =====================================================================

import cv2
import torch
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
import traceback # Detaylı hata loglama için

# ---------------------------------------------------------------------
# 1)  YOL & PARAMETRELER  ------------------------------------------------
FLOW_ROOT = Path("/content/drive/MyDrive/Crime/Preprocessed_All")   # 224×224 klipler (.pt formatında RGB)
MASK_ROOT = Path("/content/drive/MyDrive/Crime/Masks_All")          # eşlenik maskeler (.pt formatında)
DATASETS  = {  # Hangi alt setler işlenecek? FLOW_ROOT altındaki ANA KLASÖR ADLARI OLMALI
    "Anomaly-Part-1", "Anomaly-Part-2",
    "Anomaly-Part-3", "Anomaly-Part-4",
    "Normal-Event-Recognition","Training-Normal-1",
    "Training-Normal-2", "Testing-Normal-Anomaly",
}
SCALE   = 20.0
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2)  Algoritma seçimi (Bu kısım aynı kalıyor) -------------------------
ALGO = ""
tvl1_fun = None
try:
    tvl1_gpu = cv2.cuda_OpticalFlowDual_TVL1.create()
    ALGO, tvl1_fun = "GPU_TVL1", lambda prev_gray, current_gray: tvl1_gpu.calc(prev_gray, current_gray, None).download()
    print(f"➜  Optical-flow algo: {ALGO} (GPU)")
except AttributeError:
    print("⚠️ GPU TVL1 bulunamadı veya CUDA destekli OpenCV derlenmemiş.")
    try:
        tvl1_cpu = cv2.optflow.DualTVL1OpticalFlow_create()  # type: ignore
        ALGO, tvl1_fun = "CPU_TVL1", lambda prev_gray, current_gray: tvl1_cpu.calc(prev_gray, current_gray, None)
        print(f"➜  Optical-flow algo: {ALGO} (CPU)")
    except AttributeError:
        print("⚠️ CPU TVL1 (optflow modülü) bulunamadı.")
        ALGO = "FARNEBACK"
        print(f"➜  Optical-flow algo: {ALGO} (OpenCV Dahili)")

def farneback(prev_gray, current_gray):
    return cv2.calcOpticalFlowFarneback(
        prev_gray, current_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 3)  Tek klip akış hesabı (GÜNCELLENDİ: output_npy_target_dir parametresi eklendi) ---
def compute_flow(clip_path: Path, mask_path: Path, output_npy_target_dir: Path):
    """
    Verilen bir RGB klip (.pt) ve maske dosyasından (.pt)
    u, v optik akış bileşenlerini ve gri tonlama karelerini hesaplar,
    belirtilen output_npy_target_dir altına .npy olarak kaydeder.
    """
    clip_tensor = torch.load(clip_path)
    clip_numpy = clip_tensor.numpy()
    masks_tensor = torch.load(mask_path)
    masks_numpy = masks_tensor.numpy().astype(bool)

    if clip_numpy.ndim != 4 or masks_numpy.ndim != 3:
        tqdm.write(f"‼️ HATA (boyut): {clip_path.name} veya {mask_path.name} beklenmedik boyutta. Klip: {clip_numpy.shape}, Maske: {masks_numpy.shape}")
        return

    C, T, H, W = clip_numpy.shape

    if masks_numpy.shape[0] != T or masks_numpy.shape[1] != H or masks_numpy.shape[2] != W:
        tqdm.write(f"‼️ HATA (eşleşme): {clip_path.name} ({T} frame) ve maske ({masks_numpy.shape[0]} frame) frame sayıları veya boyutları eşleşmiyor.")
        return

    num_flow_frames = T - 1
    if num_flow_frames < 1:
        tqdm.write(f"ℹ️ {clip_path.name} optik akış için yeterli frame sayısına sahip değil (Frame sayısı: {T}). Atlanıyor.")
        return

    u_flows = np.zeros((num_flow_frames, H, W), np.float32)
    v_flows = np.zeros((num_flow_frames, H, W), np.float32)
    g_frames = np.zeros((num_flow_frames, H, W), np.float32)

    for t in range(num_flow_frames):
        frame1_rgb_uint8 = (clip_numpy[:, t, :, :].transpose(1, 2, 0) * 255).astype(np.uint8)
        frame2_rgb_uint8 = (clip_numpy[:, t + 1, :, :].transpose(1, 2, 0) * 255).astype(np.uint8)
        frame1_gray = cv2.cvtColor(frame1_rgb_uint8, cv2.COLOR_RGB2GRAY)
        frame2_gray = cv2.cvtColor(frame2_rgb_uint8, cv2.COLOR_RGB2GRAY)

        flow = None
        if ALGO == "GPU_TVL1" or ALGO == "CPU_TVL1":
            if tvl1_fun is not None: flow = tvl1_fun(frame1_gray, frame2_gray)
        elif ALGO == "FARNEBACK":
            flow = farneback(frame1_gray, frame2_gray)

        if flow is None:
            tqdm.write(f"‼️ HATA (akış hesabı): {clip_path.name}, frame {t} için optik akış hesaplanamadı. Algoritma: {ALGO}")
            flow = np.zeros((H,W,2), dtype=np.float32)

        flow[~masks_numpy[t, :, :]] = 0
        u_flows[t, :, :] = flow[..., 0]
        v_flows[t, :, :] = flow[..., 1]
        g_frames[t, :, :] = frame1_gray.astype(np.float32) / 255.0

    # Dosya adının kökünü al (örn: Abuse001_x264). Bu kök adı, çıktı dosya adlarında kullanılacak.
    output_file_stem = clip_path.stem

    try:
        # GÜNCELLENMİŞ KAYIT YERİ: Belirtilen output_npy_target_dir klasörüne kaydet
        output_npy_target_dir.mkdir(parents=True, exist_ok=True) # Eğer hedef klasör yoksa oluştur
        np.save(output_npy_target_dir / f"{output_file_stem}_flow_u.npy", u_flows / SCALE)
        np.save(output_npy_target_dir / f"{output_file_stem}_flow_v.npy", v_flows / SCALE)
        np.save(output_npy_target_dir / f"{output_file_stem}_flow_g.npy", g_frames)
    except Exception as e:
        tqdm.write(f"‼️ HATA ({output_file_stem}): .npy dosyaları '{output_npy_target_dir}' dizinine kaydedilirken sorun: {e}")
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 4)  Toplu tarama (GÜNCELLENDİ: Kayıt yeri mantığı değişti) -----------
print(f"\n--- Script C Başlatılıyor: Optik Akış & Gri Tonlama Üretimi ---")
print(f"Kaynak .pt Kök Dizini (FLOW_ROOT): {FLOW_ROOT}")
print(f"Kaynak Maske Kök Dizini (MASK_ROOT): {MASK_ROOT}")
print(f"İşlenecek Dataset Klasörleri: {DATASETS}")
print(f".npy Çıktıları '{FLOW_ROOT}/[DatasetKlasörAdi]/' altına kaydedilecek.\n")

start_time_overall = time.time()
total_clips_to_check = 0
processed_count = 0
skipped_count = 0
error_count = 0
mask_missing_count = 0

for dataset_folder_name in DATASETS:
    # Her bir dataset için .npy'lerin kaydedileceği hedef ana klasör
    # Örn: /content/drive/MyDrive/Crime/Preprocessed_All/Anomaly-Part-1/
    target_npy_output_dir_for_dataset = FLOW_ROOT / dataset_folder_name

    # Bu dataset içindeki .pt dosyalarını (alt klasörler dahil) bul
    current_dataset_pt_path = FLOW_ROOT / dataset_folder_name
    if not current_dataset_pt_path.is_dir():
        print(f"UYARI: DATASETS içinde belirtilen '{dataset_folder_name}' klasörü {FLOW_ROOT} altında bulunamadı. Atlanıyor.")
        continue

    # .pt uzantılı dosyaları rekürsif olarak bul
    dataset_specific_pt_files = list(current_dataset_pt_path.rglob("*.pt"))
    # _mask.pt ile bitenleri hariç tut
    original_clip_files_in_dataset = [p for p in dataset_specific_pt_files if not p.name.endswith("_mask.pt")]

    if not original_clip_files_in_dataset:
        print(f"ℹ️ '{dataset_folder_name}' içinde işlenecek .pt dosyası bulunamadı.")
        continue

    total_clips_to_check += len(original_clip_files_in_dataset)
    print(f"\n'{dataset_folder_name}' içinde {len(original_clip_files_in_dataset)} klip işlenecek/kontrol edilecek.")

    for clip_file_path in tqdm(original_clip_files_in_dataset, desc=f"Dataset: {dataset_folder_name}"):
        # clip_file_path örn: .../Preprocessed_All/Anomaly-Part-1/Abuse/Abuse001_x264.pt

        # Maske dosyasının yolu, orijinal .pt dosyasının tam yapısını MASK_ROOT altında arar
        try:
            relative_path_for_mask_lookup = clip_file_path.relative_to(FLOW_ROOT)
        except ValueError:
            tqdm.write(f"‼️ HATA (göreli yol): {clip_file_path} dosyası {FLOW_ROOT} altında değil gibi. Atlanıyor.")
            error_count +=1
            continue

        mask_file_path = MASK_ROOT / relative_path_for_mask_lookup.parent / f"{clip_file_path.stem}_mask.pt"

        if not mask_file_path.exists():
            tqdm.write(f"⚠️ Maske ({mask_file_path.name}) bulunamadı. Atlanıyor: {relative_path_for_mask_lookup}")
            mask_missing_count += 1
            continue

        # Çıktı .npy dosyalarının tam yollarını oluştur (artık target_npy_output_dir_for_dataset altında)
        # Dosya adı olarak orijinal .pt dosyasının kök adını kullan (örn: Abuse001_x264)
        output_file_stem_for_npy = clip_file_path.stem

        u_flow_output_path = target_npy_output_dir_for_dataset / f"{output_file_stem_for_npy}_flow_u.npy"
        v_flow_output_path = target_npy_output_dir_for_dataset / f"{output_file_stem_for_npy}_flow_v.npy"
        g_flow_output_path = target_npy_output_dir_for_dataset / f"{output_file_stem_for_npy}_flow_g.npy"

        # Eğer u, v, VE g dosyalarının HEPSİ zaten hedefte varsa bu klibi atla
        if u_flow_output_path.exists() and \
           v_flow_output_path.exists() and \
           g_flow_output_path.exists():
            skipped_count += 1
            continue

        try:
            # compute_flow fonksiyonuna hedef kayıt dizinini de gönderiyoruz
            compute_flow(clip_file_path, mask_file_path, target_npy_output_dir_for_dataset)
            processed_count += 1
        except Exception as e:
            tqdm.write(f"‼️ HATA (compute_flow): {clip_file_path.name} işlenirken sorun: {e}")
            tqdm.write(traceback.format_exc()) # Daha detaylı hata için
            error_count += 1

total_duration_minutes = (time.time() - start_time_overall) / 60
print(f"\n✅ Optik Akış & Gri Tonlama Üretimi Tamamlandı.")
print(f"   Toplam Kontrol Edilen Klip Sayısı: {total_clips_to_check}")
print(f"   İşlenen Yeni Klip Sayısı: {processed_count}")
print(f"   Atlanan Klip Sayısı (Çıktıları Zaten Mevcuttu): {skipped_count}")
print(f"   Maskesi Eksik Olduğu İçin Atlanan Klip Sayısı: {mask_missing_count}")
print(f"   Hata Alınan Klip Sayısı: {error_count}")
print(f"   Toplam Süre: {total_duration_minutes:.1f} dk")