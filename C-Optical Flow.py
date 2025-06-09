"""## **D- Vivit Model**

### 4-A ▸ Flow-pencerelerini (32 frame) ViViT formatına dönüş

---
"""

# =====================================================================
# Script 4-A: Optik Akış (u,v) ve Gri Tonlama (g) Pencerelerini Oluşturma
# =====================================================================
# Strateji: Script C tarafından üretilen _flow_u.npy, _flow_v.npy ve
#           _flow_g.npy (gri tonlama) dosyalarını okur.
#           Bu 3 kanalı birleştirir, N_FRAMES uzunluğunda pencereler oluşturur
#           (gerekirse padding yapar) ve .pt dosyaları olarak kaydeder.
# =====================================================================

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm # İlerleme çubuğu için
import os # Belki bazı dosya işlemleri için (opsiyonel)

# ----- 1. TEMEL AYARLAR VE YOL TANIMLARI -----
print("--- Script 4-A Başlatılıyor: u,v,g_grayscale Pencereleri Oluşturma ---")

# Girdi: Script C'nin .npy dosyalarını kaydettiği ana klasör
FLOW_ROOT = Path("/content/drive/MyDrive/Crime/Preprocessed_All")

# Çıktı: Oluşturulacak 3 kanallı pencere (.pt) dosyalarının kaydedileceği ana klasör
OUT_ROOT  = Path("/content/drive/MyDrive/Crime/Flow_Windows_32_uvg")
OUT_ROOT.mkdir(parents=True, exist_ok=True) # Eğer yoksa oluştur

# Modelin beklediği pencere (klip) uzunluğu (zaman boyutu)
N_FRAMES = 32

# Kayan pencereler için pencereler arası kaydırma adımı
# Eğer orijinal akış N_FRAMES'ten uzunsa bu kullanılır.
STRIDE = 15

# Çıktı pencerelerindeki kanal sayısı (u, v, g_grayscale)
NUM_CHANNELS = 3

print(f"Girdi .npy Kök Dizini (FLOW_ROOT): {FLOW_ROOT}")
print(f"Çıktı .pt Pencere Kök Dizini (OUT_ROOT): {OUT_ROOT}")
print(f"Pencere Uzunluğu (N_FRAMES): {N_FRAMES}")
print(f"Kayan Pencere Adımı (STRIDE): {STRIDE}")
print(f"Çıktı Kanal Sayısı (NUM_CHANNELS): {NUM_CHANNELS}")

# ----- 2. İŞLENECEK DOSYALARIN BULUNMASI -----

# FLOW_ROOT altındaki tüm '_flow_u.npy' dosyalarını rekürsif olarak bul
# Bu, 'Anomaly-Part-X/KategoriAdi/' gibi alt klasörleri de tarar.
try:
    u_files = sorted(list(FLOW_ROOT.rglob("*_flow_u.npy")))
except Exception as e:
    print(f"‼️ HATA: {FLOW_ROOT} altında dosya aranırken sorun oluştu: {e}")
    u_files = [] # Hata durumunda boş liste ile devam etmeyi dene veya scripti sonlandır

if not u_files:
    print(f"UYARI: {FLOW_ROOT} dizininde ve alt klasörlerinde hiç '*_flow_u.npy' dosyası bulunamadı.")
    print("--- Script 4-A Sonu (İşlenecek dosya yok) ---")
    # exit() # Eğer dosya yoksa scripti burada sonlandırabilirsiniz
else:
    print(f"Toplam {len(u_files)} adet '_flow_u.npy' dosyası bulundu ve işlenecek.")

# ----- 3. PENCERELEME VE KAYDETME FONKSİYONU -----

def create_and_save_windows(u_file_path: Path):
    """
    Verilen bir _flow_u.npy dosyası ve eşlenik _v.npy, _g.npy dosyalarından
    N_FRAMES uzunluğunda pencereler oluşturur ve .pt olarak kaydeder.
    """

    # Eşlenik _v.npy ve _g.npy dosyalarının yollarını oluştur
    # Path.with_name, dosya adını değiştirirken yolu (klasörü) korur.
    v_file_path = u_file_path.with_name(u_file_path.name.replace("_flow_u.npy", "_flow_v.npy"))
    g_file_path = u_file_path.with_name(u_file_path.name.replace("_flow_u.npy", "_flow_g.npy"))

    # DEBUG: Hangi dosyaların arandığını yazdır
    # tqdm.write(f"🔍 İşlenen u: {u_file_path.name}")
    # tqdm.write(f"     Beklenen v: {v_file_path.name} (Yol: {v_file_path})")
    # tqdm.write(f"     Beklenen g: {g_file_path.name} (Yol: {g_file_path})")

    # Eşlenik dosyaların varlığını kontrol et
    if not v_file_path.exists():
        tqdm.write(f"⚠️ Eşlenik v dosyası ({v_file_path.name}) bulunamadı. Atlanıyor: {u_file_path.name}")
        return 0 # Kaydedilen pencere sayısı

    if not g_file_path.exists():
        # Bu print, sizin loglarınızda gördüğümüz "Eşlenik g dosyası bulunamadı" hatasını verir.
        tqdm.write(f"⚠️ Eşlenik g_grayscale dosyası ({g_file_path.name}) bulunamadı. Atlanıyor: {u_file_path.name}")
        return 0 # Kaydedilen pencere sayısı

    # Dosyaları yükle
    try:
        u_original_data = np.load(u_file_path)      # Beklenen şekil: (T_flow, H, W)
        v_original_data = np.load(v_file_path)      # Beklenen şekil: (T_flow, H, W)
        g_original_data = np.load(g_file_path)      # Beklenen şekil: (T_flow, H, W), değerler [0,1]
    except Exception as e:
        tqdm.write(f"‼️ HATA: {u_file_path.name} veya eşlenik .npy dosyaları yüklenirken sorun: {e}")
        return 0

    # Yüklenen dizilerin boyutlarını ve frame sayılarını kontrol et
    if not (u_original_data.ndim == 3 and v_original_data.ndim == 3 and g_original_data.ndim == 3):
        tqdm.write(f"UYARI: {u_file_path.name} veya eşleniklerinin boyut sayısı (ndim) 3 değil. Atlanıyor.")
        return 0

    if not (u_original_data.shape[0] == v_original_data.shape[0] == g_original_data.shape[0]):
        tqdm.write(f"UYARI: {u_file_path.name} için u,v,g frame sayıları eşleşmiyor. Atlanıyor. "
                   f"u:{u_original_data.shape[0]}, v:{v_original_data.shape[0]}, g:{g_original_data.shape[0]}")
        return 0

    # Yükseklik ve genişlik (H, W) değerlerinin de tutarlı olduğunu varsayıyoruz (genellikle 224, 224)
    # Onları da kontrol etmek isterseniz:
    # if not (u_original_data.shape[1:] == v_original_data.shape[1:] == g_original_data.shape[1:]):
    #     tqdm.write(f"UYARI: {u_file_path.name} için u,v,g H,W boyutları eşleşmiyor. Atlanıyor.")
    #     return 0

    current_num_frames = u_original_data.shape[0] # Mevcut akış/gri tonlama kare sayısı

    # Çıktı için göreli klasör yapısını FLOW_ROOT'a göre belirle
    try:
        relative_dir_structure = u_file_path.parent.relative_to(FLOW_ROOT)
    except ValueError:
        # Bu durum, u_file_path'in FLOW_ROOT altında olmaması gibi beklenmedik bir durumda oluşabilir.
        tqdm.write(f"‼️ HATA: {u_file_path} için göreli yol hesaplanamadı (FLOW_ROOT: {FLOW_ROOT}). Atlanıyor.")
        return 0

    output_target_dir = OUT_ROOT / relative_dir_structure
    output_target_dir.mkdir(parents=True, exist_ok=True) # Gerekirse oluştur

    # Dosya adının _flow_u kısmını çıkarıp temel adını al
    base_filename_stem = u_file_path.stem.replace("_flow_u", "")

    windows_saved_for_this_clip = 0

    # Durum 1: Mevcut frame sayısı N_FRAMES'ten KISA ise DOLDURMA (padding) yap
    if current_num_frames < N_FRAMES:
        if current_num_frames == 0: # Hiç frame yoksa atla
            tqdm.write(f"ℹ️ {u_file_path.name} hiç frame içermiyor (uzunluk: {current_num_frames}). Atlanıyor.")
            return 0

        padding_size = N_FRAMES - current_num_frames
        # ((öncesine_pad, sonrasına_pad), (H_ön, H_son), (W_ön, W_son))
        # Sadece zaman (ilk) boyutta sona doğru padding yapıyoruz.
        pad_width = ((0, padding_size), (0,0), (0,0))
        try:
            u_padded = np.pad(u_original_data, pad_width, mode='edge')
            v_padded = np.pad(v_original_data, pad_width, mode='edge')
            g_padded = np.pad(g_original_data, pad_width, mode='edge')
        except Exception as e:
            tqdm.write(f"‼️ HATA: {u_file_path.name} için padding sırasında hata: {e}")
            return 0

        # Doldurma sonrası tek bir pencere oluşur
        u_window, v_window, g_window = u_padded, v_padded, g_padded

        # Kanalları birleştir: (3, N_FRAMES, H, W)
        stacked_channels_clip = np.stack([u_window, v_window, g_window], axis=0)
        stacked_channels_clip = stacked_channels_clip.astype(np.float32) # Veri tipini float32 yap

        # Doldurulmuş klipler için tek bir pencere (örn: _win00000.pt)
        output_pt_filename = output_target_dir / f"{base_filename_stem}_win{0:05d}.pt"
        try:
            torch.save(torch.from_numpy(stacked_channels_clip), output_pt_filename)
            windows_saved_for_this_clip = 1
            # tqdm.write(f"ℹ️ Klip {u_file_path.stem} (uzunluk: {current_num_frames}) {N_FRAMES}'e DOLDURULDU ve '{output_pt_filename.name}' olarak kaydedildi.")
        except Exception as e:
            tqdm.write(f"‼️ HATA: {output_pt_filename.name} dosyası kaydedilirken sorun: {e}")

    # Durum 2: Mevcut frame sayısı N_FRAMES'e EŞİT veya DAHA UZUN ise kayan pencere uygula
    else:
        for s in range(0, current_num_frames - N_FRAMES + 1, STRIDE):
            u_window = u_original_data[s : s + N_FRAMES]
            v_window = v_original_data[s : s + N_FRAMES]
            g_window = g_original_data[s : s + N_FRAMES]

            stacked_channels_clip = np.stack([u_window, v_window, g_window], axis=0)
            stacked_channels_clip = stacked_channels_clip.astype(np.float32)

            output_pt_filename = output_target_dir / f"{base_filename_stem}_win{s:05d}.pt"
            try:
                torch.save(torch.from_numpy(stacked_channels_clip), output_pt_filename)
                windows_saved_for_this_clip += 1
            except Exception as e:
                tqdm.write(f"‼️ HATA: {output_pt_filename.name} dosyası kaydedilirken sorun: {e}")
        # if windows_saved_for_this_clip > 0:
        #     tqdm.write(f"ℹ️ Klip {u_file_path.stem} için {windows_saved_for_this_clip} pencere kaydedildi.")

    return windows_saved_for_this_clip

# ----- 4. TOPLU İŞLEME DÖNGÜSÜ -----
total_windows_created = 0
if u_files: # Eğer işlenecek .npy dosyası bulunduysa
    for u_single_file_path in tqdm(u_files, desc="Pencereler oluşturuluyor (u,v,g_gray)"):
        total_windows_created += create_and_save_windows(u_single_file_path)
    print(f"\nToplam {total_windows_created} pencere .pt dosyası oluşturuldu.")
else:
    print("İşlenecek _flow_u.npy dosyası bulunmadığı için pencere oluşturma işlemi yapılmadı.")

print(f"✅ Script 4-A Tamamlandı. Pencereler '{OUT_ROOT}' klasörüne kaydedilmeye çalışıldı.")

# ----- 5. (OPSİYONEL) KONTROL -----
# OUT_ROOT altındaki .pt dosyalarının sayısını kontrol edebilirsiniz:
# !find "/content/drive/MyDrive/Crime/Flow_Windows_32_uvg" -name "*.pt" | wc -l

!tree -L 2 /content/drive/MyDrive/Crime/Flow_Windows_32 | head -n 40

"""### **4-B ▸ PyTorch Dataset & DataLoader**"""

# ---------- 4-B ▸ PyTorch Dataset & DataLoader (u,v,g_grayscale) ----------
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import random
import traceback # Hata ayıklama için

# FLOW32_ROOT, Script 4-A'nın ÇIKIŞ KLASÖRÜ olmalı
FLOW32_ROOT = Path("/content/drive/MyDrive/Crime/Flow_Windows_32_uvg")
N_FRAMES = 32 # Script 4-A'daki N_FRAMES ile aynı olmalı

print(f"--- Script 4-B Başlatılıyor ---")
print(f"Kullanılacak Pencere Dosyaları Kök Dizini (FLOW32_ROOT): {FLOW32_ROOT}")
print(f"Beklenen Frame Sayısı (N_FRAMES): {N_FRAMES}")

# Etiketleme fonksiyonu
def is_anomaly(p: Path) -> int:
    """
    Dosya yoluna bakarak anomali (1) veya normal (0) etiketi üretir.
    Yolun herhangi bir kısmında 'normal' kelimesi (büyük/küçük harf duyarsız) geçerse normal kabul edilir.
    """
    try:
        # FLOW32_ROOT'a göreli yola bakmak yerine doğrudan tüm yola bakmak daha genel olabilir.
        # Çünkü p.parts tüm yol segmentlerini içerir.
        for part_name in p.parts:
            if "normal" in part_name.lower():
                return 0 # Normal
    except Exception as e:
        print(f"is_anomaly içinde hata ({p}): {e}") # Hata durumunda logla
        pass
    return 1 # Anomaly (eğer 'normal' bulunamazsa veya bir hata oluşursa)

# Örnek testler (isteğe bağlı, çalıştırmadan önce yolların varlığından emin olun)
# print("\n--- is_anomaly Fonksiyon Testleri ---")
# test_anomaly_file = FLOW32_ROOT / "Anomaly-Part-1" / "Abuse" / "Abuse001_x264win00000.pt" # Örnek dosya adı
# test_normal_file = FLOW32_ROOT / "Training-Normal-1" / "Normal_Videos_001_x264win00000.pt" # Örnek dosya adı
#
# if test_anomaly_file.exists():
#     print(f"'{test_anomaly_file.name}' için etiket: {is_anomaly(test_anomaly_file)} (Beklenen: 1)")
# else:
#     print(f"Test dosyası bulunamadı: {test_anomaly_file}")
#
# if test_normal_file.exists():
#     print(f"'{test_normal_file.name}' için etiket: {is_anomaly(test_normal_file)} (Beklenen: 0)")
# else:
#     print(f"Test dosyası bulunamadı: {test_normal_file}")


class FlowWindowDataset(Dataset):
    """(3, N_FRAMES, 224, 224) flow window (u,v,g_grayscale) + binary label"""
    def __init__(self, root: Path, n_frames: int, split="all", seed=42):
        self.root = root
        self.n_frames = n_frames

        print(f"\n'{self.split_name(split)}' Veri Seti için '{root}' taranıyor (beklenen frame sayısı: {n_frames})...")

        if not root.exists() or not root.is_dir():
            print(f"UYARI: Belirtilen kök dizin '{root}' bulunamadı veya bir klasör değil!")
            self.paths = []
            return # Hata durumunda boş dataset ile devam et

        paths = sorted([p for p in root.rglob("*win*.pt")]) # _win içeren .pt dosyaları

        if not paths:
            print(f"UYARI: '{root}' altında hiç '*win*.pt' dosyası bulunamadı!")

        if split != "all" and paths: # paths boş değilse split yap
            random.Random(seed).shuffle(paths)
            cut = int(len(paths) * 0.8) # %80 train, %20 val
            if split == "train":
                paths = paths[:cut]
            elif split == "val": # "val" veya "test" için
                paths = paths[cut:]
            else: # Tanımsız split adı
                print(f"UYARI: Tanımsız split adı '{split}'. Tüm veriler kullanılıyor.")

        self.paths = paths

        if not self.paths:
             print(f"UYARI: '{self.split_name(split)}' veri seti için hiç dosya yüklenemedi/bulunamadı (`self.paths` boş).")
        else:
             print(f"-> '{self.split_name(split)}' veri seti için {len(self.paths)} dosya bulundu.")

    def split_name(self, split_val):
        if split_val == "all": return "Tümü"
        if split_val == "train": return "Eğitim"
        if split_val == "val": return "Doğrulama"
        return split_val # Bilinmeyen split adı için olduğu gibi döndür

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if idx >= len(self.paths): # Olası bir index hatasını önle
            print(f"HATA: Geçersiz index {idx}, dataset boyutu {len(self.paths)}")
            return None

        p = self.paths[idx]
        try:
            clip = torch.load(p, map_location="cpu") # (3, N_FRAMES, H, W)
        except Exception as e:
            print(f"‼️ HATA: {p.name} yüklenirken sorun: {e}. Bu örnek atlanacak.")
            return None

        if not isinstance(clip, torch.Tensor):
            print(f"UYARI: {p.name} geçerli bir tensor yüklemedi (tip: {type(clip)}). Atlanacak.")
            return None

        if clip.shape[0] != NUM_CHANNELS or clip.shape[1] != self.n_frames: # NUM_CHANNELS global olmalı (veya parametre)
            print(f"UYARI: {p.name} beklenmedik boyutta: {clip.shape}. Beklenen ({NUM_CHANNELS}, {self.n_frames}, H, W). Atlanacak.")
            return None

        y = torch.tensor(is_anomaly(p), dtype=torch.long)
        return clip, y

# None (hatalı/atlanmış) örnekleri DataLoader'da filtrelemek için collate_fn
def collate_fn_skip_none(batch):
    # Gelen batch'i None olmayanlarla filtrele
    valid_batch = [item for item in batch if item is not None]

    if not valid_batch: # Eğer tüm batch (veya tek elemanlı batch) None ise
        # print("UYARI: collate_fn_skip_none: Tüm batch None, bu durum sorun yaratabilir.")
        return None # Çağıran kodun bunu işlemesi gerekir (örn: DataLoader döngüsünde if batch_data is None: continue)

    # Geçerli öğeler varsa, standart PyTorch collate fonksiyonunu kullan
    try:
        return torch.utils.data.dataloader.default_collate(valid_batch)
    except RuntimeError as e: # Eğer collate sırasında bir hata olursa (örn: tensor boyutları eşleşmiyorsa)
        print(f"HATA: collate_fn_skip_none içinde default_collate hatası: {e}")
        # Hatalı durumu anlamak için batch içeriğini yazdırabilirsiniz
        # for i, item in enumerate(valid_batch):
        #     if item is not None and isinstance(item, tuple) and len(item) > 0 and isinstance(item[0], torch.Tensor):
        #         print(f"  valid_batch[{i}] tensor şekli: {item[0].shape}")
        #     else:
        #         print(f"  valid_batch[{i}] beklenmedik yapıda: {item}")
        return None # Hata durumunda None döndür

# Script 4-A'dan gelen NUM_CHANNELS değişkenini burada da tanımlayalım (FlowWindowDataset içinde kullanılıyor)
NUM_CHANNELS = 3 # u,v,g_grayscale

print(f"\n--- DataLoader Örnek Kullanımı (N_FRAMES={N_FRAMES}) ---")
try:
    # Dataset nesnelerini oluştur
    train_ds = FlowWindowDataset(FLOW32_ROOT, n_frames=N_FRAMES, split="train")
    val_ds   = FlowWindowDataset(FLOW32_ROOT, n_frames=N_FRAMES, split="val")

    # Sadece dataset'ler boş değilse DataLoader oluştur
    if len(train_ds) > 0 and len(val_ds) > 0:
        BATCH_SIZE = 4 # Deneyler için küçük bir batch boyutu
        WORKERS    = 2 # Colab için genellikle 2 iyi bir başlangıçtır
        PIN_MEMORY = torch.cuda.is_available() # GPU varsa pin_memory kullan

        train_loader = DataLoader(train_ds,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=WORKERS,
                                  pin_memory=PIN_MEMORY,
                                  collate_fn=collate_fn_skip_none,
                                  drop_last=True) # Eğitimde son, tam olmayan batch'i atla

        val_loader   = DataLoader(val_ds,
                                  batch_size=BATCH_SIZE,
                                  shuffle=False,
                                  num_workers=WORKERS,
                                  pin_memory=PIN_MEMORY,
                                  collate_fn=collate_fn_skip_none)

        print(f"\nTrain DataLoader: {len(train_ds):,} pencere, {len(train_loader) if train_loader else 0} batch.")
        print(f"Val   DataLoader: {len(val_ds):,} pencere, {len(val_loader) if val_loader else 0} batch.")

        # Hızlı bir "sanity check" (sağlamlık kontrolü)
        print("\n--- Sanity Check: Eğitim DataLoader'dan Örnek Batch ---")
        if train_loader and len(train_loader) > 0:
            batch_sample = next(iter(train_loader)) # İlk batch'i al

            if batch_sample is not None: # collate_fn None döndürmediyse
                x_batch, y_batch = batch_sample
                print("Örnek batch tensör ve etiket boyutları:")
                print(f"  x_batch.shape: {x_batch.shape}") # Beklenen: [BATCH_SIZE, NUM_CHANNELS, N_FRAMES, Yükseklik, Genişlik]
                print(f"  y_batch.shape: {y_batch.shape}") # Beklenen: [BATCH_SIZE]
                if x_batch.numel() > 0 : # Eğer batch boş değilse
                    print(f"  Bu batch'teki anomali oranı (yaklaşık): {y_batch.float().mean().item():.3f}")
            else:
                print("Sanity check için train_loader'dan geçerli bir batch alınamadı (muhtemelen tüm örnekler hatalıydı).")
        else:
            print("Sanity check için train_loader boş veya oluşturulamadı.")
    else:
        print("\nUYARI: Eğitim veya doğrulama veri seti boş olduğu için DataLoader'lar oluşturulmadı.")
        print("Lütfen FLOW32_ROOT yolunun doğru olduğundan ve Script 4-A'nın başarıyla çalışıp .pt dosyaları ürettiğinden emin olun.")

except FileNotFoundError as fnf_err:
    print(f"‼️ HATA (Dosya Bulunamadı): {fnf_err}")
    print(f"Lütfen FLOW32_ROOT yolunun ('{FLOW32_ROOT}') doğru olduğundan ve Script 4-A'nın .pt dosyalarını bu klasöre oluşturduğundan emin olun.")
except Exception as e:
    print(f"‼️ HATA: Dataset/DataLoader oluşturulurken genel bir sorun oluştu: {e}")
    print(traceback.format_exc())

print("\n--- Script 4-B Tamamlandı ---")

"""### 4-C ▸ ViViT-B-16×2 modelini yükleme (+ GPU’ya taşıma, Fine Tune'a hazırlık)"""

# B) doğru sürümü kurun  (0.1.5 son kararlı, 0.1.3 de olur)
!pip install pytorchvideo==0.1.5            # veya 0.1.3
!pip install --upgrade einops safetensors   # 4‑C kodunuz için
!pip install transformers torch accelerate # Accelerate genellikle önerilir

# =====================================================================
# Script 4-C: ViViT Modelini Yükleme ve Uyarlama (u,v,g_grayscale için)
# =====================================================================
# Strateji: Hugging Face'den önceden eğitilmiş ViViT modelini yükler.
#           Giriş katmanını 3 kanallı (u,v,g_grayscale) girdiyi kabul
#           edecek şekilde uyarlar. Sınıflandırıcı katmanını kendi
#           problemimizdeki sınıf sayısına göre ayarlar.
#           Belirli katmanları dondurup, bazılarını eğitim için açar.
# =====================================================================

import torch
import torch.nn as nn
import torch.optim as optim # Optimizer tanımlaması için
from transformers import AutoModelForVideoClassification
from pathlib import Path # Gerekirse diye eklendi, doğrudan kullanılmıyor olabilir
import traceback # Hata ayıklama için

print("--- Script 4-C Başlatılıyor: ViViT Modeli Yükleme ve Uyarlama ---")

# ----- 1. TEMEL YAPILANDIRMA -----
MODEL_NAME_HF = "prathameshdalal/vivit-b-16x2-kinetics400-UCF-Crime"
NUM_CLASSES = 14  # Sizin projenizdeki hedef sınıf sayısı
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılacak Cihaz: {DEVICE}")
print(f"Hedef Sınıf Sayısı: {NUM_CLASSES}")

# Diferansiyel Öğrenme Oranları (PreTrain bölümünüzdeki gibi)
LR_CLASSIFIER = 3e-4
LR_PATCH_EMBED_PROJECTION = 5e-4 # Sadece projeksiyon katmanı için veya tüm patch_embeddings için
LR_ENCODER_UNFROZEN_BLOCKS = 3e-5
WEIGHT_DECAY_CONFIG = 1e-2

hf_model = None # Modeli bu script kapsamında tanımlayıp hazırlayacağız
model_ready_for_training = False
optimizer = None

# ----- 2. MODELİ YÜKLEME VE UYARLAMA -----
try:
    print(f"\n'{MODEL_NAME_HF}' modeli Hugging Face'den yükleniyor...")
    hf_model_base = AutoModelForVideoClassification.from_pretrained(MODEL_NAME_HF)
    print(f"-> Model başarıyla yüklendi. Orijinal veri tipi: {next(hf_model_base.parameters()).dtype}")

    # --- 2a. Sınıflandırıcı (Classifier) Katmanını Değiştirme ---
    original_num_classes = hf_model_base.classifier.out_features
    if original_num_classes != NUM_CLASSES:
        in_features = hf_model_base.classifier.in_features
        hf_model_base.classifier = torch.nn.Linear(in_features, NUM_CLASSES)
        print(f"-> Sınıflandırıcı katmanı {original_num_classes} sınıftan {NUM_CLASSES} sınıfa değiştirildi.")
    else:
        print(f"-> Sınıflandırıcı katmanı zaten {NUM_CLASSES} sınıflı.")

    # --- 2b. Giriş Projeksiyon Katmanını 3 Kanala (u,v,g_grayscale) Uyarlama ---
    print("Giriş projeksiyon katmanı 3 KANAL (u,v,g_grayscale) için uyarlanıyor...")
    original_proj_layer = hf_model_base.vivit.embeddings.patch_embeddings.projection

    # Orijinal projeksiyon katmanının parametrelerini al
    original_in_channels = original_proj_layer.in_channels
    out_channels = original_proj_layer.out_channels
    kernel_size = original_proj_layer.kernel_size
    stride = original_proj_layer.stride
    padding = original_proj_layer.padding
    dilation = original_proj_layer.dilation
    groups = original_proj_layer.groups
    has_bias = (original_proj_layer.bias is not None)
    print(f"  Orijinal projeksiyon katmanı: in_channels={original_in_channels}, out_channels={out_channels}, kernel_size={kernel_size}")

    # Yeni projeksiyon katmanını bizim 3 giriş kanalımız için oluştur
    # (u,v,g_grayscale de 3 kanal olduğu için in_channels=3 olacak)
    new_in_channels = 3
    new_projection_layer = nn.Conv3d(
        in_channels=new_in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=has_bias
    )

    with torch.no_grad(): # Gradyan hesaplamasını bu blokta durdur
        # Orijinal ağırlıkları (genellikle RGB için [out_ch, 3, Kt, Kh, Kw]) al
        original_weights = original_proj_layer.weight.data

        if original_in_channels == 3: # Eğer orijinal model RGB (3 kanal) ise
            # Strateji: Orijinal RGB ağırlıklarının ortalamasını alıp yeni 3 kanal için kullan.
            # Bu, her çıkış filtresi için R,G,B ağırlıklarının ortalamasını alır.
            mean_original_channel_weights = torch.mean(original_weights, dim=1, keepdim=True)
            # Bu ortalama ağırlığı bizim yeni 3 kanalımız (u,v,g_grayscale) için tekrarla.
            adapted_weights = mean_original_channel_weights.repeat(1, new_in_channels, 1, 1, 1)
            new_projection_layer.weight.data.copy_(adapted_weights)
            print("  -> Projeksiyon katmanı ağırlıkları, orijinal 3 (RGB) kanal ağırlıklarının ortalaması alınarak "
                  f"yeni {new_in_channels} (u,v,g_grayscale) kanal için kopyalandı.")
        else:
            # Orijinal model 3 kanallı değilse (beklenmedik durum) veya farklı bir strateji izlemek isterseniz.
            # Şimdilik, eğer orijinal 3 değilse, yeni katman rastgele başlatılmış ağırlıklarıyla kalır.
            # Ya da isterseniz burada hata verebilir veya farklı bir başlatma yapabilirsiniz.
            print(f"  UYARI: Orijinal projeksiyon katmanı {original_in_channels} kanallıydı (3 değil). "
                  f"Yeni {new_in_channels} kanallı katmanın ağırlıkları varsayılan (rastgele) başlatma ile bırakıldı.")

        if has_bias and new_projection_layer.bias is not None:
            new_projection_layer.bias.data.copy_(original_proj_layer.bias.data)
            print("  -> Projeksiyon katmanı bias'ı (varsa) kopyalandı.")

    # Yeni oluşturulan ve ağırlıkları uyarlanan katmanı modele ata
    hf_model_base.vivit.embeddings.patch_embeddings.projection = new_projection_layer
    hf_model = hf_model_base # Modeli hf_model değişkenine ata
    print(f"  -> Yeni projeksiyon katmanı (in_channels={new_in_channels}) modele atandı. "
          f"Ağırlık şekli: {new_projection_layer.weight.shape}")

    # --- 2c. Katmanları Dondurma / Eğitime Açma ---
    # "PreTrain" bölümünüzdeki stratejiyi izleyerek:
    #   - Önce tüm parametreleri dondur.
    #   - Sonra sınıflandırıcıyı, tüm patch embeddings modülünü ve encoder'ın son birkaç bloğunu aç.
    print("Katmanlar donduruluyor ve belirtilenler eğitime açılıyor...")
    for param_name, param in hf_model.named_parameters():
        param.requires_grad = False # Önce tümünü dondur

    # Sınıflandırıcıyı (classifier) eğitilebilir yap
    for param_name, param in hf_model.classifier.named_parameters():
        param.requires_grad = True
    print("  -> Sınıflandırıcı (classifier) katmanı eğitilebilir yapıldı.")

    # Tüm Patch Embeddings modülünü (projeksiyon dahil) eğitilebilir yap
    # (vivit.embeddings.patch_embeddings)
    for param_name, param in hf_model.vivit.embeddings.patch_embeddings.named_parameters():
        param.requires_grad = True
    print("  -> Tüm Patch Embeddings modülü (patch_embeddings) eğitilebilir yapıldı.")

    # ViViT encoder bloklarının bir kısmını aç (opsiyonel, "PreTrain"deki gibi son 6 blok)
    if hasattr(hf_model, 'vivit') and hasattr(hf_model.vivit, 'encoder') and hasattr(hf_model.vivit.encoder, 'layer'):
        total_encoder_blocks = len(hf_model.vivit.encoder.layer)
        num_blocks_to_unfreeze_target = 6 # Eğitime açılacak son blok sayısı

        num_blocks_to_unfreeze = 0
        if total_encoder_blocks > 0:
            num_blocks_to_unfreeze = min(num_blocks_to_unfreeze_target, total_encoder_blocks)
            if num_blocks_to_unfreeze_target > total_encoder_blocks:
                print(f"  UYARI: İstenen {num_blocks_to_unfreeze_target} açılacak blok sayısı, toplam {total_encoder_blocks} bloktan fazla. "
                      f"Son {total_encoder_blocks} blok açılacak.")
            elif num_blocks_to_unfreeze_target <= 0:
                 num_blocks_to_unfreeze = 0 # Hiç blok açma

        if num_blocks_to_unfreeze > 0:
            print(f"  -> ViViT encoder'ın son {num_blocks_to_unfreeze} bloğu (toplam {total_encoder_blocks} bloktan) eğitilebilir yapılıyor:")
            for i in range(total_encoder_blocks - num_blocks_to_unfreeze, total_encoder_blocks):
                print(f"      Encoder blok {i} açılıyor...")
                for param_name, param in hf_model.vivit.encoder.layer[i].named_parameters():
                    param.requires_grad = True
        elif total_encoder_blocks > 0 : # num_blocks_to_unfreeze == 0 ise
             print("  -> ViViT encoder blokları eğitilebilir yapılmayacak (0 blok hedeflendi).")
        else: # total_encoder_blocks == 0 ise
            print("  Encoder'da açılabilecek blok bulunmuyor (muhtemelen model yapısı farklı).")
    else:
        print("  UYARI: Model yapısı beklenenden farklı (vivit.encoder.layer bulunamadı), encoder blokları otomatik açılamadı.")

    # --- 2d. Modeli Cihaza Taşıma ve Son Kontroller ---
    hf_model.to(DEVICE)
    print(f"Model '{DEVICE}' cihazına taşındı (Veri tipi: FP32).")

    num_total_params = sum(p.numel() for p in hf_model.parameters())
    num_trainable_params = sum(p.numel() for p in hf_model.parameters() if p.requires_grad)
    print(f"\nModel Parametreleri:")
    print(f"  Toplam parametre sayısı    : {num_total_params:,}")
    print(f"  Eğitilebilir parametre sayısı: {num_trainable_params:,} (%{100 * num_trainable_params / num_total_params:.2f})")

    model_ready_for_training = True
    print("\n--- Model Hazırlığı (3-Kanal u,v,g_grayscale için) Tamamlandı ---")

except Exception as e:
    print(f"‼️ HATA: Script 4-C'de model yükleme veya uyarlama sırasında bir sorun oluştu: {e}")
    print(traceback.format_exc())

# ----- 3. OPTIMIZER TANIMLAMASI (PreTrain bölümündeki gibi diferansiyel LR ile) -----
# Bu kısım, model başarıyla hazırlandıysa ve eğitim script'inizin bir parçasıysa çalıştırılır.
# Eğer optimizer'ı ana eğitim döngüsünde tanımlıyorsanız, bu bloğu orada kullanabilirsiniz.
if model_ready_for_training and hf_model is not None:
    print("\nOptimizer (AdamW) diferansiyel öğrenme oranlarıyla tanımlanıyor...")

    param_groups = []

    # Sınıflandırıcı parametreleri
    classifier_params = [p for n, p in hf_model.named_parameters() if 'classifier' in n and p.requires_grad]
    if classifier_params:
        param_groups.append({'params': classifier_params, 'lr': LR_CLASSIFIER, 'name': 'classifier'})
        print(f"  -> Classifier parametreleri ({sum(p.numel() for p in classifier_params):,}) için LR: {LR_CLASSIFIER}")

    # Patch Embeddings (projeksiyon dahil) parametreleri
    patch_embed_params = [p for n, p in hf_model.named_parameters() if 'vivit.embeddings.patch_embeddings' in n and p.requires_grad]
    if patch_embed_params:
        param_groups.append({'params': patch_embed_params, 'lr': LR_PATCH_EMBED_PROJECTION, 'name': 'patch_embeddings'})
        print(f"  -> Patch Embeddings parametreleri ({sum(p.numel() for p in patch_embed_params):,}) için LR: {LR_PATCH_EMBED_PROJECTION}")

    # Geri kalan eğitilebilir backbone parametreleri (açılan encoder blokları vb.)
    other_trainable_backbone_params = [
        p for n, p in hf_model.named_parameters()
        if p.requires_grad and
           'classifier' not in n and
           'vivit.embeddings.patch_embeddings' not in n
    ]
    if other_trainable_backbone_params:
        param_groups.append({'params': other_trainable_backbone_params, 'lr': LR_ENCODER_UNFROZEN_BLOCKS, 'name': 'opened_encoder_blocks'})
        print(f"  -> Diğer açılmış backbone parametreleri ({sum(p.numel() for p in other_trainable_backbone_params):,}) için LR: {LR_ENCODER_UNFROZEN_BLOCKS}")

    if param_groups:
        optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY_CONFIG)
        print(f"-> Diferansiyel AdamW optimizer başarıyla tanımlandı. Weight Decay: {WEIGHT_DECAY_CONFIG}")
    else:
        print("UYARI: Optimizer için hiç eğitilebilir parametre grubu bulunamadı!")
        optimizer = None # Hata durumunda optimizer'ı None yap
else:
    if not model_ready_for_training:
        print("\nModel hazırlanamadığı için optimizer tanımlanmadı.")

# Artık `hf_model` ve `optimizer` (eğer tanımlandıysa) ana eğitim döngünüzde kullanılmaya hazırdır.
# Eğitim için bir `scaler` (torch.cuda.amp.GradScaler) de gerekebilir eğer AMP kullanacaksanız.
# `USING_HALF = False` (veya True) ve `scaler = torch.cuda.amp.GradScaler(enabled=USING_HALF)`
# tanımlamalarını eğitim döngünüzün başına ekleyebilirsiniz.