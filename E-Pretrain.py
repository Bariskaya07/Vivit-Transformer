# =====================================================================
# PreTrain - BİRLEŞTİRİLMİŞ ANA EĞİTİM VE DEĞERLENDİRME BLOĞU (BİNARY İÇİN - TAMAMI - DÜZELTİLMİŞ + JSON ÇIKTISI)
# =====================================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path, PurePosixPath
import numpy as np
import pandas as pd # Pandas importunu başa alalım
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import contextlib
import random
import time
import math
import traceback
import collections
import re
from transformers import get_linear_schedule_with_warmup

print("\n--- PreTrain: Birleştirilmiş Ana Eğitim Bloğu (BİNARY İÇİN + DÜZELTİLMİŞ + JSON ÇIKTISI) Başlatılıyor ---")

# ----- Gerekli Değişkenlerin Varlığını Kontrol Etme (Blok 0/1'den) -----
required_vars_from_block01_exist = True
vars_to_check_from_block01 = [
    'model_ready', 'optimizer_ok', 'hf_model', 'optimizer', 'DEVICE', 'DATA_ROOT',
    'TRAIN_FLOW_ROOT_LIST', 'TRAIN_SPLIT_FILE', 'TEST_SPLIT_FILE', 'VAL_SCAN_ROOT_LIST',
    'CLASS_NAME_TO_IDX', 'IDX_TO_CLASS', 'CLASS_NAME_TO_IDX_ORIGINAL',
    'NUM_CLASSES_GLOBAL', 'NUM_EPOCHS', 'BATCH_SIZE', 'ACCUMULATION_STEPS',
    'TARGET_FRAMES_GLOBAL', 'WARMUP_RATIO', 'WEIGHT_DECAY', 'MAX_GRAD_NORM',
    'LABEL_SMOOTHING_FACTOR', 'FOCAL_LOSS_GAMMA', 'EXPERIMENT_VERSION',
    'BEST_MODEL_PATH', 'LAST_MODEL_PATH'
]

for var_name in vars_to_check_from_block01:
    if var_name not in globals():
        print(f"‼️ KRİTİK HATA: Gerekli global değişken '{var_name}' bulunamadı!")
        print("    Lütfen bir önceki 'Model Hazırlama ve Hiperparametreler (Blok 0/1)' hücresini başarıyla çalıştırdığınızdan emin olun.")
        required_vars_from_block01_exist = False
        break

if required_vars_from_block01_exist and model_ready and optimizer_ok:

    print("\n--- Gerekli Fonksiyonlar ve Sınıflar Tanımlanıyor ---")

    if 'TARGET_FRAMES_GLOBAL' not in globals(): TARGET_FRAMES_GLOBAL = 32
    if 'FLOW_STD_DIVISOR' not in globals(): FLOW_STD_DIVISOR = 1.5
    if 'FLOW_CLIP_VAL' not in globals(): FLOW_CLIP_VAL = 10.0
    # DATA_ROOT'un Path objesi olduğundan emin olalım
    if 'DATA_ROOT' in globals() and isinstance(DATA_ROOT, str):
        DATA_ROOT = Path(DATA_ROOT)


    class FocalLoss(nn.Module):
        def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
            super(FocalLoss, self).__init__(); self.alpha,self.gamma,self.reduction = alpha,gamma,reduction
        def forward(self, inputs, targets):
            ce = nn.functional.cross_entropy(inputs,targets,weight=self.alpha,reduction='none');pt=torch.exp(-ce)
            f_loss=((1-pt)**self.gamma * ce); return f_loss.mean()if self.reduction=='mean'else f_loss.sum()if self.reduction=='sum'else f_loss

    def get_video_stem(file_path:Path)->str|None:
        try:
            stem=file_path.stem; suffixes_to_remove=["_flow_u","_flow_v","_flow_g","_mask"]
            stem=re.sub(r'_win\d+$','',stem)
            for s_suffix in suffixes_to_remove:
                if stem.endswith(s_suffix): stem=stem[:-len(s_suffix)]
            return stem.split('_f')[0]
        except: return None

    def get_vg_paths(u_abs_path:Path, script_data_root:Path)->tuple[Path|None,Path|None]:
        u_fn=u_abs_path.name; v_fn=u_fn.replace("_flow_u.npy","_flow_v.npy"); g_fn=u_fn.replace("_flow_u.npy","_flow_g.npy")
        p_dir=u_abs_path.parent; return p_dir/v_fn,p_dir/g_fn

    def load_flow_pair(u_path: Path, is_train: bool, script_data_root: Path):
        try:
            v_path, g_path = get_vg_paths(u_path, script_data_root)
            if not u_path.exists(): print(f"‼️ HATA (load_flow_pair): u_path bulunamadı -> {u_path}"); return None
            if not v_path.exists(): print(f"‼️ HATA (load_flow_pair): v_path bulunamadı -> {v_path} (u: {u_path.name})"); return None
            if not (g_path and g_path.exists()): print(f"‼️ HATA (load_flow_pair): g_path bulunamadı -> {g_path} (u: {u_path.name})"); return None

            u_loaded, v_loaded, g_loaded = np.load(u_path), np.load(v_path), np.load(g_path)
            current_frames = u_loaded.shape[0]
            if not (v_loaded.shape[0] == current_frames and g_loaded.shape[0] == current_frames):
                print(f"UYARI (load_flow_pair): {u_path.name} frame sayıları tutarsız. u:{current_frames},v:{v_loaded.shape[0]},g:{g_loaded.shape[0]}. Atlanıyor.")
                return None
            if g_loaded.ndim == 2 and current_frames == 1: g_loaded = np.expand_dims(g_loaded, axis=0)
            elif g_loaded.ndim == 2 and current_frames != 1: print(f"UYARI (load_flow_pair): {u_path.name} g_loaded.ndim=2 ama T={current_frames}. Atlanıyor."); return None
            elif g_loaded.ndim != 3:
                if not (g_loaded.ndim == 2 and current_frames == 1): print(f"UYARI (load_flow_pair): {u_path.name} g_loaded.ndim={g_loaded.ndim} beklenmedik. Atlanıyor."); return None
            start_idx_base = 0
            if current_frames > TARGET_FRAMES_GLOBAL:
                max_start_offset = current_frames - TARGET_FRAMES_GLOBAL
                if is_train: start_idx_base = random.randint(0, max_start_offset)
                else: start_idx_base = max_start_offset // 2
            temporal_jitter_offset = 0
            if is_train and current_frames > TARGET_FRAMES_GLOBAL + 4:
                temporal_jitter_offset = random.randint(-2, 2)
                start_idx_base = max(0, min(start_idx_base + temporal_jitter_offset, current_frames - TARGET_FRAMES_GLOBAL))
            u_segment,v_segment,g_segment_gray = u_loaded[start_idx_base:start_idx_base+TARGET_FRAMES_GLOBAL], v_loaded[start_idx_base:start_idx_base+TARGET_FRAMES_GLOBAL], g_loaded[start_idx_base:start_idx_base+TARGET_FRAMES_GLOBAL]
            processed_frames = u_segment.shape[0]; u, v, g_gray = u_segment, v_segment, g_segment_gray
            if processed_frames < TARGET_FRAMES_GLOBAL :
                if processed_frames == 0: return None
                num_pad = TARGET_FRAMES_GLOBAL - processed_frames
                u,v,g_gray = np.pad(u, ((0,num_pad),(0,0),(0,0)), 'edge'), np.pad(v, ((0,num_pad),(0,0),(0,0)), 'edge'), np.pad(g_gray, ((0,num_pad),(0,0),(0,0)), 'edge')
            if is_train and random.random() < 0.5: u,v,g_gray,u = np.ascontiguousarray(np.flip(u,2)), np.ascontiguousarray(np.flip(v,2)), np.ascontiguousarray(np.flip(g_gray,2)), -u
            u_norm, v_norm, g_gray_norm = (u/FLOW_STD_DIVISOR).clip(-FLOW_CLIP_VAL,FLOW_CLIP_VAL), (v/FLOW_STD_DIVISOR).clip(-FLOW_CLIP_VAL,FLOW_CLIP_VAL), g_gray
            t_stacked = np.stack([u_norm,v_norm,g_gray_norm],0).astype(np.float32)
            for i in range(t_stacked.shape[0]):
                channel_data_loop = t_stacked[i, ...]
                mean_val_loop = np.mean(channel_data_loop)
                std_dev_loop = np.std(channel_data_loop)
                t_stacked[i, ...] = (channel_data_loop - mean_val_loop) / (std_dev_loop + 1e-6)
            return torch.from_numpy(t_stacked)
        except FileNotFoundError as e_fnf: print(f"‼️ HATA (load_flow_pair - FileNotFoundError): {u_path.name} için dosya bulunamadı: {e_fnf}"); return None
        except Exception as e_load: print(f"‼️ HATA (load_flow_pair - Genel Hata): {u_path.name} işlenirken sorun: {e_load}\n{traceback.format_exc(limit=1)}"); return None

    class FlowDataset(Dataset):
        def __init__(self, roots_to_scan_physical_files, split_file_path_str, class_name_to_idx_map_binary, class_name_to_idx_map_original, is_train: bool):
            self.is_train = is_train
            self.target_physical_roots = [Path(r) for r in roots_to_scan_physical_files if Path(r).is_dir()]
            if not self.target_physical_roots: raise FileNotFoundError(f"Tarama kök dizini: {roots_to_scan_physical_files}")
            self.data_root_path_obj = DATA_ROOT # DATA_ROOT global değişkenini kullan
            self.path_from_split_to_label_idx = {}
            self.samples_per_class = collections.defaultdict(int)
            self.final_flow_u_paths_to_load = [] # Bunlar mutlak (absolute) Path nesneleri olacak
            self.final_relative_paths_for_json = [] # Bunlar DATA_ROOT'a göreceli string yollar olacak
            self.final_labels_for_paths = []

            current_split_file = Path(split_file_path_str)
            if not current_split_file.exists(): raise FileNotFoundError(f"Split dosyası: {current_split_file}")
            print(f"Etiketler (Binary) '{current_split_file.name}' dosyasından hazırlanıyor (is_train={self.is_train})...")
            lines, loaded_paths, unknown_orig_cls_log, no_stem_log = 0,0,set(),set()
            with open(current_split_file, 'r') as f:
                for line_from_file in f:
                    line_from_file = line_from_file.strip(); lines +=1
                    if not line_from_file: continue
                    # Split dosyasındaki yolun DATA_ROOT'a göreceli olduğu varsayılıyor
                    relative_path_from_split_str = line_from_file
                    current_path_object_from_split = Path(relative_path_from_split_str) # Bu göreceli bir Path

                    video_stem_from_path = get_video_stem(current_path_object_from_split)
                    if not video_stem_from_path: no_stem_log.add(relative_path_from_split_str); continue

                    original_class_name = "Unknown"; path_parts = current_path_object_from_split.parts
                    if path_parts:
                        is_normal_original_path = any(nk.lower() in p.lower() for p in path_parts for nk in ["normal","training-normal","testing-normal"])
                        if is_normal_original_path or video_stem_from_path.lower().startswith("normal_"): original_class_name = "Normal"
                        else:
                            found_original_class = False
                            if len(path_parts)>1 and path_parts[-2] in class_name_to_idx_map_original and path_parts[-2]!="Normal":
                                original_class_name, found_original_class = path_parts[-2], True
                            if not found_original_class:
                                for prefix_class_original in class_name_to_idx_map_original:
                                    if prefix_class_original!="Normal" and video_stem_from_path.lower().startswith(prefix_class_original.lower()):
                                        original_class_name, found_original_class = prefix_class_original, True; break
                    binary_label_idx = -1
                    if original_class_name == "Normal": binary_label_idx = class_name_to_idx_map_binary.get("Normal", -1)
                    elif original_class_name!="Unknown" and original_class_name in class_name_to_idx_map_original: binary_label_idx = class_name_to_idx_map_binary.get("Anomaly", -1)
                    else: unknown_orig_cls_log.add(f"'{original_class_name}' ({relative_path_from_split_str})"); continue

                    if binary_label_idx != -1:
                        # self.path_from_split_to_label_idx'ı göreceli yolla dolduruyoruz
                        self.path_from_split_to_label_idx[relative_path_from_split_str] = binary_label_idx
                        loaded_paths+=1

            print(f"-> Split'ten {loaded_paths} binary etiketli yol {lines} satırdan yüklendi.")
            if unknown_orig_cls_log: print(f"  UYARI:Orijinal sınıfı belirlenemeyen:{len(unknown_orig_cls_log)}.Örnek:{list(unknown_orig_cls_log)[:1]}")
            if no_stem_log: print(f"  UYARI:Kök adı çıkarılamayan:{len(no_stem_log)}.")
            if loaded_paths == 0: raise ValueError("Split dosyasından hiç etiketli yol yüklenemedi!")

            print(f"Fiziksel dosyalar ({[r.name for r in self.target_physical_roots]}) altında taranıyor...")
            all_u_abs_paths = [f for rp in self.target_physical_roots for f in rp.glob("**/*_flow_u.npy")]
            print(f"Toplam {len(all_u_abs_paths)} potansiyel fiziksel '*_flow_u.npy' bulundu.")

            skip_not_in_split, skip_comp_missing, skip_rel_path_error = 0,0,0
            for u_abs_path in tqdm(all_u_abs_paths, desc=f"Fiziksel dosyalar filtreleniyor (is_train={self.is_train})"):
                try:
                    # Mutlak yoldan DATA_ROOT'a göre göreceli yolu elde et
                    u_relative_path_str = str(PurePosixPath(u_abs_path.relative_to(self.data_root_path_obj)))
                except ValueError:
                    skip_rel_path_error += 1
                    continue # DATA_ROOT altında değilse veya başka bir sorun varsa atla

                if u_relative_path_str in self.path_from_split_to_label_idx:
                    v_c_path, g_c_path = get_vg_paths(u_abs_path, self.data_root_path_obj)
                    if v_c_path.exists() and (g_c_path and g_c_path.exists()):
                        lbl_bin = self.path_from_split_to_label_idx[u_relative_path_str]
                        self.final_flow_u_paths_to_load.append(u_abs_path) # Mutlak yolu sakla
                        self.final_relative_paths_for_json.append(u_relative_path_str) # JSON için göreceli yolu sakla
                        self.final_labels_for_paths.append(lbl_bin)
                        self.samples_per_class[lbl_bin] +=1
                    else: skip_comp_missing +=1
                else: skip_not_in_split +=1

            print(f"Dataset (is_train={self.is_train}, Binary): {len(self.final_flow_u_paths_to_load)} geçerli dosya.")
            print(f"  Atlanan (split'te olmayan): {skip_not_in_split}")
            print(f"  Atlanan (DATA_ROOT dışı veya göreceli yol hatası): {skip_rel_path_error}")
            print(f"  Atlanan (v/g_grayscale eksik): {skip_comp_missing}")

            if not self.final_flow_u_paths_to_load: print(f"UYARI: Dataset boş (is_train={self.is_train})!")
            if 'IDX_TO_CLASS' in globals() and 'CLASS_NAME_TO_IDX' in globals() and len(CLASS_NAME_TO_IDX)==NUM_CLASSES_GLOBAL:
                for i in range(NUM_CLASSES_GLOBAL): print(f"  Sınıf {i} ({globals()['IDX_TO_CLASS'].get(i,'?')}): {self.samples_per_class.get(i,0)} örnek")

        def __len__(self): return len(self.final_flow_u_paths_to_load)

        def __getitem__(self, idx):
            # load_flow_pair mutlak yol ile çalışır
            absolute_u_path = self.final_flow_u_paths_to_load[idx]
            tensor = load_flow_pair(absolute_u_path, self.is_train, self.data_root_path_obj)

            if tensor is not None:
                label = torch.tensor(self.final_labels_for_paths[idx], dtype=torch.long)
                # JSON için kullanılacak göreceli yolu döndür
                relative_clip_path_str = self.final_relative_paths_for_json[idx]
                return tensor, label, relative_clip_path_str
            else:
                # Eğer load_flow_pair None dönerse, bu örnek atlanacak collate_fn tarafından
                return None

        def get_sampler_weights(self):
            if not self.final_labels_for_paths: return torch.DoubleTensor([])
            counts=np.array([self.samples_per_class.get(i,0)for i in range(NUM_CLASSES_GLOBAL)],dtype=float)
            ifw=1.0/np.maximum(counts,1e-6);sw=np.array([ifw[lbl]for lbl in self.final_labels_for_paths],dtype=float)
            return torch.DoubleTensor(sw if np.sum(sw)>0 else[1.0]*len(self.final_labels_for_paths)if self.final_labels_for_paths else[])

    def collate_fn_skip_none(batch):
        # Önce None olan item'ları filtrele (load_flow_pair veya __getitem__ None döndürdüyse)
        valid_batch_items = [item for item in batch if item is not None]
        if not valid_batch_items: return None # Eğer tüm batch None ise
        # Sonra tensörü None olanları filtrele (load_flow_pair None döndürdüyse ama __getitem__ None döndürmediyse - olası değil)
        # valid_batch_items = [item for item in valid_batch_items if item[0] is not None] # item[0] tensör
        # if not valid_batch_items: return None

        try:
            # valid_batch_items şimdi [(tensor1, label1, path1), (tensor2, label2, path2), ...] şeklinde
            # default_collate tensörleri, etiketleri ve yolları (string listesi) ayrı ayrı batch'ler.
            return torch.utils.data.dataloader.default_collate(valid_batch_items)
        except RuntimeError as e:
            print(f"HATA (collate_fn): Bir batch toplanamadı: {e}")
            # Sorunlu batch'i atlamak için None döndür
            return None
        except Exception as e_collate_general:
            print(f"BEKLENMEDİK HATA (collate_fn): {e_collate_general}\n{traceback.format_exc(limit=1)}")
            return None


    print("--- Gerekli Fonksiyonlar ve Sınıflar (Binary için ayarlandı) Tanımlandı ---")

    # ----- DataLoader'ların Hazırlanması -----
    dataloaders_ok=False;train_loader,val_loader,val_ds=None,None,None # val_ds'i global yapalım
    train_ds_len_sched,train_cls_counts_loss=0,None
    try:
        print("\nEğitim DataLoader'ı(B)hazırlanıyor...");
        train_ds=FlowDataset(TRAIN_FLOW_ROOT_LIST,str(TRAIN_SPLIT_FILE),CLASS_NAME_TO_IDX,CLASS_NAME_TO_IDX_ORIGINAL,True)
        if not train_ds or len(train_ds)==0:raise ValueError("Eğitim veri seti(train_ds)boş/oluşturulamadı!")
        train_sw=train_ds.get_sampler_weights();train_samp,train_shuf=None,True
        if torch.is_tensor(train_sw)and train_sw.numel()>0 and torch.sum(train_sw).item()>0:
            train_samp,train_shuf=WeightedRandomSampler(train_sw,len(train_ds),True),False
            print(f"  ->Eğitim DataLoader'ı(B):{len(train_ds)}örnek,WeightedRandomSampler aktif.")
        else:print(f"  ->Eğitim DataLoader'ı(B):{len(train_ds)}örnek,shuffle={train_shuf}aktif.")
        train_loader=DataLoader(train_ds,BATCH_SIZE,sampler=train_samp,shuffle=train_shuf,num_workers=2,pin_memory=torch.cuda.is_available(),collate_fn=collate_fn_skip_none,persistent_workers=(True if 2>0 else False),drop_last=True)
        train_ds_len_sched=len(train_loader);print(f"  ->Train loader efektif batch sayısı(B):{train_ds_len_sched}");train_cls_counts_loss=train_ds.samples_per_class

        print("\nDoğrulama DataLoader'ı(B)hazırlanıyor...");
        val_ds=FlowDataset(VAL_SCAN_ROOT_LIST,str(TEST_SPLIT_FILE),CLASS_NAME_TO_IDX,CLASS_NAME_TO_IDX_ORIGINAL,False) # val_ds'i burada tanımla
        if not val_ds or len(val_ds)==0:raise ValueError("Doğrulama veri seti(val_ds)boş/oluşturulamadı!")
        val_loader=DataLoader(val_ds,BATCH_SIZE*2,shuffle=False,num_workers=2,pin_memory=torch.cuda.is_available(),collate_fn=collate_fn_skip_none,persistent_workers=(True if 2>0 else False), drop_last=False) # drop_last=False önemli
        print(f"  ->Doğrulama DataLoader'ı(B):{len(val_ds)}örnek,{len(val_loader)}batch.");dataloaders_ok=True
    except Exception as e:print(f"‼️HATA:DataLoader(B)oluşturma başarısız:{e}\n{traceback.format_exc()}");dataloaders_ok=False

    criterion=None
    if dataloaders_ok:
        alpha_bin=None
        if train_cls_counts_loss and FOCAL_LOSS_GAMMA > 0:
            cts_bin=np.array([train_cls_counts_loss.get(i,0)for i in range(NUM_CLASSES_GLOBAL)],dtype=float);tot_bin=np.sum(cts_bin)
            if tot_bin>0 and NUM_CLASSES_GLOBAL>0:w_bin=tot_bin/(NUM_CLASSES_GLOBAL*np.maximum(cts_bin,1e-6));alpha_bin=torch.tensor(w_bin,dtype=torch.float32).to(DEVICE);print(f"\nBinary Focal Loss alpha ağırlıkları:{[f'{w:.2f}'for w in w_bin]}")
        cg=FOCAL_LOSS_GAMMA
        if 'FOCAL_LOSS_GAMMA_BINARY' in globals(): cg = globals()['FOCAL_LOSS_GAMMA_BINARY']
        if FOCAL_LOSS_GAMMA > 0: criterion=FocalLoss(alpha=alpha_bin,gamma=cg);print(f"Kayıp Fonksiyonu:Focal Loss(B)(gamma={cg},alpha={'var'if alpha_bin is not None else'yok'})")
        else:criterion=nn.CrossEntropyLoss(weight=alpha_bin,label_smoothing=LABEL_SMOOTHING_FACTOR);print(f"Kayıp Fonksiyonu:CrossEntropyLoss(B)(ağırlık={'var'if alpha_bin is not None else'yok'},ls={LABEL_SMOOTHING_FACTOR})")
    else:print("‼️Criterion(B)tanımlanamadı.")

    scheduler=None;scheduler_ok=False
    if dataloaders_ok and criterion and train_loader and train_ds_len_sched>0 and optimizer:
        try:
            if ACCUMULATION_STEPS<=0:ACCUMULATION_STEPS=1
            spe=train_ds_len_sched;ts=spe*NUM_EPOCHS;ws=int(ts*WARMUP_RATIO)
            if ts<=0:raise ValueError(f"total_steps({ts})pozitif olmalı.")
            scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=ws,num_training_steps=ts)
            print(f"Linear scheduler(B):Total steps={ts},Warmup steps={ws}");scheduler_ok=True
        except Exception as e:print(f"‼️HATA:Scheduler(B)oluşturma:{e}\n{traceback.format_exc()}");scheduler_ok=False
    else:print("‼️Scheduler(B)tanımlanamadı.")

    USING_HALF_TRAINING=False;scaler_training=None
    if USING_HALF_TRAINING and DEVICE.type=='cuda':scaler_training=torch.cuda.amp.GradScaler(enabled=True)

    def run_epoch(model, loader, optimizer_epoch, criterion_epoch, device_epoch, scheduler_epoch=None, is_train_phase=True, current_epoch_num=0, grad_accum_steps=1, max_norm_grad=1.0, use_half_precision=False, grad_scaler=None):
        model.train(is_train_phase)
        total_loss_epoch, samples_proc_epoch = 0.0, 0
        all_preds_epoch, all_golds_epoch, all_clip_paths_epoch, all_logits_epoch = [], [], [], [] # GÜNCELLENDİ

        desc = f"Epoch {current_epoch_num} {'Eğitim(B)' if is_train_phase else 'Doğrulama(B)'}({EXPERIMENT_VERSION})"
        autocast_ctx = torch.amp.autocast(device_type=device_epoch.type, dtype=torch.float16, enabled=(use_half_precision and device_epoch.type == 'cuda'))
        loop = tqdm(loader, desc=desc, leave=False)

        if is_train_phase and grad_accum_steps > 1: optimizer_epoch.zero_grad(set_to_none=True)

        for batch_idx, batch_data in enumerate(loop):
            if batch_data is None: continue # collate_fn None döndürdüyse

            # GÜNCELLENDİ: batch_data artık (tensor_batch, label_batch, clip_paths_list_batch) şeklinde
            x, y, clip_paths_batch = batch_data

            if x is None or y is None or x.size(0) == 0: continue # İçerik kontrolü
            if batch_idx == 0 and current_epoch_num == 1 and is_train_phase: print(f"\nDEBUG:İlk train batch x_batch min:{x.min():.4f},max:{x.max():.4f},mean:{x.mean():.4f},std:{x.std():.4f}"); print(f"DEBUG:İlk train batch y_batch:{y[:5]}")

            bs = x.size(0); x, y = x.to(device_epoch, non_blocking=True), y.to(device_epoch, non_blocking=True)
            exp_ch = 3
            if x.dim() != 5 or x.shape[1] != exp_ch or x.shape[2] != TARGET_FRAMES_GLOBAL: tqdm.write(f"UYARI(Batch{batch_idx}):Giriş şekli {x.shape}.Atlanıyor."); continue

            x_p = x.permute(0, 2, 1, 3, 4)
            try:
                if is_train_phase:
                    if grad_accum_steps == 1: optimizer_epoch.zero_grad(set_to_none=True)
                    with autocast_ctx:
                        out = model(x_p); logits = out.logits if hasattr(out, 'logits') else out; loss = criterion_epoch(logits, y)
                        if batch_idx == 0 and current_epoch_num == 1: print(f"DEBUG(Train)İlk batch logits(ilk örnek):{logits[0].detach().cpu().numpy()}"); print(f"DEBUG(Train)İlk batch y(ilk örnek):{y[0].cpu().numpy()}"); print(f"DEBUG(Train)İlk batch loss:{loss.item():.4f}")
                        if grad_accum_steps > 1: loss = loss / grad_accum_steps
                    if torch.isnan(loss) or torch.isinf(loss): tqdm.write(f"NaN/Inf loss({loss.item()})!Atlanıyor."); continue
                    if use_half_precision and grad_scaler: grad_scaler.scale(loss).backward()
                    else: loss.backward()
                    if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(loader):
                        if use_half_precision and grad_scaler: grad_scaler.unscale_(optimizer_epoch)
                        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad and p.grad is not None, model.parameters()), max_norm_grad)
                        if use_half_precision and grad_scaler: grad_scaler.step(optimizer_epoch); grad_scaler.update()
                        else: optimizer_epoch.step()
                        if scheduler_epoch:
                            scheduler_epoch.step()
                            if current_epoch_num == 1 and (batch_idx + 1) / grad_accum_steps <= 5 and optimizer_epoch.param_groups: lr_chk = optimizer_epoch.param_groups[0]['lr']; print(f"DEBUG:Optimizer adımı(batch_idx {batch_idx}, optim_step ~{(batch_idx + 1) // grad_accum_steps}),LR:{lr_chk:.2e}")
                        optimizer_epoch.zero_grad(set_to_none=True)
                    curr_loss_log = loss.item() * (grad_accum_steps if grad_accum_steps > 1 and loss is not None else 1.0)
                else: # Doğrulama fazı
                    with torch.inference_mode(), autocast_ctx:
                        out = model(x_p); logits = out.logits if hasattr(out, 'logits') else out
                        if criterion_epoch: curr_loss_log = criterion_epoch(logits, y).item()
                        else: curr_loss_log = 0.0
                        if batch_idx == 0 and current_epoch_num == 1: print(f"DEBUG(Val)İlk batch logits(ilk örnek):{logits[0].detach().cpu().numpy()}"); print(f"DEBUG(Val)İlk batch y(ilk örnek):{y[0].cpu().numpy()}"); print(f"DEBUG(Val)İlk batch loss:{curr_loss_log:.4f}")

                total_loss_epoch += curr_loss_log * bs; samples_proc_epoch += bs

                # GÜNCELLENDİ: Logitleri ve yolları da sakla
                with torch.no_grad():
                    preds_batch = torch.argmax(logits, 1).cpu().tolist()
                    golds_batch = y.cpu().tolist()
                    logits_batch_cpu = logits.detach().cpu().tolist() # Ham logitler

                    all_preds_epoch.extend(preds_batch)
                    all_golds_epoch.extend(golds_batch)
                    all_logits_epoch.extend(logits_batch_cpu)
                    all_clip_paths_epoch.extend(clip_paths_batch) # clip_paths_batch zaten string listesi

                lr_log = "";
                if is_train_phase and optimizer_epoch and hasattr(optimizer_epoch, 'param_groups') and optimizer_epoch.param_groups:
                    for i, grp in enumerate(optimizer_epoch.param_groups): lr_log += f"{grp.get('name', f'g{i}')}_LR:{grp['lr']:.1e} "
                loop.set_postfix(loss=f"{curr_loss_log:.4f}", lr=lr_log.strip())
            except RuntimeError as e:
                if "CUDA out of memory" in str(e).lower(): tqdm.write(f"\n!!KRİTİK(Batch{batch_idx}):CUDA Bellek Doldu!{e}"); raise e
                else: tqdm.write(f"\n!!HATA(Batch{batch_idx}):{e}"); traceback.print_exc(); continue
            except Exception as e: tqdm.write(f"\n!!BEKLENMEDİK HATA(Batch{batch_idx}):{e}"); traceback.print_exc(); continue

        if samples_proc_epoch == 0:
            print(f"UYARI: Epoch {current_epoch_num} {'Eğitim(B)' if is_train_phase else 'Doğrulama(B)'} fazında hiç örnek işlenemedi.")
            return 0.0, 0.0, 0.0, 0.0, 0.0, [], [], [], [] # GÜNCELLENDİ

        avg_loss = total_loss_epoch / samples_proc_epoch
        if not all_golds_epoch or not all_preds_epoch:
             return avg_loss, 0.0, 0.0, 0.0, 0.0, all_golds_epoch, all_preds_epoch, all_clip_paths_epoch, all_logits_epoch # GÜNCELLENDİ

        try:
            acc = accuracy_score(all_golds_epoch, all_preds_epoch)
            prec, rec, f1, _ = precision_recall_fscore_support(all_golds_epoch, all_preds_epoch, average="macro", zero_division=0, labels=list(range(NUM_CLASSES_GLOBAL)))
        except ValueError: # Eğer all_preds_epoch boşsa veya tek bir sınıf içeriyorsa
            acc = accuracy_score(all_golds_epoch, all_preds_epoch) if all_golds_epoch and all_preds_epoch else 0.0
            prec, rec, f1 = 0.0, 0.0, 0.0
            # GÜNCELLENDİ
            return avg_loss, acc, prec, rec, f1, all_golds_epoch, all_preds_epoch, all_clip_paths_epoch, all_logits_epoch

        # GÜNCELLENDİ
        return avg_loss, acc, prec, rec, f1, all_golds_epoch, all_preds_epoch, all_clip_paths_epoch, all_logits_epoch

    # ----- Ana Eğitim Döngüsü (Binary) -----
    if dataloaders_ok and criterion and scheduler_ok and hf_model and optimizer :
        print(f"\n---Eğitim Başlatılıyor(BINARY:{EXPERIMENT_VERSION})---");best_val_f1=-1.0;epochs_no_imp=0
        PATIENCE_BINARY=5;total_train_time_start=time.time();last_epoch_completed_successfully=False
        for ep_num in range(NUM_EPOCHS):
            ep_time_start=time.time();last_epoch_completed_successfully=False
            print(f"\n=====Epoch {ep_num+1}/{NUM_EPOCHS}(Binary)=====")
            try:
                # Eğitim fazı için dönen clip_paths ve logits kullanılmayacak
                tr_loss,tr_acc,tr_prec,tr_rec,tr_f1,_,_,_,_ = run_epoch(hf_model,train_loader,optimizer,criterion,DEVICE,scheduler,True,ep_num+1,ACCUMULATION_STEPS,MAX_GRAD_NORM,USING_HALF_TRAINING,scaler_training)

                # Doğrulama fazı için tüm dönen değerler alınacak
                val_loss,val_acc,val_prec,val_rec,val_f1,val_golds,val_preds,val_clip_paths,val_logits = run_epoch(hf_model,val_loader,None,criterion,DEVICE,None,False,ep_num+1,use_half_precision=USING_HALF_TRAINING)

                last_epoch_completed_successfully=True;lr_log_print=""
                if optimizer and hasattr(optimizer,'param_groups') and optimizer.param_groups:
                    for i,grp in enumerate(optimizer.param_groups):lr_log_print+=f"{grp.get('name',f'g{i}')}_LR:{grp['lr']:.1e} "
                print(f"Epoch {ep_num+1}/{NUM_EPOCHS}[{time.time()-ep_time_start:.1f}s]{lr_log_print.strip()}")
                print(f"  Eğitim(B)   ->Kayıp:{tr_loss:.4f} Acc:{tr_acc:.3f} P(M):{tr_prec:.3f} R(M):{tr_rec:.3f} F1(M):{tr_f1:.3f}")
                print(f"  Doğrulama(B)->Kayıp:{val_loss:.4f} Acc:{val_acc:.3f} P(M):{val_prec:.3f} R(M):{val_rec:.3f} F1(M):{val_f1:.3f}")

                if val_golds and val_preds: # val_clip_paths ve val_logits de dolu olmalı bu durumda
                    target_n_binary=[IDX_TO_CLASS.get(i,str(i))for i in range(NUM_CLASSES_GLOBAL)]
                    print(f"  Sınıflandırma Raporu(Doğrulama-Binary-Epoch {ep_num+1}):")
                    try:print(classification_report(val_golds,val_preds,target_names=target_n_binary,digits=3,zero_division=0,labels=list(range(NUM_CLASSES_GLOBAL))))
                    except Exception as r_err:print(f"  Epoch {ep_num+1} Rapor hatası:{r_err}")

                if val_f1 > best_val_f1:
                    print(f"  ✓Val F1(M)(Binary)iyileşti({best_val_f1:.4f} -> {val_f1:.4f}).Model:{BEST_MODEL_PATH.name}");best_val_f1=val_f1;epochs_no_imp=0
                    try:torch.save(hf_model.state_dict(),BEST_MODEL_PATH)
                    except Exception as s_exc:print(f"  ‼️HATA:En iyi binary model kaydı:{s_exc}")
                else:epochs_no_imp+=1;print(f"  Val F1(M)(Binary)iyileşmedi({best_val_f1:.4f}sabit).Sabır:{epochs_no_imp}/{PATIENCE_BINARY}")

                if epochs_no_imp>=PATIENCE_BINARY and NUM_EPOCHS>PATIENCE_BINARY:print(f"\n{PATIENCE_BINARY}epoch iyileşme yok.Erken durduruluyor(Binary)...");break
            except RuntimeError as e_rt_ep:
                if"CUDA out of memory"in str(e_rt_ep).lower():print(f"EPOCH {ep_num+1}KRİTİK(Binary):CUDA Bellek Doldu.");print(traceback.format_exc());break
                else:print(f"EPOCH {ep_num+1}Runtime Hatası(Binary):{e_rt_ep}");print(traceback.format_exc());break
            except Exception as e_gen_ep:print(f"EPOCH {ep_num+1}beklenmedik hata(Binary):{e_gen_ep}");print(traceback.format_exc());break

        completed_eps_binary=(ep_num+1)if'ep_num'in locals()and last_epoch_completed_successfully else(locals().get('ep_num',0))
        if completed_eps_binary > 0 :
            print(f"\nSon({completed_eps_binary}.epoch)binary model kaydediliyor:{LAST_MODEL_PATH.name}")
            try:torch.save(hf_model.state_dict(),LAST_MODEL_PATH);print(f"  ✓Son binary model kaydedildi.")
            except Exception as s_exc:print(f"  ‼️HATA:Son binary model kaydı:{s_exc}")
        else:print("\nHiç binary epoch tamamlanamadı,son model kaydedilmedi.")

        print(f"\n---Binary Eğitim Bitti[Toplam Süre:{(time.time()-total_train_time_start)/60:.1f}dk]---")
        print(f"▸Deney(Binary):{EXPERIMENT_VERSION},Tamamlanan Epoch:{completed_eps_binary},En İyi Val F1(M):{best_val_f1:.4f}")

        # ----- JSON ÇIKTISI OLUŞTURMA -----
        if BEST_MODEL_PATH.exists():
            print(f"\n---En İyi Binary Model({BEST_MODEL_PATH.name})ile Son Değerlendirme ve JSON Çıktısı---")
            try:
                hf_model.load_state_dict(torch.load(BEST_MODEL_PATH,map_location=DEVICE));print("  ->En iyi binary model ağırlıkları yüklendi.")
                # Son değerlendirme için run_epoch çağrısı
                fin_loss, fin_acc, fin_prec, fin_rec, fin_f1, fin_golds, fin_preds, fin_clip_paths, fin_logits = run_epoch(
                    hf_model, val_loader, None, criterion, DEVICE, None, False, completed_eps_binary, use_half_precision=USING_HALF_TRAINING
                )
                print(f"\nSonuçlar(En İyi Binary Model-Doğrulama Seti):")
                print(f"  Kayıp:{fin_loss:.4f} Acc:{fin_acc:.3f} P(M):{fin_prec:.3f} R(M):{fin_rec:.3f} F1(M):{fin_f1:.3f}")

                if fin_golds and fin_preds and fin_clip_paths and fin_logits: # Tüm listelerin dolu olduğunu kontrol et
                    target_n_fin_bin=[IDX_TO_CLASS.get(i,str(i))for i in range(NUM_CLASSES_GLOBAL)]
                    print("\nSınıflandırma Raporu(En İyi Binary Model-Val Seti):");print(classification_report(fin_golds,fin_preds,target_names=target_n_fin_bin,digits=3,zero_division=0,labels=list(range(NUM_CLASSES_GLOBAL))))
                    print("\nKarmaşıklık Matrisi(En İyi Binary Model-Val Seti):");print(confusion_matrix(fin_golds,fin_preds,labels=list(range(NUM_CLASSES_GLOBAL))))
                    f1_per_class_bin=precision_recall_fscore_support(fin_golds,fin_preds,labels=list(range(NUM_CLASSES_GLOBAL)),average=None,zero_division=0)[2]
                    print("\nHer Sınıf İçin F1 Skorları(En İyi Binary Model-Val Seti):")
                    for i,f1_s in enumerate(f1_per_class_bin):print(f"  {target_n_fin_bin[i]if i<len(target_n_fin_bin)else f'Sınıf{i}':<20}:{f1_s:.3f}")

                    # JSON dosyasına kaydetme bölümü
                    print(f"\nDoğrulama seti tahminleri JSON dosyasına kaydediliyor...")
                    output_json_path = Path("/content/drive/MyDrive/Crime/outputs/binary_val_preds.json")
                    output_json_path.parent.mkdir(parents=True, exist_ok=True)

                    results_data_for_json = []
                    # 'Anomaly' sınıfının index'ini CLASS_NAME_TO_IDX'ten al
                    # Yaygın kullanım: {"Normal": 0, "Anomaly": 1} -> anomaly_idx = 1
                    # Eğer {"Anomaly":0, "Normal":1} ise -> anomaly_idx = 0
                    # Bu değeri kendi CLASS_NAME_TO_IDX haritanıza göre doğrulayın.
                    anomaly_class_name = "Anomaly" # Aradığımız sınıf adı
                    anomaly_idx = CLASS_NAME_TO_IDX.get(anomaly_class_name, None)

                    if anomaly_idx is None:
                        print(f"‼️ UYARI: '{anomaly_class_name}' sınıfı CLASS_NAME_TO_IDX içinde bulunamadı. Skorlar doğru olmayabilir.")
                        # Varsayılan olarak son indeksi veya 1'i kullanabilirsiniz, ancak bu doğru olmayabilir.
                        # Şimdilik skorları None olarak bırakalım veya logitleri direkt yazalım.
                        anomaly_idx = 1 # Geçici bir varsayım, kullanıcı kontrol etmeli

                    for path, true_label, pred_label, logits_one_sample in zip(fin_clip_paths, fin_golds, fin_preds, fin_logits):
                        # Logitleri softmax'e çevirip skor alabiliriz
                        softmax_scores_tensor = torch.softmax(torch.tensor(logits_one_sample), dim=0)
                        softmax_scores_list = softmax_scores_tensor.tolist()

                        # Anomali skorunu al (eğer anomaly_idx geçerliyse)
                        anomaly_score = softmax_scores_list[anomaly_idx] if anomaly_idx < len(softmax_scores_list) else None

                        results_data_for_json.append({
                            "clip_path": path,        # FlowDataset'ten gelen göreceli yol
                            "label_true": true_label,
                            "label_pred": pred_label,
                            "score_anomaly": anomaly_score, # Sadece anomali sınıfının softmax skoru
                            "scores_all_classes": softmax_scores_list # Tüm sınıfların softmax skorları
                        })

                    results_df_to_json = pd.DataFrame(results_data_for_json)
                    try:
                        results_df_to_json.to_json(output_json_path, orient="records", indent=4, force_ascii=False)
                        print(f"  ✓ Doğrulama tahminleri başarıyla kaydedildi: {output_json_path}")
                    except Exception as e_json:
                        print(f"  ‼️ HATA: JSON dosyasına kaydetme sırasında sorun oluştu: {e_json}")
                        print(traceback.format_exc())
                else:
                    print("UYARI: JSON oluşturmak için yeterli veri yok (fin_golds, fin_preds, fin_clip_paths veya fin_logits eksik).")

            except Exception as fin_rep_exc:print(f"‼️HATA:Final binary raporu veya JSON oluşturma:{fin_rep_exc}\n{traceback.format_exc()}")
        else:print(f"\nUYARI:En iyi binary model({BEST_MODEL_PATH.name})bulunamadı,final raporu ve JSON çıktısı yok.")
    else:print("\n‼️Gerekli bileşenler hazır olmadığı için binary eğitim başlatılamadı.")
else:
    print(f"\n‼️ Ana Eğitim Bloğu çalıştırılamadı: Gerekli temel değişkenler (Blok 0/1'den) eksik "
          f"(required_vars_from_block01_exist={globals().get('required_vars_from_block01_exist', 'Tanımsız')}) "
          f"veya model hazır değil (model_ready={globals().get('model_ready', 'Tanımsız')}) "
          f"veya optimizer hazır değil (optimizer_ok={globals().get('optimizer_ok', 'Tanımsız')}).")