import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np

# استيراد الملفات الخاصة بك
from dataset import MultiTaskDataset
from model_factory import MultiTaskModelFactory, TASK_CONFIGURATIONS
from utils import multi_task_collate_fn, evaluate, set_seed

# الإعدادات (يجب أن تطابق إعدادات التدريب)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'best_model.pth'
DATA_ROOT_PATH = '/root/baseline/train'
RANDOM_SEED = 42
BATCH_SIZE = 8

def run_post_training_validation():
    set_seed(RANDOM_SEED)
    print(f"--- Running Validation using Device: {DEVICE} ---")

    # 1. إعداد البيانات (نفس تقسيم التدريب)
    full_dataset = MultiTaskDataset(data_root=DATA_ROOT_PATH)
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * 0.2)
    train_size = dataset_size - val_size
    
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    _, val_indices = torch.utils.data.random_split(range(dataset_size), [train_size, val_size], generator=generator)
    
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)
    val_loader = DataLoader(
        val_subset, 
        batch_size=BATCH_SIZE,
        shuffle=False, 
        num_workers=4, 
        collate_fn=multi_task_collate_fn
    )

    # 2. تحميل الموديل بالأوزان النهائية
    model = MultiTaskModelFactory(encoder_name='efficientnet-b4', task_configs=TASK_CONFIGURATIONS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

    # 3. تشغيل وظيفة التقييم (الموجودة في utils.py لديك)
    print("\n--- Generating Metrics Report ---")
    val_results_df = evaluate(model, val_loader, DEVICE)

    # 4. عرض النتائج النهائية
    if not val_results_df.empty:
        print("\n" + "="*60)
        print("         FINAL VALIDATION PERFORMANCE REPORT")
        print("="*60)
        print(val_results_df.to_string(index=False))
        
        # حساب متوسط الأداء العام
        score_cols = [col for col in val_results_df.columns if 'MAE' not in col and isinstance(val_results_df[col].iloc[0], (int, float))]
        if score_cols:
            final_avg = val_results_df[score_cols].mean().mean()
            print(f"\n>>> OVERALL VALIDATION SCORE: {final_avg:.4f}")
        print("="*60)
    else:
        print("❌ Error: No validation results generated.")

if __name__ == "__main__":
    run_post_training_validation()