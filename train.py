import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch.losses as smp_losses
import numpy as np
import random
import os

# Import local modules
from dataset import MultiTaskDataset, MultiTaskUniformSampler
from model_factory import MultiTaskModelFactory, TASK_CONFIGURATIONS
from utils import (
    multi_task_collate_fn, 
    evaluate, 
    DetectionLoss, 
    set_seed
)

# Training configuration
LEARNING_RATE = 1e-4
BATCH_SIZE = 20
NUM_EPOCHS = 50 
DATA_ROOT_PATH = 'data/train'
ENCODER = 'efficientnet-b4'
ENCODER_WEIGHTS = 'imagenet'
RANDOM_SEED = 42
MODEL_SAVE_PATH = 'best_model.pth' 
VAL_SPLIT = 0.2

# --- MIXUP HELPER FUNCTIONS (PHASE 3) ---
def mixup_data(x, y, alpha=0.4, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def main():
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    # --- AUGMENTATION PIPELINE (PHASE 1 - Final) ---
    train_transforms = A.Compose([
        A.Resize(256, 256),
        
        # 1. Physics Simulation
        A.OneOf([
            A.GaussNoise(p=0.5), 
            A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        ], p=0.5),

        # 2. Geometric Deformations
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        ], p=0.8),

        # 3. Regularization
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            fill_value=0, 
            p=0.3
        ),

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1, check_each_transform=False))
    
    # Validation transforms (No Augmentation)
    val_transforms = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))

    # Create datasets
    temp_dataset = MultiTaskDataset(data_root=DATA_ROOT_PATH, transforms=train_transforms)
    dataset_size = len(temp_dataset)
    val_size = int(dataset_size * VAL_SPLIT)
    train_size = dataset_size - val_size
    
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    indices = list(range(dataset_size))
    train_indices, val_indices = torch.utils.data.random_split(indices, [train_size, val_size], generator=generator)
    
    train_dataset = MultiTaskDataset(data_root=DATA_ROOT_PATH, transforms=train_transforms)
    val_dataset = MultiTaskDataset(data_root=DATA_ROOT_PATH, transforms=val_transforms)
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices.indices)
    train_subset.dataframe = train_dataset.dataframe.iloc[train_indices.indices].reset_index(drop=True)
    
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples")
    
    # --- DATALOADER OPTIMIZATION (PHASE 4) ---
    # Weighted Sampler from Phase 2
    train_sampler = MultiTaskUniformSampler(train_subset, batch_size=BATCH_SIZE)
    
    train_loader = torch.utils.data.DataLoader(
        train_subset, 
        batch_sampler=train_sampler, 
        num_workers=min(os.cpu_count(), 8), # Optimized workers
        pin_memory=True,
        prefetch_factor=2,                  # Keep GPU fed
        persistent_workers=True,            # Avoid recreating workers
        collate_fn=multi_task_collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_subset, 
        batch_size=8,
        shuffle=False, 
        num_workers=min(os.cpu_count(), 8), 
        pin_memory=True,
        collate_fn=multi_task_collate_fn
    )
    
    # Model Setup
    model = MultiTaskModelFactory(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, task_configs=TASK_CONFIGURATIONS).to(device)
    
    loss_functions = {
        'segmentation': smp_losses.DiceLoss(mode='multiclass'), 
        'classification': nn.CrossEntropyLoss(),
        'Regression': nn.MSELoss(), 
        'detection': DetectionLoss()
    }
    task_id_to_name = {cfg['task_id']: cfg['task_name'] for cfg in TASK_CONFIGURATIONS}

    # Optimization
    param_groups = [
        {'params': model.encoder.parameters(), 'lr': LEARNING_RATE * 1},
    ]
    for task_id, head in model.heads.items():
        param_groups.append({'params': head.parameters(), 'lr': LEARNING_RATE * 10.0})

    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_val_score = -float('inf')
    print("\n" + "="*50 + "\n--- Start Training ---")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_train_losses = defaultdict(list)
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        
        for batch in loop:
            images = batch['image'].to(device)
            task_ids = batch['task_id']
            # All samples in batch should have same task_id, so safely stack labels
            try:
                labels = torch.stack([lbl.to(device) if isinstance(lbl, torch.Tensor) else torch.tensor(lbl, dtype=torch.float32, device=device) for lbl in batch['label']])
            except RuntimeError:
                # If shapes don't match, process individually - this should not happen with uniform sampler
                labels = [lbl.to(device) if isinstance(lbl, torch.Tensor) else torch.tensor(lbl, dtype=torch.float32, device=device) for lbl in batch['label']]

            current_task_id = task_ids[0]
            task_name = task_id_to_name[current_task_id]

            # --- MIXUP LOGIC (PHASE 3) ---
            # Apply MixUp only to Classification/Segmentation and stop near end of training
            apply_mixup = (task_name in ['classification', 'segmentation']) and (epoch < NUM_EPOCHS - 5)
            
            if apply_mixup:
                images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.4, device=device)
                
                # Forward pass
                outputs = model(images, task_id=current_task_id)
                
                # Calculate loss
                if task_name == 'detection':
                    # Detection special handling skipped for MixUp to stay safe, 
                    # but logic here for structure. 
                    pass 
                else:
                    loss = mixup_criterion(loss_functions[task_name], outputs, targets_a, targets_b, lam)
            else:
                # Standard training
                outputs = model(images, task_id=current_task_id)
                
                # Grid-based detection logic
                if task_name == 'detection':
                    _, _, h, w = outputs.shape
                    gt_center_x = (labels[:, 0] + labels[:, 2]) / 2.0
                    gt_center_y = (labels[:, 1] + labels[:, 3]) / 2.0
                    coord_h = torch.clamp((gt_center_y * h).long(), 0, h - 1)
                    coord_w = torch.clamp((gt_center_x * w).long(), 0, w - 1)
                    
                    final_outputs = torch.zeros((images.shape[0], 5), device=device)
                    for i in range(images.shape[0]):
                        final_outputs[i] = outputs[i, :, coord_h[i], coord_w[i]]
                    outputs = final_outputs

                loss = loss_functions[task_name](outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_losses[current_task_id].append(loss.item())
            loop.set_postfix(loss=loss.item(), task=current_task_id, lr=scheduler.get_last_lr()[0])

        # Validation & Saving (Same as before)
        val_results_df = evaluate(model, val_loader, device)
        score_cols = [col for col in val_results_df.columns if 'MAE' not in col and isinstance(val_results_df[col].iloc[0], (int, float))]
        
        avg_val_score = 0
        if not val_results_df.empty and score_cols:
            avg_val_score = val_results_df[score_cols].mean().mean()

        print(f"\nEpoch {epoch+1} Val Score: {avg_val_score:.4f}")

        if avg_val_score > best_val_score:
            best_val_score = avg_val_score
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"-> Saved Best Model: {best_val_score:.4f}")
        
        scheduler.step()

    print(f"\n--- Training Finished ---\nBest model: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()