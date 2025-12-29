import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch.losses as smp_losses
import numpy as np
import random

# Import local modules
from dataset import MultiTaskDataset, MultiTaskUniformSampler
from model_factory import MultiTaskModelFactory, TASK_CONFIGURATIONS
from utils import (
    multi_task_collate_fn, 
    evaluate, 
    DetectionLoss, 
    set_seed,
    DiceFocalLoss
)

# Training configuration
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 50 
DATA_ROOT_PATH = r"E:\nu\deep\proj\Data\train"
ENCODER = 'efficientnet-b7' # Expert 1: Updated to B7
ENCODER_WEIGHTS = 'imagenet'
RANDOM_SEED = 42
MODEL_SAVE_PATH = 'best_model.pth' 
VAL_SPLIT = 0.2

# Gradient Accumulation & Warmup
ACCUMULATION_STEPS = 4  # Increased to 4 (Effective BS=16) for B7 stability
WARMUP_EPOCHS = 3 

# --- Custom Scheduler ---
class WarmupCosineScheduler:
    """Combines linear warmup with cosine annealing decay."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.current_epoch = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            warmup_factor = self.current_epoch / self.warmup_epochs
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i] * warmup_factor
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.min_lr + (self.base_lrs[i] - self.min_lr) * cosine_decay
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

# --- Expert 3: Loss Wrapper ---
class MultiTaskLossWrapper(nn.Module):
    def __init__(self, model, task_names):
        super(MultiTaskLossWrapper, self).__init__()
        self.model = model
        self.log_vars = nn.ParameterDict({
            task: nn.Parameter(torch.zeros(1)) for task in task_names
        })

    def forward(self, x, task_id, task_name, targets, criterion):
        outputs = self.model(x, task_id=task_id)
        
        # Detection Logic (Moved inside wrapper)
        if task_name == 'detection':
            _, _, h, w = outputs.shape
            gt_center_x = (targets[:, 0] + targets[:, 2]) / 2.0
            gt_center_y = (targets[:, 1] + targets[:, 3]) / 2.0
            coord_h = torch.clamp((gt_center_y * h).long(), 0, h - 1)
            coord_w = torch.clamp((gt_center_x * w).long(), 0, w - 1)
            final_outputs = torch.zeros((x.shape[0], 5), device=x.device)
            for i in range(x.shape[0]):
                final_outputs[i] = outputs[i, :, coord_h[i], coord_w[i]]
        else:
            final_outputs = outputs

        raw_loss = criterion(final_outputs, targets)
        
        # Uncertainty Weighting
        precision = torch.exp(-self.log_vars[task_name])
        weighted_loss = precision * raw_loss + self.log_vars[task_name]
        
        return weighted_loss, raw_loss

def main():
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")
    
    scaler = GradScaler()
    print(f"AMP Initialized. Gradient Accumulation: {ACCUMULATION_STEPS} steps.")

    # --- Data Setup ---
    # EXPERT 2 NOTE: Using Safe Augmentations to prevent crashes
    train_transforms = A.Compose([
        A.Resize(384, 384), # Expert 1: 384x384
        A.OneOf([
            A.CLAHE(clip_limit=4.0, p=0.7),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ], p=0.8),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.0, check_each_transform=False))
    
    val_transforms = A.Compose([
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.0, check_each_transform=False))

    train_dataset = MultiTaskDataset(data_root=DATA_ROOT_PATH, transforms=train_transforms)
    val_dataset = MultiTaskDataset(data_root=DATA_ROOT_PATH, transforms=val_transforms)
    
    indices = list(range(len(train_dataset)))
    split = int(len(indices) * VAL_SPLIT)
    train_indices, val_indices = indices[split:], indices[:split]

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    train_subset.dataframe = train_dataset.dataframe.iloc[train_indices].reset_index(drop=True)

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_sampler=MultiTaskUniformSampler(train_subset, BATCH_SIZE),
        num_workers=4, pin_memory=True, collate_fn=multi_task_collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=8, shuffle=False, num_workers=4, collate_fn=multi_task_collate_fn
    )

    # --- Model Setup ---
    base_model = MultiTaskModelFactory(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        task_configs=TASK_CONFIGURATIONS
    ).to(device)
    
    task_names = ['segmentation', 'classification', 'Regression', 'detection']
    model = MultiTaskLossWrapper(base_model, task_names).to(device)

    # Expert 3 Losses
    loss_functions = {
        'segmentation': DiceFocalLoss(gamma=2.0, alpha=0.25), 
        'classification': nn.CrossEntropyLoss(label_smoothing=0.1),
        'Regression': nn.MSELoss(), 
        'detection': DetectionLoss()
    }
    task_id_to_name = {cfg['task_id']: cfg['task_name'] for cfg in TASK_CONFIGURATIONS}

    # Optimizer & Scheduler
    param_groups = [
        {'params': base_model.encoder.parameters(), 'lr': LEARNING_RATE},
        {'params': model.log_vars.parameters(), 'lr': LEARNING_RATE},
        {'params': base_model.heads.parameters(), 'lr': LEARNING_RATE * 10.0}
    ]

    optimizer = optim.AdamW(param_groups)
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS)

    # --- Training Loop (Corrected) ---
    best_val_score = -float('inf')
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_train_losses = defaultdict(list)
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(loop):
            images = batch['image'].to(device)
            task_ids = batch['task_id']
            labels = torch.stack(batch['label']).to(device)
            
            current_task_id = task_ids[0]
            task_name = task_id_to_name[current_task_id]

            # --- AMP Forward Pass ---
            with autocast():
                # The wrapper handles Forward + Loss + Uncertainty Weighting
                weighted_loss, raw_loss = model(
                    images, 
                    task_id=current_task_id, 
                    task_name=task_name, 
                    targets=labels, 
                    criterion=loss_functions[task_name]
                )
                
                # Gradient Accumulation Scaling
                loss = weighted_loss / ACCUMULATION_STEPS

            # --- AMP Backward Pass ---
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # --- Logging ---
            epoch_train_losses[current_task_id].append(raw_loss.item())
            
            if batch_idx % 10 == 0:
                with torch.no_grad():
                    current_sigma = torch.exp(model.log_vars[task_name]).sqrt().item()
                
                loop.set_postfix(
                    task=task_name,
                    loss=f"{raw_loss.item():.4f}",
                    sigma=f"{current_sigma:.3f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}"
                )

        # End of Epoch
        scheduler.step()
        
        print("\n--- Train Report ---")
        for t_id, losses in epoch_train_losses.items():
            print(f"  - {t_id}: {np.mean(losses):.4f}")

        # Evaluation
        val_results_df = evaluate(base_model, val_loader, device)
        # Calculate simplistic score (custom logic)
        score_cols = [c for c in val_results_df.columns if isinstance(val_results_df[c].iloc[0], (int, float))]
        avg_val_score = val_results_df[score_cols].mean().mean() if not val_results_df.empty else 0
        
        print(f"Val Score: {avg_val_score:.4f}")

        if avg_val_score > best_val_score:
            best_val_score = avg_val_score
            torch.save(base_model.state_dict(), MODEL_SAVE_PATH)
            print("-> Model Saved!")

if __name__ == '__main__':   
    main()