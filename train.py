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
ENCODER = 'efficientnet-b4'
ENCODER_WEIGHTS = 'imagenet'
RANDOM_SEED = 42
MODEL_SAVE_PATH = 'best_model.pth' 
VAL_SPLIT = 0.2

class MultiTaskLossWrapper(nn.Module):
    """
    Phase 3: Automatic Multi-Task Loss Weighting.
    Learns to balance gradients between Seg, Det, Reg, and Cls.
    """
    def __init__(self, model, task_names):
        super(MultiTaskLossWrapper, self).__init__()
        self.model = model
        # Create a learnable log variance parameter (log(sigma^2)) for each task
        # Initializing at 0.0 means sigma=1.0 (equal weighting initially)
        self.log_vars = nn.ParameterDict({
            task: nn.Parameter(torch.zeros(1)) for task in task_names
        })

    def forward(self, x, task_id, task_name, targets, criterion):
        outputs = self.model(x, task_id=task_id)
        
        # --- Handle Detection Logic (from your train.py) ---
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

        # --- Calculate Weighted Loss ---
        raw_loss = criterion(final_outputs, targets)
        
        # The Kendall et al. formula: (1 / 2*sigma^2) * L + log(sigma)
        # We use log_var (s) to represent log(sigma^2) for numerical stability
        precision = torch.exp(-self.log_vars[task_name])
        weighted_loss = precision * raw_loss + self.log_vars[task_name]
        
        return weighted_loss, raw_loss

def main():
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    # --- 1. Data Setup (Standard) ---
    train_transforms = A.Compose([
        A.Resize(256, 256), 
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.1), 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))
    
    val_transforms = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))

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

    # --- 2. Phase 3 & 4: Model and Loss Wrapper Setup ---
    # Create the base model architecture
    base_model = MultiTaskModelFactory(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        task_configs=TASK_CONFIGURATIONS
    ).to(device)
    
    # Wrap the model for Automatic Multi-Task Loss Weighting
    task_names = ['segmentation', 'classification', 'Regression', 'detection']
    model = MultiTaskLossWrapper(base_model, task_names).to(device)

    # Define Loss Functions with Phase 1 (DiceFocal) and Phase 4 (Label Smoothing)
    loss_functions = {
        'segmentation': DiceFocalLoss(gamma=2.0, alpha=0.25), 
        'classification': nn.CrossEntropyLoss(label_smoothing=0.1), # Phase 4
        'Regression': nn.MSELoss(), 
        'detection': DetectionLoss() # Phase 2 CIoU is inside this
    }
    task_id_to_name = {cfg['task_id']: cfg['task_name'] for cfg in TASK_CONFIGURATIONS}

    # --- 3. Phase 3: Optimized Parameter Groups ---
    print("\n--- Setting parameter groups (Expert 3 Configuration) ---")
    param_groups = [
        # Encoder learning rate
        {'params': base_model.encoder.parameters(), 'lr': LEARNING_RATE},
        # Learning rate for the Task Uncertainty Weights (Log Vars)
        {'params': model.log_vars.parameters(), 'lr': LEARNING_RATE} 
    ]
    
    # Add Task Heads with a higher LR for faster adaptation
    for task_id, head in base_model.heads.items():
        param_groups.append({'params': head.parameters(), 'lr': LEARNING_RATE * 10.0})

    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # --- 4. Training Loop ---
    # --- 4. Training Loop with Weight Logger ---
    best_val_score = -float('inf')
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_train_losses = defaultdict(list)
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for batch in loop:
            images = batch['image'].to(device)
            task_ids = batch['task_id']
            labels = torch.stack(batch['label']).to(device)
            
            current_task_id = task_ids[0]
            task_name = task_id_to_name[current_task_id]

            # Wrapper handles forward pass and uncertainty math
            weighted_loss, raw_loss = model(
                images, 
                task_id=current_task_id, 
                task_name=task_name, 
                targets=labels, 
                criterion=loss_functions[task_name]
            )
            
            optimizer.zero_grad()
            weighted_loss.backward()
            
            # Phase 5: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # --- Logger: Monitor Learnable Weights (Sigma) ---
            # Every 10 batches, calculate the current sigma for the active task
            if loop.n % 10 == 0:
                with torch.no_grad():
                    # sigma = sqrt(exp(log_var))
                    current_log_var = model.log_vars[task_name]
                    current_sigma = torch.exp(current_log_var).sqrt().item()
                
                # Update the progress bar with both raw loss and the learnable weight
                loop.set_postfix(
                    task=task_name,
                    loss=f"{raw_loss.item():.4f}",
                    sigma=f"{current_sigma:.3f}"
                )
            
            epoch_train_losses[current_task_id].append(raw_loss.item())

        # --- 5. Validation (Evaluates the base model) ---
        val_results_df = evaluate(base_model, val_loader, device)
        score_cols = [col for col in val_results_df.columns if 'MAE' not in col and isinstance(val_results_df[col].iloc[0], (int, float))]
        avg_val_score = val_results_df[score_cols].mean().mean() if not val_results_df.empty else 0

        print(f"\n--- Epoch {epoch+1} Average Val Score: {avg_val_score:.4f} ---")

        if avg_val_score > best_val_score:
            best_val_score = avg_val_score
            torch.save(base_model.state_dict(), MODEL_SAVE_PATH)
            print(f"-> Saved New Best Multi-Task Model!\n")
        
        scheduler.step()

    print(f"\n--- Mission Complete! Best Score: {best_val_score:.4f} ---")

if __name__ == '__main__':   
    main()