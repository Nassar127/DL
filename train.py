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
    set_seed
)

# Custom Learning Rate Scheduler with Warmup
class WarmupCosineScheduler:
    """Combines linear warmup with cosine annealing decay.
    
    During warmup phase (first warmup_epochs), learning rate increases linearly
    from near-zero to the base learning rate. After warmup, follows cosine annealing.
    This prevents shocking pre-trained weights while random task heads stabilize.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.current_epoch = 0
        
        # Store base learning rates for each parameter group
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self):
        """Update learning rate based on current epoch."""
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup phase: LR goes from ~0 to base_lr
            warmup_factor = self.current_epoch / self.warmup_epochs
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i] * warmup_factor
        else:
            # Cosine annealing phase after warmup
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.min_lr + (self.base_lrs[i] - self.min_lr) * cosine_decay
    
    def get_last_lr(self):
        """Return current learning rates for all parameter groups."""
        return [group['lr'] for group in self.optimizer.param_groups]

# Training configuration
LEARNING_RATE = 1e-4
BATCH_SIZE = 20
NUM_EPOCHS = 50 
DATA_ROOT_PATH = '/root/baseline/train'
ENCODER = 'efficientnet-b4'
ENCODER_WEIGHTS = 'imagenet'
RANDOM_SEED = 42
MODEL_SAVE_PATH = 'best_model.pth' 
VAL_SPLIT = 0.2

# Gradient Accumulation Configuration
# Simulates larger batch sizes by accumulating gradients over multiple micro-batches
# Effective batch size = BATCH_SIZE * ACCUMULATION_STEPS
# This helps stabilize batch normalization statistics without requiring massive GPU memory
ACCUMULATION_STEPS = 2  # Effective batch size will be 20 * 2 = 40

# Learning Rate Warmup Configuration
# Gradually increases learning rate from near-zero to target LR over first few epochs
# Prevents "shocking" pre-trained backbone weights with high LR while random heads stabilize
WARMUP_EPOCHS = 3  # Number of epochs for linear warmup phase

def main():
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")
    
    # Initialize Automatic Mixed Precision (AMP) gradient scaler
    # This scales loss values to prevent gradient underflow when using FP16 precision
    # Reduces memory usage by ~50% and speeds up training on modern GPUs with Tensor Cores
    scaler = GradScaler()
    print("AMP GradScaler initialized for mixed precision training")
    print(f"Gradient Accumulation: {ACCUMULATION_STEPS} steps (Effective batch size: {BATCH_SIZE * ACCUMULATION_STEPS})")

    # Data loading and splitting
    # Training transforms with augmentation
    train_transforms = A.Compose([
        A.Resize(256, 256), 
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.1), 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))
    
    # Validation transforms without augmentation
    val_transforms = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))

    # Create full dataset to get indices
    temp_dataset = MultiTaskDataset(data_root=DATA_ROOT_PATH, transforms=train_transforms)
    dataset_size = len(temp_dataset)
    val_size = int(dataset_size * VAL_SPLIT)
    train_size = dataset_size - val_size
    
    # Split indices
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    indices = list(range(dataset_size))
    train_indices, val_indices = torch.utils.data.random_split(indices, [train_size, val_size], generator=generator)
    
    # Create separate datasets with different transforms
    train_dataset = MultiTaskDataset(data_root=DATA_ROOT_PATH, transforms=train_transforms)
    val_dataset = MultiTaskDataset(data_root=DATA_ROOT_PATH, transforms=val_transforms)
    
    # Create subsets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices.indices)
    
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples")
    
    # Fix dataframe reference for subset
    train_subset.dataframe = train_dataset.dataframe.iloc[train_indices.indices].reset_index(drop=True)
    
    train_sampler = MultiTaskUniformSampler(train_subset, batch_size=BATCH_SIZE)
    train_loader = torch.utils.data.DataLoader(
        train_subset, 
        batch_sampler=train_sampler, 
        num_workers=4, 
        pin_memory=True,
        collate_fn=multi_task_collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_subset, 
        batch_size=8,
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        collate_fn=multi_task_collate_fn
    )
    
    # Model and loss setup
    model = MultiTaskModelFactory(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, task_configs=TASK_CONFIGURATIONS).to(device)
    
    loss_functions = {
        'segmentation': smp_losses.DiceLoss(mode='multiclass'), 
        'classification': nn.CrossEntropyLoss(),
        'Regression': nn.MSELoss(), 
        'detection': DetectionLoss()
    }
    task_id_to_name = {cfg['task_id']: cfg['task_name'] for cfg in TASK_CONFIGURATIONS}

    # Optimization setup
    print("\n--- Setting parameter groups ---")
    param_groups = [
        {'params': model.encoder.parameters(), 'lr': LEARNING_RATE * 1},
    ]
    print(f"  - Shared Encoder                 -> LR: {LEARNING_RATE * 1}")
    
    for task_id, head in model.heads.items():
        lr_multiplier = 10.0
        current_lr = LEARNING_RATE * lr_multiplier
        param_groups.append({'params': head.parameters(), 'lr': current_lr})
        print(f"  - Task Head '{task_id:<25}' -> LR: {current_lr}")

    optimizer = optim.AdamW(param_groups)
    
    # Learning rate scheduler with warmup and cosine annealing
    # Warmup protects pre-trained backbone from high initial LR shock
    # Cosine annealing gradually reduces LR for fine-tuning
    scheduler = WarmupCosineScheduler(
        optimizer, 
        warmup_epochs=WARMUP_EPOCHS, 
        total_epochs=NUM_EPOCHS, 
        min_lr=1e-6
    )
    print(f"\n--- Warmup + Cosine Annealing Scheduler configured ---")
    print(f"  - Warmup epochs: {WARMUP_EPOCHS}")
    print(f"  - Total epochs: {NUM_EPOCHS}")
    print(f"  - Min LR: 1e-6")

    best_val_score = -float('inf')
    print("\n" + "="*50 + "\n--- Start Training ---")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_train_losses = defaultdict(list)
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        
        for batch in loop:
            images = batch['image'].to(device)
            task_ids = batch['task_id']
            # Manually stack labels list to tensor
            labels = torch.stack(batch['label']).to(device)

            # All samples in batch belong to the same task due to sampler
            current_task_id = task_ids[0]
            task_name = task_id_to_name[current_task_id]

            # Enable automatic mixed precision for forward pass and loss calculation
            # Operations like convolutions and matrix multiplications run in FP16 for speed
            # while critical operations like reductions and loss calculations use FP32 for numerical stability
            with autocast():
                outputs = model(images, task_id=current_task_id)
                
                # Grid-based detection logic
                if task_name == 'detection':
                    _, _, h, w = outputs.shape
                    
                    # Calculate center of GT box (normalized)
                    gt_center_x = (labels[:, 0] + labels[:, 2]) / 2.0
                    gt_center_y = (labels[:, 1] + labels[:, 3]) / 2.0

                    # Map to grid coordinates
                    coord_h = torch.clamp((gt_center_y * h).long(), 0, h - 1)
                    coord_w = torch.clamp((gt_center_x * w).long(), 0, w - 1)

                    # Extract prediction from the specific grid cell
                    final_outputs = torch.zeros((images.shape[0], 5), device=device)
                    for i in range(images.shape[0]):
                        final_outputs[i] = outputs[i, :, coord_h[i], coord_w[i]]
                else:
                    final_outputs = outputs
                
                # Compute loss inside autocast context
                loss = loss_functions[task_name](final_outputs, labels)
                
                # Scale loss by accumulation steps to maintain correct gradient magnitude
                # When accumulating gradients, we need to average them across micro-batches
                loss = loss / ACCUMULATION_STEPS
            
            # Scale the loss to prevent gradient underflow in FP16
            # The scaler multiplies loss by a large factor (e.g., 2^16) before backward pass
            scaler.scale(loss).backward()
            
            # Only update weights after accumulating gradients from multiple micro-batches
            # This simulates training with a larger batch size without the memory cost
            if (loop.n + 1) % ACCUMULATION_STEPS == 0:
                # Unscale gradients internally and perform optimizer step
                # If gradients contain inf/NaN, this step is skipped automatically
                scaler.step(optimizer)
                
                # Update the scale factor for next iteration based on gradient health
                # Increases scale if gradients are healthy, decreases if inf/NaN detected
                scaler.update()
                
                # Zero gradients only after optimizer step
                optimizer.zero_grad()
            
            epoch_train_losses[current_task_id].append(loss.item() * ACCUMULATION_STEPS)
            loop.set_postfix(loss=loss.item() * ACCUMULATION_STEPS, task=current_task_id, lr=scheduler.get_last_lr()[0])

        # Train reporting
        print("\n--- Epoch {} Average Train Loss Report ---".format(epoch + 1))
        sorted_task_ids = sorted(epoch_train_losses.keys())
        for task_id in sorted_task_ids:
            avg_loss = np.mean(epoch_train_losses[task_id])
            print(f"  - Task '{task_id:<25}': {avg_loss:.4f}")
        print("-" * 40)

        # Validation
        val_results_df = evaluate(model, val_loader, device)
        
        score_cols = [col for col in val_results_df.columns if 'MAE' not in col and isinstance(val_results_df[col].iloc[0], (int, float))]
        avg_val_score = 0
        if not val_results_df.empty and score_cols:
            avg_val_score = val_results_df[score_cols].mean().mean()

        print("\n--- Epoch {} Validation Report ---".format(epoch + 1))
        if not val_results_df.empty:
            print(val_results_df.to_string(index=False))
        print(f"--- Average Val Score (Higher is better): {avg_val_score:.4f} ---")

        if avg_val_score > best_val_score:
            best_val_score = avg_val_score
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"-> New best model saved! Score improved to: {best_val_score:.4f}\n")
        
        # Update scheduler
        scheduler.step()

    print(f"\n--- Training Finished ---\nBest model saved at: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()
