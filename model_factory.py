import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import List, Dict


# Task configuration list
TASK_CONFIGURATIONS = [
    {'task_name': 'Regression', 'num_classes': 2, 'task_id': 'FUGC'},
    {'task_name': 'Regression', 'num_classes': 3, 'task_id': 'IUGC'},
    {'task_name': 'Regression', 'num_classes': 2, 'task_id': 'fetal_femur'},
    {'task_name': 'classification', 'num_classes': 2, 'task_id': 'breast_2cls'},
    {'task_name': 'classification', 'num_classes': 3, 'task_id': 'breast_3cls'},
    {'task_name': 'classification', 'num_classes': 8, 'task_id': 'fetal_head_pos_cls'},
    {'task_name': 'classification', 'num_classes': 6, 'task_id': 'fetal_plane_cls'},
    {'task_name': 'classification', 'num_classes': 8, 'task_id': 'fetal_sacral_pos_cls'},
    {'task_name': 'classification', 'num_classes': 2, 'task_id': 'liver_lesion_2cls'},
    {'task_name': 'classification', 'num_classes': 2, 'task_id': 'lung_2cls'},
    {'task_name': 'classification', 'num_classes': 3, 'task_id': 'lung_disease_3cls'},
    {'task_name': 'classification', 'num_classes': 6, 'task_id': 'organ_cls'},
    {'task_name': 'detection', 'num_classes': 1, 'task_id': 'spinal_cord_injury_loc'},
    {'task_name': 'detection', 'num_classes': 1, 'task_id': 'thyroid_nodule_det'},
    {'task_name': 'detection', 'num_classes': 1, 'task_id': 'uterine_fibroid_det'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'breast_lesion'},
    {'task_name': 'segmentation', 'num_classes': 4, 'task_id': 'cardiac_multi'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'carotid_artery'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'cervix'},
    {'task_name': 'segmentation', 'num_classes': 3, 'task_id': 'cervix_multi'},
    {'task_name': 'segmentation', 'num_classes': 5, 'task_id': 'fetal_abdomen_multi'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'fetal_head'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'fetal_heart'},
    {'task_name': 'segmentation', 'num_classes': 3, 'task_id': 'head_symphysis_multi'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'lung'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'ovary_tumor'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'thyroid_nodule'},
]

# Task specific heads

class SmpClassificationHead(nn.Module):
    """Wrapper for SMP Classification Head."""
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.head = smp.base.ClassificationHead(
            in_channels=in_channels,
            classes=num_classes,
            pooling="avg",
            dropout=0.2,
            activation=None,
        )
        
    def forward(self, features: list):
        # Use the last feature map from encoder
        return self.head(features[-1])

class RegressionHead(nn.Module):
    """Custom head for regression tasks."""
    def __init__(self, in_channels: int, num_points: int):
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        
        # Hidden layer dimension (expand slightly to capture relationships)
        hidden_dim = 1024 
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.SiLU(inplace=True),       # Modern Swish activation
            nn.Dropout(p=0.3),           # Regularization
            nn.Linear(hidden_dim, num_points * 2)
        )

    def forward(self, features: list):
        x = self.pooling(features[-1])
        x = self.flatten(x)
        return self.mlp(x)

class FPNGridDetectionHead(nn.Module):
    """Detection head designed for FPN outputs."""
    def __init__(self, fpn_out_channels: int, num_classes: int = 1, num_anchors: int = 1):
        super().__init__()
        mid_channels = 128
        num_outputs = num_anchors * (4 + num_classes)
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(fpn_out_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, num_outputs, kernel_size=1)
        )

    def forward(self, fpn_features: torch.Tensor):
        # Input fpn_features is already a single fused tensor from FPN Decoder
        predictions_map = self.conv_block(fpn_features)
        
        # Apply sigmoid to bbox coordinates (first 4 channels)
        predictions_map[:, :4] = torch.sigmoid(predictions_map[:, :4])
        
        return predictions_map

# ====================================================================
# --- 2. SCSEModule ---
# ====================================================================

class SCSEModule(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (scSE).
    Crucial for Ultrasound to suppress speckle noise and highlight organic boundaries.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Attention = Channel_Attention * x + Spatial_Attention * x
        return x * self.cSE(x) + x * self.sSE(x)

# ====================================================================
# --- 3. Mish and SegmentationHead ---
# ====================================================================

class Mish(nn.Module):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function"""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class CustomSegmentationHead(nn.Module):
    """
    Deep Segmentation Head with Mish Activation.
    Conv3x3 -> BN -> Mish -> Conv1x1 -> Upsample
    """
    def __init__(self, in_channels, out_channels, upsampling=4):
        super().__init__()
        mid_channels = in_channels // 2
        self.block = nn.Sequential(
            # Context Layer (3x3)
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            Mish(),  # <--- The Modern Activation
            # Projection Layer (1x1)
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
            # Upsampling
            nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        )
        
    def forward(self, x):
        return self.block(x)

# ====================================================================
# --- 4. Multi-Task Model Factory ---
# ====================================================================

class MultiTaskModelFactory(nn.Module):
    def __init__(self, encoder_name: str, encoder_weights: str, task_configs: List[Dict]):
        super().__init__()
        
        # Initialize shared encoder
        print(f"Initializing shared encoder: {encoder_name}")
        self.encoder = smp.encoders.get_encoder(
            name=encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights,
        )
        
        # Initialize shared FPN decoder
        temp_fpn_model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1, 
        )
        self.fpn_decoder = temp_fpn_model.decoder

        print("Initializing scSE Attention...")
        # scSE operates on the FPN output channels
        self.attention = SCSEModule(in_channels=self.fpn_decoder.out_channels)
        
        # Initialize task heads
        self.heads = nn.ModuleDict()
        
        print(f"Creating heads for {len(task_configs)} tasks...")
        for config in task_configs:
            task_id = config['task_id']
            task_name = config['task_name']
            num_classes = config['num_classes']
            
            head_module = None
            if task_name == 'segmentation':
                head_module = CustomSegmentationHead(
                    in_channels=self.fpn_decoder.out_channels, 
                    out_channels=num_classes, 
                    upsampling=4 
                )

            elif task_name == 'classification':
                head_module = SmpClassificationHead(
                    in_channels=self.encoder.out_channels[-1],
                    num_classes=num_classes
                )

            elif task_name == 'Regression':
                num_points = config['num_classes']
                head_module = RegressionHead(
                    in_channels=self.encoder.out_channels[-1],
                    num_points=num_points
                )

            elif task_name == 'detection':
                head_module = FPNGridDetectionHead(
                    fpn_out_channels=self.fpn_decoder.out_channels,
                    num_classes=num_classes
                )

            if head_module:
                self.heads[task_id] = head_module
            else:
                print(f"Warning: Unknown task type '{task_name}' for {task_id}")

    def forward(self, x: torch.Tensor, task_id: str) -> torch.Tensor:
        features = self.encoder(x)
        
        if task_id not in self.heads:
            raise ValueError(f"Task ID '{task_id}' not found.")

        task_config = next((item for item in TASK_CONFIGURATIONS if item["task_id"] == task_id), None)
        task_name = task_config['task_name'] if task_config else None

        # Route features based on task type
        if task_name in ['segmentation', 'detection']:
            # Use FPN features for dense prediction tasks
            fpn_features = self.fpn_decoder(features)
            fpn_features = self.attention(fpn_features)
            output = self.heads[task_id](fpn_features)
        else: 
            # Use encoder features directly for global prediction tasks
            output = self.heads[task_id](features)
            
        return output

# Example usage

if __name__ == '__main__':
    model = MultiTaskModelFactory(
        encoder_name='efficientnet-b7',
        encoder_weights='imagenet',
        task_configs=TASK_CONFIGURATIONS
    )

    print("\n--- Model Architecture Verification ---")
    # Test with the new resolution
    dummy_image_batch = torch.randn(2, 3, 384, 384) 

    # Test specific tasks
    test_tasks = ['cardiac_multi', 'fetal_plane_cls', 'FUGC', 'thyroid_nodule_det']
    
    for t_id in test_tasks:
        try:
            out = model(dummy_image_batch, task_id=t_id)
            print(f"Task: {t_id:<25} | Output Shape: {out.shape}")
        except Exception as e:
            print(f"Task: {t_id:<25} | Error: {e}")