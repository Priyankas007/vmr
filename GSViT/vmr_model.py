import torch
import torch.nn as nn
from EfficientViT.classification.model.build import EfficientViT_M5
from typing import Tuple, List, Optional

class SurgicalStepClassifier(nn.Module):
    def __init__(self, in_features: int = 384, num_classes: int = 8, 
                 hidden_dims: Tuple[int, int] = (2048, 512), dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, 384, 4, 4] -> [B, 384, 1, 1]
            nn.Flatten(),              # [B, 384, 1, 1] -> [B, 384]

            nn.Linear(in_features, hidden_dims[0]),  # [B, 384] -> [B, 2048]
            nn.ELU(),
            nn.LayerNorm(hidden_dims[0]),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[0], hidden_dims[1]),  # [B, 2048] -> [B, 512]
            nn.ELU(),
            nn.LayerNorm(hidden_dims[1]),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[1], num_classes)  # [B, 512] -> [B, num_classes]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class GSViT(nn.Module):
    def __init__(self, num_classes: int, finetune_mode: str, dropout: float):
        super(GSViT, self).__init__()
        self.evit = EfficientViT_M5(pretrained='efficientvit_m5')
        # self.evit = EfficientViT_M5() # no pretrained efficient vit weights
        self.evit = torch.nn.Sequential(*list(self.evit.children())[:-1])  # shape [B, 384, 4, 4]
        self.classifier = SurgicalStepClassifier(in_features=384, 
                                                 num_classes=num_classes, 
                                                 hidden_dims=(2048, 512),
                                                 dropout=dropout)
        
        if finetune_mode == 'linear_probe':
            print("Finetuning mode: linear_probe (only the classification head is trainable)")
            for param in self.evit.parameters():
                param.requires_grad = False
        elif finetune_mode == 'finetune':
            print("Finetuning mode: finetune (all parameters are trainable)")
            for param in self.evit.parameters():
                assert param.requires_grad
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.evit(x)  # [B, 384, 4, 4]
        out = self.classifier(features)  # [B, num_classes]
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GSViT(num_classes=8, finetune_mode='finetune', dropout=0.1).to(device)

dummy_input = torch.randn(1, 3, 224, 224).to(device)
output = model(dummy_input)
print(output.shape)
