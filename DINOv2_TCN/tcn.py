import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class CausalTemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(CausalTemporalBlock, self).__init__()
        
        self.causal_padding = (kernel_size - 1) * dilation
        
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, dilation=dilation))
        
        self.dropout = nn.Dropout(dropout)
        self.downsample = weight_norm(nn.Conv1d(n_inputs, n_outputs, 1)) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

        nn.init.normal_(self.conv1.weight, 0, 0.01)
        nn.init.normal_(self.conv2.weight, 0, 0.01)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0, 0.01)

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        
        out = F.pad(x, (self.causal_padding, 0))
        out = self.conv1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = F.pad(out, (self.causal_padding, 0))
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        return self.relu(out + res)

class CausalDinoTCN(nn.Module):
    def __init__(self, 
                 num_classes=8, 
                 sequence_length=16, 
                 num_levels = 3,
                 num_channels=64, 
                 kernel_size=3, 
                 dropout=0.2,
                 model_name='dinov2_vits14',
                 train_end_to_end=True):
        super(CausalDinoTCN, self).__init__()
    
        self.feature_extractor = torch.hub.load(
            'facebookresearch/dinov2',
            model_name,
            pretrained=True
        )
        
        self.dino_dim = 384 if model_name == 'dinov2_vits14' else 768
        self.num_channels = num_channels
        self.sequence_length = sequence_length
        self.train_end_to_end = train_end_to_end
        
        if not train_end_to_end:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            print("Finetuning mode: linear_probe (only the TCN head is trainable)")
        else:
            print("Finetuning mode: finetune (all parameters are trainable)")
        
        
        self.input_projection = nn.Linear(self.dino_dim, num_channels)
        
        layers = []
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_channels
            out_channels = num_channels
            layers.append(CausalTemporalBlock(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation_size,
                dropout=dropout
            ))
        
        self.temporal_layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_channels, num_classes)

    def forward(self, x):  # x is [B, T, C, H, W]
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # dino
        x = x.view(-1, x.size(2), x.size(3), x.size(4))  # [B*T, C, H, W]
        embeddings = self.feature_extractor(x)
        embeddings = embeddings.view(batch_size, seq_len, self.dino_dim)
        
        x = self.input_projection(embeddings)  # [B, T, num_channels]
        x = x.transpose(1, 2)  # [B, num_channels, T]
        
        # tcn
        x = self.temporal_layers(x)
        
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalDinoTCN(num_classes=8, sequence_length=16, train_end_to_end=True).to(device)
    
    dummy_input = torch.randn(4, 16, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f'Output shape: {output.shape}')
