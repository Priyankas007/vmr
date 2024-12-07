import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .lora import LoRA
from .linear_decoder import LinearClassifier

# from scratch
class DINOv2ClassifierScratch(nn.Module):
    def __init__(self, model_name, num_classes):
        super(DINOv2ClassifierScratch, self).__init__()
	
        # load in pretrained DINOv2 weights
        self.feature_extractor = torch.hub.load(
            'facebookresearch/dinov2',  # GitHub repository
            model_name,            # Model variant (e.g., ViT-B/14)
            pretrained=False             # Load pretrained weights
        )

        dim = 768 if model_name == 'dinov2_vitb14' else 384

        self.feature_extractor.head = nn.Identity()  # Remove original classifier
        self.classifier = nn.Linear(dim, num_classes)  # Adjust for your dataset

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

# linear probing
class DINOv2ClassifierLinearProbe(nn.Module):
    def __init__(self, model_name, num_classes):
        super(DINOv2ClassifierLinearProbe, self).__init__()
	
        # load in pretrained DINOv2 weights
        self.feature_extractor = torch.hub.load(
            'facebookresearch/dinov2',  # GitHub repository
            model_name,            # Model variant (e.g., ViT-B/14)
            pretrained=True             # Load pretrained weights
        )

        dim = 768 if model_name == 'dinov2_vitb14' else 384

        self.feature_extractor.head = nn.Identity()  # Remove original classifier
        self.classifier = nn.Linear(dim, num_classes)  # Adjust for your dataset
        
         # Freeze the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

# finetune
class DINOv2ClassifierFinetune(nn.Module):
    def __init__(self, model_name, num_classes):
        super(DINOv2ClassifierFinetune, self).__init__()
	
        # load in pretrained DINOv2 weights
        self.feature_extractor = torch.hub.load(
            'facebookresearch/dinov2',  # GitHub repository
            model_name,                 # Model variant (e.g., ViT-B/14)
            pretrained=True             # Load pretrained weights
        )

        dim = 768 if model_name == 'dinov2_vitb14' else 384

        self.feature_extractor.head = nn.Identity()  # Remove original classifier
        self.classifier = nn.Linear(dim, num_classes)  # Adjust for your dataset

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)


# with LoRA
class DINOv2ClassifierFinetuneLoRA(nn.Module):
    def __init__(self, 
                model_name,
                num_classes,
                r: int = 3,
                img_dim: tuple[int, int] = (224, 224)
                ):
        super(DINOv2ClassifierFinetuneLoRA, self).__init__()
        assert r > 0
        
        self.inter_layers = 4 # number of previous layers to use as inupt
	
        # load in pretrained DINOv2 weights
        self.feature_extractor = torch.hub.load(
            'facebookresearch/dinov2',  # GitHub repository
            model_name,            # Model variant (e.g., ViT-B/14)
            pretrained=True     # Load pretrained weights
        )
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        #dim = 768 if model_name == 'dinov2_vitb14' else 384
        self.dim = self.feature_extractor.embed_dim
        
        self.feature_extractor.head = nn.Identity()  # Remove original classifier
        self.classifier = nn.Linear(self.dim, num_classes)  # Adjust for your dataset
        
        # Decoder
        #self.decoder = LinearClassifier(
            #self.dim,
            #patch_h=int(img_dim[0] / self.feature_extractor.patch_size),
            #patch_w=int(img_dim[1] / self.feature_extractor.patch_size),
            #n_classes=num_classes,
        #)
        
        # LoRA blocks
        self.lora_layers = list(range(len(self.feature_extractor.blocks)))
        print(f"self.lora_layers: {self.lora_layers}")
        self.w_a = []
        self.w_b = []
        
        for i, block in enumerate(self.feature_extractor.blocks):
            if i not in self.lora_layers:
                continue
            w_qkv_linear = block.attn.qkv
            dim = w_qkv_linear.in_features
            
            w_a_linear_q, w_b_linear_q = self._create_lora_layer(self.dim, r)
            w_a_linear_v, w_b_linear_v = self._create_lora_layer(self.dim, r)
            
            self.w_a.extend([w_a_linear_q, w_a_linear_v])
            self.w_b.extend([w_b_linear_q, w_b_linear_v])

            block.attn.qkv = LoRA(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
            )
            
        self._reset_lora_parameters()

    def _create_lora_layer(self, dim: int, r: int):
        w_a = nn.Linear(dim, r, bias=False)
        w_b = nn.Linear(r, dim, bias=False)
        return w_a, w_b

    def _reset_lora_parameters(self) -> None:
        for w_a in self.w_a:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_b:
            nn.init.zeros_(w_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)

        # get the patch embeddings - so we exclude the CLS token
        # patch_embeddings = feature["x_norm_patchtokens"]
        # logits s= self.decoder(patch_embeddings)
        #logits = self.decoder(features)
        #logits = logits.mean(dim=(2, 3))

        #logits = F.interpolate(
            #logits,
           # size=x.shape[2:],
          #  mode="bilinear",
         #   align_corners=False,
        #)
        return self.classifier(features)


