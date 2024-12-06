import torch
import torch.nn as nn




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
            model_name,            # Model variant (e.g., ViT-B/14)
            pretrained=True             # Load pretrained weights
        )

        dim = 768 if model_name == 'dinov2_vitb14' else 384

        self.feature_extractor.head = nn.Identity()  # Remove original classifier
        self.classifier = nn.Linear(dim, num_classes)  # Adjust for your dataset

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

