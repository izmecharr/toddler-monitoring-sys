from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from multiprocessing import freeze_support

class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        
        # Get layers for feature extraction
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels
        
    def forward(self, x):
        # Store intermediate features
        features = []
        
        x = self.layer0(x)
        x = self.layer1(x)
        features.append(x)  # P3
        
        x = self.layer2(x)
        features.append(x)  # P4
        
        x = self.layer3(x)
        features.append(x)  # P5
        
        x = self.layer4(x)
        features.append(x)  # P6
        
        return features

# Modify YOLOv8 model
def create_resnet_yolo():
    # Load base YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    # Replace backbone
    model.model.backbone = ResNetBackbone(pretrained=True)
    
    return model

class ResNetYOLOAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        # Adaptation layers to match YOLOv8 expected channels
        self.adapt_p3 = nn.Conv2d(256, 128, 1)
        self.adapt_p4 = nn.Conv2d(512, 256, 1)
        self.adapt_p5 = nn.Conv2d(1024, 512, 1)
        self.adapt_p6 = nn.Conv2d(2048, 1024, 1)
        
        self.bn_p3 = nn.BatchNorm2d(128)
        self.bn_p4 = nn.BatchNorm2d(256)
        self.bn_p5 = nn.BatchNorm2d(512)
        self.bn_p6 = nn.BatchNorm2d(1024)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, features):
        p3 = self.relu(self.bn_p3(self.adapt_p3(features[0])))
        p4 = self.relu(self.bn_p4(self.adapt_p4(features[1])))
        p5 = self.relu(self.bn_p5(self.adapt_p5(features[2])))
        p6 = self.relu(self.bn_p6(self.adapt_p6(features[3])))
        
        return [p3, p4, p5, p6]

class ResNetYOLOv8(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = ResNetBackbone(pretrained=True)
        self.adapter = ResNetYOLOAdapter()
        # Rest of YOLOv8 architecture remains the same

def train_resnet_yolo(data_yaml, epochs=10):
    # Create model with ResNet50 backbone
    model = create_resnet_yolo()
    
    # Training configuration
    train_args = dict(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=16,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        amp =False,
        optimizer='AdamW',
        lr0=0.001,  # Initial learning rate
        lrf=0.01,   # Final learning rate
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # Feature extraction phase
        freeze=[f'backbone.layer{i}' for i in range(2)],  # Freeze early layers
    )
    
    # Train the model
    results = model.train(**train_args)
    return model, results

def implement_gradual_unfreeze(model, epoch):
    """Gradually unfreeze ResNet layers during training"""
    if epoch < 10:
        # Freeze entire backbone
        for param in model.backbone.parameters():
            param.requires_grad = False
    elif epoch < 20:
        # Unfreeze layer4
        for param in model.backbone.layer4.parameters():
            param.requires_grad = True
    elif epoch < 30:
        # Unfreeze layer3
        for param in model.backbone.layer3.parameters():
            param.requires_grad = True
    else:
        # Unfreeze all layers
        for param in model.backbone.parameters():
            param.requires_grad = True

# Save the custom model


if __name__ == '__main__':
    freeze_support()
    data_yaml = 'coco128.yaml'
    model, results = train_resnet_yolo(data_yaml)
