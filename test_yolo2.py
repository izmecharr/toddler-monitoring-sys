from ultralytics.nn.modules import Conv, C2f
import torch.nn as nn

class CustomConvBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        # First additional conv layer with 4x4 filter
        self.conv1 = Conv(
            c1=in_channels,
            c2=512,
            k=4,
            s=1,
            p=1
        )
        
        # Second additional conv layer with 3x3 filter
        self.conv2 = Conv(
            c1=512,
            c2=256,  # Match the original output channels
            k=3,
            s=1,
            p=1
        )
        
        # Add Batch Normalization and Dropout for regularization
        self.bn = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x

def modify_yolo_model(model):
    """
    Modify YOLOv8 model by adding new convolutional layers before the detection head
    """
    # Create our custom conv block
    custom_block = CustomConvBlock(in_channels=256)  # Input from SPPF layer
    
    # Get the model's sequential layers
    sequential = model.model.model
    
    # Find the index of the Detect layer
    detect_idx = None
    for i, layer in enumerate(sequential):
        if 'Detect' in str(type(layer)):
            detect_idx = i
            break
            
    if detect_idx is None:
        raise ValueError("Could not find Detect layer")
    
    # Insert our custom block before the detection head
    # We'll insert it after the last C2f layer but before Detect
    new_sequence = nn.ModuleList()
    
    # Add all layers up to the detection head
    for i in range(detect_idx):
        new_sequence.append(sequential[i])
    
    # Add our custom block
    new_sequence.append(custom_block)
    
    # Add the detection head
    detect_layer = sequential[detect_idx]
    # Update detection layer input channels if needed
    detect_layer.cv2[0][0].conv.in_channels = 256  # Match with our output channels
    detect_layer.cv2[1][0].conv.in_channels = 256
    detect_layer.cv2[2][0].conv.in_channels = 256
    
    new_sequence.append(detect_layer)
    
    # Replace the model's sequential layers
    model.model.model = new_sequence
    
    return model

# Usage example:
if __name__ == "__main__":
    from ultralytics import YOLO
    
    # Load base model
    model = YOLO('yolov8n.pt')
    
    # Modify backbone
    modified_model = modify_yolo_model(model)
    
    # Training configuration
    training_args = {
        'data':'coco128.yaml',
        'epochs': 50,
        'batch': 16,
        'patience': 5,
        'amp': False,
        'optimizer': 'AdamW',
        'weight_decay': 0.0001,
        'lr0': 0.002,
        'lrf': 0.01,
        'warmup_epochs': 3,
        'cos_lr': True,
        'augment': True,
        'mixup': 0.1,
        'mosaic': 1.0,
        'close_mosaic': 10,
        'label_smoothing': 0.1
    }
    
    # Train the modified model
    modified_model.train(**training_args)