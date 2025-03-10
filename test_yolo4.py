from ultralytics.nn.modules import Conv, C2f
import torch.nn as nn
import os
import torch
import json
from datetime import datetime

class CustomConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # First convolution with kernel=5
        self.conv1 = Conv(
            c1=in_channels,
            c2=out_channels,  # Use same channel as original for intermediate
            k=5,
            s=1,
            p=2  # Padding=(kernel_size-1)//2 to maintain spatial dimensions
        )
        
        # Second convolution with kernel=3
        self.conv2 = Conv(
            c1=out_channels,
            c2=out_channels,  # Match the original output channels
            k=3,
            s=1,
            p=1
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=0.1)
        
        # Add residual connection
        self.use_residual = (in_channels == out_channels)

    def forward(self, x):
        identity = x if self.use_residual else None
        
        x = self.conv1(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.conv2(x)
        
        if self.use_residual:
            x = x + identity
        
        return x

def modify_yolo_model(model):
    sequential = model.model.model
    new_sequence = nn.ModuleList()
    
    i = 0
    while i < len(sequential):
        layer = sequential[i]
        layer_type = str(type(layer))
        
        if 'Conv' in layer_type and 'C2f' not in layer_type:
            # Get input channels from current Conv layer
            in_channels = layer.conv.in_channels
            out_channels = layer.conv.out_channels
            
            # Create double convolution block
            double_conv = CustomConvBlock(
                in_channels=in_channels,
                out_channels=out_channels  # Maintain original output channels
            )
            new_sequence.append(double_conv)
            
        elif 'C2f' in layer_type:
            # Keep C2f layer as is
            new_sequence.append(layer)
            
        elif 'SPPF' in layer_type:
            # Keep SPPF layer and all subsequent layers (detection head)
            while i < len(sequential):
                new_sequence.append(sequential[i])
                i += 1
            break
            
        i += 1
    
    # Update the model's sequence
    model.model.model = new_sequence
    return model

class CheckpointManager:
    def __init__(self, save_dir='checkpoints'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(save_dir, 'best_checkpoint.pt')
        self.metadata_file = os.path.join(save_dir, 'training_metadata.json')
        self.best_metrics = None
        self.load_metadata()

    def load_metadata(self):
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'iterations': 0,
                'best_epoch': 0,
                'best_metrics': None,
                'training_history': []
            }

    def is_better_performance(self, current_metrics, best_metrics):
        """
        Compare current metrics with best metrics to determine if performance improved.
        Returns True only if both precision and mAP metrics are better.
        """
        if best_metrics is None:
            return True
            
        # Get mAP metrics (using both mAP50-95 and mAP50 for robustness)
        current_map = current_metrics.get('metrics/mAP50-95(B)', 0)
        best_map = best_metrics.get('metrics/mAP50-95(B)', 0)
        
        current_map50 = current_metrics.get('metrics/mAP50(B)', 0)
        best_map50 = best_metrics.get('metrics/mAP50(B)', 0)
        
        # Get precision metrics
        current_precision = current_metrics.get('metrics/precision(B)', 0)
        best_precision = best_metrics.get('metrics/precision(B)', 0)
        
        # Check if both mAP metrics and precision have improved
        map_improved = current_map > best_map and current_map50 >= best_map50
        precision_improved = current_precision > best_precision
        
        return map_improved and precision_improved

    def save_checkpoint(self, model, metrics, epoch):
        """
        Save checkpoint only if both precision and mAP metrics have improved.
        Returns True if a new checkpoint was saved, False otherwise.
        """
        if self.is_better_performance(metrics, self.best_metrics):
            self.best_metrics = metrics
            
            # Save model weights
            torch.save({
                'model_state_dict': model.model.state_dict(),
                'epoch': epoch,
                'metrics': metrics
            }, self.checkpoint_file)
            
            # Update metadata
            self.metadata['best_epoch'] = epoch
            self.metadata['best_metrics'] = metrics
            self.save_metadata()
            
            return True
            
        print(f"\nCurrent performance not better than best checkpoint. Keeping previous weights.")
        print(f"Best mAP50-95: {self.best_metrics.get('metrics/mAP50-95(B)', 0):.4f}, "
              f"Current mAP50-95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"Best precision: {self.best_metrics.get('metrics/precision(B)', 0):.4f}, "
              f"Current precision: {metrics.get('metrics/precision(B)', 0):.4f}")
        return False

    def load_checkpoint(self, model):
        if os.path.exists(self.checkpoint_file):
            checkpoint = torch.load(self.checkpoint_file)
            
            # Modify the model first
            model = modify_yolo_model(model)
            
            try:
                # Try loading the state dict
                model.model.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError as e:
                print("Standard loading failed, attempting to fix state dict keys...")
                
                # Get the state dict from checkpoint
                state_dict = checkpoint['model_state_dict']
                
                # Get the current model state dict
                model_state_dict = model.model.state_dict()
                
                # Create a new state dict with matching keys
                new_state_dict = {}
                for name, param in model_state_dict.items():
                    if name in state_dict:
                        new_state_dict[name] = state_dict[name]
                    else:
                        print(f"Initializing new parameter: {name}")
                        new_state_dict[name] = param
                
                # Load the fixed state dict
                model.model.load_state_dict(new_state_dict)
                
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
            print(f"Best metrics - mAP50-95: {checkpoint['metrics'].get('metrics/mAP50-95(B)', 0):.4f}, "
                  f"precision: {checkpoint['metrics'].get('metrics/precision(B)', 0):.4f}")
            self.best_metrics = checkpoint['metrics']
            return checkpoint
        return None

    def save_metadata(self):
        self.metadata['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=4)

    def update_training_history(self, metrics, epoch):
        self.metadata['iterations'] += 1
        self.metadata['training_history'].append({
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        self.save_metadata()

def custom_on_train_epoch_end(trainer):
    """Custom callback function for end of training epoch"""
    metrics = trainer.metrics
    checkpoint_manager = trainer.custom_checkpoint_manager
    if checkpoint_manager.save_checkpoint(trainer.model, metrics, trainer.epoch):
        print(f"\nNew best model saved! Epoch: {trainer.epoch}, mAP50-95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
    checkpoint_manager.update_training_history(metrics, trainer.epoch)

def train_with_checkpoints(model, training_args, checkpoint_manager=None):
    """
    Train the model with checkpoint management
    """
    if checkpoint_manager is None:
        checkpoint_manager = CheckpointManager()
    
    # Load best checkpoint if exists
    checkpoint_manager.load_checkpoint(model)
    
    # Add checkpoint manager to model's trainer
    def on_trainer_init(trainer):
        trainer.custom_checkpoint_manager = checkpoint_manager
        trainer.add_callback('on_train_epoch_end', custom_on_train_epoch_end)
    
    # Add the initialization callback
    model.add_callback('on_train_start', on_trainer_init)
    
    # Train the model
    results = model.train(**training_args)
    return results

if __name__ == "__main__":
    from ultralytics import YOLO
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager()
    
    # Load base model
    model = YOLO('yolov8n.pt')
    
    # Modify backbone
    modified_model = modify_yolo_model(model)
    
    # Training configuration
    training_args = {
        'data': 'coco128.yaml',
        'epochs': 20,
        'batch': 16,
        'patience': 10,
        'amp': False,
        'optimizer': 'AdamW',
        'weight_decay': 0.0001,
        'lr0': 0.002,
        'lrf': 0.001,
        'warmup_epochs': 5,
        'cos_lr': True,
        'augment': True,
        'mixup': 0.1,
        'mosaic': 1.0,
        'close_mosaic': 10,
        'label_smoothing': 0.1
    }
    
    # Train the modified model with checkpoint management
    results = train_with_checkpoints(modified_model, training_args, checkpoint_manager)