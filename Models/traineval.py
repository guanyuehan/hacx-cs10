import numpy as np
import glob
import os
import pickle
import struct
import torch
from torch import nn
from torchmetrics import ConfusionMatrix
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

@torch.no_grad()
def test_model(test_dataloader: DataLoader, model: nn.Module, loss_fn=nn.CrossEntropyLoss(), DEVICE="cpu"):
    # confusion matrix
    confmat = ConfusionMatrix(num_classes=3, task="multiclass").to(DEVICE)
    total_loss = 0.0

    model.eval()
    model.to(DEVICE)
    for data_batch in tqdm(test_dataloader):
        imgs, labels = data_batch
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        total_loss += loss_fn(outputs, labels).item()
        preds = torch.argmax(outputs, dim=1)
        confmat.update(preds, labels)

    confmat = confmat.compute().cpu().numpy()
    accuracy = np.trace(confmat) / np.sum(confmat)
    loss = torch.tensor(total_loss / len(test_dataloader))

    precision = confmat[1, 1] / np.sum(confmat[:, 1])
    recall = confmat[1, 1] / np.sum(confmat[1, :])
    f1_score = 2 * (precision * recall) / (precision + recall)
    return loss, confmat, accuracy, precision, recall, f1_score

def train_model_epoch(train_loader: DataLoader, test_loader: DataLoader, model: nn.Module, criterion, optimizer, writer: SummaryWriter, DEVICE="cpu", epoch=0):
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
    mean_train_loss = 0.0
    for idx, (images, labels) in loop:
        model.train()
        optimizer.zero_grad()

        images, labels = images.to(DEVICE), labels.to(DEVICE)
        pred_logits = model(images)
        loss = criterion(pred_logits, labels)
        mean_train_loss += loss.item()

        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=f"{loss.item():.4f}")

    mean_train_loss /= len(train_loader)
        
    loss, confmat, accuracy, precision, recall, f1_score = test_model(test_loader, model, DEVICE=DEVICE)
    writer.add_scalar('Train/Loss', mean_train_loss, epoch)
    writer.add_scalar('Test/Loss', loss.item(), epoch * len(train_loader) + idx)
    writer.add_scalar('Test/Accuracy', accuracy, epoch * len(train_loader) + idx)

def parse_gguf_metadata(file_path):
    try:
        with open(file_path, 'rb') as f:
            # Read magic number (4 bytes)
            magic = f.read(4)
            if magic != b'GGUF':
                return None
            
            # Read version (4 bytes, uint32)
            version = struct.unpack('<I', f.read(4))[0]
            
            # Read tensor count (8 bytes, uint64 for v2+, uint32 for v1)
            if version >= 2:
                tensor_count = struct.unpack('<Q', f.read(8))[0]
                metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
            else:
                tensor_count = struct.unpack('<I', f.read(4))[0]
                metadata_kv_count = struct.unpack('<I', f.read(4))[0]
            
            # Try to get parameter count from file size as estimation
            # (More accurate parsing would require traversing the entire metadata)
            file_size = os.path.getsize(file_path)
            
            # Rough estimation: assuming 4 bytes per parameter for FP32
            # For quantized models, this will be less, but we'll use tensor_count
            # as a rough indicator
            return f"~{tensor_count} tensors (GGUF)"
            
    except Exception as e:
        print(f"Warning: Could not parse GGUF file: {str(e)}")
        return None

def count_model_parameters(weights_dir):
    model_params = {}
    
    # Look for common model weight file formats
    weight_patterns = ['*.pth', '*.pt', '*.pkl', '*.h5', '*.weights', '*.gguf']
    
    for pattern in weight_patterns:
        weight_files = glob.glob(os.path.join(weights_dir, pattern))
        
        for weight_file in weight_files:
            model_name = os.path.basename(weight_file)
            
            try:
                # Try loading as PyTorch model
                if weight_file.endswith(('.pth', '.pt')):
                    # Try with weights_only=True first (safer), fall back to False for custom models
                    try:
                        checkpoint = torch.load(weight_file, map_location='cpu', weights_only=True)
                    except Exception:
                        checkpoint = torch.load(weight_file, map_location='cpu', weights_only=False)
                    
                    total_params = 0
                    
                    # Handle different checkpoint formats
                    if isinstance(checkpoint, dict):
                        # Try common keys for state dict
                        state_dict = None
                        for key in ['model_state_dict', 'state_dict', 'model']:
                            if key in checkpoint:
                                state_dict = checkpoint[key]
                                break
                        
                        # If no common key found, assume the dict itself is the state dict
                        if state_dict is None:
                            state_dict = checkpoint
                        
                        # Count parameters from state dict
                        if isinstance(state_dict, dict):
                            total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
                        elif hasattr(state_dict, 'parameters'):
                            # If it's a model object stored in the dict
                            total_params = sum(p.numel() for p in state_dict.parameters())
                    
                    elif hasattr(checkpoint, 'parameters'):
                        # If the checkpoint itself is a model
                        total_params = sum(p.numel() for p in checkpoint.parameters())
                    
                    model_params[model_name] = total_params if total_params > 0 else 'N/A'
                
                # Try loading as pickle
                elif weight_file.endswith('.pkl'):
                    with open(weight_file, 'rb') as f:
                        model = pickle.load(f)
                    
                    # Try to count parameters if it's a PyTorch model
                    if hasattr(model, 'parameters'):
                        total_params = sum(p.numel() for p in model.parameters())
                        model_params[model_name] = total_params
                    else:
                        model_params[model_name] = 'N/A'
                
                # Try loading as GGUF
                elif weight_file.endswith('.gguf'):
                    param_info = parse_gguf_metadata(weight_file)
                    if param_info:
                        model_params[model_name] = param_info
                    else:
                        model_params[model_name] = 'N/A (GGUF)'
                
            except Exception as e:
                print(f"Warning: Could not load {model_name}: {str(e)}")
                model_params[model_name] = 'Error'
    
    return model_params