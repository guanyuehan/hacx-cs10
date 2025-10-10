import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import os
import glob
import torch
import pickle
import struct
import argparse


def compute_weighted_f1(predictions_file):
    # Read predictions
    df = pd.read_csv(predictions_file)
    
    # Remove any empty rows
    df = df.dropna()
    
    # Get true and predicted labels
    y_true = df['actual_class'].values
    y_pred = df['predicted_class'].values
    
    # Compute weighted F1 score
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    return f1_weighted


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


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Compute evaluation metrics from model predictions.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-p', '--predictions',
        type=str,
        default='./output/predictions.csv',
        help='Path to predictions CSV file with columns: file_name, predicted_class, actual_class'
    )
    parser.add_argument(
        '-w', '--weights',
        type=str,
        default='./weights',
        help='Directory containing model weight files'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='./output/eval_result.csv',
        help='Path to output CSV file for evaluation results'
    )
    
    args = parser.parse_args()
    
    # Paths
    predictions_file = args.predictions
    weights_dir = args.weights
    output_file = args.output
    
    # Compute weighted F1 score
    print("Computing weighted F1 score...")
    f1_weighted = compute_weighted_f1(predictions_file)
    print(f"Weighted F1 Score: {f1_weighted:.4f}")
    
    # Count model parameters
    print(f"\nCounting model parameters from {weights_dir}...")
    model_params = count_model_parameters(weights_dir)
    
    # Prepare results
    results = []
    
    if model_params:
        for model_name, param_count in model_params.items():
            results.append({
                'model_name': model_name,
                'weighted_f1_score': f1_weighted,
                'num_parameters': param_count
            })
        print(f"\nFound {len(model_params)} model(s)")
    else:
        # If no models found, still output the F1 score
        results.append({
            'model_name': 'N/A',
            'weighted_f1_score': f1_weighted,
            'num_parameters': 'N/A'
        })
        print("\nNo model weight files found in weights directory")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Display results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60)


if __name__ == '__main__':
    main()

