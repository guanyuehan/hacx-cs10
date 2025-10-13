# Evaluation System - Quick Guide

Simple evaluation system for satellite imagery classification models.

## Folder Structure

```
evaluation/
├── app/                          # Training, Inference code and dockerfile
│
├── test_data/                    # Test images (.tif files)
│   ├── cloud_1.tif
│   ├── haze_100.tif
│   ├── smoke_1005.tif
│   └── ...
│
├── weights/                      # Model weights
│   ├── resnet50_best.pth        # Trained model
│   └── resnet50_history.json    # Training history
│
├── output/                       # Results (OUTPUT)
│   ├── predictions.csv          # Model predictions
│   ├── eval_result.csv          # F1 & parameters
│   └── timing_results.json      # Timing statistics
│
├── complete_evaluation.sh        # Main evaluation script
├── compute_metrics.py           # Compute F1 & count parameters
├── run_docker_evaluation.py     # Docker evaluation with timing
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Quick Start

### Prerequisites

Install Python dependencies:
```bash
pip install -r requirements.txt
```

### Running Evaluation

```bash
# Make script executable
chmod +x complete_evaluation.sh

# Run evaluation
./complete_evaluation.sh
```

### Interactive Prompts

The script will ask:
```
Enter Docker image name (default: satellite-classifier:latest): [type your image name]
Enter predictions CSV filename (default: predictions.csv): [press Enter or type name]
```

Or provide as arguments:
```bash
./complete_evaluation.sh YOUR_IMAGE:TAG predictions.csv
```

## What It Does

The script performs 2 steps:

### Step 1: Docker Evaluation
- Starts Docker container
- Runs inference on all test images
- Monitors predictions.csv in real-time
- Measures timing

### Step 2: Compute Metrics
- Calculates weighted F1 score
- Counts model parameters
- Saves results to eval_result.csv

## Output

At the end, you'll see:

```
======================================================================
EVALUATION RESULTS
======================================================================
Model Name: resnet50_best.pth
Number of Test Images: 1247
Throughput: 114.15 files/second
Weighted F1 Score: 0.8943
Model Size: 24,611,832 parameters
======================================================================
Results saved to: ./output/timing_results.json
======================================================================
```

## Output Files

After running, check these files:

1. **`output/predictions.csv`** - Model predictions
   ```csv
   file_name,predicted_class,actual_class
   cloud_1.tif,2,2
   haze_100.tif,1,1
   smoke_1005.tif,0,0
   ```

2. **`output/timing_results.json`** - Detailed timing and metrics
   ```json
   {
     "image_name": "YOUR_IMAGE:TAG",
     "num_test_files": 1247,
     "weighted_f1_score": 0.8943,
     "model_parameters": 24611832,
     "full_evaluation": {
       "total_time": 12.59,
       "time_to_last_prediction": 11.53
     }
   }
   ```

3. **`output/eval_result.csv`** - Metrics summary
   ```csv
   model_name,weighted_f1_score,num_parameters
   resnet50_best.pth,0.8943,24611832
   ```

## Class Mapping

- **0** = smoke (includes wildfire)
- **1** = haze
- **2** = normal (cloud, land, seaside, dust)

## Manual Usage

If you prefer to run steps separately:

### Step 1: Run Docker Evaluation
```bash
python run_docker_evaluation.py \
    --image YOUR_IMAGE:TAG \
    --test-data ./test_data \
    --output ./output \
    --weights ./weights \
    --predictions-file predictions.csv
```

### Step 2: Compute Metrics
```bash
python compute_metrics.py \
    --predictions ./output/predictions.csv \
    --weights ./weights \
    --output ./output/eval_result.csv
```

## Troubleshooting

### Docker image not found
```bash
cd app
./build_docker.sh YOUR_IMAGE TAG
cd ..
```

### Permission denied
```bash
chmod +x complete_evaluation.sh
chmod +x app/build_docker.sh
```

### No test data
Ensure test_data/ contains .tif files:
```bash
ls test_data/*.tif | head
```

### CUDA/GPU errors
The system automatically falls back to CPU if GPU is unavailable.

## Advanced Options

### Custom paths
```bash
python run_docker_evaluation.py \
    --image YOUR_IMAGE:TAG \
    --test-data /path/to/test/data \
    --output /path/to/output \
    --weights /path/to/weights
```

### Different model
```bash
python compute_metrics.py \
    --predictions ./output/predictions.csv \
    --weights /path/to/different/weights
```