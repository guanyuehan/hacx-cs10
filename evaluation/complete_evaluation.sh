#!/bin/bash
# Complete evaluation pipeline: Build Docker, Run inference, Compute metrics

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Ask for image name and predictions file if not provided as arguments
if [ -z "$1" ]; then
    echo "=========================================="
    echo "Complete Docker Evaluation Pipeline"
    echo "=========================================="
    echo ""
    read -p "Enter Docker image name (default: satellite-classifier:latest): " IMAGE_NAME
    IMAGE_NAME="${IMAGE_NAME:-satellite-classifier:latest}"
    
    read -p "Enter predictions CSV filename (default: predictions.csv): " PREDICTIONS_FILE
    PREDICTIONS_FILE="${PREDICTIONS_FILE:-predictions.csv}"
else
    IMAGE_NAME="$1"
    PREDICTIONS_FILE="${2:-predictions.csv}"
fi

echo ""
echo "=========================================="
echo "Complete Docker Evaluation Pipeline"
echo "=========================================="
echo "Image: $IMAGE_NAME"
echo "Predictions file: $PREDICTIONS_FILE"
echo ""

# Step 1: Run Docker evaluation with timing
echo "Step 1/2: Running Docker evaluation..."
python run_docker_evaluation.py \
    --image "$IMAGE_NAME" \
    --test-data ./test_data \
    --output ./output \
    --weights ./weights \
    --predictions-file "$PREDICTIONS_FILE"
echo "Evaluation complete"
echo ""

# Step 2: Compute metrics
echo "Step 2/2: Computing metrics..."
python compute_metrics.py \
    --predictions "./output/$PREDICTIONS_FILE" \
    --weights ./weights \
    --output ./output/eval_result.csv
echo "Metrics computed"
echo ""

# Display results
echo ""
echo "=========================================="
echo "EVALUATION RESULTS"
echo "=========================================="

# Extract metrics from JSON and CSV files
MODEL_NAME=$(python -c "import json; print(json.load(open('./output/timing_results.json')).get('model_name', 'N/A'))" 2>/dev/null || echo "N/A")
NUM_IMAGES=$(python -c "import json; print(json.load(open('./output/timing_results.json'))['num_test_files'])" 2>/dev/null || echo "N/A")
TIME_TO_LAST=$(python -c "import json; d=json.load(open('./output/timing_results.json')); print(f\"{d['num_test_files']/d['full_evaluation']['time_to_last_prediction']:.2f}\")" 2>/dev/null || echo "N/A")
F1_SCORE=$(python -c "import pandas as pd; print(f\"{pd.read_csv('./output/eval_result.csv')['weighted_f1_score'].iloc[0]:.4f}\")" 2>/dev/null || echo "N/A")
NUM_PARAMS=$(python -c "import pandas as pd; print(f\"{int(pd.read_csv('./output/eval_result.csv')['num_parameters'].iloc[0]):,}\")" 2>/dev/null || echo "N/A")

echo "Model Name: $MODEL_NAME"
echo "Number of Test Images: $NUM_IMAGES"
echo "Throughput: $TIME_TO_LAST files/second"
echo "Weighted F1 Score: $F1_SCORE"
echo "Model Size: $NUM_PARAMS parameters"
echo "=========================================="

