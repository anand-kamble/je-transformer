#!/bin/bash

# Set up logging that works with JupyterLab
LOG_FILE="/home/akamble/training_$(date +%Y%m%d_%H%M%S).log"

echo "Starting training at $(date)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"

# Change to project directory
cd /home/akamble/je-transformer
source venv/bin/activate

# Run training (no sudo needed since you're already akamble)
python train_small.py \
    --encoder "prajjwal1/bert-tiny" \
    --hidden-dim 384 \
    --max-lines 40 \
    --batch-size 32 \
    --epochs 30 \
    --lr 3e-4 \
    --output-dir "gs://dev-rai-files/outputs" \
    --limit 200 \
    2>&1 | tee -a "$LOG_FILE"

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully at $(date)" | tee -a "$LOG_FILE"
    
    # Upload log to GCS
    gsutil cp "$LOG_FILE" gs://dev-rai-files/logs/
    
    echo "Shutting down instance in 30 seconds..." | tee -a "$LOG_FILE"
    sleep 30
    
    # Shutdown the instance
    sudo poweroff
else
    echo "Training failed with exit code $TRAIN_EXIT_CODE at $(date)" | tee -a "$LOG_FILE"
    gsutil cp "$LOG_FILE" gs://dev-rai-files/logs/
    # Don't shutdown on failure
    exit $TRAIN_EXIT_CODE
fi
