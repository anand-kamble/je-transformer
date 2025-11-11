#!/bin/bash

# Log everything
exec > >(tee /var/log/training-startup.log)
exec 2>&1

echo "Starting training at $(date)"

# Wait for GPU drivers if needed
sleep 30

# Run training as your user (not root)
sudo -u akamble bash << 'EOF'
cd /home/akamble/je-transformer
source venv/bin/activate

python train_small.py \
    --encoder "prajjwal1/bert-tiny" \
    --hidden-dim 384 \
    --max-lines 40 \
    --batch-size 32 \
    --epochs 30 \
    --lr 3e-4 \
    --output-dir "gs://dev-rai-files/outputs"

EOF

echo "Training completed at $(date)"

# Shutdown
sleep 10
sudo poweroff
