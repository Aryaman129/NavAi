#!/bin/bash
# Quick training script for NavAI
cd ml
python train_local.py --model-type cnn --batch-size 16 --num-epochs 50
