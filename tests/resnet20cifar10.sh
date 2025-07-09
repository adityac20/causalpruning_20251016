#!/bin/bash

PRUNE_AMT_VALS=(0.1 0.3 0.5 0.7 0.9 0.95 0.98)
mkdir -p ../tensorboard
mkdir -p ../checkpoints/resnet20_cifar10

for i in {1..3}; do
  mkdir -p "../checkpoints/resnet20_cifar10/resnet20_cifar10_${i}"

  # No pruning
  uv run main.py --model=resnet20 --dataset=cifar10 --no-prune \
    --num_train_epochs=400 --checkpoint_dir="../checkpoints/resnet20_cifar10/resnet20_cifar10_${i}"

  for PRUNE_AMT in "${PRUNE_AMT_VALS[@]}"; do    
      uv run main.py --model=resnet20 --dataset=cifar10 --prune \
    --pruner=causalpruner --total_prune_amount=$PRUNE_AMT \
    --num_pre_prune_epochs=10 --num_prune_iterations=10 \
    --num_train_epochs_before_pruning=10 --num_prune_epochs=1 \
    --num_train_epochs=300 --checkpoint_dir="../checkpoints/resnet20_cifar10/resnet20_cifar10_${i}"

    uv run main.py --model=resnet20 --dataset=cifar10 --prune \
    --pruner=magpruner --total_prune_amount=$PRUNE_AMT \
    --num_pre_prune_epochs=10 --num_prune_iterations=10 \
    --num_train_epochs_before_pruning=10 --num_prune_epochs=1 \
    --num_train_epochs=300 --checkpoint_dir="../checkpoints/resnet20_cifar10/resnet20_cifar10_${i}"

    uv run main.py --model=resnet20 --dataset=cifar10 --prune \
    --pruner=causalpruner --total_prune_amount=$PRUNE_AMT \
    --num_pre_prune_epochs=10 --num_prune_iterations=1 \
    --num_train_epochs_before_pruning=10 --num_prune_epochs=1 \
    --num_train_epochs=300 --checkpoint_dir="../checkpoints/resnet20_cifar10/resnet20_cifar10_${i}"
  done
done