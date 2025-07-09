mkdir -p ../checkpoints_all_results
mkdir -p ../tensorboard


uv run main.py --model=lenet --dataset=mnist --prune \
    --train_lr=1e-3 --max_train_lr=0.1 \
    --pruner=causalpruner --total_prune_amount=0.993 --num_prune_iterations=30 \
    --checkpoint_dir="../checkpoints_all_results"

