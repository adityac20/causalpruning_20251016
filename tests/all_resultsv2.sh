mkdir -p ../checkpoints_all_results
mkdir -p ../tensorboard


uv run main.py --model=vgglike --dataset=cifar10 --prune \
    --train_lr=1e-3 --max_train_lr=0.1 \
    --pruner=causalpruner --total_prune_amount=0.98 --num_prune_iterations=30 \
    --checkpoint_dir="../checkpoints_all_results"

uv run main.py --model=vgglike --dataset=cifar10 --no-prune --checkpoint_dir="../checkpoints_all_results"


uv run main.py --model=vgglike --dataset=tinyimagenet --prune --train_lr=1e-3 --max_train_lr=0.1  --pruner=causalpruner --total_prune_amount=0.98 --num_prune_iterations=30  --checkpoint_dir="../checkpoints_all_results" --num_train_epochs_before_pruning=60 --num_pre_prune_epochs=120 --num_prune_epochs=10

uv run main.py --model=vgglike --dataset=tinyimagenet --no-prune --checkpoint_dir="../checkpoints_all_results"

uv run main.py --model=vgglike_trained --dataset=cifar10 --prune --train_lr=1e-3 --max_train_lr=0.1  --pruner=causalpruner --total_prune_amount=0.98 --num_prune_iterations=30  --checkpoint_dir="../checkpoints_all_results" --num_train_epochs_before_pruning=60 --num_pre_prune_epochs=10 --num_prune_epochs=10 --causal_pruner_init_lr=1e-4

uv run main.py --model=resnet50_trained --dataset=imagenet --prune --train_lr=1e-3 --max_train_lr=0.1  --pruner=causalpruner --total_prune_amount=0.8 --num_prune_iterations=1  --checkpoint_dir="../checkpoints_all_results" --num_train_epochs_before_pruning=1 --num_pre_prune_epochs=1 --num_prune_epochs=1

uv run main.py --model=mobilenet_trained --dataset=imagenet --prune --train_lr=1e-3 --max_train_lr=0.1  --pruner=causalpruner --total_prune_amount=0.8 --num_prune_iterations=1  --checkpoint_dir="../checkpoints_all_results2" --num_train_epochs_before_pruning=1 --num_pre_prune_epochs=1 --num_prune_epochs=1


uv run main.py --model=resnet20_trained --dataset=cifar10 --prune --train_lr=1e-3 --max_train_lr=0.1  --pruner=causalpruner --total_prune_amount=0.98 --num_prune_iterations=40  --checkpoint_dir="../checkpoints_all_results" --num_train_epochs_before_pruning=60 --num_pre_prune_epochs=10 --num_prune_epochs=1

uv run main.py --model=resnet20_trained --dataset=cifar10 --prune --train_lr=1e-3 --max_train_lr=0.1  --pruner=magpruner --total_prune_amount=0.9 --num_prune_iterations=1  --checkpoint_dir="../checkpoints_all_results" --num_train_epochs_before_pruning=1 --num_pre_prune_epochs=1 --num_prune_epochs=1

uv run main.py --model=resnet20_trained --dataset=cifar10 --prune --train_lr=1e-3 --max_train_lr=0.1  --pruner=causalpruner --total_prune_amount=0.9 --num_prune_iterations=1  --checkpoint_dir="../checkpoints_all_results" --num_train_epochs_before_pruning=1 --num_pre_prune_epochs=1 --num_prune_epochs=1

 uv run main.py --model=resnet20 --dataset=cifar10 --prune --train_lr=1e-3 --max_train_lr=0.1  --pruner=causalpruner --total_prune_amount=0.9 --num_prune_iterations=1  --checkpoint_dir="../checkpoints_all_results2" --num_train_epochs_before_pruning=1 --num_pre_prune_epochs=30  --num_prune_epochs=1 --causal_pruner_max_iter=50 --causal_pruner_num_iter_no_change=5