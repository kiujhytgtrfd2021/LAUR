GPU_ID=$1

CUDA_VISIBLE_DEVICES=$GPU_ID python train_sentiment.py --approach LAUR --alpha 1 --beta 1 --gamma 0.5 --logname '1_LAUR' --seed 1 --tasks_order 1
CUDA_VISIBLE_DEVICES=$GPU_ID python train_sentiment.py --approach LAUR --alpha 1 --beta 1 --gamma 0.5 --logname '2_LAUR' --seed 2 --tasks_order 1
CUDA_VISIBLE_DEVICES=$GPU_ID python train_sentiment.py --approach LAUR --alpha 1 --beta 1 --gamma 0.5 --logname '3_LAUR' --seed 3 --tasks_order 1
