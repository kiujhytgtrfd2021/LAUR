GPU_ID=$1

CUDA_VISIBLE_DEVICES=$GPU_ID python train_text.py --alpha 1 --beta 1 --gamma 1 --approach LAUR --logname 'order1_seed42' --seed 42 --tasks_order 1 --min_import 1.05 --max_import 3.0
CUDA_VISIBLE_DEVICES=$GPU_ID python train_text.py --alpha 1 --beta 1 --gamma 1 --approach LAUR --logname 'order2_seed42' --seed 42 --tasks_order 2 --min_import 1.05 --max_import 3.0
CUDA_VISIBLE_DEVICES=$GPU_ID python train_text.py --alpha 1 --beta 1 --gamma 1 --approach LAUR --logname 'order3_seed42' --seed 42 --tasks_order 3 --min_import 1.05 --max_import 3.0
CUDA_VISIBLE_DEVICES=$GPU_ID python train_text.py --alpha 1 --beta 1 --gamma 1 --approach LAUR --logname 'order4_seed42' --seed 42 --tasks_order 4 --min_import 1.05 --max_import 3.0