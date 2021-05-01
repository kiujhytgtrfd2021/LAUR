CUDA_VISIBLE_DEVICES=6 python ./SQuAD/SQUAD_train.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name ./SQuAD/squad_preprocessing.py \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --eval_accumulation_steps 10\
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./result/SQuAD \

CUDA_VISIBLE_DEVICES=6 python3 WikiSQL/sqlova/train_decoder_layer.py --seed 1 --bS 16 --accumulate_gradients 1 --bert_type_abb uS --fine_tune --lr 0.001 --lr_bert 5e-5 --load_model --old_model_path best_model.pt --use_regular --tepoch 5 --do_train

CUDA_VISIBLE_DEVICES=6 python SST/SST_train.py \
  --model_name_or_path bert-base-uncased \
  --train_file data/SST/train.csv \
  --validation_file data/SST/test.csv \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --output_dir ./result/SST \
  --overwrite_output_dir \
  --load_model \
  --old_model_path best_WikiSQL_decoder_5.pt

CUDA_VISIBLE_DEVICES=6 python ./SRL/SRL_train.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name ./SRL/SRL_preprocessing.py \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --eval_accumulation_steps 10\
  --learning_rate 5e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./result/SRL \
  --load_model \
  --old_model_path best_SST.pt