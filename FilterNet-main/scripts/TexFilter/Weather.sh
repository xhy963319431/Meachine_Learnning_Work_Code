export CUDA_VISIBLE_DEVICES=0

model_name=Transformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/ \
  --data_path data_ml.csv \
  --model_id ml_96_240\
  --model $model_name \
  --data custom \
  --features MS \
  --enc_in 15 \
  --dec_in 15 \
  --c_out 15 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 240 \
  --embed_size 128 \
  --hidden_size 128 \
  --dropout 0 \
  --train_epochs 20 \
  --batch_size 128 \
  --patience 6 \
  --learning_rate 0.01 \
  --des 'Exp' \
  --itr 1
