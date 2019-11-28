#!/usr/bin/env bash

FOLDIR="../data/test_data"
python ./run.py \
--model_type bert \
--task_name funcup \
--model_name_or_path bert_chinese_tw \
--do_test \
--do_lower_case \
--data_dir $FOLDIR \
--learning_rate 5e-5 \
--num_train_epochs 4 \
--max_seq_length 512 \
--output_dir model \
--per_gpu_eval_batch_size=2 \
--per_gpu_train_batch_size=2 \
--gradient_accumulation_steps 2 \
--overwrite_output