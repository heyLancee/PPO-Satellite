#!/bin/bash

# Script to reproduce results

# 切换到上一级目录
cd ..

# 打印当前目录，确认是否切换成功
echo "当前工作目录: $(pwd)"

# pip
pip install pandas
pip install scikit-learn

# 启动训练任务，并将输出保存在不同的文件夹中
python DynamicNet.py \
--hidden_size 128 \
--model_save_path "./models/dynamic_net/attitude_dynamics_model.pth" \
--data_path "./sample_data/data.csv" \
--test_size_per 0.2 \
--num_epochs 500 \
--batch_size 128 \
--lr 0.001 \
--seed 0 \
--save_model &

# 等待所有任务完成
wait
