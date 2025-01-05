#!/bin/bash

# Script to reproduce results

# 切换到上一级目录
cd ..

# 打印当前目录，确认是否切换成功
echo "当前工作目录: $(pwd)"

# pip
pip install pandas
pip install scikit-learn
pip install gym

# 启动多个训练任务，分别保存到不同的文件夹
for i in {1..5}; do
    # 启动训练任务，并将输出保存在不同的文件夹中
    python train.py \
    --seq_num ${i} \
    --env "SunPointFaultSatellite" \
    --random_seed $i \
    --hidden_dim 256 \
    --dyn_hidden_size 64 128 \
    --dyn_net_path "models/dynamic_net/attitude_dynamics_model.pth" \
    --max_ep_len 2000 \
    --max_training_timesteps 2000000  &
done

# 等待所有任务完成
wait
