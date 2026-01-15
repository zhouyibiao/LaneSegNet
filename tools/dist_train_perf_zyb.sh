#!/usr/bin/env bash
# set -x

# 基础配置
CONFIG_FILE="/data/yibiao.zhou/LaneSegNet-Adaption/LaneSegNet/projects/configs/lanesegnet_r50_8x1_24e_olv2_subset_A_perf.py"
SAMPLES_PER_GPU=(8 10 12)
GPU_CARDS=(8)
LOG_ROOT_DIR="work_dirs/training_performances_20260113"
# 定义CHANNEL_LAST的取值（0/1）
CHANNEL_LAST_VALUES=(1)
rm -r $LOG_ROOT_DIR
mkdir -p $LOG_ROOT_DIR
MASTER_PORT=28510

for gpu_num in "${GPU_CARDS[@]}"; do
    echo "========================================"
    echo "开始测试 GPU 卡数: $gpu_num"
    echo "----------------------------------------"
    
    if [ $gpu_num -eq 8 ]; then
        GPU_DEVICE="0,1,2,3,4,5,6,7"
        echo "  已设置8卡模式，设备号：${GPU_DEVICE}"
    else
        GPU_DEVICE="7"
        echo "  已设置单卡模式，设备号：${GPU_DEVICE}"
    fi
    
    for channel_last in "${CHANNEL_LAST_VALUES[@]}"; do
        echo "  ----------------------------------------"
        echo "  开始测试 CHANNEL_LAST: ${channel_last}"
        echo "  ----------------------------------------"
        
        for samples in "${SAMPLES_PER_GPU[@]}"; do
            # 1. 定义日志路径（融入channel-last-${channel_last}）
            LOG_DIR="${LOG_ROOT_DIR}/gpu-num-${gpu_num}_channel-last-${channel_last}_samples-per-gpu_${samples}"
            mkdir -p $LOG_DIR
            TRAIN_LOG="${LOG_DIR}/gpu-num-${gpu_num}_channel-last-${channel_last}_samples-per-gpu_${samples}.log"
            touch ${TRAIN_LOG}
            MEM_LOG="${LOG_DIR}/gpu-num-${gpu_num}_channel-last-${channel_last}_samples-per-gpu_${samples}_mem.log"
            touch ${MEM_LOG}

            # 2. 修改配置文件中的 samples_per_gpu
            wokers=${samples}
            python /data/yibiao.zhou/LaneSegNet-Adaption/LaneSegNet/tools/set_batch_size.py $CONFIG_FILE $samples $wokers

            # 3. 后台执行训练命令，记录进程PID（导出CHANNEL_LAST环境变量）
            echo "    启动训练：GPU卡数=$gpu_num | CHANNEL_LAST=${channel_last} | 单卡样本数=$samples | 设备号=${GPU_DEVICE}"
            # 关键：导出CHANNEL_LAST环境变量，动态传入GPU_DEVICE
            MUSA_VISIBLE_DEVICES=${GPU_DEVICE} CHANNEL_LAST=${channel_last} python -m torch.distributed.run \
                --nproc_per_node=${gpu_num} \
                --master_port=${MASTER_PORT} \
                tools/train_zyb.py $CONFIG_FILE \
                --launcher pytorch \
                --work-dir $LOG_DIR > ${TRAIN_LOG} 2>&1 &
            TRAIN_PID=$!

            # 4. 并行启动显存监控（每秒执行一次，直到训练进程结束）
            echo "    启动显存监控，日志保存到：${MEM_LOG}"
            # 用while循环替代watch，更易管控生命周期（避免watch成为孤儿进程）
            (
                while kill -0 ${TRAIN_PID} 2>/dev/null; do  # 检查训练进程是否存活
                    mthreads-gmi | grep S5000 >> ${MEM_LOG}
                    echo "----------------------------------------" >> ${MEM_LOG}
                    sleep 1
                done
            ) &
            MEM_MONITOR_PID=$!

            wait ${TRAIN_PID}
            wait ${MEM_MONITOR_PID}  # 确保显存监控也正常退出
            echo "    训练+显存监控完成：GPU卡数=$gpu_num | CHANNEL_LAST=${channel_last} | 单卡样本数=$samples"
            
            # 6. 端口号自增（避免后续循环端口冲突）
            MASTER_PORT=$((MASTER_PORT + 1))
            sleep 30
        done
        
        echo "  CHANNEL_LAST ${channel_last} 测试完成（GPU卡数 ${gpu_num}）"
        echo "  "
        sleep 30  # CHANNEL_LAST切换间隔
    done
    
    echo "GPU 卡数 $gpu_num 测试完成"
    echo ""
    sleep 30  # 卡数切换间隔
done

echo "所有配置测试完成！"