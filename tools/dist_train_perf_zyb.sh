#!/usr/bin/env bash
# set -x

# 基础配置
WORKSPACE_DIR="/data/yibiao.zhou/LaneSegNet-Adaption/LaneSegNet/"
CONFIG_FILE="${WORKSPACE_DIR}/projects/configs/lanesegnet_r50_8x1_24e_olv2_subset_A_perf.py"
SAMPLES_PER_GPU=(1 2 4 8 10 12)
GPU_CARDS=(1 8)
LOG_ROOT_DIR="${WORKSPACE_DIR}/work_dirs/training_performances_20260116/"  
rm -rf "${LOG_ROOT_DIR}"
mkdir -p "${LOG_ROOT_DIR}"
MASTER_PORT=28510

if command -v nvidia-smi &> /dev/null; then
    platform="nv"
elif command -v mthreads-gmi &> /dev/null; then
    platform="musa"
else
    platform="unknown"
fi
echo "[INFO] Detected hardware: $platform"

for gpu_num in "${GPU_CARDS[@]}"; do
    echo "========================================"
    echo "开始测试 GPU 卡数: $gpu_num"
    echo "----------------------------------------"
    
    if [ $gpu_num -eq 1]; then
        if [ "${platform}" = "nv" ]; then
            export CUDA_VISIBLE_DEVICES=1
        elif [ "${platform}" = "musa" ]; then
            export MUSA_VISIBLE_DEVICES=1
        else
            echo "❌ 不支持的平台类型：${platform}，仅支持 musa / nv"
            exit 1
        fi
        echo "已设置8卡模式"
    else
        if [ "${platform}" = "nv" ]; then
            export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        elif [ "${platform}" = "musa" ]; then
            export MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        else
            echo "❌ 不支持的平台类型：${platform}，仅支持 musa / nv"
            exit 1
        fi
        echo "已设置单卡模式"
    fi

    for samples in "${SAMPLES_PER_GPU[@]}"; do
        LOG_DIR="${LOG_ROOT_DIR}/gpu-num-${gpu_num}_samples-per-gpu_${samples}"
        mkdir -p "${LOG_DIR}"
        TRAIN_LOG="${LOG_DIR}/gpu-num-${gpu_num}_samples-per-gpu_${samples}.log"
        touch "${TRAIN_LOG}"
        MEM_LOG="${LOG_DIR}/gpu-num-${gpu_num}_samples-per-gpu_${samples}_mem.log"
        touch "${MEM_LOG}"

        wokers=${samples}
        # wokers=8
        python3 "${WORKSPACE_DIR}/tools/set_batch_size.py" $CONFIG_FILE $samples $wokers

        echo "    启动训练：GPU卡数=$gpu_num | 单卡样本数=$samples"
        python3 -m torch.distributed.run \
            --nproc_per_node=${gpu_num} \
            --master_port=${MASTER_PORT} \
            ./tools/train_zyb.py $CONFIG_FILE \
            --launcher pytorch \
            --work-dir $LOG_DIR > ${TRAIN_LOG} 2>&1 &
        TRAIN_PID=$!

        echo "    启动显存监控，日志保存到：${MEM_LOG}"
        (
            while kill -0 ${TRAIN_PID} 2>/dev/null; do  # 检查训练进程是否存活
                if [ "${platform}" = "nv" ]; then
                    nvidia-smi | grep 81920MiB >> ${MEM_LOG}
                elif [ "${platform}" = "musa" ]; then
                    mthreads-gmi | grep S5000 >> ${MEM_LOG}
                else
                    echo "未知平台" >> ${MEM_LOG}
                    exit 1
                fi
                echo "----------------------------------------" >> ${MEM_LOG}
                sleep 1
            done
        ) &
        MEM_MONITOR_PID=$!

        wait ${TRAIN_PID}
        wait ${MEM_MONITOR_PID}
        echo "    训练+显存监控完成：GPU卡数=$gpu_num | 单卡样本数=$samples"
        
        MASTER_PORT=$((MASTER_PORT + 1))
        sleep 30
    done
    
    echo "GPU 卡数 $gpu_num 测试完成"
    echo ""
    sleep 30  # 卡数切换间隔
done

echo "所有配置测试完成！"