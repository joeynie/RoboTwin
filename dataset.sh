#!/usr/bin/env bash
set -e
set -o pipefail

# ========== CONFIG ==========
ROOT=/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/niexian/RoboTwin
HF_OUT=/inspire/hdd/project/wuliqifa/public/niexian/huggingface/lerobot
ATT_OUT=/inspire/hdd/project/wuliqifa/public/niexian/attention_maps

# ========== ARGUMENTS ==========
TASK_NAME=$1
MODE=${2:-demo_randomized_mask}
N_FRAMES=${3:-1000}

if [ -z "$TASK_NAME" ]; then
    echo "Usage:"
    echo "  bash run_all.sh <task_name> [mode=demo_randomized_mask] [num_frames=1000]"
    exit 1
fi

echo "==================================================="
echo "Running data pipeline:"
echo "  TASK       = $TASK_NAME"
echo "  MODE       = $MODE"
echo "  N_FRAMES   = $N_FRAMES"
echo "==================================================="

cd $ROOT

# 1. 采集数据
echo "[1/5] Collecting raw data..."
bash collect_data.sh "$TASK_NAME" "$MODE" 0

# 2. 生成 attention maps
echo "[2/5] Generating attention maps..."
uv run --no-sync script/process_target_mask.py \
    --data_path data/$TASK_NAME/$MODE/data \
    --output attention_maps/${TASK_NAME}-${MODE}-${N_FRAMES}.h5

# 3. pi0 预处理
echo "[3/5] Processing data for pi0..."
cd policy/pi05
bash process_data_pi0.sh "$TASK_NAME" "$MODE" "$N_FRAMES"

# 4. 生成 HuggingFace 数据集缓存
echo "[4/5] Generating HF dataset..."
export XDG_CACHE_HOME=$(pwd)/cache
bash generate.sh processed_data/${TASK_NAME}-${MODE}-${N_FRAMES}/ \
    ${TASK_NAME}-${MODE}-${N_FRAMES}

# 5. 拷贝到公共目录
echo "[5/5] Copying results..."
cp -r cache/huggingface/lerobot/${TASK_NAME}-${MODE}-${N_FRAMES} \
    $HF_OUT/

cd ../..
cp attention_maps/${TASK_NAME}-${MODE}-${N_FRAMES}.h5 \
    $ATT_OUT/

echo "==================================================="
echo "DONE! Output saved to:"
echo "  HF:  $HF_OUT/${TASK_NAME}-${MODE}-${N_FRAMES}"
echo "  ATT: $ATT_OUT/${TASK_NAME}-${MODE}-${N_FRAMES}.h5"
echo "==================================================="
