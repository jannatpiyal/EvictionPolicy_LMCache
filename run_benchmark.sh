#!/bin/bash
#SBATCH --job-name=kv-bench-real
#SBATCH --qos=blanca-clearlab1 --account=blanca-clearlab1 --nodelist=bgpu-g4-u30
#SBATCH --gres=gpu:h100_3g.40gb:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/%j.out
#SBATCH --error=outputs/%j.err

module purge
module load cuda/11.8  # Adjust to available CUDA version

# Print CUDA info
echo "CUDA Version:"
nvcc --version
echo ""

# Activate conda environment
source /scratch/alpine/jana7431/Intune/jana7431-gpt/bin/activate

MODEL="meta-llama/Llama-3.1-8B-Instruct"

echo "============================================"
echo "Job started at $(date)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) available"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================"

# --- Single GPU test ---
echo ""
echo "=== Single GPU: LRU vs LFU ==="
python main.py \
    --model $MODEL \
    --num-gpus 1 \
    --policies lru lfu semantic learned\
    --requests 100 --prefixes 10 --reuse-ratio 0.7 \
    --gpu-mb 512 --max-tokens 30 \
    --output results_1gpu

# --- Multi GPU test (3 GPUs) ---
echo ""
echo "=== Multi GPU (3 workers): LRU vs LFU ==="
python main.py \
    --model $MODEL \
    --num-gpus 3 \
    --policies lru lfu \
    --requests 100 --prefixes 10 --reuse-ratio 0.7 \
    --gpu-mb 512 --max-tokens 30 \
    --output results_3gpu

echo ""
echo "============================================"
echo "Job finished at $(date)"
echo "============================================"
