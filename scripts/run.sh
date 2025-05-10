#!/bin/bash
#SBATCH -J test_sr
#SBATCH --account=ml4science
#SBATCH --mail-user=sepidehfatemi@vt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=l40s_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:1

set -e  # Exit on any error

DATA=$1
PORT=$2
MODEL=$3
ID=$4
echo "run for: DATA ${DATA}, PORT ${PORT}, MODEL:${MODEL}"

# Setup environment
module reset
echo "Module reset"

module load Miniconda3/23.10.0-1
echo "Modules loaded"

source activate llmsr || conda activate llmsr
echo "Conda env 'llmsr' activated"

LOG_PATH=./logs/${ID}/${DATA}
mkdir -p ${LOG_PATH}


# Start server in background
python ./llm_engine/engine.py \
    --model_path ${MODEL} \
    --gpu_ids 0 \
    --port ${PORT} \
    --quantization > ${LOG_PATH}/server_${DATA}.txt 2>&1 &

ENGINE_PID=$!

# Wait for Flask to be ready on port
echo "Waiting for Flask server to start on port ${PORT}..."
for i in {1..30}; do
    if nc -z localhost ${PORT}; then
        echo "Flask server is up!"
        break
    else
        echo "Port ${PORT} not open yet... attempt $i"
        sleep 20
    fi
done

echo "Server started in background (PID: $ENGINE_PID)"

# Start client
python main.py \
    --problem_name ${DATA} \
    --spec_path ./specs/specification_${DATA}_numpy.txt \
    --log_path ${LOG_PATH} \
    --port ${PORT} > ${LOG_PATH}/${DATA}.txt 2>&1

echo "Client finished"

# Kill the server
kill $ENGINE_PID || echo "Server already stopped"

echo "Done"
