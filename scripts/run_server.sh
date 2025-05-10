python ./llm_engine/engine.py --model_path mistralai/Mixtral-8x7B-Instruct-v0.1 --gpu_ids 0 --port 5000 --quantization > server.txt 2>&1 &

python ./llm_engine/engine.py --model_path mistralai/Mixtral-8x7B-Instruct-v0.1 --gpu_ids 0 --port 5001 --quantization > server.txt 2>&1 &

python ./llm_engine/engine.py --model_path meta-llama/Meta-Llama-3-8B-Instruct --gpu_ids 0 --port 5050 --quantization > server.txt 2>&1 &
python ./llm_engine/engine.py --model_path meta-llama/Meta-Llama-3-8B-Instruct --gpu_ids 0 --port 5050 --quantization


# python main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt --log_path ./logs/oscillator1_local --port 5001 > test_osc1.txt 2>&1 &
# python main.py --problem_name all --spec_path ./specs/specification_all_numpy.txt --log_path ./logs/test --port 5000 > test_All.txt 2>&1 &
