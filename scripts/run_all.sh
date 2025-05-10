
sbatch -J m1 ./scripts/run.sh m1 6000 mistralai/Mixtral-8x7B-Instruct-v0.1 mixtral1
sbatch -J m2 ./scripts/run.sh m2 6001 mistralai/Mixtral-8x7B-Instruct-v0.1 mixtral1
sbatch -J m3 ./scripts/run.sh m3 6002 mistralai/Mixtral-8x7B-Instruct-v0.1 mixtral1
sbatch -J m4 ./scripts/run.sh m4 6003 mistralai/Mixtral-8x7B-Instruct-v0.1 mixtral1

sbatch -J m1_llama --time=2-00:00:00 ./scripts/run.sh m1 7000 meta-llama/Meta-Llama-3-8B-Instruct llama1
sbatch -J m2_llama --time=2-00:00:00 ./scripts/run.sh m2 7001 meta-llama/Meta-Llama-3-8B-Instruct llama1
sbatch -J m3_llama --time=2-00:00:00 ./scripts/run.sh m3 7002 meta-llama/Meta-Llama-3-8B-Instruct llama1
sbatch -J m4_llama --time=2-00:00:00 ./scripts/run.sh m4 7003 meta-llama/Meta-Llama-3-8B-Instruct llama1


sbatch -J all_llama --time=2-00:00:00 ./scripts/run.sh all 8003 meta-llama/Meta-Llama-3-8B-Instruct llama1
sbatch -J all_mixtral --time=2-00:00:00 ./scripts/run.sh all 8004 mistralai/Mixtral-8x7B-Instruct-v0.1 mixtral1

