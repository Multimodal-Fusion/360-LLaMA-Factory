python -m src.r2e_edits.agenthub.agent.localize localize_multiple --traj_dir "./traj" --exp_name "qwen-32B-test-v1" --max_steps 10 --start_idx 0 --max_workers 1 --k 1 --llm_name "Qwen/Qwen2.5-Coder-32B-instruct"

 python -m src.r2e_edits.agenthub.agent.localize localize_multiple --traj_dir "./traj" --exp_name "qwen-14B-test-v1" --max_steps 10 --start_idx 11 --max_workers 1 --k 1 --llm_name "Qwen/Qwen2.5-Coder-14B-instruct"


python -m src.r2e_edits.agenthub.agent.localize localize_multiple --traj_dir "./traj" --exp_name "qwen-7B-test-v1" --max_steps 10 --start_idx 1 --max_workers 1 --k 1 --llm_name "Qwen/Qwen2.5-Coder-7B-instruct"


python -m src.r2e_edits.agenthub.agent.localize localize_multiple --traj_dir "./traj" --exp_name "qwen-14B-test-v3" --max_steps 5 --start_idx 0 --max_workers 32 --k 200 --llm_name "Qwen/Qwen2.5-Coder-14B-instruct"

python -m src.r2e_edits.agenthub.agent.localize localize_multiple --traj_dir "./traj" --exp_name "gpt4o-loc-maxstep5-v2" --max_steps 5 --start_idx 0 --max_workers 32 --k 200 --llm_name "gpt-4o"


python -m src.r2e_edits.agenthub.agent.localize localize_multiple --traj_dir "./traj" --exp_name "gpt4o-loc-maxstep10-v2" --max_steps 10 --start_idx 0 --max_workers 32 --k 200 --llm_name "gpt-4o"


python -m src.r2e_edits.agenthub.agent.localize localize_multiple --traj_dir "./traj" --exp_name "gpt4o-loc-maxstep10-v2" --max_steps 10 --start_idx 200 --max_workers 32 --k 300 --llm_name "gpt-4o"


python -m src.r2e_edits.train.localize.data_preparation.dataset_preparation create_hf_dataset \
    --traj_file_path=./traj/gpt4o-loc-maxstep10-v2.jsonl \
    --output_dataset_path=./hf_datasets/gpt4o-loc-maxstep10-v2 \
    --hub_repo_name=gpt4o-loc-maxstep10-v2 \
    --push_to_hub=True \
    --max_workers=8

python -m src.r2e_edits.train.localize.data_preparation.dataset_preparation create_hf_dataset     --traj_file_path=./traj/gpt4o-loc-maxstep10-v2.jsonl     --output_dataset_path=./hf_datasets/gpt4o-loc-maxstep10-v2     --hub_repo_name=gpt4o-loc-maxstep10-v2     --push_to_hub=True     --max_workers=32


CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "Qwen/Qwen2.5-Coder-14B-instruct" --port 8000 --gpu-memory-utilization 0.95 --tensor-parallel-size 4

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /home/zha439/scratch/project/r2e/LLaMA-Factory/saves/qwen25coder-14b-instruct-v2/full/sft --port 8000 --gpu-memory-utilization 0.95 --tensor-parallel-size 4

python -m src.r2e_edits.agenthub.agent.localize localize_multiple --traj_dir "./traj" --exp_name "qwen-14B-finetuned-loc-maxstep10-v3" --max_steps 10 --start_idx 0 --max_workers 32 --k 500 --llm_name "/home/zha439/scratch/project/r2e/LLaMA-Factory/saves/qwen25coder-14b-instruct-v3/full/sft"

/home/zha439/scratch/project/r2e/LLaMA-Factory/saves/qwen25coder-14b-instruct-v3/full/sft/checkpoint-1300

python -m src.r2e_edits.agenthub.agent.localize localize_multiple --traj_dir "./traj" --exp_name "qwen-14B-finetuned-loc-maxstep10-v2" --max_steps 10 --start_idx 0 --max_workers 32 --k 500 --llm_name "/home/zha439/scratch/project/r2e/LLaMA-Factory/saves/qwen25coder-14b-instruct-v3/full/sft/checkpoint-1300"


python -m src.r2e_edits.agenthub.agent.localize localize_multiple --traj_dir "./traj" --exp_name "qwen-14B-finetuned-loc-maxstep10-testset-v3" --max_steps 10 --start_idx 500 --max_workers 32 --k 200 --llm_name "/home/zha439/scratch/project/r2e/LLaMA-Factory/saves/qwen25coder-14b-instruct-v3/full/sft"


python -m src.r2e_edits.agenthub.agent.localize localize_multiple --traj_dir "./traj" --exp_name "qwen-14B-loc-maxstep10-testset" --max_steps 10 --start_idx 500 --max_workers 32 --k 200 --llm_name "Qwen/Qwen2.5-Coder-14B-instruct"


CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "/home/zha439/scratch/project/r2e/LLaMA-Factory/saves/qwen25coder-14b-instruct-v3/full/sft" --port 8000 --gpu-memory-utilization 0.95 --tensor-parallel-size 4


python -m src.r2e_edits.agenthub.utils.utils push_model_to_hf_hub --local_model_path ../LLaMA-Factory/saves/qwen25coder-14b-instruct-v3/full/sft --repo_id qwen25coder-14b-instruct-v3


python -m src.r2e_edits.agenthub.agent.localize localize_multiple --traj_dir "./traj" --repo_name 'pandas' --exp_name  "qwen-14B-loc-maxstep10-pandas-testset" --max_steps 10 --start_idx 0 --max_workers 32 --k 1 --llm_name "Qwen/Qwen2.5-Coder-14B-instruct"


python -m src.r2e_edits.agenthub.agent.localize localize_multiple --traj_dir "./traj" --repo_name 'pandas' --exp_name "gpt4o-loc-maxstep10-pandas-testset" --max_steps 10 --start_idx 0 --max_workers 32 --k 2 --llm_name gpt-4o


python -m src.r2e_edits.agenthub.agent.localize localize_multiple --traj_dir "./traj" --repo_name 'pandas' --exp_name "gpt4o-loc-maxstep10-testset" --max_steps 10 --start_idx 1 --max_workers 32 --k 1 --llm_name gpt-4o


python -m src.r2e_edits.agenthub.agent.localize localize_multiple --traj_dir "./traj" --repo_name 'pandas' --exp_name  "qwen-14B-finetunedv3-loc-maxstep10-pandas-testset" --max_steps 10 --start_idx 1 --max_workers 32 --k 199 --llm_name "/home/zha439/scratch/project/r2e/LLaMA-Factory/saves/qwen25coder-14b-instruct-v3/full/sft"


python -m src.r2e_edits.agenthub.agent.localize localize_multiple --traj_dir "./traj" --repo_name 'pandas' --exp_name "gpt4o-loc-maxstep10-testset-pandas" --max_steps 10 --start_idx 0 --max_workers 32 --k 200 --llm_name gpt-4o


python -m src.r2e_edits.agenthub.agent.localize localize_multiple --traj_dir "./traj" --repo_name 'pillow' --exp_name "gpt4o-loc-maxstep10-testset-pillow" --max_steps 10 --start_idx 0 --max_workers 32 --k 200 --llm_name gpt-4o


python -m src.r2e_edits.agenthub.agent.localize localize_multiple --traj_dir "./traj" --repo_name 'pillow' --exp_name "qwen-14B-finetunedv3-loc-maxstep10-testset-pillow" --max_steps 10 --start_idx 0 --max_workers 32 --k 200 --llm_name "/home/zha439/scratch/project/r2e/LLaMA-Factory/saves/qwen25coder-14b-instruct-v3/full/sft"

python -m src.r2e_edits.agenthub.agent.localize localize_multiple --traj_dir "./traj" --repo_name 'pillow' --exp_name "qwen-14B-loc-maxstep10-testset-pillow" --max_steps 10 --start_idx 0 --max_workers 32 --k 200 --llm_name "Qwen/Qwen2.5-Coder-14B-instruct"


python -m src.r2e_edits.agenthub.utils.utils print_results_mt --max_workers 32 --path './traj/qwen-14B-finetunedv3-loc-maxstep10-testset-pillow.jsonl' 


python -m src.r2e_edits.agenthub.utils.utils print_results_mt --max_workers 32 --path './traj/traj.jsonl' 


python -m src.r2e_edits.agenthub.run.localize localize_multiple --traj_dir "./traj" --repo_name 'pillow' --exp_name "test" --max_steps 10 --start_idx 0 --max_workers 32 --k 10 --llm_name gpt-4o




---
# TestGen
---

python -m src.r2e_edits.agenthub.run.testgen testgen_multiple --traj_dir "./traj" --exp_name "gpt4o-testgen-maxstep20-v2" --max_steps 30 --start_idx 5 --max_workers 32 --k 100 --llm_name "gpt-4o"


python -m src.r2e_edits.agenthub.run.edit editagent_multiple --traj_dir "./traj" --exp_name "gpt4o-edit-maxstep30-v2" --max_steps 30 --start_idx 0 --max_workers 32 --k 100 --llm_name "gpt-4o"


python -m src.r2e_edits.agenthub.run.edit editagent_multiple --traj_dir "./traj" --exp_name "claudev-edit-maxstep30-v2" --max_steps 30 --start_idx 0 --max_workers 32 --k 100 --llm_name "vertex_ai/claude-3-5-sonnet-v2@20241022"


python -m src.r2e_edits.agenthub.run.edit editagent_multiple --traj_dir "./traj" --exp_name "gpt4o-edit-maxstep50-swebench-verified-v1" --max_steps 50 --start_idx 0 --max_workers 32 --k 500 --llm_name "gpt-4o"


python -m src.r2e_edits.agenthub.run.edit editagent_multiple --traj_dir "./traj" --exp_name "gpt4o-edit-maxstep30-swebench-verified-v2" --max_steps 30 --start_idx 0 --max_workers 32 --k 500 --llm_name "gpt-4o"

CUDA_VISIBLE_DEVICES=2,3 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tensor-parallel-size 2 --max-model-len 32768 --enforce-eager --port=8000 --gpu-memory-utilization 0.95


CUDA_VISIBLE_DEVICES=1,2,3,4 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tensor-parallel-size 4 --max-model-len 32768 --enforce-eager --port=8000 --gpu-memory-utilization 0.95


---
# Dev Set Training data gen
---

python -m r2e_edits.agenthub.run.edit runagent_multiple --traj_dir "./traj"  --max_workers 32 --exp_name "gpt4o-edit-r2e-dev_10pr_v1-maxstep30-v2" --max_steps 30 --start_idx 0 --k 100 --dataset "r2e-edits/r2e-dockers-v1" --split "dev_10pr_v1" --llm_name "gpt-4o"

python -m r2e_edits.agenthub.run.edit runagent_multiple --traj_dir "./traj"  --max_workers 16 --exp_name "sonnet-edit-r2e-dev_10pr_v1-maxstep30-v1" --max_steps 30 --start_idx 0 --k 100 --dataset "r2e-edits/r2e-dockers-v1" --split "dev_10pr_v1" --llm_name "vertex_ai/claude-3-5-sonnet-v2@20241022"

python -m r2e_edits.agenthub.run.edit runagent_multiple --traj_dir "./traj"  --max_workers 16 --exp_name "sonnet-end2end-r2e-dev_100pr_v1-maxstep40-v1" --max_steps 40 --start_idx 0 --k 1000 --dataset "r2e-edits/r2e-dockers-v1" --split "dev_100pr_v1" --llm_name "vertex_ai/claude-3-5-sonnet-v2@20241022"



python -m r2e_edits.agenthub.run.edit runagent_multiple --traj_dir "./traj"  --max_workers 16 --exp_name "sonnet-edit-r2e-dev_100pr_v1-maxstep30-v1" --max_steps 30 --start_idx 0 --k 1000 --dataset "r2e-edits/r2e-dockers-v1" --split "dev_100pr_v1" --llm_name "vertex_ai/claude-3-5-sonnet-v2@20241022"


export PYTHONPATH="/home/gcpuser/project/r2e/r2e-edits-internal/src:$PYTHONPATH"



gcsfuse --implicit-dirs edits-bucket ~/buckets/edits_data


python -m src.r2e_edits.agenthub.train.utils pull_images --dataset "r2e-edits/r2e-dockers-v1" --split "dev_100pr_v1"


python -m src.r2e_edits.agenthub.train.data_prep create_hf_dataset --traj_file_path=./traj/sonnet-edit-r2e-dev_100pr_v1-maxstep30-v1.jsonl --output_dataset_path=./hf_datasets/sonnet-edit-r2e-dev_100pr_v1-maxstep30-v1 --hub_repo_name=sonnet-edit-r2e-dev_100pr_v1-maxstep30-v1 --push_to_hub=True     --max_workers=32



CUDA_VISIBLE_DEVICES=1,3 vllm serve "Qwen/Qwen2.5-Coder-14B-instruct" --port 8000 --gpu-memory-utilization 0.95 --tensor-parallel-size 2


python -m r2e_edits.agenthub.run.edit runagent_multiple --traj_dir "./traj"  --max_workers 16 --exp_name "qwen25_32B_base-edit-r2e-dev_100pr_v1-maxstep30-v1" --max_steps 30 --start_idx 0 --k 100 --dataset "r2e-edits/r2e-dockers-v1" --split "dev_100pr_v1" --llm_name "hosted_vllm/Qwen/Qwen2.5-Coder-32B-instruct"

CUDA_VISIBLE_DEVICES=1,3 vllm serve "/home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-14b-instruct-edit-rft-16k-v3/checkpoint-100" --port 8000 --gpu-memory-utilization 0.95 --tensor-parallel-size 2

python -m r2e_edits.agenthub.run.edit runagent_multiple --traj_dir "./traj"  --max_workers 16 --exp_name "qwen25_32B_rft-edit-r2e-dev_100pr_v1-maxstep30-v1" --max_steps 30 --start_idx 0 --k 100 --dataset "r2e-edits/r2e-dockers-v1" --split "dev_100pr_v1" --llm_name "hosted_vllm//home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-14b-instruct-edit-rft-16k-v3/checkpoint-100"


python -m r2e_edits.agenthub.run.edit runagent_multiple --traj_dir "./traj"  --max_workers 16 --exp_name "qwen25_32B_rft-edit-r2e-dev_100pr_v1-maxstep30-v1" --max_steps 30 --start_idx 0 --k 100 --dataset "r2e-edits/r2e-dockers-v1" --split "dev_100pr_v1" --llm_name "hosted_vllm//home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-14b-instruct-edit-rft-16k-v3/checkpoint-100"


python -m r2e_edits.agenthub.run.edit runagent_multiple --traj_dir "./traj"  --max_workers 16 --exp_name "qwen25_14B_rft-edit-r2e-dev_100pr_v1-maxstep30-v2" --max_steps 30 --start_idx 0 --k 100 --dataset "r2e-edits/r2e-dockers-v1" --split "dev_100pr_v1" --llm_name "hosted_vllm//home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-14b-instruct-edit-rft-20k-v1"

CUDA_VISIBLE_DEVICES=1,3 vllm serve "/home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-14b-instruct-edit-rft-20k-v1" --port 8000 --gpu-memory-utilization 0.95 --tensor-parallel-size 4


python -m r2e_edits.agenthub.run.edit runagent_multiple --traj_dir "./traj"  --max_workers 16 --exp_name "qwen25_14B_rft-edit-swebv-test-maxstep30-v3" --max_steps 30 --start_idx 0 --k 500 --dataset "r2e-edits/swebench-verified-v1" --split "test" --llm_name "hosted_vllm//home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-14b-instruct-edit-rft-20k-v1"



CUDA_VISIBLE_DEVICES=1,3 vllm serve "/home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-32b-instruct-edit-rft_lora-20k-v1" --port 8000 --gpu-memory-utilization 0.95 --tensor-parallel-size 4

CUDA_VISIBLE_DEVICES=1,3 vllm serve "Qwen/Qwen2.5-Coder-32B-instruct" --enable-lora --lora-modules rft-lora="/home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-32b-instruct-edit-rft_lora-20k-v1" --port 8000 --gpu-memory-utilization 0.95 --tensor-parallel-size 4

DISABLE_VERSION_CHECK=1 API_PORT=8000 llamafactory-cli api examples/inference/qwen_lora_sft.yaml 

python -m r2e_edits.agenthub.run.edit runagent_multiple --traj_dir "./traj"  --max_workers 16 --exp_name "qwen25_32B_rft_lora-edit-swebv-test-maxstep30-v1" --max_steps 30 --start_idx 0 --k 500 --dataset "r2e-edits/swebench-verified-v1" --split "test" --llm_name "hosted_vllm/Qwen/Qwen2.5-Coder-32B-instruct"

cat sonnet-end2end-r2e-dev_100pr_v2-maxstep40-v1.jsonl \
    sonnet-end2end-r2e-dev_100pr_v1-maxstep40-v1.jsonl \
    > sonnet-end2end-r2e-dev_100pr_1_2-maxstep40-v1.jsonl


cat gpt4o-end2end-r2e-dev_100pr_v1-maxstep40-v2.jsonl \
    gpt4o-end2end-r2e-dev_100pr_v2-maxstep40-v2.jsonl \
    > gpt4o-end2end-r2e-dev_100pr_1_2-maxstep40-v2.jsonl


sonnet-end2end-r2e-dev_100pr_v1-maxstep40-v2.jsonl
sonnet-end2end-r2e-dev_100pr_v2-maxstep40-v1.jsonl
sonnet-end2end-r2e-dev_100pr_v3-maxstep40-v2.jsonl
sonnet-end2end-r2e-dev_100pr_v3-maxstep40-v4.jsonl
sonnet-end2end-r2e-dev_100pr_v4-maxstep40-v1.jsonl
sonnet-end2end-r2e-dev_100pr_v4-maxstep40-v2.jsonl
sonnet-end2end-r2e-dev_100pr_v5-maxstep40-v1.jsonl
sonnet-end2end-r2e-dev_100pr_v5-maxstep40-v2.jsonl
sonnet-end2end-r2e-dev_100pr_v5-maxstep40-v3.jsonl
sonnet-end2end-r2e-dev_100pr_v6-maxstep40-v1.jsonl
sonnet-end2end-r2e-dev_100pr_v6-maxstep40-v2.jsonl
sonnet-end2end-r2e-dev_100pr_v6-maxstep40-v3.jsonl

cat sonnet-end2end-r2e-dev_100pr_v1-maxstep40-v2.jsonl \
    sonnet-end2end-r2e-dev_100pr_v2-maxstep40-v1.jsonl \
    sonnet-end2end-r2e-dev_100pr_v3-maxstep40-v2.jsonl \
    sonnet-end2end-r2e-dev_100pr_v3-maxstep40-v4.jsonl \
    sonnet-end2end-r2e-dev_100pr_v4-maxstep40-v1.jsonl \
    sonnet-end2end-r2e-dev_100pr_v4-maxstep40-v2.jsonl \
    sonnet-end2end-r2e-dev_100pr_v5-maxstep40-v1.jsonl \
    sonnet-end2end-r2e-dev_100pr_v5-maxstep40-v2.jsonl \
    sonnet-end2end-r2e-dev_100pr_v5-maxstep40-v3.jsonl \
    sonnet-end2end-r2e-dev_100pr_v6-maxstep40-v1.jsonl \
    sonnet-end2end-r2e-dev_100pr_v6-maxstep40-v2.jsonl \
    sonnet-end2end-r2e-dev_100pr_v6-maxstep40-v3.jsonl \
    > sonnet-end2end-r2e-dev_100pr_1_6-maxstep40-v1.jsonl

python -m src.r2e_edits.agenthub.train.data_prep create_hf_dataset --traj_file_path=/home/gcpuser/buckets/edits_data/sonnet-end2end-r2e-dev_100pr_1_2-maxstep40-v1.jsonl --output_dataset_path=./hf_datasets/sonnet-end2end-r2e-dev_100pr_v1_2-maxstep40-v1 --hub_repo_name=sonnet-end2end-r2e-dev_100pr_v1_2-maxstep40-v1 --push_to_hub=True     --max_workers=32

python -m src.r2e_edits.agenthub.train.data_prep create_hf_dataset --traj_file_path=/home/gcpuser/buckets/edits_data/traj/gpt4o-end2end-r2e-dev_100pr_1_2-maxstep40-v2.jsonl --output_dataset_path=./hf_datasets/gpt4o-end2end-r2e-dev_100pr_1_2-maxstep40-v2 --hub_repo_name=gpt4o-end2end-r2e-dev_100pr_1_2-maxstep40-v2 --push_to_hub=True     --max_workers=32 --trajectory_format='new'



CUDA_VISIBLE_DEVICES=1,3 vllm serve "/home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-14b-instruct-end2end-rft-20k-v1" --port 8000 --gpu-memory-utilization 0.95 --tensor-parallel-size 2


CUDA_VISIBLE_DEVICES=1,3 vllm serve "/home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-14b-instruct-end2end-rft-20k-v1" --port 8000 --gpu-memory-utilization 0.9 --tensor-parallel-size 4


python -m r2e_edits.agenthub.run.edit runagent_multiple --traj_dir "./traj"  --max_workers 16 --exp_name "qwen25coder-14b-instruct-end2end-rft-20k-swebv_test-maxstep40-v1" --max_steps 40 --start_idx 0 --k 500 --dataset "r2e-edits/swebench-verified-v1" --split "test" --llm_name "hosted_vllm//home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-14b-instruct-end2end-rft-20k-v1"

python push_model.py \
  --local_model_path="/home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-14b-instruct-end2end-rft-20k-v1" \
  --repo_id="r2e-edits/qwen25coder-14b-instruct-end2end-rft-20k-v1" \
  --private=False \
  --replace=True \
  --commit_message="Add model"

  uv run src/r2e_edits/agenthub/trajectory/analyze_logs.py traj/qwen25coder-14b-instruct-end2end-rft-20k-swebv_test-maxstep40-v1.jsonl


vllm serve "/home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-14b-instruct-
edit-rft-20k-v1" --port 8000 --gpu-memory-utilization 0.95 --tensor-parallel-size 4 --enable-prefix-caching


qwen25coder-14b-instruct-end2end_edit-rft-20k-v1

vllm serve "/home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-14b-instruct-end2end_edit-rft-20k-v1" --port 8000 --gpu-memory-utilization 0.95 --tensor-parallel-size 4 --enable-prefix-caching


python -m r2e_edits.agenthub.run.edit runagent_multiple --traj_dir "./traj"  
--max_workers 16 --exp_name "qwen25coder-14b-instruct-end2end-rft-20k-swebv_test-maxstep40-v1" 
--max_steps 40 --start_idx 0 --k 500 --dataset "r2e-edits/swebench-verified-v1" --split "test" 
--llm_name "hosted_vllm//home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-14b-instruct-end2end-rft-20k-v1"


python -m r2e_edits.agenthub.run.edit runagent_multiple --traj_dir "./traj" --max_workers 16 --exp_name "qwen25coder-14b-instruct-end2end_edit-rft-20k-swebv_test-maxstep40-v1" --max_steps 40 --start_idx 0 --k 500 --dataset "r2e-edits/swebench-verified-v1" --split "test" --llm_name "hosted_vllm//home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-14b-instruct-end2end_edit-rft-20k-v1"

vllm serve "/home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-14b-instruct-end2end_gpt4o-rft-20k-v1" --port 8000 --gpu-memory-utilization 0.95 --tensor-parallel-size 4 --enable-prefix-caching

python -m r2e_edits.agenthub.run.edit runagent_multiple --traj_dir "./traj" --max_workers 16 --exp_name  "qwen25coder-14b-instruct-end2end_gpt4o-rft-20k-swebv_test-maxstep40-v1" --max_steps 40 --start_idx 0 --k 500 --dataset "r2e-edits/swebench-verified-v1" --split "test" --llm_name "hosted_vllm//home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-14b-instruct-end2end_gpt4o-rft-20k-v1"


python -m src.r2e_edits.agenthub.train.data_prep create_hf_dataset --traj_file_path=/home/gcpuser/buckets/edits_data/finaltraj/sonnet-end2end-r2e-dev_100pr_1_6-maxstep40-v1.jsonl --output_dataset_path=./hf_datasets/sonnet-end2end-r2e-dev_100pr_1_6-maxstep40-v1 --hub_repo_name=sonnet-end2end-r2e-dev_100pr_1_6-maxstep40-v1 --push_to_hub=True     --max_workers=32 --trajectory_format='new'

sonnet-end2end-r2e-dev_100pr_1_6-maxstep40-v1.jsonl


vllm serve "/home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-14b-instruct-end2end_sonnet_2717-rft-20k-v1/checkpoint-700" --port 8000 --gpu-memory-utilization 0.95 --tensor-parallel-size 4 --enable-prefix-caching
qwen25coder-14b-instruct-end2end_sonnet_2717-rft-20k_bz32_epoch2-v1


python -m r2e_edits.agenthub.run.edit runagent_multiple --traj_dir "./traj" --max_workers 16 --exp_name  "qwen25coder-14b-instruct-end2end_sonnet_2717_ckpt700-rft-20k-swebv_test-maxstep40-v1" --max_steps 40 --start_idx 0 --k 500 --dataset "r2e-edits/swebench-verified-v1" --split "test" --llm_name "hosted_vllm//home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-14b-instruct-end2end_sonnet_2717-rft-20k-v1/checkpoint-700"

python -m r2e_edits.agenthub.run.edit runagent_multiple --traj_dir "./traj" --max_workers 16 --exp_name  "qwen25coder-14b-instruct-end2end_sonnet_1_6-rft-20k-swebv_test-maxstep40-v1" --max_steps 40 --start_idx 0 --k 500 --dataset "r2e-edits/swebench-verified-v1" --split "test" --llm_name "hosted_vllm//home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-14b-instruct-end2end_sonnet_1_6-rft-20k-v1"

vllm serve "/home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-14b-instruct-end2end_sonnet_1_6-rft-20k-v1" --port 8000 --gpu-memory-utilization 0.95 --tensor-parallel-size 4 --enable-prefix-caching


qwen25coder-14b-instruct-end2end_sonnet_1_6-rft-20k-v1


vllm serve "/home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-14b-instruct-end2end_sonnet_2717-rft-20k_bz32_epoch2-v1" --port 8000 --gpu-memory-utilization 0.95 --tensor-parallel-size 4 --enable-prefix-caching



python -m r2e_edits.agenthub.run.edit runagent_multiple --traj_dir "./traj_ablation" --max_workers 16 --exp_name  "qwen25coder-14b-instruct-end2end_sonnet_2717-rft-20k_bz32_epoch2-v1" --max_steps 40 --start_idx 0 --k 150 --dataset "r2e-edits/swebench-verified-v1" --split "test" --llm_name "hosted_vllm//home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-14b-instruct-end2end_sonnet_2717-rft-20k_bz32_epoch2-v1"


python -m src.r2e_edits.agenthub.train.data_prep create_hf_dataset --traj_file_path=/home/gcpuser/buckets/edits_data/finaltraj/sonnet-end2end-r2e-dev_100pr_combined-maxstep40-v1.jsonl --output_dataset_path=./hf_datasets/sonnet-end2end-r2e-dev_100pr_combined-maxstep40_context20k-v1 --hub_repo_name=sonnet-end2end-r2e-dev_100pr_combined-maxstep40_context20k-v1 --push_to_hub=True     --max_workers=32 --max_tokens=20480

export HF_HUB_CHUNK_SIZE=52428800

python push_model.py \
  --local_model_path="/home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/14B_ablations_maxsamples1600_sonnet_combined_maxstep40_rft-20k_bz8_epoch2_lr1en5-v1" \
  --repo_id="r2e-edits/14B_ablations_maxsamples1600_sonnet_combined_maxstep40_rft-20k_bz8_epoch2_lr1en5-v1" \
  --private=False \
  --replace=True \
  --commit_message="Add model"


python push_model.py \
  --local_model_path="/home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-14b-instruct-end2end_32bonpolicy_3_6_k-rft-20k_bz8_epoch2_lr1en5-v1" \
  --repo_id="r2e-edits/qwen25coder-14b-instruct-end2end_32bonpolicy_2_8_k-rft-20k_bz8_epoch2_lr1en5-v1" \
  --private=False \
  --replace=True \
  --commit_message="Add model"

qwen25coder-14b-instruct-end2end_32bonpolicy_3_6_k-rft-20k_bz8_epoch2_lr1en5-v1

git add notebooks/agent-training.ipynb \
    src/r2e_edits/agenthub/run/edit.py \
    run_restarts.sh \
    src/r2e_edits/agenthub/config/testgen-v2.yaml \
    src/r2e_edits/agenthub/train/edit-rft-restart.yaml 



bash src/docker_bash_utils/remove_containers.sh ; time uv run python -m r2e_edits.agenthub.run.edit runagent_multiple --traj_dir "./traj"  --max_workers 54 --start_idx 0 --k 500 --dataset "r2e-edits/swebench-verified-v1" --split "test" --llm_name openai/r2e-edits/qwen25coder-14b-instruct-end2end_32bonpolicy_2_8_k-rft-20k_bz8_epoch2_lr1en5-v1 --exp_name qwen25coder-14b-instruct-end2end_32bonpolicy_2_8_k-rft-20k_bz8_epoch2_lr1en5-v1 --temperature 0 --max_steps 40 ; 




python -m src.r2e_edits.agenthub.train.verifier_data_prep create_hf_dataset --traj_file_path=/home/gcpuser/buckets/edits_data/trajensemble/traj/32b_ense_swebv_temp09_no-testpatch_N3_rollout_merged.jsonl --output_dataset_path=./hf_datasets/32b_ense_swebv_temp09_no-testpatch_N3_rollout_merged --hub_repo_name=32b_ense_swebv_temp09_no-testpatch_N3_rollout10_merged --push_to_hub=True     --max_workers=32



bash src/docker_bash_utils/remove_containers.sh ; time uv run python -m r2e_edits.agenthub.run.testgen runagent_multiple --traj_dir "./traj"  --max_workers 32 --start_idx 0 --k 30 --dataset "r2e-edits/swebench-verified-v1" --split "test" --llm_name "vertex_ai/claude-3-5-sonnet-v2@20241022" --exp_name testgen_sonnet_swebv30_rollout1-v1 --temperature 0.8 --max_steps 30 ;

bash src/docker_bash_utils/remove_containers.sh ; python -m src.r2e_edits.agenthub.eval.testverifier_eval --num_workers=16 --output_csv="testverifier_output-v1.csv"



python -m src.r2e_edits.agenthub.train.data_prep create_hf_dataset --traj_file_path=/home/gcpuser/buckets/edits_data/finaltraj/sonnet-end2end-r2e-dev_100pr_combined-maxstep40-v1.jsonl --output_dataset_path=./hf_datasets/sonnet-end2end-r2e-dev_100pr_combined-maxstep40_context32k-sft-v1 --hub_repo_name=sonnet-end2end-r2e-dev_100pr_combined-maxstep40_context32k-sft-v1  --push_to_hub=True     --max_workers=32 --include_thoughts=True


python -m src.r2e_edits.agenthub.train.data_prep create_hf_dataset --traj_file_path=/home/gcpuser/buckets/edits_data/finaltraj/sonnet-end2end-r2e-dev_100pr_combined-maxstep40-v1.jsonl --output_dataset_path=./hf_datasets/sonnet-end2end-r2e-dev_100pr_combined-maxstep40_context32k_nothoughts-v1 --hub_repo_name=sonnet-end2end-r2e-dev_100pr_combined-maxstep40_context32k_nothoughts-v1 --push_to_hub=True     --max_workers=32 --include_thoughts=False

python -m src.r2e_edits.agenthub.train.data_prep create_hf_dataset --traj_file_path=./traj_sft/sonnet-end2end-r2e-dev_100pr_combined-maxstep40-v1.jsonl --output_dataset_path=./hf_datasets/sonnet-end2end-r2e-dev_100pr_combined-maxstep40_context32k-sft-v1 --hub_repo_name=sonnet-end2end-r2e-dev_100pr_combined-maxstep40_context32k-sft-v1  --push_to_hub=True     --max_workers=32 --include_thoughts=True

====================
squeue | grep zha439
scancel 59723200
squeue -j 5288914 --start
sinteractive -n 2 -c 4 -g gpu:1 -m 48gb -A OD-227441 -t 1:59:00
sinteractive -n 4 -c 8 -g gpu:4 -m 95gb -A OD-227441 -t 23:58:00
sinteractive -n 4 -c 8 -g gpu:4 -m 95gb -A OD-227441 -t 1:59:00
sinteractive -n 4 -c 8 -g gpu:4 -m 192gb -A OD-227441 -t 23:58:00

sinteractive -n 2 -c 4 -g gpu:1 -m 48gb -A OD-227441 -t 1:59:00

srun --nodes=2 --ntasks-per-node=1 --cpus-per-task=32 --gres=gpu:4 --mem=500GB -A OD-227441 -t 1:59:00 --pty bash
env | grep SLURM_NODE


# Commands used after 02/12/2024
sinteractive -n 1 -c 8 -g gpu:1 -m 32gb -A OD-227441 -t 02:00:00
sinteractive -n 4 -c 8 -g gpu:4 -m 192gb -A OD-227441 -t 02:00:00
srun --nodes=2 --ntasks-per-node=1 --cpus-per-task=32 --gres=gpu:4 --mem=500G -A OD-227441 -t 1:59:59 --pty bash
srun --nodes=2 --ntasks-per-node=1 --cpus-per-task=32 --gres=gpu:4 --mem=500G -A OD-227441 -t 1-00:00:00 --pty bash
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --gres=gpu:1 --mem=64G -A OD-227441 -t 1:59:59 --pty bash
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --gres=gpu:2 --mem=128G -A OD-227441 -t 1:59:59 --pty bash

ssh -N -f -L 7234:localhost:7234 g033

# Notes
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=64 --gres=gpu:4 --mem=500G -A OD-227441 -t 1:59:59 --pty bash

srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --gres=gpu:1 --mem=500G -A OD-227441 -t 1:59:59 --pty bash

srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --gres=gpu:1 --mem=500G -A OD-227441 -t 1:59:59 --pty bash


srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=64 --gres=gpu:4 --mem=200G -A OD-227441 -t 1:59:59 --pty bash


srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --gres=gpu:4 --mem=200G -A OD-227441 -t 1:00:00:00 --pty bash


srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=128 --gres=gpu:4 \
--mem=500G -A OD-227441 -t 29-23:59:59 --pty bash



srun --nodes=2 --ntasks-per-node=1 --cpus-per-task=64 --gres=gpu:4 \
--mem=500G -A OD-227441 -t 6-23:59:59 --pty bash

srun --nodes=4 --ntasks-per-node=1 --cpus-per-task=32 --gres=gpu:4 --mem=384G -A OD-227441 -t 23:59:59 --pty bash

CUDA_VISIBLE_DEVICES=2,3 python3 -m sglang.launch_server --model-path r2e-edits/qwen25coder-32b-instruct-end2end_sonnet_combined_maxstep40-rft-16k_bz8_epoch2_lr1en5-v1 --host 0.0.0.0 --port 8000 --tp-size 4

CUDA_VISIBLE_DEVICES=3 python3 -m sglang.launch_server --model-path r2e-edits/qwen25coder-14b-instruct-end2end_sonnet_combined_maxstep40_filter15-rft-20k_bz8_epoch2_lr1en5-v1 --host 0.0.0.0 --port 8000 --tp-size 1



python3 -m sglang.launch_server --model-path r2e-edits/qwen25coder-32b-instruct-end2end_sonnet_combined_maxstep40-rft-16k_bz8_epoch2_lr1en5-v1 --host 0.0.0.0 --port 8000 --tp-size 8 --dist-init-addr sgl-dev-0:50000 --nnodes 2 --node-rank 0

python3 -m sglang.launch_server --model-path r2e-edits/qwen25coder-32b-instruct-end2end_sonnet_combined_maxstep40-rft-16k_bz8_epoch2_lr1en5-v1 --host 0.0.0.0 --port 8000 --tp-size 8 --dist-init-addr sgl-dev-0:50000 --nnodes 2 --node-rank 1

python3 -m sglang.launch_server --model-path r2e-edits/qwen25coder-14b-instruct-end2end_sonnet_combined_maxstep40_filter15-rft-20k_bz8_epoch2_lr1en5-v1 --host 0.0.0.0 --port 8000 --tp-size 8 --dist-init-addr sgl-dev-0:50000 --nnodes 2 --node-rank 1



python push_model.py   --local_model_path="/home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/14B_ablations_maxsamples800_sonnet_combined_maxstep40_rft-20k_bz8_epoch2_lr1en5-v1"   --repo_id="r2e-edits/14B_ablations_maxsamples800_sonnet_combined_maxstep40_rft-20k_bz8_epoch2_lr1en5-v1"   --private=False   --replace=True   --commit_message="Add model"




CUDA_VISIBLE_DEVICES=2,3 python push_model.py   --local_model_path="/home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-14b-instruct_end2end_sonnet_combined_maxstep40_sft-32k_bz8_epoch2_lr1en5-v1"   --repo_id="r2e-edits/qwen25coder-14b-end2end_sonnet_combined_maxstep40_sft-32k_bz8_epoch2_lr1en5-v1"   --private=False   --replace=True   --commit_message="Add model"


CUDA_VISIBLE_DEVICES=1,3 python push_model.py   --local_model_path="/home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/14B-agent-filter15_testgenverifier_noagg_rft-20k_bz8_epoch2_lr1en5-v1"   --repo_id="r2e-edits/14B-agent-filter15_testgenverifier_noagg_rft-20k_bz8_epoch2_lr1en5-v1"   --private=False   --replace=True   --commit_message="Add model"


CUDA_VISIBLE_DEVICES=1,3 python push_model.py   --local_model_path="/home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/14B_ablations_maxsamples100_sonnet_combined_maxstep40_rft-20k_bz8_epoch2_lr1en5-v1" --repo_id="r2e-edits/14B_ablations_maxsamples100_sonnet_combined_maxstep40_rft-20k_bz8_epoch2_lr1en5-v1"   --private=False   --replace=True   --commit_message="Add model"


CUDA_VISIBLE_DEVICES=1,3 python push_model.py   --local_model_path="/home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-7b-instruct-end2end_sonnet_combined_maxstep40-rft-20k_bz8_epoch2_lr1en5-v1" --repo_id="r2e-edits/qwen25coder-7b-instruct-end2end_sonnet_combined_maxstep40-rft-20k_bz8_epoch2_lr1en5-v1"   --private=False   --replace=True   --commit_message="Add model"


CUDA_VISIBLE_DEVICES=2,3 python push_model.py   --local_model_path="/home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-7b-instruct-end2end_sonnet_combined_maxstep40-rft-20k_bz8_epoch2_lr1en5-v2" --repo_id="r2e-edits/qwen25coder-7b-instruct-end2end_sonnet_combined_maxstep40-rft-20k_bz8_epoch2_lr1en5-v2"   --private=False   --replace=True   --commit_message="Add model"


/scratch3/zha439/miniconda3/bin/activate



CUDA_VISIBLE_DEVICES=2,3 python push_model.py   --local_model_path="/home/zha439/scratch/project/r2e/r2e-edits-internal/LLaMA-Factory/saves/qwen25coder-14b-instruct_end2end_sonnet_combined_maxstep40_sft-exitreason-agent-32k_bz8_epoch5_lr1en5-v1" --repo_id="r2e-edits/qwen25coder-14b_end2end_sonnet_combined_maxstep40_sft-exitreason-agent-32k_bz8_epoch5_lr1en5-v1"   --private=False   --replace=True   --commit_message="Add model"


qwen25coder-14b-instruct_end2end_sonnet_combined_maxstep40_sft-exitreason-agent-32k_bz8_epoch5_lr1en5-v1


python3 -m sglang.launch_server --model-path r2e-edits/qwen25coder-14b_end2end_sonnet_combined_maxstep40_sft-exitreason-agent-32k_bz8_epoch5_lr1en5-v1 --host 0.0.0.0 --port 8000 --tp-size 8

docker run --gpus all \
    --shm-size 32g \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    python -m sglang.launch_server --model-path r2e-edits\qwen25coder-14b_end2end_sonnet_combined_maxstep40_sft-exitreason-agent-32k_bz8_epoch_lr1en5-v1 --host 0.0.0.0 --port 8000 --tp-size 8




##########################
qwen 32b
##########################

python -m sglang.launch_server --model-path r2e-edits/qwen3-14b_end2end_sonnet_combined_maxstep40_sft-exitreason-agent-32k_bz8_epoch2_lr1en5-v1 --reasoning-parser qwen3 --host 0.0.0.0 --port 8000 --tp-size 8

uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=r2e

saves/
qwen3-32b_end2end_sonnet_combined_maxstep40_sft-exitreason-agent-32k_bz8_epoch2_lr1en5-v1

/home/naman/project/r2e-edits-internal

CUDA_VISIBLE_DEVICES=2,3 python push_model.py   --local_model_path="/home/naman/project/qwen3/LLaMA-Factory/saves/qwen3-32b_end2end_sonnet_combined_maxstep40_sft-exitreason-agent-32k_bz8_epoch2_lr1en5-v1" --repo_id="r2e-edits/qwen3-32b_end2end_sonnet_combined_maxstep40_sft-exitreason-agent-32k_bz8_epoch2_lr1en5-v1" --private=False   --replace=True   --commit_message="Add model"


python -m sglang.launch_server --model-path "/home/naman/project/qwen3/LLaMA-Factory/saves/qwen3-32b_end2end_sonnet_combined_maxstep40_sft-exitreason-agent-32k_bz8_epoch2_lr1en5-v1" --reasoning-parser qwen3 --host 0.0.0.0 --port 8000 --tp-size 8

/home/naman/project/qwen3/LLaMA-Factory/saves/qwen3-32b_end2end_sonnet_combined_maxstep40_sft-exitreason-agent-32k_bz8_epoch2_lr1en5-v1

bash src/docker_bash_utils/remove_containers.sh ; time uv run python -m r2e_edits.agenthub.run.edit runagent_multiple --traj_dir "./traj"  --max_workers 54 --start_idx 0 --k 500 --dataset "r2e-edits/swebench-verified-v1" --split "test" --llm_name openai//home/naman/project/qwen3/LLaMA-Factory/saves/qwen3-32b_end2end_sonnet_combined_maxstep40_sft-exitreason-agent-32k_bz8_epoch2_lr1en5-v1 --exp_name qwen3-32b_end2end_sonnet_combined_maxstep40_sft-exitreason-agent-32k_bz8_epoch2_lr1en5-v1 --temperature 0 --max_steps 40 ;

r2e-edits/qwen3-14b_end2end_sonnet_combined_maxstep40_sft-exitreason-agent-32k_bz8_epoch2_lr1en5-v1

python -m sglang.launch_server --model-path r2e-edits/qwen3-14b_end2end_sonnet_combined_maxstep40_sft-exitreason-agent-32k_bz8_epoch2_lr1en5-v1 --reasoning-parser qwen3 --host 0.0.0.0 --port 8000 --tp-size 8

bash src/docker_bash_utils/remove_containers.sh ; time uv run python -m r2e_edits.agenthub.run.edit runagent_multiple --traj_dir "./traj"  --max_workers 54 --start_idx 0 --k 500 --dataset "r2e-edits/swebench-verified-v1" --split "test" --llm_name openai/r2e-edits/qwen3-14b_end2end_sonnet_combined_maxstep40_sft-exitreason-agent-32k_bz8_epoch2_lr1en5-v1 --exp_name qwen3-14b_end2end_sonnet_combined_maxstep40_sft-exitreason-agent-32k_bz8_epoch2_lr1en5-v1 --temperature 0 --max_steps 40 ;



##########################
swesmith
##########################

SWE-bench/SWE-agent-LM-32B

python3 -m sglang.launch_server --model-path r2e-edits/qwen25coder-32b-instruct-end2end_sonnet_combined_maxstep40-rft-16k_bz8_epoch2_lr1en5-v1 --host 0.0.0.0 --port 8000 --tp-size 8


python -m sglang.launch_server --model-path SWE-bench/SWE-agent-LM-32B --host 0.0.0.0 --port 8000 --tp-size 8

/home/naman/project/qwen3/LLaMA-Factory/saves/qwen3-14b_end2end_sonnet_combined_maxstep40_sft-exitreason-agent-32k_bz8_epoch5_lr1en5-v1
qwen25coder-14b_swe-smith-trajectories-R2E-32k_bz8_epoch3_lr1en5-v1

python -m sglang.launch_server --model-path "/home/naman/project/qwen3/LLaMA-Factory/saves/qwen25coder-14b_swe-smith-trajectories-R2E-32k_bz8_epoch3_lr1en5-v1" --host 0.0.0.0 --port 8000 --tp-size 8

bash src/docker_bash_utils/remove_containers.sh ; time PYTHONPATH="$PWD/src:$PYTHONPATH" uv run python -m r2e_edits.agenthub.run.edit runagent_multiple --traj_dir "./traj" --max_workers 54 --start_idx 0 --k 500 --dataset "r2e-edits/swebench-verified-v1" --split "test" --llm_name openai//home/naman/project/qwen3/LLaMA-Factory/saves/qwen25coder-14b_swe-smith-trajectories-R2E-32k_bz8_epoch3_lr1en5-v1 --exp_name qwen25coder-14b_swe-smith-trajectories-R2E-32k_bz8_epoch3_lr1en5-v1 --temperature 0 --max_steps 40

PYTHONPATH=$PYTHONPATH:/home/naman/project/r2e-edits-internal bash src/docker_bash_utils/remove_containers.sh ; time uv run python -m src.r2e_edits.agenthub.run.edit runagent_multiple --traj_dir "./traj"  --max_workers 54 --start_idx 0 --k 500 --dataset "r2e-edits/swebench-verified-v1" --split "test" --llm_name openai//home/naman/project/qwen3/LLaMA-Factory/saves/qwen25coder-14b_swe-smith-trajectories-R2E-32k_bz8_epoch3_lr1en5-v1 --exp_name qwen25coder-14b_swe-smith-trajectories-R2E-32k_bz8_epoch3_lr1en5-v1 --temperature 0 --max_steps 40 ;

PYTHONPATH="$PWD/src:$PYTHONPATH" watch uv run python src/r2e_edits/agenthub/trajectory/analyze_logs.py traj/qwen25coder-14b_swe-smith-trajectories-R2E-32k_bz8_epoch3_lr1en5-v1.jsonl



/home/naman/project/qwen3/LLaMA-Factory/saves/qwen25coder-32b_swe-smith-trajectories-R2E-v2_r2e-RFT-trajs-32k_bz8_epoch1_lr1en5-v1

python -m sglang.launch_server --model-path /home/naman/project/qwen3/LLaMA-Factory/saves/qwen25coder-32b_swe-smith-trajectories-R2E-v2_r2e-RFT-trajs-32k_bz8_epoch1_lr1en5-v1
 --host 0.0.0.0 --port 8000 --tp-size 8 
 
 bash src/docker_bash_utils/remove_containers.sh ; time PYTHONPATH="$PWD/src:$PYTHONPATH" uv run python -m r2e_edits.agenthub.run.edit runagent_multiple --traj_dir "./traj" --max_workers 54 --start_idx 0 --k 500 --dataset "r2e-edits/swebench-verified-v1" --split "test" --llm_name openai//home/naman/project/qwen3/LLaMA-Factory/saves/qwen25coder-32b_swe-smith-trajectories-R2E-v2_r2e-RFT-trajs-32k_bz8_epoch1_lr1en5-v1 --exp_name qwen25coder-32b_swe-smith-trajectories-R2E-v2_r2e-RFT-trajs-32k_bz8_epoch1_lr1en5-v1 --temperature 0 --max_steps 40

##########################
# vllm testing
##########################

R2E-Gym/R2EGym-14B-Agent

sudo docker run --gpus all \
-v ~/.cache/huggingface:/root/.cache/huggingface \
-v /home/ubuntu/r2e-edits-internal/LLaMA-Factory/saves:/models \
-p 8000:8000     --ipc=host     vllm/vllm-openai:latest \
--model /models/qwen25coder-32b-instruct-end2end_edit-rft-20k-v1 \
--host 0.0.0.0 --port 8000 --tensor-parallel-size 8 --gpu-memory-utilization 0.95 \
--dtype bfloat16

sudo docker run --gpus all \
-v ~/.cache/huggingface:/root/.cache/huggingface \
-p 8000:8000     --ipc=host     vllm/vllm-openai:latest \
--model R2E-Gym/R2EGym-14B-Agent \
--host 0.0.0.0 --port 8000 --tensor-parallel-size 8 --gpu-memory-utilization 0.95 \
--dtype bfloat16

R2E-Gym/R2EGym-32B-Agent
python -m sglang.launch_server --model-path R2E-Gym/R2EGym-32B-Agent  --host 0.0.0.0 --port 8000 --tp-size 8 

bash src/docker_bash_utils/remove_containers.sh ; time uv run python -m r2egym.agenthub.run.edit runagent_multiple --traj_dir "./traj"  --max_workers 54 --start_idx 0 --k 500 --dataset "r2e-edits/swebench-verified-v1" --split "test" --llm_name openai/r2e-edits/qwen3-14b_end2end_sonnet_combined_maxstep40_sft-exitreason-agent-32k_bz8_epoch2_lr1en5-v1 --exp_name qwen3-14b_end2end_sonnet_combined_maxstep40_sft-exitreason-agent-32k_bz8_epoch2_lr1en5-v1 --temperature 0 --max_steps 40 ;

uv run python src/r2egym/agenthub/run/edit.py runagent_multiple \
  --traj_dir "./traj" \
  --max_workers 54 \
  --start_idx 0 \
  --k 500 \
  --dataset "R2E-Gym/SWE-Bench-Verified" \
  --split "test" \
  --llm_name 'openai/R2E-Gym/R2EGym-32B-Agent' \
  --use_fn_calling False \
  --exp_name r2egym-32B-test-v1 \
  --temperature 0 \
  --max_steps 40

uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/R2E-Gym/R2EGym-32B-Agent'   --use_fn_calling False   --exp_name r2egym-32B-test-v1   --temperature 0   --max_steps 40

uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/R2E-Gym/R2EGym-14B-Agent'   --use_fn_calling False   --exp_name r2egym-14B-test-v1   --temperature 0   --max_steps 40


 python -m sglang.launch_server --model-path r2e-edits/qwen25coder-14b-end2end_sonnet_combined_maxstep40_sft-exitreason-agent-32k_bz8_epoch2_lr1en5-v1  --host 0.0.0.0 --port 8000 --tp-size 8 

 uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 20   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/R2E-Gym/R2EGym-14B-Agent'   --use_fn_calling False   --exp_name r2egym-14B-test-backend-docker-v4   --temperature 0   --max_steps 40   --backend "docker"

  uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 20   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/R2E-Gym/R2EGym-14B-Agent'   --use_fn_calling False   --exp_name r2egym-14B-test-backend-kubernetes-v1   --temperature 0   --max_steps 40   --backend "kubernetes"

    uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 20   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/R2E-Gym/R2EGym-14B-Agent'   --use_fn_calling False   --exp_name r2egym-14B-test-backend-kubernetes-v4   --temperature 0   --max_steps 40   --backend "kubernetes"

        uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/R2E-Gym/R2EGym-14B-Agent'   --use_fn_calling False   --exp_name r2egym-14B-test-backend-kubernetes-fulleval-v1   --temperature 0   --max_steps 40   --backend "kubernetes"
        uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/R2E-Gym/R2EGym-14B-Agent'   --use_fn_calling False   --exp_name r2egym-14B-test-backend-docker-fulleval-v1   --temperature 0   --max_steps 40   --backend "docker"

        uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/R2E-Gym/R2EGym-32B-Agent'   --use_fn_calling False   --exp_name r2egym-32B-test-backend-docker-fulleval-v1   --temperature 0   --max_steps 40   --backend "docker"

##########################
# swe smith
##########################

SWE-bench/SWE-agent-LM-32B
python -m sglang.launch_server --model-path SWE-bench/SWE-agent-LM-32B  --host 0.0.0.0 --port 8000 --tp-size 8 --dtype bfloat16

r2e-edits/SWE-agent-LM-32B
python -m sglang.launch_server --model-path r2e-edits/SWE-agent-LM-32B  --host 0.0.0.0 --port 8000 --tp-size 8


 uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/r2e-edits/SWE-agent-LM-32B'   --use_fn_calling False   --exp_name swesmith-32B-test-backend-docker-fulleval-wrapper-v2   --temperature 0   --max_steps 40   --backend "docker"

  uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/r2e-edits/SWE-agent-LM-32B'   --use_fn_calling False   --exp_name swesmith-32B-test-backend-docker-fulleval-wrapper-topkcondenser-v1   --temperature 0   --max_steps 40   --backend "docker"
  uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/r2e-edits/SWE-agent-LM-32B'   --use_fn_calling False   --exp_name swesmith-32B-test-backend-docker-fulleval-wrapper-topkcondenser-v2   --temperature 0   --max_steps 40   --backend "docker"
  uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/r2e-edits/SWE-agent-LM-32B'   --use_fn_calling False   --exp_name swesmith-32B-test-backend-docker-fulleval-wrapper-topkcondenser-v3   --temperature 0   --max_steps 40   --backend "docker"
    uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/r2e-edits/SWE-agent-LM-32B'   --use_fn_calling False   --exp_name swesmith-32B-test-backend-docker-fulleval-wrapper-topkcondenser-v4   --temperature 0   --max_steps 40   --backend "docker"

bash src/docker_bash_utils/remove_containers.sh; uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/r2e-edits/qwen32B_swesmith_orig-32b_32k-bz8_epoch2_lr1en5-v1'   --use_fn_calling False   --exp_name qwen32B_swesmith_orig-32b_32k-bz8_epoch2_lr1en5-v1-backend-docker-fulleval-wrapper-nocondenser-v1   --temperature 0   --max_steps 40   --backend "docker"

bash src/docker_bash_utils/remove_containers.sh; uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/r2e-edits/qwen25coder-14b_swesmith-traj-R2E-v2_r2e-RFT-trajs-32k_bz8_epoch1_lr1en5-v1'   --use_fn_calling False   --exp_name qwen32B_swesmith_orig-32b_32k-bz8_epoch2_lr1en5-v1-backend-docker-fulleval-wrapper-nocondenser-v1   --temperature 0   --max_steps 40   --backend "docker"


bash src/docker_bash_utils/remove_containers.sh; uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/r2e-edits/qwen25coder-14b_swesmith-traj-R2E-v2_r2e-RFT-trajs-32k_bz8_epoch1_lr1en5-v1'   --use_fn_calling False   --exp_name qwen25coder-14b_swesmith-traj-R2E-v2_r2e-RFT-trajs-32k_bz8_epoch1_lr1en5-v1-normalcondenser-v1 --temperature 0   --max_steps 40   --backend "docker" --swesmith_wrapper False

bash src/docker_bash_utils/remove_containers.sh; uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/$MODEL_NAME'   --use_fn_calling False   --exp_name qwen25coder-14b_swesmith-traj-R2E-v2_r2e-RFT-trajs-32k_bz8_epoch1_lr1en5-v1-normalcondenser-v1 --temperature 0   --max_steps 40   --backend "docker" --swesmith_wrapper False



##########################
# test time encoding
##########################

python -m sglang.launch_server --model-path R2E-Gym/R2EGym-14B-Agent  --host 0.0.0.0 --port 8000 --tp-size 8

max_reward_calc_time
bash src/docker_bash_utils/remove_containers.sh; uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/R2E-Gym/R2EGym-14B-Agent'   --use_fn_calling False   --exp_name R2EGym-14B-Agent-reward-calc-time-nocondenser-v1 --temperature 0   --max_steps 40   --backend "docker" --swesmith_wrapper False --max_reward_calc_time 1200



##########################
qwen 3 debugging
##########################

docker run --gpus all \
    --shm-size 32g \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env 
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path Qwen/Qwen3-14B --host 0.0.0.0 --port 8000 --tp-size 8

lmsysorg/sglang:v0.4.6.post2-cu124
docker run --gpus all \
    --shm-size 32g \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env 
    --ipc=host \
    lmsysorg/sglang:v0.4.6.post2-cu124 \
    python3 -m sglang.launch_server --model-path Qwen/Qwen3-14B --host 0.0.0.0 --port 8000 --tp-size 8


r2e-edits/qwen3-14b-rl-step100
docker run --gpus all \
    --shm-size 32g \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env 
    --ipc=host \
    lmsysorg/sglang:v0.4.6.post2-cu124 \
    python3 -m sglang.launch_server --model-path r2e-edits/qwen3-14b-rl-step100 --host 0.0.0.0 --port 8000 --tp-size 8


##########################
devstral
##########################

vllm serve mistralai/Devstral-Small-2505 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice --tensor-parallel-size 8 --host 0.0.0.0 --port 8000

bash src/docker_bash_utils/remove_containers.sh; uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/mistralai/Devstral-Small-2505'   --use_fn_calling False   --exp_name Devstral-Small-2505_condenser_nonfncalling_maxsteps40-v1 --temperature 0   --max_steps 40   --backend "docker" --swesmith_wrapper False

bash src/docker_bash_utils/remove_containers.sh; uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/mistralai/Devstral-Small-2505'   --use_fn_calling False   --exp_name Devstral-Small-2505_nocondenser_swesmith_maxsteps40-v1 --temperature 0   --max_steps 40   --backend "docker" --swesmith_wrapper True

bash src/docker_bash_utils/remove_containers.sh; uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/mistralai/Devstral-Small-2505'   --use_fn_calling False   --exp_name Devstral-Small-2505_nocondenser_swesmith_maxsteps50-v1 --temperature 0   --max_steps 50 --absolute_max_steps 60  --backend "docker" --swesmith_wrapper True

# above but with higher max tokens
bash src/docker_bash_utils/remove_containers.sh; uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/mistralai/Devstral-Small-2505'   --use_fn_calling False   --exp_name Devstral-Small-2505_nocondenser_swesmith_maxsteps50_maxtokens64k-v1 --temperature 0   --max_steps 100 --absolute_max_steps 110  --backend "docker" --swesmith_wrapper True

bash src/docker_bash_utils/remove_containers.sh; uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/mistralai/Devstral-Small-2505'   --use_fn_calling False   --exp_name Devstral-Small-2505_nocondenser_swesmith_maxsteps50_maxtokens64k-v2 --temperature 0   --max_steps 50 --absolute_max_steps 60  --backend "docker" --swesmith_wrapper True

python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path gold \
    --max_workers 54 \
    --run_id gold-test-v1
    # use --predictions_path 'gold' to verify the gold patches
    # use --run_id to name the evaluation run

python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --split test \
    --predictions_path all_preds.jsonl \
    --max_workers 54 \
    --run_id oh-devstral-verified-v1

##########################
# better scaffolding, devstral, openhands
##########################
bash src/docker_bash_utils/remove_containers.sh; uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/mistralai/Devstral-Small-2505'   --use_fn_calling False   --exp_name Devstral-Small-2505_nocondenser_swesmith_openhands-instance-prompt_maxsteps50_maxtokens32k-v1 --temperature 0   --max_steps 50 --absolute_max_steps 60  --backend "docker" --swesmith_wrapper True
bash src/docker_bash_utils/remove_containers.sh; uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/mistralai/Devstral-Small-2505'   --use_fn_calling False   --exp_name Devstral-Small-2505_nocondenser_swesmith_openhands-instance-prompt-v2_maxsteps50_maxtokens32k-v1 --temperature 0   --max_steps 50 --absolute_max_steps 60  --backend "docker" --swesmith_wrapper True
bash src/docker_bash_utils/remove_containers.sh; uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/mistralai/Devstral-Small-2505'   --use_fn_calling False   --exp_name Devstral-Small-2505_nocondenser_swesmith_openhands-instance-prompt-v2_maxsteps50_maxtokens32k_no-demo-v1 --temperature 0   --max_steps 50 --absolute_max_steps 60  --backend "docker" --swesmith_wrapper True

python -m swebench.harness.run_evaluation  --dataset_name princeton-nlp/SWE-bench_Verified  --split test     --predictions_path all_preds/Devstral-Small-2505_nocondenser_swesmith_openhands-instance-prompt-v2_maxsteps50_maxtokens32k_no-demo-v1.jsonl     --max_workers 48  --run_id Devstral-Small-2505_nocondenser_swesmith_openhands-instance-prompt-v2_maxsteps50_maxtokens32k_no-demo-v2
python -m swebench.harness.run_evaluation  --dataset_name princeton-nlp/SWE-bench_Verified  --split test     --predictions_path all_preds/Devstral-Small-2505_nocondenser_swesmith_openhands-instance-prompt-v2_maxsteps50_maxtokens32k_no-demo-v1.jsonl     --max_workers 48  --run_id Devstral-Small-2505_nocondenser_swesmith_openhands-instance-prompt-v2_maxsteps50_maxtokens32k_no-demo-v2

# with new scaffold
bash src/docker_bash_utils/remove_containers.sh; uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/mistralai/Devstral-Small-2505'   --use_fn_calling False   --exp_name Devstral-Small-2505_new-scaffold-nocondenser_openhands-instance-prompt-v2_maxsteps50_maxtokens32k_no-demo-v1 --temperature 0   --max_steps 50 --absolute_max_steps 60  --backend "docker" --swesmith_wrapper False
bash src/docker_bash_utils/remove_containers.sh; uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/mistralai/Devstral-Small-2505'   --use_fn_calling False   --exp_name Devstral-Small-2505_new-scaffold-nocondenser_openhands-instance-prompt-v2_maxsteps100_maxtokens32k_no-demo-v1 --temperature 0   --max_steps 100 --absolute_max_steps 110  --backend "docker" --swesmith_wrapper False
bash src/docker_bash_utils/remove_containers.sh; uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/mistralai/Devstral-Small-2505'   --use_fn_calling False   --exp_name Devstral-Small-2505_new-scaffold-nocondenser_openhands-instance-prompt-v2_maxsteps100_maxtokens32k_context128k_no-demo-v1 --temperature 0   --max_steps 100 --absolute_max_steps 110  --backend "docker" --swesmith_wrapper False

# 64k window / context, max_iterations = 3
bash src/docker_bash_utils/remove_containers.sh; uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/mistralai/Devstral-Small-2505'   --use_fn_calling False   --exp_name Devstral-Small-2505_new-scaffold-nocondenser_openhands-inst1ance-prompt-v2_maxsteps50_maxtokens64k_maxiter3_no-demo-v2 --temperature 0   --max_steps 50 --max_steps_absolute 60  --backend "docker" --swesmith_wrapper False --max_iterations 3
watch -n 2 uv run python -W ignore analyze_logs.py traj/Devstral-Small-2505_new-scaffold-nocondenser_openhands-instance-prompt-v2_maxsteps100_maxtokens64k_maxiter3_no-demo-v1.jsonl

bash src/docker_bash_utils/remove_containers.sh; uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/mistralai/Devstral-Small-2505'   --use_fn_calling False   --exp_name Devstral-Small-2505_new-scaffold-nocondenser_openhands-instance-prompt-v2_maxsteps100_maxtokens64k_maxiter3_no-demo-v2 --temperature 0   --max_steps 100 --max_steps_absolute 110  --backend "docker" --swesmith_wrapper False --max_iterations 3
bash src/docker_bash_utils/remove_containers.sh; uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/mistralai/Devstral-Small-2505'   --use_fn_calling False   --exp_name Devstral-Small-2505_new-scaffold-nocondenser_openhands-instance-prompt-v2_maxsteps100_maxtokens64k_maxiter1_no-demo-v2 --temperature 0   --max_steps 100 --max_steps_absolute 110  --backend "docker" --swesmith_wrapper False --max_iterations 1

bash src/docker_bash_utils/remove_containers.sh; uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/mistralai/Devstral-Small-2505'   --use_fn_calling False   --exp_name Devstral-Small-2505_new-scaffold-nocondenser_openhands-instance-prompt-v2_maxsteps100_maxtokens64k_maxiter1_no-demo-v3 --temperature 0   --max_steps 100 --max_steps_absolute 110  --backend "docker" --swesmith_wrapper False --max_iterations 1

bash src/docker_bash_utils/remove_containers.sh; uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/mistralai/Devstral-Small-2505'   --use_fn_calling False   --exp_name Devstral-Small-2505_new-scaffold-nocondenser_openhands-instance-prompt-v2_maxsteps100_maxiter1_no-demo-finalrun-v2 --temperature 0   --max_steps 100 --max_steps_absolute 110  --backend "docker" --swesmith_wrapper False --max_iterations 1



https://huggingface.co/r2e-edits/qwen3-32b-rl-step200
# sglang server
python3 -m sglang.launch_server --model-path r2e-edits/qwen3-32b-rl-step200 --host 0.0.0.0 --port 8000 --tp-size 8 --dist-init-addr sgl-dev-0:50000 --nnodes 2 --node-rank 0
python3 -m sglang.launch_server --model-path r2e-edits/qwen3-32b-rl-step200 --host 0.0.0.0 --port 8000 --tp-size 8
docker run --gpus all \
    --shm-size 32g \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env  \
    --ipc=host \
    lmsysorg/sglang:v0.4.6.post2-cu124 \
    python3 -m sglang.launch_server --model-path r2e-edits/qwen3-32b-rl-step200 --host 0.0.0.0 --port 8000 --tp-size 8

bash src/docker_bash_utils/remove_containers.sh; uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/r2e-edits/qwen3-32b-rl-step200'   --use_fn_calling False   --exp_name qwen3-32b-rl-step200_nocondenser_openhands-instance-prompt-v2_maxsteps100_maxtokens64k_maxiter1_no-demo-v3 --temperature 0   --max_steps 100 --max_steps_absolute 110  --backend "docker" --swesmith_wrapper False --max_iterations 1

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve r2e-edits/qwen3-32b-rl-step200 --tensor-parallel-size 8 --host 0.0.0.0 --port 8000 --max-model-len 65536


bash src/docker_bash_utils/remove_containers.sh; uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name "openai/r2e-edits/qwen3-32b-rl-step200"   --use_fn_calling False   --exp_name r2egym-qwen3-32b-v1   --temperature 0   --max_steps 100 --max_steps_absolute 110  --backend "docker"   --condense_history False   --swesmith_wrapper False   --max_reward_calc_time 1200

bash src/docker_bash_utils/remove_containers.sh; uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 32   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name "openai/r2e-edits/qwen3-32b-rl-step200"   --use_fn_calling False   --exp_name r2egym-qwen3-32b-v5   --temperature 0   --max_steps 100 --max_steps_absolute 110  --backend "docker"   --condense_history False   --swesmith_wrapper False   --max_reward_calc_time 1200


bash src/docker_bash_utils/remove_containers.sh; uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/mistralai/Devstral-Small-2505'   --use_fn_calling False   --exp_name Devstral-Small-2505_new-scaffold-nocondenser_openhands-instance-prompt-v2_maxsteps100_maxtokens64k_maxiter1_no-demo-v4 --temperature 0   --max_steps 100 --max_steps_absolute 110  --backend "docker" --swesmith_wrapper False --max_iterations 1
bash src/docker_bash_utils/remove_containers.sh; uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name 'openai/mistralai/Devstral-Small-2505'   --use_fn_calling False   --exp_name Devstral-Small-2505_new-scaffold-nocondenser_openhands-instance-prompt-v2_maxsteps100_maxtokens64k_maxiter5_no-demo-v4 --temperature 0   --max_steps 100 --max_steps_absolute 110  --backend "docker" --swesmith_wrapper False --max_iterations 5


bash src/docker_bash_utils/remove_containers.sh; time uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 24   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name "openai/r2e-edits/qwen3-32b-rl-step200"   --use_fn_calling False   --exp_name r2egym-qwen3-32b-maxtoken64k-nworkers24-rewardcalc1200-maxiter3-version1-v2   --temperature 0   --max_steps 100 --max_steps_absolute 110  --backend "docker"   --condense_history False   --swesmith_wrapper False   --max_reward_calc_time 1200 --max_iterations 3 --version "v1"

# lets use 70k tokens
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve r2e-edits/qwen3-32b-rl-step200   --tensor-parallel-size 8   --max-model-len 70000   --hf-overrides '{"max_position_embeddings": 70000}' --gpu-memory-utilization 0.95 --dtype bfloat16 --enable-prefix-caching
bash src/docker_bash_utils/remove_containers.sh; time uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 10   --start_idx 0   --k 10   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name "openai/r2e-edits/qwen3-32b-rl-step200"   --use_fn_calling False   --exp_name r2egym-qwen3-32b-maxtoken64k-vllm70k-nworkers10-k10-rewardcalc1200-maxiter1-version1-v2   --temperature 0   --max_steps 100 --max_steps_absolute 110  --backend "docker"   --condense_history False   --swesmith_wrapper False   --max_reward_calc_time 1200 --max_iterations 1 --version "v1"

bash src/docker_bash_utils/remove_containers.sh; time uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 10   --start_idx 0   --k 10   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name "openai/r2e-edits/qwen3-32b-rl-step200"   --use_fn_calling False   --exp_name r2egym-qwen3-32b-maxtoken64k-vllm70k-nworkers10-k10-rewardcalc1200-maxiter3-version1-v2   --temperature 0   --max_steps 100 --max_steps_absolute 110  --backend "docker"   --condense_history False   --swesmith_wrapper False   --max_reward_calc_time 1200 --max_iterations 3 --version "v1"

# on master branch -- all 500 samples
bash src/docker_bash_utils/remove_containers.sh; time uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 24   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name "openai/r2e-edits/qwen3-32b-rl-step200"   --use_fn_calling False   --exp_name r2egym-qwen3-32b-maxtoken64k-nworkers24-maxiter1-masterbranch-v1  --temperature 0   --max_steps 100 --max_steps_absolute 110  --backend "docker"   --condense_history False   --swesmith_wrapper False --max_iterations 1
bash src/docker_bash_utils/remove_containers.sh; time uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name "openai/r2e-edits/qwen3-32b-rl-step200"   --use_fn_calling False   --exp_name r2egym-qwen3-32b-maxtoken64k-nworkers54-maxiter1-swesmithdockerbranch-v1  --temperature 0   --max_steps 100 --max_steps_absolute 100  --backend "docker"   --condense_history False   --swesmith_wrapper False --max_reward_calc_time 1200

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve r2e-edits/qwen3-32b-rl-step200   --tensor-parallel-size 8   --max-model-len 70000   --hf-overrides '{"max_position_embeddings": 70000}' --dtype bfloat16 --enable-prefix-caching
sudo docker run --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface -p 8000:8000     --ipc=host     vllm/vllm-openai:latest --model /models/qwen25coder-14b-instruct-end2end_sonnet_combined_maxstep40_30-rft-20k_bz8_epoch5_lr1en4-v1/checkpoint-900 --host 0.0.0.0 --port 8000 --tensor-parallel-size 8 --gpu-memory-utilization 0.95 --dtype bfloat16 --enable-prefix-caching

bash src/docker_bash_utils/remove_containers.sh; time uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name "openai/r2e-edits/qwen3-32b-rl-step200"   --use_fn_calling False   --exp_name r2egym-qwen3-32b-maxtoken64k-nworkers54-maxiter1-swesmithdockerbranch-fixededitor-v1  --temperature 0   --max_steps 100 --max_steps_absolute 100  --backend "docker"   --condense_history False   --swesmith_wrapper False --max_reward_calc_time 1200
bash src/docker_bash_utils/remove_containers.sh; time uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name "openai/r2e-edits/qwen3-32b-rl-step200"   --use_fn_calling False   --exp_name r2egym-qwen3-32b-maxtoken64k-nworkers54-maxiter5-swesmithdockerbranch-fixededitor-v1  --temperature 0   --max_steps 100 --max_steps_absolute 100  --backend "docker"   --condense_history False   --swesmith_wrapper False --max_reward_calc_time 1200 --max_iterations 5

bash src/docker_bash_utils/remove_containers.sh; time uv run python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name "openai/r2e-edits/qwen3-32b-rl-step200"   --use_fn_calling False   --exp_name r2egym-qwen3-32b-maxtoken64k-nworkers54-maxiter1-swesmithdockerbranch-fixededitor-temp1-v1  --temperature 0.1   --max_steps 100 --max_steps_absolute 100  --backend "docker"   --condense_history False   --swesmith_wrapper False --max_reward_calc_time 1200

########################################################
# hybrid scaling
########################################################

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve r2e-edits/qwen3-32b-rl-step200   --tensor-parallel-size 8   --max-model-len 70000   --hf-overrides '{"max_position_embeddings": 70000}'

time python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/SWE-Bench-Verified"   --split "test"   --llm_name "openai/r2e-edits/qwen3-32b-rl-step200"   --use_fn_calling False   --exp_name r2egym-qwen3-rl-100-steps --temperature 0   --max_steps_absolute 100   --backend "docker"   --condense_history False   --swesmith_wrapper False   --max_reward_calc_time 1200

time python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 54   --start_idx 0   --k 500   --dataset "R2E-Gym/R2E-Gym-Subset"   --split "train"   --llm_name "openai/r2e-edits/qwen3-32b-rl-step200"   --use_fn_calling False   --exp_name qwen3-rl-100-absolute-40-maxsteps-temp0n5-r2egym-maxworkers54-scaffold-v1 --temperature 0.5   --max_steps_absolute 100   --backend "docker"   --condense_history False   --swesmith_wrapper False

bash src/docker_bash_utils/remove_containers.sh; time python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 32   --start_idx 0   --k 4000   --dataset "R2E-Gym/R2E-Gym-Subset"   --split "train"   --llm_name "openai/r2e-edits/qwen3-32b-rl-step200"   --use_fn_calling False   --exp_name qwen3-rl-100-absolute-40-maxsteps-temp5-r2egym-maxworkers54-scaffold-v4 --temperature 0.5   --max_steps_absolute 100   --backend "docker"   --condense_history False   --swesmith_wrapper False

bash src/docker_bash_utils/remove_containers.sh; time python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 32   --start_idx 0   --k 500   --dataset "R2E-Gym/R2E-Gym-Subset"   --split "train"   --llm_name "openai/r2e-edits/qwen3-32b-rl-step200"   --use_fn_calling False   --exp_name qwen3-rl-100-absolute-40-maxsteps-temp5-r2egym-maxworkers54-scaffold-k500-v2 --temperature 0.5   --max_steps_absolute 100   --backend "docker"   --condense_history False   --swesmith_wrapper False --prepull_images True

bash src/docker_bash_utils/remove_containers.sh; time python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 32   --start_idx 501   --k 1000   --dataset "R2E-Gym/R2E-Gym-Subset"   --split "train"   --llm_name "openai/r2e-edits/qwen3-32b-rl-step200"   --use_fn_calling False   --exp_name qwen3-rl-100-absolute-40-maxsteps-temp5-r2egym-maxworkers54-scaffold-k500-1000-v2 --temperature 0.5   --max_steps_absolute 100   --backend "docker"   --condense_history False   --swesmith_wrapper False --prepull_images True


bash src/docker_bash_utils/remove_containers.sh; time python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 32   --start_idx 501   --k 500   --dataset "R2E-Gym/R2E-Gym-Subset"   --split "train"   --llm_name "openai/r2e-edits/qwen3-32b-rl-step200"   --use_fn_calling False   --exp_name qwen3-rl-100-absolute-40-maxsteps-temp5-r2egym-maxworkers54-scaffold-k500-1000-v1 --temperature 0.5   --max_steps_absolute 100   --backend "docker"   --condense_history False   --swesmith_wrapper False --prepull_images True

bash src/docker_bash_utils/remove_containers.sh; time python src/r2egym/agenthub/run/edit.py runagent_multiple   --traj_dir "./traj"   --max_workers 32   --start_idx 0   --k 1500   --dataset "R2E-Gym/R2E-Gym-Subset"   --split "train"   --llm_name "openai/r2e-edits/qwen3-32b-rl-step200"   --use_fn_calling False   --exp_name qwen3-rl-100-absolute-40-maxsteps-temp5-r2egym-maxworkers54-scaffold-k500-1000-v1-run2 --temperature 0.9  --max_steps_absolute 100   --backend "docker"   --condense_history False   --swesmith_wrapper False --prepull_images True



########################################################
# verifier data prep
########################################################

python verifier_data_prep_optimized.py create_optimized
   --verifier_traj_dir="./verifier_traj" --output_dataset
  _path="./verifier-deepswe_dataset_2582_balanced_64k-v1"
  --push_to_hub=True --debug=False --max_workers=8

python verifier_data_prep_optimized.py create_optimized --push_to_hub=True --hub_repo_name="deepswe-verifier-2582-v1" --max_workers=54

python verifier_data_prep.py create --push_to_hub=True --hub_repo_name="deepswe-verifier-2582-v1" --max_workers=54

python verifier_data_prep_condense.py create --push_to_hub=True --hub_repo_name="deepswe-verifier-debug-condense-v1" --max_workers=54 --filter_method="agent_priority"

python verifier_data_prep_condense.py create --push_to_hub=True --hub_repo_name="r2e-edits/deepswe-verifier-exitreason-agent-priority-v2" --max_workers=54 --filter_method="mixed_pairs"


python verifier_data_prep_condense.py create --push_to_hub=True --hub_repo_name="r2e-edits/deepswe-swebv-eval-n16-verifier-v1" --max_workers=54 --filter_method="none" --verifier_traj_dir="./deepswe"


export DS_SKIP_CUDA_CHECK=1 

export DS_SKIP_CUDA_CHECK=1 
export DISABLE_VERSION_CHECK=1  # if necessary
export NCCL_TIMEOUT=3600000  # 30 minutes
export NCCL_BLOCKING_WAIT=1



DISABLE_VERSION_CHECK=1
deepspeed src/train.py \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen2.5-Coder-14B-Instruct \
    --dataset deepswe-verifier-only-matching-pairs-v1 \
    --template qwen \
    --output_dir output/deepswe-verifier-only-matching-pairs-v1 \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 74752 \
    --drop_exceed_length_data True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 2000 \
    --learning_rate 1e-5 \
    --num_train_epochs 2 \
    --plot_loss \
    --save_only_model True \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --bf16 True \
    --flash_attn fa2 \
    --gradient_checkpointing True \
    --seed 42 \
    --sequence_parallel_size 8 \
    --packing False \
    --preprocessing_num_workers 64 \
    --finetuning_type lora \
    --lora_rank 64 \
    --lora_target all \
    --ddp_timeout 180000000 \
    --warmup_ratio 0.05 

 --use_unsloth_gc True \

    --enable_liger_kernel True \
    ### train
flash_attn: fa2
enable_liger_kernel: true
use_unsloth_gc: true
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 1.0e-5
num_train_epochs: 5.0
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
ddp_timeout: 180000000

########################################################
# verifier eval
########################################################

vllm serve Qwen/Qwen2.5-Coder-14B-instruct --enable-lora --lora-modules a=/home/ubuntu/360-LLaMA-Factory/output/deepswe-verifier-only-matching-pairs-v1 --port 8000 --gpu-memory-utilization 0.95 --dtype bfloat16 --max-lora-rank 64 --tensor-parallel-size 8  --max-model-len 76800   --hf-overrides '{"max_position_embeddings": 76800}'


VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve Qwen/Qwen2.5-Coder-14B-instruct --max-model-len 76800 --hf-overrides '{"max_position_embeddings": 76800}' --enable-lora --lora-modules a=/home/ubuntu/360-LLaMA-Factory/output/deepswe-verifier-only-matching-pairs-v1 --port 8000 --dtype bfloat16 --max-lora-rank 64 --tensor-parallel-size 8

r2e-edits/deepswe-verifier-qwen25-14b-merged-v1

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve r2e-edits/deepswe-verifier-qwen25-14b-merged-v1 --max-model-len 76800 --hf-overrides '{"max_position_embeddings": 76800}' --port 8000 --dtype bfloat16 --tensor-parallel-size 8

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve Qwen/Qwen2.5-Coder-14B-instruct --max-model-len 76800 --hf-overrides '{"max_position_embeddings": 76800}' --port 8000 --dtype bfloat16 --tensor-parallel-size 8


llm serve Qwen/Qwen2.5-Coder-14B-instruct --max-model-len 76800 --hf-overrides '{"max_position_embeddings": 76800}' --port 8000 --dtype bfloat16 --tensor-parallel-size 8


--rope-scaling '{"rope_type":"yarn","factor":2.288818359375,"original_max_position_embeddings":32768}' \
  --max-model-len 75000

vllm serve Qwen/Qwen3-14B --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072  --tensor-parallel-size 8



Qwen/Qwen3-8B

vllm serve Qwen/Qwen2.5-Coder-14B-instruct \
  --enable-lora \
  --lora-modules a=/home/ubuntu/360-LLaMA-Factory/output/deepswe-verifier-only-matching-pairs-v1 \
  --port 8000 \
  --gpu-memory-utilization 0.95 \
  --dtype bfloat16 \
  --max-lora-rank 64 \
  --tensor-parallel-size 8 \
  --rope-scaling '{"rope_type":"yarn","factor":2.288818359375,"original_max_position_embeddings":32768}' \
  --max-model-len 75000
  
python verifier_eval.py --llm_name "hosted_vllm//home/ubuntu/360-LLaMA-Factory/output/deepswe-verifier-only-matching-pairs-v1" --eval_dataset "r2e-edits/deepswe-swebv-eval-n16-verifier-v1" --out_file 'deepswe-verifier-only-matching-pairs-qwen25coder-14b-lora64_N16_traindata_p2p-14B-75k-lr1en5-v1.csv'

python verifier_eval.py --llm_name "hosted_vllm/a" --eval_dataset "r2e-edits/deepswe-swebv-eval-n16-verifier-v1" --out_file 'deepswe-verifier-only-matching-pairs-qwen25coder-14b-lora64_N16_traindata_p2p-14B-75k-lr1en5-v1.csv'

python verifier_eval.py --llm_name "hosted_vllm/r2e-edits/deepswe-verifier-qwen25-14b-merged-v1" --eval_dataset "r2e-edits/deepswe-swebv-eval-n16-verifier-v1" --out_file 'deepswe-verifier-only-matching-pairs-qwen25coder-14b-lora64_N16_traindata_p2p-14B-75k-lr1en5-v1.csv'

python verifier_eval.py --llm_name "hosted_vllm/Qwen/Qwen2.5-Coder-14B-instruct" --eval_dataset "r2e-edits/deepswe-swebv-eval-n16-verifier-v1" --out_file 'deepswe-verifier-only-matching-pairs-qwen25coder-14b-lora64_N16_traindata_p2p-14B-75k-lr1en5-v1.csv'

python verifier_eval.py --llm_name "hosted_vllm/Qwen/Qwen3-14B" --eval_dataset "r2e-edits/deepswe-swebv-eval-n16-verifier-v1" --out_file 'deepswe-verifier-only-matching-pairs-qwen25coder-14b-lora64_N16_traindata_p2p-14B-75k-lr1en5-v1.csv'


vllm serve Qwen/Qwen3-14B --max-model-len 76800 --hf-overrides '{"max_position_embeddings": 76800}' --enable-lora --lora-modules a=/home/ubuntu/360-LLaMA-Factory/output/verifier --port 8000 --dtype bfloat16 --max-lora-rank 64 --tensor-parallel-size 8  
python verifier_eval.py --llm_name "hosted_vllm/a" --eval_dataset "r2e-edits/deepswe-swebv-eval-n16-verifier-v1" --out_file 'deepswe-verifier-only-matching-pairs-qwen3-14b-lora64_N16_traindata_p2p-14B-75k-lr1en5-v1.csv'
