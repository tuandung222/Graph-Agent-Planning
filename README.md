# Parallel-Agent-Workflow-Planning
<div align="center">
  <a href='https://arxiv.org/abs/2510.25320'><img src='https://img.shields.io/badge/Paper GAP-arXiv-d63031?logo=arxiv&logoColor=white'></a>
  <!-- <a href='https://huggingface.co/collections/PersonalAILab/afm-689200e11d0b21a67c015ba8'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Models-Huggingface-yellow'></a> -->
  <a href='https://huggingface.co/datasets/Chtistina777/GAP-MHQA-SFT-7K'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-Huggingface-yellow'></a>
</div>
This is the repository for our paper "GAP: Graph-Based Agent Planning with Parallel Tool Use and Reinforcement Learning". We propose a new paradigm for LLM reasoning that enables parallel workflow planning and end-to-end RL trianing within a single model. The fine-tuning dataset and model checkpoint will be released soon.
<div align="center">
  <img src="./assets/gap1.png" width="90%" height="auto" />
</div>

## Training
### Supervised Fine-tuning
#### 1. Env Setup
```bash
conda create -n llama_factory python=3.10
conda activate llama_factory
pip install deepspeed
pip install swanlab
cd LLaMA-Factory
pip install -e '.[torch,metrics]'
```

#### 2. Prepare SFT Dataset
Download SFT Dataset for MHQA Agent:
```py  
python ./Agent/data/mhqa_agent/download.py 
```

Add the downloaded dataset filepath to `LLaMA-Factory/data/dataset_info.json`, for example:
```json
"mhqa_agent_sft": {
  "file_name": "path/to/downloaded/MHQA-SFT-Dataset/MHQA-SFTDataset.json"
}
```

#### 3. Start training with default parameters
The training scripts are list in `./train`. 
Example of sft for MHQA agent:
```bash
bash ./Agent/train/mhqa_agent/sft/sft_qwen2.5_3b.sh
```

Note `DATA_DATA` in the training bash script should be the key in `LLaMA-Factory/data/dataset_info.json`, like `web_agent_sft`, `mhqa_agent_sft`.

Logs output to output_dir/training.log. We use [SwanLab](https://swanlab.cn/) for visualization (requires setup):
```bash
--swanlab_api_key xxx  # Your SWANLAB_API_KEY
--swanlab_project xxx  # Your SWANLAB_PROJECT
```

Key Configurable Parameters
```
ignore_observation=true # Whether to mask content within special tokens
ignore_observation_token=observation # Specify special token
```
**Note: Check if special tokens are properly masked and data length is appropriate after starting.**

### Reinforement Learning
#### 1. Env Setup
```bash
# Create virtual environment. 
conda create -n parallel-agent python=3.10.14 -y
conda activate parallel-agent

# Phase 1
pip install symeval@git+https://github.com/tongyx361/symeval.git@54c1a844ea4a6db486c5af8b5b4d2f383224a83b
pip install latex2sympy2==1.9.1
pip install --force-reinstall antlr4-python3-runtime==4.9.3

# Phase 2
cd verl
pip install -r requirements.txt

# Phase 3
pip install --force-reinstall protobuf==5.29.5
pip install --force-reinstall --no-deps grpcio-status==1.71.0 selenium==4.33.0

# Phase 4
cd ..
git clone https://github.com/NVIDIA/apex.git  
cd apex
python -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..

# Phase 5
cd verl
pip install -r requirements_sglang.txt
cd ..
```


#### 2. Tool usage
##### 2.1 Search Servers
We have developed two server-side components to support web interactions:
- A web search server
- A page crawling server

For detailed deployment instructions, please refer to `Agent/tool_servers/tool_server_readme.md`.

#### 3. Configuration
1. Edit the `environment.sh` file and fill in your API keys and other required credentials
2. Apply the environment settings:
```bash
source environment.sh
```

#### 4. Dataset Processing
The `./Agent/data/README.md` contains scripts and instructions for processing search agent model related data.

The final mhqa_agent dataset format is shown below and stored in .parquet: 
```python
{
    "data_source": data_source,
    "prompt": [
        {"role": "user", "content": sys_prompt + question}
    ],
    "reward_model": {
        "ground_truth": {"target": answer}
    },
    "extra_info": {
        "need_tools_kwargs": True,
        "question": question,
        "answer": answer,
        "tools_kwargs": tools_kwargs
    }
}
```


#### 5. Training
To start a training run:

1. All Agentic-RL script examples are listed:
    -  MHQA Agent: `./Agent/train/mhqa_agent/rl/train_dapo_mhqa_agent.sh`
2. Edit the corresponding script to specify your downloaded dataset and model
3. Make sure you have already fill in the `environment.sh` and source
4. All tool configs are listed and have been specified in training scripts: 
    - web_search and crawl_page: `verl/verl/tools/config/search_tool_config/training_servers_config.yaml`
    - wiki_search: `verl/verl/tools/config/search_tool_config/wiki_rag_config.yaml`
5. Execute the training script like:
```bash
bash ./Agent/train/mhqa_agent/rl/train_dapo_mhqa_agent_wiki.sh
```


## Evaluation
### Multi Hop QA (MHQA) Evaluation
1. To evaluate MHQA datasets, you should first download the MHQA-Agent-3B-rl model and test datasets
2. Then fill the corresponding dataset and model in scripts below and run
  ```bash
  bash ./Agent/evaluation/mhqa_agent/eval_mhqa_agent.sh
  ```


# Acknowledgement
We would like to express our sincere gratitude to the original authors and contributors of LLaMA-Factory and verl, an excellent open-source project that provided a solid foundation for our work. Our implementation has been adapted from the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [verl-agent](https://github.com/langfengQ/verl-agent), [AFM](https://github.com/OPPO-PersonalAI/Agent_Foundation_Models) and [Search-R1](https://github.com/PeterGriffinJin/Search-R1). 
For the VeRL framework, we have enhanced it with parallel tool calling in sgalng rollout phase for RL training, along with specified reward design, and related supporting features.