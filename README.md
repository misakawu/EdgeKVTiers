# EdgeKVTiers Reproducible Environment

本文档用于在一台新服务器上复现本项目当前环境，包括代码、三个 Conda 环境、第三方源码、模型、ShareGPT 数据和基础验证命令。路径示例沿用当前机器：`/DATACENTER3/zhenxiang.wang/work/EdgeKVTiers`；迁移到其他服务器时可替换为自己的工作目录，但需要同步修改命令中的绝对路径。

## 1. 目标环境

当前已验证环境如下：

| 项 | 当前值 |
| --- | --- |
| 项目目录 | `/DATACENTER3/zhenxiang.wang/work/EdgeKVTiers` |
| Conda 根目录 | `/DATACENTER3/zhenxiang.wang/miniforge3` |
| 数据目录 | `/DATACENTER3/zhenxiang.wang/data` |
| GPU | `3 x NVIDIA GeForce RTX 2080 Ti` |
| 单卡显存 | `11264 MiB` |
| Compute capability | `7.5` |
| NVIDIA driver | `525.105.17` |
| `nvidia-smi` CUDA | `12.0` |
| 记录日期 | `2026-06-17` |

当前机器可以复现 H0 vLLM prefix-cache smoke、H3 adapter smoke、H0/H1/H2/H4/H5 仿真和 KIVI/H2O baseline 环境。正式大规模真实引擎实验建议使用显存更大、驱动更新且 GPU 空闲的服务器。

## 2. 新服务器前置条件

建议先准备：

- Linux x86_64 服务器。
- NVIDIA GPU 和可用 `nvidia-smi`。
- NVIDIA driver 至少能运行 PyTorch CUDA 12.x wheel。当前复现过的机器是 driver `525.105.17`。
- `git`、`git-lfs`、`gcc/g++`、`make`、`curl` 或等价工具。
- Miniforge/Conda。
- 能访问 GitHub、Hugging Face、PyPI；如 Hugging Face 受限，可使用 ModelScope 下载 Qwen 模型。

安装系统工具示例：

```bash
# Ubuntu/Debian 示例
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential curl

git lfs install
```

安装 Miniforge 示例：

```bash
curl -L -o Miniforge3-Linux-x86_64.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /DATACENTER3/zhenxiang.wang/miniforge3
source /DATACENTER3/zhenxiang.wang/miniforge3/etc/profile.d/conda.sh
conda config --set channel_priority flexible
```

## 3. 获取项目代码

推荐目录布局：

```bash
mkdir -p /DATACENTER3/zhenxiang.wang/work /DATACENTER3/zhenxiang.wang/data
cd /DATACENTER3/zhenxiang.wang/work
```

克隆项目：

```bash
# SSH
git clone git@github.com:misakawu/EdgeKVTiers.git

# 或 HTTPS
# git clone https://github.com/misakawu/EdgeKVTiers.git

cd EdgeKVTiers
```

后续命令默认在项目根目录执行。

## 4. 创建三个 Conda 环境

本项目拆成三个环境，避免 vLLM/LMCache、KIVI、H2O 的依赖互相覆盖。

| 环境名 | Python | 用途 |
| --- | --- | --- |
| `h3-lmcache-blog` | `3.12.13` | vLLM + LMCache、H0 real-engine smoke、H3 adapter smoke |
| `edgekv-h2o` | `3.10.20` | H2O baseline / Heavy-Hitter 相关代码 |
| `edgekv-kivi` | `3.10.20` | KIVI KV-cache quantization baseline |

### 4.1 `h3-lmcache-blog`

```bash
conda create -y -n h3-lmcache-blog python=3.12
conda activate h3-lmcache-blog
python -m pip install --upgrade pip setuptools wheel

python -m pip install vllm==0.8.5.post1 lmcache==0.3.15 transformers==5.11.0 \
  modelscope==1.37.1 openai==2.41.1 numpy==1.26.4

# vLLM 0.8.5.post1 会安装匹配的 torch==2.6.0、torchvision==0.21.0、torchaudio==2.6.0。
# 若从旧环境升级后 pip check 报旧观测包冲突，可移除非 H1 必需包：
python -m pip uninstall -y opentelemetry-exporter-prometheus google-api-core opencensus
```

验证：

```bash
python - <<'PY'
import torch, vllm, lmcache, transformers, numpy
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('vllm', vllm.__version__)
print('lmcache', getattr(lmcache, '__version__', 'unknown'))
print('transformers', transformers.__version__)
print('numpy', numpy.__version__)
import lmcache.c_ops
print('lmcache.c_ops ok')
PY
```

已验证关键版本：

- `torch == 2.6.0+cu124`
- `vllm == 0.8.5.post1`
- `lmcache == 0.3.15`
- `transformers == 5.11.0`
- `numpy == 1.26.4`

### 4.2 `edgekv-h2o`

```bash
conda create -y -n edgekv-h2o python=3.10
conda activate edgekv-h2o
python -m pip install --upgrade pip setuptools wheel

python -m pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install transformers==4.43.1 datasets==5.0.0 numpy==2.2.6 \
  pandas==2.3.3 accelerate==1.14.0 sentencepiece==0.2.1 psutil==7.2.2
```

验证：

```bash
python - <<'PY'
import torch, transformers, datasets, numpy, pandas
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('transformers', transformers.__version__)
print('datasets', datasets.__version__)
print('numpy', numpy.__version__)
print('pandas', pandas.__version__)
PY
```

已验证关键版本：

- `torch == 2.4.1+cu121`
- `transformers == 4.43.1`
- `datasets == 5.0.0`
- `numpy == 2.2.6`
- `pandas == 2.3.3`

### 4.3 `edgekv-kivi`

```bash
conda create -y -n edgekv-kivi python=3.10
conda activate edgekv-kivi
python -m pip install --upgrade pip setuptools wheel ninja

python -m pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install transformers==4.43.1 datasets==5.0.0 numpy==2.2.6 \
  pandas==2.3.3 accelerate==1.14.0 sentencepiece==0.2.1 \
  packaging==24.0 attributedict==0.3.0 fastchat==0.1.0 protobuf ipdb toml
```

KIVI 源码下载后再执行 editable 安装，见第 5 节。当前已验证 `kivi == 0.1.0` 以 editable 方式安装到该环境。

验证：

```bash
python - <<'PY'
import torch, transformers, datasets, numpy, pandas
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('transformers', transformers.__version__)
print('datasets', datasets.__version__)
print('numpy', numpy.__version__)
print('pandas', pandas.__version__)
PY
```

## 5. 下载第三方源码

第三方源码放在 `third_party/`。

| 名称 | 来源网站 | 本地路径 | 固定 commit |
| --- | --- | --- | --- |
| H2O | `https://github.com/FMInference/H2O.git` | `third_party/H2O` | `ac75c2a8a9e76832b2a4139b9363373b56336bfb` |
| KIVI | `https://github.com/jy-yuan/KIVI.git` | `third_party/KIVI` | `876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6` |

下载命令：

```bash
mkdir -p third_party

git clone https://github.com/FMInference/H2O.git third_party/H2O
git -C third_party/H2O checkout ac75c2a8a9e76832b2a4139b9363373b56336bfb

git clone https://github.com/jy-yuan/KIVI.git third_party/KIVI
git -C third_party/KIVI checkout 876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6
```

安装 KIVI 到 `edgekv-kivi`：

```bash
conda activate edgekv-kivi
python -m pip install -e third_party/KIVI
```

可选：编译 KIVI CUDA 扩展。只有需要运行 `kivi_gemv` CUDA kernel 时才需要这一步；如果服务器没有匹配 CUDA toolkit 或 `nvcc`，可以先跳过。

```bash
conda activate edgekv-kivi
cd third_party/KIVI/quant
python setup.py build_ext --inplace
cd ../../..
```

第三方仓库自带 requirements 仅作参考。本项目复现以第 4 节记录的实际已验证版本为准；例如 `third_party/KIVI/requirements.txt` 中有旧版锁定项，但当前实际可用环境是 `torch 2.4.1 + transformers 4.43.1`。

Python 包来源：

| 包 | 当前使用版本 | 网站 |
| --- | --- | --- |
| vLLM | `0.6.6.post1` | `https://github.com/vllm-project/vllm` / `https://pypi.org/project/vllm/` |
| LMCache | `0.3.15` | `https://github.com/LMCache/LMCache` / `https://pypi.org/project/lmcache/` |
| PyTorch | `2.5.1+cu124`、`2.4.1+cu121` | `https://pytorch.org/` |
| Transformers | `5.11.0`、`4.43.1` | `https://github.com/huggingface/transformers` / `https://pypi.org/project/transformers/` |
| datasets | `5.0.0` | `https://github.com/huggingface/datasets` / `https://pypi.org/project/datasets/` |

## 6. 下载模型

模型放在项目内 `models/`。当前使用两个模型：

| 模型 | 来源网站 | 下载标识 | 本地路径 | 当前大小 | 用途 |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B-Instruct | `https://huggingface.co/Qwen/Qwen2.5-7B-Instruct` | `Qwen/Qwen2.5-7B-Instruct` | `models/Qwen2.5-7B-Instruct` | `15G` | H0 vLLM prefix-cache 回放主模型 |
| facebook/opt-125m | `https://huggingface.co/facebook/opt-125m` | `facebook/opt-125m` | `models/facebook_opt_125m` | `241M` | 小模型 smoke / 低资源验证 |

Hugging Face 下载：

```bash
conda activate h3-lmcache-blog
mkdir -p models

huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir models/Qwen2.5-7B-Instruct \
  --local-dir-use-symlinks False

huggingface-cli download facebook/opt-125m \
  --local-dir models/facebook_opt_125m \
  --local-dir-use-symlinks False
```

如果 Hugging Face 网络不可用，可从 ModelScope 下载 Qwen：

- `https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct`

```bash
conda activate h3-lmcache-blog
modelscope download --model Qwen/Qwen2.5-7B-Instruct \
  --local_dir models/Qwen2.5-7B-Instruct
```

也可以使用 `git-lfs`：

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct models/Qwen2.5-7B-Instruct
git clone https://huggingface.co/facebook/opt-125m models/facebook_opt_125m
```

下载后检查：

```bash
ls models/Qwen2.5-7B-Instruct/config.json
ls models/Qwen2.5-7B-Instruct/model-00001-of-00004.safetensors
ls models/facebook_opt_125m/config.json
```

## 7. 下载 ShareGPT 数据

当前默认数据文件：

- 本机路径：`/DATACENTER3/zhenxiang.wang/data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json`
- 当前大小：`640M`
- 主要来源：`https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered`
- 备用来源：`https://huggingface.co/datasets/RyokoAI/ShareGPT52K`

下载当前代码默认文件名：

```bash
mkdir -p /DATACENTER3/zhenxiang.wang/data

huggingface-cli download anon8231489123/ShareGPT_Vicuna_unfiltered \
  ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json \
  --repo-type dataset \
  --local-dir /DATACENTER3/zhenxiang.wang/data \
  --local-dir-use-symlinks False
```

如果下载得到的文件名或数据目录不同，可以在运行时通过 `--trace-path` 指定实际路径。也可以建立软链接让代码默认路径可用：

```bash
ln -s /your/data/path/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json \
  /DATACENTER3/zhenxiang.wang/data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json
```

用 `datasets` 验证数据源：

```bash
conda activate edgekv-h2o
python - <<'PY'
from datasets import load_dataset

ds = load_dataset('anon8231489123/ShareGPT_Vicuna_unfiltered')
print(ds)
# 备用：ds = load_dataset('RyokoAI/ShareGPT52K')
PY
```

项目中引用该数据的位置：

- `h0/run_h0_vllm.py`
- `h0/configs/vllm_qwen25_7b_h0.json`
- `pre实验/sim.py`
- `论文规划/实验文档/*`

## 8. 验证项目环境

### 8.1 H3 adapter smoke

```bash
conda activate h3-lmcache-blog
python h3/run_h3.py --trace-source sharegpt --event-limit 100 --out h3/out/adapter_smoke
```

预期输出目录包含：

- `h3/out/adapter_smoke/h3_adapter_contract.json`
- `h3/out/adapter_smoke/h3_hook_events.sample.jsonl`
- `h3/out/adapter_smoke/summary.json`

### 8.2 H0 vLLM prefix-cache smoke

启动 vLLM server。RTX 2080 Ti 11 GiB 机器上建议 `tensor-parallel-size=2` 或 `3`；更大显存机器可以按实际 GPU 调整。

```bash
conda activate h3-lmcache-blog

CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
  --model /DATACENTER3/zhenxiang.wang/work/EdgeKVTiers/models/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --dtype half \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching
```

另开终端运行回放：

```bash
conda activate h3-lmcache-blog
cd /DATACENTER3/zhenxiang.wang/work/EdgeKVTiers

python h0/run_h0_vllm.py \
  --endpoint http://127.0.0.1:8000 \
  --trace-path /DATACENTER3/zhenxiang.wang/data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json \
  --max-sessions 200 \
  --max-requests 1024 \
  --max-model-len 1024 \
  --max-tokens 8 \
  --tensor-parallel-size 2 \
  --out h0/out/h0_vllm_prefix_cache_qwen25_7b_tp2_1024_200
```

当前机器已跑过的 H0 输出：

- 输出目录：`h0/out/h0_vllm_prefix_cache_qwen25_7b_tp2_1024_200`
- 模型：`models/Qwen2.5-7B-Instruct`
- ShareGPT sessions：`200`
- 请求数：`894`
- `max_model_len == 1024`
- `max_tokens == 8`
- `tensor_parallel_size == 2`

说明：当前 vLLM `0.6.6.post1` OpenAI responses 未暴露逐请求真实 prefix-cache hit flag，输出中的 `hit` 是 trace-side session prefix reuse 推断。

### 8.3 H1 vLLM V1 KV Offload

当前仓库已在 `h0` 下实现 H1 自定义驱逐策略入口：`lru`、`lfu`、`lpe-score` 和 `vllm-default`。LRU、LFU、LPE 分别实现为 `h0/edgekv_v1_offload/cache_policy.py` 中的 `LRUCachePolicy`、`LFUCachePolicy`、`LPECachePolicy`，共同继承 `CachePolicy`；`LPECachePolicy` 按 §6.4 的 `score=p_reuse*c_recomp/size` 选择最低单位显存收益对象，并按 `theta_keep` 决定 `offload` 或 `drop`。先运行环境门禁：

```bash
conda run -n h3-lmcache-blog python h0/run_h1_vllm_offload.py \
  --mode env-check \
  --out h0/out/h1_env_check
```

当前已验证的 `vllm==0.8.5.post1` 已包含 `vllm.distributed.kv_transfer.kv_connector.v1`，`h0/out/h1_env_check/v1_env_status.json` 中 `ok=true`。真实接入命令：

```bash
conda run -n h3-lmcache-blog python h0/run_h1_vllm_offload.py \
  --config h0/configs/vllm_qwen25_7b_h1_offload.json \
  --mode real-v1 \
  --policy lpe-score
```

如果切回旧 vLLM 环境，可以用 `--mode shadow` 复用 H0 OpenAI replay 链路，生成 `policy_decisions.jsonl`；shadow 只代表策略旁路决策，不声明真实 KV offload 已生效。H1 输出目录为 `h0/out/h1_vllm_offload_qwen25_7b`，包含 `events.jsonl`、`policy_decisions.jsonl`、`summary.csv`、`config.resolved.json` 和 `gpu_memory_samples.jsonl`。

### 8.4 小模型或低资源 smoke

如果 7B 模型在目标服务器上 OOM，可先用 `models/facebook_opt_125m` 验证 vLLM server 和回放脚本链路，再切回 Qwen。

## 9. 常见问题

- `h3-lmcache-blog` 已升级到 `vllm 0.8.5.post1`，可以导入 `KVConnectorBase_V1`，H1 `env-check` 已通过。升级前环境锁在 `env_locks/h3-lmcache-blog.pre-vllm-upgrade.explicit.txt`，升级后环境锁在 `env_locks/h3-lmcache-blog.post-vllm-085.explicit.txt`。
- `Qwen2.5-7B-Instruct` fp16 在 11 GiB 单卡上不适合单卡运行，应使用 tensor parallel、更大显存 GPU 或更小模型。
- ShareGPT 数据文件不在项目目录内。迁移服务器时需要同步数据，或通过 `--trace-path` 指向新路径。
- 使用 Hugging Face 下载模型和数据前，可能需要先执行 `huggingface-cli login`。
- 当前根目录 `requirement.txt` 只记录早期轻量依赖：`matplotlib>=3.7`，不能代表三个 Conda 环境的完整依赖。复现请以本文第 4 节为准。
