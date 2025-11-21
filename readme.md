# 中文情感分析模型环境配置与使用指南

本项目展示如何在本地配置环境、加载预训练 BERT 中文模型，并使用 HuggingFace 上的 ChnSentiCorp 数据集进行中文情感分析任务。

## 🚀 1. 环境配置

### 1.1 创建 Conda 环境

```bash
conda create -n hgf python=3.12 -y
conda activate hgf
```

### 1.2 安装 Transformers、Datasets、torch

若需要 GPU 加速，请提前安装与你 CUDA 版本匹配的 PyTorch。

🔹 安装 Transformers（含 SentencePiece）

```bash
pip install transformers[sentencepiece]
```

🔹 安装 Datasets

```bash
pip install datasets
```

🔹安装 PyTorch（GPU 版本）

请根据你的 CUDA 版本选择安装命令（示例：CUDA 12）：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 🤖 2. 加载预训练 BERT 中文模型与分词器（tokenizer）

以 bert-base-chinese 为例：（可用下列脚本在python上安装，也可去Hugging Face官网找到对应模型下载）

 ```python
import transformers
from transformers import AutoModel,AutoTokenizer
model_name="bert-base-chinese"
model_dir=r"D:\cccc\usense_work\hgf_test\model"  ### 模型保存路径
model=AutoModel.from_pretrained(model_name,cache_dir=model_dir)
tokenizer=AutoTokenizer.from_pretrained(model_name,cache_dir=model_dir)
```

## 📦 3. 加载中文情感分析数据集（ChnSentiCorp）

HuggingFace 数据集仓库：https://huggingface.co/datasets/lansinuote/ChnSentiCorp

```python
from datasets import load_dataset
data_dir=r"D:\cccc\usense_work\hgf_test\data"
dataset = load_dataset("lansinuote/ChnSentiCorp",cache_dir=data_dir,num_proc=1)
dataset.save_to_disk(r"D:\cccc\usense_work\hgf_test\data\lansinuote___chn_senti_corp\data_csc")###数据集保存路径
print(dataset["train"][0])
```


