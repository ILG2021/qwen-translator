# Qwen3.5-4B 翻译模型微调项目

本项目使用 [Unsloth](https://github.com/unslothai/unsloth) 框架，针对 Qwen3.5-4B 模型进行微调，以实现高质量、纯净输出的翻译功能。最终模型将导出为 4-bit GGUF 格式，部署于 Ollama。

## 目录结构
- `train.py`: 主要微调脚本。
- `Modelfile`: Ollama 部署配置文件。
- `translate_data.json`: 您的训练数据文件（需自行准备）。

## 快速开始

### 1. 环境准备
确保您的系统中已安装 Python 3.10+ 和 CUDA。

```bash
pip install unsloth[colab-new] xformers trl peft accelerate bitsandbytes
```

### 2. 准备数据
将您的翻译数据保存为 `translate_data.json`（支持多行 JSON/JSONL），格式如下：
```json
{"messages": [{"role": "user", "content": "将以下文本翻译为中文..."}, {"role": "assistant", "content": "你好"}]}
```
您无需进行额外转换，训练脚本会自动识别此格式。

### 3. 开始微调
执行训练脚本。您可以指定数据文件并设置训练轮数（epochs）。显存占用约 10GB 左右（使用 4-bit 加载）。
```bash
# 正常开始训练
python train.py --data_file translate_data.json --epochs 3

# 如果训练中断，可以使用 --resume 恢复
python train.py --resume
```
训练完成后，会在项目目录下生成 `model_q4_k_m` 文件夹，其中包含导出的 GGUF 文件。

### 4. 部署到 Ollama

## 注意事项
- **显存要求**: 推荐使用 12GB 或更高显存的 GPU（如 RTX 3060 12G, 4070 等）。
- **训练量**: `train.py` 中的 `max_steps` 默认为 60，请根据您的数据量进行调整。
- **自定义模板**: Qwen3.5 默认使用 ChatML 模板，Unsloth 会自动处理。
