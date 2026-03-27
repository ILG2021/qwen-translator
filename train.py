import argparse
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from unsloth import is_bfloat16_supported

# 1. 参数解析
parser = argparse.ArgumentParser(description="Unsloth Qwen 3.5 翻译微调脚本")
parser.add_argument("--data_file", type=str, default="translate_data.json", help="训练数据路径 (JSON 格式)")
parser.add_argument("--epochs", type=int, default=3, help="训练轮数 (建议 1-3 轮)")
parser.add_argument("--resume", action="store_true", help="从最新的检查点恢复训练")
args = parser.parse_args()

# 2. 配置
model_name = "unsloth/Qwen3.5-4B-Instruct"
max_seq_length = 8192 # 增加到 8192 以支持长文本翻译
dtype = None # None 会根据 GPU 自动选择
load_in_4bit = True # 开启 4-bit 量化以节省显存

# 3. 加载模型和分词器
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 4. 添加 LoRA 适配器
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 5. 数据处理与 Chat Template
def formatting_prompts_func(examples):
    instructions = examples["messages"]
    texts = []
    for messages in instructions:
        # 使用 Qwen2.5/3.5 标准的 ChatML 模板
        text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = False)
        texts.append(text)
    return { "text" : texts, }

# 加载数据
dataset = load_dataset("json", data_files={"train": args.data_file}, split="train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

# 6. 设置训练参数
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        # warmup_ratio 相比 warmup_steps 更适合按轮数 (epochs) 训练
        warmup_ratio = 0.1, 
        num_train_epochs = args.epochs, # 使用轮数代替步数
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        # 增加保存机制，以便恢复
        save_strategy = "steps",
        save_steps = 500,
        save_total_limit = 5,
    ),
)

# 6. 开始训练
# 如果指定了 --resume，则从最新的检查点恢复
trainer_stats = trainer.train(resume_from_checkpoint = args.resume)

# 7. 保存 LoRA 适配器
model.save_pretrained("lora_model") # 保存适配器
tokenizer.save_pretrained("lora_model")

# 8. 直接导出为 GGUF (用于 Ollama)
# 这里选择导出为 q4_k_m 量化格式，这是 Ollama 常用的高质量 4-bit 分支
model.save_pretrained_gguf("model_q4_k_m", tokenizer, quantization_method = "q4_k_m")

print("\n" + "="*50)
print("训练完成！")
print("LoRA 适配器已保存至: lora_model")
print("GGUF 模型已保存至: model_q4_k_m")
print("="*50)
