from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel
from generate import generate

device = "cuda" if torch.cuda.is_available() else "cpu"
# 与脚本同目录下的 LLaDA-8B-Base（可从 Hugging Face GSAI-ML/LLaDA-8B-Base 下载完整权重）
model_path = str(Path(__file__).resolve().parent / "LLaDA-8B-Base")

# 1. 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True,
)

# 2. 加载 model
model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
).to(device).eval()

# 3. 一些必要检查
if tokenizer.padding_side != "left":
    tokenizer.padding_side = "left"

assert tokenizer.pad_token_id != 126336, "pad_token_id 不能等于 mask token id 126336"

# 4. Base 模型直接喂普通文本，不要用 chat template
prompts = [
    "Question: What is the capital of France?\nAnswer:",
    "Question: 1+1=?\nAnswer:",
]

# 5. tokenize
inputs = tokenizer(
    prompts,
    add_special_tokens=False,
    padding=True,
    return_tensors="pt"
)

input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# 6. 推理
with torch.no_grad():
    outputs = generate(
        model=model,
        prompt=input_ids,
        attention_mask=attention_mask,
        steps=128,
        gen_length=128,
        block_length=32,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
    )

# 7. 只解码新生成部分
gen_tokens = outputs[:, input_ids.shape[1]:]
texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

for i, text in enumerate(texts):
    print(f"\n===== sample {i} =====")
    print(text)
