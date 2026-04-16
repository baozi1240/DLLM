import argparse
import time
import torch
from transformers import AutoModel, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_length", type=int, default=None)
    return parser.parse_args()

def select_device():
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"

args = parse_args()

# --- Model Loading ---
model_path = "./models/Dream-v0-Instruct-7B"
device = select_device()
dtype_by_device = {
    "cuda": torch.bfloat16,
    "mps": torch.float16,
    "cpu": torch.float32,
}
dtype = dtype_by_device[device]
print(f"Using device: {device} (dtype={dtype})")

model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = model.to(device).eval()

messages = [
     {"role": "user", "content": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"},
 ]
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
)
input_ids = inputs.input_ids.to(device)
attention_mask = inputs.attention_mask.to(device)

output = model.diffusion_generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=512,
    output_history=True,
    return_dict_in_generate=True,
    steps=512,
    temperature=0.0,
    top_p=0.95,
    block_length=args.block_length,
    alg="entropy",
    alg_temp=0.,
)
generations = [
    tokenizer.decode(g[len(p) :].tolist())
    for p, g in zip(input_ids, output.sequences)
]

print(generations[0].split(tokenizer.eos_token)[0])
