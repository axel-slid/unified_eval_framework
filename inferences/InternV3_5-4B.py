# %%

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

MODEL_PATH = "models/InternVL3_5-4B"
IMAGE_PATH = "tests/images/001.jpg.jpg"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    use_fast=False,
)

model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = model.eval().to(device)

image = Image.open(IMAGE_PATH).convert("RGB")

question = "<image>\nDescribe this image in detail."

response = model.chat(
    tokenizer=tokenizer,
    pixel_values=None,
    question=question,
    generation_config={
        "max_new_tokens": 256,
        "do_sample": False,
    },
    image=image,
)

print("\n=== Response ===\n")
print(response)