# %%
import os
os.environ["HF_HOME"] = "/mnt/shared/dils/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/mnt/shared/dils/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/shared/dils/hf_cache"

import torch
from PIL import Image
from transformers import AutoProcessor, SmolVLMForConditionalGeneration

# Point directly at local download — no network, no cache writes
MODEL_PATH = "/mnt/shared/dils/models/SmolVLM2-2.2B-Instruct"
IMAGE_PATH = "inferences/images/001.jpg"

processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
processor.image_processor.size = {"longest_edge": 378}
processor.image_processor.max_image_size = {"longest_edge": 378}

model = SmolVLMForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    local_files_only=True,  # never touch the network or cache
)
model = model.eval()

image = Image.open(IMAGE_PATH).convert("RGB")
image = image.resize((378, 378), Image.LANCZOS)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image in detail."},
        ],
    }
]

text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

inputs = processor(
    text=text_prompt,
    images=image,
    return_tensors="pt",
).to("cuda")

print("pixel_values shape:", inputs["pixel_values"].shape)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )

generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
response = processor.decode(generated_ids[0], skip_special_tokens=True)
print(response)
# %%
