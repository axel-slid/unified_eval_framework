# %%
import os
os.environ["HF_HOME"] = "/mnt/shared/dils/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/mnt/shared/dils/hf_cache"

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH = "/mnt/shared/dils/models/Qwen3-VL-4B-Instruct"
IMAGE_PATH = "inferences/images/001.jpg"

processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    local_files_only=True,
)
model = model.eval()

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": IMAGE_PATH},
            {"type": "text",  "text": "Describe this image in detail."},
        ],
    }
]

text_prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text_prompt],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda")

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )

generated_ids_trimmed = [
    out_ids[len(in_ids):]
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

response = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]

print(response)
# %%
