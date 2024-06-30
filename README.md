# Lawbuddy + Illegal squad method

## Model
[Here](https://huggingface.co/Nongpeamm/Lawbuddy-typhoon-7b) Lawbuddy-typhoon-7b


## Usage

```py
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, TrainingArguments
from datasets import load_dataset

prompt_template = """
Instruction: {instruction}
Context: {context}
Response:
"""

prompt = prompt_template.format(
    instruction="your instruction",
    context="your context"
)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/content/drive/MyDrive/typhoon_DPO/checkpoint-195",
    max_seq_length=4096,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Tokenize input
inputs = tokenizer(
    [prompt] * 1,
    return_tensors="pt"
).to("cuda")

# Generate response
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=1028)
```
