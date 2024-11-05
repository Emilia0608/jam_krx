# +
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

import json
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset, concatenate_datasets
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported


max_seq_length = 2048
dtype = None
load_in_4bit = True 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="model_v4",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    device_map={'':torch.cuda.current_device()}, 
    # token="hf_...", # gated model을 사용할 경우 허깅페이스 토큰을 입력해주시길 바랍니다.
)

print(model)


# LoRA Adapter 선언
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=False,
    loftq_config=None
)


EOS_TOKEN = tokenizer.eos_token
# prompt_format = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# {}

# ### 정답:
# {}"""

prompt_format = """
{}

### 정답:
{}"""

with open('data/new_krx_report.json', 'r', encoding='utf-8') as file:
    report = json.load(file)
with open('data/new_stock.json', 'r', encoding='utf-8') as file:
    stock = json.load(file)
with open('data/new_code_company_v2_1103.json', 'r', encoding='utf-8') as file:
    samples = json.load(file)
with open('data/data_v11_1104.json', 'r', encoding='utf-8') as file:
    v11_json = json.load(file)
with open('data/data_qa_v11_1104.json', 'r', encoding='utf-8') as file:
    v11_qa_json = json.load(file)
    
new_data=[]
for item in report:
    tmp={}
    company_report=item["company_report"]+EOS_TOKEN
    tmp["formatted_text"]=company_report
    new_data.append(tmp)
    
for item in samples:
    tmp={}
    instruction=item["prompt"]
    output=item["response"]
    text = prompt_format.format(instruction, output) + EOS_TOKEN
    tmp["formatted_text"]=text
    new_data.append(tmp)
    
for item in stock:
    tmp={}
    instruction=item["prompt"]
    output=item["response"]
    text = prompt_format.format(instruction, output) + EOS_TOKEN
    tmp["formatted_text"]=text
    new_data.append(tmp)

for item in v11_json:
    tmp={}
    tmp["formatted_text"]=item["formatted_text"] + EOS_TOKEN
    new_data.append(tmp)
    
for item in v11_qa_json:
    tmp={}
    instruction=item["prompt"]
    output=item["response"]
    text = prompt_format.format(instruction, output) + EOS_TOKEN
    tmp["formatted_text"]=text
    new_data.append(tmp)
    
    
dataset = Dataset.from_dict({"formatted_text": [item["formatted_text"] for item in new_data]})

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="formatted_text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # True로 설정하면 짧은 텍스트 데이터에 대해서는 더 빠른 학습 속도로를 보여줍니다.
    args=TrainingArguments(  # TrainingArguments는 자신의 학습 환경과 기호에 따라 적절하게 설정하면 됩니다.
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        warmup_steps=5,
        num_train_epochs=1,  # full train을 위해 에포크 수 설정
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="outputs_v11",
    ),
)

trainer_stats = trainer.train()
model.save_pretrained_merged("model_v11", tokenizer, save_method="merged_16bit")

