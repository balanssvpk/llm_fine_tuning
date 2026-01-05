# %% [markdown]
# # LLM Fine‑Tuning

# %% [markdown]
# ### Set Devide and Model name

# %%
device="cuda"
# device="cpu"
model_name="Qwen/Qwen2.5-3B-Instruct"

# %% [markdown]
# ### 1. Create pipeline from tranformers

# %%
from transformers import pipeline

ask_llm = pipeline(
    model=model_name,
    device=device
)


# %% [markdown]
# ### Get response from base model

# %%
print(ask_llm("Who is Mariya Sha ?")[0]["generated_text"])

# %%


# %% [markdown]
# ### 2. Create dataset for fine tuning

# %%
from datasets import load_dataset

# %%
raw_data = load_dataset("json", data_files='mariya.json')
raw_data

# %%
raw_data['train'][0]

# %%


# %% [markdown]
# ### 3. Create tokens for dataset

# %% [markdown]
# #### Load the tokenizer

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
)

# %% [markdown]
# #### Inspect the template

# %%
print(tokenizer.chat_template)


# %%
def preprocess(sample):
    sample = sample['prompt'] + ' \n ' + sample['completion']
    
    tokenized = tokenizer(
        sample,
        max_length=128,
        truncation=True,
        padding='max_length'    
    )
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized

# %%
data = raw_data.map(preprocess)

# %%
display(
    print(data['train'][8])
    )

# %%


# %% [markdown]
# ### 4. create PEFT and LORA config

# %%
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map = device,
    dtype = torch.float16
)

lora_config = LoraConfig(
    task_type = TaskType.CAUSAL_LM,
    target_modules = ["q_proj", "k_proj", "v_proj"]
)

model = get_peft_model(model, lora_config)

# %%
model

# %%


# %% [markdown]
# ### 5. create training arguments and trainer

# %%
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    num_train_epochs=15,
    learning_rate=0.001,
    logging_steps=25,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data["train"]
)


# %% [markdown]
# ### 6. train model

# %%
%%time
trainer.train()

# %%


# %% [markdown]
# ### 7. save saves LoRA adapter weights

# %%
trainer.save_model("./my_qwen")
tokenizer.save_pretrained("./my_qwen")

# %% [markdown]
# ### 8. save finetuned model(Load base + merge LoRA) and tokennizer

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = model_name   

# Load tokenizer from the base model
tokenizer = AutoTokenizer.from_pretrained(base)

# Load base model
model = AutoModelForCausalLM.from_pretrained(base, dtype="auto")

# Load LoRA / PEFT adapter
model = PeftModel.from_pretrained(model, "./my_qwen")

# Merge LoRA weights into the base model
model = model.merge_and_unload()   # IMPORTANT

# Save merged model + tokenizer
model.save_pretrained("./merged_qwen")
tokenizer.save_pretrained("./merged_qwen")

# %%


# %% [markdown]
# ### 9. Get response from finetuned model

# %%
from transformers import pipeline

ask_llm = pipeline(
    model="./my_qwen",
    tokenizer="./my_qwen",
    device=device
)
print(ask_llm("Who is Mariya Sha")[0]["generated_text"])

# %%


# %% [markdown]
# ### 10. Convert finetuned model as GGUF

# %% [markdown]
# A. Make Sure merged model contains both model and tokenizer
# 
# B. Clone llama.cpp
#     git clone https://github.com/ggerganov/llama.cpp
#     cd llama.cpp
# 
# C. Install Python dependencies
# 
# D. Convert HF → GGUF (FP16)
# 
#     1. cd C:\Users\balan\OneDrive\Desktop\llama.cpp
# 
#     2. conda activate llm (activate llm(environment))
# 
#     3. (llm) C:\Users\balan\OneDrive\Desktop\llama.cpp>
#     python convert_hf_to_gguf.py --outtype f16 "C:\Users\balan\OneDrive\Desktop\LLM_Fine_Tuning\merged_qwen"
# 
# E. "Merged_Qwen-3.1B-F16.gguf" file will be created in folder "C:\Users\balan\OneDrive\Desktop\LLM_Fine_Tuning\merged_qwen"
# 
# 

# %%


# %% [markdown]
# ### 11. Create ollama custom model from finetuned model using modelfile

# %% [markdown]
# A. create model file using GGUF,
# 
#     # Base model file
#     FROM ./Merged_Qwen-3.1B-F16.gguf
# 
#     # Optional: model parameters
#     PARAMETER temperature 0.0
#     PARAMETER top_p 0.9
#     PARAMETER repeat_penalty 1.1
#     PARAMETER num_ctx 4096
# 
#     # Set the system behavior
#     SYSTEM """
#     You are Qwen‑3.1B, a fine‑tuned assistant created by Bala.
#     You respond clearly, concisely, and with helpful reasoning.
#     Avoid hallucinations and always ask for clarification when needed.
#     """
# 
#     # Optional: custom prompt formatting
#     TEMPLATE """
#     {{ if .System }}<|system|>
#     {{ .System }}{{ end }}
# 
#     <|user|>
#     {{ .Prompt }}
# 
#     <|assistant|>
#     """
# 
# 
# B. ollama create "qwen-3.1b-fine_tuned" -f C:\Users\balan\OneDrive\Desktop\LLM_Fine_Tuning\merged_qwen\Modelfile

# %%



