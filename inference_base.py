import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM , AutoTokenizer
import torch
df = pd.read_csv('data\gsm_dataset.csv')

tokenizer = AutoTokenizer.from_pretrained('path')
model = AutoModelForCausalLM.from_pretrained(
    'path',
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
def role(question):
  messages = [
    {"role": "system", "content": "You are a logical reasoning assistant. When given multiple premises, your task is to draw logical connections between the premises and solve it step by step and conclude with answer."},
    {"role": "user", "content": f"Here are the premises for the logical reasoning question: {question}"},
  ]
  return messages

# def role(question):
#   messages = [
#     {"role": "system", "content": "You are a logical reasoning assistant. When given multiple premises, your task is to create a clear and one paragraph abstract that conveys all relevant details of the problem statement in a logical and coherent order."},
#     {"role": "user", "content": f"Here are the premises for the logical reasoning question: {question}"},
#   ]
#   return messages

def generate_answer(question):
  messages = role( question)
  input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
  ).to(model.device)
  terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]
  outputs = model.generate(
    input_ids,
    max_new_tokens=200,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
  )

  response = outputs[0][input_ids.shape[-1]:]
  answer = tokenizer.decode(response, skip_special_tokens=True)

  return answer

from tqdm import tqdm
answers = []
for i in tqdm(range(len(df["question"]))):
  question = df["question"][i]
  abstract = generate_answer(question)
  answers.append(abstract)


df_answers = pd.DataFrame(answers, columns=["answer_without_abstract"])
df_save = pd.concat([df, df_answers], axis=1)
df_save.to_csv('data\gsm_base_permuted.csv', index=False)