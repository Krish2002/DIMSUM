import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM , AutoTokenizer
import torch
df = pd.read_csv('/tmpdir/m24047krsh/llama_project/llama_inference/code/data/orca_dataset.csv')
df_structure = pd.read_csv('/tmpdir/m24047krsh/llama_project/llama_inference/code/data/orca_structure.csv')

tokenizer = AutoTokenizer.from_pretrained('/tmpdir/m24047krsh/Llama-3.1-70B-Instruct/')
model = AutoModelForCausalLM.from_pretrained(
    '/tmpdir/m24047krsh/Llama-3.1-70B-Instruct/',
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


system_prompt = '''
Task: Given a set of premises and an abstract that defines their properties, determine the answer to the question using only the information provided in the abstract. The abstract provides a structured relational framework, ensuring logical consistency in reasoning.

Instructions:

    Identify Topics and Premises:
        - Assign meaningful topic labels to the key elements of the problem (e.g., [topic-a]: Initial context, [topic-b]: Events influencing the outcome).
        - List the premises (P1, P2, ..., Pn), ensuring they contain all necessary descriptive statements.

    Apply Narrative Sequencing Rules Between Premises:
        - Maintain chronological order unless explicitly defined otherwise in the abstract.
        - If a premise introduces a temporal or logical shift, attach it accordingly using appropriate relations (e.g., Narr for sequential events, Bkg for background information).
        - Resolve premise reordering constraints based on given discourse markers.

    Use the Abstract’s Structured Information to Derive the Answer:
        - Follow the relationships and premises as structured in the abstract.
        - Do not introduce external knowledge; rely solely on the provided abstract and premises.
        - If numerical reasoning is required, apply logical derivations based on the premises.

Output Format:

    <answer>
    - Provide the final computed or inferred answer.
    - Ensure it aligns with the logical structure in the abstract.
    </answer>
'''

user_input_1 = "Question : Ian won $100 in the lottery.  He decided to use the money to pay off debts. He paid $20 to Colin. He then paid twice as much to Helen, as he had paid to Colin. Then finally, he paid half as much to Benedict, as he had paid to Helen.  How much money, in dollars, does he have left after paying off debts?"


abstract_1 = '''
<abstract>
Topics
[topic-a]: Ian winning the lottery
[topic-b]: Ian paying off debts

Relationships
t1--elab--t2

Premises
P1: Ian won $100 in the lottery.
P2: He decided to use the money to pay off debts.
P3: He paid $20 to Colin.
P4: He then paid twice as much to Helen, as he had paid to Colin.
P5: Then finally, he paid half as much to Benedict, as he had paid to Helen.

Narrative Structure
P1--narr--P2
P2--narr--P3
P3--narr--P4
P4--narr--P5
</abstract>
'''
answer_with_abstract_1 = '''
<answer>

To determine how much money Ian has left after paying off debts, we will follow the premises in the given order.

1. Ian won $100 in the lottery (P1).
2. He decided to use the money to pay off debts (P2).
3. He paid $20 to Colin (P3). Remaining money: $100 - $20 = $80.
4. He then paid twice as much to Helen, as he had paid to Colin (P4). He paid 2 * $20 = $40 to Helen. Remaining money: $80 - $40 = $40.
5. Then finally, he paid half as much to Benedict, as he had paid to Helen (P5). He paid $40 / 2 = $20 to Benedict. Remaining money: $40 - $20 = $20.

Therefore, Ian has $20 left after paying off debts.

</answer> ''' 


user_input_2 = "A bear is preparing to hibernate for the winter and needs to gain 1000 pounds. At the end of summer, the bear feasts on berries and small woodland animals. During autumn, it devours acorns and salmon. It gained a fifth of the weight it needed from berries during summer, and during autumn, it gained twice that amount from acorns. Salmon made up half of the remaining weight it had needed to gain. How many pounds did it gain eating small animals?"


abstract_2 = '''<abstract>
Topics
[topic-a]: Bear preparing for hibernation
[topic-b]: Bear’s weight gains from different food sources

Relationships
t1--elab--t2

Premises
P1: A bear is preparing to hibernate for the winter and needs to gain 1000 pounds.
P2: At the end of summer, the bear feasts on berries and small woodland animals.
P3: During autumn, it devours acorns and salmon.
P4: It gained a fifth of the weight it needed from berries during summer.
P5: During autumn, it gained twice that amount from acorns.
P6: Salmon made up half of the remaining weight it had needed to gain.

Narrative Structure
P1--narr--P2
P2--narr--P3
P3--bkg--P4
P4--narr--P5
P5--narr--P6
</abstract>
'''


answer_with_abstract_2 = '''<answer>
To find out how many pounds the bear gained eating small animals, we need to follow the relationships and premises as structured in the abstract.

From P1, we know the bear needs to gain 1000 pounds.

From P4, we know it gained a fifth of the weight it needed from berries during summer. 
1/5 * 1000 = 200 pounds from berries.

From P5, we know during autumn, it gained twice that amount from acorns. 
2 * 200 = 400 pounds from acorns.

The total weight gained from berries and acorns is 200 + 400 = 600 pounds.

From P6, we know salmon made up half of the remaining weight it had needed to gain. 
The remaining weight is 1000 - 600 = 400 pounds.
Half of the remaining weight is 1/2 * 400 = 200 pounds from salmon.

Now, we know the bear gained weight from berries (200 pounds), acorns (400 pounds), and salmon (200 pounds), totaling 800 pounds.

Since the bear also feasts on small woodland animals during summer (P2), the remaining weight must come from these animals. 
The total weight gained is 1000 pounds, and we've accounted for 800 pounds from berries, acorns, and salmon.

Therefore, the bear gained 1000 - 800 = 200 pounds eating small animals.

</answer>
'''

def role_answer(premise, abstract):
    user_input = f"The given question is: {premise}, structured according to the discourse format: {abstract}."
    f_1 = f"The given question is: {user_input_1}, structured according to the discourse format: {abstract_1}."
    f_2 = f"The given question is: {user_input_2}, structured according to the discourse format: {abstract_2}."

    chat = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": f_1,
        },
        {
            "role": "assistant",
            "content": answer_with_abstract_1,
        },
        {
            "role": "user",
            "content": f_2,
        },
        {
            "role": "assistant",
            "content": answer_with_abstract_2,
        },
        {
            "role": "user",
            "content": user_input,
        },
    ]

    return chat


def generate_structure(question , structure):
  messages = role_answer(question , structure)
  input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
  ).to(model.device)
  terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]
  tokenizer.pad_token_id = 18610
  tokenizer.padding_side = "right"
  outputs = model.generate(
    input_ids,
    max_new_tokens=400,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.4,
    top_p=0.9,
  )


  response = outputs[0][input_ids.shape[-1]:]
  structure = tokenizer.decode(response, skip_special_tokens=True)

  return structure

from tqdm import tqdm
answers = []
for i in tqdm(range(len(df["question"]))):
  question = df["question"][i]
  answer = generate_structure(question , df_structure["abstracts"][i])
  answers.append(answer)

df_structure = pd.DataFrame({'pred_answer': answers })
df_save = pd.concat([df, df_structure], axis=1)

df_structure.to_csv('/tmpdir/m24047krsh/llama_project/llama_inference/code/data/orca_answer.csv', index=False)