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


system_prompt = '''
Task: Analyze a short story or scenario and represent its elements using a structured relational framework. Follow the rules for narrative progression, temporal adverbials, and premise reordering to derive the final structure. Only include topics, relationships, premises, and narrative structure in the output. Do not include any resolution or calculations.

Instructions:

    Identify Topics and Premises:
        Assign meaningful topic labels to the key elements of the story (e.g., [topic-a]: Initial context/situation, [topic-b]: Character's key decision/action).
        Break the story into premises (P1, P2, ..., Pn) representing descriptive statements or key actions/events.

    Apply Narrative Sequencing Rules Between Premises:
        Same Tense, No Special Markers:
            Attach the premise (Pi) to the previous premise (Pi-1) using Narr.
        Shift in Tense:
            If a premise (Pi) introduces a tense shift, attach it to the relevant earlier premise (Pi-1) using Background (bkg) to indicate reverse order.
        Days of the Week:
            If days are mentioned and are in proper chronological order, attach Pi to Pi-1 using Narr.
            If Pi: "Day n B" and Pi-1: "Day n+m C" (where m > 1):
                Shuffle the premises until Pi connects with Pj where Pj: "Day n+1 C" using Narr.
        Discourse Adverbials:
            "Then," "Next," "XY later": Attach Pi to Pi-1 using Narr.
            "XY before": Attach Pi to Pi-1 using Narr to indicate reverse temporal progression.
            "While": Use Elaboration (elab) to connect Pi to Pi-1.
        Special Markers:
            If Pi contains "Finally," and Pi is not the last premise, shuffle to make Pi the final premise.
            If Pi contains "First," and Pi is not the first premise, shuffle to make Pi the initial premise.
        Day-Specific Rules:
            If Pi: "Day n B" and Pi+1: "Day n+1 B," attach Pi to Pi+1 using Narr.
            If Pi: "Day n B" and Pi-1: "Day n+m C," shuffle until Pi connects with Pj where Pj: "Day n+1 C" using Narr.

    Label Relationships Between Elements:
        Use t1--relation--t2 for topic-level relationships (e.g., elaboration, cause-effect, etc.).
        Use PX--relation--PY for premise-level relationships.

    Output Requirements:
        Only include the following in the output:
            Topics: List all identified topics with brief descriptions.
            Relationships: Describe relationships between topics and premises.
            Premises: List all premises (P1, P2, ..., Pn) in logical order.
            Narrative Structure: Show how the premises are connected using narrative relationships.
        Do not include any resolution, calculations, or final answers.

'''

user_input_1 = '''

Story: Ken created a care package to send to his brother, who was away at boarding school. Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds. And finally, he added enough gummy worms to double the weight once again. Then, he added enough brownies to cause the weight to triple. Next, he added another 2 pounds of jelly beans. What was the final weight of the box of goodies, in pounds?

'''

sys_out_1 = '''

<abstract>
Topics
[topic-a]: Ken’s care package context
[topic-b]: The incremental additions to the box

Relationships
t1--elab--t2

Premises
P1: Ken created a care package to send to his brother, who was away at boarding school.
P2: Ken placed a box on a scale and poured in enough jelly beans to bring the weight to 2 pounds.
P3: Then, he added enough brownies to cause the weight to triple.
P4: Next, he added another 2 pounds of jelly beans.
P5: Finally, he added enough gummy worms to double the weight once again.

Narrative Structure
P1--narr--P2
P2--narr--P3
P3--narr--P4
P4--narr--P5
</abstract>
'''
user_input_2 = '''
Story: A bear is preparing to hibernate for the winter. By the time it went into hibernation, it weighed 900 pounds. In November, it gained an additional 50 pounds. In October, it ate enough to gain another 100 pounds. It gained 200 pounds during the summer months. How much did the bear weigh at the start of summer?
'''

sys_out_2 = '''
<abstract>
Topics
[topic-a]: Bear preparing for hibernation
[topic-b]: Bear’s weight gains over time

Relationships
t1--elab--t2

Premises
P1: A bear is preparing to hibernate for the winter.
P2: It gained 200 pounds during the summer months.
P3: In October, it gained another 100 pounds.
P4: In November, it gained an additional 50 pounds.
P5: By the time it went into hibernation, it weighed 900 pounds.

Narrative Structure
P1--narr--P2
P2--narr--P3
P3--narr--P4
P4--narr--P5
</abstract>
'''
def role(question):
    chat = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_input_1,
        },
        {
            "role": "assistant",
            "content": sys_out_1,
        },
        {
            "role": "user",
            "content": user_input_2,
        },
        {
            "role": "assistant",
            "content": sys_out_2,
        },
        {
           "role": "user",
            "content": question
        }
    ]
    return chat



def generate_structure(question):
  messages = role(question)
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

def extract_premises(question):
  premises = question.split('.')[:-1]
  premises = ' '.join(premises)
  return premises

from tqdm import tqdm
abstracts = []
for i in tqdm(range(len(df["question"]))):
  question = df["question"][i]
  premises = extract_premises(question)
  abstract = generate_structure(premises)
  abstracts.append(abstract)

df_structure = pd.DataFrame({'abstracts': abstracts})
df_save = pd.concat([df, df_structure], axis=1)

df_structure.to_csv('data\gsm_dataset_answers.csv', index=False)