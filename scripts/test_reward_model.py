import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

reward_model = PeftModel.from_pretrained(model, "./models/reward_model_gpt2")
reward_model = reward_model.merge_and_unload()
reward_model.eval()

reward_model.to("cuda")

def get_score(model, tokenizer, prompt, response):
    kwargs = {"padding": "max_length", "truncation": True, "max_length": 512, "return_tensors": "pt"}
    inputs = tokenizer.encode_plus(prompt+"\n"+response, **kwargs).to("cuda")

    with torch.no_grad():
        logits = model(**inputs).logits

    # predicted_class = logits.argmax(dim=-1).item()

    return logits

def test(x):
    a = get_score(reward_model, tokenizer, x['instruction'], x['choice_w'])
    b = get_score(reward_model, tokenizer, x['instruction'], x['choice_l'])

    if a.sum() > b.sum():
        return "Works"
    elif a.sum() < b.sum():
        return "Fails"

df = pd.read_csv("./data/llama3-loan-mortgage-ranked-responses.csv")

for i in df.iloc[:10].iterrows():
    print(i[1]['instruction'])
    print(i[1]['choice_w'])
    print(i[1]['choice_l'])
    print(test(i[1]))
    print()
    print("==="*20)