import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

model = AutoModelForCausalLM.from_pretrained("./models/reward_model_gpt2")
tokenizer = AutoTokenizer.from_pretrained("./models/reward_model_gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

model.eval()
model.to("cuda")

def get_score(model, tokenizer, prompt, response):
    kwargs = {"padding": "max_length", "truncation": True, "max_length": 512, "return_tensors": "pt"}
    inputs = tokenizer.encode_plus(prompt+"\n"+response, **kwargs).to("cuda")

    with torch.no_grad():
        logits = model(**inputs).logits

    # predicted_class = logits.argmax(dim=-1).item()

    return logits

def test(x):
    a = get_score(model, tokenizer, x['instruction'], x['choice_w'])
    b = get_score(model, tokenizer, x['instruction'], x['choice_l'])

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