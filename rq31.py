import json
from pathlib import Path
from datasets import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from thesis_datasets import germeval, omp, schmidt

# Dict path
dict_path = "dicts/"

ds_name,_,ds = germeval("train")
# ds_name,_,ds = omp("full[50%:]")
# ds_name,_,ds = schmidt("train")

model_name = "google/gemma-1.1-7b-it"
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_name = "mistralai/Mistral-7B-Instruct-v0.2"

prompt = "You are provided with a text, and your task is to analyze the sentiment. Anlyze also each sentiment-bearing word and emotion in the context of the text. For each sentiment-bearing word, assign a sentiment value between -1 (most negative) and 1 (most positive). Neutral words or those not bearing sentiment should not be assigned any value. Format: word - value - reasoning"

wordlist = []

text_p = "Fix mit der @DB_Bahn nach Mannheim und auf zu Volkers Pispers. Freue mich riesig :)) #koeln #mannheim #pispers # ICE http://t.co/qGSQ2Ituc4"
text_n = "bz-berlin.de Alexanderplatz: Mann in S-Bahn brutal attackiert Gleich vier Personen griffen zwischen Ostbahnhof und Alexanderplatz einen Fahrgast in der S-Bahn an."
 # Some examples
template = [
    {"role": "user", "content": f"{prompt} Text: {text_n}",
     "role": "assistant", "content": """* brutal -0.9 - This word describes the attack as extremely violent and harsh, carrying a strong negative sentiment. \n * attackiert -0.8 - The verb "attackiert" means "attacked", which is inherently negative, indicating aggression or violence. \n * griffen -0.6 - "Griffen" means "grabbed" or "attacked", indicating a confrontation, which is a negative action. \n * angegriffen - -0.7 - Another form relating to "attack", conveying a negative sentiment due to the aggressive nature. (Note: Although "angegriffen" is not directly in the provided text, the term "griffen an" can be derived to it for clarity. If stick strictly to the text, no need to analyze "angegriffen".)"""},
    {"role": "user", "content": f"{prompt} Text: Wise Guys Deutsche Bahn Live bei 3Sat Dies ist eines der beliebtesten und bekanntesten Lieder der Wise Guys Viel Spaß beim hören :D.",
     "role": "assistant", "content": "* beliebtesten 0.2172 - Indicates positive reception and high approval, popular among people. \n * bekanntesten 0.004 - Recognized and widely known, positive for visibility and reputation.\n * Spaß 0.2823 - High level of enjoyment and amusement.\n"},
]

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.config.pad_token_id = model.config.eos_token_id

# Generate dictionary
for document in tqdm(ds):
    inContext = list(template)
    document_prompt = {"role": "user", "content": f"{prompt} Text: {document['text']}"}
    inContext.append(document_prompt)
    chat_prompt = tokenizer.apply_chat_template(inContext, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(chat_prompt, add_special_tokens=False, return_tensors="pt")
    outputs = model.generate(input_ids=inputs.to("cuda"), max_new_tokens=150)
    answer = tokenizer.decode(outputs[0]).rsplit("<|end_header_id|>")[-1]

    split_a = answer.replace(",","").replace("*","").replace('"',"").split(" ")
    split_t = document["text"].replace(",","").replace(".","").replace("*","").split(" ")
    for idx, element in enumerate(split_a):
        clean_element = element.replace(":", "").replace('"', "")
        if clean_element in split_t: # Some postprocessing to cut the strings into words with values.
            substring = "\t".join(split_a[idx:idx+4])
            if ("0" in substring) or ("1" in substring):
                for number in split_a[idx:idx+4]:
                    clean_number = number.replace('.','',1).lstrip("-+").isdigit()
                    if clean_number:
                        if (float(number) >= -1 ) and (float(number) <= 1):
                            wordlist.append([split_a[idx], float(number)])


# Save wordlist
Path(dict_path).mkdir(parents=True, exist_ok=True)
filename = dict_path + f"rq31_{ds_name}_{model_name}".replace("/","_")

with open(f"{filename}.txt", "w") as f:
    json.dump(wordlist, f, ensure_ascii=False)
