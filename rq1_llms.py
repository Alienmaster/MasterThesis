import json
from pathlib import Path
import sys
from datasets import tqdm
from evaluate import load
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from termcolor import colored
from transformers import AutoTokenizer, AutoModelForCausalLM
from thesis_datasets import germeval, omp, schmidt, wikipedia

# Parameters
model_info = sys.argv[1] # "google/gemma-1.1-7b-it <start_of_turn>model"
model_name = sys.argv[1].split(" ")[0] # google/gemma-1.1-7b-it
splitter = sys.argv[1].split(" ")[1] # <start_of_turn>model

ds_name = sys.argv[2].split(" ")[0] # germeval
split_comm = sys.argv[2].split(" ")[1] # test_syn

# Folder for results
rq1_results_path = "results_rq1/"

# Load model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

label_ds = "sentiment"
label_model = "label"
text_ds = "text"

# Load dataset
if ds_name == "germeval":
    dataset, split, dataset_loaded = germeval(split_comm)
if ds_name == "omp":
    dataset, split, dataset_loaded = omp()
if ds_name == "schmidt":
    dataset, split, dataset_loaded = schmidt(split_comm)
if ds_name == "wikipedia":
    dataset, split, dataset_loaded = wikipedia()
results_comp = []

prompt = f"Classify the sentiment of the text into ONE of the three classes: neutral, negative or positive. Split the answer in two parts: Label and Reasoning. Text: "

# Prediction
for snippet in tqdm(dataset_loaded):
    text = snippet["text"]
    chat_template = [{  "role" : "user",
                        "content" : f"{prompt} {text}"}]
    datapoint = {"text" : text, "gt" : snippet[label_ds]}
    chat_prompt = tokenizer.apply_chat_template(chat_template, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(chat_prompt, add_special_tokens=False, return_tensors="pt")
    outputs = model.generate(input_ids=inputs.to("cuda"), max_new_tokens=150)
    answer = tokenizer.decode(outputs[0]).lower()
    part = answer.split(splitter)[1].replace("\n", "").replace("<eos>","")
    part = part.split("reasoning")[0]
    # Check if the output before the reasoning contains only one class
    if "negative" in part and not any (x in part for x in ["positiv", "neutral"]):
        pred = "negative"
    elif "positive" in part and not any (x in part for x in ["negativ", "neutral"]):
        pred = "positive"
    elif "neutral" in part and not any (x in part for x in ["negativ", "positiv"]):
        pred = "neutral"
    else:
        print(f"{colored('The prediction is not clear. The answer was:', 'red', attrs=['bold'])} {answer}")
        continue
    datapoint["pred"] = pred
    results_comp.append(datapoint)
    label = snippet[label_ds]
    if label == pred:
        col = "green"
    else:
        col = "red"
    print(f"Prediction:{colored(pred, col)}\tLabel:{label}\tText: {part}")

results = {"metrics": {}, "prompt": prompt, "details": []}
fn_kwargs={}
label2id = {"negative": 0, "positive": 1, "neutral": 2}
label2id_ds = {"negative": 0, "positive": 1, "neutral": 2}

# Metrics
accuracy = load("accuracy")
precision_weighted = load("precision")
recall_weighted = load("recall")
f1_weighted = load("f1")
conf_matrix = load("confusion_matrix")

references = []
predictions_calc = []
count = 0
for res in results_comp:
    ref_id = label2id_ds[res["gt"]]
    pred_id = label2id[res["pred"]]
    references.append(ref_id)
    predictions_calc.append(pred_id)
    count+=1
    results["details"].append(res)

results["metrics"]["quantity"] = count
results["metrics"]["accuracy"] = accuracy.compute(references = references, predictions = predictions_calc)
precision_result = list(precision_weighted.compute(references = references, predictions = predictions_calc, average = "weighted", labels=[0,1,2]).values())[0]
results["metrics"]["precision_weighted"] = precision_result
recall_result = list(recall_weighted.compute(references = references, predictions = predictions_calc, average = "weighted", labels=[0,1,2]).values())[0]
results["metrics"]["recall_weighted"] = recall_result
f1_result = list(f1_weighted.compute(references = references, predictions = predictions_calc, average = "weighted", labels=[0,1,2]).values())[0]
results["metrics"]["f1_weighted"] = f1_result
conf_matrix_result = str(list(conf_matrix.compute(references = references, predictions = predictions_calc, labels=[0,1,2]).values())[0])
results["metrics"]["conf_matrix"] = conf_matrix_result
cm = confusion_matrix(y_true = references, y_pred = predictions_calc, labels=[0,1,2])

# Save results
Path(rq1_results_path).mkdir(parents=True, exist_ok=True)
filename = rq1_results_path + f"{model_name}_{dataset}_{split}".replace("/", "_")
cm_display = ConfusionMatrixDisplay(cm).plot().figure_.savefig(f"{filename}.pdf")
with open(f"{filename}.txt", "w") as f:
    json.dump(results, f, ensure_ascii=False)