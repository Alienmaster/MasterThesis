import json
from pathlib import Path
import sys
from datasets import tqdm
from evaluate import load
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from termcolor import colored
from transformers import AutoTokenizer, AutoModelForCausalLM
from thesis_datasets import germeval, omp, schmidt

model_info = sys.argv[1]
model_name = sys.argv[1].split(" ")[0]
splitter = sys.argv[1].split(" ")[1]
ds_name = sys.argv[2].split(" ")[0]
split_comm = sys.argv[2].split(" ")[1]


# Load model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
model.config.pad_token_id = model.config.eos_token_id

label_ds = "sentiment"
label_model = "label"
text_ds = "text"

# Load dataset
if ds_name == "schmidt":
    dataset, split, dataset_loaded = schmidt(split_comm)
if ds_name == "omp":
    dataset, split, dataset_loaded = omp()
if ds_name == "germeval":
    dataset, split, dataset_loaded = germeval(split_comm)

prompt_incontext = "Classify the sentiment of the text into ONE of the three classes: neutral, negative or positive. Text: "

results_comp = []
# GermEval
context_ge = {"context": "GermEval",
                    "neutral": "köln: wo sich in der bahn ein mittfünfziger im trikot neben dich setzt und dir lebenstipps gibt. ich würde nicht tauschen wollen.",
                    "negative": "@DB_Bahn manchmal fragt man sich, warum es euch überhaupt noch gibt!!",
                    "positive": "RT @holgi: Hui, die neuen QR-Lesegeräte der Bahn sind mal sauschnell... Huiuiui"
                    }

# Schmidt
context_schmidt = {"context": "Schmidt",
                  "neutral": "Das ist ein Problem, was zu einem massiven Risiko für Kinder und Eltern führt. Wenn Ältere geimpft sind, steigt Risiko Umgeimpfter stark, weil Schutzmassnahmen entfallen da Gesamtinzidenz nicht hoch ist. Dabei riskieren 7% der Kinder und 14% der Kinder #LongCovid Malte Kreutzfeldt @MKreutzfeldt · 13. Apr. Wenn 25 % geschützt sind, bezieht sich die Inzidenz also nur noch auf die verbliebenen 75 %. In dieser Gruppe entspricht eine Gesamt-Inzidenz von 100 dann einer faktischen Inzidenz von 133. Das Risiko für die Ungeimften steigt bei gleichem Grenzwert also permanent an. [11/11] ",
                  "negative": "#Altmaier leistet Beihilfe zu perfidem #Lohndumping bei der #Lufthansa. Statt mit zig Steuermilliarden private Verluste zu sozialisieren und dann Beschäftigte dafür bluten zu lassen muss die #GroKo dem Lohndumping auf Staatskosten einen Riegel vorschieben! tagesschau.de Lufthansa: Lohndumping auf Staatskosten? Lufthansa wickelt Konzerngesellschaften wie Germanwings ab, während es eine neue Airline gründet: Eurowings Discover. Dort wird nach Recherchen von Report Mainz zum Teil weniger Gehalt bezahlt -...",
                  "positive": "Seien wir ehrlich, wir hätten uns ein besseres Ergebnis gewünscht, aber das Wichtige ist: Wir haben uns etabliert, wir haben eine Kernwählerschaft, wir sind gekommen, um zu bleiben. Danke an alle Wähler und vor allem auch an alle fleißigen Wahlkampfhelfer! "
                  }
# OMP
context_omp = {"context": "OMP",
                  "neutral": "Wenn man den Treibhausgasausstoß pro Kopf betrachtet sind Länder wie China, Indien und Brasilien noch weit von Frankreich und Italien entfernt.",
                  "negative": "Einfach mal den Spieß umdrehen! Erdogan mal eine klare Absage erteilen und ihn daran erinnern, dass 90% der Türkei ein bäuerliches Land zw. dritter und zweiter Welt darstellen!",
                  "positive": "du kannst sie auch ausdrucken und unters Kopfpolster legen, dann kannst du dich jeden Morgen daran erfreuen. Bei mir hängt sie im Büro an der Wand, nur zur Sicherheit, falls das Internet mal ausfallen sollte."
                  }

# Select context
context_selection = [context_ge, context_omp, context_schmidt]
# Set a name for the results
context_name = "GermEval" # OMP, Schmidt, All
# Select folder to save results
rq21_results_path = "results_rq21/"

# Building context
context = []
for c in context_selection:
    history1_u = {"role" : "user",
                "content": f"{prompt_incontext}{c['neutral']}"}
    history1_a = {"role" : "assistant",
                "content": "neutral"}
    history2_u = {"role" : "user",
                "content" : f"{prompt_incontext}{c['positive']}"}
    history2_a = {"role" : "assistant",
                "content" : "positive"}
    history3_u = {"role" : "user",
                "content" : f"{prompt_incontext}{c['negative']}"}
    history3_a = {"role" : "assistant",
                "content" : "negative"}
    context.append(history1_u)
    context.append(history1_a)
    context.append(history2_u)
    context.append(history2_a)
    context.append(history3_u)
    context.append(history3_a)

# Prediction
for snippet in tqdm(dataset_loaded):
    text = snippet["text"]
    actual_text = {"role" : "user",
                     "content" : f"{prompt_incontext} {text}"}
    chat_template = list(context)
    chat_template.append(actual_text)
    datapoint = {"text" : text, "gt" : snippet[label_ds]}
    chat_prompt = tokenizer.apply_chat_template(chat_template, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(chat_prompt, add_special_tokens=False, return_tensors="pt")
    outputs = model.generate(input_ids=inputs.to("cuda"), max_new_tokens=150)
    answer = tokenizer.decode(outputs[0]).lower()
    part = answer.rsplit(splitter)[-1].replace("</s>","").replace("<eos>","")
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

# Metrics
results = {"metrics": {}, "prompt": prompt_incontext, "context" : context_name, "details": []}
fn_kwargs={}
label2id = {"negative": 0, "positive": 1, "neutral": 2}
label2id_ds = {"negative": 0, "positive": 1, "neutral": 2}

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
Path(rq21_results_path).mkdir(parents=True, exist_ok=True)
filename = rq21_results_path + f"{model_name}_{dataset}_{split}_{context_name}".replace("/", "_")
cm_display = ConfusionMatrixDisplay(cm).plot().figure_.savefig(f"{filename}.png")
with open(f"{filename}.txt", "w") as f:
    f.write(json.dumps(results, ensure_ascii=False))