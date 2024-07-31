import json
from pathlib import Path
from datasets import load_dataset
from evaluate import load
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import torch
from thesis_datasets import germeval, omp, schmidt, wikipedia

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Select Model
# model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
model_name = "oliverguhr/german-sentiment-bert"


label_ds = "sentiment"
label_model = "label"
text_ds = "text"

# Folder for results
rq1_results_path = "results_rq1/"

# Choose dataset
dataset, split, dataset_loaded = germeval("test_syn")
# dataset, split, dataset_loaded = omp()
# dataset, split, dataset_loaded = schmidt("test")
# dataset, split, dataset_loaded = wikipedia()

def guhr(ds):
    from germansentiment import SentimentModel
    # texts = []
    predictions = []
    model = SentimentModel()
    
    for text in ds[text_ds]:
        result = model.predict_sentiment([text])[0]
        pred = {"label": result, "text": text}
        predictions.append(pred)
    return predictions

def lxyuan(ds):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    predictions = []
    fn_kwargs={"padding": "max_length", "truncation": True, "max_length": 512}
    tokenizer = AutoTokenizer.from_pretrained("lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    model = AutoModelForSequenceClassification.from_pretrained("lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    model.to(device)
    for text in ds[text_ds]:
        inputs = tokenizer(text, return_tensors="pt", **fn_kwargs)
        inputs.to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_class_id = logits.argmax().item()
        result = model.config.id2label[predicted_class_id]
        pred = {"label": result, "text": text}
        predictions.append(pred)
    return predictions

if model_name == "lxyuan/distilbert-base-multilingual-cased-sentiments-student":
    predictions = lxyuan(dataset_loaded)
elif model_name == "oliverguhr/german-sentiment-bert":
    predictions = guhr(dataset_loaded)

results = {"metrics": {}, "details": []}
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
for ref, pred in zip(dataset_loaded, predictions):
    ref_id = label2id_ds[ref[label_ds]]
    pred_id = label2id[pred["label"]]
    references.append(ref_id)
    predictions_calc.append(pred_id)
    result = {"text": ref[text_ds], "gt": ref[label_ds], "prediction": pred[label_model]}
    results["details"].append(result)
    count+=1

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