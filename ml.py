from transformers import pipeline
from datasets import load_dataset
from evaluate import load
import json
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
# model_name = "oliverguhr/german-sentiment-bert"
label_ds = "label"
label_model = "label"
text_ds = "text"

def omp():
    dataset = "Alienmaster/omp_sa"
    split = "full"
    def prepro(datapoint):
        if not datapoint["Headline"]:
            datapoint["text"] = datapoint["Body"]
        elif not datapoint["Body"]:
            datapoint["text"] = datapoint["Headline"]
        else:
            datapoint["text"]  = f"{datapoint['Headline']}. {datapoint['Body']}"
        datapoint["label"] = datapoint["Category"].lower()
        return datapoint
    ds = load_dataset(dataset, split=split)
    ds_update = ds.map(prepro)
    return dataset, split, ds_update

def germeval2017():
    dataset = "uhhlt/GermEval2017"
    # split = "train"
    # split = "test_dia"
    split = "test_syn"
    # split = "dev"
    def prepro(datapoint):
        datapoint["label"] = datapoint["Sentiment"]
        datapoint["text"] = datapoint["Text"]
        return datapoint
    ds = load_dataset(dataset, split=split)
    ds_f = ds.filter(lambda x: x["Relevance"] == 1)
    ds_fm = ds_f.map(prepro)
    return dataset, split, ds_fm

def schmidt2022():
    dataset = "Alienmaster/german_politicians_twitter_sentiment"
    # split = "train"
    split = "test"
    def prepro(datapoint):
        if datapoint["majority_sentiment"] == 1:
            datapoint["label"] = "positive"
        elif datapoint["majority_sentiment"] == 2:
            datapoint["label"] = "negative"
        else:
            datapoint["label"] = "neutral"
        del datapoint["majority_sentiment"]
        return datapoint
    ds = load_dataset(dataset, split=split)
    ds_m = ds.map(prepro)
    return dataset, split, ds_m

def wikipedia():
    dataset = "Alienmaster/wikipedia_leipzig_de_2021"
    split = "10k"
    ds = load_dataset(dataset, split=split)
    return dataset, split, ds

# dataset, split, dataset_loaded = omp()
# dataset, split, dataset_loaded = germeval2017()
# dataset, split, dataset_loaded = schmidt2022()
dataset, split, dataset_loaded = wikipedia()

# lib
from germansentiment import SentimentModel

texts = []
predictions = []
libmodel = SentimentModel()
for text in dataset_loaded[text_ds]:
    result = libmodel.predict_sentiment([text])[0]
    pred = {"label": result}
    predictions.append(pred)

results = {"metrics": {}, "details": []}
fn_kwargs={"padding": "max_length", "truncation": True, "max_length": 512}
# predictions = classifier(dataset_loaded[text_ds], **fn_kwargs)
label2id = {"negative": 0, "positive": 1, "neutral": 2}
label2id_ds = {"negative": 0, "positive": 1, "neutral": 2}
print(predictions)
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
    result = {"text": ref[text_ds], "gt": ref[label_ds], "prediction": pred[label_model]} #, "score": pred["score"]}
    results["details"].append(result)

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

filename = "results/" + f"{model_name}__{dataset}__{split}".replace("/", "_")
cm_display = ConfusionMatrixDisplay(cm).plot().figure_.savefig(f"{filename}.png")
with open(f"{filename}.txt", "w") as f:
    f.write(json.dumps(results, ensure_ascii=False))