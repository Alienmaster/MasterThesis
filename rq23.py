import json
from pathlib import Path
import sys
from datasets import ClassLabel
from evaluate import load
from setfit import SetFitModel, Trainer, TrainingArguments
from thesis_datasets import germeval, omp, schmidt

size = int(sys.argv[1])
run = int(sys.argv[2])

model_name = "intfloat/multilingual-e5-large-instruct"
# Results path
rq23_results_path = "results_rq23/"
Path(rq23_results_path).mkdir(parents=True, exist_ok=True)

def add_prompt(datapoint):
    prompt = "Instruct: Classify the sentiment of a given text as either positive, negative, or neutral.\n Query:"
    datapoint["text"] = f"{prompt} {datapoint['text']}"
    return datapoint

dataset_name, _, train_dataset = germeval("train")
_, _, eval_dataset = germeval("dev[:100]")
_, _, test_dataset = germeval("test_syn")

# dataset_name, _, train_dataset = omp("full[50%:]")
# _, _, eval_dataset = omp("full[:100]")
# _, _, test_dataset = omp("full[:50%]")

# dataset_name, _, train_dataset = schmidt2022("train")
# _, _, eval_dataset = schmidt2022("test[:100]")
# _, _, test_dataset = schmidt2022("test")


# Dataset preprocessing
train_dataset = train_dataset.map(add_prompt)
test_dataset = test_dataset.map(add_prompt)
test_dataset = test_dataset.cast_column("sentiment", ClassLabel(names=["negative", "positive", "neutral"]))

# Split the dataset and keep the distribution
train_dataset = train_dataset.cast_column("sentiment", ClassLabel(names=["negative", "positive", "neutral"]))
train_dataset = train_dataset.train_test_split(test_size=3, train_size=size, stratify_by_column="sentiment")["train"]

model = SetFitModel.from_pretrained(
    model_name,
    labels=["0","1","2"],
)

args = TrainingArguments(
    batch_size=32,
    num_epochs=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=10
)

def metrics(y_pred, y_test):
    precision_metric = load("precision")
    recall_metric = load("recall")
    f1_metric= load("f1")
    accuracy_metric = load("accuracy")
    metrics = {}
    metrics["accuracy"] = accuracy_metric.compute(predictions=y_pred, references=y_test)["accuracy"]
    metrics["precision"] = precision_metric.compute(predictions=y_pred, references=y_test, average = "weighted")["precision"]
    metrics["recall"] = recall_metric.compute(predictions=y_pred, references=y_test, average = "weighted")["recall"]
    metrics["f1"] = f1_metric.compute(predictions=y_pred, references=y_test, average = "weighted")["f1"]
    return metrics

args.eval_strategy = args.evaluation_strategy
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    metric=metrics,
    column_mapping={"text": "text", "sentiment": "label"}  # Map dataset columns to text/label as expected by trainer
)

# Train
trainer.train()

# Evaluate
metrics = trainer.evaluate(test_dataset)

# Save metrics
rq23_results_path
filename = rq23_results_path + f"{model_name}_{dataset_name}_{size}_{run}".replace("/", "_")
with open(filename, "w") as f:
    f.write(json.dumps(metrics))

print(f"{size=}")
print(f"{metrics=}")