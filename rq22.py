import json
from pathlib import Path
import sys
import time
from datasets import tqdm, Dataset, ClassLabel
from evaluate import load
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import (accuracy_score, 
                             classification_report, 
                             ConfusionMatrixDisplay,
                             confusion_matrix)
from termcolor import colored
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding, pipeline, Trainer, TrainingArguments, AutoModelForSequenceClassification
from thesis_datasets import germeval, omp, schmidt

# Measure training time
start = time.perf_counter()

size = int(sys.argv[1])
model_name = sys.argv[2]
run = int(sys.argv[3])

# Set dataset
# ds_name, split, dataset = germeval("train")
# _, _, dataset_test = germeval("test_syn")
# ds_name, split, dataset = omp("full[50%:]")
# _, _, dataset_test = omp("full[:50%]")
ds_name, split, dataset = schmidt("train")
_, _, dataset_test = schmidt("test")


# Adapter configuration
rank = 128
alpha = 128
lr = 2e-05
dropout = 0.05
epochs = 2
batch_size = 8

# Output paths
rq22_results_path = "results_rq22/"
Path(rq22_results_path).mkdir(parents=True, exist_ok=True)

rq22_models_path = "models_rq22/"
Path(rq22_models_path).mkdir(parents=True, exist_ok=True)

output_name = f"{model_name}_{ds_name}_bs{batch_size}_r{rank}_a{alpha}_{lr}_lqv_d{dropout}_e{epochs}_s{size}_ch_r{run}".replace("/", "_")

results_path = rq22_results_path + output_name
models_path =  rq22_models_path + output_name

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

def dataset_preperation(ds: Dataset, size):
    test_size = int(size*0.1)
    def prepro(datapoint):
        if datapoint["sentiment"] == "negative":
            datapoint["label"] = 0
        elif datapoint["sentiment"] == "positive":
            datapoint["label"] = 1
        else:
            datapoint["label"] = 2
        return datapoint
    
    def preprocessing_function(examples):
        return tokenizer(examples['text'], truncation=True)
    col_to_delete = ['ID', 'text']
    ds = ds.map(prepro, remove_columns=["sentiment"])
    ds = ds.cast_column("label", ClassLabel(names=[0, 1, 2]))
    ds = ds.train_test_split(test_size=test_size, train_size=size, stratify_by_column="label")
    # For the weighted cross entropy
    class_weights = {}
    class_weights["neg"] = len(ds["train"].to_pandas()) / (2 * ds["train"].to_pandas().label.value_counts()[0])
    class_weights["pos"] = len(ds["train"].to_pandas()) / (2 * ds["train"].to_pandas().label.value_counts()[1])
    class_weights["neu"] = len(ds["train"].to_pandas()) / (2 * ds["train"].to_pandas().label.value_counts()[2])

    tokenized_datasets = ds.map(preprocessing_function, batched=True, remove_columns=col_to_delete)
    tokenized_datasets.set_format("torch")
    return tokenized_datasets, class_weights

# Some configurations for the model before training
def model_configuration_training(model_name):
    model =  AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name,
            num_labels=3,
            device_map="auto",
            offload_folder="offload",
            trust_remote_code=True
            )
    model.config.use_cache = False
    model.config.pad_token_id = model.config.eos_token_id
    # Freeze other layers
    for param in model.parameters():
        param.requires_grad=False
    
    llama_peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        target_modules=[
            "q_proj",
            "v_proj",
            "all-linear"
        ],
    )
    model = get_peft_model(model, llama_peft_config)
    model.print_trainable_parameters()
    model = model.cuda()
    return model

def train(model, class_weights):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        precision_metric = load("precision")
        recall_metric = load("recall")
        f1_metric= load("f1")
        accuracy_metric = load("accuracy")

        logits, labels = eval_pred # eval_pred is the tuple of predictions and labels returned by the model
        predictions = np.argmax(logits, axis=-1)
        precision = precision_metric.compute(predictions=predictions, references=labels, average = "weighted")["precision"]
        recall = recall_metric.compute(predictions=predictions, references=labels, average = "weighted")["recall"]
        f1 = f1_metric.compute(predictions=predictions, references=labels, average = "weighted")["f1"]
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
        # The trainer is expecting a dictionary where the keys are the metrics names and the values are the scores. 
        return {"precision": precision, "recall": recall, "f1-score": f1, 'accuracy': accuracy}

    class WeightedCELossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            # Get model's predictions
            outputs = model(**inputs)
            logits = outputs.get("logits")
            # Compute custom loss
            loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([class_weights["neg"], class_weights["pos"], class_weights["neu"]], device=model.device, dtype=logits.dtype))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=models_path,
        learning_rate=lr,
        lr_scheduler_type= "constant",
        warmup_ratio=0.1,
        max_grad_norm= 0.3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=1
    )

    trainer = WeightedCELossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(models_path)
    # Clear memory after training
    import gc
    del model
    gc.collect()
    torch.cuda.empty_cache()

def predict(test, model, tokenizer):
    y_pred = []
    label_output_to_name = {"LABEL_0" : "negative", "LABEL_1" : "positive", "LABEL_2": "neutral"}
    tqdm_obj = tqdm(test)
    for sample in tqdm_obj:
        text = sample["text"]
        pipe = pipeline(task="text-classification", 
                        model=model, 
                        tokenizer=tokenizer,
                    )
        result = pipe(text)[0]
        pred = label_output_to_name[result["label"]]
        y_pred.append(pred)
        label = sample["sentiment"]
        if label == pred:
            col = "green"
        else:
            col = "red"
        tqdm_obj.set_description(f"P:{colored(pred, col)} L:{label}")
    return y_pred

def evaluate(y_true, y_pred):
    mapping = {'negative': 0, 'positive': 1, 'neutral': 2 }
    def map_func(x):
        return mapping.get(x, 1)

    y_true = np.vectorize(map_func)(y_true)
    y_pred = np.vectorize(map_func)(y_pred)

    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'Accuracy: {accuracy:.3f}')

    # Generate accuracy report
    unique_labels = set(y_true)  # Get unique labels

    for label in unique_labels:
        label_indices = [i for i in range(len(y_true)) 
                        if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred = [y_pred[i] for i in label_indices]
        accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {label}: {accuracy:.3f}')

    # Generate classification report
    class_report = classification_report(y_true=y_true, y_pred=y_pred)
    print('\nClassification Report:')
    print(class_report)
    # Own evaluation
    results = {}
    accuracy = load("accuracy")
    precision_weighted = load("precision")
    recall_weighted = load("recall")
    f1_weighted = load("f1")
    conf_matrix = load("confusion_matrix")
    references = y_true
    predictions_calc = y_pred
    results["accuracy"] = accuracy.compute(references = references, predictions = predictions_calc)
    precision_result = list(precision_weighted.compute(references = references, predictions = predictions_calc, average = "weighted", labels=[0,1,2]).values())[0]
    results["precision_weighted"] = precision_result
    recall_result = list(recall_weighted.compute(references = references, predictions = predictions_calc, average = "weighted", labels=[0,1,2]).values())[0]
    results["recall_weighted"] = recall_result
    f1_result = list(f1_weighted.compute(references = references, predictions = predictions_calc, average = "weighted", labels=[0,1,2]).values())[0]
    results["f1_weighted"] = f1_result
    conf_matrix_result = str(list(conf_matrix.compute(references = references, predictions = predictions_calc, labels=[0,1,2]).values())[0])
    results["conf_matrix"] = conf_matrix_result
    results["time"] = time.perf_counter()-start
    print(f'It took {results["time"]} seconds.')
    print(results)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2])
    print('\nConfusion Matrix:')
    print(conf_matrix)
    
    # Save results
    ConfusionMatrixDisplay(conf_matrix).plot().figure_.savefig(f"{results_path}.png")
    with open(f"{results_path}.txt", "w") as f:
        json.dump(results, f, ensure_ascii=False)

# Setup dataset
tokenized_datasets, class_weights = dataset_preperation(dataset, size)

# Setup model
model = model_configuration_training(model_name=model_name)

# Train
train(model=model, class_weights = class_weights)

# Evaluate
finetuned_model = output_name
tokenizer = AutoTokenizer.from_pretrained(models_path)

model =  AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=Path(models_path),
        num_labels=3,
        device_map="auto",
        trust_remote_code=True
        )

y_pred = predict(dataset_test, model, tokenizer)
y_true = dataset_test["sentiment"]
evaluate(y_true, y_pred)