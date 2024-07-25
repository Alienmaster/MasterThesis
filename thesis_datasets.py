from datasets import Dataset, load_dataset, concatenate_datasets
from typing import Tuple

def germeval(split: str, relevance=True) -> Tuple[str, str, Dataset]:
    dataset = "uhhlt/GermEval2017"
    # split = "train" "test_dia" "test_syn" "dev" "full"
    def prepro(datapoint):
        datapoint["sentiment"] = datapoint["Sentiment"]
        datapoint["text"] = datapoint["Text"]
        return datapoint
    if split == "full":
        ds1 = load_dataset(dataset, split="train")
        ds2 = load_dataset(dataset, split="test_dia")
        ds3 = load_dataset(dataset, split="test_syn")
        ds4 = load_dataset(dataset, split="dev")
        ds = concatenate_datasets([ds1, ds2, ds3, ds4])
    else:
        ds = load_dataset(dataset, split=split)
    if relevance:
        ds = ds.filter(lambda x: x["Relevance"] == 1)
    ds = ds.map(prepro)
    ds = ds.remove_columns(["Sentiment", "Text", "Aspect:Polarity", "Relevance"])
    return dataset, split, ds

def omp(split="full") -> Tuple[str, str, Dataset]:
    dataset = "Alienmaster/omp_sa"
    def prepro(datapoint):
        if not datapoint["Headline"]:
            datapoint["text"] = datapoint["Body"]
        elif not datapoint["Body"]:
            datapoint["text"] = datapoint["Headline"]
        else:
            datapoint["text"]  = f"{datapoint['Headline']}. {datapoint['Body']}"
        datapoint["sentiment"] = datapoint["Category"].lower()
        datapoint["ID"] = datapoint["ID_Post"]
        return datapoint
    ds = load_dataset(dataset, split=split)
    ds = ds.map(prepro)
    ds = ds.remove_columns(["Category", "Headline", "Body", "ID_Post"])
    return dataset, split, ds

def schmidt(split) -> Tuple[str, str, Dataset]:
    dataset = "Alienmaster/german_politicians_twitter_sentiment"
    def prepro(datapoint):
        if datapoint["majority_sentiment"] == 1:
            datapoint["sentiment"] = "positive"
        elif datapoint["majority_sentiment"] == 2:
            datapoint["sentiment"] = "negative"
        else:
            datapoint["sentiment"] = "neutral"
        return datapoint
    if split == "full":
        ds2 = load_dataset(dataset, split="test")
        ds1 = load_dataset(dataset, split="train")
        ds = concatenate_datasets([ds1, ds2])
    else:
        ds = load_dataset(dataset, split=split)
    ds_m = ds.map(prepro)
    ds_m = ds_m.remove_columns("majority_sentiment")
    return dataset, split, ds_m

def wikipedia(split="10k") -> Tuple[str, str, Dataset]:
    dataset = "Alienmaster/wikipedia_leipzig_de_2021"
    def prepro(datapoint):
        datapoint["sentiment"] = datapoint["label"]
        return datapoint
    ds = load_dataset(dataset, split=split)
    ds_m = ds.map(prepro)
    ds_m = ds_m.remove_columns("label")
    return dataset, split, ds_m