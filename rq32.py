# Based on: https://discuss.huggingface.co/t/generate-raw-word-embeddings-using-transformer-models-like-bert-for-downstream-process/2958/2
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import weaviate
from datasets import tqdm
from tokenizers import pre_tokenizers, normalizers
from thesis_datasets import germeval, schmidt, omp

# Weaviate client
client = weaviate.Client(
    url="http://localhost:8080",  # Replace with your endpoint
    )
print('is_ready:', client.is_ready())

# Choose model and tokenizer

# Gemma - Otherwise the Tokenizer is not able to return a valid word_ids list
model_name = "google/gemma-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer._tokenizer.normalizer = normalizers.Sequence([])
tokenizer._tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split("▁","merged_with_next")])
model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to("cuda")

# # Llama 3 - The tokenizer needs a pad_token
# model_name = "meta-llama/Meta-Llama-3-8B"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token
# model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to("cuda")

# Dataset
# dataset_name, _, ds = germeval(split="full", relevance=False)
# dataset_name, _, ds = omp()
dataset_name, _, ds = schmidt(split="full")

collection = f"{model_name}_{dataset_name}".replace('/', '_')

def get_hidden_states(encoded, words, model, layers):
    with torch.no_grad():
        output = model(**encoded)
    states = output.hidden_states
    # Stack and sum all requested layers
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()
    # Only select the tokens that constitute the requested word
    for word in words:
        position_ids_token = word["position_ids_token"]
        word_tokens_output = output[position_ids_token].mean(dim=0)
        word["word_tokens_output"] = word_tokens_output
    return words

def get_word_vector(sent, tokenizer: AutoTokenizer, model, layers):
    """Get a word vector by first tokenizing the input sentence, getting all token idxs
    that make up the word of interest, and then `get_hidden_states`."""
    encoded = tokenizer(sent,padding=True, truncation=True, return_tensors="pt").to("cuda")
    word_ids = encoded.word_ids()
    input_ids = encoded["input_ids"].tolist()[0]
    words = []
    # Iterate over the unique values in word_ids. 
    for word_id in sorted(set(word_ids) - {None}):
        # Adds an empty list to the actual word
        token_ids = [input_ids[i] for i in range(len(input_ids)) if word_ids[i] == word_id]
        word = tokenizer.decode(token_ids, clean_up_tokenization_spaces=True).strip()
        position_ids_token = np.where(np.array(encoded.word_ids()) == word_id)
        words.append({"word" : word, "word_id": word_id, "token_ids": token_ids, "position_ids_token": position_ids_token})

    return get_hidden_states(encoded, words, model, layers)

def main(sentence: str, layers=None) -> list:
    # Use last four layers by default
    layers = [-4, -3, -2, -1] if layers is None else layers
    embeddings = get_word_vector(sentence, tokenizer, model, layers)
    return embeddings

def create_class(collection):
    class_obj = {
        "class": collection,
        "vectorizer": "text2vec-openai", # This is only to activate T-SNE in weaviate! Otherwise with own embeddings it is not possible to use T-SNE
    }
    # Add the class to the schema
    client.schema.create_class(class_obj)

def add_document_to_weaviate(document, embeddings):
    with client.batch as batch:
        for element in embeddings:
            word = element["word"]
            vector = element["word_tokens_output"]
            properties = {
                    "source_text": document,
                    "word": word
                }
            batch.add_data_object(properties, collection, vector=vector)

def clear_database(collection):
    client.schema.delete_class(collection)

if __name__ == '__main__':
    clear_database(collection)
    create_class(collection)
    for document in tqdm(ds):
        text = document["text"]
        embeddings = main(sentence = text)
        add_document_to_weaviate(text, embeddings)