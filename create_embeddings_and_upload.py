# https://discuss.huggingface.co/t/generate-raw-word-embeddings-using-transformer-models-like-bert-for-downstream-process/2958/2
import numpy as np
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import sys
import weaviate
import json
from datasets import tqdm
from tokenizers import pre_tokenizers, normalizers, decoders
sys.path.append("../../")
from thesis_datasets import germeval, schmidt, omp

# Weaviate client
client = weaviate.Client(
    url="http://localhost:8080",  # Replace with your endpoint
    )
print('is_ready:', client.is_ready())

# model_name = "google/gemma-7b"
model_name = "meta-llama/Meta-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name)#, legacy=False, from_slow=True)

# tokenizer._tokenizer.normalizer = normalizers.Sequence([]) # Gemma
# tokenizer._tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split("▁","merged_with_next")]) # Gemma
model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to("cuda")
tokenizer.pad_token = tokenizer.eos_token # Llama3

# Dataset
# dataset_name, _, ds = germeval(split="full", relevance=False)
# dataset_name, _, ds = omp()
dataset_name, _, ds = schmidt(split="full")

# collection = f"{model_name}_{dataset_name}".replace('/', '_')
# collection = "GoogleEmb_GE2017"
# collection = "LlamaEmb_GE2017"
# collection = "GoogleEmb_omp"
# collection = "LlamaEmb_omp"
# collection = "GoogleEmb_schmidt"
collection = "LlamaEmb_schmidt"

# def last_token_pool(last_hidden_states: Tensor,
#                  attention_mask: Tensor) -> Tensor:
#     left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
#     if left_padding:
#         return last_hidden_states[:, -1]
#     else:
#         sequence_lengths = attention_mask.sum(dim=1) - 1
#         batch_size = last_hidden_states.shape[0]
#         return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_hidden_states(encoded, words, model, layers):
    with torch.no_grad():
        output = model(**encoded)

    # Get all hidden states
    # print(output)
    # exit()
    states = output.hidden_states
    # Stack and sum all requested layers
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()
    # Only select the tokens that constitute the requested word
    for word in words:
        position_ids_token = word["position_ids_token"]
        word_tokens_output = output[position_ids_token].mean(dim=0)
        word["word_tokens_output"] = word_tokens_output
    return words


def get_word_vector(sent, tokenizer: AutoTokenizer, model, layers, end_pos):
    """Get a word vector by first tokenizing the input sentence, getting all token idxs
    that make up the word of interest, and then `get_hidden_states`."""
    max_length = 4096
    encoded = tokenizer(sent,padding=True, truncation=True, return_tensors="pt").to("cuda")
    word_ids = encoded.word_ids()
    input_ids = encoded["input_ids"].tolist()[0]
    # print(word_ids)
    words = []
    # Iteriere über die unique Werte in word_ids (dies stellt sicher, dass wir über jedes Wort iterieren)
    for word_id in sorted(set(word_ids) - {None}):
        # Füge eine leere Liste für das aktuelle Wort hinzu
        token_ids = [input_ids[i] for i in range(len(input_ids)) if word_ids[i] == word_id]
        word = tokenizer.decode(token_ids, clean_up_tokenization_spaces=True).strip()
        position_ids_token = np.where(np.array(encoded.word_ids()) == word_id)
        if end_pos:
            if np.any((position_ids_token <= end_pos)):
                continue
        # print(words)
        words.append({"word" : word, "word_id": word_id, "token_ids": token_ids, "position_ids_token": position_ids_token})

    return get_hidden_states(encoded, words, model, layers)

def main(sentence: str, layers=None, end_pos=None) -> list:
    # Use last four layers by default
    layers = [-4, -3, -2, -1] if layers is None else layers
    embeddings = get_word_vector(sentence, tokenizer, model, layers, end_pos)
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

def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'

def get_end_pos_of_instrcution(task):
    words = []
    # max_length = 4096
    encoded = tokenizer(task,padding=True, truncation=True, return_tensors="pt").to("cuda")
    word_ids = encoded.word_ids()
    input_ids = encoded["input_ids"].tolist()[0]
    for word_id in sorted(set(word_ids) - {None}):
        # Füge eine leere Liste für das aktuelle Wort hinzu
        token_ids = [input_ids[i] for i in range(len(input_ids)) if word_ids[i] == word_id]
        word = tokenizer.decode(token_ids, clean_up_tokenization_spaces=True).strip()
        position_ids_token = np.where(np.array(encoded.word_ids()) == word_id)
        words.append({"word" : word, "word_id": word_id, "token_ids": token_ids, "position_ids_token": position_ids_token})
    # instruction = words[-1]["position_ids_token"][0]
    instruct_end = words[-1]["position_ids_token"][0][-1]
    # print(instruction)
    # if np.any((instruction > instruct_end)):
    #     print("happy")
    # elif np.any((instruction <= instruct_end)):
    #     print("end")
    return instruct_end

if __name__ == '__main__':
    # task = "Classify the sentiment of a given text into negative, positive, or neutral"
    # instruct_text = get_detailed_instruct(task, "")
    # end_pos = get_end_pos_of_instrcution(instruct_text)
    clear_database(collection)
    create_class(collection)
    for document in tqdm(ds):
        text = document["text"]
        # query = get_detailed_instruct(task, text)
        # print(query)
        embeddings = main(sentence = text)#, end_pos=end_pos)#, layers=[-1])
        add_document_to_weaviate(text, embeddings)
    # print(database.keys())
    # find_embedding("Lokführer", database)
    # for word in database:
    #     print(len(word))