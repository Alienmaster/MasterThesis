import ast
import json
import numpy as np
from  tqdm import tqdm
import weaviate
from thesis_datasets import germeval, omp, schmidt

# select dataset
dataset_name, _, _ = germeval("dev")
# dataset_name, _, _ = omp("full")
# dataset_name, _, _ = schmidt("test")

# select the model for the embeddings
model_name = "google/gemma-7b"
model_name = "meta-llama/Meta-Llama-3-8B"


# Select seed dict
seed_dict = "dicts/uhhlt_GermEval2017_cTFIDF_dict500.txt"

collection = f"{model_name}_{dataset_name}".replace('/', '_')

client = weaviate.Client(
    url="http://localhost:8080",  # Replace with your endpoint
    )

def read_llm_dict(filename):

    with open(filename) as f:
        data = f.read()
        llm_dict = ast.literal_eval(data)
    return llm_dict

def get_seed_embeddings(collection_name, word):
    # Get embeddings for the word in the seed lexicon
    query = (
        client.query
        .get(collection_name,"word")
        .with_where({
            'path': "word",
            'operator': 'Equal',
            'valueString': word}
        )
        .with_additional("id")
        .with_limit(5)
    )
    result = query.do()
    if "data" in result.keys():
        return result["data"]["Get"][collection_name]
    else:
        return None

def get_similar_words(batch, seed_word):
    # Get similar words compared by embeddings
    # Every embedding is only taken once for each seed word
    similar_words = []
    ids = set()
    for element in batch:
        id = element["_additional"]["id"]
        results = (
            client.query
            .get(collection, ["source_text", "word"])
            .with_where({
                    'path': "word",
                    'operator': 'NotEqual',
                    'valueString': seed_word})
            .with_near_object(
                {"id" : id,
                "distance": 0.30}
            )
            .with_limit(5)
            .with_additional(["id"])
            .do()
        )["data"]["Get"][collection]
        for result in results:
            id = result["_additional"]["id"]
            if id in ids:
                continue
            ids.add(id)
            word = result["word"]
            source_text = result["source_text"]
            candidate = {"word": word, "source_text": source_text}
            similar_words.append(candidate)
    return similar_words


# Load seed dict
input_dict = read_llm_dict(seed_dict)

extended_lexicon = {}

for seed_word in tqdm(input_dict):
    ids = get_seed_embeddings(collection, seed_word)
    if ids:
        candidates = get_similar_words(ids, seed_word)
        # Every word is taken only once as addition word for every seed word
        candidate_words = set(cword["word"] for cword in candidates if cword["word"] not in input_dict.keys())
        # the seed word and the new words are added to the new dict.
        # the sentiment value of the new words is the value from the seed word
        extended_lexicon[seed_word] = input_dict[seed_word]
        for cword in candidate_words:
            extended_lexicon[cword] = input_dict[seed_word]

filename = collection + "_waeviate.txt"
with open(filename,"w") as f:
    json.dump(extended_lexicon, f, ensure_ascii=False)

