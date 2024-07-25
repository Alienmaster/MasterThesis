import ast
import json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import numpy as np
import spacy

# Font configuration
rc_fonts = {
    "font.family": "serif",
    "font.size": 22,
    # 'figure.figsize': (5, 3),
    "text.usetex": True,
    'text.latex.preamble':
        r"""
        \usepackage{libertine}
        """,
}
# mpl.rcParams.update(rc_fonts)

# Input dictionary
input_dictionary = "LlamaEmb_schmidt_waeviate.txt"

# Dict path
dict_path = "dicts/"

# Preprocessing parameter
remove_stopwords = True
lemma = True
remove_duplicates = True
filter_weak = False

# Import stop words in German
german_stop_words = stopwords.words("german")

# Load spacy lemmatizer
nlp = spacy.load("de_core_news_sm", disable=["tagger", "transformer", "parser", "attribute_ruler", "morphologizer"])


list_filenames_gemma = [
    # "dicts/rq31_uhhlt_GermEval2017_google_gemma-1.1-7b-it.txt"
    # "dicts/rq31_Alienmaster_omp_sa_google_gemma-1.1-7b-it.txt",
    "dicts/rq31_Alienmaster_german_politicians_twitter_sentiment_google_gemma-1.1-7b-it.txt",
]

list_filenames_llama = [
    # "dicts/rq31_uhhlt_GermEval2017_meta-llama_Meta-Llama-3-8B-Instruct.txt",
    # "dicts/rq31_Alienmaster_omp_sa_meta-llama_Meta-Llama-3-8B-Instruct.txt",
    "dicts/rq31_Alienmaster_german_politicians_twitter_sentiment_meta-llama_Meta-Llama-3-8B-Instruct.txt",
]

list_filenames_mistral = [
    # "dicts/rq31_uhhlt_GermEval2017_mistralai_Mistral-7B-Instruct-v0.2.txt",
    # "dicts/rq31_Alienmaster_omp_sa_mistralai_Mistral-7B-Instruct-v0.2.txt",
    "dicts/rq31_Alienmaster_german_politicians_twitter_sentiment_mistralai_Mistral-7B-Instruct-v0.2.txt",
]

def postprocessing(wordlist, remove_stopwords=False, lemma=False, remove_duplicates=False, filter_weak=False):
    negatives = [x for x in wordlist if x[1] > 0]
    positives = [x for x in wordlist if x[1] < 0]
    lexicon = {}
    
    # Stopword removal and lemmatization
    for inputList in [negatives, positives]:
        for element in inputList:
            word = element[0]
            value = element[1]
            if remove_stopwords:
                if word in german_stop_words:
                    continue
            if lemma:
                if(len(word)== 0):
                    continue
                word = nlp(word)[0].lemma_
            if word not in lexicon.keys():
                lexicon[word] = [value]
            else:
                lexicon[word].append(value)

    # Remove duplicates
    if remove_duplicates:
        new_value_lexicon = {}
        for word in lexicon:
            negative = any(i < 0 for i in lexicon[word])
            positive = any(i > 0 for i in lexicon[word])
            if negative and positive:
                continue
            else:
                new_value_lexicon[word] = lexicon[word]
        lexicon = new_value_lexicon
    
    # Calculate the mean sentiment value
    for word in lexicon:
        lexicon[word] = np.mean(lexicon[word])
    
    # Give all words with the same lemma the same value
    if lemma:
        lemma_lexicon = {}
        for inputList in [negatives, positives]:
            for element in inputList:
                word = element[0]
                value = element[1]
                if(len(word)== 0):
                    continue
                lemma = nlp(word)[0].lemma_
                if lemma in lexicon.keys():
                    lemma_lexicon[word] = lexicon[lemma]
        lexicon = lemma_lexicon

    if filter_weak:
        lexicon = filter_weak_words(lexicon)

    lexicon_list = list(map(list, lexicon.items()))
    return lexicon_list

def filter_weak_words(dict):
    filtered_dict = {}
    for entry in dict:
        word = entry[0]
        value = entry[1]
        if (value < 0.3) and (value > -0.3):
            continue
        filtered_dict[word] = value
    # llm_dict_p = [word for word in llm_dict if (word[1] > 0.30)]
    # llm_dict_n = [word for word in llm_dict if (word[1] < -0.30)]
    # llm_dict = llm_dict_n+llm_dict_p
    return filtered_dict

def save_dict(dict, name):
    gd = {}
    for entry in dict:
        gd[entry[0]] = entry[1]

    with open(name, "w") as f:
        json.dump(gd, f, ensure_ascii=False)

def read_llm_dict(filename):
    wordlist = []
    with open(filename) as f:
        data = f.read()
        llm_dict = ast.literal_eval(data)

    for entry in llm_dict:
        value = llm_dict[entry]
        wordlist.append([entry, value])
    return llm_dict, wordlist

# Load lexicon from file
ctfidf_dict, ctfidf_list = read_llm_dict(input_dictionary)
# Postprocess lexicon
ctfidf_clean_dict = postprocessing(ctfidf_list, remove_stopwords=remove_stopwords, lemma=lemma, remove_duplicates=remove_duplicates, filter_weak=filter_weak)

# Save filtered lexicon
Path(dict_path).mkdir(parents=True, exist_ok=True)

filtered_dict_filename = (
f"""{dict_path}\
{input_dictionary}\
{'_duplicates' if remove_duplicates else ''}\
{'_lemma' if lemma else ''}\
{'_stopwords' if remove_stopwords else ''}\
{'_weak' if filter_weak else ''}\
.txt""")
save_dict(ctfidf_clean_dict,filtered_dict_filename)