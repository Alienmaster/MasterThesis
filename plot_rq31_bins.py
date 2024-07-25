
import ast
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from nltk.corpus import stopwords
import numpy as np
import spacy
from sqlalchemy import all_

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

# Plot path
plot_path = "plots/"

# Preprocessing parameter
remove_stopwords = True
lemma = True
remove_duplicates = True


# Import stop words in German
german_stop_words = stopwords.words("german")

# Load spacy lemmatizer
nlp = spacy.load("de_core_news_sm", disable=["tagger", "transformer", "parser", "attribute_ruler", "morphologizer"])

dataset = "GermEval" # OMP, Schmidt

# Input dictionaries
Input_dictionaries = {
    "GermEval" : {
        "Llamaaaa" : "dicts/Alienmaster_omp_sa_google_gemma-1.1-7b-it_pos_full.txt"
        # "Gemma" : "dicts/rq31_uhhlt_GermEval2017_google_gemma-1.1-7b-it.txt",
        # "Llama" : "dicts/rq31_uhhlt_GermEval2017_meta-llama_Meta-Llama-3-8B-Instruct.txt",
        # "Mistral" : "dicts/rq31_uhhlt_GermEval2017_mistralai_Mistral-7B-Instruct-v0.2.txt",
    },
    "OMP" : {
        "Gemma" : "dicts/rq31_Alienmaster_omp_sa_google_gemma-1.1-7b-it.txt",
        "Llama" : "dicts/rq31_Alienmaster_omp_sa_meta-llama_Meta-Llama-3-8B-Instruct.txt",
        "Mistral" : "dicts/rq31_Alienmaster_omp_sa_mistralai_Mistral-7B-Instruct-v0.2.txt",
    },
    "Schmidt" : {
        "Gemma" : "dicts/rq31_Alienmaster_german_politicians_twitter_sentiment_google_gemma-1.1-7b-it.txt",
        "Llama" : "dicts/rq31_Alienmaster_german_politicians_twitter_sentiment_meta-llama_Meta-Llama-3-8B-Instruct.txt",
        "Mistral" : "dicts/rq31_Alienmaster_german_politicians_twitter_sentiment_mistralai_Mistral-7B-Instruct-v0.2.txt",
    }
}


def exclusive(neglist, poslist):
    wordlist_n = [x[0] for x in neglist]
    wordlist_p = [x[0] for x in poslist]
    duplicates = [x for x in wordlist_n if x in wordlist_p]
    lexicon = {}
    for inputList in [neglist, poslist]:
        for element in inputList:
            if element[0] in duplicates:
                continue
            if element[0] not in lexicon.keys():
                lexicon[element[0]] = [element[1]]
            else:
                lexicon[element[0]].append(element[1])

    for word in lexicon:
        lexicon[word] = np.mean(lexicon[word])
    lex_list = list(map(list, lexicon.items()))
    return lex_list

def load_preprocess_list(filelist):
    raw_dict = []
    final_dict = []
    poslist = []
    neglist = []
    for filename in filelist:
        with open (filename) as f:
            result = json.load(f)
        raw_dict = raw_dict + result
    
    for element in raw_dict:
        word = element[0]
        wordlemma=word
        if len(word) == 0:
            continue
        if any(char.isdigit() for char in wordlemma):
            continue
        if word == "-":
            continue
        final_dict.append([word, element[1]])
        if element[1] < 0:
            neglist.append([word, element[1]])
        else:
            poslist.append([word, element[1]])
    return final_dict, neglist, poslist

def preprocessing(llm_dicts, remove_stopwords=False, lemma=False, remove_duplicates=False):
    filtered_dicts = {}
    for model in llm_dicts:
        unfiltered = llm_dicts[model]
        negatives = [x for x in unfiltered if x[1] > 0]
        positives = [x for x in unfiltered if x[1] < 0]
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

        filtered_dicts[model] = list(map(list, lexicon.items()))
        
    return filtered_dicts


    # value_lexicon = {}
    # lexicon = {}
    
    # # Get the values for the lowercases
    # for inputList in [neglist, poslist]:
    #     for element in inputList:
    #         word = element[0]
    #         value = element[1]
    #         if remove_stopwords:
    #             if word in german_stop_words:
    #                 continue
    #         if lemma:
    #             if(len(word)== 0):
    #                 continue
    #             word = nlp(word)[0].lemma_
    #             if remove_stopwords:
    #                 if word in german_stop_words:
    #                     continue
    #         if word not in value_lexicon.keys():
    #             value_lexicon[word] = [value]
    #         else:
    #             value_lexicon[word].append(value)
    
    # if remove_duplicates:
    #     new_value_lexicon = {}
    #     for word in value_lexicon:
    #         negative = any(i < 0 for i in value_lexicon[word])
    #         positive = any(i > 0 for i in value_lexicon[word])
    #         if negative and positive:
    #             continue
    #         else:
    #             new_value_lexicon[word] = value_lexicon[word]
    #     value_lexicon = new_value_lexicon
    
    # for word in value_lexicon:
    #     value_lexicon[word] = np.mean(value_lexicon[word])
    # if lemma and gervader:
    #     for inputList in [neglist, poslist]:
    #         for element in inputList:
    #             word = element[0]
    #             value = element[1]
    #             if(len(word)== 0):
    #                 continue
    #             lemma = nlp(word)[0].lemma_
    #             if lemma in value_lexicon.keys():
    #                 lexicon[word] = value_lexicon[lemma]
    #             # if word not in value_lexicon.keys():
    #             #     value_lexicon[word] = [element[1]]
    #             # else:
    #             #     value_lexicon[word].append(element[1])
    #     value_lexicon = lexicon

    # lex_list = list(map(list, value_lexicon.items()))
    # return lex_list

def filter_weak_words(llm_dict):
    llm_dict_p = [word for word in llm_dict if (word[1] > 0.30)]
    llm_dict_n = [word for word in llm_dict if (word[1] < -0.30)]
    llm_dict = llm_dict_n+llm_dict_p
    return llm_dict

def create_dict(dict, name, ds, filter=True):
    if filter:
        dict = filter_weak_words(dict)
    gd = {}
    for entry in dict:
        gd[entry[0]] = entry[1]

    with open(f"{name}_{ds}.txt", "w") as f:
        json.dump(gd, f, ensure_ascii=False)

def read_llm_dict(filenames):
    llm_dicts = {}
    for model in filenames:
        filename = filenames[model]
        with open(filename) as f:
            data = f.read()
            llm_dicts[model] = ast.literal_eval(data)
    return llm_dicts


llm_dicts = read_llm_dict(Input_dictionaries[dataset])
llm_filtered = preprocessing(llm_dicts, remove_stopwords=remove_stopwords, lemma=lemma, remove_duplicates=remove_duplicates)


# Plot limits
num_bins = 20
all_values = []

for model in llm_filtered:
    llm_dict = llm_filtered[model]
    values = [item[1] for item in llm_dict]
    all_values.extend(values)

min_value = min(all_values)
max_value = max(all_values)
bins = np.linspace(min_value, max_value, num_bins + 1)

model_values = []
names = []
# Plot the bins
for model in llm_filtered:
    llm_dict = llm_filtered[model]
    values = [item[1] for item in llm_dict]
    model_values.append(values)
    names.append(model)

plt.hist(model_values, bins=bins, label=names, stacked=True)

# Labeling
plt.xlabel('Sentiment value')
plt.ylabel('Frequency')
# plt.yticks([])
# plt.title(f'Sentiment values of generated lexicon entrys for {ds}')
plt.legend()
plt.xlim((-1,1))

# Save the plot
Path(plot_path).mkdir(parents=True, exist_ok=True)

plot_filename = (
f"""{plot_path}\
RQ31_plot_histogram_\
{dataset}\
{'_duplicates' if remove_duplicates else ''}\
{'_lemma' if lemma else ''}\
{'_stopwords' if remove_stopwords else ''}\
.pdf""")

plt.savefig(plot_filename, bbox_inches='tight')