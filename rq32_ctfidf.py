import json
from pathlib import Path
import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from cTFIDF.ctfidf import CTFIDFVectorizer
# Import the datasets from the parentdirectory
sys.path.append("..")
from thesis_datasets import germeval, schmidt, omp

# Dict path
dict_path = "dicts/"

# Size of seed lexicon, if 0 full lexicon
seed_size = 500

# Get data
dataset_name, _, dataset = germeval("train")
# dataset_name, _, dataset = omp("full[50%:]")
# dataset_name, _, dataset = schmidt("train")

docs = pd.DataFrame({'Document': dataset["text"], 'Class': dataset["sentiment"]})
docs_per_class = docs.groupby(['Class'], as_index=False).agg({'Document': ' '.join})

# Create bag of words
count_vectorizer = CountVectorizer(max_df=2).fit(docs_per_class.Document)
count = count_vectorizer.transform(docs_per_class.Document)
words = count_vectorizer.get_feature_names_out()

# Extract top words
ctfidf = CTFIDFVectorizer().fit_transform(count, n_samples=len(docs_per_class)).toarray()

tfidf_per_class = pd.DataFrame(ctfidf.T, index=words, columns=docs_per_class['Class'].values)
label2id = {"negative": 0, "positive": 1, "neutral": 2}
words_per_class = {label: [words[index] for index in ctfidf[label2id[label]].argsort()[-100:]] for label in docs_per_class.Class}
positive = tfidf_per_class[tfidf_per_class['positive'] > 0].positive.sort_values(ascending=False)
negative = tfidf_per_class[tfidf_per_class['negative'] > 0].negative.sort_values(ascending=False)
# neutral = tfidf_per_class[tfidf_per_class['neutral'] > 0].neutral.sort_values(ascending=False)

# Normalization (normalize by 500 first otherwise very small numbers are getting lost in float64)
# Values of the negative class are set to a negative value
negative = ((negative-negative.min())/(negative.max()-negative.min()))*500
negative = -((negative-negative.min())/(negative.max()-negative.min()))
positive = (positive-positive.min())/(positive.max()-positive.min())*500
positive = (positive-positive.min())/(positive.max()-positive.min())

# If seed_size > 0 create seed lexicon otherwise full lexicon
if seed_size > 0:
    positive = positive.head(seed_size)
    negative = negative.head(seed_size)

complete = positive._append(negative)
complete_dict = complete.to_dict()

# Save seed dictionary
Path(dict_path).mkdir(parents=True, exist_ok=True)
filename = dict_path + f"{dataset_name}_cTFIDF_dict{seed_size if seed_size>0 else ''}.txt".replace("/","_")
with open(filename,"w") as f:
    json.dump(complete_dict, f, ensure_ascii=False)
