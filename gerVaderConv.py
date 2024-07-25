import csv
from thesis_datasets import germeval, omp, schmidt, wikipedia

temp_filename = "temp.csv"
output_filename = "dataset_tests.tsv"
# dataset, split, dataset_loaded = germeval("test_syn")
# dataset, split, dataset_loaded = omp()
# dataset, split, dataset_loaded = schmidt("test")
dataset, split, dataset_loaded = wikipedia()
dataset_loaded.to_csv(temp_filename)

count=0
converted_dataset = []
with open(temp_filename) as f:
    reader = csv.DictReader(f)
    for row in reader:
        count+=1
        if "ID" in row.keys():
            converted_dataset.append([row["ID"], row["sentiment"], row["text"]])
        else:
            converted_dataset.append([count, row["sentiment"], row["text"]])

with open("test_new.tsv", "w") as f:
    for a in converted_dataset:
        f.write(f"{a[0]}\t{a[1]}\t{a[2]}\n")