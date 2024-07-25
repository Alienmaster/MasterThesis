# Enhancing Sentiment Analysis: Model Comparison, Domain Adaptation, and Lexicon Evolution in German Data
The code in this repository is part of the Masterthesis handed in by Robert Georg Geislinger\
All used code, datasets and models are stated below.\
Also how to use the code and where to change the variables.
## Datasets
[GermEval](https://huggingface.co/datasets/uhhlt/GermEval2017)\
[OMP](https://huggingface.co/datasets/Alienmaster/omp_sa)\
[Schmidt](https://huggingface.co/datasets/Alienmaster/german_politicians_twitter_sentiment)\
[Wikipedia](https://huggingface.co/datasets/Alienmaster/wikipedia_leipzig_de_2016)

## Models
[GerVADER](https://github.com/KarstenAMF/GerVADER)\
[Guhr](https://github.com/oliverguhr/german-sentiment-lib)\
[Lxyuan](https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student)\
[Gemma 7B](https://huggingface.co/google/gemma-7b)\
[Gemma 7B Instruct](https://huggingface.co/google/gemma-7b-it)\
[Llama 2 13B Chat](https://huggingface.co/meta-llama/Llama-2-13b-chat)\
[Llama 3 8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)\
[Llama 3 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)\
[Mistral 7B Instruct v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)\
[Mistral 8x7B Instruct v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)\
[multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct)

## Code and explanation
### RQ1: Zero-shot
#### Information
- Tested Models: GerVADER, Guhr, Lxyuan, Gemma 7B Instruct, Llama 2 13B Chat, Llama 3 8B Instruct, Mistral 7B Instruct, Mistral 8x7B Instruct
- Tested Datasets: GermEval, OMP, Schmidt, Wikipedia\
Not every model runs on every dataset on a nvidia A100 80G

##### GerVADER
- Clone GerVADER
- Convert the dataset with `GerVaderConv.py` into the GerVADER format
- Copy the file into the GerVADER directory
- Run GerVADER: `python GERvaderModule.py`
    - Select 1
    - Choose dataset (GerVADER only supports lowercase filename!)
- Evaluate the results

#### Guhr / Lxyuan
-- TODO --

#### Large Language Models
- `rq1_llms.sh` - Runs the experiments with different models on different datasets
- `rq1_llms.py` - The actual llm code

Results are stored in `results_rq1`

### RQ2.1: In-context learning
#### Information
- Tested Models: Gemma 7B Instruct, Llama 3 8B Instruct, Mistral 7B Instruct
- Tested Datasets: GermEval, OMP, Schmidt

#### Experiments
- `rq21.sh` - Runs the experiments
- `rq21.py` - The actual code. Select here which context to use. GermEval, OMP, Schmidt or all

Results are stored in `results_rq21`

### RQ2.2: LoRA adapters
#### Information
- Tested Models: Gemma 7B Instruct, Llama 3 8B Instruct, Mistral 7B Instruct
- Tested Datasets: GermEval, OMP, Schmidt

#### Experiments
- `rq22.sh` - Runs the experiments. Each configuration three times. Set here the training sizes and the model
- `rq22.py` - The actual training code. Set here the dataset.

### RQ2.3: SetFit
#### Information
- Tested Models: multilingual-e5-large-instruct
- Tested Datasets: GermEval, OMP, Schmidt
#### Experiments
- 

### RQ3 GerVADER adaptations
#### Prerequirements
To change the GerVADER dict, small changes a necessary.\
Replace the following lines in `vaderSentimentGER.py` at line 310 to load a custom lexicon:

old:
```python
        with open('outputmap.txt', 'w') as f:
            print (lex_dict, file=f)
```
new:
```python
        with open('outputmap.txt', 'r') as f:
            new_str_dict = f.read()
            lex_dict = json.loads(new_str_dict)
```
Replace the `outputmap.txt` file with the generated lexicon

## RQ3.1: Lexicon creation by prompting LLMs
### Information
- Tested Models: Gemma, Llama 3, Mistral
- Tested Datasets: GermEval, OMP, Schmidt

### Experiment
- `rq31.py` - The actual code. Set here the dataset or change the context.
    - The dictionary is saved as `dicts/rq31_$dataset$_$model$.txt`
- `rq3_postprocessing.py` - Postprocessing of the Lexicon.

The resulting lexicon can be directly used in GerVADER by replacing the existing `outputmap.txt`.

## RQ3.2: Lexicon extension with LLM embeddings
### Information
- Tested Models: Gemma, Llama 3
- Tested Datasets: GermEval, OMP, Schmidt

### c-TF-IDF
Create and load a seperate virtual environment (e.g., Python 3.10).\
Newer sklearn is not compatible with the cTFIDF implementation.
- `python -m venv envcTFIDF`
- `source envcTFIDF/bin/activate`
- clone git repository: `https://github.com/MaartenGr/cTFIDF.git`
- Install the requirements:
    - `pip install -r cTFIDF/requirements.txt`
- run the script. Set the variable `seed_size` for the size of the seed lexicon. 0 = full lexicon
    - `rq32_ctfidf.py`

### Create embeddings and extend lexicon
- create a folder `weaviate/`
- Start weaviate in a docker container:
    - `docker compose -f docker-compose.yml start -d`
- Create the embeddings for each model and each dataset
- `rq32_create_embeddings.py` - Select the model and the dataset
- `rq32_extend_lexicon.py` - Select the seed lexicon, model and dataset. Extend the seed lexicon.
- `rq3_postprocessing.py` - Postprocessing of the lexicon.

## Plots
All plots can be used with latex backend for font generaration.
- `plot_confucionMatrices.py` - Plot and results for the Confusion Matrices (RQ1 RQ2.1 RQ2.2)
- `plot_rq22.py` - Lineplots with scattering (RQ2.2)
- `plot_rq23.py` - Lineplot with scattering (RQ2.3)
- `plot_rq31_bins.py` - Histograms (RQ3.1)

Generated plots are stored in `plots/`