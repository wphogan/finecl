<div align="center">

# Fine-grained Contrastive Learning for Relation Extraction

</div>

### Description

This repo contains the source code for the EMNLP 2022 paper [Fine-grained Contrastive Learning for Relation Extraction](https://coming_soon). 

### Project Structure

```
├── configs                <- Configuration files
│   ├── experiment               <- Experiment configs
│   ├── mode                     <- Mode configs
│   ├── trainer                  <- Trainer configs
│   └── config.yaml              <- Main config 
│
├── data                   <- Project data
│
├── logs                   <- Logs and saved checkpoints
│
├── preprocess             <- Preprocessing scripts
│
├── saved_models           <- Saved models
│
├── src                    <- Source code
│   ├── stage_1                  <- Stage 1 code: record learning order
│   ├── stage_2                  <- Stage 2 code: contrastive pre-training
│   └── stage_3                  <- Stage 2 code: fine-tuning
│       ├── doc_level            <- Fine-tune for doc-level relation extraction
│       └── sentence_level       <- Fine-tune for sentence-level relation extraction
│
├── requirements.txt       <- File for installing python dependencies
├── run.py                 <- Controller
└── README.md
```


## How to run

#### Pre-run Notes:
- For convenience, we provide the complete [preprocessed datasets, saved models, and logs](https://drive.google.com/drive/folders/13-iTHhde8B5BQPNk8bCA0z6dxxo42ov1?usp=sharing) for all of our experiments.
- By using the preprocessed data and saved models, you can jump to any stage detailed below without needing to run the previous stages.
- This repo uses [Hyrda](https://hydra.cc/) to run and configure experiments. You can override any parameter from command line like this: `CUDA_VISIBLE_DEVICES=0 python run.py trainer.learning_rate=1e-5`.
- Please see the [FineCL paper](https://coming_soon) for details about each stage of training and evaluation.
- The code in this repo was developed using Python (v3.9), PyTorch (v1.12.1), Hugginface transformers (v2.5.0), and CUDA (v11.6)

### Initalize
- Install dependencies from `requirements.txt`
- Install [Apex](https://github.com/NVIDIA/apex)
- Download the data and saved models via the command line: <br>`gdown --no-check-certificate --folder https://drive.google.com/drive/u/1/folders/13-iTHhde8B5BQPNk8bCA0z6dxxo42ov1`
- Unzip `data.zip` and then move both `data` and `saved_models` into the project's root directory.
- [OPTIONAL] All the preprocessed data is provided, but if you'd like to preprocess the data yourself, run: <br> `python run.py mode=preprocess.yaml`
      
### Stage 1 – Record learning order:
- The exact learning order data used in the paper is provided in `data/erica_data/order_dict_augmented.json`. However, if you'd like to generate a learning order of relation instances from scratch, do the following:
- There are 10 separate distantly labeled training data files numbered 0 through 9. Run: <br>`CUDA_VISIBLE_DEVICES=0 python run.py experiment=stage_1.yaml erica_file_num=[FILE_NUM]` for each of the 10 files, replacing `[FILE_NUM]` with the appropriate file number for each run.
- Each run outputs a json file containing the epoch and a list of learned unique identifiers (UIDs). A UID identifies a relational instance in the dataset.
- Merge the outputs of each run into a single json file with the format: `{str(UID): int(epoch_leared), ...}`. For example: `{"39582": 2, "49243": 12, ...}`
- This merged file will replace `data/erica_data/order_dict_augmented.json` in Stage 2 of training.
  
### Stage 2 – Fine-grained contrastive pre-training:
- To pre-train a model using Fine-grained contrastive learning (FineCL), run: 
```
CUDA_VISIBLE_DEVICES=0 python run.py experiment=stage_2.yaml
```
    
### Stage 3 – Downstream fine-tuning:

- #### Document-level RE fine-tuning:
  - For document-level fine-tuning, we load the pre-trained model from Stage 2 and train on annotated data from the [DocRED dataset](https://aclanthology.org/P19-1074.pdf). 
    ```
    CUDA_VISIBLE_DEVICES=0 python run.py experiment=stage_3_doc_re.yaml
    ```
    To evaluate the performance of the fine-tuned model from Stage 3, run: 
    ```
    CUDA_VISIBLE_DEVICES=0 python run.py experiment=stage_3_eval.yaml
    ```
    
- #### Sentence-level RE fine-tuning:
  - For sentence-level fine-tuning, we again load the pre-trained model from Stage 2 and train on the following annotated datasets:
    - [TACRED](https://nlp.stanford.edu/projects/tacred/).
    - [SemEval-2010 Task 8](https://aclanthology.org/S10-1006.pdf).
    
    ```
    CUDA_VISIBLE_DEVICES=0 python run.py experiment=stage_3_sentence_re.yaml
    ```
<br>
<br>
<br>

---

**Credits:** This work began as a fork of the [ERICA](https://github.com/thunlp/ERICA) repository.
If you found our code useful, please consider citing:
```
@inproceedings{
    fincl,
    title={Fine-grained Contrastive Learning for Relation Extraction},
    author={William Hogan, Jiacheng Li, and Jingbo Shang},
    booktitle={EMNLP},
    year={2022},
    url={https://coming-soon}
}
```