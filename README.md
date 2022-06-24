<div align="center">

# Fine-grained Contrastive Learning

</div>

## Description

Repository for Fine-grained Contrastive Learning for Relation Extraction. 

## How to run

1. Install dependencies

```yaml
# clone project
git clone [repo link]
cd finecl

# [OPTIONAL] create conda environment
conda create -n finecl python=3.8
conda activate finecl

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install apex accoring to instructions
# https://github.com/NVIDIA/apex

# install requirements
pip install -r requirements.txt
```

2. For convenience, we provide the complete pre-processed datasets for all our experiments. Download each dataset and place it in (&rarr;) the corresponding directory:
   - [DocRED RoBERTa dataset](https://coming_soon) &rarr; `/data/docred_roberta`
   - [DocRED BERT dataset](https://coming_soon) &rarr; `/data/docred_bert`
   - [ERICA pre-training dataset](https://coming_soon) &rarr; `/data/pretrain_data`
   - [SEMEVAL-2010 Task 8](https://coming_soon) &rarr; `/data/semeval`
   - [TACRED](https://coming_soon) &rarr; `/data/tacred`

3. Download the following pre-trained models and place them in `/pretrained_models/`:
   - [Fine-grained Contrastive Learning](https://coming_soon)
   - [ERICA RoBERTa](https://coming_soon)
   - [ERICA BERT](https://coming_soon)
   - [WCL](https://coming_soon)

4. Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

    Stage 1 -- Collect learning order:
    ```yaml
    CUDA_VISIBLE_DEVICES=0 python run.py experiment=stage_1.yaml
    ```
    Stage 2 -- Fine-grained Contrastive Pre-training:
    ```yaml
    CUDA_VISIBLE_DEVICES=0 python run.py experiment=stage_2.yaml
    ```
    Stage 3 -- Downstream fine-tuning:
    ```yaml
    CUDA_VISIBLE_DEVICES=0 python run.py experiment=stage_3_sentence_re.yaml
    ```

You can override any parameter from command line like this

```yaml
CUDA_VISIBLE_DEVICES=0 python run.py trainer.max_epoch=20
```

#### 
**Credits:** This work began as a fork of the [ERICA](https://github.com/thunlp/ERICA) repository.    
<br>
