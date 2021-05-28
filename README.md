# COMP0087 Group Project - Python Code Generator

Code generation is becoming a crucial and trending field in natural language processing (NLP), as it may help to improve programming productivity by developing automatic code. TranX is a transition-based neural abstract syntax parser for code generation, it achieves state-of-the-art results on the conala dataset.

However, existing code generation models suffer from various problems, ranging from ... to ... For example, when reviewing the code generation results from tranX, we spot that it has bad performance for the... generation task as illustrated below. TranX uses standard bidirectional Long Short-term Memory (LSTM) network as the encoder and decoder, which may lead to this issue as ...

This project presents potential solutions by using tranX as the baseline, experimenting and modifying the encoder and decoder with different networks like Gated Recurrent Units (GRUs) and Transformer. Please see the performance results on the dataset conala using these three systems.
(hopefully) both candidate models beat the current state-of-the-art tranX model on conala dataset.

| Model             | Results      | Metric             |
| -------           | ------------ | ------------------ |
| tranX_LSTM        | ?            | Corpus BLEU            |
| tranX_GRU         | ?            | Corpus BLEU            |
| tranX_Transformer | ?            | Corpus BLEU            |

## 1 System Architecture

##########insert pic##########

Figure 1 and 2 gives brief overview of the system for tranX_GRU and tranX_Transformer respectively.


![Sysmte Architecture](doc/system.png)

## 2 Project Setup
This project can be either run on colab or the local machine. Please find the project set up in the corresponding subsection below. To run it without CUDA, please simply remove the "--cuda" flag from the command line argument in all shell scripts under the file named "scripts".

### 2.1 Colab Setup
```bash
# Setup
from google import colab
colab.drive.mount('/content/drive')

# Imports, login, connect drive
import os
from pathlib import Path
import requests
from google.colab import auth
auth.authenticate_user()
from googleapiclient.discovery import build
drive = build('drive', 'v3').files()

# Recursively get names
def get_path(file_id):
    f = drive.get(fileId=file_id, fields='name, parents').execute()
    name = f.get('name')
    if f.get('parents'):
        parent_id = f.get('parents')[0]  # assume 1 parent
        return get_path(parent_id) / name
    else:
        return Path(name)

# Change directory
def chdir_notebook():
    d = requests.get('http://172.28.0.2:9000/api/sessions').json()[0]
    file_id = d['path'].split('=')[1]
    path = get_path(file_id)
    nb_dir = 'drive' / path.parent
    os.chdir(nb_dir)
    return nb_dir

!cd /
chdir_notebook()
```
### 2.2 Local Machine Setup
```bash
# Clone our project repository into the local machine
git clone https://github.com/kzCassie/ucl_nlp
# Enter the project file
cd ucl_nlp

# Create virtual environments
python3 -m venv config/env
# Activate virtual environment
source config/env/bin/activate
# Install all the required packages
pip install -r requirements.txt
```
## 3 Data Loading & Data Preprocessing

Run the following shell script to get the Conala json file from http://www.phontron.com/download/conala-corpus-v1.1.zip, and download the preprocessed Conala zipfile from GoogleDrive.

```
# Download original Conala json dataset
# Download pre-processed Colana zipfile from GoogleDrive
!bash pull_data.sh
```

###Clarification on Data Preprocessing

Please note the data were preprocessed with the downloaded mined file and topk=100000 (First k number from mined file) using the code below.

```
mined_data_file = "data/conala-corpus/conala-mined.jsonl" # path to the downloaded mined file
topk = 100000 # number of pretraining data to be preprocessed
!python datasets/conala/dataset.py --pretrain=$mined_data_file --topk=$topk
```
We preprocess the json files into several bin files and save them to the folder named `data/canola/${topk}`. These preprocessed files are then used in the next section for training, fine-tuning and testing.

In particular, we preprocess the mined json file (conala-mined.jsonl) and save the results into *mined_100000.bin*, which is then used for the model pre-training. Next, the gold training data are preprocessed with the downloaded train file (*conala-train.json*), these preprocessed files are stored in *train.gold.full.bin*, and they are used for fine-tuning. At the end, we preprocess the test json file *conala-test.json* to *test.bin* and use it for model testing. In total, we use around 100000, 2500 and 500 instances for training, fine-tuning and testing respectively.

Please see the example of the preprocessed data below.
```
# example of pre-processed data.
from components.dataset import Dataset
n_example = 3
train_set = Dataset.from_bin_file("data/conala/train.gold.full.bin")
for src, tgt in zip(train_set.all_source[:n_example],train_set.all_targets[:n_example]):
    print(f'Source:{src} \nTarget:{tgt} \n')
```
![IMAGE](https://github.com/kzCassie/ucl_nlp/blob/master/nlp.jpg)

## 4 Model Training & Fine-tuning

### tranX_LSTM (Baseline)
```
# tranX baseline model
# pretrain with mined_num = 100000
！bash scripts/tranX/pretrain.sh 100000

# fine-tune with mined_num = 100000
! bash scripts/tranX/finetune.sh 100000
```

### tranX_GRU

```
# GRU model
# pretrain with mined_num = 100000
！bash scripts/GRU/pretrain.sh 100000

# fine-tune with mined_num = 100000
! bash scripts/GRU/finetune.sh 100000
```

### tranX_Transformer

```
# Transformer
# pretrain with mined_num = 100000
！bash scripts/transformer/pretrain.sh 100000

# fine-tune with mined_num = 100000
! bash scripts/transformer/finetune.sh 100000
```

## 5 Model Testing

### tranX_LSTM (Baseline)
```
# tranX baseline model
# test with mined_num = 100000
!bash scripts/tranX/test.sh 100000
```
### tranX_GRU
```
# GRU model
# test with mined_num = 100000
!bash scripts/GRU/test.sh 100000
```
### tranX_Transformer
```
# Transformer
# test with mined_num = 100000
!bash scripts/transformer/test.sh 100000
```

## Reference

TranX is described/used in the following two papers:

```
@inproceedings{yin18emnlpdemo,
    title = {{TRANX}: A Transition-based Neural Abstract Syntax Parser for Semantic Parsing and Code Generation},
    author = {Pengcheng Yin and Graham Neubig},
    booktitle = {Conference on Empirical Methods in Natural Language Processing (EMNLP) Demo Track},
    year = {2018}
}

@inproceedings{yin18acl,
    title = {Struct{VAE}: Tree-structured Latent Variable Models for Semi-supervised Semantic Parsing},
    author = {Pengcheng Yin and Chunting Zhou and Junxian He and Graham Neubig},
    booktitle = {The 56th Annual Meeting of the Association for Computational Linguistics (ACL)},
    url = {https://arxiv.org/abs/1806.07832v1},
    year = {2018}
}
```

## Thanks

We are also grateful to the following papers that inspire this work :P
```
Abstract Syntax Networks for Code Generation and Semantic Parsing.
Maxim Rabinovich, Mitchell Stern, Dan Klein.
in Proceedings of the Annual Meeting of the Association for Computational Linguistics, 2017

The Zephyr Abstract Syntax Description Language.
Daniel C. Wang, Andrew W. Appel, Jeff L. Korn, and Christopher S. Serra.
in Proceedings of the Conference on Domain-Specific Languages, 1997
```

We also thank [Li Dong](http://homepages.inf.ed.ac.uk/s1478528/) for all the helpful discussions and sharing the data-preprocessing code for ATIS and GEO used in our Web Demo.
