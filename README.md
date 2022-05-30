# COMP0087 Group Project - Python Code Generator

Code generation is becoming an important and trending field in natural language processing (NLP), as it could potentially help to improve programming productivity by developing automatic code. Given some natural language (NL) utterances, the code generator aims to output some source code that completes the task described in the NL intents. Many models for the code generation task have been proposed by the researchers. In particular, TranX is a transition-based neural abstract syntax parser for code generation, it achieves state-of-the-art results on the CoNaLa dataset.

However, existing code generation models suffer from various problems. For example, TRANX often leads to disadvantageous performance when dealing with long and complex code generation tasks. Furthermore, current code generators suffer from learning dependencies between distant positions. In particular, TRANX uses standard bidirectional Long Short-term Memory (LSTM) network as the encoder and decoder, which may lead to this issue due to its sequential computation. TranX also has high complexity and high computational cost due to its recurrent layer type.

To solve these problems, this project explores potential solutions by using TRANX as the baseline, experimenting and modifying the encoder with different networks like Gated Recurrent Units (GRUs) and attentional encoder. In particular, for the code generation task on CoNaLa dataset, both candidate models achieved slightly better results than TRANX in terms of the exact match. In addition, the training time per epoch for our candidate models is less if compared with TRANX. Furthermore, we found that GRU is less inclined to use implicit information. The attentional encoder has a stronger ability in dealing with long context intent and distinguishing ambiguous words.



| Model                     | Corpus BLEU  | Exact Match  |Training Time/Epoch  |
| ------------------------- | ------------ | ------------ |---------------------|
| tranX_LSTM                | 0.301        | 0.017        | 570s                |
| tranX_GRU                 | 0.286        | 0.030        | 400s                |
| tranX_attentional_encoder | 0.282        | 0.024        | 375s                |

## 1 System Architecture

TRANX is a seq-to-action model, in which the input is the natural language utterances that described the task and the output is a series of actions corresponding to some Python source code that completes the task. Please find the workflow of TRANX below.

![IMAGE](https://raw.githubusercontent.com/kzCassie/ucl_nlp/master/IMAGE/Work_flow.png)

TRANX employs an encoder-decoder structure to score AST by measuring the probability of a series of actions. TRANX uses a Bi-LSTM network for the encoder and a standard LSTM for the decoder. This project explores and replaces the encoder with two different network structures: Gated Recurrent Units (GRUs) and attentional encoder.

Figure below gives brief overview of the partial system for the original TRANX model.

![IMAGE](https://raw.githubusercontent.com/kzCassie/ucl_nlp/master/IMAGE/TRANX.jpg)

For TRANX_GRU, we replace the encoder part with a GRU network. In graphical representations, we change the LSTM encoder (highlighted by the dotted squared) in the TRANX architecture above with a GRU network as shown in the figure below.

![IMAGE](https://raw.githubusercontent.com/kzCassie/ucl_nlp/master/IMAGE/GRU.png)

For TRANX_attentional_encoder, we change the encoder part with an attentional encoder, which is also the encoder of the transformer. The corresponding changed part is shown below.

![IMAGE](https://raw.githubusercontent.com/kzCassie/ucl_nlp/master/IMAGE/Transformer_enc.png)

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

The CoNaLa dataset was originally released by CMU. This dataset contains mined corpus of natural language intents and Python code snippets from Stack Overflow. The original data is in JSON format and comes in 3 parts: first, 2379 manually-curated training examples, second 500 human-curated test examples and thirdly 600k automatically-mined examples.

```
# Download original Conala json dataset
# Download pre-processed Colana zipfile from GoogleDrive
!bash pull_data.sh
```

###Clarification on Data Preprocessing

Please note the data were preprocessed with the downloaded files and topk=100000 (First k number from mined file) using the code below. This means we use all of the human-curated data, but only 100k automatically-mined examples for training.

```
train_file = 'data/conala-corpus/conala-train.json'
test_file = 'data/conala-corpus/conala-test.json'
mined_data_file = 'data/conala-corpus/conala-mined.jsonl' # path to the downloaded mined file
topk = 100000 # number of pretraining data to be preprocessed
!python datasets/conala/dataset.py --pretrain=$mined_data_file --topk=$topk
```

After data pre-processing, the NL intents were transformed into lists of tokens, while the Python source code snippets become series of actions through the AST parsing process as explained before. Please see the example of the preprocessed data below.
```
# example of pre-processed data.
from components.dataset import Dataset
n_example = 3
train_set = Dataset.from_bin_file("data/conala/train.gold.full.bin")
for src, tgt in zip(train_set.all_source[:n_example],train_set.all_targets[:n_example]):
    print(f'Source:{src} \nTarget:{tgt} \n')
```
![IMAGE](https://raw.githubusercontent.com/kzCassie/ucl_nlp/master/IMAGE/pre-processed_data.jpg)

We preprocess the json files into several bin files and save them to the folder named `data/canola/${topk}`. These preprocessed files are then used in the next section for training, fine-tuning and testing. In particular, we held out 200 examples from the 2379 manually-curated training data as the validation set. We first trained the model from scratch with the 100k un-curated examples along with the 200 validation examples. Next, we fine-tuned the obtained model using the 2179 remaining training data and evaluated the same 200 evaluation examples to form the final model. In the end, we applied the fine-tuned model on the 500 test set on which the results are reported.

## 4 Model Training & Fine-tuning

Hint: It takes several hours to train and fine-tune the model. The best-pre-trained models can be downloaded using the following shell script, **if you just want to test the model to see the model performance, please go to section 5.**

### Download best pretrained models from GoogleDrive

We have uploaded the best-pretrained models to an open-accessed GoogleDrive file, please kindly use the shell script *pull_best_models.sh* to download these models.

```
# Download best pretrained models zipfile from GoogleDrive
!bash pull_best_models.sh
```

### tranX_LSTM (Baseline)
```
# tranX baseline model
# pretrain with mined_num = 100000
！bash scripts/tranX/pretrain.sh 100000

# fine-tune with the gold set
! bash scripts/tranX/finetune.sh 100000
```

### tranX_GRU

```
# GRU model
# pretrain with mined_num = 100000
！bash scripts/GRU/pretrain.sh 100000

# fine-tune with the gold set
! bash scripts/GRU/finetune.sh 100000
```

### tranX_attentional_encoder

```
# Attentional encoder
# pretrain with mined_num = 100000
！bash scripts/transformer/pretrain.sh 100000

# fine-tune with the gold set
! bash scripts/transformer/finetune.sh 100000
```

## 5 Model Testing with the test set provided in CoNaLa dataset.

### tranX_LSTM (Baseline)
```
# tranX baseline model
!bash scripts/tranX/test.sh best_pretrained_models/finetune.conala.default_parser.hidden256.embed128.action128.field64.type64.dr0.3.lr0.001.lr_de0.5.lr_da15.beam15.vocab.src_freq3.code_freq3.mined_100000.bin.mined_100000.bin.glorot.par_state.seed0.bin 100000 default_parser
```

### tranX_GRU
```
# GRU model
!bash scripts/tranX/test.sh best_pretrained_models/finetune.conala.gru_parser.hidden256.embed128.action128.field64.type64.dr0.3.lr0.001.lr_de0.5.lr_da15.beam15.vocab.src_freq3.code_freq3.mined_100000.bin.mined_100000.bin.glorot.par_state.seed0.bin 100000 gru_parser
```
### tranX_attentional_encoder
```
# Transformer
!bash scripts/transformer/test.sh best_pretrained_models/finetune.conala.transformer_enc_parser.enc_nhead2.enc_nlayer1.hidden256.embed128.action128.field64.type64.dr0.3.lr0.001.lr_de0.5.lr_da15.beam15.vocab.src_freq3.code_freq3.mined_100000.bin.mined_100000.bin.glorot.par_state.seed0.bin 100000 transformer_enc_parser
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
