#!/bin/bash
set -e
#set -x

mined_num=$1
parser="transformer_enc_parser"

echo "Training using ${mined_num} mined data."
echo "Parser=${parser}."

###########
echo "****Pretraining with mined data****"
train_file="data/conala/${mined_num}/mined_${mined_num}.bin"
dev_file="data/conala/${mined_num}/dev.bin"
vocab="data/conala/${mined_num}/vocab.src_freq3.code_freq3.mined_${mined_num}.bin"
finetune_file="data/conala/${mined_num}/train.gold.full.bin"

seed=0
dropout=0.3
enc_nhead=2
enc_nlayer=1
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=64
type_embed_size=64
lr=0.001
lr_decay=0.5
batch_size=64
max_epoch=100
patience=20
max_num_trial=10
beam_size=15
lstm='lstm'  # lstm
lr_decay_after_epoch=15
valid_every_epoch=5
model_name=${parser}.enc_nhead${enc_nhead}.enc_nlayer${enc_nlayer}.hidden${hidden_size}.embed${embed_size}.action${action_embed_size}.field${field_embed_size}.type${type_embed_size}.dr${dropout}.lr${lr}.lr_de${lr_decay}.lr_da${lr_decay_after_epoch}.beam${beam_size}.$(basename ${vocab}).$(basename ${train_file}).glorot.par_state.seed${seed}

#######
echo "****Fine tuning with gold data****"
finetuned_model_name=finetune.conala.${model_name}

echo "**** Writing results to logs/conala/${finetuned_model_name}.log ****"
mkdir -p logs/conala
echo commit hash: "$(git rev-parse HEAD)" > logs/conala/"${finetuned_model_name}".log

python -u exp.py \
    --parser ${parser} \
    --cuda \
    --seed ${seed} \
    --mode train \
    --batch_size 10 \
    --evaluator conala_evaluator \
    --asdl_file asdl/lang/py3/py3_asdl.simplified.txt \
    --transition_system python3 \
    --train_file ${finetune_file} \
    --dev_file ${dev_file} \
    --pretrain saved_models/conala/${model_name}.bin \
    --vocab ${vocab} \
    --lstm ${lstm} \
    --no_parent_field_type_embed \
    --no_parent_production_embed \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --action_embed_size ${action_embed_size} \
    --field_embed_size ${field_embed_size} \
    --type_embed_size ${type_embed_size} \
    --dropout ${dropout} \
    --patience 10 \
    --max_num_trial 10 \
    --glorot_init \
    --lr ${lr} \
    --lr_decay ${lr_decay} \
    --lr_decay_after_epoch ${lr_decay_after_epoch} \
    --max_epoch 80 \
    --beam_size ${beam_size} \
    --log_every 50 \
    --valid_every_epoch ${valid_every_epoch} \
    --enc_nhead ${enc_nhead} \
    --enc_nlayer ${enc_nlayer} \
    --save_to saved_models/conala/${finetuned_model_name} 2>&1 | tee logs/conala/${finetuned_model_name}.log

. scripts/transformer/test.sh saved_models/conala/${finetuned_model_name}.bin ${mined_num} ${parser} 2>&1 | tee -a logs/conala/${finetuned_model_name}.log