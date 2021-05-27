#!/bin/bash

test_model=$1
mined_num=$2
test_file="data/conala/${mined_num}/test.bin"
parser="gru_parser"

python exp.py \
    --parser ${parser} \
    --cuda \
    --mode test \
    --load_model $1 \
    --beam_size 15 \
    --test_file ${test_file} \
    --evaluator conala_evaluator \
    --save_decode_to decodes/conala/$(basename $1).test.decode \
    --decode_max_time_step 100

