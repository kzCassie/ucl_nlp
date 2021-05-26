#!/bin/bash

test_model=$1
mined_num=$2
test_file="data/conala/${mined_num}/test.bin"

python exp.py \
    --mode test \
    --load_model ${test_model} \
    --beam_size 15 \
    --test_file ${test_file} \
    --evaluator conala_evaluator \
    --save_decode_to decodes/conala/$(basename $1).test.decode \
    --decode_max_time_step 100

