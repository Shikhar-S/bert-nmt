#!/bin/bash

SRC=nlc
TGT=cmd
SCRIPT_PATH=./script/
OUTPUT_DIR=./data

TOKENIZER_SCRIPT=${SCRIPT_PATH}tokenizer.py
PREPROCESSING_SCRIPT=${SCRIPT_PATH}preprocessing.py


echo "Tokenizing source and target files"
python ${TOKENIZER_SCRIPT} --file ./nlc_cmd/nl2bash-data.json --output_dir $OUTPUT_DIR --src_land $SRC --tgt_lang $TGT
echo "creating train, valid, test..."


echo "running preprocessing script to create fairseq dictionaries."


