import argparse
from transformers import BertTokenizer
import json

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--input_file')
arg_parser.add_argument('--output_dir')
arg_parser.add_argument('--src_lang')
arg_parser.add_argument('--tgt_lang')
arg_parser.add_argument('--bert_model')
args = arg_parser.parse_args()

def split_data(args):
    print('Splitting data')
    with open(args.input_file,'r') as f:
        json_data=json.load(f)
    
    with open(args.output_dir+'/raw.'+args.src_lang,'w') as raw_src,open(args.output_dir+'/raw.'+args.tgt_lang,'w') as raw_tgt:
        for key in json_data.keys():
            invocation = json_data[key]['invocation']
            cmd = json_data[key]['cmd']
            print(invocation,file=raw_src)
            print(cmd,file=raw_tgt)

def tokenize(args,lang):
    print(f'tokenizing {lang}')

    tokenizer=BertTokenizer.from_pretrained(args.bert_model)
    with open(args.output_dir+'/raw.'+lang,'r') as f_in,open(args.output_dir+'/tkn.'+lang,'w') as f_out:
        for line in f_in:
            print(tokenizer.tokenize(line),file=f_out)

split_data(args)

tokenize(args,args.src_lang)
tokenize(args,args.tgt_lang)