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
    
    with open(args.output_dir+'/train.'+args.src_lang,'w') as train_src,open(args.output_dir+'/train.'+args.tgt_lang,'w') as train_tgt,open(args.output_dir+'/test.'+args.src_lang,'w') as test_src,open(args.output_dir+'/test.'+args.tgt_lang,'w') as test_tgt,open(args.output_dir+'/valid.'+args.src_lang,'w') as valid_src,open(args.output_dir+'/valid.'+args.tgt_lang,'w') as valid_tgt:
        i=0
        for key in json_data.keys():
            i+=1
            invocation = json_data[key]['invocation']
            cmd = json_data[key]['cmd']
            if i%10==0:
                if i%20==0:
                    output_file_src=valid_src
                    output_file_tgt=valid_tgt
                else:
                    output_file_src=test_src
                    output_file_tgt=test_tgt
            else:
                output_file_src=train_src
                output_file_tgt=train_tgt
            
            print(invocation,file=output_file_src)
            print(cmd,file=output_file_tgt)

def tokenize(args,lang):
    print(f'tokenizing {lang}')
    tokenizer=BertTokenizer.from_pretrained(args.bert_model)
    with open(args.output_dir+'/raw.'+lang,'r') as f_in,open(args.output_dir+'/tkn.'+lang,'w') as f_out:
        for line in f_in:
            print(tokenizer.tokenize(line),file=f_out)

split_data(args)

# tokenize(args,args.src_lang)
# tokenize(args,args.tgt_lang)