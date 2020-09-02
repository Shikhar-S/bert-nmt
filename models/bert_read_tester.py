from bert_nmt_task import BertNMTTask
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src_lang',help = 'source language file path')
parser.add_argument('--trg_lang',help='target language file path')
parser.add_argument('--data_dir',help='data directory')
parser.add_argument('--split',help='name of split')
parser.add_argument('--bert_model',help='bert model name',default='bert-base-cased')

args = parser.parse_args()


task  = BertNMTTask.setup_task(args)
task.load_dataset(args.split)

for epoch in range(1):
    split=task.dataset(args.split)
    itr = task.get_batch_iterator(split)
    print(itr.iterations_in_epoch)
    print(itr.end_of_epoch())
    print(len(itr))
    for batch in itr:
        print(batch)
    for num_updates, batch in enumerate(itr):
        print(batch)
