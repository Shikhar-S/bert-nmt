from fairseq.tasks import FairseqTask, register_task
from transformers import BertTokenizer
from fairseq.data.bert_nmt_dataset import BertNMTDataset
import os

@register_task('bert_nmt_task')
class BertNMTTask(FairseqTask):
    @staticmethod
    def add_args(parser):
        # Add some command-line arguments for specifying where the data is
        # located and the maximum supported input length.
        parser.add_argument('--src_lang',help = 'source language file path')
        parser.add_argument('--trg_lang',help='target language file path')
        parser.add_argument('--data_dir',help='data directory')
        parser.add_argument('--split',help='name of split')
        parser.add_argument('--bert_model',help='bert model name',default='bert-base-cased')
    

    @classmethod
    def setup_task(cls, args, **kwargs):
        # Here we can perform any setup required for the task. This may include
        # loading Dictionaries, initializing shared Embedding layers, etc.
        # In this case we'll just load the Dictionaries.
        tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        return BertNMTTask(args, tokenizer,args.src_lang,args.trg_lang)

    def __init__(self, args,tokenizer, source_lang, target_lang):
        super().__init__(args)
        self.tokenizer=tokenizer
        self.src_l = source_lang
        self.trg_l = target_lang
        self.max_len = tokenizer.model_max_length

    def load_dataset(self, split_path, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""
        print('Loading',split_path)

        prefix = os.path.join(self.args.data_dir,split_path)+'.'

        # Read input sentences.
        
        english, english_lengths = [], []
        with open(prefix + self.src_l, 'r') as file:
            for line in file:
                sentence = line.strip()

                # Tokenize the sentence, splitting on spaces
                tokens = self.tokenizer.tokenize(sentence)

                english.append(tokens)
                english_lengths.append(len(tokens))
                break
        #english = self.tokenizer(texts = english,return_tensors='pt',is_pretokenized=True,padding='max_length')

        # Read input sentences.
        command, command_lengths = [], []
        with open(prefix + self.trg_l, 'r') as file:
            for line in file:
                sentence = line.strip()

                # Tokenize the sentence, splitting on spaces
                tokens = self.tokenizer.tokenize(sentence)

                command.append(tokens)
                command_lengths.append(len(tokens))
                break

        #command = self.tokenizer(texts = command,return_tensors='pt',is_pretokenized=True,padding='max_length')

        assert len(english_lengths) == len(command_lengths)
        print(f'Dataset size {len(english_lengths)}')

        self.datasets[split_path] = BertNMTDataset(src = english,src_len = english_lengths, trg = command, trg_len = command_lengths,tokenizer =self.tokenizer)


    def max_positions(self):
        """Return the max input length allowed by the task."""
        # The source should be less than *args.max_positions*
        return (self.max_len, 1)