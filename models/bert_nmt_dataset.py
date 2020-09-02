from fairseq.data import FairseqDataset

class BertNMTDataset(FairseqDataset):
    """A dataset that provides helpers for batching."""

    def __init__(self,src,src_len,trg,trg_len,tokenizer):
        self.src= src
        self.src_len = src_len
        self.trg = trg
        self.trg_len = trg_len
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        print('Getting item')
        src_tensor=self.tokenizer(texts = self.src[index],return_tensors='pt',is_pretokenized=True,padding='max_length')
        trg_tensor=self.tokenizer(texts = self.sr[index],return_tensors='pt',is_pretokenized=True,padding='max_length')

        item_dic ={ 'src': src_tensor, 'trg' : trg_tensor,'src_len':self.src_len[index],'trg_len':self.trg_len[index]}
        return item_dic

    def __len__(self):
        return len(self.src_len)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        print('Collating')
        SRC = []
        TRG = []
        SRC_LEN = []
        TRG_LEN = []

        for sample in samples:
            SRC.append(sample['src'])
            TRG.append(sample['trg'])
            SRC_LEN.append(sample['src_len'])
            TRG_LEN.append(sample['trg_len'])

        SRC = torch.Tensor(SRC)
        TRG = torch.Tensor(TRG)
        SRC_LEN = torch.Tensor(SRC_LEN)
        TRG_LEN = torch.Tensor(TRG_LEN)

        return {'src':SRC,'trg':TRG,'src_len':SRC_LEN,'trg_len':TRG_LEN}


    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.src_len[index]


    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_len[index],self.trg_len[index])