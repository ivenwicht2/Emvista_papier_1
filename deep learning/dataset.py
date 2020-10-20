import torch
import torchtext
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import Iterator, BucketIterator
from transformers import AutoTokenizer



def data_extraction(train,valid,device,BATCH_SIZE=16,token = 'default'):
    if token == 'default' :
        tokenize = lambda x: x.split()
    else :
        tokenize = AutoTokenizer.from_pretrained(token)
        tokenize = tokenize.tokenize

    TEXT = Field(sequential=True, tokenize=tokenize, lower=True)
    LABEL = Field(sequential=False)

    tv_datafields = [ ("ID",None),
                 ("QUESTION", TEXT), 
                 ("ENG", None),
                 ("TYPE_ENTITE", LABEL),
                 ("ROLE_SEMANTIQUE", None),
                 ("COMMENTAIRES", None)]

    trn, vld = TabularDataset.splits(
            path="data", 
            train= train, validation=valid,
            format='csv',
            skip_header=True, 
            fields=tv_datafields)

    
    TEXT.build_vocab(trn)
    LABEL.build_vocab(vld)


    
    train_iter, val_iter = BucketIterator.splits(
            (trn, vld),
            batch_sizes=(BATCH_SIZE, BATCH_SIZE),
            device=device, 
            sort_key=lambda x: len(x.QUESTION), 
            sort_within_batch=False,
            repeat=False
    )



    train_dl = BatchWrapper(train_iter, "QUESTION", ['TYPE_ENTITE'])
    valid_dl = BatchWrapper(val_iter, "QUESTION", ['TYPE_ENTITE'])

    return train_dl , valid_dl , len(TEXT.vocab), LABEL.vocab.itos


class BatchWrapper:
    def __init__(self, dl, x_var, y_vars):
        self.dl, self.x_var, self.y_vars = dl, x_var, y_vars
    
    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_var) 
            
            if self.y_vars is not None: 
                y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1).float()
            else:
                y = torch.zeros((1))
   

            yield (x, y)
    
    def __len__(self):
        return len(self.dl)
