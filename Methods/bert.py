import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from math import sqrt
import copy
import json
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertModel, BertConfig, BertTokenizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

config = BertConfig()
embed_model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class AdvDataset(Dataset):
    """Generate samples of the dataset given an array/list of embeddings/texts
    and targets.
    """

    def __init__(self, idx_dict, mapping_dict, targets_dict, db):
        """Initialize attributes of the dataset.
        Parameters
        ----------
        idx_dict : dict[str]
            Dictionary mapping case IDs to their indices in the provided
            database `db`.
        mapping_dict : dict[List]
            Dictionary mapping advocates to their cases.
        targets_dict : dict[List]
            Dictionary mapping cases to their advocates.
        db : List[str] or array_like
            If model learns representations, it is a list of strings which are
            case texts. If representations are provided, it is a `torch.Tensor`
            where each row is an embedding.
        Notes
        -----
        Structure of idx_dict
        ---------------------
        idx_dict = {
                    case_id_1 : 0,
                    case_id_2 : 1,
                    .
                    .
                    .
                    }
        Structure of mapping_dict
        -------------------------
        mapping_dict = {
                        adv_a : [
                                case_id_1,
                                case_id_2,
                                .
                                .
                                .
                                ],
                        adv_b : [
                                case_id_10,
                                case_id_13,
                                .
                                .
                                .
                                ],
                        .
                        .
                        .
                        }
        Structure of targets_dict
        -------------------------
        targets_dict = {
                        case_id_1 : [
                                    adv_a,
                                    adv_c
                                    ],
                        case_id_2 : [
                                    adv_x,
                                    adv_y,
                                    adv_z
                                    ],
                        .
                        .
                        .
                        }
        """

        super(AdvDataset, self).__init__()
        self.db = db
        self.targets_dict = targets_dict
        self.mapping_dict = mapping_dict
        self.idx_dict = idx_dict
        self.targets_db = self.create_targets()

    def create_targets(self):
        """Create targets of given cases using self.mapping_dict,
        self.idx_dict and self.targets_dict.
        """

        targets = []
        advs = list(self.mapping_dict.keys())
        for idx,case in self.idx_dict.items():
            names = [adv[2:] for adv in self.targets_dict[case]]
            targets.append(torch.tensor([int(adv in names)
                                         for adv in advs],
                                        dtype=torch.float32))
        return torch.stack(targets, dim=0)

    def __len__(self):
        """Return number of samples."""
        if (type(self.db) == list):
            return len(self.db)
        return self.db.shape[0]

    def __getitem__(self, idx):
        """Return a sample and its target."""
        if (type(self.db) == list):
            return self.db[idx], self.targets_db[idx, :]
        return self.db[idx, :].view(1, -1), self.targets_db[idx, :]





class Attention(nn.Module):
    """Scaled-dot product attention."""
    def __init__(self, input_dim, name=None):
        """Defines sizes of all relevant layers.
        
        Parameters
        ----------
        db_size : int
            Size of database with which to compute attention weights.
        input_dim : int
            Input dimensionality of both query and database entries.
        name : str
            Name of the layer to identify with each advocate.
        """
        super(Attention, self).__init__()
        self.input_dim = input_dim
        
        # Defining layers
        self.query = nn.Linear(in_features=input_dim,
                              out_features=input_dim,
                               bias=True)
        self.keys = nn.Linear(in_features=input_dim,
                            out_features=input_dim,
                            bias=True)
        self.values = nn.Linear(in_features=input_dim,
                                out_features=input_dim,
                                bias=True)
        self.name = name
        
    def forward(self, q, v):
        """Carries out forward pass on one set of data and a query.
        
        Parameters
        ----------
        q : torch.Tensor, shape: (1, input_dim)
            Query Tensor.
        v : torch.Tensor, shape: (database_size, input_dim)
            Database of tensors to compute attention against.
        
        Returns
        -------
        context_vector : torch.Tensor, shape: (1, input_dim)
            Attention-weighted sum of database vectors.
        """
        transform_q = self.query(q)
        transform_v = self.values(v)
        transform_k = self.keys(v)
          
        align_wts = torch.matmul(transform_q, transform_k.t())
        attn_wts = F.softmax(align_wts, dim=-1)
        return torch.matmul(attn_wts, transform_v)/transform_k.shape[-1]     


class Embedding_Model(nn.Module):
    def __init__(self, output_dim):
        super(Embedding_Model, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.train()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.fc1 = nn.Linear(in_features=768, out_features=output_dim, bias=False)
        
    def forward(self, x):
        inputs = self.tokenizer(x, return_tensors='pt', truncation=True, padding=True)
        model_outputs = self.model(**inputs)
        outputs = self.fc1(model_outputs.last_hidden_state[:,0,:])
        return outputs


path="./Resource/fold/fold_0/high_count_advs_0.json"
case_target="./Resource/fold/fold_0/case_targets_0.json"
file_path="./Resource/facts/"
f = open (path, "r")
high_court_advs = json.loads(f.read())
f = open (case_target, "r")
case_tar = json.loads(f.read())

def create_mapping(high_court_advs,term):   
    idx_dict={}
    mapping_dict={}
    db=[]
    i=0
    for key in high_court_advs:
        li=[]
        #j=0
        for k in high_court_advs[key][term]:
            _,case_no=k.split('_')
            idx_dict[i]=case_no
            li.append(case_no)
            db.append(open(file_path+case_no+".txt").read())
            i+=1
            #j+=1
            #if j==2:
            #    break
        mapping_dict[key]=li

    return idx_dict,mapping_dict,db,

def doc_vec():
    pass

def create_concatenated(high_court_advs,term):
    db=[]
    for key in high_court_advs:
        li=""
        j=0
        for k in high_court_advs[key][term]:
            _,case_no=k.split('_')
            li+= open(file_path+case_no+".txt").read() 
            j+=1
            if j==2:
                break
        db.append(li)
    print("Concatented representaion created....\n")
    return db


idx_dict_db, mapping_dict_db,db_db=create_mapping(high_court_advs,'db')
#idx_dict_train, mapping_dict_train,db_train=create_mapping(high_court_advs,'train')
idx_dict_test, mapping_dict_test,db_test=create_mapping(high_court_advs,'test')

Db = AdvDataset(idx_dict_db, mapping_dict_db, case_tar ,db_db)
#train_dl=AdvDataset(idx_dict_train, mapping_dict_train, case_tar ,db_train)
train_dl=create_concatenated(high_court_advs,'train')
test_dl= AdvDataset(idx_dict_test, mapping_dict_test, case_tar ,db_test)




model_embd=Embedding_Model(len(high_court_advs))
model_att=Attention(len(high_court_advs))


momentum=0.9
criterion = nn.MSELoss()
optimizer = optim.SGD(model_att.parameters(), lr=0.01)


"""
def create_embedding(model_embd,model_att,q):
    j=0
    for key in high_court_advs:
        #i=0
        for k in high_court_advs[key]['db']:
            _,case_no=k.split('_')
            rep=model_embd(open(file_path+case_no+".txt").read())
            train(model_att,q,rep,dataset.targets_db[j].reshape([1,len(dataset.targets_db[j])]))
            j+=1
            print("\rTraining for cases {} total case done {}...".format(case_no,j),end="")
            #i+=1
            #if i==2:
            #    break

def create_embedding(q,db,target,epochs):
    j=0
    for case in db:
        rep=model_embd(case)
        train(rep,target.reshape([1,len(target)]),q,epochs)
        j+=1
        print("\Total training cases done {}/{}...".format(j,len(db)),end="")

"""

def train(target,test_q_embd,db,epochs):
    for i in range(epochs):
        j=0
        for case in db:
            rep=model_embd(case)
            yhat = model_att(test_q_embd,rep)
            loss = criterion(yhat, target.reshape([1,len(target)]))
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            j+=1
            print("\rTotal training cases done {}/{} for epoch {}/{}...".format(j,len(db),i+1,epochs),end="")
            #if j==4:
            #    break
        print(f"Loss:{loss.item()}")
    
#test_q=db_test[0]
test_q_embd=model_embd(db_test[0])

#create_embedding(test_q_embd,db_train,test_dl.targets_db[0],10)
train(test_dl.targets_db[0],test_q_embd,train_dl,2)

#weights={}

for name, param in model_att.named_parameters():
    if param.requires_grad:
        print(name,param.data)

#with open("weights.json", "w") as write_file:
#        json.dump(weights, write_file,indent = 4)
