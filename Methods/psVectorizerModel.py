import pslegal as psl
import nltk
import NNP_extractor as npe
from sklearn.datasets import fetch_20newsgroups
import json
from nltk.stem import PorterStemmer
import pickle

#training on tokenize documents

path="./Resource/high_count_advs.json"
file_path="./Resource/masked_continuous_doc/"

#variable to store tokenized documents,non legal docs and content of legal docs
legal_doc=[]
nonlegal_doc=[]
legal_content=[]

#non legal data (20 news group data)
nl_data = fetch_20newsgroups(remove = ('headers', 'footers', 'quotes')) 

#extraction of advocates names and their cases 
f = open (path, "r")
data = json.loads(f.read())
Adv_name=[]
file_length=[]
case_number=[]

for key in data:
    Adv_name.append(key)#scanning each advocate name at a time
    k=0
    for entries in data[key]:#this loop will go through all train and test cases for advocates 
        for entry in data[key][entries]:
            _,case_no=entry.split('_')#extracting the case number 
            case_number.append(case_no)
            file_content=open(file_path+case_no+".txt").read()#reading file content and appending it in variable legal content
            legal_content.append(file_content)
            legal_doc.append(nltk.word_tokenize(file_content))#tokenizing each cases
            k=k+1
    file_length.append(k)#to count total number of cases(test + train) for each advocate

#tokenizing non legal data
for i in range(len(nl_data.data)):
    nonlegal_doc.append(nltk.word_tokenize(nl_data.data[i])) 


ps = PorterStemmer()
#Using pslegal with version==1
psvectorizer = psl.PSlegalVectorizer(version=1)
psvectorizer.fit(legal_doc,nonlegal_doc)

#save model using pickel as trianing takes very long time 
save_model="./re/saved_model/"
filename = 'finalized_model.sav'
pickle.dump(psvectorizer, open(filename, 'wb'))

#To load model using pickel
#loaded_model = pickle.load(open(filename, 'rb'))



save="./re/ALLCASES/"
s=0
j=0
#Creating a json file for each advocate containg case number,phrase and score and term score

for l in file_length:
    #l is total number of cases of Adv_name[j]
    #dividing legal cases for each advocate
    files=legal_content[s:s+l]
    c=case_number[s:s+l]
    s+=l
    
    result={
        Adv_name[j]:{
            "Case_number":[],
            "Phrase_and_get_score":[],
            "term_score":[],
        }       
    }

    print("\nCases for {} .....:".format(Adv_name[j]))
    #below for loop will run for cases for advocate[j]
    for i in range(l):
        print("\n\tCase no {} :".format(c[i]))
        NNP_list = npe.start(files[i])#extracting noun phrases
        test_token=nltk.word_tokenize(files[i])#tokenizing each case
        #applying stemming
        stemm=[]
        for w in test_token:
            #print(w, " : ", ps.stem(w))
            stemm.append(ps.stem(w))
        print("\t\tStemming complete.....")
        #calculation of term score
        term_score=psvectorizer.fit_doc(stemm)
        print("\t\tDoc fitting complete.....")

        #calculating pslegal score for each phrases in NNP_list
        comb=[]
        for key in NNP_list:
            ke=nltk.word_tokenize(key)
            ne=[]
            for k in ke:
                ne.append(ps.stem(k))
            #new_nnp_list.append(ne)
            score=psvectorizer.get_score(ne)
            comb.append((key,score))
            
        new_list= sorted(comb, key = lambda x: x[1],reverse=True)
        result[Adv_name[j]]['Case_number'].append(c[i])
        result[Adv_name[j]]['Phrase_and_get_score'].append(new_list)
        result[Adv_name[j]]['term_score'].append(term_score)
        print("\tcases {}/{} completed.....".format(i+1,l))
    
    with open(save+Adv_name[j]+".json", "w") as write_file:
        json.dump(result, write_file,indent = 4)
    
    j=j+1   
