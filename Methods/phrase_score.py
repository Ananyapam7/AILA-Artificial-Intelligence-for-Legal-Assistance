from pathlib import Path
from keras.preprocessing.text import text_to_word_sequence
import pslegal as psl
import NNP_extractor as npe


path = "./Resource/adv-hireinsharma-cases-docs/100588213"

path1 = "./Resource/adv-hireinsharma-cases-docs/4096809"

original_text= path

text=Path(path).read_text()
result = text_to_word_sequence(text)
text1=Path(path1).read_text()
result1 = text_to_word_sequence(text1) 

#print(result)
NNP_list = npe.start(text)
#print('\n'.join(NNP_list))

"""
legal_tokenized_documents = [['law','reports','or','reporters','are','series','of','books','that','contain','judicial','opinions','from','a','selection','of','case','law','decided','by','courts'],
['when','a','particular','judicial','opinion','is','referenced,','the','law','report','series','in','which','the','opinion','is','printed','will','determine','the','case','citation','format'],
] #two legal documents

nonlegal_tokenized_documents = [['the','data','is','organized','into','20','different','newsgroups,','each','corresponding','to','a','different','topic'],
['some','of','the','newsgroups','are','very','closely','related','to','each','other'],
['the','following','file','contains','the','vocabulary','for','the','indexed','data'],
] #three non-legal documents


psvectorizer = psl.PSlegalVectorizer()
psvectorizer.fit( legal_tokenized_documents, nonlegal_tokenized_documents)

psvectorizer.fit_doc(result)
phrase_score = psvectorizer.get_score(result1)

print(phrase_score)
"""

legal_tokenized_documents=NNP_list
psvectorizer = psl.PSlegalVectorizer()
psvectorizer.fit_legal(legal_tokenized_documents)
psvectorizer.fit_doc(result)
phrase_score = psvectorizer.get_score(['allegation','the' ,'supreme',' court'])

print("\n",phrase_score)

