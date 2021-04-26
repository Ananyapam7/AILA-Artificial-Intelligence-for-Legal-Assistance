from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(corpus_array_processed)]

model = Doc2Vec(tagged_data, vector_size=20, window=2, min_count=1, workers=4, epochs = 100)

model.save("test_doc2vec.model")

model= Doc2Vec.load("test_doc2vec.model")

count = 0
for i in range(50):
    for j in model.docvecs.most_similar(positive=[model.infer_vector(word_tokenize(query_array_processed[i]))],topn=10)[0][0]:
        temp = evaluate.loc[evaluate['Query_Number'] =="AILA_Q"+str(i+1)]["Document"]
        for k in temp.str.replace('C', ''):
            if (j==k):
                count=count+1

print(count)

Precision = count/500
Recall = count/195
print(Precision)
print(Recall)
