!pip install rank_bm25

from rank_bm25 import BM25Okapi

query_array_processed = [0]*50
corpus_array_processed = [0]*2914
train_array=df.iloc[:,1:].values

for i in range(2914):
    corpus_array_processed[i] = train_array[i][0]
    query_array=test.iloc[:,1:].values

#test["Query_processed"]
#test.values(columns=[test["Query_processed"]])
#query_array[49][0]

for i in range(50):
    query_array_processed[i] = query_array[i][0]

train_array=df.iloc[:,1:].values
tokenized_corpus = [doc.split(" ") for doc in corpus_array_processed]

bm25 = BM25Okapi(tokenized_corpus)

name = df["Name"]
name = name.str.rstrip('.txt')

# bm25.get_top_n(corpus_array_processed[4].split(" "), name, n=10)

evaluate = evaluate.loc[evaluate['Relevance'] == 1]
# evaluate


count = 0
for i in range(50):
    for j in bm25.get_top_n(query_array_processed[i].split(" "), name, n=10):
        for k in evaluate.loc[evaluate['Query_Number'] == "AILA_Q"+str(i+1)]["Document"]:
            if (j==k):
                count=count+1

print(count)

Precision = count/500
Recall = count/195

print(Precision)
print(Recall)
