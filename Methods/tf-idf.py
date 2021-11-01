#TF-IDF implementation

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import texthero as hero

df['tfidf'] = hero.do_tfidf(df['clean_text'])

test = pd.read_csv("Query_doc.txt",delimiter ="|",header=None)

test.columns = ["AILA","NAN", "Query"]

test = test.drop(columns=["AILA","NAN"])

test['clean_text'] = test['Query'].apply(lambda x: utils_preprocess_text(x,flg_stemm = False, flg_lemm=True))

test['tfidf'] = hero.do_tfidf(test['clean_text'])

# Vectorise the data
vec = TfidfVectorizer()
X = vec.fit_transform([df['clean_text'][0]]) # `X` will now be a TF-IDF representation of the data, the first row of `X` corresponds to the first sentence in `data`
Y = vec.transform([test['clean_text'][0]])

max_index_array = []
max_similarity_score_array = []
for i in range(len(test)):
    Y = vec.fit_transform([test['clean_text'][i]])
    max_similarity = 0
    max_index = -1
    for j in range(len(df)):
        X = vec.transform([df['clean_text'][j]])
        S = cosine_similarity(X,Y)
        #print(S[i][0])
        if (S[0][0]>max_similarity):
            max_similarity = S[0][0]
            max_index = j

        max_index_array.append(max_index)
        max_similarity_score_array.append(max_similarity)


print(max_index_array)
print(max_similarity_score_array)
