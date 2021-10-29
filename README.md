# AILA-Artificial-Intelligence-for-Legal-Assistance

## Motivation

In countries following the Common Law system (e.g. UK, USA, Canada, Aus-
tralia, India), there are two primary sources of law – Statutes (established laws,
such as the Constitution of a country) and Precedents (prior cases decided in
courts of law). Statutes deal with applying legal principles to a situation (facts /
scenario / circumstances which lead to filing the case). Precedents or prior cases
help a lawyer understand how the Court has dealt with similar scenarios in the
past, and prepare the legal reasoning accordingly.

When a lawyer is presented with a situation (that will potentially lead to filing
of a case), it will be very beneficial to him/her if there is an automatic system
that identifies a set of related prior cases involving similar situations as well
as statutes/acts that can be most suited to the purpose in the given situation.

## Benefits

Such a system shall not only help a lawyer but also benefit a common man,
in a way of getting a preliminary understanding of the legal aspects pertaining
to a situation, even before he/she approaches a lawyer. The system can assist
him/her in identifying where his/her legal problem fits, what legal actions he/she
can proceed with (through statutes) and what were the outcomes of similar cases
(through precedents).

Motivated by the above scenario, the FIRE 2019 track on ‘Artificial Intelligence for Legal Assistance’ (AILA) proposed two tasks: 
1. Identifying relevant prior cases for a given situation (Precedent Retrieval) 
2. Identifying most relevant statutes for a given situation (Statute Retrieval). 

This is essentially a task in the domain of natural language processing, information retrieval and data mining. 
[This](http://ceur-ws.org/Vol-2517/T1-1.pdf) paper provides an overview of the FIRE 2019 AILA Track.

This repository contains python implimentations of the various methods described in the paper which include the following main techniques:

1. Cosine Similarity
2. Jaccard Similarity
3. Doc2Vec
4. BM 25
5. Tf-Idf
6. textRank
7. Word2Vec Embeddings
8. Word2Vec + Glove Vectors
9. FastText
10. Sent2Vec
11. Bigram / Unigram model + linear Interpolation
12. BERT
13. IFB2 Weighting Model
14. LexPageRank algorithm

used in conjunction with other mechanisms.

Citations and references: 
1. http://ceur-ws.org/Vol-2517/T1-1.pdf
2. https://sites.google.com/view/fire-2019-aila/
3. https://zenodo.org/record/4063986#.YGGZunUzaV6
