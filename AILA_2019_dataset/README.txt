This folder contains the dataset of the AILA Track organized in FIRE 2019.

Track Website : https://sites.google.com/view/fire-2019-aila/
Conference Website : http://fire.irsi.res.in/fire/2019/home

======================================
Description of the folders :

=======================================
1. Folder Name : Object_casedocs
=======================================
* No. of files : 2914
* Description : Contains prior case documents, some of which are relevant to the given queries. In Task 1, the prior cases relevant to each query should be retrieved from this set of documents. [These documents can also be used for Task 2 to construct a training set for supervised models.] 
* Format : Each file in the folder is named as C<id>.txt, e.g., C1.txt, C2.txt, ..., C77.txt, ..., C2914.txt. The numbers in the file names are the identifiers of the cases.  

=======================================
2. Folder Name : Object_statutes
=======================================
* No. of files : 197
* Description : Contains the title and description of 197 statutes, that are relevant to some of the given queries. 
*Format : 
a. Each file in the folder is named as S<id>.txt; the numbers in the file names are the identifiers of the statutes.   
b. Each file contains 2 lines. The first line is the title of the statute. The format is Title:<space><Titletext>
c. The second line is the description. The format is Desc:<space><Descriptiontext>
For example, the file name "S1.txt" contains the title and description of S1 statute.
The first line gives the title: "Title: Power of High Courts to issue certain writs"
The second line gives the description: "Desc: (1) Notwithstanding anything in Article 32 every High Court shall have powers, throughout the territories..."

=======================================
3. File Name : Query_doc.txt
=======================================
* Description : This file contains the 50 queries which are description of situations. Each query has an id such as AILA_Q1, AILA_Q2, ..., AILA_Q50.
* Format of each line: QueryId||<QueryText>

===================================================
4.  File Name : relevance_judgments_priorcases.txt
===================================================
* Description : contains the relevant prior-cases for a query
* Format : <query-id> Q0 <document-id> <relevance>
where, query_id : query identifier. eg., AILA_Q1, AILA_Q2,...., AILA_Q50
document-id : the prior case document (C1,C2,...,C2914)
Relevance : 1, if the document is the correct answer to the query ; 0, otherwise
*Example :
AILA_Q1 Q0 C2341 0 ==> for the queryid AILA_Q1, C2341 is a wrong prior case
AILA_Q4 Q0 C182 1 ==> for the queryid AILA_Q4, C182 is a correct prior case

===================================================
5.  File Name : relevance_judgments_statutes.txt
===================================================
* Description : contains the relevant statutes for a query
* Format : <query-id> Q0 <document-id> <relevance>
where, query_id : query identifier. eg., AILA_Q1, AILA_Q2,...., AILA_Q50
document-id : the prior case document (S1,S2,...,S197)
Relevance : 1, if the statute is the correct answer to the query ; 0, otherwise
*Example :
AILA_Q1 Q0 S10 1 : for the queryid AILA_Q1, S10 is the correct statute
AILA_Q4 Q0 S184 0 : for the queryid AILA_Q4, S184 is the wrong statute

===================================================
Evaluation : The trec_eval tool (version 8.1) was usued for evaluation (downloaded from https://trec.nist.gov/trec_eval/)

* The tool requires the "relevance judgement file" (i.e., relevance_judgments_priorcases.txt / relevance_judgments_statutes.txt depending on the task) and the top file (documents retreived by the system) 
* In the command file type the following command : trec_eval trec_rel_file trec_top_file 

* The format of the trec_top_file is:
<qid> <iter> <docno> <rank> <sim> <run_id>
where,
qid : query identifier. eg., AILA_Q1, AILA_Q2,...., AILA_Q50
iter : ignored by trec_eval, could be anything, eg. "Q0" for all the queries
docno : the statute or prior case document
rank : document position in the ranking
sim : a similarity value returned by the algorithm based on which the documents are ranked
run_id : in case you are submitting multiple runs, use an alphanumeric name

*Examples :
AILA_Q1 Q0 S14 5 0.11026615969581749 R1
For the queryid AILA_Q1 , statute S14 has rank 5 with cosine similarity value of 0.11026615969581749 in runid R1.

AILA_Q2 Q0 C395 744 0.0308289159925753 R2
For the queryid AILA_Q2 , prior case C395 has rank 744 with cosine similarity value of 0.0308289159925753 in runid R2.


* Read More here :
1. https://www-nlpir.nist.gov/projects/trecvid/trecvid.tools/trec_eval_video/A.README
2. http://www.rafaelglater.com/en/post/learn-how-to-use-trec_eval-to-evaluate-your-information-retrieval-system

===================================================
Note that : for the Tasks, we had provided the first 10 queries (AILA_Q1 - AILA_Q10) as training data. The evaluations were done for the remaining 40 queries (AILA_Q11 - AILA_Q50)
===================================================

If you wish to use the dataset in your research, please cite the AILA 2019 overview paper (bibtex information given below). The overview paper is included in this folder, and gives further details of the track and performance of various methods.

@inproceedings{bhattacharya2019fire,
  title={Overview of the FIRE 2019 AILA Track: Artificial Intelligence for Legal Assistance},
  author={Bhattacharya, Paheli and Ghosh, Kripabandhu and Ghosh, Saptarshi and Pal, Arindam and Mehta, Parth and Bhattacharya, Arnab and Majumder, Prasenjit},
  booktitle={CEUR Workshop Proceedings, Vol. 2517 -- Working Notes of the Conference of the Forum for Information Retrieval Evaluation (FIRE)},
  pages={1--12},
  year={2019}
}
