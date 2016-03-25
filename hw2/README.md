Team Name: TAIWAN NO. 1 
Team Member: sshiang, Fred, Bernie

A. Result: (Dev set)
Precision: 0.544490

B. Usage:
Fred
glove vectors
Bernie:
python metoer6.py > output.txt

C. Preprocsssing
1. lower case
2. remove punctuation
3. stemming/lemmatization

D. Translation Evaluation Algorithms
0. Framentation (complete Meteor funcationalies)
1. Exact match
2. POS tag + Exact match
3. n-gram match (n=2~4)
4. stem match
5. similarity match by word embeddings (Glove)
6. Doc2Vec (treat each sentence as a doc)

E. Fusion Method and Other Tricks
0. Average late fusion of the inididual score ouput
1. MLP fusion on train-dev set
2. Grid search for (alpha, beta, gamma)
3. Rank SVM
4. Majority vote of our results

F. Tools
1. NLTK: stemmer, stopwords, POS tags
2. Glove for word embeddings


There are three Python programs here (`-h` for usage):

 - `./evaluate` evaluates pairs of MT output hypotheses relative to a reference translation using counts of matched words
 - `./check` checks that the output file is correctly formatted
 - `./grade` computes the accuracy

The commands are designed to work in a pipeline. For instance, this is a valid invocation:

    ./evaluate | ./check | ./grade


The `data/` directory contains the following two files:

 - `data/train-test.hyp1-hyp2-ref` is a file containing tuples of two translation hypotheses and a human (gold standard) translation. The first 26208 tuples are training data. The remaining 24131 tuples are test data.

 - `data/train.gold` contains gold standard human judgements indicating whether the first hypothesis (hyp1) or the second hypothesis (hyp2) is better or equally good/bad for training data.

Until the deadline the scores shown on the leaderboard will be accuracy on the training set. After the deadline, scores on the blind test set will be revealed and used for final grading of the assignment.
