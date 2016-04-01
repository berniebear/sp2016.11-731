eadme for Github

Usage: ./decode7 -s [stack_size] -a [distance parameter]> [output]
( the other paramters are the same as described in example code.)

The best score that we achieved: -4830.7803, which is the combination of sshiang, Fred and Bernie’s results, evaluated by the grading function sentence by sentence.


What we tried: 
insert phrase into arbitrary place.
scan sentence both forward and backard, and select the one with highest score.
Distance score for insertion point. Tuning by a parameter alpha with range [0,0.2]
beam search with future prediction: If scanning the sentence from the beginning and seeing the position i in the sentence, we use score of (i,end) sub-sentence from backward scanning.
Parallel processing of all sentences to speed up. We tried some large beam search space ranging from [1000~10000]

--
There are three Python programs here (`-h` for usage):

 - `./decode` a simple non-reordering (monotone) phrase-based decoder
 - `./grade` computes the model score of your output

The commands are designed to work in a pipeline. For instance, this is a valid invocation:

    ./decode | ./grade


The `data/` directory contains the input set to be decoded and the models

 - `data/input` is the input text

 - `data/lm` is the ARPA-format 3-gram language model

 - `data/tm` is the phrase translation model

