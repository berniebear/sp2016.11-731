#!/us:r/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
import word2vec.test_word2vec as wv
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
import codecs,sys,string
from nltk import pos_tag
import cPickle as pickle
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
exclude = set(string.punctuation)

model_name = 'word2vec/' + 'glove.6B.50d.txt'
w2v_dic, w2v_model = wv.load_word2vec(model_name)

def remove_punctuation(context):
    return ''.join(ch for ch in context if ch not in exclude)
 

# convention to allow import of this file as a module
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
    # PEP8: use ' and not " for strings
    parser.add_argument('-i', '--input', default='data/train-test.hyp1-hyp2-ref',
            help='input file (default data/train-test.hyp1-hyp2-ref)')
    parser.add_argument('-n', '--num_sentences', default=None, type=int,
            help='Number of hypothesis pairs to evaluate')
    # note that if x == [1, 2, 3], then x[:None] == x[:] == x (copy); no need for sys.maxint
    parser.add_argument('-b','--beta', type=float, default=3, help='beta of metoer')
    parser.add_argument('-g','--gamma', type=float, default=0.5, help='gamma of meteor')
    parser.add_argument('-a','--alpha', type=float, default=0.7, help='alpha of meteor')
    opts = parser.parse_args()

    wnl = WordNetLemmatizer() 
    stemmer = PorterStemmer()
    # we create a generator and avoid loading all sentences into a list
    
    def sentences():
        with codecs.open(opts.input,encoding='utf-8') as f:
            for pair in f:
                #yield [sentence.strip().split() for sentence in pair.split(' ||| ')]
                yield [sentence.strip().lower().split() for sentence in pair.split(' ||| ')]
    # note: the -n option does not work in the original code
    counter = 0
    sent_w2v = np.zeros((opts.num_sentences,50*3))
    for h1, h2, ref in islice(sentences(), opts.num_sentences):
        counter += 1
        if counter%100 == 0:
            print counter
        # remove [unctuation
        h1 = remove_punctuation(' '.join(h1)).split(' ')
        h2 = remove_punctuation(' '.join(h2)).split(' ')
        ref = remove_punctuation(' '.join(ref)).split(' ')
        #sys.stderr.write(str(counter) + '\n')
        h1_pos = pos_tag(h1)
        h2_pos = pos_tag(h2)
        ref_pos = pos_tag(ref)

        h1_w2v = np.zeros((1,50))
        h2_w2v = np.zeros((1,50))
        ref_w2v = np.zeros((1,50))
        for i in range(len(h1_pos)):
            h1_pos[i] = list(h1_pos[i])
            temp = wnl.lemmatize(h1_pos[i][0])
            try:
                vec = w2v_model[temp];
                h1_w2v += vec
            except:
                continue
        h1_w2v = h1_w2v/len(h1_pos)
                
        for i in range(len(h2_pos)):
            h2_pos[i] = list(h2_pos[i])
            temp = wnl.lemmatize(h2_pos[i][0])
            try:
                vec = w2v_model[temp];
                h2_w2v += vec
            except:
                continue
        h2_w2v = h2_w2v/len(h2_pos)
        
        for i in range(len(ref_pos)):
            ref_pos[i] = list(ref_pos[i])
            temp = wnl.lemmatize(ref_pos[i][0])
            try:
                vec = w2v_model[temp];
                ref_w2v += vec
            except:
                continue
        ref_w2v = ref_w2v/len(ref_pos)
        x = np.hstack((h1_w2v, h2_w2v, ref_w2v))
        
        sent_w2v[counter-1] = x
    np.savetxt('feat_w2v.csv', sent_w2v, delimiter=",", fmt="%s")        



