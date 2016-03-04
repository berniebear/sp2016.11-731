#!/us:r/bin/env python
# -*- coding: utf-8 -*-
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet as wn
from scipy import linalg, dot

import gensim
from gensim.models import Doc2Vec
from  gensim.models.doc2vec import *

import codecs,sys,string
exclude = set(string.punctuation)
output = open('meteor_score.csv','w')


def remove_punctuation(context):
    return ''.join(ch for ch in context if ch not in exclude)

def doc2vec_score(h, ref):
    return 0
 
def cos_similar(a,b):
    return dot(a,b.T)/linalg.norm(a)/linalg.norm(b)


#def main():
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
 
    stemmer = PorterStemmer()
    
    # we create a generator and avoid loading all sentences into a list
    def sentences():
        with codecs.open(opts.input,encoding='utf-8') as f:
            for pair in f:
                #yield [sentence.strip().split() for sentence in pair.split(' ||| ')]
                yield [sentence.strip().lower().split() for sentence in pair.split(' ||| ')]
    # note: the -n option does not work in the original code
    '''
    ss =[]
    idx = 0
    for h1, h2, ref in islice(sentences(), opts.num_sentences):
        # remove punctuation
        h1 = remove_punctuation(' '.join(h1)).split(' ')
        h2 = remove_punctuation(' '.join(h2)).split(' ')
        ref = remove_punctuation(' '.join(ref)).split(' ')
        idx += 1
        ss.append(LabeledSentence(words=ref, tags=['ref_'+str(idx)]))
        ss.append(LabeledSentence(words=h1, tags=['h1_'+str(idx)]))
        ss.append(LabeledSentence(words=h2, tags=['h2_'+str(idx)]))
   
    model = Doc2Vec(size=300, alpha=0.25, min_alpha=0.25, workers=8, window=2, min_count=0)  # use fixed learning rate
    model.build_vocab(ss)
    
    

    for epoch in range(10):
        model.train(ss)
        model.alpha -= 0.02  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay

    model.save('./tmp/my_model.doc2vec')
    '''
    # load the model back
    model_loaded = Doc2Vec.load('./tmp/my_model.doc2vec')
    idx = 0
    for h1, h2, ref in islice(sentences(), opts.num_sentences):
        idx += 1;
        
        ref = model_loaded.docvecs["ref_"+str(idx)]
        h1 = model_loaded.docvecs["h1_"+str(idx)]
        h2 = model_loaded.docvecs["h2_"+str(idx)]

        
        print(-1 if cos_similar(ref,h1) > cos_similar(ref,h2) + 0.01  else # \begin{cases}
                (0 if abs(cos_similar(ref,h1) - cos_similar(ref,h2))<=0.01
                    else 1)) # \end{cases}

        #output.write('%s,%s\n'%(str(h1_score),str(h2_score)))
# convention to allow import of this file as a module
#if __name__ == '__main__':
#    main()
