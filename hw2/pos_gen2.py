#!/us:r/bin/env python
# -*- coding: utf-8 -*-
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
import codecs,sys,string
from nltk import pos_tag
import cPickle as pickle
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
exclude = set(string.punctuation)

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
    sent_pos = []
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
        for i in range(len(h1_pos)):
            h1_pos[i] = list(h1_pos[i])
            h1_pos[i][0] = wnl.lemmatize(h1_pos[i][0])
            h1_pos[i][1] = h1_pos[i][1][0:2]        
            h1_pos[i].append(stemmer.stem(h1_pos[i][0]))
        for i in range(len(h2_pos)):
            h2_pos[i] = list(h2_pos[i])
            h2_pos[i][0] = wnl.lemmatize(h2_pos[i][0])
            h2_pos[i][1] = h2_pos[i][1][0:2]
            h2_pos[i].append(stemmer.stem(h2_pos[i][0]))
        for i in range(len(ref_pos)):
            ref_pos[i] = list(ref_pos[i])
            ref_pos[i][0] = wnl.lemmatize(ref_pos[i][0])
            ref_pos[i][1] = ref_pos[i][1][0:2]
            ref_pos[i].append(stemmer.stem(ref_pos[i][0]))

        sent_pos.append([h1_pos,h2_pos,ref_pos]) 
    pickle.dump(sent_pos, open('sent3_pos.pkl', 'wb'))

