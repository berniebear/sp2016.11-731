#!/us:r/bin/env python
# -*- coding: utf-8 -*-
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet as wn
from features import *
import codecs,sys,string
exclude = set(string.punctuation)
output = open('meteor_score.csv','w')

# compute fragments
def frag(match_pair) :
    if len(match_pair) == 0:
        return 0
    h = match_pair[0][0]
    ref = match_pair[0][1]
    del match_pair[0]
    fragment = 1
    # search for consective  matches
    for i in xrange(1, len(match_pair) + 1) :
        if [h + i, ref + i] in match_pair :
            match_pair.remove([h + i, ref + i])
        else :
            break
    # iterative search
    if len(match_pair) != 0 :
        fragment += frag(match_pair)
    return fragment

# check synonym
def check_syn(w, ref):
    synsets = wn.synsets(w)
    for synset in synsets:
        for ref_w in ref:
            if ref_w in synset.lemma_names():
                return 1
    return 0

# meteor implemetation
def meteor_score(h, ref, stemmer, alpha, beta, gamma):
    h2 = list(h)
    ref2 = list(ref)
    t = len(h)
    r = len(ref2)
    match_pair = []
    # exact match
    exact = 0.0
    for idx1, w in enumerate(h):
        if w in ref2:
            exact += 1
            ref_count = ref2.count(w)
            org_count = ref.count(w)
            match_pair.append([idx1, ref.index(w,org_count-ref_count)])
            ref2.remove(w)
            h2.remove(w)
    # stemming
    h_stem_org = [stemmer.stem(x) for x in h]
    r_stem_org = [stemmer.stem(x) for x in ref]
    h_stem = [stemmer.stem(x) for x in h2]
    r_stem = [stemmer.stem(x) for x in ref2]
    h_stem2 = list(h_stem)
    r_stem2 = list(r_stem)    

    stem = 0.0
    for idx1, w in enumerate(h_stem):
        if  w in r_stem2:
            stem += 1
            ref_count = r_stem2.count(w)
            org_count = r_stem_org.count(w)
            ref_count_h = h_stem2.count(w)
            org_count_h = h_stem_org.count(w)
            match_pair.append([h_stem_org.index(w,org_count_h-ref_count_h), r_stem_org.index(w,org_count-ref_count)])
            h_stem2.remove(w)
            r_stem2.remove(w)

    # synonymy
    synonym = 0.0
    #synonymy = sum(1 for w in h2 if check_syn(w, ref2) == 1)
 
    m = exact + stem + synonym

    precision = m/t
    recall = m/r
    
    # compute F-score
    if precision != 0.0 and recall != 0.0:
        fmean = precision*recall/(alpha*precision + (1-alpha)*recall)
    else:
        fmean = 0.0

    # compute fragments
    if not (precision == 0.0 or recall == 0.0) and t != 1:
        fragmentation = (frag(match_pair) - 1) / float(t - 1)
        DF = gamma * pow(fragmentation, beta)
        final = fmean * (1 - DF)
    else:
        final = fmean

    return final

def remove_punctuation(context):
    return ''.join(ch for ch in context if ch not in exclude)
 
def main():
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
    counter = 0
    for h1, h2, ref in islice(sentences(), opts.num_sentences):
        counter += 1
        # remove [unctuation
        h1 = remove_punctuation(' '.join(h1)).split(' ')
        h2 = remove_punctuation(' '.join(h2)).split(' ')
        ref = remove_punctuation(' '.join(ref)).split(' ')
        #rset = set(ref)
        h1_score = meteor_score(h1, ref, stemmer, opts.alpha, opts.beta, opts.gamma)
        h2_score = meteor_score(h2, ref, stemmer, opts.alpha, opts.beta, opts.gamma)
        sys.stderr.write(str(counter) + '\n')

        #output.write('%s,%s\n'%(str(h1_score),str(h2_score)))
        #h1_score = 1 #pos_score(h1,ref)
        #h2_score = 1 #pos_score(h2,ref)
        print(-1 if h1_score > h2_score else # \begin{cases}
                (0 if h1_score == h2_score
                    else 1)) # \end{cases}
 

# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
