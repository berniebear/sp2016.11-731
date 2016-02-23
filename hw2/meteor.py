#!/us:r/bin/env python
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet as wn
import codecs,sys,string

fp = codecs.open('debug0','w', encoding='utf-8')
exclude = set(string.punctuation)


beta = 3
gamma = 0.3
syn = 1

def frag(match_pair) :
    h = match_pair[0][0]
    ref = match_pair[0][1]
    del match_pair[0]

    for i in xrange(1, len(match_pair) + 1) :
        if [h + i, ref + i] in match_pair :
            match_pair.remove([h + i, ref + i])
        else :
            break

    fragment = 1

    if len(match_pair) != 0 :
        fragment += frag(match_pair)

    return fragment

def check_syn(w, ref):
    synsets = wn.synsets(w)
    for synset in synsets:
        for ref_w in ref:
            if ref_w in synset.lemma_names():
                return 1
    return 0

# DRY
def meteor_score(h, ref, stemmer, alpha):
    h2 = list(h)
    ref2 = list(ref)
    t = len(h)
    r = len(ref2)
    matched_pair = []
    # exact
    exact = 0.0
    for idx1, w in enumerate(h):
        if w in ref2:
            exact += 1
            matched_pair.append([idx1, ref2.index(w)])
            ref2.remove(w)
            h2.remove(w)
    # stem
    h_stem = [stemmer.stem(x) for x in h2]
    r_stem = [stemmer.stem(x) for x in ref2]

    stem = 0.0
    for idx1, w in enumerate(h_stem):
        if  w in r_stem:
            stem += 1
            matched_pair.append([idx1, r_stem.index(w)])
            del h2[idx1]
            del ref2[r_stem.index(w)]
            h_stem.remove(w)
            r_stem.remove(w)

    # synonymy
    #synonymy = sum(1 for w in h2 if check_syn(w, ref2) == 1)
    
    
    
    
    m = exact + stem #+synonymy

    precision = m/t

    # recall
    recall = m/r
    


    #fp.write("%d,%d,%d,%f,%f\n"%(m, t, r, precision, recall))
    #fp.write("%s\n"%(' '.join(ref)))
    #fp.write("%s\n"%(' '.join(h)))
    #sys.stderr.write("%d,%d,%d,%f,%f\n"%(m, t, r, precision, recall))
    if not (precision == 0.0 or recall == 0.0):
        fmean = precision*recall/(alpha*precision + (1-alpha)*recall)
    else:
        fmean = 0.0



    ''' 
    if not (precision == 0.0 or recall == 0.0) and t != 1:
        fragmentation = (frag(matched_pair) - 1) / float(t - 1)
        DF = gamma * pow(fragmentation, beta)
        final = fmean * (1 - DF)
    else :
        final = fmean
    '''
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
    opts = parser.parse_args()
 
    stemmer = PorterStemmer()
    
    # we create a generator and avoid loading all sentences into a list
    def sentences():
        with codecs.open(opts.input,encoding='utf-8') as f:
            for pair in f:
                #yield [sentence.strip().split() for sentence in pair.split(' ||| ')]
                yield [sentence.strip().lower().split() for sentence in pair.split(' ||| ')]
 
    # note: the -n option does not work in the original code
    for h1, h2, ref in islice(sentences(), opts.num_sentences):
        h1 = remove_punctuation(' '.join(h1)).split(' ')
        h2 = remove_punctuation(' '.join(h2)).split(' ')
        ref = remove_punctuation(' '.join(ref)).split(' ')
        #rset = set(ref)
        h1_score = meteor_score(h1, ref, stemmer, 0.7)
        h2_score = meteor_score(h2, ref, stemmer, 0.7)
        print(-1 if h1_score > h2_score else # \begin{cases}
                (0 if h1_score == h2_score
                    else 1)) # \end{cases}
 

# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
