#!/us:r/bin/env python
# -*- coding: utf-8 -*-
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet as wn
from features import *
import codecs,sys,string
import cPickle as pickle
import numpy as np
exclude = set(string.punctuation)
o_match = open('match.csv','w')
o_match_r = open('match_r.csv','w')
o_match_pos = open('match_pos.csv','w')
o_match_pos_r = open('match_pos_r.csv','w')
o_2gram = open('2gram.csv','w')
o_2gram_r = open('2gram_r.csv','w')
o_3gram = open('3gram.csv','w')
o_3gram_r = open('3gram_r.csv','w')
o_w2v = open('w2v.csv','w')
o_w2v_r = open('w2v_r.csv','w')


def  gen_bigram(source,n):
    target_len = len(source)-n
    output = []
    for i in range(target_len):
        output.append(source[i]+source[i+1])
    return output

def  gen_ngram(source,n):
    target_len = len(source)-n+1
    output = []
    for i in range(target_len):
        tt = ''
        for j in range(n):
            tt = tt+source[i+j]
        output.append(tt)
    return output



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
                return True
    return False

# meteor implemetation
def meteor_score7(h, ref, h_pos, ref_pos, stemmer, alpha, beta, gamma):

    t = len(h)
    r = len(ref)
    if t==0 or r == 0:
        return 0.0
    matched_pair = []
    matched_pair2 = []
    h_matched = np.zeros((t))
    ref_matched = np.zeros((r))
    
    ref_matched2 = np.zeros((r))
    
    exact = 0.0
    stem = 0.0
    synonym = 0.0
    
    # exact match
    for idx1, x in enumerate(h):
        for idx2, y in enumerate(ref):
            if x==y and (ref_matched[idx2]==0):
                exact += 1
                matched_pair.append([idx1, idx2])
                ref_matched[idx2] = 1
    
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
        fragmentation = (frag(matched_pair) - 1) / float(t - 1)
        DF = gamma * pow(fragmentation, beta)
        final = fmean * (1 - DF)
    else:
        final = fmean
    return final






def w2v_score7(h, ref, h_pos, ref_pos, stemmer, alpha, beta, gamma):

    h2 = []
    ref2 = []
    for idx, x in enumerate(h):
        h2.append(x+'_'+h_pos[idx])
    for idx, x in enumerate(ref):
        ref2.append(x+'_'+ref_pos[idx])

    t = len(h)
    r = len(ref)
    if t==0 or r == 0:
        return 0.0
    matched_pair = []
    h_matched = np.zeros((t))
    ref_matched = np.zeros((r))
    
    exact = 0.0
    
    # exact match
    for idx1, x in enumerate(h2):
        for idx2, y in enumerate(ref2):
            if x==y and (ref_matched[idx2]==0):
                exact += 1
                matched_pair.append([idx1, idx2])
                ref_matched[idx2] = 1

    for idx1, x in enumerate(h):
        max_sim = 0.0
        max_idx = sys.maxint
        for idx2, y in enumerate(ref):
            if (ref_matched[idx2]==0):
                try:
                    sim = wv.similarity(x,y, w2v_model)
                except:
                    sim = 0
                if sim > max_sim:
                    max_sim = sim
                    max_idx = idx2
        if max_sim>0.5:
            exact += 1
            matched_pair.append([idx1, max_idx])
            ref_matched[max_idx] = 1

    m = exact 

    precision = m/t
    recall = m/r
    
    # compute F-score
    if precision != 0.0 and recall != 0.0:
        fmean = precision*recall/(alpha*precision + (1-alpha)*recall)
    else:
        fmean = 0.0

    # compute fragments
    if not (precision == 0.0 or recall == 0.0) and t != 1:
        fragmentation = (frag(matched_pair) - 1) / float(t - 1)
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
    parser.add_argument('-b','--beta', type=float, default=2, help='beta of metoer')
    parser.add_argument('-g','--gamma', type=float, default=0.3, help='gamma of meteor')
    parser.add_argument('-a','--alpha', type=float, default=0.8, help='alpha of meteor')
    opts = parser.parse_args()
 
    stemmer = PorterStemmer()
    
    fp = open('sent3_pos.pkl','r') 
    sent_pos = pickle.load(fp)
    # we create a generator and avoid loading all sentences into a list
    def sentences():
        with codecs.open(opts.input,encoding='utf-8') as f:
            for pair in f:
                #yield [sentence.strip().split() for sentence in pair.split(' ||| ')]
                yield [sentence.strip().lower().split() for sentence in pair.split(' ||| ')]
    # note: the -n option does not work in the original code
    counter = 0
    for h1, h2, ref in islice(sentences(), opts.num_sentences):
        # remove [unctuation
        h1a = remove_punctuation(' '.join(h1)).split(' ')
        h2a = remove_punctuation(' '.join(h2)).split(' ')
        refa = remove_punctuation(' '.join(ref)).split(' ')
        if counter%100 == 0:
            sys.stderr.write(str(counter) + '\n')

        index = counter
        h1_sent_pos = sent_pos[index][0]
        h2_sent_pos = sent_pos[index][1]
        ref_sent_pos = sent_pos[index][2]
        
        h1 = [x[0] for x in h1_sent_pos]
        h1_pos = [x[1] for x in h1_sent_pos]
        h1_stem = [x[2] for x in h1_sent_pos]

        h2 = [x[0] for x in h2_sent_pos]
        h2_pos = [x[1] for x in h2_sent_pos]
        h2_stem = [x[2] for x in h2_sent_pos]
        
        r = [x[0] for x in ref_sent_pos]
        r_pos = [x[1] for x in ref_sent_pos]
        r_stem = [x[2] for x in ref_sent_pos]

        # 1. match
        h1_score = meteor_score7(h1_stem, r_stem, h1_pos, r_pos, stemmer, opts.alpha, opts.beta, opts.gamma)
        h2_score = meteor_score7(h2_stem, r_stem, h2_pos, r_pos, stemmer, opts.alpha, opts.beta, opts.gamma) 
        o_match.write('%s,%s\n'%(str(h1_score),str(h2_score)))

        # 2. match_reverse
        h1_score_temp = meteor_score7(r_stem, h1_stem, r_pos, h1_pos, stemmer, opts.alpha, opts.beta, opts.gamma)
        h2_score_temp = meteor_score7(r_stem, h2_stem, r_pos, h2_pos, stemmer, opts.alpha, opts.beta, opts.gamma)
        h1_score += h1_score_temp
        h2_score += h2_score_temp
        o_match_r.write('%s,%s\n'%(str(h1_score_temp),str(h2_score_temp)))

        
        h1_stem_pos = []
        h2_stem_pos = []
        r_stem_pos = []
        for idx, x in enumerate(h1_stem):
            h1_stem_pos.append(x+'_'+h1_pos[idx])
        for idx, x in enumerate(h2_stem):
            h2_stem_pos.append(x+'_'+h2_pos[idx])
        for idx, x in enumerate(r_stem):
            r_stem_pos.append(x+'_'+r_pos[idx])

        # 3. match_pos
        h1_score_temp = meteor_score7(h1_stem_pos, r_stem_pos, h1_pos, r_pos, stemmer, opts.alpha, opts.beta, opts.gamma)
        h2_score_temp = meteor_score7(h2_stem_pos, r_stem_pos, h2_pos, r_pos, stemmer, opts.alpha, opts.beta, opts.gamma) 
        h1_score += h1_score_temp
        h2_score += h2_score_temp
        o_match_pos.write('%s,%s\n'%(str(h1_score_temp),str(h2_score_temp)))

        # 4. match_pos_reverst
        h1_score_temp = meteor_score7(r_stem_pos, h1_stem_pos, r_pos, h1_pos, stemmer, opts.alpha, opts.beta, opts.gamma)
        h2_score_temp = meteor_score7(r_stem_pos, h2_stem_pos, r_pos, h2_pos, stemmer, opts.alpha, opts.beta, opts.gamma)
        h1_score += h1_score_temp
        h2_score += h2_score_temp
        o_match_pos_r.write('%s,%s\n'%(str(h1_score_temp),str(h2_score_temp)))

        # 5. w2v 
        h1_score_temp = w2v_score7(h1, r, h1_pos, r_pos, stemmer, opts.alpha, opts.beta, opts.gamma)
        h2_score_temp = w2v_score7(h2, r, h2_pos, r_pos, stemmer, opts.alpha, opts.beta, opts.gamma)
        h1_score += h1_score_temp
        h2_score += h2_score_temp
        o_w2v.write('%s,%s\n'%(str(h1_score_temp),str(h2_score_temp)))
        
        # 6. w2v reverse 
        h1_score_temp = w2v_score7(r, h1, r_pos, h1_pos, stemmer, opts.alpha, opts.beta, opts.gamma)
        h2_score_temp = w2v_score7(r, h2, r_pos, h2_pos, stemmer, opts.alpha, opts.beta, opts.gamma)
        h1_score += h1_score_temp
        h2_score += h2_score_temp
        o_w2v_r.write('%s,%s\n'%(str(h1_score_temp),str(h2_score_temp)))

        # 7. bigram match
        h_1 = gen_ngram(h1_stem,2)
        h_2 = gen_ngram(h2_stem,2)
        r_ = gen_ngram(r_stem,2)

        h1_score_temp = meteor_score7(h_1, r_, h1_pos, r_pos, stemmer, opts.alpha, opts.beta, opts.gamma)
        h2_score_temp = meteor_score7(h_2, r_, h2_pos, r_pos, stemmer, opts.alpha, opts.beta, opts.gamma)
        h1_score += h1_score_temp
        h2_score += h2_score_temp
        o_2gram.write('%s,%s\n'%(str(h1_score_temp),str(h2_score_temp)))

        # 8. bigram reverse
        h1_score_temp = meteor_score7(r_, h_1, r_pos, h1_pos, stemmer, opts.alpha, opts.beta, opts.gamma)
        h2_score_temp = meteor_score7(r_, h_2, r_pos, h2_pos, stemmer, opts.alpha, opts.beta, opts.gamma)
        h1_score += h1_score_temp
        h2_score += h2_score_temp
        o_2gram_r.write('%s,%s\n'%(str(h1_score_temp),str(h2_score_temp)))


        # 9. bigram match
        h_1 = gen_ngram(h1_stem,3)
        h_2 = gen_ngram(h2_stem,3)
        r_ = gen_ngram(r_stem,3)

        h1_score_temp = meteor_score7(h_1, r_, h1_pos, r_pos, stemmer, opts.alpha, opts.beta, opts.gamma)
        h2_score_temp = meteor_score7(h_2, r_, h2_pos, r_pos, stemmer, opts.alpha, opts.beta, opts.gamma)
        h1_score += h1_score_temp
        h2_score += h2_score_temp
        o_3gram.write('%s,%s\n'%(str(h1_score_temp),str(h2_score_temp)))

        # 10. bigram reverse
        h1_score_temp = meteor_score7(r_, h_1, r_pos, h1_pos, stemmer, opts.alpha, opts.beta, opts.gamma)
        h2_score_temp = meteor_score7(r_, h_2, r_pos, h2_pos, stemmer, opts.alpha, opts.beta, opts.gamma)
        h1_score += h1_score_temp
        h2_score += h2_score_temp
        o_3gram_r.write('%s,%s\n'%(str(h1_score_temp),str(h2_score_temp)))
        

        print(-1 if h1_score > h2_score else # \begin{cases}
                (0 if h1_score == h2_score
                    else 1)) # \end{cases}
        counter += 1
 

# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
