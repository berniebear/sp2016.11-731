#!/us:r/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
import word2vec.test_word2vec as wv

model_name = 'word2vec/' + 'glove.6B.300d.txt'
w2v_dic, w2v_model = wv.load_word2vec(model_name)

#def w2v_score(h,ref):
#    wv.similarity('apple', 'banana', w2v_model)


'''
def pos_score(h, ref):
    #h_pos = [ h[i][0] for i in range(len(h)) ]
    #ref_pos = [ ref[i][0]  for i in range(len(ref)) ]    
        
    for i in range(len(h)):
        h[i] = list(h[i])
        h[i][1] =  h[i][1][0:2]
    for i in range(len(ref)):
        ref[i] = list(ref[i])
        ref[i][1] = ref[i][1][0:2]
    
    h_pos = [ u'_'.join(h[i]) for i in range(len(h)) ]
    ref_pos = [ u'_'.join(ref[i]) for i in range(len(ref)) ]
    return sum(1 for w in h_pos if w in ref_pos)
'''
