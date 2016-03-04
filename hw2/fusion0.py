#!/us:r/bin/env python
# -*- coding: utf-8 -*-
import codecs,sys,string
import cPickle as pickle
import numpy as np

def load_feat(fp):
    feat = []
    with fp as lines:
        for line in lines:
            f = [float(x) for x in line.strip().split(',')]
            feat.append(f)
    return np.asarray(feat)
feature_dir = 'feature/'
o_match = open(feature_dir + 'match.csv','r')
o_match_r = open(feature_dir + 'match_r.csv','r')
o_match_pos = open(feature_dir + 'match_pos.csv','r')
o_match_pos_r = open(feature_dir + 'match_pos_r.csv','r')
o_2gram = open(feature_dir + '2gram.csv','r')
o_2gram_r = open(feature_dir + '2gram_r.csv','r')
o_3gram = open(feature_dir + '3gram.csv','r')
o_3gram_r = open(feature_dir + '3gram_r.csv','r')
o_4gram = open(feature_dir + '4gram.csv','r')
o_4gram_r = open(feature_dir + '4gram_r.csv','r')
o_w2v = open(feature_dir + 'w2v.csv','r')
o_w2v_r = open(feature_dir + 'w2v_r.csv','r')


f_match =  load_feat(o_match)
f_match_r =  load_feat(o_match_r)
f_match_pos =  load_feat(o_match_pos)
f_match_pos_r =  load_feat(o_match_pos_r)
f_2gram =  load_feat(o_2gram)
f_2gram_r =  load_feat(o_2gram_r)
f_3gram =  load_feat(o_3gram)
f_3gram_r =  load_feat(o_3gram_r)
f_4gram =  load_feat(o_4gram)
f_4gram_r =  load_feat(o_4gram_r)
f_w2v =  load_feat(o_w2v)
f_w2v_r =  load_feat(o_w2v_r)

feat_set = [f_match,f_match_r,f_match_pos, f_match_pos_r, f_2gram, f_2gram_r, f_3gram, f_3gram_r, f_4gram, f_4gram_r, f_w2v, f_w2v_r]


feat_all = np.hstack((f_match,f_match_r,f_match_pos, f_match_pos_r, f_2gram, f_2gram_r, f_3gram, f_3gram_r, f_4gram, f_4gram_r, f_w2v, f_w2v_r))

size = feat_all.shape[0]
dim = feat_all.shape[1]
train_size = 10000

#print feat_all.shape
weight = np.asarray([1,-1,1,-1,0.5,-0.5,0,0,0.5,-0.5,0,0,0,0,0,0,0,0,0,0,2,-2,0,0])
for i in range(size):
    score =  np.dot(weight,feat_all[i])
    print (-1 if score > 0 else (0 if score == 0 else 1)) # \end{cases}
