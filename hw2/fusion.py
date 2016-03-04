#!/us:r/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function


import codecs,sys,string
import cPickle as pickle
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils


def load_feat(fp):
    feat = []
    with fp as lines:
        for line in lines:
            f = [float(x) for x in line.strip().split(',')]
            feat.append(f)
    return np.asarray(feat)

label_file = 'data/train2.gold'
label = [] 
with open(label_file) as lines:
    for line in lines:
        label.append(int(line.strip()))
label = np.asarray(label)

feature_dir = 'feature2/'
o_match = open(feature_dir + 'match.csv','r')
o_match_r = open(feature_dir + 'match_r.csv','r')
o_match_pos = open(feature_dir + 'match_pos.csv','r')
o_match_pos_r = open(feature_dir + 'match_pos_r.csv','r')
o_2gram = open(feature_dir + '2gram.csv','r')
o_2gram_r = open(feature_dir + '2gram_r.csv','r')
o_w2v = open(feature_dir + 'w2v.csv','r')
o_w2v_r = open(feature_dir + 'w2v_r.csv','r')


f_match =  load_feat(o_match)
f_match_r =  load_feat(o_match_r)
f_match_pos =  load_feat(o_match_pos)
f_match_pos_r =  load_feat(o_match_pos_r)
f_2gram =  load_feat(o_2gram)
f_2gram_r =  load_feat(o_2gram_r)
f_w2v =  load_feat(o_w2v)
f_w2v_r =  load_feat(o_w2v_r)
#feat_set = [f_match,f_match_r,f_match_pos, f_match_pos_r, f_2gram, f_2gram_r, f_3gram, f_3gram_r, f_4gram, f_4gram_r, f_w2v, f_w2v_r]
#feat_all = np.hstack((f_match,f_match_r,f_match_pos, f_match_pos_r, f_2gram, f_2gram_r, f_3gram, f_3gram_r, f_4gram, f_4gram_r, f_w2v, f_w2v_r))
#feat_all = np.hstack((f_match,f_match_r,f_match_pos, f_match_pos_r, f_2gram, f_2gram_r, f_w2v, f_w2v_r))
feat_all = np.hstack((f_match,f_match_r,f_match_pos, f_2gram, f_w2v, np.log(f_match+0.0001)))

size = feat_all.shape[0]
dim = feat_all.shape[1]
train_size = 26208


training_data = feat_all[0:train_size]
testing_data=feat_all[train_size:26208]

training_label = label[0:train_size]
testing_label = label[train_size:26208]
#print label.shape

batch_size = 100
nb_classes = 2
nb_epoch = 20


#X_train = X_train.reshape(60000, 784)
X_all = feat_all
X_train = training_data
#X_test = X_test.reshape(10000, 784)
X_test = testing_data
X_all = X_all.astype('float32')
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(training_label, nb_classes)
Y_test = np_utils.to_categorical(testing_label, nb_classes)
#print (len(X_test), len(Y_test))

model = Sequential()
model.add(Dense(100, input_shape=(dim,), init ='lecun_uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.3))
model.add(Dense(20, input_shape=(dim,), init ='lecun_uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(10, input_shape=(dim,), init ='lecun_uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.1))
model.add(Dense(10, input_shape=(dim,), init ='lecun_uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.1))
#model.add(Dense(nb_classes, input_shape=(dim,)))
model.add(Dense(nb_classes, init='lecun_uniform'))
model.add(Activation('softmax'))

rms = RMSprop()
#sgd = SGD(lr=0.1, momentum=0.0, decay=0.00499, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=rms)

model.fit(X_train, Y_train,
          batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=2,
          validation_data=(X_test, Y_test))
#score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
#print('Test score:', score[0])
#print('Test accuracy:', score[1])


temp = model.predict(X_all, batch_size=10, verbose=0)


fp = open('a','w')
for i in range(len(label)):
    h1_score = temp[i][0]
    h2_score = temp[i][1]
    if h1_score > h2_score:
        fp.write('-1\n')
    elif h1_score == h2_score:
        fp.write('0\n')
    else:
        fp.write('1\n')

'''
#print feat_all.shape
weight = np.asarray([1,-1,1,-1,0.5,-0.5,0,0,0.5,-0.5,0,0,0,0,0,0,0,0,0,0,2,-2,0,0])
for i in range(size):
    score =  np.dot(weight,feat_all[i])
    print (-1 if score > 0 else (0 if score == 0 else 1)) # \end{cases}
'''
