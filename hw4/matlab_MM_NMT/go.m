%addpath(genpath(sprintf('%s/..', pwd)));
%trainLSTM('../data/wmt-lower/train', '../data/wmt-lower/val', '../data/wmt-lower/val', 'en', 'de', '../data/wmt-lower/train.en.vocab', '../data/wmt-lower/train.de.vocab', '../output/wmt-test_h2_7', 'lstmSize', 256, 'attnFunc', 1, 'attnOpt', 2, 'isReverse', 1, 'feedInput', 1, 'dropout', 0.7, 'isResume', 0, 'seed', 3284  )
%testLSTM('../output/wmt-test_h2_7/modelRecent.mat', 2, 10, 1, '../output/wmt-test_h2_7/translations.txt')

trainLSTM('../data2/train1', '../data2/dev1', '../data2/dev1', 'src', 'tgt', '../data2/train1.src.vocab.3436', '../data2/train1.tgt.vocab.3121', '../output/test5', 'attnFunc', 2,'attnOpt', 2, 'isResume', 0)
testLSTM('../output/test5/modelRecent.mat', 2, 10, 1, '../output/test5/translations2.txt', 'testPrefix', '../data2/test1')
testLSTM('../output/basic/modelRecent.mat', 2, 10, 1, '../output/basic/translations.txt', 'testPrefix', '../output/id.1000/valid.100')
