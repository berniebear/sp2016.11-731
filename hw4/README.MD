Group: sshiang, fred, bernie 
(We are all together through this semester... However, it's a pity that we are too busy for the finals to specify being together this time....)

Method: We use a simplified version of our multi-modality neural machine translation (MM-NMT) with global attention on parallel thread LSTM.
For this specific chinese-english translation, we load other modalities (images) as all zero and mask them out (a head-insertion version), so it will be transparent as pure text result. Adopt from reference [1], the attention function we use is the generalized  dot product  h_src*W_a * h_t. Also, we modify the basic lstm unit and rnn structure with extended masking and dropout functionality for MM. TrainLSTM prepares data, the main encode/decode and realted forward/backward functionalities are in lstmCostGrad. The torch version is a simple implementation of encoder/decoder with global attention.

All training/testing data are lower-cased and transferom to discrete (one hot) under data_chen(ch-en). Word embeddings are randomly initialized withni +-0.1. Enter matlab under code/ and run the go.m script would do the whole training/decoding process. We also write a torch version, which will upload by Fred. Run the doall.lua script for training and testing. The final result is the ensemble of the best models with different parameters. Other example outputs are under the same folder.

[1] Effective Approaches to Attention-based Neural Machine Translation, Luong, Minh-Thang  and  Pham, Hieu  and  Manning, Christopher D., Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processings (EMNLP), 2015
