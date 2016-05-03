function [params] = parameter_parser(trainPrefix,validPrefix,testPrefix,srcLang,tgtLang,srcVocabFile,tgtVocabFile,outDir,varargin)
  %% Argument Parser

  p = inputParser;
  % required
  addRequired(p,'trainPrefix',@ischar);
  addRequired(p,'validPrefix',@ischar);
  addRequired(p,'testPrefix',@ischar);
  addRequired(p,'srcLang',@ischar);
  addRequired(p,'tgtLang',@ischar);
  addRequired(p,'srcVocabFile',@ischar);
  addRequired(p,'tgtVocabFile',@ischar);
  addRequired(p,'outDir',@ischar);
  
  % important hyperparameters
  addOptional(p,'numLayers', 1, @isnumeric); % stacking architecture
  addOptional(p,'lstmSize', 256, @isnumeric); % number of cells, also embedding size
  addOptional(p,'initRange', 0.1, @isnumeric);
  addOptional(p,'learningRate', 1.0, @isnumeric);
  addOptional(p,'maxGradNorm', 5.0, @isnumeric); % to scale large grads
  addOptional(p,'batchSize', 64, @isnumeric);
  addOptional(p,'numEpoches', 16, @isnumeric); % num epoches
  addOptional(p,'epochFraction', 1.0, @isnumeric);
  addOptional(p,'finetuneEpoch', 8, @isnumeric); % epoch > finetuneEpoch, start halving learning rate every epochFraction of an epoch, e.g., every 0.5 epoch
  addOptional(p,'finetuneRate', 0.5, @isnumeric); % multiply learning rate by this factor each time we finetune
  % hack
  addOptional(p,'epochIter', 0, @isnumeric); % if our train file is too large and we want to know the number of iterations in a epoch beforehand
  
  % advanced features
  addOptional(p,'dropout', 0.8, @isnumeric); % keep prob for dropout, i.e., 1 no dropout, <1: dropout
  addOptional(p,'isReverse', 0, @isnumeric); % 1: reseverse source sentence. We expect file $prefix.$srcLang.reversed (instead of $prefix.$srcLang)
  addOptional(p,'feedInput', 0, @isnumeric); % 1: feed the softmax vector to the next timestep input
  addOptional(p,'lstmOpt', 0, @isnumeric); % lstmOpt=0: basic model (I have always been using this!), 1: no tanh for c_t.
    
  % training
  addOptional(p,'isBi', 1, @isnumeric); % isBi=0: mono model, isBi=1: bi (encoder-decoder) model.
  addOptional(p,'isClip', 1, @isnumeric); % isClip=1: clip forward 50, clip backward 1000.
  addOptional(p,'maxSentLen', 51, @isnumeric); % limit sentence length on each side during training. Default: 50 + 1 (eos).
  addOptional(p,'logFreq', 10, @isnumeric); % how frequent (number of batches) we want to log stuffs
  addOptional(p,'isResume', 1, @isnumeric); % isResume=1: check if a model file exists, continue training from there.
  addOptional(p,'sortBatch', 1, @isnumeric); % 1: each time we read in 100 batches, we sort sentences by length.
  addOptional(p,'shuffle', 1, @isnumeric); % 1: shuffle training batches
  addOptional(p,'loadModel', '', @ischar); % To start training from
  addOptional(p,'saveHDF', 0, @isnumeric); % 1: to save in HDF5 format
  
  % decoding
  addOptional(p,'decode', 1, @isnumeric); % 1: decode during training
  addOptional(p,'minLenRatio', 0.5, @isnumeric);
  addOptional(p,'maxLenRatio', 1.5, @isnumeric);
  
  % debugging
  addOptional(p,'isGradCheck', 0, @isnumeric); % set 1 to check the gradient, no need input arguments as toy data is automatically generated.
  addOptional(p,'dataType', 'single', @ischar); % Note: use double precision for grad check
  addOptional(p,'isProfile', 0, @isnumeric);
  addOptional(p,'debug', 0, @isnumeric); % 0: no debug, 1: debug
  addOptional(p,'assert', 0, @isnumeric); % 0: no sanity check, 1: yes
  addOptional(p,'seed', 0, @isnumeric); % 0: seed based on current clock time, else use the specified seed
  
  %% attention! %%
  addOptional(p,'align', 0, @isnumeric);
  addOptional(p,'attnFunc', 0, @isnumeric);
  addOptional(p,'attnOpt', 0, @isnumeric);
  addOptional(p,'posWin', 10, @isnumeric); % relative window, used for attnFunc~=1
  
  %% system options
  addOptional(p,'onlyCPU', 0, @isnumeric); % 1: avoid using GPUs
  addOptional(p,'gpuDevice', 1, @isnumeric); % choose the gpuDevice to use: 0 -- no GPU.

  p.KeepUnmatched = true; % store  unmatched terms
  parse(p,trainPrefix,validPrefix,testPrefix,srcLang,tgtLang,srcVocabFile,tgtVocabFile,outDir,varargin{:})
  params = p.Results;
    %% Setup params
  params.chunkSize = params.batchSize*100;
  params.baseIndex = 0; %  the minimum value in all sequences of integers (often 0). Required to convert them to 1-indexed for Matlab.
  % clip
  params.clipForward = 50; % clip c_t, h_t
  params.clipBackward = 1000; % clip dc, dh
  
  % act functions for gate
  params.nonlinear_gate_f = @sigmoid;
  params.nonlinear_gate_f_prime = @sigmoidPrime;
  
  % act functions for others
  params.nonlinear_f = @tanh;
  params.nonlinear_f_prime = @tanhPrime;
 
  % decode params
  params.beamSize = 12;
  params.stackSize = 100;
  params.unkPenalty = 0;
  params.forceDecoder = 0;
  
  % rand seed
  if params.isGradCheck || params.isProfile || params.seed
    s = RandStream('mt19937ar','Seed',params.seed); % assign specific seed
  else
    s = RandStream('mt19937ar','Seed','shuffle'); % seed baed on current time
  end
  RandStream.setGlobalStream(s);
  
  % check GPUs
  params.isGPU = 0;
  if params.gpuDevice
    n = gpuDeviceCount;  
    if n>0 % GPU exists
      fprintf(2, '# %d GPUs exist. So, we will use GPUs.\n', n);
      params.isGPU = 1;
      gpuDevice(params.gpuDevice)
    else
      params.dataType = 'double';
    end
  else
    params.dataType = 'double';
  end
  
  % grad check
  if params.isGradCheck
    params.lstmSize = 2;
    params.batchSize = 10;
    params.batchId = 1;
    params.maxSentLen = 7;
    params.posWin = 1;
  end
  
    %% attention
  params.align = (params.attnFunc>0); % for the decoder 
  params.attnGlobal = 0;
  params.attnLocalMono = 0;
  params.attnLocalPred = 0;
  if params.attnFunc>0
    params.attnGlobal = (params.attnFunc==1);
  end
end