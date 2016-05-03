function [costs, grad] = lstmCostGrad(model, trainData, params, isTest)
%%%
 %
% Compute cost/grad for LSTM. 
% When params.predictPos>0, returns costs.pos and costs.word
% If isTest==1, this method only computes cost (for testing purposes).
% Po-Yao Huang, Parallel lstm enc/dec for multiple modality 
% Thang Luong @ 2014, 2015, <lmthang@stanford.edu>
%
%%%

  %%%%%%%%%%%%
  %%% INIT %%%
  %%%%%%%%%%%%
  curBatchSize = size(trainData.tgtInput, 1);
  if params.isBi
    srcMaxLen = trainData.srcMaxLen; 
  else % monolingual
    srcMaxLen = 1;
  end
  
  params.curBatchSize = curBatchSize;
  params.srcMaxLen = srcMaxLen;
  params.srcMaxLen_img = srcMaxLen; % for image

  
  
  
  %[params] = setAttnParams(params);
  [params] = setAttnParams_img(params);
  [grad, params] = initGrad(model, params);
  zeroBatch = zeroMatrix([params.lstmSize, params.curBatchSize], params.isGPU, params.dataType);
  
  % initState
  [zeroState] = createZeroState(params);
  
  % init costs
  costs = initCosts();
  
  %%%%%%%%%%%%%%%%%%%%
  %%% FORWARD PASS %%%
  %%%%%%%%%%%%%%%%%%%%
  %% encoder
  lastEncState = zeroState;
  if params.isBi
    encRnnFlags = struct('decode', 0, 'test', isTest, 'attn', params.attnFunc, 'feedInput', 0);
    
    % init
    %if encRnnFlags.attn %&& rnnFlags.decode == 0 % encoder
        %trainData.srcHidVecsOrig = zeroMatrix([params.lstmSize, params.curBatchSize, params.numSrcHidVecs*3], params.isGPU, params.dataType);
        trainData.srcHidVecsOrig = zeroMatrix([params.lstmSize, params.curBatchSize, params.numSrcHidVecs], params.isGPU, params.dataType);
    %end
      
      
    
    [encStates1, trainData, ~] = rnnLayerForward_enc(model.b_img, model.W_img, model.W_src, model.W_emb_src, zeroState, trainData.srcInput, trainData.srcMask, trainData.imgMask, ...
      params, encRnnFlags, trainData, model, 1, []);
    lastEncState1 = encStates1{end};
    
    %[encStates2, trainData, ~] = rnnLayerForward_enc(model.b_img, model.W_img, model.W_src, model.W_emb_src, zeroState, trainData.srcInput, trainData.srcMask, trainData.imgMask, ...
    %  params, encRnnFlags, trainData, model, 2, encStates1);
    %lastEncState2 = encStates2{end};
    
    %[encStates3, trainData, ~] = rnnLayerForward_enc(model.b_img, model.W_img, model.W_src, model.W_emb_src, zeroState, trainData.srcInput, trainData.srcMask, trainData.imgMask, ...
    %  params, encRnnFlags, trainData, model, 3, encStates1);
    %lastEncState3 = encStates3{end};
    
    % merge lastEncState
    
    %lastEncState = encoder_fusion(lastEncState1, lastEncState2, lastEncState3);
    lastEncState = lastEncState1;
    
    % feed input
    if params.feedInput
      lastEncState{end}.softmax_h = zeroBatch;
    end
  end
  
  %% decoder
  decRnnFlags = struct('decode', 1, 'test', isTest, 'attn', params.attnFunc, 'feedInput', params.feedInput);
  [decStates, ~, attnInfos] = rnnLayerForward_dec([], [], model.W_tgt, model.W_emb_tgt, lastEncState, trainData.tgtInput, trainData.tgtMask, ...
    params, decRnnFlags, trainData, model);
  
  %% softmax
  [costs.total, grad.W_soft, dec_top_grads] = softmaxCostGrad(decStates, model.W_soft, trainData.tgtOutput, trainData.tgtMask, ...
    params, isTest);
  costs.word = costs.total;
  
  if isTest==1 % don't compute grad % testing phase
    return;
  end
  
  %%%%%%%%%%%%%%%%%%%%%
  %%% BACKWARD PASS %%%
  %%%%%%%%%%%%%%%%%%%%%
  % h_t and c_t gradients accumulate over time per layer
  dh = cell(1, 1);
  dc = cell(1, 1); 
  for ll=params.numLayers:-1:1 % layer
    dh{ll} = zeroBatch;
    dc{ll} = zeroBatch;
  end
  
  %% decoder
  [~,~, dc, dh, grad.W_tgt, grad.W_emb_tgt, grad.indices_tgt, attnGrad, grad.srcHidVecs] = rnnLayerBackprop_dec([], [], model.W_tgt, ...
    decStates, lastEncState, dec_top_grads, dc, dh, trainData.tgtInput, trainData.tgtMask, params, decRnnFlags, ...
    attnInfos, trainData, model);
    [grad] = copyStruct(attnGrad, grad);
    
  %dec_top_grads2 = decoder_fusion(dec_top_grads);
  %% encoder
    
    enc_top_grads = cell(srcMaxLen - 1 + 1, 1); % remove eos, add img
    for tt=1:params.numSrcHidVecs% attention +1
      enc_top_grads{tt} = grad.srcHidVecs(:,:,tt);
    end
    
    [grad.b_img1, grad.W_img1, ~, ~, grad.W_src1, grad.W_emb_src1, grad.indices_src1, ~, ~] = rnnLayerBackprop_enc(model.b_img, model.W_img, model.W_src, encStates1, zeroState, ...
    enc_top_grads, dc, dh, trainData.srcInput, trainData.srcMask, trainData.imgMask,params, encRnnFlags, attnInfos, trainData, model, 1);
    

    %enc_top_grads = cell(srcMaxLen - 1 + 1, 1); % remove eos, add img
    %for tt=1:params.numSrcHidVecs% attention +1
    %  enc_top_grads{tt} = grad.srcHidVecs(:,:,params.numSrcHidVecs+tt);
    %end
    
    %[grad.b_img2, grad.W_img2, ~, ~, grad.W_src2, grad.W_emb_src2, grad.indices_src2, ~, ~] = rnnLayerBackprop_enc(model.b_img, model.W_img, model.W_src, encStates2, zeroState, ...
    %enc_top_grads, dc, dh, trainData.srcInput, trainData.srcMask, trainData.imgMask,params, encRnnFlags, attnInfos, trainData, model, 2);
                        

    %enc_top_grads = cell(srcMaxLen - 1 + 1, 1); % remove eos, add img
    %for tt=1:params.numSrcHidVecs% attention +1
    %  enc_top_grads{tt} = grad.srcHidVecs(:,:,2*params.numSrcHidVecs+tt);
    %end
    
    %[grad.b_img3, grad.W_img3, ~, ~, grad.W_src3, grad.W_emb_src3, grad.indices_src3, ~, ~] = rnnLayerBackprop_enc(model.b_img, model.W_img, model.W_src, encStates3, zeroState, ...
    %enc_top_grads, dc, dh, trainData.srcInput, trainData.srcMask, trainData.imgMask,params, encRnnFlags, attnInfos, trainData, model, 3);
    

    %grad.b_img = (grad.b_img1 + grad.b_img2 + grad.b_img3);
    %grad.W_img = (grad.W_img1 + grad.W_img2 + grad.W_img3);
    %grad.W_src{1} = (grad.W_src1{1} + grad.W_src2{1} + grad.W_src3{1});
    %grad.W_emb_src = (grad.W_emb_src1 + grad.W_emb_src2 + grad.W_emb_src3);
    %grad.indices_src = (grad.indices_src1 + grad.indices_src2 + grad.indices_src3)/3; % should be same
    
    grad.b_img = grad.b_img1;
    grad.W_img = grad.W_img1;
    grad.W_src{1} = grad.W_src1{1};
    grad.W_emb_src = grad.W_emb_src1;
    grad.indices_src = grad.indices_src1; % should be same

    
  % remove unused variables
  grad = rmfield(grad, 'srcHidVecs');
end

function [grad, params] = initGrad(model, params)
  %% grad
  for ii=1:length(params.varsDenseUpdate)
    field = params.varsDenseUpdate{ii};
    if iscell(model.(field))
      for jj=1:length(model.(field)) % cell, like W_src, W_tgt
        grad.(field){jj} = zeroMatrix(size(model.(field){jj}), params.isGPU, params.dataType);
      end
    else
      grad.(field) = zeroMatrix(size(model.(field)), params.isGPU, params.dataType);
    end
  end
  
  %% backprop to src hidden states for attention and positional models
  if params.attnFunc>0
    %grad.srcHidVecs = zeroMatrix([params.lstmSize, params.curBatchSize, params.numSrcHidVecs*3], params.isGPU, params.dataType);
    grad.srcHidVecs = zeroMatrix([params.lstmSize, params.curBatchSize, params.numSrcHidVecs], params.isGPU, params.dataType);
  end
end