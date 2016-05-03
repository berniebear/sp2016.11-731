function [grad_b_img, grad_W_img, dc, dh, grad_W_rnn, grad_W_emb, grad_emb_indices, attnGrad, grad_srcHidVecs_total] = rnnLayerBackprop_enc(b_img, W_img, W_rnn, rnnStates, initState, ...
  top_grads, dc, dh, input, masks, im_masks, params, rnnFlags, attnInfos, trainData, model, obj_idx)
% Running Multi-layer RNN for one time step.
% Input:
%   W_rnn: recurrent connections of multiple layers, e.g., W_rnn{ll}.
%   initState: begin state
%   input: indices for the current batch
%   isDecoder: 1 -- on the decoder side
%
% Thang Luong @ 2015, <lmthang@stanford.edu>

T = size(input, 2);
grad_W_img = [];
grad_b_img = [];
T =  T + 1;

% emb
totalWordCount = params.curBatchSize * T;
allEmbGrads = zeroMatrix([params.lstmSize, totalWordCount], params.isGPU, params.dataType);
allEmbIndices = zeros(totalWordCount, 1);
wordCount = 0;

% attention
if params.attnFunc && rnnFlags.decode
  grad_srcHidVecs_total = zeroMatrix([params.lstmSize, params.curBatchSize, params.numSrcHidVecs], params.isGPU, params.dataType); % +1
else
  grad_srcHidVecs_total = [];
  attnGrad = [];
end

% masks
[maskInfos] = prepareMask(masks+im_masks);
[word_maskInfos] = prepareMask(masks);
[im_maskInfos] = prepareMask(im_masks); % no update on word embedding, so OK

for tt=T:-1:1 % time
  % attention
  cur_top_grad = top_grads{tt};
  if tt>1
    prevState = rnnStates{tt-1};
  else
    prevState = initState;
  end

  %% multi-layer RNN backprop
  [dc, dh, d_emb, d_W_rnn, d_feed_input] = rnnStepLayerBackprop(W_rnn, prevState, rnnStates{tt}, cur_top_grad, dc, dh, maskInfos{tt}, ... 
    params, rnnFlags.feedInput);

  % recurrent grad
  for ll=params.numLayers:-1:1 % layer
    if tt==T
      grad_W_rnn{ll} = d_W_rnn{ll};
    else
      grad_W_rnn{ll} = grad_W_rnn{ll} + d_W_rnn{ll};
    end
  end

  % softmax feedinput, bottom grad send back to top grad in the previous
  % time step
  if rnnFlags.feedInput && tt>1
    top_grads{tt-1} = top_grads{tt-1} + d_feed_input;
  end

  
    img_pooling = repmat(im_masks(:, tt)',size(d_emb,1),1);
    d_emp_img = d_emb.*img_pooling;
  
    [~, grad_W_img, grad_b_img] = linearLayerBackpropBias(b_img, W_img, d_emp_img, trainData.srcFc7(:,:,obj_idx)); 
    %[~, grad_W_img, grad_b_img] = linearLayerBackpropBias(b_img, W_img, d_emb(1:params.lstmSize, :),trainData.srcFc7); 
    
    
    if (tt>1)
        unmaskedIds = word_maskInfos{tt}.unmaskedIds;
        numWords = length(unmaskedIds);
        allEmbIndices(wordCount+1:wordCount+numWords) = input(unmaskedIds, tt-1);
        allEmbGrads(:, wordCount+1:wordCount+numWords) = d_emb(1:params.lstmSize, unmaskedIds);
        wordCount = wordCount + numWords;
    end
  
end % end for time

allEmbGrads(:, wordCount+1:end) = [];
allEmbIndices(wordCount+1:end) = [];
[grad_W_emb, grad_emb_indices] = aggregateMatrix(allEmbGrads, allEmbIndices, params.isGPU, params.dataType);