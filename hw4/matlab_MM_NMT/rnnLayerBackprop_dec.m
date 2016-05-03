function [grad_b_img, grad_W_img, dc, dh, grad_W_rnn, grad_W_emb, grad_emb_indices, attnGrad, grad_srcHidVecs_total] = rnnLayerBackprop_dec(~, ~, W_rnn, rnnStates, initState, ...
  top_grads, dc, dh, input, masks, params, rnnFlags, attnInfos, trainData, model)
% Running Multi-layer RNN for one time step with multiple modalities
% Input:
%   W_rnn: recurrent connections of multiple layers, e.g., W_rnn{ll}.
%   initState: begin state
%   input: indices for the current batch
%   isDecoder: 1 -- on the decoder side
% Po-Yao Huang
% With contribution from
% Thang Luong @ 2015, <lmthang@stanford.edu>

T = size(input, 2);
grad_W_img = [];
grad_b_img = [];

% emb
totalWordCount = params.curBatchSize * T;
allEmbGrads = zeroMatrix([params.lstmSize, totalWordCount], params.isGPU, params.dataType);
allEmbIndices = zeros(totalWordCount, 1);
wordCount = 0;

% attention
%if params.attnFunc 
  %grad_srcHidVecs_total = zeroMatrix([params.lstmSize, params.curBatchSize, params.numSrcHidVecs*3], params.isGPU, params.dataType); % +1
  grad_srcHidVecs_total = zeroMatrix([params.lstmSize, params.curBatchSize, params.numSrcHidVecs], params.isGPU, params.dataType); % +1
%end

% masks
[maskInfos] = prepareMask(masks);

for tt=T:-1:1 % time
  % attention
  if rnnFlags.attn 
    % attention: softmax_h -> h_t
    h2sInfo = attnInfos{tt};
    [cur_top_grad, attnStepGrad, grad_srcHidVecs] = attnLayerBackprop(model, top_grads{tt}, trainData, h2sInfo, params, maskInfos{tt});
    fields = fieldnames(attnStepGrad);
    for ii=1:length(fields)
      field = fields{ii};
      if tt==T
        attnGrad.(field) = attnStepGrad.(field);
      else
        attnGrad.(field) = attnGrad.(field) + attnStepGrad.(field);
      end
    end

    % srcHidVecs
   grad_srcHidVecs_total = grad_srcHidVecs_total + grad_srcHidVecs;

  else % non-attention
    cur_top_grad = top_grads{tt};
  end

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

  
    % emb grad
    unmaskedIds = maskInfos{tt}.unmaskedIds;
    numWords = length(unmaskedIds);
    allEmbIndices(wordCount+1:wordCount+numWords) = input(unmaskedIds, tt);
    allEmbGrads(:, wordCount+1:wordCount+numWords) = d_emb(1:params.lstmSize, unmaskedIds);
    wordCount = wordCount + numWords;
  
end % end for time

allEmbGrads(:, wordCount+1:end) = [];
allEmbIndices(wordCount+1:end) = [];
[grad_W_emb, grad_emb_indices] = aggregateMatrix(allEmbGrads, allEmbIndices, params.isGPU, params.dataType);