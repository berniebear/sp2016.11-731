function [lstmStates, attnData, attnInfos] = rnnLayerForward_enc(b_img, W_img, W_rnn, W_emb, prevState, input, masks, im_masks,params, rnnFlags, attnData, model, obj_idx, parallel_lstm)
% Running Multi-layer RNN for one time step.
% Input:
%   W_rnn: recurrent connections of multiple layers, e.g., W_rnn{ll}.
%   prevState: previous hidden state, e.g., for LSTM, prevState.c{ll}, prevState.h{ll}.
%   input: indices for the current batch
%   isTest: 1 -- don't store intermediate results in each state
%   isAttn: for attention, require attnData to be non-empty, has
%     attnData.srcHidVecsOrig and attnData.srcLens.
%   isDecoder: 0 -- encoder, 1 -- decoder
% Output:
%   nextState
%   attnData : training data (to be attn)
% Thang Luong @ 2015, <lmthang@stanford.edu>


T = size(input, 2);
T = T + 1; % append image

lstmStates = cell(T, 1);

% attention
attnInfos = cell(T, 1);
% move out to lstmCostGrad
%if rnnFlags.attn %&& rnnFlags.decode == 0 % encoder
%  assert(T <= params.numSrcHidVecs);
%  attnData.srcHidVecsOrig = zeroMatrix([params.lstmSize, params.curBatchSize, params.numSrcHidVecs], params.isGPU, params.dataType);
%end


for tt=1:T % time
  % local monotonic alignment
  if params.attnLocalMono
    if rnnFlags.decode == 0 % encoder
        if tt < T
            attnData.tgtPos = tt;  
        end
    else
        attnData.tgtPos = tt;
    end
  end
  
  %masks(:, tt),
  %im_masks
  W_img_emb = linearLayerForwardBias(b_img, W_img, attnData.srcFc7(:,:,obj_idx));
  %W_img_emb = linearLayerForwardBias(b_img, W_img, attnData.srcFc7());
  img_pooling = repmat(im_masks(:, tt)',size(W_img_emb,1),1);
  W_img_emb2 = W_img_emb.*img_pooling;
  if tt > 1
    W_emb_temp = W_emb(:, input(:, tt-1)); % the orginal mask is shift right by 1, mask k for word k-1. only apply when retireve input
  else
    W_emb_temp = 0*W_img_emb;
  end
  word_pooling = repmat(masks(:, tt)',size(W_emb_temp,1),1);
  W_emb2 = W_emb_temp.*word_pooling;
  emb = W_emb2 + W_img_emb2;
  %[prevState, attnInfos{tt}] = rnnStepLayerForward(W_rnn, W_img_emb, prevState, ones(size(masks,1),1), params, rnnFlags, attnData, model);
  
  
  if isempty(parallel_lstm) == 0 && rnnFlags.test == 0
    dropoutmask = parallel_lstm{tt}{1}.dropoutMask;
    [prevState, attnInfos{tt}] = rnnStepLayerForward(W_rnn, emb, prevState, masks(:, tt)+im_masks(:, tt), params, rnnFlags, attnData, model, dropoutmask); % 
  else
    [prevState, attnInfos{tt}] = rnnStepLayerForward(W_rnn, emb, prevState, masks(:, tt)+im_masks(:, tt), params, rnnFlags, attnData, model, []); % 
  end
  
  
  
    
  if rnnFlags.attn %&& rnnFlags.decode == 0 % encoder
      attnData.srcHidVecsOrig(:, :, tt + (obj_idx-1)*params.numSrcHidVecs) = prevState{end}.h_t;
  end
  % store all states
  lstmStates{tt} = prevState;
end