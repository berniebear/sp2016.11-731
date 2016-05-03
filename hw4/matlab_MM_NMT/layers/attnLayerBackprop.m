%%%
%
% Attentional Layer Backprop from softmax hidden state to lstm hidden state.
% Po-Yao Huang, add parallel lstm support for multiple modality in
% contextVecs
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%% 
function [grad_ht, attnGrad, grad_srcHidVecs] = attnLayerBackprop(model, grad_softmax_h, trainData, h2sInfo, params, curMask)
  % softmax_h -> h_t
  % f'(outVec).*outGrad
  tmpResult = params.nonlinear_f_prime(h2sInfo.softmax_h).*grad_softmax_h;  
  
  % grad_W
  attnGrad.W_h = tmpResult*h2sInfo.input';
  
  % inGrad
  grad_input = model.W_h'*tmpResult;
  
  
  % global
  srcHidVecs = trainData.srcHidVecsOrig;
  % grad_contextVecs -> grad_srcHidVecs, grad_alignWeights
  grad_contextVecs = grad_input(1:params.lstmSize, :);
    % change from grad_contextVecs lstmSize*batchSize -> lstmSize*batchSize*1
  unmaskedIds = curMask.unmaskedIds;
  grad_contextVecs = permute(grad_contextVecs, [1, 2, 3]); 
  
  %% grad_srcHidVecs
  grad_srcHidVecs = zeroMatrix(size(srcHidVecs), params.isGPU, params.dataType);
  grad_srcHidVecs(:, unmaskedIds, :) = bsxfun(@times, grad_contextVecs(:, unmaskedIds), permute(h2sInfo.alignWeights(:, unmaskedIds), [3, 2, 1])); % change alignWeights -> 1 * batchSize * numPositions
  
  %% grad_alignWeights
  grad_alignWeights = zeroMatrix(size(h2sInfo.alignWeights), params.isGPU, params.dataType);
  if size(srcHidVecs, 3)==1
    grad_alignWeights(:, unmaskedIds) = sum(bsxfun(@times, srcHidVecs(:, unmaskedIds, :), grad_contextVecs(:, unmaskedIds)), 1); % sum across lstmSize: numPositions * batchSize
  else
    grad_alignWeights(:, unmaskedIds) = squeeze(sum(bsxfun(@times, srcHidVecs(:, unmaskedIds, :), grad_contextVecs(:, unmaskedIds)), 1))'; % sum across lstmSize: numPositions * batchSize
  end
  
   
  % grad_alignWeights -> grad_scores
  tmpResult = h2sInfo.alignWeights.*grad_alignWeights; % numAttnPositions * curBatchSize
  grad_scores = tmpResult - bsxfun(@times, h2sInfo.alignWeights, sum(tmpResult, 1));
  % grad_scores -> grad_ht, grad_W_a / grad_srcHidVecs
  if params.attnOpt==1
    % s_t = H_src * h_t
    grad_scores = permute(grad_scores, [3, 2, 1]); % 1 * batchSize * numPositions
    grad_ht = sum(bsxfun(@times, srcHidVecs, grad_scores), 3); % sum along numPositions: lstmSize * batchSize
    grad_srcHidVecs1 = bsxfun(@times, h_t, grad_scores);
  
  
  elseif params.attnOpt==2
    % s_t = H_src * W_a * h_t
     grad_scores = permute(grad_scores, [3, 2, 1]); % 1 * batchSize * numPositions
     grad_ht = sum(bsxfun(@times, srcHidVecs, grad_scores), 3); % sum along numPositions: lstmSize * batchSize
     grad_srcHidVecs1 = bsxfun(@times, h2sInfo.transform_ht, grad_scores);
    
    % grad_h_t -> W_a * h_t
    % todo: add bias
    % grad_W
    attnGrad.W_a = grad_ht*h2sInfo.h_t';
    % inGrad
    grad_ht = model.W_a'*grad_ht;
  end
  
  grad_srcHidVecs = grad_srcHidVecs + grad_srcHidVecs1; % add to the existing grad_srcHidVecs
  
  % grad_ht
  grad_ht = grad_ht + grad_input(params.lstmSize+1:end, :);
end