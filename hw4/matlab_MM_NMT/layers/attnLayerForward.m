function [attnInfo] = attnLayerForward(h_t, params, model, attnData, maskInfo)
%
% Attentional Layer: from lstm hidden state to softmax hidden state 
% Input: 
%   attnData: require attnData.srcHidVecsOrig and attnData.srcLens
%
% Po-Yao Huang, add parallel lstm support for multiple modality in
% contextVecs
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
  attnInfo = [];
  % global attn
  srcHidVecs = attnData.srcHidVecsOrig;
  attnInfo.srcMaskedIds = [];
  
  
  % compute alignScores: numAttnPositions * curBatchSize
  if params.attnOpt==1 || params.attnOpt==2 % dot product or general dot product
    if params.attnOpt==1 % dot product
        alignScores = squeeze(sum(bsxfun(@times, srcHidVecs, attnInfo.transform_ht), 1))'; % numPositions * curBatchSize
        if params.curBatchSize==1 || params.numAttnPositions==1 % handle special cases that causing squeezing to transpose row/col vectors.
            alignScores = alignScores';
        end
    elseif params.attnOpt==2 % general dot product
      attnInfo.transform_ht = model.W_a * h_t; 
      alignScores = squeeze(sum(bsxfun(@times, srcHidVecs, attnInfo.transform_ht), 1))'; % numPositions * curBatchSize
        if params.curBatchSize==1 || params.numAttnPositions==1 % handle special cases that causing squeezing to transpose row/col vectors.
            alignScores = alignScores';
        end
    end
  end  
  
  % normalize -> alignWeights
  maskedIds = attnInfo.srcMaskedIds;
  scores = alignScores;
  scores(maskedIds) = 0;
  % subtract max elements, scores: numClasses * ...
  mx = max(scores, [], 1);
  scores = bsxfun(@minus, scores, mx); 
  % probs
  probs = exp(scores);
  probs(maskedIds) = 0;
  
  norms = sum(probs, 1); % normalization factors
  norms(norms==0) = 1; % for zero columns, set to 1.
  attnInfo.alignWeights = bsxfun(@rdivide, probs, norms); % normalize
  attnInfo.alignWeights(:, maskInfo.maskedIds) = 0;
  
  % alignWeights, srcHidVecs -> contextVecs
  unmaskedIds = maskInfo.unmaskedIds;
  contextVecs = zeroMatrix([params.lstmSize, params.curBatchSize], params.isGPU, params.dataType);
  contextVecs(:, unmaskedIds) = squeeze(sum(bsxfun(@times, srcHidVecs(:, unmaskedIds, :), permute(attnInfo.alignWeights(:, unmaskedIds), [3, 2, 1])), 3)); % lstmSize * batchSize

  % f(W_h*[context_t; h_t])
  attnInfo.input = [contextVecs; h_t];
  attnInfo.h_t = h_t;
  softmax_h = params.nonlinear_f(model.W_h*attnInfo.input);
  attnInfo.softmax_h = softmax_h; % attentional vectors

end

