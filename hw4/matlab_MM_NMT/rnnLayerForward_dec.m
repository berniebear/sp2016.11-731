function [lstmStates, attnData, attnInfos] = rnnLayerForward_dec(~, ~, W_rnn, W_emb, prevState, input, masks, params, rnnFlags, attnData, model)
% Running Multi-layer RNN for one time step with multiple modalities
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
% Po-Yao Huang, 
% Thang Luong @ 2015, <lmthang@stanford.edu>


T = size(input, 2);
lstmStates = cell(T, 1);

% attention
attnInfos = cell(T, 1);
for tt=1:T % time
  [prevState, attnInfos{tt}] = rnnStepLayerForward(W_rnn, W_emb(:, input(:, tt)), prevState, masks(:, tt), params, rnnFlags, attnData, model, []);  
  % store all states
  lstmStates{tt} = prevState;
end