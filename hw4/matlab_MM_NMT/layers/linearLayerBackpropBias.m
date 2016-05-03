%% 
% Forward outVec = W*inVec
% Compute inGrad, grad_W
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%%
function [inGrad, grad_W, grad_b] = linearLayerBackprop(b, W, outGrad, inVec)  
  % grad_W
  grad_W = outGrad*inVec';

  % grad_b
  grad_b = outGrad*ones(size(inVec,2),1);
  
  % inGrad
  inGrad = W'*outGrad;
end