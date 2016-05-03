%%%
%
% Linear transformation W*inVec
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%% 
function [outVec] = linearLayerForward(b, W, inVec)
  outVec = W*inVec + b*ones(1,size(inVec,2));
end