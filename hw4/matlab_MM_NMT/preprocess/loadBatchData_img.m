function [fc7, sents, numSents, sentLens] = loadBatchData_img(img_fid, fid, baseIndex, batchSize) %, varargin)
%%%
%
% Load a number of sentences (integers per line) from a file.
%   baseIndex: minimum integer value in the input file.
%   batchSize: number of sents to read (if batchSize==-1, read all).
%
% Thang Luong @ 2013-2015 <lmthang@stanford.edu>
%%%
  %batchSize
  sents = cell(1, batchSize);
  fc7 = zeros(4096, batchSize);
  sentLens = zeros(1, batchSize);
  numSents = 0;
  while ~feof(fid)
    indices = sscanf(fgetl(fid), '%d') + (1-baseIndex);
    
    if isempty(indices) % ignore empty lines
      continue
    end
    numSents = numSents + 1;

    if img_fid ~= -1
      imageName = fgetl(img_fid); % read line by line
      tmp = load(['../../fc7/' strrep(num2str(imageName), '.jpg', '') '_fc7.mat']);
      feat = tmp.fc7;
      feat = feat/10;
      % normalization
      %feat =feat-mean(feat);
      %feat = feat/norm(feat);
      fc7(:, numSents) =feat;
    end
    
        
    sents{numSents} = indices; %[indices' suffix];
    sentLens(numSents) = length(sents{numSents});
    if numSents==batchSize   
      break;
    end
  end
  
  % delete empty values
  sents((numSents+1):end) = []; 
  sentLens((numSents+1):end) = [];
  %sents

end
