function [fc7, sents, numSents, sentLens] = loadBatchData_img_mm(img_fid, fid, baseIndex, batchSize) %, varargin)
%%%
%
% Load a number of sentences (integers per line) from a file.
%   baseIndex: minimum integer value in the input file.
%   batchSize: number of sents to read (if batchSize==-1, read all).
%
% Po-Yao Huang, multi-modal data load
% Thang Luong @ 2013-2015 <lmthang@stanford.edu>
%%%
  %batchSize
  sents = cell(1, batchSize);
  %fc7 = zeros(4096, batchSize);
  fc7 = cell(batchSize,1);
  sentLens = zeros(1, batchSize);
  numSents = 0;
  while ~feof(fid)
    indices = sscanf(fgetl(fid), '%d') + (1-baseIndex);
    
    if isempty(indices) % ignore empty lines
      continue
    end
    numSents = numSents + 1;

    if 0 %img_fid ~= -1
      image_mats = fgetl(img_fid); % read line by line
      image_mats = strsplit(image_mats,' ');
      fc7{numSents} = zeros(4096, length(image_mats));
      for ii=1:1:length(image_mats)
          %tmp = load(strcat('../fc7/',image_mats{ii}));
          %feat = tmp.fc7;
          feat = zeros(4096,1);%tmp.fc7;
          feat = feat/10;
          % normalization
          %feat =feat-mean(feat);
          %feat = feat/norm(feat);
        fc7{numSents}(:,ii) =feat;    
      end
    else
        fc7{numSents} = zeros(4096, 3);
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
