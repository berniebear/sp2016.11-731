function [srcfc7, srcSents, tgtSents, numSents] = loadBiData(params, prefix, srcVocab, tgtVocab, varargin)
  chunkSize = -1;
  hasTgt = 1; % 1 -- has tgt file
  
  if length(varargin) >= 1
    chunkSize = varargin{1};
  end
  if length(varargin) >= 2
    hasTgt = varargin{2};
  end
  assert(params.isBi || hasTgt==1);
  
  % src
  if params.isBi
    if params.isReverse
      srcFile = sprintf('%s.%s.reversed', prefix, params.srcLang);
    else
      srcFile = sprintf('%s.%s', prefix, params.srcLang);
    end
    srcImgFile = sprintf('%s_images.txt', prefix);
    [srcfc7, srcSents, numSents] = loadMonoData(srcImgFile, srcFile, chunkSize, params.baseIndex, srcVocab, 'src');
  else
    srcSents = {};
  end
  
  % tgt
  if hasTgt
    tgtFile = sprintf('%s.%s', prefix, params.tgtLang);
    [~, tgtSents, numSents] = loadMonoData('', tgtFile, chunkSize, params.baseIndex, tgtVocab, 'tgt');
  else
    tgtSents = repmat({[]}, 1, numSents);
  end
end

function [fc7, sents, numSents] = loadMonoData(imgFile, file, numSents, baseIndex, vocab, label)
  fprintf(2, '# Loading data %s from file %s\n', label, file);
  fid = fopen(file, 'r');
  if fid == -1
      fprintf(2, 'Error: Cannot open %s\n', file);
  end
  
  fc7 = [];
  if isempty(imgFile) == 0
      img_fid = fopen(imgFile, 'r');
      if img_fid == -1
          fprintf(2, 'Error: Cannot open %s\n', imgFile);
      end
      [fc7, sents, numSents, ~] = loadBatchData_img(img_fid, fid, baseIndex, numSents);
  else
      [sents, numSents, ~] = loadBatchData(fid, baseIndex, numSents);
  end
  
  
  fclose(fid);
  printSent(2, sents{1}, vocab, ['  ', label, ' 1:']);
  printSent(2, sents{end}, vocab, ['  ', label, ' end:']);


end
