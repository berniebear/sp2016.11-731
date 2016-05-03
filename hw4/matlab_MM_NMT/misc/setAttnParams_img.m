function [params] = setAttnParams_img(params)
  if params.attnFunc>0
    params.numSrcHidVecs = params.srcMaxLen_img;
    
    if params.attnGlobal % global
      params.numAttnPositions = params.numSrcHidVecs;
    else % local
      params.numAttnPositions = 2*params.posWin + 1;
    end
  else
    params.numSrcHidVecs = 0;
  end
end