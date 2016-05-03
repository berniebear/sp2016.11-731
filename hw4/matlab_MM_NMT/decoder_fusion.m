function [out] = decoder_fusion(in)
    out =cell(size(in));
    for ii=1:length(in)
        out{ii} = in{ii}/3;
    end
end