function [out] = wrap_fc7(fc7_in)
    len_fc7 = size(fc7_in,1);
    out=zeros(4096,len_fc7,3); % remark: assume #obj=3
    for ii =1:len_fc7 % remark: assume batch=128
        out(:,ii,:) = fc7_in{ii};
    end
end