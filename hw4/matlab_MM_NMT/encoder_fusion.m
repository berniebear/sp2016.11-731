function last = encoder_fusion(last1, last2, last3)
    last = last1;
    for fields = fieldnames(last1{1})
        for ii=1:length(fields)
            last{1}.(fields{ii}) = (last1{1}.(fields{ii}) + last2{1}.(fields{ii}) + last3{1}.(fields{ii}))/3 ;
        end
    end
end