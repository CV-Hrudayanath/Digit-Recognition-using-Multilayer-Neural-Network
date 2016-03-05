function si = sigmoid( input , derivative )
% Sigmoid function calculation 
% if derivative is 1 then it returns sigmoid derivative value else if 0 then it simply returns sigmoid value.

val = 1.0/(1.0+ exp(-input));
if derivative == 1
    dval = val*(1-val);
    si = dval;
    return;
end

si = val;
return;



end

