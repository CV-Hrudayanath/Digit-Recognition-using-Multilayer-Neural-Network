function si = sigmoid( input , derivative )
% Sigmoid function calculation 

val = 1.0/(1.0+ exp(-input));
if derivative == 1
    dval = val*(1-val);
    si = dval;
    return;
end

si = val;
return;



end

