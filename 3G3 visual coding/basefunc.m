function dcost=basefunc(a,B,s)
%s is the subimage vector
%B is the base matrix
%a is the activity vector

% dcost is the returned vector of the partial derivatives 
% of the cost with respect to the bases
% you should calculate dcost here 


dcost= -2*s*transpose(a) + 2*B*a*transpose(a);