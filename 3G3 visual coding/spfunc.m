function [cost,dcost]=spfunc(a,B,s,sigma,lambda)

%s is the subimage vector
%B is the base matrix
%a is the activations
% cost is the cost
% dcost is the vector of the derivatives of the cost with respect to the
% activations

cost= transpose(s-B*a)*(s-B*a) + lambda*sum(log(1+(power(a,2))./power(sigma,2)));

dcost= -2*transpose(B)*s + 2*transpose(B)*B*a + (2*lambda*a)./(sigma^2+a.^2);
%dcost= -s.'*B - B.'*s + (2*B*B.')*a + (2*lambda*a)./(sigma^2+a.^2)

