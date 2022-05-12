% The polynomial approach does not really work unless given a good enough
% initial value. 
%
% I am doing this in Matlab because numpy polyfit for some reason gives 
% numerical singular issues.
%

a = 500;
b = 5;
c = 8;

meow = @(t) a * b * cot(t) .* csc(t) .* exp(-b * csc(t)) + c;

dt = 0.01;
T = 314;
ts = [dt:dt:dt*T];

order = 10;
coeffs = polyfit(ts, meow(ts), order);

fprintf('%e\n', flip(coeffs))

plot(ts, meow(ts))
hold on
plot(ts, polyval(coeffs, ts))
