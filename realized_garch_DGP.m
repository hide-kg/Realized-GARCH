function [DGP] = realized_garch_DGP(para, type, T)
% Realized GARCHÇÃÉfÅ[É^ê∂ê¨âﬂíˆ

omega   = para(1);
beta    = para(2);
gamma   = para(3);
xi      = para(4);
phi     = para(5);
sigma_u = para(6);
tau1    = para(7);
tau2    = para(8);

u = zeros(T, 1);
h = zeros(T, 1);
x = zeros(T, 1);
r = zeros(T, 1);

h(1) = 0.2;
x(1) = 0.4;

if type == 1
    h(1) = log(h(1));
    x(1) = log(x(1));
end

if type == 0
    for t = 2:T
        h(t) = omega + beta * h(t-1) + gamma * x(t-1);
        z = normrnd(0,1);
        tau = tau1 * z + tau2 * (z^2 - 1);
        u(t) = normrnd(0, sigma_u^2);
        x(t) = xi + phi * h(t) + tau + u(t);
        r(t) = sqrt(h(t)) * z;
    end
    DGP = struct();
    DGP.realized = x;
    DGP.cond_vol = h;
    DGP.return = r;
elseif type == 1
    for t = 2:T
        u(t) = normrnd(0, sigma_u^2);
        h(t) = omega + beta * h(t-1) + gamma * x(t-1);
        z = normrnd(0,1);
        tau = tau1 * z + tau2 * (z^2 - 1);
        x(t) = xi + phi * h(t) + tau + u(t);
        r(t) = sqrt(exp(h(t)))*z;
    end
    DGP = struct();
    DGP.realized = exp(x);
    DGP.cond_vol = exp(h);
    DGP.return = r;
end


