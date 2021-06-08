function DGP = realized_egarch_DGP(para0, T)
%
% Realized EGARCHÇÃÉfÅ[É^ê∂ê¨âﬂíˆ
%

mu = para0(1);
omega = para0(2);
beta = para0(3);
tau1 = para0(4);
tau2 = para0(5);
gamma = para0(6);
xi = para0(7);
phi = para0(8);
delta1 = para0(9);
delta2 = para0(10);
sigma_u = para0(11);

u = zeros(T, 1);
r = zeros(T, 1);

logh(1) = 0.2;
logx(1) = 0.4;


for t = 2:T
    if t == 2
        u(1) = normrnd(0, sigma_u^2);
        z(1) = normrnd(0, 1);
    end
    u(t) = normrnd(0, sigma_u^2);
    z(t) = normrnd(0, 1);
    tau(t-1) = tau1 * z(t-1) + tau2 * (z(t-1)^2 - 1);
    logh(t) = omega + beta * logh(t-1) + tau(t-1) + gamma * u(t-1);
    delta(t) = delta1 * z(t) + delta2 * (z(t)^2 - 1);
    logx(t) = xi + phi * logh(t) + delta(t) + u(t);
    r(t) = mu + sqrt(exp(logh(t))) * z(t);
end
    
DGP = struct();
DGP.realized = exp(logx);
DGP.cond_vol = exp(logh);
DGP.return = r;
