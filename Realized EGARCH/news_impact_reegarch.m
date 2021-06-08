clear all
% Documents/2021/研究/2021jafee冬 にある
load('vechRMK_63_20142020.mat')
load('intraday_return.mat')
clc
%close all
r = [];
RV = [];


r = intraday_return;
for i = 1:1624
    RMK(:,:,i) = ivech(MX(:,i));
end

i = 30;
for t = 1:1624
    RV(t,1) = RMK(i,i,t);
end

test_start = 1000;

daily_return = r(:,i);

[estimpara, forecast_fit_egarch, logL, stderror, tstat] = realized_egarch(daily_return, RV, test_start);
estimpara

resi = forecast_fit_egarch.residual;
u = forecast_fit_egarch.residual_realized;

mu = estimpara.return;
omega = estimpara.garch(1);
beta = estimpara.garch(2);
tau1 = estimpara.garch(3);
tau2 = estimpara.garch(4);
gamma = estimpara.garch(5);
xi = estimpara.measurement(1);
phi = estimpara.measurement(2);
delta1 = estimpara.measurement(3);
delta2 = estimpara.measurement(4);
sigma_u = estimpara.measurement(5);

sample_var = var(daily_return - mu);

h = forecast_fit_egarch.cond_vol;
z = forecast_fit_egarch.residual;
logx = log(RV);

T = length(daily_return);
for t = 2:T
    % GARCH式
    tau(t-1) = tau1 * z(t-1) + tau2 * (z(t-1)^2 - 1);
    logh(t) = omega + beta * sample_var + tau(t-1);
    
    % リターン式から計算された標準化残差
    
    % 対数尤度
    h(t) = exp(logh(t));
end

figure
plot(forecast_fit_egarch.residual(1:T-1), h(2:T),'.')

