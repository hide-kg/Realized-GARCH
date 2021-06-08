function [estimpara, forecast_fit, logL, stderror, tstat] = realized_egarch(daily_return, RV, test_start)
% 
% 2021/4/13
%   Hansen and Huang (2016)のRealized EGARCH(1,1)モデルのプログラム
%   ただし, 測定方程式の実現測度の個数は1つとする. 
% 
% input - 
%   daily_return : 日次リターン
%   RV : 実現ボラティリティ
%   type : 0(通常), 1(対数をとる)
%   test_start : テスト期間の始まり
%
% variable -
%   mu : リターン式の定数項
%   omega, beta, tau1, tau2, gamma : ReGARCHのGARCH式のパラメータ
%   xi, phi, delta1, delta2, sigma_u : Measurement式のパタメータ(測定方程式の実現測度の個数は1つ)
%
% パラメータの初期値
%   [mu, omega, beta, tau1, tau2, gamma, xi, phi, delta1, delta2, sigma_u]
%

para0_return = 0.1;
para0_garch = [0.1, 0.9, 0.1, 0.1, 0.1];
para0_measu = [0.1, 0.9, 0.1, 0.1, sqrt(0.5)];

para0 = [para0_return, para0_garch, para0_measu];

ver = 0;
llh = @(x0) -realized_egarch_llh(x0, daily_return(1:test_start), RV(1:test_start), test_start, ver);
options = optimoptions('fminunc','Display', 'off', 'MaxFunEvals', 1e15, 'MaxIter', 1e15);
para = fminunc(llh, para0, options);

para_return = para(1);
para_garch = para(2:6);
para_measu = para(7:11);

ver = 1;
[llh, ~, h, z, u] = realized_egarch_llh(para, daily_return, RV, test_start, ver);

ver = 0;
fun = @(x0) realized_egarch_llh(x0, daily_return(1:test_start), RV(1:test_start), test_start, ver);
VCV = vcv(fun, para);
tstats = para./sqrt(diag(VCV)');

estim_std = sqrt(diag(VCV)');

n = length(para);

aic = -2 * llh + 2 * n;
bic = -2 * llh + n * log(test_start-1);

estimpara = struct();
estimpara.return = para_return;
estimpara.garch = para_garch;
estimpara.measurement = para_measu;

forecast_fit = struct();
forecast_fit.cond_vol = h;
forecast_fit.residual = z;
forecast_fit.residual_realized = u;

logL = struct();
logL.llh = llh;
logL.aic = aic;
logL.bic = bic;

stderror = struct();
stderror.return = estim_std(1);
stderror.garch = estim_std(2:6);
stderror.measurement = estim_std(7:11);

tstat = struct();
tstat.return = tstats(1);
tstat.garch = tstats(2:6);
tstat.measurement = tstats(7:11);

end

function [llh, llhs, h, z, u] = realized_egarch_llh(para0, daily_return, RV, test_start, ver)
%
% Realized EGARCHの対数尤度関数の計算
%
% input - 
%   ver : 0(パラメータ推定), 1(予測)

if ver == 0
    T = test_start - 1;
elseif ver == 1
    T = length(daily_return);
end

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

logh = zeros(T, 1);
z = zeros(T, 1);
u = zeros(T, 1);
tau = zeros(T, 1);
delta = zeros(T, 1);
h = zeros(T, 1);
llhs = zeros(T, 1);

logx = log(RV);
h(1) = mean((daily_return - mu).^2);
z(1) = (daily_return(1) - mu)./sqrt(h(1));
logh(1) = log(h(1));


for t = 2:T
    % GARCH式
    tau(t-1) = tau1 * z(t-1) + tau2 * (z(t-1)^2 - 1);
    logh(t) = omega + beta * logh(t-1) + tau(t-1) + gamma * u(t-1);
    
    % リターン式から計算された標準化残差
    z(t) = (daily_return(t) - mu)./sqrt(exp(logh(t)));
    
    % 測定方程式
    delta(t) = delta1 * z(t) + delta2 * (z(t)^2 - 1);
    u(t) = logx(t) - xi - phi * logh(t) - delta(t);
    
    % 対数尤度
    h(t) = exp(logh(t));
    llhs(t) = -1/2 * (log(h(t)) + z(t)^2 + log(sigma_u^2) + u(t)^2/sigma_u^2);
end

if ver == 0
    llh = sum(llhs);
elseif ver == 1
    llh = sum(llhs(1:test_start-1));
end
end



function [VCV,scores,gross_scores]=vcv(fun,theta,varargin)
% Compute Variance Covariance matrix numerically only based on gradient
%
% USAGE:
%     [VCV,A,SCORES,GROSS_SCORES]=vcv(FUN,THETA,VARARGIN)
%
% INPUTS:
%     FUN           - Function name ('fun') or function handle (@fun) which will
%                       return the sum of the log-likelihood (scalar) as the 1st output and the individual
%                       log likelihoods (T by 1 vector) as the second output.
%     THETA         - Parameter estimates at the optimum, usually from fmin*
%     VARARGIN      - Other inputs to the log-likelihood function, such as data
%
% OUTPUTS:
%     VCV           - Estimated robust covariance matrix (see White 1994)
%     SCORES        - T x num_parameters matrix of scores
%     GROSS_SCORES  - Numerical scores (1 by num_parameters) of the objective function, usually for diagnostics
%
% COMMENTS:
%     For (Q)MLE estimation

% Michael Stollenwerk
% michael.stollenwerk@live.com
% 05.02.2019

% Adapted from robustvcv by Kevin Sheppard:
% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 1    Date: 9/1/2005

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input Argument Checking
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if size(theta,1)<size(theta,2)
    theta=theta';
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input Argument Checking
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


k=length(theta);
h=abs(theta*eps^(1/3));
h=diag(h);

% funの出力引数によって, ここの出力引数を変更する
% likeは1期間ごとの対数尤度となる
% 今, funはrealized_garch_llhで, 前から2つ目の出力引数はllhs(1期間ごとの対数尤度としている)
[~,like]=feval(fun,theta,varargin{:});

t=length(like);

LLFp=zeros(k,1);
LLFm=zeros(k,1);
likep=zeros(t,k);
likem=zeros(t,k);
for i=1:k
    thetaph=theta+h(:,i);
    [LLFp(i),likep(:,i)]=feval(fun,thetaph,varargin{:});
    thetamh=theta-h(:,i);
    [LLFm(i),likem(:,i)]=feval(fun,thetamh,varargin{:});
end

scores=zeros(t,k);
gross_scores=zeros(k,1);
h=diag(h);
for i=1:k
    scores(:,i)=(likep(:,i)-likem(:,i))./(2*h(i));
    gross_scores(i)=(LLFp(i)-LLFm(i))./(2*h(i));
end

B=cov(scores);
VCV=inv(B)/t;
end


























