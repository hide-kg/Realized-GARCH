function [estimpara, forecast_fit, logL, stderror, tstat] = realized_egarch_midas_vol2(daily_return, RV, test_start, K, N)
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
%   K : 長期成分のラグの数
%   N : 長期成分のrollingの数
%
% variable -
%   mu : リターン式の定数項
%   beta, tau1, tau2, alpha : ReGARCHのGARCH式のパラメータ
%   xi, phi, delta1, delta2, sigma_u : Measurement式のパタメータ(測定方程式の実現測度の個数は1つ)
%   omega, lambda, gamma : MIDAS式のパラメータ
%
% パラメータの初期値
%   [mu, beta, tau1, tau2, alpha, xi, phi, delta1, delta2, sigma_u, omega, lambda, gamma]
%



para0_return = 0.1;
para0_garch = [0.9, -0.1, 0.1, 0.1];
para0_measu = [0.1, 0.9, -0.1, 0.1, sqrt(0.5)];
para0_midas = [0.2, 0.2, 5];

para0 = [para0_return, para0_garch, para0_measu, para0_midas];

ver = 0;
llh = @(x0) -realized_egarch_midas_llh(x0, daily_return(1:test_start), RV(1:test_start), test_start, ver, K, N);
options = optimoptions('fminunc','Display', 'off', 'MaxFunEvals', 1e15, 'MaxIter', 1e15);
para = fminunc(llh, para0, options);

para_return = para(1);
para_garch = para(2:5);
para_measu = para(6:10);
para_midas = para(11:13);

ver = 1;
[llh, ~, sigma, z, h, g] = realized_egarch_midas_llh(para, daily_return, RV, test_start, ver, K, N);

ver = 0;
fun = @(x0) realized_egarch_midas_llh(x0, daily_return(1:test_start), RV(1:test_start), test_start, ver, K, N);
VCV = vcv(fun, para);
tstats = para./sqrt(diag(VCV)');

estim_std = sqrt(diag(VCV)');

n = length(para);

aic = -2 * llh + 2 * n;
bic = -2 * llh + n * log(test_start-1);

para_midas(2) = abs(para_midas(2));
para_midas(3) = abs(para_midas(3));

estimpara = struct();
estimpara.return = para_return;
estimpara.garch = para_garch;
estimpara.measurement = para_measu;
estimpara.midas = para_midas;

forecast_fit = struct();
forecast_fit.cond_vol = sigma;
forecast_fit.residual = z;
forecast_fit.short = h;
forecast_fit.long = g;

logL = struct();
logL.llh = llh;
logL.aic = aic;
logL.bic = bic;

stderror = struct();
stderror.return = estim_std(1);
stderror.garch = estim_std(2:5);
stderror.measurement = estim_std(6:10);
stderror.midas = estim_std(11:13);

tstat = struct();
tstat.return = tstats(1);
tstat.garch = tstats(2:5);
tstat.measurement = tstats(6:10);
tstat.midas = tstats(11:13);

end

function [llh, llhs, sigma, z, h, g] = realized_egarch_midas_llh(para0, daily_return, RV, test_start, ver, K, N)
%
% Realized EGARCHの対数尤度関数の計算
%
% input - 
%   ver : 0(パラメータ推定), 1(予測)
%   K : 長期成分のラグの数
%
% variable -
%   mu : リターン式の定数項
%   beta, tau1, tau2, alpha : ReGARCHのGARCH式のパラメータ
%   xi, phi, delta1, delta2, sigma_u : Measurement式のパタメータ(測定方程式の実現測度の個数は1つ)
%   omega, lambda, gamma : MIDAS式のパラメータ
%
if ver == 0
    T = test_start - 1;
elseif ver == 1
    T = length(daily_return);
end


mu = para0(1);
beta = para0(2);
tau1 = para0(3);
tau2 = para0(4);
alpha = para0(5);
xi = para0(6);
phi = para0(7);
delta1 = para0(8);
delta2 = para0(9);
sigma_u = para0(10);
omega = para0(11);
lambda = abs(para0(12));
gamma = abs(para0(13));


% sigma : 条件付き分散
% h, logh : ボラティリティの短期成分とその対数
% x, logx : 実現ボラティリティとその対数
% g, logg : ボラティリティの長期成分とその対数
% z : 標準化残差
% aveRV : 対数RVの過去の値
% y : 対数RVの平均
% Gamma : Beta weight の値
% long_term : 長期成分の日によって変動する値
sigma = zeros(T, 1);
h = zeros(T, 1);
logh = zeros(T, 1);
x = zeros(T, 1);
logx = log(RV);
g = zeros(T, 1);
logg = zeros(T, 1);
z = zeros(T, 1);
y = zeros(T, K);
long_term = zeros(T, K);

% u : 実現ボラティリティの誤差項
% tau, delta : 短期成分のボラティリティと実現分散のレバレッジ項
% llhs : 対数尤度
u = zeros(T, 1);
tau = zeros(T, 1);
delta = zeros(T, 1);
llhs = zeros(T, 1);

l = 1:K;
Gamma = beta_weight(l, K, gamma);

for t = N * K + 2:T
    if t == N * K + 2
        % 各値の初期値
        sigma(t-1) = mean((daily_return - mu).^2);
        z(t-1) = (daily_return(t-1) - mu)./sqrt(sigma(t-1));
        h(t-1) = sigma(t-1)/RV(t-1);
        logh(t-1) = log(h(t-1));
        delta(t-1) = delta1 * z(t-1) + delta2 * (z(t-1)^2 - 1);
        u(t-1) = logx(t-1) - xi - phi * log(sigma(t-1)) - delta(t-1);
    end
    
    % 短期成分の計算 (GARCH式)
    tau(t-1) = tau1 * z(t-1) + tau2 * (z(t-1)^2 - 1);
    logh(t) = beta * logh(t-1) + tau(t-1) + alpha * u(t-1);
    h(t) = exp(logh(t));
    
    % 長期成分の計算
    for k = 1:K
        for i = 1:N
            y(t-1,k) = y(t-1,k) + logx(t-N*(k-1)-i-1);
        end
    end
    y = y .* 1/N;
    for k = 1:K
        long_term(t-1,k) = Gamma(k) * y(t-1,k);
    end
    logg(t) = omega + lambda * sum(long_term(t-1,:));
    
    g(t) = exp(logg(t));
    
    % 条件付きボラティリティの計算
    sigma(t) = h(t) * g(t);
    
    % リターン式から計算された標準化残差
    z(t) = (daily_return(t) - mu)./sqrt(sigma(t));
    
    % 測定方程式
    delta(t) = delta1 * z(t) + delta2 * (z(t)^2 - 1);
    u(t) = logx(t) - xi - phi * log(sigma(t)) - delta(t);
    
    % 対数尤度
    llhs(t) = -1/2 * (log(sigma(t)) + z(t)^2 + log(sigma_u^2) + u(t)^2/sigma_u^2);
end

if ver == 0
    llh = sum(llhs);
elseif ver == 1
    llh = sum(llhs(1:test_start-1));
end
if beta >= 1 
    llh = -inf;
end
if round(gamma,3) <= 1
    llh = -inf;
end

end

function phi_ell = beta_weight(k, K, gamma2)
% MIDAS項のbeta weight
gamma1 = 1;
j = 1:K;

phi_ell_upp = ((k/K).^(gamma1-1)) .* ((1 - k/K).^(gamma2-1));
phi_ell_low = sum(((j./K).^(gamma1-1)) .* (1 - j./K).^(gamma2-1));

phi_ell = phi_ell_upp./phi_ell_low;

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
