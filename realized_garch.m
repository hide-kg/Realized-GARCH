function [estimpara, forecast_fit, logL, stderror, tstat] = realized_garch(daily_return, RV, type, test_start)
%
% 2021/4/8
%   Hansen et al.(2012)のRealized GARCH(1,1)モデルのプログラム
%
% input -
%   daily_return : 日次リターン
%   RV : 実現ボラティリティ
%   type : 0(通常), 1(対数をとる)
%   test_start : テスト期間の始まり
%
% variable -
%   omega, beta, gamma : ReGARCHのGARCH式のパラメータ
%   xi, phi, tau1, tau2, sigma_u : Measurement式のパタメータ


% パラメータの初期値
%   [omega, beta, gamma, xi, phi, sigma_u, tau1, tau2]
para0 = [0.1, 0.3, 0.6, 0.1, 0.9, 0.4, 0.1, 0.1];

ver = 0;
llh = @(x0) -realized_garch_llh(x0, daily_return(1:test_start), RV(1:test_start), type, test_start, ver);
%[para, ~, ~, ~, ~, hessian] = fminunc(llh, para0);
options = optimoptions('fminunc','Display', 'off', 'MaxFunEvals', 1e15, 'MaxIter', 1e15);
para = fminunc(llh, para0, options);
omega   = para(1);
beta    = para(2);
gamma   = para(3);
xi      = para(4);
phi     = para(5);
sigma_u = para(6);
tau1    = para(7);
tau2    = para(8);

ver = 1;
[llh, llhs, h, z] = realized_garch_llh(para, daily_return, RV, type, test_start, ver);

ver = 0;
fun = @(x0) realized_garch_llh(x0, daily_return(1:test_start), RV(1:test_start), type, test_start, ver);
%[VCV,scores,gross_scores]=vcv(fun,para);
VCV = vcv(fun, para);
tstats = para./sqrt(diag(VCV)');

estim_std = sqrt(diag(VCV)');

n = length(para);

aic = -2 * llh + 2 * n;
bic = -2 * llh + n * log(test_start-1);

estimpara = struct();
estimpara.garch = [omega, beta, gamma];
estimpara.measurement = [xi, phi, sigma_u^2, tau1, tau2];

forecast_fit = struct();
forecast_fit.cond_vol = h;
forecast_fit.residual = z;

logL = struct();
logL.llh = llh;
logL.aic = aic;
logL.bic = bic;

stderror = struct();
stderror.garch = [estim_std(1), estim_std(2), estim_std(3)];
stderror.measurement = [estim_std(4), estim_std(5), estim_std(6), estim_std(7), estim_std(8)];

tstat = struct();
tstat.garch = [tstats(1), tstats(2), tstats(3)];
tstat.measurement = [tstats(4), tstats(5), tstats(6), tstats(7), tstats(8)];

end



function [llh, llhs, h, z] = realized_garch_llh(para0, daily_return, RV, type, test_start, ver)
% Realized GARCHの対数尤度関数の計算
%
% input -
%   type : 0(通常), 1(ボラティリティに対数をとる)
%   test_start : テスト期間の始まり(ただし, パラメータの推定の時は使わない)
%   ver : 0(パラメータ推定), 1(ボラティリティの予測)
%
% variable -
%   h : 日次リターンの条件付き分散 h = var(r|F_t-1)
%   x : ボラティリティの実現測度

if ver == 0
    T = test_start-1;
elseif ver == 1
    T = length(daily_return);
end

omega   = para0(1);
beta    = para0(2);
gamma   = para0(3);
xi      = para0(4);
phi     = para0(5);
sigma_u = para0(6);
tau1    = para0(7);
tau2    = para0(8);

logh = zeros(T, 1);
h = zeros(T, 1);
z = zeros(T, 1);
u = zeros(T, 1);
tau = zeros(T, 1);

x = RV;
h(1) = mean(daily_return.^2);

llhs = zeros(T, 1);

if type == 0
    % 通常のReaized GARCH
    for t = 2:T
        % GARCH式
        if t == 2
            h(2) = omega + beta * h(1) + gamma * x(1);
        else
            h(t) = omega + beta * h(t-1) + gamma * x(t-1);
        end
        % リターン式から計算された標準化残差
        z(t) = daily_return(t)./sqrt(h(t));
        % Measurement式
        tau(t) = tau1 * z(t) + tau2 * (z(t)^2 -1);
        u(t) = x(t) - xi - phi * h(t) - tau(t);
        % 対数尤度
        llhs(t) = -1/2 * (log(h(t)) + (daily_return(t)^2)/h(t) + log(sigma_u^2) + (u(t)^2)/(sigma_u^2));
    end
    if ver == 0
        llh = sum(llhs);
    elseif ver == 1
        llh = sum(llhs(1:test_start-1));
    end
elseif type == 1
    % 対数をとったRealized GARCH
    for t = 2:T
        % GARCH式
        % ただし, hは対数化されているとする
        if t == 2
            logh(2) = omega + beta * log(h(1)) + gamma * log(x(1));
        else
            logh(t) = omega + beta * logh(t-1) + gamma * log(x(t-1));
        end
        % リターン式から計算された標準化残差
        z(t) = daily_return(t)./sqrt(exp(logh(t)));
        % Measurement式
        tau(t) = tau1 * z(t) + tau2 * (z(t)^2 - 1);
        u(t) = log(x(t)) - xi - phi * logh(t) - tau(t);
        % 対数尤度
        %llhs(t) = -1/2 * (logh(t) + (daily_return(t)^2)/exp(logh(t)) + log(sigma_u^2) + (u(t)^2)/(sigma_u^2));
        h(t) = exp(logh(t));
        llhs(t) = -1/2 * (log(h(t)) + (daily_return(t)^2)/h(t) + log(sigma_u^2) + (u(t)^2)/(sigma_u^2));
    end
    if ver == 0
        llh = sum(llhs);
    elseif ver == 1
        llh = sum(llhs(1:test_start-1));
    end
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
