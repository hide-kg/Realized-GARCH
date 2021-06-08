function [estimpara, forecast_fit, logL, stderror, tstat] = realized_egarch_midas_epu_rv(daily_return, RV, EPU, sample_in_month, test_start, L, K)
%
% 2021/4/28
%   ����������EPU��RV�ŋ쓮�����Realized EGARCH MIDAS���f���̎���
%
% �e���̉c�Ɠ��̓�����,
% Users/hide/Docuents/MATLAB/���p�x�f�[�^/TOPIX-17/sample_in_month.mat
% �Ɋi�[����Ă���. 
%
% input - 
%   daily_return : �������^�[��
%   RV : �����{���e�B���e�B
%   epu : EPU 
%   sample_in_month : 1�����ɉ����̉c�Ɠ������邩�������x�N�g��
%   test_start : �e�X�g���Ԃ̎n�܂�
%   L : ���������̃��O(���������͌����Ȃ̂�, �����ł�L�������ɂȂ�)
%
% variable -
%   mu : ���^�[�����̒萔��
%   beta, tau1, tau2, alpha : ReGARCH��GARCH���̃p�����[�^
%   xi, phi, delta1, delta2, sigma_u : Measurement���̃p�^���[�^(����������̎������x�̌���1��)
%   omega, lambda, gamma : MIDAS���̃p�����[�^
%
% �p�����[�^�̏����l
%   [mu, beta, tau1, tau2, alpha, xi, phi, delta1, delta2, sigma_u, omega, lambda1, lambda2, gamma1, gamma2]
%   lambda1, lambda2 : lambda1��RV�ɑ΂���p�����[�^��lambda2��EPU�ɑ΂���p�����[�^


para0_return = 0.1;
para0_garch = [0.5, -0.1, 0.1, 0.1];
para0_measu = [0.1, 0.9, -0.1, 0.1, sqrt(0.5)];
para0_midas = [0.2, 0.9, 0.9, 5, 5];

para0 = [para0_return, para0_garch, para0_measu, para0_midas];

ver = 0;
llh = @(x0) -realized_egarch_midas_epu_llh(x0, daily_return(1:test_start), RV(1:test_start), EPU, sample_in_month, test_start, ver, L);
options = optimoptions('fminunc','Display', 'off', 'MaxFunEvals', 1e15, 'MaxIter', 1e15);
para = fminunc(llh, para0, options);

para_return = para(1);
para_garch = para(2:5);
para_measu = para(6:10);
para_midas = para(11:15);

ver = 1;
[llh, ~, sigma, z, u, g] = realized_egarch_midas_epu_llh(para, daily_return, RV, EPU, sample_in_month, test_start, ver, L);

ver = 0;
fun = @(x0) realized_egarch_midas_epu_llh(x0, daily_return(1:test_start), RV(1:test_start), EPU, sample_in_month, test_start, ver, L);
para(12) = para(12);
para(13) = para(13);
para(14) = abs(para(14));
para(15) = abs(para(15));

VCV = vcv(fun, para);
tstats = para./sqrt(diag(VCV)');

estim_std = sqrt(diag(VCV)');

n = length(para);

aic = -2 * llh + 2 * n;
bic = -2 * llh + n * log(test_start-1);

para_midas(2) = para_midas(2)^2;
para_midas(3) = para_midas(3)^2;
para_midas(4) = abs(para_midas(4)) + 1;
para_midas(5) = abs(para_midas(5)) + 1;

estimpara = struct();
estimpara.return = para_return;
estimpara.garch = para_garch;
estimpara.measurement = para_measu;
estimpara.midas = para_midas;

forecast_fit = struct();
forecast_fit.cond_vol = sigma;
forecast_fit.residual = z;
forecast_fit.residual_realized = u;
forecast_fit.long = g;

logL = struct();
logL.llh = llh;
logL.aic = aic;
logL.bic = bic;

stderror = struct();
stderror.return = estim_std(1);
stderror.garch = estim_std(2:5);
stderror.measurement = estim_std(6:10);
stderror.midas = estim_std(11:15);

tstat = struct();
tstat.return = tstats(1);
tstat.garch = tstats(2:5);
tstat.measurement = tstats(6:10);
tstat.midas = tstats(11:15);

end

%% Realized EGARCH MIDAS EPU �̑ΐ��ޓx�֐�
function [llh, llhs, sigma, z, u, g] = realized_egarch_midas_epu_llh(para0, daily_return, RV, EPU, sample_in_month, test_start, ver, L)
%
% Realized EGARCH MIDAS EPU�̑ΐ��ޓx�֐��̌v�Z
%
% variable -
%   sample_in_month : 1�����ɉ����̉c�Ɠ������邩��\���x�N�g��
%   T_long : EPU�̂��߂̌����̐�
%

if ver == 0
    T = test_start - 1;
elseif ver == 1
    T = length(daily_return);
end

T_long = length(sample_in_month);

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
lambda1 = para0(12)^2;
lambda2 = para0(13)^2;
gamma1 = abs(para0(14)) + 1;
gamma2 = abs(para0(15)) + 1;

% sigma : �����t�����U
% h, logh : �{���e�B���e�B�̒Z�������Ƃ��̑ΐ�
% x, logx : �����{���e�B���e�B�Ƃ��̑ΐ�
% g, logg : �{���e�B���e�B�̒��������Ƃ��̑ΐ�
% z : �W�����c��
% aveRV : �ΐ�RV�̉ߋ��̒l
% y : �ΐ�RV�̕���
% Gamma : Beta weight �̒l
% long_term : ���������̓��ɂ���ĕϓ�����l
sigma = zeros(T, 1);
h = zeros(T, 1);
logh = zeros(T, 1);
x = zeros(T, 1);
logx = log(RV);
g = zeros(T_long, 1);
logg = zeros(T_long, 1);
z = zeros(T, 1);
long_term = zeros(T_long, L);

% u : �����{���e�B���e�B�̌덷��
% tau, delta : �Z�������̃{���e�B���e�B�Ǝ������U�̃��o���b�W��
% llhs : �ΐ��ޓx
u = zeros(T, 1);
tau = zeros(T, 1);
delta = zeros(T, 1);
llhs = zeros(T, 1);

%% ���������̐���
l = 1:L;
Gamma = beta_weight(l, L, gamma1);

for t = L+1:T_long
    for ell = 1:L
        long_term(t,ell) = Gamma(ell) * log(EPU(t-ell));
    end
    logg(t) = lambda1 * sum(long_term(t,:));
end

%% �Z�������̐���

% L : ���������̌����̃��O����
% start_time : ����̊J�n��

L = L+1;

start_time = sum(sample_in_month(1:L)) + 2;
cumsample = cumsum(sample_in_month);

K = 20*L;
k = 1:K;
Gamma_RV = beta_weight(k, K, gamma2);

long_term_RV = zeros(T, K);
g_RV = zeros(T, 1);

for t = start_time:T
    % RV�̒��������̌v�Z
    for ell = 1:K
        long_term_RV(t-1,ell) = Gamma_RV(ell) * logx(t-ell-1);
    end
    g_RV(t) = lambda2 * sum(long_term_RV(t-1,:));
    
    if t <= cumsample(L)
        logg(t) = g_RV(t) + logg(L);
    else
        L = L + 1;
        logg(t) = g_RV(t) + logg(L);
    end
    g(t) = exp(omega + logg(t));
    
    % �����l�̌v�Z
    if t == start_time 
        sigma(t-1) = mean((daily_return - mu).^2);
        z(t-1) = (daily_return(t-1) - mu)./sqrt(sigma(t-1));
        h(t-1) = sigma(t-1)/g(t);
        logh(t-1) = log(h(t-1));
        delta(t-1) = delta1 * z(t-1) + delta2 * (z(t-1)^2 - 1);
        u(t-1) = logx(t-1) - xi - phi * log(sigma(t-1)) - delta(t-1);
    end
    
    % GARCH���̌v�Z
    tau(t-1) = tau1 * z(t-1) + tau2 * (z(t-1)^2 - 1);
    logh(t) = beta * logh(t-1) + tau(t-1) + alpha * u(t-1);
    h(t) = exp(logh(t));
    

    
    % �����t���{���e�B���e�B�̌v�Z
    % logg�͌����f�[�^�Ȃ̂�, 1�����Ԃ͒l���Œ肵�Čv�Z����
    sigma(t) = h(t) * g(t);
    
    % ���^�[��������v�Z���ꂽ�W�����c��
    z(t) = (daily_return(t) - mu)./sqrt(sigma(t));
    
    % ���������
    delta(t) = delta1 * z(t) + delta2 * (z(t)^2 - 1);
    u(t) = logx(t) - xi - phi * log(sigma(t)) - delta(t);
    
    % �ΐ��ޓx
    
    llhs(t) = -1/2 * (log(sigma(t)) + z(t)^2 + log(sigma_u^2) + u(t)^2/sigma_u^2);
end

if ver == 0
    llh = sum(llhs(start_time:T));
elseif ver == 1
    llh = sum(llhs(1:test_start-1));
end
if beta >= 0.9999
    llh = -inf;
end

end

%% Beta weight
function phi_ell = beta_weight(k, K, gamma)
% MIDAS����beta weight
% omega > 1�ł���K�v������
j = 1:K;

phi_ell_upp = (1 - k/K).^(gamma-1);
phi_ell_low = sum((1 - j./K).^(gamma-1));

phi_ell = phi_ell_upp./phi_ell_low;

end
    
    
%% �W���덷�̌v�Z

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

% fun�̏o�͈����ɂ����, �����̏o�͈�����ύX����
% like��1���Ԃ��Ƃ̑ΐ��ޓx�ƂȂ�
% ��, fun��realized_garch_llh��, �O����2�ڂ̏o�͈�����llhs(1���Ԃ��Ƃ̑ΐ��ޓx�Ƃ��Ă���)
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






















