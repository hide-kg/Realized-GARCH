
% Documents/2021/Œ¤‹†/2021jafee“~ ‚É‚ ‚é
%load('vechRMK_63_20142020.mat')
%load('intraday_return.mat')

% TOPIX-17ƒVƒŠ[ƒY‚ğŒvZ
load('TOPIX17_RV_12.mat')

clc
%close all
%r = [];
%RV = [];


%r = intraday_return;
daily_return = daily_return_c2c;
%{
for i = 1:1624
    RMK(:,:,i) = ivech(MX(:,i));
end

i = 10;
for t = 1:1624
    RV(t,1) = RMK(i,i,t);
test_start = 1624;

daily_return = r(:,i);
%}

i = 1;
test_start = 1000;

[estimpara, forecast_fit_egarch, logL, stderror, tstat] = realized_egarch(daily_return, RV, test_start);

estimpara
stderror
tstat
logL


test_start = length(daily_return);

[estimpara, forecast_fit, logL] = egarch(daily_return, test_start);
estimpara

figure
plot(RV,'b')
hold on
plot(forecast_fit.cond_vol,'r', 'LineWidth', 1.5)
legend({'RV', '—\‘ª’l(EGARCH)'})

stat_time = 752;
qlike_reegarch = mean(loss_function(RV(stat_time:end), forecast_fit_egarch.cond_vol(stat_time:end), 0));
stein_reegarch = mean(loss_function(RV(stat_time:end), forecast_fit_egarch.cond_vol(stat_time:end), 1));
mse_reegarch = mean(loss_function(RV(stat_time:end), forecast_fit_egarch.cond_vol(stat_time:end), 3));

qlike_egarch = mean(loss_function(RV(stat_time:end), forecast_fit.cond_vol(stat_time:end), 0));
stein_egarch = mean(loss_function(RV(stat_time:end), forecast_fit.cond_vol(stat_time:end), 1));
mse_egarch = mean(loss_function(RV(stat_time:end), forecast_fit.cond_vol(stat_time:end), 3));

figure
plot(RV,'b')
hold on
plot(forecast_fit_egarch.cond_vol,'r', 'LineWidth',1.5)
legend({'RV', '—\‘ª’l(Re-EGARCH)'})


