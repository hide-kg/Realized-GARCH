


close all
clear
% 5, 6, 7”Ô–Ú‚Ìw”‚Å607‚Ì’l‚ª‚¨‚©‚µ‚¢
load('TOPIX17_RV_06.mat')
%{
load('vechRMK_63_20142020.mat')
load('intraday_return.mat')
r = [];
RV = [];
r = intraday_return;
for i = 1:1624
    RMK(:,:,i) = ivech(MX(:,i));
end

i = 2;
for t = 1:1624
    RV(t,1) = RMK(i,i,t);
end
test_start = 1624;
daily_return = r(:,i);
%}
daily_return = daily_return_o2c;
test_start = 700;

K = 12;
N = 22;
[estimpara, forecast_fit_egarch, logL, stderror, tstat] = realized_egarch_midas_vol2(daily_return, RV, test_start, K, N);

estimpara

stat_time = 752;

qlike_reegarch_midas = mean(loss_function(RV(stat_time:end), forecast_fit_egarch.cond_vol(stat_time:end), 0))
stein_reegarch_midas = mean(loss_function(RV(stat_time:end), forecast_fit_egarch.cond_vol(stat_time:end), 1))
mse_reegarch_midas = mean(loss_function(RV(stat_time:end), forecast_fit_egarch.cond_vol(stat_time:end), 3))

e = forecast_fit_egarch.long;
h = forecast_fit_egarch.short;
test_start = 1;


figure
subplot(2,1,1)
plot(RV, 'b')
hold on
plot(forecast_fit_egarch.cond_vol, 'r', 'LineWidth', 1.5)
legend({'RV', '—\‘ª’l(Re-EGARCH-MIDAS)'})

subplot(2,1,2)
plot(forecast_fit_egarch.cond_vol(test_start:end),'b', 'LineWidth',1.5)
hold on
plot(e(test_start:end),'r', 'LineWidth',1.5)

