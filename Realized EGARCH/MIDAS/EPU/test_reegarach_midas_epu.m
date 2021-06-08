
clear

clc

% 5, 6, 7”Ô–Ú‚Ìw”‚Å607‚Ì’l‚ª‚¨‚©‚µ‚¢

load('TOPIX17_RV_01.mat')
load('EPU.mat')
load('sample_in_month.mat')
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
%%
daily_return = daily_return_o2c;
test_start = cumsum(sample_in_month);

L = 12;

[estimpara, forecast_fit_egarch, logL, stderror, tstat] = realized_egarch_midas_epu(daily_return, RV, epu, sample_in_month, test_start(60), L);

estimpara


stat_time = 752;

qlike_reegarch_midas = mean(loss_function(RV(stat_time:end), forecast_fit_egarch.cond_vol(stat_time:end), 0))
stein_reegarch_midas = mean(loss_function(RV(stat_time:end), forecast_fit_egarch.cond_vol(stat_time:end), 1))
mse_reegarch_midas = mean(loss_function(RV(stat_time:end), forecast_fit_egarch.cond_vol(stat_time:end), 3))


e = forecast_fit_egarch.long;
ep = (epu - mean(epu))/std(epu);

x = 1:1708;
x_epu = 1:84;
figure
subplot(2,1,1)
plot(RV, 'b')
hold on
plot(forecast_fit_egarch.cond_vol, 'r', 'LineWidth', 1.5)
legend({'RV', '—\‘ª’l(Re-EGARCH-MIDAS)'})

subplot(2,1,2)
plot(e,'r')
hold on
plot(epu,'b', 'LineWidth',1.5)
