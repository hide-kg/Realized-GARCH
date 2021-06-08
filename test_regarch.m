
load('TOPIX17_RV_01.mat')
%close all

daily_return = daily_return_o2c;

test_start = 1000;

type = 0;
[estimpara, forecast_fit, logL, std_err, tstat] = realized_garch(daily_return, RV, type, test_start);
estimpara
std_err
tstat
logL

type = 1;
[estimpara_log, forecast_fit_log, logL_log, std_err_log, tstat] = realized_garch(daily_return, RV, type, test_start);
estimpara_log
std_err_log
tstat
logL_log

price_return = daily_return;
distribution = 'norm';
[sigma, epsilon, mu, rho, omega, alpha, beta, nu, sim_r] = garch_estimate_vol2(price_return, distribution);

qlike_rgarch = mean(loss_function(RV, forecast_fit.cond_vol, 0));
qlike_rgarch_log = mean(loss_function(RV, forecast_fit_log.cond_vol, 0));
qlike_garch = mean(loss_function(RV, sigma, 0));

stein_rgarch = mean(loss_function(RV, forecast_fit.cond_vol, 1));
stein_rgarch_log = mean(loss_function(RV, forecast_fit_log.cond_vol, 1));
stein_garch = mean(loss_function(RV, sigma, 1));

mse_rgarch = mean(loss_function(RV, forecast_fit.cond_vol, 3));
mse_rgarch_log = mean(loss_function(RV, forecast_fit_log.cond_vol, 3));
mse_garch = mean(loss_function(RV, sigma, 3));

figure
plot(RV,'b')
hold on
plot(forecast_fit.cond_vol,'r', 'LineWidth',1.5)
legend({'RV', '—\‘ª’l(Re-GARCH)'})


figure
plot(RV,'b')
hold on
plot(forecast_fit_log.cond_vol,'r', 'LineWidth',1.5)
legend({'RV', '—\‘ª’l(Re-GARCH(log))'});

figure
plot(RV,'b')
hold on
plot(sigma,'r', 'LineWidth', 1.5)
legend({'RV', '—\‘ª’l(GARCH)'})

figure
plot(RV, 'r')
hold on
plot(daily_return, 'b')

