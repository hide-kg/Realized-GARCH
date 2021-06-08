clc
close all
clear

para = [0.14, 0.6, 0.3, -0.01, 1, sqrt(0.2), -0.03, 0.12];
type = 0;
T = 1000;

[DGP] = realized_garch_DGP(para, type, T);
daily_return = DGP.return(2:1000);
RV = DGP.realized(2:1000);
test_start = 999;
[estimpara, forecast_fit, logL, std_err, tstat] = realized_garch(daily_return, RV, type, test_start);
estimpara

figure
plot(forecast_fit.cond_vol,'r', 'LineWidth',1.5)
hold on
plot(RV,'b')
legend({'—\‘ª’l(Re-GARCH)', 'RV'})

clear
para = [0.14, 0.6, 0.3, -0.01, 1, sqrt(0.2), -0.03, 0.12];
type = 1;
T = 1000;

[DGP] = realized_garch_DGP(para, type, T);
daily_return = DGP.return(2:1000);
RV = DGP.realized(2:1000);
test_start = 999;
[estimpara_log, forecast_fit_log, logL_log, std_err_log, tstat] = realized_garch(daily_return, RV, type, test_start);
estimpara_log


figure
plot(forecast_fit_log.cond_vol,'r', 'LineWidth',1.5)
hold on
plot(RV,'b')
legend({'—\‘ª’l(Re-GARCH(log))', 'RV'});
