clc

clear

sector = 01;

filename = sprintf('TOPIX17_RV_%02d.mat', sector);
load('EPU.mat')
load('sample_in_month.mat')

load(filename);

daily_return = daily_return_o2c;

test_start = 1000;
start_time = 1000;
%% GARCH
[estimpara_garch, forecast_fit_garch, logL_garch] = garch(daily_return, test_start);
qlike_garch = mean(loss_function(RV, forecast_fit_garch.cond_vol, 0));
stein_garch = mean(loss_function(RV, forecast_fit_garch.cond_vol, 1));
mse_garch = mean(loss_function(RV, forecast_fit_garch.cond_vol, 3));

%% EGARCH
[estimpara_egarch, forecast_fit_egarch, logL_egarch] = egarch(daily_return, test_start);
qlike_egarch = mean(loss_function(RV(start_time:end), forecast_fit_egarch.cond_vol(start_time:end), 0));
stein_egarch = mean(loss_function(RV(start_time:end), forecast_fit_egarch.cond_vol(start_time:end), 1));
mse_egarch = mean(loss_function(RV(start_time:end), forecast_fit_egarch.cond_vol(start_time:end), 3));

%% Realized GARCH
type = 0;
[estimpara, forecast_fit, logL, std_err, tstat] = realized_garch(daily_return, RV, type, test_start);
qlike_rgarch = mean(loss_function(RV(start_time:end), forecast_fit.cond_vol(start_time:end), 0));
stein_rgarch = mean(loss_function(RV(start_time:end), forecast_fit.cond_vol(start_time:end), 1));
mse_rgarch = mean(loss_function(RV(start_time:end), forecast_fit.cond_vol(start_time:end), 3));

%% Realized GARCH (log)
type = 1;
[estimpara_log, forecast_fit_log, logL_log, std_err_log, tstat_log] = realized_garch(daily_return, RV, type, test_start);
qlike_rgarch_log = mean(loss_function(RV(start_time:end), forecast_fit_log.cond_vol(start_time:end), 0));
stein_rgarch_log = mean(loss_function(RV(start_time:end), forecast_fit_log.cond_vol(start_time:end), 1));
mse_rgarch_log = mean(loss_function(RV(start_time:end), forecast_fit_log.cond_vol(start_time:end), 3));

%% Realized EGARCH
[estimpara_reegarch, forecast_fit_reegarch, logL_reegarch, stderror_reegarch, tstat_reegarch] = realized_egarch(daily_return, RV, test_start);
qlike_reegarch = mean(loss_function(RV(start_time:end), forecast_fit_reegarch.cond_vol(start_time:end), 0));
stein_reegarch = mean(loss_function(RV(start_time:end), forecast_fit_reegarch.cond_vol(start_time:end), 1));
mse_reegarch = mean(loss_function(RV(start_time:end), forecast_fit_reegarch.cond_vol(start_time:end), 3));

%% Realized EGARCH MIDAS
K = 240;
[estimpara_midas, forecast_fit_midas, logL_midas, stderror_midas, tstat_midas] = realized_egarch_midas(daily_return, RV, test_start, K);

qlike_reegarch_midas = mean(loss_function(RV(start_time:end), forecast_fit_midas.cond_vol(start_time:end), 0));
stein_reegarch_midas = mean(loss_function(RV(start_time:end), forecast_fit_midas.cond_vol(start_time:end), 1));
mse_reegarch_midas = mean(loss_function(RV(start_time:end), forecast_fit_midas.cond_vol(start_time:end), 3));

%% Realized EGARCH MIDAS EPU
L = 6;
test_start = cumsum(sample_in_month);
[estimpara_epu, forecast_fit_epu] = realized_egarch_midas_epu(daily_return, RV, epu, sample_in_month, test_start(49), L);
qlike_epu = mean(loss_function(RV(start_time:end), forecast_fit_epu.cond_vol(start_time:end),0));
stein_epu = mean(loss_function(RV(start_time:end), forecast_fit_epu.cond_vol(start_time:end),1));
mse_epu = mean(loss_function(RV(start_time:end), forecast_fit_epu.cond_vol(start_time:end),3));

%% Realized EGARCH MIDAS EPU and RV
[estimpara_epu_rv, forecast_fit_epu_rv] = realized_egarch_midas_epu_rv(daily_return, RV, epu, sample_in_month, test_start(49), L);
qlike_epu_rv = mean(loss_function(RV(start_time:end), forecast_fit_epu_rv.cond_vol(start_time:end),0));
stein_epu_rv = mean(loss_function(RV(start_time:end), forecast_fit_epu_rv.cond_vol(start_time:end),1));
mse_epu_rv = mean(loss_function(RV(start_time:end), forecast_fit_epu_rv.cond_vol(start_time:end),3));
