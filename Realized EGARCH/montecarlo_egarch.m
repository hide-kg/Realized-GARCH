clear
clc
para = [0.05, 0.1, 0.9, -0.1, 0.1, 0.1, 0.2, 0.9, -0.1, 0.1, sqrt(0.8)];
T = 500;

DGP = realized_egarch_DGP(para, T);

daily_return = DGP.return;
RV = DGP.realized;
test_start = T-1;

N = 1000;
for ite = 1:N
    DGP = realized_egarch_DGP(para, T);
    daily_return = DGP.return(2:T);
    RV = DGP.realized(2:T);
    [estimpara] = realized_egarch(daily_return, RV, test_start);
    r1(ite) = estimpara.return;
    garch1(ite,:) = estimpara.garch;
    measu1(ite,:) = estimpara.measurement;
end
para
estim_para = [mean(r1), mean(garch1), mean(measu1)]
