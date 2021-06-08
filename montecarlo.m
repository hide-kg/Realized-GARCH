clear
clc
N = 100;
para = [0.2, 0.6, 0.3, 0.1, 1, sqrt(0.8), -0.05, 0.2];
T = 500;
test_start = 499;

for ite = 1:N
    type = 1;
    DGP1 = realized_garch_DGP(para, type, T);
    daily_return1 = DGP1.return(2:T);
    RV1 = DGP1.realized(2:T);
    [estimpara1] = realized_garch(daily_return1, RV1, type, test_start);
    garch1(ite,:) = estimpara1.garch;
    measu1(ite,:) = estimpara1.measurement;
end
    


mean(garch1)
mean(measu1)
