%   Algorithm:
%	1: SDCA
%   2: SPDC
%   3: SPDC_r7

%% Set Params
clear;
%file_name = 'a9a';
file_name = 'ijcnn1';
%file_name = 'covtype';

loss = 'smooth_hinge';
%loss = 'L2_svm';
%%%the lambda should be relatively large, otherwise use SPDC_r8
lambda = 0.0001;%%0.01,0.001,0.0001
tol = 1e-15;
n_epoch = 24;
verbose = 1;

%% SDCA
algorithm = 'SDCA';
fprintf('Algorithm: %s\n', algorithm);
tic;
[dual_gap_sdca,primal_val_sdca,dual_val_sdca] = Interface(file_name, loss, algorithm, lambda, tol, n_epoch, verbose);
time = toc;
fprintf('Time: %f seconds \n', time);

%% SPDC
algorithm = 'SPDC';
fprintf('Algorithm: %s\n', algorithm);
tic;
[dual_gap_spdc,primal_val_spdc,dual_val_spdc] = Interface(file_name, loss, algorithm, lambda, tol, n_epoch, verbose);
time = toc;
fprintf('Time: %f seconds \n', time);

%% SPDC_r7
%%ASPDC with large lambda
algorithm = 'SPDC_r7';
fprintf('Algorithm: %s\n', algorithm);
tic;
[dual_gap_spdc_r7,primal_val_spdc_r7,dual_val_spdc_r7] = Interface(file_name, loss, algorithm, lambda, tol, n_epoch, verbose);
time = toc;
fprintf('Time: %f seconds \n', time);

%% find the optimal primal value and dual value 
% algorithm = 'SDCA';
% n_epoch = 240;
% tol = 1e-18;
% fprintf('Algorithm: %s\n', algorithm);
% tic;
% [dual_gap,primal_val,dual_val] = Interface(file_name, loss, algorithm, lambda, tol, n_epoch, verbose);%only array primal_val is used to find tthe optimal prima value
% time = toc;
% optimal_primal_val = min(primal_val);
% fprintf('optimal_primal_val: %20.18f\n',optimal_primal_val);
% optimal_dual_val = max(dual_val);%%尽管SPDC和SDCA的对偶函数形式优点细微差别，但是由于对偶间隙最后为0，所以SDCA和SPDC的对偶函数值是相等的
% fprintf('optimal_dual_val: %20.18f\n',optimal_dual_val);
% fprintf('Time: %f seconds \n', time);

%% Draw Dual_gap
figure
semilogy(1:size(dual_gap_sdca), dual_gap_sdca, 'g-*');
hold on,semilogy(1:size(dual_gap_spdc), dual_gap_spdc, 'm-square');
hold on,semilogy(1:size(dual_gap_spdc_r7), dual_gap_spdc_r7,'k-v');
hold off

legend('SDCA','SPDC','ASPDC', 'Location','southwest');
xlabel({'Epoch'});
ylabel({'Dual Gap'});
if strcmp(loss,'L2_svm')
    loss = 'L2\_svm';
elseif strcmp(loss,'smooth_hinge')
    loss = 'smooth\_hinge';
end 
title([loss,'   ',file_name,'    lambda = ',num2str(lambda)]);

%% Draw Primal Value 
% figure
% semilogy(0:size(primal_val_sdca)-1,primal_val_sdca - optimal_primal_val, 'g-*');
% hold on,semilogy(0:size(primal_val_spdc)-1, primal_val_spdc - optimal_primal_val, 'm-square');
% hold on,semilogy(0:size(primal_val_spdc_r7)-1,primal_val_spdc_r7 - optimal_primal_val,'k-v');
% hold off
% 
% legend('SDCA','SPDC','SPDC\_r7');
% xlabel({'Epoch'});
% ylabel({'Primal Value - optimal '});
% title([loss,'   ',file_name,'    lambda = ',num2str(lambda)]);

%% Draw Primal Value 
% figure
% semilogy(0:size(dual_val_sdca)-1,optimal_dual_val-dual_val_sdca , 'g-*');
% hold on,semilogy(0:size(dual_val_spdc)-1,optimal_dual_val- dual_val_spdc, 'm-square');
% hold on,semilogy(0:size(dual_val_spdc_r7)-1,optimal_dual_val- dual_val_spdc_r7,'k-v');
% hold off
% 
% legend('SDCA','SPDC','SPDC\_r7');
% xlabel({'Epoch'});
% ylabel({'optimal - Dual Value'});
% title([loss,'   ',file_name,'    lambda = ',num2str(lambda)]);
