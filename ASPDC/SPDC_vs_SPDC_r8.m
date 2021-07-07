%   Algorithm:
%	1: SDCA
%   2: ASDCA %%注意ASDCA是有应用条件的。
%   3: SPDC
%   4: FastSPDC（SPDC_r8）

%% Set Params
clear ;
%file_name ='phishing';
%file_name = 'cod-rna';
%file_name = 'a9a';
%file_name ='realsim';
%file_name = 'rcv1';
%file_name = 'news20';

%file_name = 'ijcnn1';
file_name = 'covtype';

loss = 'smooth_hinge';
%loss = 'L2_svm';
%%%%% lambda should be relatively small in FastSPDC(SPDC_r8), otherwise call SPDC_r7 
lambda = 1e-7;%%lambda要小于1e-6
tol = 1e-15;
EPOCH = 12;%%epoch 次数
n_epoch = EPOCH;
m = 5;
H = 2;%%FastSPDC的内循环次数
%%具体是5倍还是其他2倍或者其他值，取决于FastSPDC(SPDC_r8)中是如何实现的。为了方便和ASDCA对比，就取m=5,即在C++代码中，m=5
verbose = 1;

%% SDCA
algorithm = 'SDCA';
fprintf('Algorithm: %s\n', algorithm);
n_epoch=EPOCH*m;%%fair comparison
tic;
[dual_gap_sdca,primal_val_sdca,dual_val_sdca] = Interface(file_name, loss, algorithm, lambda, tol, n_epoch, verbose);
time = toc;
fprintf('Time: %f seconds \n', time);

%% SPDC
algorithm = 'SPDC';
fprintf('Algorithm: %s\n', algorithm);
n_epoch=EPOCH*m;
tic;
[dual_gap_spdc,primal_val_spdc,dual_val_spdc] = Interface(file_name, loss, algorithm, lambda, tol, n_epoch, verbose);
time = toc;
fprintf('Time: %f seconds \n', time);

%% ASDCA
algorithm = 'ASDCA';
fprintf('Algorithm: %s\n', algorithm);
n_epoch=EPOCH;
tic;
[dual_gap_asdca,primal_val_asdca,dual_val_asdca] = Interface(file_name, loss, algorithm, lambda, tol, n_epoch, verbose);
time = toc;
fprintf('Time: %f seconds \n', time);


%% FastSPDC(SPDC_r8)
algorithm = 'FastSPDC';
fprintf('Algorithm: %s\n', algorithm);
n_epoch=EPOCH*m/H;
tic;
%%%% value in dual_gap_spdc_r8 are absolute value 
[dual_gap_spdc_r8,primal_val_spdc_r8,dual_val_spdc_r8] = Interface(file_name, loss, algorithm, lambda, tol, n_epoch, verbose);
time = toc;
fprintf('Time: %f seconds \n', time);

%% find the optimal primal value and dual value 
%%在lambda很小的时候，不应该用SDCA去获取最优的函数值
% algorithm = 'FastSPDC';
% n_epoch = 120;
% tol = 1e-20;
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
semilogy(0:m:size(dual_gap_sdca)-1, dual_gap_sdca(1:m:size(dual_gap_sdca)), 'g-*');
hold on,semilogy(0:m:size(dual_gap_spdc)-1, dual_gap_spdc(1:m:size(dual_gap_spdc)), 'm-square');
hold on,semilogy(0:H:size(dual_gap_spdc_r8)*H-1, dual_gap_spdc_r8,'k-v');
%hold on,semilogy(0:H:size(dual_gap_asdca)*H-1, dual_gap_asdca,'r-+');
hold on,semilogy(0:m:size(dual_gap_asdca)*m-1, dual_gap_asdca,'r-+');
hold off

legend('SDCA','SPDC','FastSPDC\-s','ASDCA');
xlabel({'Number of passes of data'});
ylabel({'Dual Gap'});
%axis([0 n_epoch]);
if strcmp(loss,'L2_svm')
    loss = 'L2\_svm';
elseif strcmp(loss,'smooth_hinge')
    loss = 'smooth\_hinge';
end 
title([loss,'   ',file_name,'    lambda = ',num2str(lambda)]);

%% Draw Primal Value 
% figure
% semilogy(0:m:size(primal_val_sdca)-1,primal_val_sdca(1:m:size(primal_val_sdca)) - optimal_primal_val, 'g-*');
% hold on,semilogy(0:m:size(primal_val_spdc)-1, primal_val_spdc(1:m:size(primal_val_spdc)) - optimal_primal_val, 'm-square');
% hold on,semilogy(0:m:size(primal_val_asdca)*m-1, primal_val_asdca- optimal_primal_val,'r-+');
% hold on,semilogy(0:m:size(primal_val_spdc_r8)*m-1,primal_val_spdc_r8 - optimal_primal_val,'k-v');
% hold off
% 
% legend('SDCA','SPDC','ASDCA','FastSPDC');
% xlabel({'Number of passes of data'});
% ylabel({'Primal Value - optimal '});
% %axis([0 n_epoch]);
% title([loss,'   ',file_name,'    lambda = ',num2str(lambda)]);

%% Draw Dual Value 
% figure
% semilogy(0:m:size(dual_val_sdca)-1,optimal_dual_val-dual_val_sdca(1:m:size(dual_val_sdca)) , 'g-*');
% hold on,semilogy(0:m:size(dual_val_spdc)-1,optimal_dual_val- dual_val_spdc(1:m:size(dual_val_spdc)), 'm-square');
% hold on,semilogy(0:m:size(dual_val_asdca)*m-1, optimal_dual_val-dual_val_asdca,'r-+');
% hold on,semilogy(0:m:size(dual_val_spdc_r8)*m-1,optimal_dual_val- dual_val_spdc_r8,'k-v');
% hold off
% 
% legend('SDCA','SPDC','ASDCA','FastSPDC');
% xlabel({'Number of passes of data'});
% ylabel({'optimal - Dual Value'});
% %axis([0 n_epoch]);
% title([loss,'   ',file_name,'    lambda = ',num2str(lambda)]);


