%   Algorithm:
%	1: SDCA
%   2: ASDCA %%ע��ASDCA����Ӧ�������ġ�
%   3: SPDC
%   4: ASPDC��SPDC_r8��

%% Set Params
clear ;
%file_name = 'a9a';
%file_name = 'ijcnn1';
file_name = 'covtype';
loss = 'smooth_hinge';
%loss = 'L2_svm';
%%%%% lambda should be relatively small in ASPDC(SPDC_r8), ot herwise call SPDC_r7 
lambda = 1e-6;%%lambdaҪС��1e-6
tol = 1e-15;
EPOCH = 15;%%epoch ����
n_epoch = EPOCH;
m = 2;
H = 2;%%ASPDC����ѭ������
%%������5����������2����������ֵ��ȡ����ASPDC(SPDC_r8)�������ʵ�ֵġ�Ϊ�˷����ASDCA�Աȣ���ȡm=5,����C++�����У�m=5
verbose = 1;

Optimal_val= 0.000000897725809;
%lambda=1e-6,covtype:0.000000897725809
%lambda=1e-7,covtype:0.000000091122588

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


%% ASPDC-i(SPDC_r8)
algorithm = 'ASPDCi';
fprintf('Algorithm: %s\n', algorithm);
n_epoch=EPOCH*m/H;
tic;
%%%% value in dual_gap_ASPDCi are absolute value 
[dual_gap_ASPDCi,primal_val_ASPDCi,dual_val_ASPDCi] = Interface(file_name, loss, algorithm, lambda, tol, n_epoch, verbose);
time = toc;
fprintf('Time: %f seconds \n', time);

%% SVRG
algorithm = 'SVRG';
fprintf('Algorithm: %s\n', algorithm);
n_epoch=EPOCH*m/H;
tic;
%%%% value in dual_gap_ASPDCi are absolute value 
[dual_gap_SVRG,primal_val_SVRG,dual_val_SVRG] = Interface(file_name, loss, algorithm, lambda, tol, n_epoch, verbose);
time = toc;
fprintf('Time: %f seconds \n', time);

%% find the optimal primal value and dual value 
%%��lambda��С��ʱ�򣬲�Ӧ����SDCAȥ��ȡ���ŵĺ���ֵ
% algorithm = 'ASPDC';
% n_epoch = 120;
% tol = 1e-20;
% fprintf('Algorithm: %s\n', algorithm);
% tic;
% [dual_gap,primal_val,dual_val] = Interface(file_name, loss, algorithm, lambda, tol, n_epoch, verbose);%only array primal_val is used to find tthe optimal prima value
% time = toc;
% optimal_primal_val = min(primal_val);
% fprintf('optimal_primal_val: %20.18f\n',optimal_primal_val);
% optimal_dual_val = max(dual_val);%%����SPDC��SDCA�Ķ�ż������ʽ�ŵ�ϸ΢��𣬵������ڶ�ż��϶���Ϊ0������SDCA��SPDC�Ķ�ż����ֵ����ȵ�
% fprintf('optimal_dual_val: %20.18f\n',optimal_dual_val);
% fprintf('Time: %f seconds \n', time);

%% Draw Sub-Optimality
figure
semilogy(3:m:size(primal_val_sdca)-1, primal_val_sdca(4:m:size(primal_val_sdca))-Optimal_val, 'g-*');%size(dual_gap_sdca)=30,ÿ����epoch��һ��
hold on,semilogy(3:m:size(primal_val_spdc)-1, primal_val_spdc(4:m:size(primal_val_spdc))-Optimal_val, 'm-square'); %size(dual_gap_spdc)=30
hold on,semilogy(3:H:size(primal_val_ASPDCi)*H-1, primal_val_ASPDCi(2:1:size(primal_val_ASPDCi))-Optimal_val,'k-v'); %size(dual_gap_spdc_r8)=15
%hold on,semilogy(0:H:size(dual_gap_asdca)*H-1, dual_gap_asdca,'r-+');
hold on,semilogy(3:m:size(primal_val_asdca)*m-1, primal_val_asdca(2:1:size(primal_val_asdca))-Optimal_val,'r-+');
hold on,semilogy(3:H:size(primal_val_SVRG)*H-1, primal_val_SVRG(2:1:size(primal_val_SVRG))-Optimal_val,'k-v'); %size(dual_gap_spdc_r8)=15
hold off

legend('SDCA','SPDC','ASPDC-s','ASDCA','SVRG', 'Location','southwest');
xlabel({'Number of passes of data'});
ylabel({'Log Loss'});

%axis([0 n_epoch]);
% if strcmp(loss,'L2_svm')
%     loss = 'L2\_svm';
% elseif strcmp(loss,'smooth_hinge')
%     loss = 'smooth\_hinge';
% end 
% title([loss,'   ',file_name,'    lambda = ',num2str(lambda)]);


%% Draw dual-gap
figure
semilogy(3:m:size(dual_gap_sdca)-1, dual_gap_sdca(4:m:size(dual_gap_sdca)), 'g-*');%size(dual_gap_sdca)=30,ÿ����epoch��һ��
hold on,semilogy(3:m:size(dual_gap_sdca)-1, dual_gap_spdc(4:m:size(dual_gap_spdc)), 'm-square'); %size(dual_gap_spdc)=30
hold on,semilogy(3:H:size(dual_gap_ASPDCi)*H-1, dual_gap_ASPDCi(2:1:size(dual_gap_ASPDCi)),'k-v'); %size(dual_gap_spdc_r8)=15
%hold on,semilogy(0:H:size(dual_gap_asdca)*H-1, dual_gap_asdca,'r-+');
hold on,semilogy(3:m:size(dual_gap_asdca)*m-1, dual_gap_asdca(2:1:size(dual_gap_asdca)),'r-+');
hold off

legend('SDCA','SPDC','ASPDC-s','ASDCA', 'Location','southwest');
xlabel({'Number of passes of data'});
ylabel({'Dual Gap'});
%% Draw Primal Value 
% figure
% semilogy(0:m:size(primal_val_sdca)-1,primal_val_sdca(1:m:size(primal_val_sdca)) - optimal_primal_val, 'g-*');
% hold on,semilogy(0:m:size(primal_val_spdc)-1, primal_val_spdc(1:m:size(primal_val_spdc)) - optimal_primal_val, 'm-square');
% hold on,semilogy(0:m:size(primal_val_asdca)*m-1, primal_val_asdca- optimal_primal_val,'r-+');
% hold on,semilogy(0:m:size(primal_val_spdc_r8)*m-1,primal_val_spdc_r8 - optimal_primal_val,'k-v');
% hold off
% 
% legend('SDCA','SPDC','ASDCA','ASPDC');
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
% legend('SDCA','SPDC','ASDCA','ASPDC');
% xlabel({'Number of passes of data'});
% ylabel({'optimal - Dual Value'});
% %axis([0 n_epoch]);
% title([loss,'   ',file_name,'    lambda = ',num2str(lambda)]);


