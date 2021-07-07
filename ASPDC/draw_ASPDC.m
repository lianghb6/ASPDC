%% draw dual gap of differet  primal dual algorithms(when lambda is small)
%   Algorithm:
%	1: SDCA
%   2: SPDC
%   3: ASDCA
%   4: ASPDCi


%% Set Params

file_name = 'a9a';
for lambda = [1e-6 5*1e-7 1e-7]
    fprintf(file_name)
    if(draw(file_name, lambda)~=0)
        fprintf('Error in call draw!')
    end
end


file_name = 'ijcnn1';
for lambda = [1e-6 5*1e-7 1e-7]
    fprintf(file_name)
    if(draw(file_name, lambda)~=0)
        fprintf('Error in call draw!')
    end
end

file_name = 'covtype';
for lambda = [1e-6 5*1e-7 1e-7]
    fprintf(file_name)
    if(draw(file_name, lambda)~=0)
        fprintf('Error in call draw!')
    end
end

% file_name = 'kddb';
% for lambda = [1e-6 1e-7,1e-8]
%     fprintf(file_name)
%     if(draw(file_name, lambda)~=0)
%         fprintf('Error in call draw!')
%     end
% end

% file_name = 'news20';
% for lambda = [1e-6 1e-7,1e-8]
%     fprintf(file_name)
%     if(draw(file_name, lambda)~=0)
%         fprintf('Error in call draw!')
%     end
% end
% 
% file_name = 'rcv1_test';
% for lambda = [1e-6 1e-7,1e-8]
%     fprintf(file_name)
%     if(draw(file_name, lambda)~=0)
%         fprintf('Error in call draw!')
%     end
% end

function output= draw(file_name, lambda)
    loss = 'smooth_hinge';
    % loss = 'L2_svm';
    %%%%% lambda should be relatively small(less than 1e-6) in ASPDC(SPDC_r8), otherwise call SPDC_r7 
 
    tol = 1e-15;
    EPOCH = 60;%%epoch 次数
    n_epoch = EPOCH;
    m = 5;
    %%具体是5倍还是其他2倍或者其他值，取决于ASPDC(SPDC_r8)中是如何实现的。为了方便和ASDCA对比，就取m=5,即在C++代码中，m=5
    verbose = 1;

%     %% SDCA
%     algorithm = 'SDCA';
%     fprintf('Algorithm: %s\n', algorithm);
%     n_epoch=EPOCH*m;%%fair comparison
%     tic;
%     [dual_gap_sdca,primal_val_sdca,dual_val_sdca] = Interface(file_name, loss, algorithm, lambda, tol, n_epoch, verbose);
%     time = toc;
%     fprintf('Time: %f seconds \n', time);

%     %% SPDC
%     algorithm = 'SPDC';
%     fprintf('Algorithm: %s\n', algorithm);
%     n_epoch=EPOCH*m;
%     tic;
%     [dual_gap_spdc,primal_val_spdc,dual_val_spdc] = Interface(file_name, loss, algorithm, lambda, tol, n_epoch, verbose);
%     time = toc;
%     fprintf('Time: %f seconds \n', time);

%     %% ASDCA
%     algorithm = 'ASDCA';
%     fprintf('Algorithm: %s\n', algorithm);
%     n_epoch=EPOCH;
%     tic;
%     [dual_gap_asdca,primal_val_asdca,dual_val_asdca] = Interface(file_name, loss, algorithm, lambda, tol, n_epoch, verbose);
%     time = toc;
%     fprintf('Time: %f seconds \n', time);

    %% ASPDC(SVRG)
    algorithm = 'SVRG';
    fprintf('Algorithm: %s\n', algorithm);
    n_epoch=EPOCH;
    tic;
    %%%% value in dual_gap_spdc_r8 are absolute value 
    [dual_gap_svrg,primal_val_svrg,dual_val_svrg] = Interface(file_name, loss, algorithm, lambda, tol, n_epoch, verbose);
    time = toc;
    fprintf('Time: %f seconds \n', time);
    
    %% ASPDC(SPDC_r8)
    algorithm = 'ASPDCi';
    fprintf('Algorithm: %s\n', algorithm);
    n_epoch=EPOCH;
    tic;
    %%%% value in dual_gap_spdc_r8 are absolute value 
    [dual_gap_spdc_r8,primal_val_spdc_r8,dual_val_spdc_r8] = Interface(file_name, loss, algorithm, lambda, tol, n_epoch, verbose);
    time = toc;
    fprintf('Time: %f seconds \n', time);

    %% find the optimal primal value and dual value 
    %%在lambda很小的时候，不应该用SDCA去获取最优的函数值
    algorithm = 'ASPDCi';
    n_epoch = 120;
    tol = 1e-20;
    fprintf('Algorithm: %s\n', algorithm);
    tic;
    [dual_gap,primal_val,dual_val] = Interface(file_name, loss, algorithm, lambda, tol, n_epoch, verbose);%only array primal_val is used to find tthe optimal prima value
    time = toc;
    optimal_primal_val_fast = min(primal_val);
    fprintf('optimal_primal_val: %20.18f\n',optimal_primal_val_fast);
    optimal_dual_val = max(dual_val);%%尽管SPDC和SDCA的对偶函数形式优点细微差别，但是由于对偶间隙最后为0，所以SDCA和SPDC的对偶函数值是相等的
    fprintf('optimal_dual_val: %20.18f\n',optimal_dual_val);
    fprintf('Time: %f seconds \n', time);
    
    
    algorithm = 'SVRG';
    n_epoch = 120;
    tol = 1e-20;
    fprintf('Algorithm: %s\n', algorithm);
    tic;
    [dual_gap,primal_val,dual_val] = Interface(file_name, loss, algorithm, lambda, tol, n_epoch, verbose);%only array primal_val is used to find tthe optimal prima value
    time = toc;
    optimal_primal_val_svrg = min(primal_val);
    fprintf('optimal_primal_val: %20.18f\n',optimal_primal_val_svrg);
    optimal_dual_val = max(dual_val);%%尽管SPDC和SDCA的对偶函数形式优点细微差别，但是由于对偶间隙最后为0，所以SDCA和SPDC的对偶函数值是相等的
    fprintf('optimal_dual_val: %20.18f\n',optimal_dual_val);
    fprintf('Time: %f seconds \n', time);

%     %% Draw Dual_gap
%     figure
%     hold on,semilogy(0:m:size(dual_gap_sdca)-1, dual_gap_sdca(1:m:size(dual_gap_sdca)), 'g-*');
%     hold on,semilogy(0:m:size(dual_gap_spdc)-1, dual_gap_spdc(1:m:size(dual_gap_spdc)), 'm-square');
%     hold on,semilogy(0:m:size(dual_gap_asdca)*m-1, dual_gap_asdca,'r-+');
%     hold on,semilogy(0:m:size(dual_gap_svrg)-1, dual_gap_svrg(1:m:size(dual_gap_svrg)), '--*');
%     hold on,semilogy(0:m:size(dual_gap_spdc_r8)*m-1, dual_gap_spdc_r8,'k-v');
%     hold off
%     legend('SDCA','SPDC','ASDCA','SVRG', 'ASPDC');
%     xlabel({'Number of passes of data'});
%     ylabel({'Dual Gap'});
%     %axis([0 n_epoch]);
%     if strcmp(loss,'L2_svm')
%         loss = 'L2\_svm';
%     elseif strcmp(loss,'smooth_hinge')
%         loss = 'smooth\_hinge';
%     end 
%     title([loss,'   ',file_name,'    lambda = ',num2str(lambda)]);

    %% Draw Primal Value 
    figure
    semilogy(0:m:size(primal_val_svrg)-1,primal_val_svrg(1:m:size(primal_val_svrg))-optimal_primal_val_svrg, 'g-*');
    hold on,semilogy(0:m:size(primal_val_spdc_r8)-1, primal_val_spdc_r8(1:m:size(primal_val_spdc_r8))-optimal_primal_val_fast, 'm-square');
%     hold on,semilogy(0:m:size(primal_val_asdca)*m-1, primal_val_asdca- optimal_primal_val,'r-+');
%     hold on,semilogy(0:m:size(primal_val_spdc_r8)*m-1,primal_val_spdc_r8 - optimal_primal_val,'k-v');
    hold off

    legend('SVRG','ASPDC');
    xlabel({'Number of passes of data'});
    ylabel({'Primal Value - optimal '});
    %axis([0 n_epoch]);
    title(['smooth\_hinge','   ',file_name,'    lambda = ',num2str(lambda)]);

%     %% Draw Dual Value 
%     figure
%     semilogy(0:m:size(dual_val_sdca)-1,optimal_dual_val-dual_val_sdca(1:m:size(dual_val_sdca)) , 'g-*');
%     hold on,semilogy(0:m:size(dual_val_spdc)-1,optimal_dual_val- dual_val_spdc(1:m:size(dual_val_spdc)), 'm-square');
%     hold on,semilogy(0:m:size(dual_val_asdca)*m-1, optimal_dual_val-dual_val_asdca,'r-+');
%     hold on,semilogy(0:m:size(dual_val_spdc_r8)*m-1,optimal_dual_val- dual_val_spdc_r8,'k-v');
%     hold off
% 
%     legend('SDCA','SPDC','ASDCA','ASPDC');
%     xlabel({'Number of passes of data'});
%     ylabel({'optimal - Dual Value'});
%     %axis([0 n_epoch]);
%     title([loss,'   ',file_name,'    lambda = ',num2str(lambda)]);
%     write_file = [loss,'_',file_name,'_',num2str(lambda)];
%     save write_file;
    output = 0;
end



