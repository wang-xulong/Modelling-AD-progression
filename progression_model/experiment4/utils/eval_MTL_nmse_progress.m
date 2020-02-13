function [nmse,rMSE_t] = eval_MTL_rmse_progress (Y, X, W)
%% FUNCTION eval_MTL_mse
%   computation of root mean squared error given a specific model.
%   the value is the lower the better.
%   
%% FORMULATION
%   
% multi-task rmse = sum_t (rmse(t) * N_t) / sum_t N_t
%
%  where 
%     rmse(t) = sqrt(sum((Yt_pred - Y{t})^2))/ N_t)
%     Yt_pred = X{t} * W(:, t)
%     N_t     = length(Y{t})
%
%% INPUT
%   X: {n * d} * t - input matrix
%   Y: {n * 1} * t - output matrix
%   percent: percentage of the splitting range (0, 1)
%
%% OUTPUT
%   X_sel: the split of X that has the specifid percent of samples 
%   Y_sel: the split of Y that has the specifid percent of samples 
%   X_res: the split of X that has the 1-percent of samples 
%   Y_res: the split of Y that has the 1-percent of samples 
%   selIdx: the selection index of for X_sel and Y_sel for each task
%
    task_num = length(X);
    rMSE_t = zeros(task_num,1);
   
    nmse = 0;
    
    total_sample = 0;
    for t = 1: task_num
        y_pred = X{t} * W(:, t);
        rMSE_t(t) = sum((y_pred - Y{t}).^2);                                  %计算当前任务的MSE
        Yt_variance = var(Y{t});                                              %计算当前任务的variance
        nmse = nmse + (rMSE_t(t) / Yt_variance);                              %累计当前任务的mse/当前任务的v ariance
        rMSE_t(t) = sqrt(rMSE_t(t)/length(y_pred));                           %计算当前任务的rMSE
        %rmse = rmse + sqrt(sum((y_pred - Y{t}).^2)/length(y_pred)) * length(y_pred);
        total_sample = total_sample + length(y_pred);
    end
    nmse = nmse/total_sample;
end
