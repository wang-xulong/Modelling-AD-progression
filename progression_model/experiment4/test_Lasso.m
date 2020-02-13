function [final_performance, final_performance_t,arr_wR_Lasso ] = test_Lasso ()
 

%clear; clc;
%clear all;
addpath('../../MALSAR/functions/Lasso/');
addpath('../../MALSAR/utils/');


load_data  = csvread('./data_MMSE.csv',1);

task_num = 5;
X = cell([1,task_num]);
Y = cell([1,task_num]);

for i = 1:task_num
    X{1,i} = load_data(: , 1+1 : size(load_data,2)- task_num);
    Y{1,i} = load_data(: ,  size(load_data,2) - task_num + i);
end

% preprocessing data
for t = 1: length(X)
    X{t} = zscore(X{t});                  % normalization
    X{t} = [X{t} ones(size(X{t}, 1), 1)]; % add bias. 
end

% split data into training and testing.
training_percent = 0.9;
[X_tr, Y_tr, X_te, Y_te] = mtSplitPerc(X, Y, training_percent);

% the function used for evaluation.
 eval_func_str = 'eval_MTL_nmse_progress';
 higher_better = false;  % mse is lower the better.

% cross validation fold
cv_fold = 5;

% optimization options
opts = [];
opts.maxIter = 1000;

% model parameter range
param_range = [0.001 0.01 0.1 1 10 100 1000 10000];

fprintf('Perform model selection via cross validation: \n')
[ best_param, perform_mat] = CrossValidation1Param...
    ( X_tr, Y_tr, 'Least_Lasso', opts, param_range, cv_fold, eval_func_str, higher_better);
% [ best_param, perform_mat] = CrossValidation1Param...
%     ( X_tr, Y_tr, 'Least_Lasso', opts, param_range, cv_fold, eval_func_str, higher_better);

%disp(perform_mat) % show the performance for each parameter.

% build model using the optimal parameter 
W = Least_Lasso(X_tr, Y_tr, best_param, opts);

% show final performance
%eval_func = str2func(eval_func_str);
[final_performance, final_performance_t]= eval_MTL_nmse_progress(Y_te, X_te, W); 
arr_wR_Lasso = eval_MTL_wR(Y_te, X_te, W); 
%fprintf('Performance on test data: %.4f\n', final_performance);