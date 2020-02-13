function [final_performance, final_performance_t,arr_wR_cFSGL] = test_cFSGL()
%function [final_performance, final_performance_t] = test_cFSGL ()
%% SCRIPT test_script.m 
%   Multi-task learning training/testing example. This example illustrates
%   how to perform split data into training part and testing part, and how
%   to use training data to build prediction model (via cross validation).
%   
%% LICENSE
%   This program is free software: you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation, either version 3 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You should have received a copy of the GNU General Public License
%   along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%   Copyright (C) 2011 - 2012 Jiayu Zhou and Jieping Ye 
%
%% Related functions
%   mtSplitPerc, CrossValidation1Param, Least_Trace

addpath('../../MALSAR/functions/progression_model/cFSGL/');
addpath('../../MALSAR/utils/');
addpath('../../MALSAR/c_files/flsa');


load_data  = csvread('./MTL_data.csv',1);%สตั้9
task_num = 5;
X = cell([1,task_num]);
Y = cell([1,task_num]);

for i = 1:task_num
    X{1,i} = load_data(: , 1+1 : size(load_data,2) - task_num);
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
 eval_func_str = 'eval_MTL_rmse';
  higher_better = false;  % mse is lower the better.

% cross validation fold
cv_fold = 5;

% optimization options
opts = [];
opts.maxIter = 100;     
      
% model parameter range    
param_range = [0.001 0.01 0.1 1 10 100 1000 10000];         
      
fprintf('Perform model selection via cross validation: \n')
[ perform_mat,best_params] = progressModelCrossValidation1Param...
     ( X_tr, Y_tr, 'Least_CFGLasso', opts, param_range, cv_fold, eval_func_str, higher_better);


%disp(perform_mat) % show the performance for each parameter.

% build model using the optimal parameter 
W = Least_CFGLasso(X_tr, Y_tr, best_params(1,1),best_params(2,1),best_params(3,1), opts);

% show final performance
%eval_func = str2func(eval_func_str);
[final_performance, final_performance_t]= eval_MTL_rmse_progress(Y_te, X_te, W); 
arr_wR_cFSGL = eval_MTL_wR(Y_te, X_te, W);
%fprintf('Performance on test data: %.4f\n', final_performance);