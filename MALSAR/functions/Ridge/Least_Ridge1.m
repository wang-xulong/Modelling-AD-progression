%% FUNCTION Least_Ridge
% Sparse Structure-Regularized Learning with Least Squares Loss.
%
%% OBJECTIVE
% argmin_W { sum_i^t (0.5 * norm (Y{i} - X{i}' * W(:, i))^2)
%            + rho1 * \|W\|_1 + opts.rho_L2 * \|W\|_F^2}
%
%% INPUT
%   X: {n * d} * t - input matrix
%   Y: {n * 1} * t - output matrix
%   rho1: sprasity controlling parameter
%   opts.rho_L2: L2-norm regularization parameter
%
%% OUTPUT
%   W: model: d * t
%   funcVal: function value vector.
%
%% Related functions
%   Logistic_Lasso, init_opts
%
%% Code starts here
%w is the matrix weight
%funcVal is the function value of the last iteration of this model
%rho1 regularizaion parameter
%opts 是初始化的默认参数设置，用来指定一些算法所需的默认的参数值，算法优化迭代最大次数，迭代停止条件等等
function [W, funcVal] = Least_Ridge(X, Y, rho1, opts)

if nargin <3 %调用该函数时，所给的参数不够，则提示有误，并退出
    error('\n Inputs: X, Y, abd rho1 should be specified!\n');
end

% cell X array transpose ;every cell is n * d; when tansposed:
% row:feature number (d)
% column:sample number (n)
%X = multi_transpose(X); 
if nargin <4
    opts = [];
end

opts=init_opts(opts);                                           % 初始化默认参数选项

if isfield(opts, 'rho_L2')                                        % 是否opts中要求添加L2范式的正则化参数
    rho_L2 = opts.rho_L2;
else
    rho_L2 = 0;                                                     % 本算法并没有要求添加，因此rho_L2 = 0; 
end


task_num  = length (X);                                       % X contains many matrix(n x d)；the number of task number is length(X)

dimension = size(X{1}, 2);                                   % every cell's demension is d

funcVal = [];                                                         % store function value


% % initialize a starting point
% if opts.init==2
%     W0 = zeros(dimension, task_num);                % 本算法中，opts.init==2，初始化权重设置为零矩阵
% elseif opts.init == 0
%     W0 = W0_prep;
% else
%     if isfield(opts,'W0')
%         W0=opts.W0;
%         if (nnz(size(W0)-[dimension, task_num]))
%             error('\n Check the input .W0');
%         end
%     else
%         W0=W0_prep;
%     end
% end

W1 = zeros(dimension, task_num);
I = eye(dimension);
I(1,1) = 0;
for t_idx = 1: task_num
    W_tmp = (X{t_idx}' * X{t_idx} + rho1 * I );
    W1(: , t_idx) = W_tmp \ X{t_idx}' * Y{t_idx};
end
W = W1;

end