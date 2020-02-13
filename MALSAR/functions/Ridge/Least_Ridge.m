%% FUNCTION Least_Lasso
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
%% Related papers
%   [1] Tibshirani, J. Regression shrinkage and selection via
%   the Lasso, Journal of the Royal Statistical Society. Series B 1996
%
%% Related functions
%   Logistic_Lasso, init_opts
%
%% Code starts here
%w is the matrix weight
%funcVal is the function value of the last iteration of this model
%rho1 regularizaion parameter
%opts 是初始化的默认参数设置，用来指定一些算法所需的默认的参数值，算法优化迭代最大次数，迭代停止条件等等
function [W, funcVal] = Least_Ridge1(X, Y, rho1, opts)

if nargin <3 %调用该函数时，所给的参数不够，则提示有误，并退出
    error('\n Inputs: X, Y, abd rho1 should be specified!\n');
end

% cell X array transpose ;every cell is n * d; when tansposed:
% row:feature number (d)
% column:sample number (n)
X = multi_transpose(X); 
if nargin <4
    opts = [];
end

opts=init_opts(opts);                                           % 初始化默认参数选项

if isfield(opts, 'rho_L2')                                        % 是否opts中要求添加L2范式的正则化参数
    rho_L2 = opts.rho_L2;
else
    rho_L2 = 0;                                                     % 本算法并没有要求添加，因此rho_L2 = 0; 
end

rho_L2 = rho1;
task_num  = length (X);                                       % X contains many matrix(n x d)；the number of task number is length(X)

dimension = size(X{1}, 1);                                   % every cell's demension is d

funcVal = [];                                                         % store function value

%create cell to store result in every task
XY = cell(task_num, 1);

W0_prep = [];%

% initialize a weight
% every XY{t_idx} is the result of every task(dxn  x   nx1)
% put the result of every task(dxn  x   nx1) into W0_prep
% W0_prep为每个任务中特征值与标签值的乘积，即：X*Y
% W0_prep既是初始化模型权重的一种方式，同时为后面矩阵计算提供便利，因为同样使用到了X*Y的结果
for t_idx = 1: task_num                                        % 每个任务依次遍历
    XY{t_idx} = X{t_idx}*Y{t_idx};                          %  对应 X*Y
    W0_prep = cat(2, W0_prep, XY{t_idx});          %  将每个任务的X*Y的结果存储到W0_prep中
end

% initialize a starting point
if opts.init==2
    W0 = zeros(dimension, task_num);                % 本算法中，opts.init==2，初始化权重设置为零矩阵
elseif opts.init == 0
    W0 = W0_prep;
else
    if isfield(opts,'W0')
        W0=opts.W0;
        if (nnz(size(W0)-[dimension, task_num]))
            error('\n Check the input .W0');
        end
    else
        W0=W0_prep;
    end
end


bFlag=0;                                                             % this flag tests whether the gradient step only changes a little

%Wz和Wz_old记录上一次的模型权重和上上次的模型权重，与当前模型权重Ws有关
Wz= W0;
Wz_old = W0;
%t和t_old记录当前模型权重的调整参数，与模型参数alpha有关
t = 1;
t_old = 0;


iter = 0;                                                               % 当前迭代次数
gamma = 1;                                                        % gamma为优化目标函数过程中用到的超参数，用于搜索优化过程的步长、计算出目标函数的软阈值等
gamma_inc = 2;                                                 % gamma_inc为每次调整的gamma的力度，gamma = gamma * gamma_inc;

while iter < opts.maxIter                                     % 规定最大迭代次数为100次 ，外部函数实参定义，opts.maxIter  = 100 
    alpha = (t_old - 1) /t;                                       % alpha为调节模型权重Ws在每次迭代过程中对上次Ws和上上次Ws的综合参考度
    
    Ws = (1 + alpha) * Wz - alpha * Wz_old;        % Ws在每次迭代过程中参考上次Ws和上上次Ws

    % compute function value and gradients of the search point
    gWs  = gradVal_eval(Ws);                              % gWs是计算Ws在RSS部分的梯度值（不加任何正则化的损失函数对应的梯度值）
    Fs   = funVal_eval  (Ws);                                % Fs是RSS(残差平方和)的损失函数值
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 本部分while true 的目的是确定出满足L1范式约束下的模型权重矩阵
    % 本部分的方法是透??普希茨常?(Lipschitz Constant)来确定一个上界值（即步长：gamma）
    % Lipschitz Constant的核心思想是是存在一?常? L，?函?上任意???的斜率??值不大於
    % ??常?，斜率被限制在这个上界内，此时只要选择步长为1/gamma 即可保证函数是收敛的
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    while true
        % 本算法是通过使用近端梯度下降法来迭代求解的
        % l1_projection为计算出当前模型权重W（不含L1约束下）在L1约束空间中的投影；l1_projection可以看做是软阈值操作
        % Wzp是经过L1约束空间中投影的权重值；其sum值为l1c_wzp
       % [Wzp, l1c_wzp] = l1_projection(Ws - gWs/gamma, 2 * rho1 / gamma);
        Wzp = Ws - gWs/gamma;
        % 计算出Wzp的残差平方和
        Fzp = funVal_eval  (Wzp);
        
        % delta_Wzp为L1约束空间中投影的权重值Wzp与未经正则化约束的模型权值Ws,delta_Wzp为两者之间的变化量
        delta_Wzp = Wzp - Ws;
        % 求出其F范式
        r_sum = norm(delta_Wzp, 'fro')^2;
        % Fzp_gamma来计算损失函数在Wzp处的泰勒二阶近似展开式的结果，
        % 用于判断当前的gamma步长是否满足Lipschitz Constantd 条件
        Fzp_gamma = Fs + sum(sum(delta_Wzp .* gWs)) + gamma/2 * sum(sum(delta_Wzp.*delta_Wzp));
        
        if (r_sum <=1e-20)                                     % 如果r_sum很小了，说明约束前后模型权重值变化不大，可以停止迭代！
            bFlag=1;                                                 % this shows that, the gradient step makes little improvement
            break;
        end
        
        if (Fzp <= Fzp_gamma)                              % 判断是否满足Lipschitz Constant 条件
            break;
        else
            gamma = gamma * gamma_inc;            % 不满足条件，调整步长，gamma_inc == 2，此过程中，gamma会逐渐增长，但步长1/gamma会逐渐减小
        end
    end
    % while true循环结束，此时权重矩阵在迭代中不再变化或者已经找到满足满足Lipschitz Constant
    % 条件的步长，在这种情况下，可以保证函数是收敛的
    Wz_old = Wz;                                                 % 记录下上一次的模型权重矩阵
    Wz = Wzp;                                                      % 将本次满足L1约束的模型权重矩阵保存到Wz
    
    funcVal = cat(1, funcVal, Fzp);% 将本此结果损失函数的值Fzp + rho1 * l1c_wzp保存在funcVal中
    
    if (bFlag)                                                         % 权值已经变化很小，更新不动了，那么可以退出迭代
        % fprintf('\n The program terminates as the gradient step changes the solution very small.');
        break;
    end
    
    % test stop condition.
    switch(opts.tFlag)                                           % 迭代终止的条件，在本模型的算法里默认是选择case 1的情况
        case 0
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <= opts.tol)
                    break;
                end
            end
        case 1
            if iter>=2
                % 最后两次迭代的损失值的变化量已经小于等于倒数第二次迭代的数值乘以一个很小的数 opts.tol
                if (abs( funcVal(end) - funcVal(end-1) ) <=...
                        opts.tol* funcVal(end-1))
                    break;
                end
            end
        case 2
            if ( funcVal(end)<= opts.tol)
                break;
            end
        case 3
            if iter>=opts.maxIter
                break;
            end
    end
    
    iter = iter + 1;                                                   % 迭代次数+1
    t_old = t;                                                          % 保存下旧的变量t
    t = 0.5 * (1 + (1+ 4 * t^2)^0.5);                         % t 增长
    
end

W = Wzp;


% private functions

%     function [z, l1_comp_val] = l1_projection (v, beta)%v是更新后的权重W的值
%         %l1_projection计算软阈值(soft thresholding) 
%         % this projection calculates    
%         % argmin_z = \|z-v\|_2^2 + beta \|z\|_1
%         % z: solution
%         % l1_comp_val: value of l1 component (\|z\|_1)
%         z = sign(v).*max(0,abs(v)- beta/2);
%          
%         l1_comp_val = sum(sum(abs(z)));
%     end

    function [grad_W] = gradVal_eval(W)%得到RSS部分的偏导(梯度)值
        grad_W = [];
        for t_ii = 1:task_num
            XWi = X{t_ii}' * W(:,t_ii);
            XTXWi = X{t_ii}* XWi;
            grad_W = cat(2, grad_W, XTXWi - XY{t_ii});%得到RSS部分的偏导(梯度)值：X*  X' * w - X * Y
        end
        grad_W = grad_W + rho_L2 * 2 * W;%由于是RSS部分，所以这里rho_L2的值为0
    end

    function [funcVal] = funVal_eval (W)%得到RSS部分的损失函数值
        funcVal = 0;
        for i = 1: task_num
            funcVal = funcVal + 0.5 * norm (Y{i} - X{i}' * W(:, i), 'fro')^2;
        end
        funcVal = funcVal + rho_L2 * norm(W, 'fro')^2;
    end


end