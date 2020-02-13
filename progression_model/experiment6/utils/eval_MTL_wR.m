function [wR,Placeholder] = eval_MTL_wR(Y,X,W)
%加权关系系数的构造
%任务数=5，设置初始值为0，设置样本总数定义为0
    task_num = length(X);
    wR = 0;
    total_sample = 0;
    Placeholder = zeros(task_num,1);%占位符，为了统一函数接口，没有实际意义
    
    for i = 1: task_num
        y_pred = X{i} * W(:, i);
        corr = corrcoef(Y{i},y_pred);%计算出相关系数矩阵
        wR = wR + corr(1,2) * length(y_pred);
        total_sample = total_sample + length(y_pred);
    end
    wR = wR / total_sample;
end



