% EE609A Term project Spring 2020-21
%% =====Paper 1 LAIT implementation====
% Sparse Portfolios for High-Dimensional Financial Index Tracking
% Paper Authors - Konstantinos Benidis , Yiyong Feng, and Daniel P. Palomar
% Code by - Aditya Singh, Varun Rajesh Gadre

classdef LAIT
    properties
        w0
        X
        r_b
        nitr
        lambda
        cost_history
        convergence_history
        print
    end
    methods
        function obj = LAIT(X, r_b, numitr, lambda, print)
            obj.X = X;
            obj.r_b = r_b;
            obj.nitr = numitr;
            obj.lambda = lambda; % Regularizer
            % weight vector initialized equally
            obj.w0 = ones(size(X,2),1)/size(X,2);
            obj.convergence_history = [];
            obj.cost_history = [];
            if nargin == 5
                obj.print = print;
            else
                obj.print = "Yes";
            end
        end
        
        function y = d(obj, p, gamma, w)
            k = log(1+ gamma/p);
            if isscalar(w) == 1
                y = 1/(k * (p + w));
            else
                y = zeros('like', w);
                for i = 1:size(w)
                    y(i,1) = 1/(k * (p + w(i,1)));
                end
            end
        end
        
        function [w, conv_hist, cost_hist] = optimweight(obj)
            % lambda: regularizer, X: return of stocks in indices, r_b:
            % index return
            lambda = obj.lambda; X = obj.X; r_b = obj.r_b;
            % ndays = D, number of days of data, nstocks, = N, number of
            % stocks in index
            ndays = size(obj.X,1); nstocks = size(obj.X,2);
            for i = 1:obj.nitr
                d_vec = obj.d(1e-2, 0.05, obj.w0);
                H = (2/ndays)*(X'*X);
                f = (lambda*d_vec)-(2/ndays)*(X'*r_b);
                Aeq = ones(1,nstocks); beq = 1;
                % ub: Upper bound, No stock weight should be greater than 5 percent
                lb = 0* ones(nstocks, 1); 
                ub = 0.05*ones(nstocks, 1);
                options = optimoptions('quadprog',...
                    'Display','Off');
                % Matlab quadprog to optimize QP
                w = quadprog(H, f,[],[], Aeq, beq,...
                    lb, ub,[],options);
                % Insignificant holdings reduced to zero.
                w(w<1e-5) = 0;
                obj.cost_history(i) = norm(X*w - r_b, 2)/ndays;
                obj.convergence_history(i) = abs(norm(w -...
                    obj.w0, 2));
                if  obj.convergence_history(i) < 1e-9
                    break;
                end
                obj.w0 = w;
            end
            conv_hist = obj.convergence_history;
            cost_hist = obj.cost_history;
            % If "No" doesnt print convergence information. Since printing
            % is one of the slowest step in matlab, kept at "No" for large
            % iterations.
            if obj.print ~= "No"
                fprintf(['LAIT Converged in ', num2str(i),...
                    ' iterations\n']);
                fprintf(['Sparsity of w: ',...
                    num2str(sum(w > 1e-4)),...
                    '/50 nonzero terms\n']);
            end
        end
    end
end
        