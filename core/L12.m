% EE609A Term project Spring 2020-21
%% =====Paper 2 L1/2 implementation====
% Sparse Index Tracking Based On L1/2 Model And Algorithm
% Paper Authors - Xu Fengmin , Zongben Xu, and Honggang Xue
% Code by - Aditya Singh, Varun Rajesh Gadre

classdef L12
    properties
        w0
        X
        r_b
        K
        nitr
        lambda
        cost_history
        convergence_history
    end
    methods
        function obj = L12(X, r_b, numitr, lambda,K)
            obj.X = X;
            obj.r_b = r_b;
            obj.nitr = numitr;
            obj.lambda = lambda; % Regularizer
            % weight vector initialized equally
            obj.K = K;
            obj.w0 = ones(size(X,2),1)/size(X,2);
            obj.convergence_history = [];
            obj.cost_history = [];
        end
        
        
        function [w] = optimweight(obj)
            % lambda: regularization param
            % L12_X: Return of stocks listen in index
            % r_b: Return of index
            % ndays: Number of days in our stock data history
            % nstocks: Number of stocks in index
            % epsilon: Used to calculate mu_0 in Eq (16) of term report
            % K: Number of final postive weights required. Sparsity controlling param
            % X_norm: Frobenius normalized stock return matrix X
            lambda = obj.lambda; L12_X = obj.X; r_b = obj.r_b;
            ndays = size(obj.X,1); nstocks = size(obj.X,2);
            epsilon = 0.5;
            K = obj.K;
            X_norm = power(norm(L12_X,'fro'),2);
            mu_not = (1-epsilon)/X_norm;
            wn = obj.w0;
          
            
            for i=1:obj.nitr
                B = wn+((mu_not)*L12_X.'*(r_b-L12_X*wn));
                B_k = maxk(abs(B),K+1);
                B_k = B_k(length(B_k));
                
                % Half thresholding algorithm described in Eq (15-16) in
                % term report
                if i~=1
                    lambda = min(lambda,1.088662*X_norm*power(B_k,1.5));
                end
                
                H_input = wn+(mu_not*L12_X.'*(r_b-L12_X*wn));
                for i=1:length(H_input)
                    if power(lambda*mu_not,2/3)*power(54,1/3)/4 < abs(H_input(i))
                        theta = (2/3)*acos((lambda*mu_not*power(3,1.5))/...
                            (8*power(abs(H_input(i)),1.5)));
                        wn(i) = (2/3)*abs(H_input(i))*(1+ cos(((2*pi)/3)-theta));
                    else
                        wn(i) = 0;
                    end
                end
               
            end
            w = wn;
            % Compute S(w) as in step 5 of Algo 3 in term report.
            indices = find(w);
            new_X = L12_X(:,indices);
            
            H = new_X.'*new_X;
            f = -2*new_X.'*r_b;
            lb = 0.001*ones(1,length(indices));
            ub = 0.2*ones(1,length(indices));
            Aeq = ones(1,length(indices));
            beq = 1;
            options = optimoptions('quadprog',...
                    'Display','Off');
            % Matlab quadprog to optimize QP
            x = quadprog(H, f,[],[], Aeq, beq, lb, ub,[],options);
            for i=1:length(indices)
                w(indices(i)) = x(i);
            end
                    
            
        
        
        end
        
    end
end
