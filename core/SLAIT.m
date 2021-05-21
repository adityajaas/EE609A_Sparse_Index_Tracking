% EE609A Term project Spring 2020-21
%% =====Paper 1 SLAIT implementation====
% Sparse Portfolios for High-Dimensional Financial Index Tracking
% Paper Authors - Konstantinos Benidis , Yiyong Feng, and Daniel P. Palomar
% Code by - Aditya Singh, Varun Rajesh Gadre

classdef SLAIT
    properties
        X
        r_b
        w0
        lambda
        nitr
        cost_history
        convergence_history
        print
    end
    methods
        % Constructor
        function obj = SLAIT(X, r_b, numitr, lambda, print)
            obj.X = X; obj.r_b = r_b; obj.nitr = numitr;
            obj.lambda = lambda; obj.cost_history = [];
            obj.convergence_history = [];
            obj.w0 = ones(size(X,2),1)/size(X,2);
            if nargin == 5
                obj.print = print;
            else
                obj.print = "Yes";
            end
        end
        
        % d_{p,gamma}(w^{k}) function. Slope of the majorizing line. Eq (3)
        % in our project report
        function y = d(obj, p, gamma, w)
            k = log(1+ gamma/p);
            if isscalar(w) == 1
                y = 1/(k * (p + w));
            else
                y = zeros(length(w),1);
                y = 1./(k .* (p + w));
            end
            return;
        end
        
        % Calculate q of Eq (9) in our project report
        function y = q(obj, L1, eig_max, nstocks, ...
                ndays, X, r_b, lambda, d_vec, w)
            y = (2*(L1 - eig_max * eye(nstocks)) * w + ...
                lambda * d_vec - (2 * X'*r_b)/ndays)/eig_max;
            return;
        end
        
        % Calcuate lagrangian mu value using bisection
        % Method described in Appendex 2 of Benidis paper. Brief in Eq
        % (9-10) of our project report
        function w = optimize_mu(obj, q, u)
            n = length(q);
            w = zeros(n,1);
            [q_sort I] = sort(q);
            low = 1; high = n;
            while low <= high
                mid = floor((low + high)/2);
                mu = -(sum(q_sort(1:mid)) + 2)/mid;
                test1 = (mu + q_sort(mid) < 0);
                if mid < n
                    test2 = (mu + q_sort(mid+1) >= 0);
                else
                    test2 = 1;
                end
                if test1 && test2
                    break;
                elseif test1 && ~test2
                    low = mid + 1;
                else
                    high = mid - 1;
                end
            end
            if all(-(mu + q_sort(1:mid))/2 <= u)
                w(I(1:mid)) = -(mu + q_sort(1:mid))/2;
            else
                flag1 = 0; flag2 = 0;
                k = mid;
                while 1
                    low1 = 0; high1 = k - 1;
                    while low1 <= high1
                        mid1 = floor((low1+high1)/2);
                        mu = -(sum(q_sort(mid1+1:k) - 2*mid1*u + 2)...
                            /(k-mid1));
                        if mid1 ~= 0
                            test1 = (mu + q_sort(mid1) <= -2*u);
                        else
                            test1 = 1;
                        end

                        test2 = (-2*u < mu + q_sort(mid1+1)) && ...
                            (mu + q_sort(k) < 0);

                        if k < n
                            test3 = (mu + q_sort(k+1) >= 0);
                        else
                            test3 = 1;
                        end
                        if test1 && test2 && test3
                            flag1 = 1;
                            break;
                        elseif test1 && ~test2
                            low1 = mid1 + 1;
                        else
                            high1 = mid1 - 1;
                        end
                    end

                    if flag1
                        break
                    else
                        k = k + 1;
                    end
                    if k > n
                        flag2 = 1;
                        break;
                    end
                end

        %         flag1
        %         flag2
                if flag2
                    w(I(1:ceil(1/u))) = u;
                    return;
                else
                    w(I(1:mid1)) = u;
                    w(I(mid1+1:k)) = -(mu + q_sort(mid1+1:k))/2;
                    return;
                end
            end
        end
        
        % Main Algorithm
        function [w, conv_hist, cost_hist] = optimweight(obj)
            % lambda: regularizer, X: return of stocks in indices, r_b:
            % index return
            lambda = obj.lambda; X = obj.X; r_b = obj.r_b;
            % ndays = D, number of days of data, nstocks, = N, number of
            % stocks in index
            ndays = size(obj.X,1); nstocks = size(obj.X,2);
            L1 = X'*X/ndays;
            eig_max = max(eig(L1));
            print = obj.print;
            % No stock weight should be greater than 5 percent
            u = 0.05; 
            for i = 1:obj.nitr
                d_vec = d(obj, 1e-4, u, obj.w0);
                q1 = q(obj, L1, eig_max, nstocks, ...
                    ndays, X, r_b, lambda, d_vec, obj.w0);
                w = optimize_mu(obj, q1, u);
                w(w<1e-5) = 0;
                obj.cost_history(i) = norm(X*w - r_b, 2)/ndays;
                obj.convergence_history(i) = abs(norm(w - ...
                    obj.w0, 2));
                if  obj.convergence_history(i) < 1e-9
                    break;
                end
                obj.w0 = w;
            end
            conv_hist = obj.convergence_history;
            cost_hist = obj.cost_history;
            if print ~= "No"
                fprintf(['SLAIT Converged in ', num2str(i), ' iterations\n']);
                fprintf(['Sparsity of w: ', num2str(sum(w > 1e-4)),...
                    '/50 nonzero terms\n']);
            end
        end
    end
end