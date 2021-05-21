% EE609A Term project Spring 2020-21
%% =====Paper 3  implementation====
% DIVERSITY AND SPARSITY: A NEW PERSPECTIVE ON INDEX TRACKING
% Paper Authors - Yu Zheng, Timothy M. Hospedales, Yongxin Yang
% Code by - Aditya Singh, Varun Rajesh Gadre

classdef sparseAndDiverse
    properties
        X
        y
        Z
        distMeasType
        lambda1
        lambda2
        k
        method
        h_norm
        h_labels
    end
    methods
        
        function obj = sparseAndDiverse(stocks_return, ...
                index_return, industry_onehot, distMeasType, ...
                lambda1, lambda2, k, method)
            if nargin == 8
                obj.X = stocks_return;
                obj.y = index_return;
                obj.distMeasType = distMeasType;
                obj.lambda1 = lambda1;
                obj.lambda2 = lambda2;
                obj.method = method;
                if obj.method == "sector"
                    % Use NSE sectors
                    obj.Z = double(industry_onehot);
                    obj.k = size(obj.Z,1);
                elseif obj.method == "cluster"
                    % Calcuate sectors from spectral clustering
                    if distMeasType == "euclidean"
                        % Use euclidean distance to get affinity matrix
                        S = squareform(pdist(obj.X'));
                    else
                        % Use Spearman rank correlation to get affinity
                        % matrix
                        
                        % find rank correlation between columns of X
                        rho = corrcoef(obj.X); 
                        rho(isnan(rho))=0;
                        % Calculate Spearman distance
                        d_xixj = sqrt(2*(1-rho));
                        % sigma as median of non diagonal terms of d_xixj
                        sigma = median(d_xixj(logical(ones(size(d_xixj))...
                            - eye(size(d_xixj)))));
                        % Off diagonal term distance
                        S = exp(-(d_xixj.^2)./(sigma^2));
                        % Diagonal term distance set to zero
                        for i = 1:size(S,1)
                            S(i,i) = 0;
                        end
                    end
                    assert(sum((S<0),'all') == 0);
                    % Capital lambda is normalization term of laplacian
                    LAMBDA = diag(sum(S'));
                    L =  ((LAMBDA)^-0.5) * S * ((LAMBDA)^-0.5);
                    [V D] = eig(L); % V = eigenvector,D = diag(eigenval)
                    D = D(eye(size(D))== 1); % Diagonal matrix to vector
                    % Sort eigen values in decreasing order
                    [D permut] = sort(D, 'descend');
                    obj.k = k;
                    H = zeros(k, length(D));
                    % Create H from top k eigenvectors of Laplacian as rows
                    for i =1:k
                        H(i,:) = V(:,permut(i))';
                    end
                    norm_deno = sqrt(sum(H.^2));
                    H_norm = (H./norm_deno)'; % Each row is instance
                    % Run k means to get k sectors as clusters. Square
                    % euclidean distance used for calculating distance
                    idx = kmeans(real(H_norm), k, 'Distance','sqeuclidean');
                    obj.h_norm = H_norm;
                    obj.h_labels = idx;
                    obj.Z = (idx == 1:k)';
                end
            else
                fprintf('Numbers of arguments not equal to 7\n');
            end
        end

        function [w h_norm h_labels] = optimweight(obj)
            n = size(obj.X,2); % nstocks
            one_w = ones(n,1); % n x 1 Vector of all 1
            one_k = ones(obj.k, 1); % k x 1 Vector of all 1
            % Get h_normalized and its labels given by kmeans
            h_norm = obj.h_norm; h_labels = obj.h_labels;
            X = obj.X; Z = obj.Z; Y = obj.y;
            lambda1 = obj.lambda1; lambda2 = obj.lambda2;
            ndays = size(obj.X,1); nstocks = size(obj.X,2);
            % Use matlab quadprog to get optimal solution
            P = 2*((X'*X) + lambda1*(Z'*Z));
            q = lambda2* (one_k' * pinv(Z*Z')*Z)' - (2*X'*Y);
            G = -eye(n); 
            h = zeros(n,1);
            A = one_w'; b = 1;
            options = optimoptions('quadprog',...
                    'Display','Off');
            w = quadprog(P, q, G, h, A, b,...
                    0*ones(nstocks,1), 0.05*ones(nstocks,1),...
                    [],options);
            w(w<1e-3) = 0;
        end
    end
end