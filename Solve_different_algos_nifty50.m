clear all; clc; close all;
% Run matlab file to create return table and closing table for stocks and
% its constitutents
create_data_table_nifty50
X = return_data{:, 3:end}; % T x N return matrix for stocks
r_b = return_data{:, 2}; % T x 1 return vector for index
Y = closing_data{:, 3:end}; % T x N close price matrix of stocks
dates = closing_data{:,1}; % T x 1 vector of dates

%% Plot with monthly re-adjustment

%----NOTE-----
% The output from SADA Clus in plot changes every iteration due to
% overlapping clusters during spectral clustering which is sensitive to
% initilization. Kindly run this section again if result are not
% consistent with the plot in our report. Suggestions for improving this 
% clustering method listed in future work of our term report.
%-------------

%----NOTE----
% You may see the best result of the algorithms in these plots may still 
% not track the index, as in S&P500 in report. Is the algorithm not good?
% The issue is that the weights decided by solving the ETE (Eq. 1 in 
% report) are a result based on the past and does not guarentee a good 
% result in the by default for the future. This is what makes this still an
% open problem with active research to improve the future prediction.
%------------

clc; close all;

h = figure();
axis tight manual

% Step: Number of days after which we reconstruct the portfolio
step = 63;

lait_normalized_portfolio_value=[]; 
slait_normalized_portfolio_value=[];
L12_normalized_portfolio_value=[];
sada_sec_normalized_portfolio_value=[];
sada_clus_normalized_portfolio_value=[];
% Start date of investment: 1 trading year + 1 days (252+1 day)
% The one year history to be use to determine weights at start of
% investment
start = 253;
index_normalized = closing_data{start:end, 2};
index_normalized = closing_data{start:end, 2}/closing_data{start, 2};

for i = start:step:ndays
    % end date of last batch for prediction clamped at number of days in
    % the dataset irrespective of the step size
    e = min(ndays,i+step-1);
    
    % Call LAIT Class. Max Itrs: 20, regularization param Lambda = 10, 
    % Print convergence data = "No"
    ObjLAIT = LAIT(log(1+X(i-252:i-1,:)),...
        log(1+r_b(i-252:i-1,:)), 20, 10, "No");
    [w_lait, conv_hist_lait, cost_hist_lait] = ...
        ObjLAIT.optimweight();
    w_lait = max(w_lait, zeros(size(w_lait)));
    
    % Call SLAIT Class. Max itrs: 20, regularization param Lambda = 10,
    % Print convergence data = "No"
    ObjSLAIT = SLAIT(log(1+X(i-252:i-1,:)), ...
        log(1+r_b(i-252:i-1,:)), 20, 10,"No");
    [w_slait, conv_hist_slait, cost_hist_slait] = ...
        ObjSLAIT.optimweight();
    w_slait = max(w_slait, zeros(size(w_slait)));

    % Call L12 Class, Max itrs: 2500, regularization param Lambda = 10,
    % Sparsity required (number of non zero weights) = K
    ObjL12 = L12(log(1+X(i-252:i-1,:)),...
        log(1+r_b(i-252:i-1,:)), 2500, 100,20);
    w_L12 = ObjL12.optimweight();

    % Call SADA Class using industry sectors given by NSE,
    % industry_one_hot:Used for sparsity calculation with industry sector
    % given byNSE, 
    % distance measurement method = "rankcorr"(Spearman rank
    % correlation), 
    % Regularization param Lambda 1 = 1, 
    % Regularization param Lambda 2 = 800, 
    % Number of industry sectors to consider = NaN for NSE industry sector
    % assignment used, otherwise int value given if spectral clustering used
    % Method of calculating affinity matrix = "sector" for NSE sectors,
    % "cluster" for spectral clustering
    ObjSADA_sec = sparseAndDiverse(log(1+X(i-252:i-1,:)),...
        log(1+r_b(i-252:i-1,:)),...
        industry_onehot, "rankcorr",...
        1, 800, NaN ,"sector");
    w_sada_sec = ObjSADA_sec.optimweight();

    % Call SADA Class using industry sectors determined using spectral
    % clustering. Number of cluster kept equal to number of sectors
    % originally given by NSE.
    ObjSADA_clus = sparseAndDiverse(log(1+X(i-252:i-1,:)),...
        log(1+r_b(i-252:i-1,:)),...
        industry_onehot, ...
        "euclidean", 1, 800, 13, "cluster");
    w_sada_clus = ObjSADA_clus.optimweight();
    
    % We calculate the portfolio value for an investment of Re.1 (just for 
    % representative purpose) at the start of our investment. To change
    % this, simply multiply index_normalized defined in Line 27 by the
    % factor of our choice.
    if i == start
        Y_lait_norm = (Y(i:e,:)*w_lait)*index_normalized(i-252,:)...
            /(Y(i,:)*w_lait);
        w_lait_old = w_lait; 

        Y_slait_norm = (Y(i:e,:)*w_slait)*index_normalized(i-252,:)...
            /(Y(i,:)*w_slait);
        w_slait_old = w_slait;

        Y_L12_norm = (Y(i:e,:)*w_L12)*index_normalized(i-252,:)...
            /(Y(i,:)*w_L12);
        w_L12_old = w_L12;

        Y_sada_sec_norm = (Y(i:e,:)*w_sada_sec)*index_normalized(i-252,:)...
            /(Y(i,:)*w_sada_sec);
        w_sada_sec_old = w_sada_sec;

        Y_sada_clus_norm = (Y(i:e,:)*w_sada_clus)*index_normalized(i-252,:)...
            /(Y(i,:)*w_sada_clus);
        w_sada_clus_old = w_sada_clus;
    else
        % Before reconstructing, we calculate the value of our portfolio at
        % the end of previously invested term. Then we invest that capital
        % amount for the next term. Below equation does the same.
        Y_lait_norm = (Y(i:e,:)*w_lait) *....
            lait_normalized_portfolio_value(i-start)/(Y(i-1,:)*w_lait);
        w_lait_old = w_lait; 

        Y_slait_norm = (Y(i:e,:)*w_slait) *...
            slait_normalized_portfolio_value(i-start)/(Y(i-1,:)*w_slait);
        w_slait_old = w_slait;

        Y_L12_norm = (Y(i:e,:)*w_L12) *...
            slait_normalized_portfolio_value(i-start)/(Y(i-1,:)*w_L12);
        w_L12_old = w_L12;

        Y_sada_sec_norm = (Y(i:e,:)*w_sada_sec) *...
            sada_sec_normalized_portfolio_value(i-start)/(Y(i-1,:)*w_sada_sec);
        w_sada_sec_old = w_sada_sec;

        Y_sada_clus_norm = (Y(i:e,:)*w_sada_clus) *...
            sada_clus_normalized_portfolio_value(i-start)/(Y(i-1,:)*w_sada_clus);
        w_sada_clus_old = w_sada_clus;
    end
    
    % Keep appending terms
    lait_normalized_portfolio_value = [lait_normalized_portfolio_value; ...
        Y_lait_norm];
    slait_normalized_portfolio_value = [slait_normalized_portfolio_value; ...
        Y_slait_norm];
    L12_normalized_portfolio_value = [L12_normalized_portfolio_value; ...
        Y_L12_norm];
    sada_sec_normalized_portfolio_value = [sada_sec_normalized_portfolio_value; ...
        Y_sada_sec_norm];
    sada_clus_normalized_portfolio_value = [sada_clus_normalized_portfolio_value; ...
        Y_sada_clus_norm];
end
% Plot the portfolio return if 'invested' in NSE50, vs using algorithms
% suggested in our report.
plot(dates(start:end), lait_normalized_portfolio_value,...
    '-b', 'Linewidth',1.5); hold on;
plot(dates(start:end),slait_normalized_portfolio_value,...
    '-r', 'Linewidth',1.5);
plot(dates(start:end),L12_normalized_portfolio_value,...
    '-c', 'Linewidth',1.5);
plot(dates(start:end),sada_sec_normalized_portfolio_value,...
    '-m', 'Linewidth',1.5);
plot(dates(start:end),sada_clus_normalized_portfolio_value,...
    '-g', 'Linewidth',1.5);
plot(dates(start:end), index_normalized,...
    '-k', 'Linewidth',2.0);
set(gca, 'FontSize',11,'FontWeight','Bold');
lgnd = legend('LAIT','SLAIT','L12', 'SADA Sec','SADA Clus',...
    'NIFTY50', 'Location','NorthWest',...
    'interpreter','latex','FontSize',9.9);
lgnd.NumColumns = 3;
fig = gcf;
% Set fig size, and everything else for consistency across all PCs
x0=350; y0=385; width=500; height=175;
fig.Position = [x0,y0,width,height];
grid minor;
xlabel('Date','interpreter','latex');
ylabel('Portfolio value','interpreter','latex');
%     title({'Nifty50 Index Tracking',...
%         'Sparse index return with initial investment of Re.1',...
%         strcat('Reconstruction window: ',num2str(step),' days')});
xlim([dates(start) dates(end)]); ylim([0.65 1.95]);

drawnow