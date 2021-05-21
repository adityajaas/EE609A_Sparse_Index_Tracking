%% Script to create closing_data.txt and return_data.txt
% Load Index Data
clc;clear;
addpath('Datacrawler NSEpy');
addpath('Datacrawler NSEpy/Stocks data 5yr');
NIFTY50_table = readtable('NIFTY50_23Mar16_22Mar21.csv', ...
    'PreserveVariableNames', 1);
clc;

% Load companies list
nifty50list = readtable('nifty50list.csv', ...
    'PreserveVariableNames', 1);
symbols = nifty50list.Symbol;
industry = categorical(nifty50list.Industry);
industry_labels = categories(industry);
industry_onehot = (industry' == industry_labels);

% Load Stocks Data
ndays = length(NIFTY50_table.Close); nstocks =  length(symbols);
closing_data = table(NIFTY50_table.Date, ...
    'VariableNames', ["Date"]);
temp_table = table(NIFTY50_table.Close, ...
    'VariableNames', ["NIFTY50"]);
closing_data = [closing_data temp_table]; 

for i = 1:nstocks
    filename = strcat(string(symbols(i)), '.csv');
    stock = readtable(filename);
    temp_table = table(stock.Close,  ...
        'VariableNames', [string(symbols(i))]);
    if height(temp_table) ~= ndays
        zero_pad = table(zeros(ndays - height(temp_table),1),...
            'VariableNames', [string(symbols(i))]);
        temp_table = [zero_pad ;temp_table];
    end 
    closing_data = [closing_data temp_table];
end
% Create return price data
return_data = closing_data;

% Change the first date return to 0
% nstocks + 1 because of NIFTY50 + 50 stocks
return_data(1,2:end) = num2cell(zeros(1, nstocks + 1));

% Return r_i = (p_i - p_{i-1}) / p_i
for i = 2: ndays
            return_data(i,2:end) = num2cell((closing_data{i,2:end}...
                - closing_data{i-1,2:end})./closing_data{i-1,2:end});
end
% If stocks wasn't listed before a certain date,
% its return data would be NaN or Inf.
% Convert those to 0
for j = 2:nstocks+2
    return_data{isnan(return_data{:,j}),j} = 0;
    return_data{isinf(return_data{:,j}),j} = 0;
end
clear filename i NIFTY50_table stock symbols temp_table zero_pad j