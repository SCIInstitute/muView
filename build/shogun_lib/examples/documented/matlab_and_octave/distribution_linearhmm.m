% Trains an inhomogeneous Markov chain of order 3 on a DNA string data set. Due to
% the structure of the Markov chain it is very similar to a HMM with just one
% chain of connected hidden states - that is why we termed this linear HMM.

order=3;
gap=0;
num=12;
reverse='n'; % bit silly to not use boolean, set 'r' to yield true

addpath('tools');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');

% LinearHMM
disp('LinearHMM');

%sg('new_distribution', 'LinearHMM');
sg('add_preproc', 'SORTWORDSTRING');

sg('set_features', 'TRAIN', fm_train_dna, 'DNA');
sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
sg('attach_preproc', 'TRAIN');

%	sg('train_distribution');
%	histo=sg('get_histogram');

%	num_param=sg('get_histogram_num_model_parameters');
%	for i = 1:num,
%		for j = 1:num_param,
%			sg(sprintf('get_log_derivative %d %d', j, i));
%		end
%	end

%	sg('get_log_likelihood');
%	sg('get_log_likelihood_sample');

