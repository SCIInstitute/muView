% The Weighted Degree String kernel.
% 
% The WD kernel of order d compares two sequences X and
% Y of length L by summing all contributions of k-mer matches of
% lengths k in 1...d , weighted by coefficients beta_k. It
% is defined as
% 
%     k(X, Y)=\sum_{k=1}^d\beta_k\sum_{l=1}^{L-k+1}I(u_{k,l}(X)=u_{k,l}(Y)).
% 
% Here, $u_{k,l}(X)$ is the string of length k starting at position
% l of the sequence X and I(.) is the indicator function
% which evaluates to 1 when its argument is true and to 0
% otherwise.
% 

size_cache=10;

addpath('tools');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');
fm_test_dna=load_matrix('../data/fm_test_dna.dat');

% Weighted Degree String
disp('WeightedDegreeString');

degree=20;

sg('set_kernel', 'WEIGHTEDDEGREE', 'CHAR', size_cache, degree);

sg('set_features', 'TRAIN', fm_train_dna, 'DNA');
km=sg('get_kernel_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_dna, 'DNA');
km=sg('get_kernel_matrix', 'TEST');
