% In this example an agglomerative hierarchical single linkage clustering method
% is used to cluster a given toy data set. Starting with each object being
% assigned to its own cluster clusters are iteratively merged. Here the clusters
% are merged that have the closest (minimum distance, here set via the Euclidean
% distance object) two elements.

addpath('tools');
fm_train=load_matrix('../data/fm_train_real.dat');

% Hierarchical
disp('Hierarchical');

merges=3;

sg('set_features', 'TRAIN', fm_train);
sg('set_distance', 'EUCLIDIAN', 'REAL');
sg('new_clustering', 'HIERARCHICAL');

sg('train_clustering', merges);
[merge_distance, pairs]=sg('get_clustering');
