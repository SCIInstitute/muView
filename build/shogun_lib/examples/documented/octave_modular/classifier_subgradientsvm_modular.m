% In this example a two-class linear support vector machine classifier is trained
% on a toy data set and the trained classifier is used to predict labels of test
% examples. As training algorithm the steepest descent subgradient algorithm is
% used. The SVM regularization parameter is set to C=0.9 and the bias in the
% classification rule is switched off. The solver iterates until it finds an
% epsilon-precise solution (epsilon=1e-3) or the maximal training time
% max_train_time=1 (seconds) is exceeded. The unbiased linear rule is trained.
% 
% Note that this solver often does not converges because the steepest descent
% subgradient algorithm is oversensitive to rounding errors. Note also that this
% is an unpublished work which was predecessor of the OCAS solver (see
% classifier_svmocas).

init_shogun

addpath('tools');
label_train_twoclass=load_matrix('../data/label_train_twoclass.dat');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');

% subgradient based svm
disp('SubGradientSVM')

realfeat=RealFeatures(fm_train_real);
feats_train=SparseRealFeatures();
feats_train.obtain_from_simple(realfeat);
realfeat=RealFeatures(fm_test_real);
feats_test=SparseRealFeatures();
feats_test.obtain_from_simple(realfeat);

C=0.9;
epsilon=1e-3;
num_threads=1;
max_train_time=1.;
labels=BinaryLabels(label_train_twoclass);

svm=SubGradientSVM(C, feats_train, labels);
svm.set_epsilon(epsilon);
svm.parallel.set_num_threads(num_threads);
svm.set_bias_enabled(false);
svm.set_max_train_time(max_train_time);
svm.train();

svm.set_features(feats_test);
svm.apply().get_labels();

