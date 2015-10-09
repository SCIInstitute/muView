# In this example a two-class linear support vector machine classifier (SVM) is
# trained on a toy data set and the trained classifier is used to predict labels
# of test examples. As training algorithm the SVMLIN solver is used with the SVM
# regularization parameter C=0.9 and the bias in the classification rule switched
# on and the precision parameter epsilon=1e-5. The example also shows how to
# retrieve parameters (vector w and bias b)) of the trained linear classifier.
# 
# For more details on the SVMLIN solver see
#  V. Sindhwani, S.S. Keerthi. Newton Methods for Fast Solution of Semi-supervised
#  Linear SVMs. Large Scale Kernel Machines MIT Press (Book Chapter), 2007

from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')
label_traindat = lm.load_labels('../data/label_train_twoclass.dat')

parameter_list = [[traindat,testdat,label_traindat,0.9,1e-5,1],[traindat,testdat,label_traindat,0.8,1e-5,1]]

def classifier_svmlin_modular (fm_train_real=traindat,fm_test_real=testdat,label_train_twoclass=label_traindat,C=0.9,epsilon=1e-5,num_threads=1):
	from shogun.Features import RealFeatures, SparseRealFeatures, BinaryLabels
	from shogun.Classifier import SVMLin

	realfeat=RealFeatures(fm_train_real)
	feats_train=SparseRealFeatures()
	feats_train.obtain_from_simple(realfeat)
	realfeat=RealFeatures(fm_test_real)
	feats_test=SparseRealFeatures()
	feats_test.obtain_from_simple(realfeat)

	labels=BinaryLabels(label_train_twoclass)

	svm=SVMLin(C, feats_train, labels)
	svm.set_epsilon(epsilon)
	svm.parallel.set_num_threads(num_threads)
	svm.set_bias_enabled(True)
	svm.train()

	svm.set_features(feats_test)
	svm.get_bias()
	svm.get_w()
	svm.apply().get_labels()
	predictions = svm.apply()
	return predictions, svm, predictions.get_labels()


if __name__=='__main__':
	print('SVMLin')
	classifier_svmlin_modular(*parameter_list[0])
