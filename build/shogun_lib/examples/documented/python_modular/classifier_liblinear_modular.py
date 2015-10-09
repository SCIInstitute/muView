# In this example a two-class linear support vector machine classifier is trained
# on a toy data set and the trained classifier is then used to predict labels of
# test examples. As training algorithm the LIBLINEAR solver is used with the SVM
# regularization parameter C=0.9 and the bias in the classification rule switched
# on and the precision parameters epsilon=1e-5.
# 
# For more details on LIBLINEAR see
#     http://www.csie.ntu.edu.tw/~cjlin/liblinear/

from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')
label_traindat = lm.load_labels('../data/label_train_twoclass.dat')

parameter_list = [[traindat,testdat,label_traindat,0.9,1e-3],[traindat,testdat,label_traindat,0.8,1e-2]]

def classifier_liblinear_modular(fm_train_real, fm_test_real,
		label_train_twoclass, C, epsilon):

	from shogun.Features import RealFeatures, SparseRealFeatures, BinaryLabels
	from shogun.Classifier import LibLinear, L2R_L2LOSS_SVC_DUAL
	from shogun.Mathematics import Math_init_random
	Math_init_random(17)

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)
	labels=BinaryLabels(label_train_twoclass)

	svm=LibLinear(C, feats_train, labels)
	svm.set_liblinear_solver_type(L2R_L2LOSS_SVC_DUAL)
	svm.set_epsilon(epsilon)
	svm.set_bias_enabled(True)
	svm.train()

	svm.set_features(feats_test)
	svm.apply().get_labels()
	predictions = svm.apply()
	return predictions, svm, predictions.get_labels()



if __name__=='__main__':
	print('LibLinear')
	classifier_liblinear_modular(*parameter_list[0])


