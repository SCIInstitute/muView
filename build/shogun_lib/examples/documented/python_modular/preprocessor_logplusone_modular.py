# In this example a kernel matrix is computed for a given real-valued data set.
# The kernel used is the Chi2 kernel which operates on real-valued vectors. It
# computes the chi-squared distance between sets of histograms. It is a very
# useful distance in image recognition (used to detect objects). The preprocessor
# LogPlusOne adds one to a dense real-valued vector and takes the logarithm of
# each component of it. It is most useful in situations where the inputs are
# counts: When one compares differences of small counts any difference may matter
# a lot, while small differences in large counts don't. This is what this log
# transformation controls for.

from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')

parameter_list = [[traindat+10,testdat+10,1.4,10],[traindat+10,testdat+10,1.5,10]]

def preprocessor_logplusone_modular (fm_train_real=traindat,fm_test_real=testdat,width=1.4,size_cache=10):

	from shogun.Kernel import Chi2Kernel
	from shogun.Features import RealFeatures
	from shogun.Preprocessor import LogPlusOne

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	preproc=LogPlusOne()
	preproc.init(feats_train)
	feats_train.add_preprocessor(preproc)
	feats_train.apply_preprocessor()
	feats_test.add_preprocessor(preproc)
	feats_test.apply_preprocessor()

	
	kernel=Chi2Kernel(feats_train, feats_train, width, size_cache)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()

	return km_train,km_test,kernel

if __name__=='__main__':
	print('LogPlusOne')
	preprocessor_logplusone_modular(*parameter_list[0])
