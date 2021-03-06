# The Weighted Degree Position String kernel (Weighted Degree kernel with shifts).
# 
# The WD-shift kernel of order d compares two sequences X and
# Y of length L by summing all contributions of k-mer matches of
# lengths k in 1...d, weighted by coefficients beta_k
# allowing for a positional tolerance of up to shift s.
# 

from tools.load import LoadMatrix
from sg import sg
lm=LoadMatrix()

traindna=lm.load_dna('../data/fm_train_dna.dat')
testdna=lm.load_dna('../data/fm_test_dna.dat')
parameter_list=[[traindna,testdna,10,20],
		[traindna,testdna,11,21]]

def kernel_weighteddegreepositonstring (fm_train_dna=traindna,fm_test_dna=testdna,
			size_cache=10,degree=20):

	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	sg('set_features', 'TEST', fm_test_dna, 'DNA')
	sg('set_kernel', 'WEIGHTEDDEGREEPOS', 'CHAR', size_cache, degree)
	km=sg('get_kernel_matrix', 'TRAIN')
	km=sg('get_kernel_matrix', 'TEST')
	return km

if __name__=='__main__':
	print('WeightedDegreePositionString')
	kernel_weighteddegreepositonstring(*parameter_list[0])
