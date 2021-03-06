# This is an example for the initialization of the local alignment kernel on 
# DNA sequences, where each column of the matrices of type char corresponds to 
# one training/test example. 

from tools.load import LoadMatrix
from sg import sg
lm=LoadMatrix()

traindna=lm.load_dna('../data/fm_train_dna.dat')
testdna=lm.load_dna('../data/fm_test_dna.dat')
parameter_list=[[traindna,testdna,10],
		[traindna,testdna,11]]

def kernel_localalignmentstring (fm_train_dna=traindna,fm_test_dna=testdna,
			    size_cache=10):

	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	sg('set_features', 'TEST', fm_test_dna, 'DNA')
	sg('set_kernel', 'LOCALALIGNMENT', 'CHAR', size_cache)
	km=sg('get_kernel_matrix', 'TRAIN')
	km=sg('get_kernel_matrix', 'TEST')
	return km

if __name__=='__main__':
	print('LocalAlignmentString')
	kernel_localalignmentstring(*parameter_list[0])
