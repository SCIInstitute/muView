# This example initializes the locality improved string kernel. The locality improved string 
# kernel is defined on sequences of the same length and inspects letters matching at 
# corresponding positions in both sequences. The kernel sums over all matches in windows of 
# length l and takes this sum to the power of 'inner_degree'. The sum over all these 
# terms along the sequence is taken to the power of 'outer_degree'. 

from tools.load import LoadMatrix
from sg import sg
lm=LoadMatrix()

traindna=lm.load_dna('../data/fm_train_dna.dat')
testdna=lm.load_dna('../data/fm_test_dna.dat')
trainlabel=lm.load_labels('../data/label_train_dna.dat')
parameter_list=[[traindna,testdna,trainlabel,10,5,5,7],
		[traindna,testdna,trainlabel,11,6,6,8]]

def kernel_localityimprovedstring (fm_train_dna=traindna,fm_test_dna=testdna,
				 label_train_dna=trainlabel,size_cache=10,
				 length=5,inner_degree=5,outer_degree=7):

	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	sg('set_features', 'TEST', fm_test_dna, 'DNA')
	sg('set_kernel', 'LIK', 'CHAR', size_cache, length, inner_degree, outer_degree)
	km=sg('get_kernel_matrix', 'TRAIN')
	km=sg('get_kernel_matrix', 'TEST')
	return km

if __name__=='__main__':
	print('LocalityImprovedString')
	kernel_localityimprovedstring(*parameter_list[0])
