parameter_list =[[23],[24]]

def kernel_const_modular (c=23):
	from shogun.Features import DummyFeatures
	from shogun.Kernel import ConstKernel

	feats_train=DummyFeatures(10)
	feats_test=DummyFeatures(17)
	
	kernel=ConstKernel(feats_train, feats_train, c)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('Const')
	kernel_const_modular(*parameter_list[0])
