// The WeightedCommWordString kernel may be used to compute the weighted
// spectrum kernel (i.e. a spectrum kernel for 1 to K-mers, where each k-mer
// length is weighted by some coefficient \f$\beta_k\f$) from strings that have
// been mapped into unsigned 16bit integers.
// 
// These 16bit integers correspond to k-mers. To applicable in this kernel they
// need to be sorted (e.g. via the SortWordString pre-processor).
// 
// It basically uses the algorithm in the unix "comm" command (hence the name)
// to compute:
// 
// k({\bf x},({\bf x'})= \sum_{k=1}^K\beta_k\Phi_k({\bf x})\cdot \Phi_k({\bf x'})
// 
// where \f$\Phi_k\f$ maps a sequence \f${\bf x}\f$ that consists of letters in
// \f$\Sigma\f$ to a feature vector of size \f$|\Sigma|^k\f$. In this feature
// vector each entry denotes how often the k-mer appears in that \f${\bf x}\f$.
// 
// Note that this representation is especially tuned to small alphabets
// (like the 2-bit alphabet DNA), for which it enables spectrum kernels
// of order 8.
// 
// For this kernel the linadd speedups are quite efficiently implemented using
// direct maps.
// 

using System;

public class kernel_weighted_comm_word_string_modular {
	public static void Main() {
		modshogun.init_shogun_with_defaults();
		int degree = 20;

		String[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");
		String[] fm_test_dna = Load.load_dna("../data/fm_test_dna.dat");

		StringCharFeatures feats_train = new StringCharFeatures(fm_train_dna, EAlphabet.DNA);
		StringCharFeatures feats_test = new StringCharFeatures(fm_test_dna, EAlphabet.DNA);

		WeightedDegreePositionStringKernel kernel = new WeightedDegreePositionStringKernel(feats_train, feats_train, degree);

		double[,] km_train = kernel.get_kernel_matrix();
		kernel.init(feats_train, feats_test);
		double[,] km_test = kernel.get_kernel_matrix();
		modshogun.exit_shogun();
	}
}
