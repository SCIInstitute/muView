// The class FKFeatures implements Fischer kernel features obtained from
// two Hidden Markov models.
// 
// It was used in
// 
// K. Tsuda, M. Kawanabe, G. Raetsch, S. Sonnenburg, and K.R. Mueller. A new
// discriminative kernel from probabilistic models. Neural Computation,
// 14:2397-2414, 2002.
// 
// which also has the details.
// 
// Note that FK-features are computed on the fly, so to be effective feature
// caching should be enabled.
// 
// It inherits its functionality from CSimpleFeatures, which should be
// consulted for further reference.
// 

import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.DNA;
import static org.shogun.BaumWelchViterbiType.BW_NORMAL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class kernel_fisher_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		boolean reverse = false;
		int N = 1;
		int M = 64;
		double pseudo = 1e-5;
		int order = 3;
		int gap = 0;

		String[] fm_train_dna = Load.load_dna("../data/fm_train_dna.dat");
		String[] fm_test_dna = Load.load_dna("../data/fm_test_dna.dat");
		DoubleMatrix label_train_dna = Load.load_labels("../data/label_train_dna.dat");

		
		ArrayList fm_hmm_pos_builder = new ArrayList();
		ArrayList fm_hmm_neg_builder = new ArrayList();
		for(int i = 0; i < label_train_dna.getColumns(); i++) {
			if (label_train_dna.get(i) == 1)			
				fm_hmm_pos_builder.add(fm_train_dna[i]);
			else
				fm_hmm_neg_builder.add(fm_train_dna[i]);			
		}

		int pos_size = fm_hmm_pos_builder.size();
		int neg_size = fm_hmm_neg_builder.size();
		String[] fm_hmm_pos = new String[pos_size];
		String[] fm_hmm_neg = new String[neg_size];
		for (int i = 0; i < pos_size; i++)
			fm_hmm_pos[i] = (String)fm_hmm_pos_builder.get(i);
		for (int i = 0; i < neg_size; i++)
			fm_hmm_neg[i] = (String)fm_hmm_neg_builder.get(i);

		StringCharFeatures charfeat = new StringCharFeatures(fm_hmm_pos, DNA);
		StringWordFeatures hmm_pos_train = new StringWordFeatures(charfeat.get_alphabet());
		hmm_pos_train.obtain_from_char(charfeat, order-1, order, gap, reverse);

		HMM pos = new HMM(hmm_pos_train, N, M, pseudo);
		pos.baum_welch_viterbi_train(BW_NORMAL);

		charfeat = new StringCharFeatures(fm_hmm_neg, DNA);
		StringWordFeatures hmm_neg_train = new StringWordFeatures(charfeat.get_alphabet());
		hmm_neg_train.obtain_from_char(charfeat, order-1, order, gap, reverse);

		HMM neg = new HMM(hmm_neg_train, N, M, pseudo);
		neg.baum_welch_viterbi_train(BW_NORMAL);

		charfeat = new StringCharFeatures(fm_train_dna, DNA);
		StringWordFeatures wordfeats_train = new StringWordFeatures(charfeat.get_alphabet());
		wordfeats_train.obtain_from_char(charfeat, order-1, order, gap, reverse);

		charfeat = new StringCharFeatures(fm_test_dna, DNA);
		StringWordFeatures wordfeats_test = new StringWordFeatures(charfeat.get_alphabet());
		wordfeats_test.obtain_from_char(charfeat, order-1, order, gap, reverse);

		pos.set_observations(wordfeats_train);
		neg.set_observations(wordfeats_train);
		FKFeatures feats_train = new FKFeatures(10, pos, neg);
		feats_train.set_opt_a(-1);
		PolyKernel kernel = new PolyKernel(feats_train, feats_train, 1, true);
		DoubleMatrix km_train = kernel.get_kernel_matrix();

		HMM pos_clone = new HMM(pos);
		HMM neg_clone = new HMM(neg);
		pos_clone.set_observations(wordfeats_test);
		neg_clone.set_observations(wordfeats_test);
		FKFeatures feats_test = new FKFeatures(10, pos_clone, neg_clone);
		feats_test.set_a(feats_train.get_a());
		kernel.init(feats_train, feats_test);
		DoubleMatrix km_test=kernel.get_kernel_matrix();

		modshogun.exit_shogun();
	}
}
