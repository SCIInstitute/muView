// In this example a two-class support vector machine classifier is trained on a
// toy data set and the trained classifier is used to predict labels of test
// examples. As training algorithm the LIBSVM solver is used with SVM
// regularization parameter C=1 and a Gaussian kernel of width 2.1 and the
// precision parameter epsilon=1e-5. The example also shows how to retrieve the
// support vectors from the train SVM model.
// 
// For more details on LIBSVM solver see http://www.csie.ntu.edu.tw/~cjlin/libsvm/

import org.shogun.*;
import org.jblas.*;

import static org.shogun.BinaryLabels.obtain_from_generic;

public class classifier_libsvm_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		double width = 2.1;
		double epsilon = 1e-5;
		double C = 1.0;

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		DoubleMatrix trainlab = Load.load_labels("../data/label_train_twoclass.dat");

		RealFeatures feats_train = new RealFeatures();
		feats_train.set_feature_matrix(traindata_real);
		RealFeatures feats_test = new RealFeatures();
		feats_test.set_feature_matrix(testdata_real);

		GaussianKernel kernel = new GaussianKernel(feats_train, feats_train, width);

		BinaryLabels labels = new BinaryLabels(trainlab);

		LibSVM svm = new LibSVM(C, kernel, labels);
		svm.set_epsilon(epsilon);
		svm.train();

		kernel.init(feats_train, feats_test);
		DoubleMatrix out_labels = obtain_from_generic(svm.apply()).get_labels();
		System.out.println(out_labels.toString());

		modshogun.exit_shogun();
	}
}
