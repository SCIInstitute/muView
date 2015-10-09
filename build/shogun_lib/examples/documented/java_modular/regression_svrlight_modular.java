// In this example a support vector regression algorithm is trained on a
// real-valued toy data set. The underlying library used for the SVR training is
// SVM^light. The SVR is trained with regularization parameter C=1 and a gaussian
// kernel with width=2.1. The the label of both the train and the test data are
// fetched via svr.classify().get_labels().
// 
// For more details on the SVM^light see
//  T. Joachims. Making large-scale SVM learning practical. In Advances in Kernel
//  Methods -- Support Vector Learning, pages 169-184. MIT Press, Cambridge, MA USA, 1999.

import org.shogun.*;
import org.jblas.*;

import static org.shogun.RegressionLabels.obtain_from_generic;

public class regression_svrlight_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		double width = 0.8;
		int C = 1;
		double epsilon = 1e-5;
		double tube_epsilon = 1e-2;
		int num_threads = 3;

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		DoubleMatrix trainlab = Load.load_labels("../data/label_train_twoclass.dat");

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);
		GaussianKernel kernel= new GaussianKernel(feats_train, feats_train, width);

		RegressionLabels labels = new RegressionLabels(trainlab);

		SVRLight svr = new SVRLight(C, epsilon, kernel, labels);
		svr.set_tube_epsilon(tube_epsilon);
		//svr.parallel.set_num_threads(num_threads);
		svr.train();

		kernel.init(feats_train, feats_test);
		DoubleMatrix out_labels = obtain_from_generic(svr.apply()).get_labels();
		System.out.println(out_labels.toString());

		modshogun.exit_shogun();
	}
}
