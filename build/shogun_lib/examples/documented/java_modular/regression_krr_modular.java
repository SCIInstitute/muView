// In this example a kernelized version of ridge regression (KRR) is trained on a
// real-valued data set. The KRR is trained with regularization parameter tau=1e-6
// and a gaussian kernel with width=0.8. The labels of both the train and the test
// data can be fetched via krr.classify().get_labels().

import org.shogun.*;
import org.jblas.*;

import static org.shogun.RegressionLabels.obtain_from_generic;

public class regression_krr_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		double width = 0.8;
		double tau = 1e-6;

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		DoubleMatrix trainlab = Load.load_labels("../data/label_train_twoclass.dat");

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);
		GaussianKernel kernel= new GaussianKernel(feats_train, feats_train, width);

		RegressionLabels labels = new RegressionLabels(trainlab);

		KernelRidgeRegression krr = new KernelRidgeRegression(tau, kernel, labels);
		krr.train(feats_train);

		kernel.init(feats_train, feats_test);
		DoubleMatrix out_labels = obtain_from_generic(krr.apply()).get_labels();
		System.out.println(out_labels.toString());

		modshogun.exit_shogun();
	}
}
