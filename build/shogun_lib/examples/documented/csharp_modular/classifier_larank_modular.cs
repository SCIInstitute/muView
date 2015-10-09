// In this example a multi-class support vector machine classifier is trained on a
// toy data set and the trained classifier is then used to predict labels of test
// examples. As training algorithm the LaRank algorithm is used with SVM
// regularization parameter C=1 and a Gaussian kernel of width 2.1 and a precision
// set to epsilon=1e-5.
// 
// For more details on LaRank see
//    Bordes, A. and Bottou, L. and Gallinari, P. and Weston, J.
//    Solving MultiClass Support Vector Machines with LaRank. ICML 2007.
// 

using System;

public class classifier_larank_modular {
	public static void Main() {
		modshogun.init_shogun_with_defaults();
		double width = 2.1;
		double epsilon = 1e-5;
		double C = 1.0;

		double[,] traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		double[,] testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		double[] trainlab = Load.load_labels("../data/label_train_multiclass.dat");

		RealFeatures feats_train = new RealFeatures();
		feats_train.set_feature_matrix(traindata_real);
		RealFeatures feats_test = new RealFeatures();
		feats_test.set_feature_matrix(testdata_real);

		GaussianKernel kernel = new GaussianKernel(feats_train, feats_train, width);

		MulticlassLabels labels = new MulticlassLabels(trainlab);

		LaRank svm = new LaRank(C, kernel, labels);
		svm.set_batch_mode(false);
		svm.set_epsilon(epsilon);
		svm.train();
		double[] out_labels = MulticlassLabels.obtain_from_generic(svm.apply(feats_train)).get_labels();

		foreach(double item in out_labels) {
			Console.Write(item);
		}

		modshogun.exit_shogun();
	}
}
