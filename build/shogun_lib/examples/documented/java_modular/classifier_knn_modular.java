// This example shows usage of a k-nearest neighbor (KNN) classification rule on
// a toy data set. The number of the nearest neighbors is set to k=3 and the distances
// are measured by the Euclidean metric. Finally, the KNN rule is applied to predict
// labels of test examples. 

import org.shogun.*;
import org.jblas.*;

import static org.shogun.MulticlassLabels.obtain_from_generic;

public class classifier_knn_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		int k = 3;

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		DoubleMatrix trainlab = Load.load_labels("../data/label_train_multiclass.dat");

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);
		EuclidianDistance distance = new EuclidianDistance(feats_train, feats_train);

		MulticlassLabels labels = new MulticlassLabels(trainlab);

		KNN knn = new KNN(k, distance, labels);
		knn.train();
		DoubleMatrix out_labels = obtain_from_generic(knn.apply(feats_test)).get_labels();
		System.out.println(out_labels.toString());

		modshogun.exit_shogun();
	}
}
