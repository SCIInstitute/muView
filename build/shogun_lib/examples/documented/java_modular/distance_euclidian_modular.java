// An approach as applied below, which shows the processing of input data
// from a file becomes a crucial factor for writing your own sample applications.
// This approach is just one example of what can be done using the distance
// functions provided by shogun.
// 
// First, you need to determine what type your data will be, because this
// will determine the distance function you can use.
// 
// This example loads two stored matrices of real values from different 
// files and initializes the matrices to 'RealFeatures'.
// Each column of the matrices corresponds to one data point.
// 
// The distance initialized by two data sets (the same data set as shown in the 
// first call) controls the processing of the given data points, where a pairwise 
// distance matrix is computed by 'get_distance_matrix'.
// 
// The resulting distance matrix can be reaccessed by 'get_distance_matrix'.
// 
// The method call 'init'* binds the given data sets, where a pairwise distance 
// matrix between these two data sets is computed by 'get_distance_matrix'.
// 
// The resulting distance matrix can be reaccessed by 'get_distance_matrix'.
// 
// *Note that the previous computed distance matrix can no longer be 
// reaccessed by 'get_distance_matrix'.
// 
// For more details see doc/classshogun_1_1CEuclidianDistance.html.
// 
// Obviously, using the Euclidian distance is not limited to this showcase
// example.

import org.shogun.*;
import org.jblas.*;

public class distance_euclidian_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);

		EuclidianDistance distance = new EuclidianDistance(feats_train, feats_train);
		
		DoubleMatrix dm_train = distance.get_distance_matrix();
		distance.init(feats_train, feats_test);
		DoubleMatrix dm_test = distance.get_distance_matrix();

		System.out.println(dm_train.toString());
		System.out.println(dm_test.toString());

		modshogun.exit_shogun();
	}
}
