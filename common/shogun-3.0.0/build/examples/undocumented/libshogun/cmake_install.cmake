# Install script for directory: D:/Code/Common/shogun-3.0.0/examples/undocumented/libshogun

# Set the install prefix
IF(NOT DEFINED CMAKE_INSTALL_PREFIX)
  SET(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/shogun")
ENDIF(NOT DEFINED CMAKE_INSTALL_PREFIX)
STRING(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
IF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  IF(BUILD_TYPE)
    STRING(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  ELSE(BUILD_TYPE)
    SET(CMAKE_INSTALL_CONFIG_NAME "Release")
  ENDIF(BUILD_TYPE)
  MESSAGE(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
ENDIF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)

# Set the component getting installed.
IF(NOT CMAKE_INSTALL_COMPONENT)
  IF(COMPONENT)
    MESSAGE(STATUS "Install component: \"${COMPONENT}\"")
    SET(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  ELSE(COMPONENT)
    SET(CMAKE_INSTALL_COMPONENT)
  ENDIF(COMPONENT)
ENDIF(NOT CMAKE_INSTALL_COMPONENT)

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "libshogun-examples")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/shogun/examples/libshogun" TYPE FILE FILES
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/basic_minimal"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/classifier_libsvm"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/classifier_libsvm_probabilities"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/classifier_minimal_svm"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/classifier_mklmulticlass"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/classifier_gaussian_process_binary_classification"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/classifier_gaussiannaivebayes"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/classifier_multiclasslibsvm"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/classifier_qda"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/classifier_lda"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/classifier_multiclasslinearmachine"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/classifier_knn"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/kernel_gaussian"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/kernel_revlin"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/kernel_custom"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/library_dyn_int"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/library_gc_array"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/library_indirect_object"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/library_hash"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/parameter_set_from_parameters"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/parameter_iterate_float64"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/parameter_iterate_sgobject"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/parameter_modsel_parameters"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/evaluation_cross_validation_classification"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/evaluation_cross_validation_regression"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/evaluation_cross_validation_locked_comparison"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/evaluation_cross_validation_multiclass"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/evaluation_cross_validation_multiclass_mkl"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/evaluation_cross_validation_mkl_weight_storage"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/modelselection_parameter_combination_test"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/regression_gaussian_process_fitc"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/regression_gaussian_process_gaussian"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/regression_gaussian_process_sum"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/regression_gaussian_process_product"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/regression_gaussian_process_ard"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/regression_gaussian_process_laplace"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/regression_gaussian_process_simple_exact"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/modelselection_model_selection_parameters_test"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/modelselection_parameter_tree"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/modelselection_apply_parameter_tree"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/modelselection_grid_search_linear"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/features_subset_labels"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/modelselection_grid_search_kernel"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/modelselection_grid_search_string_kernel"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/modelselection_grid_search_multiclass_svm"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/modelselection_combined_kernel_sub_parameters"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/features_dense_real_modular"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/features_subset_stack"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/features_subset_simple_features"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/features_copy_subset_simple_features"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/features_copy_subset_sparse_features"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/labels_binary_fit_sigmoid"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/mathematics_confidence_intervals"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/clustering_kmeans"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/base_parameter_map"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/base_load_file_parameters"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/base_load_all_file_parameters"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/base_map_parameters"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/base_migration_type_conversion"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/base_migration_dropping_and_new"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/base_migration_multiple_dependencies"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/splitting_stratified_crossvalidation"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/splitting_standard_crossvalidation"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/library_set"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/library_map"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/mathematics_lapack"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/converter_locallylinearembedding"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/converter_localtangentspacealignment"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/converter_hessianlocallylinearembedding"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/converter_kernellocallylinearembedding"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/converter_multidimensionalscaling"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/converter_isomap"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/converter_jade_bss"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/converter_diffusionmaps"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/converter_laplacianeigenmaps"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/converter_neighborhoodpreservingembedding"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/converter_linearlocaltangentspacealignment"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/converter_localitypreservingprojections"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/converter_stochasticproximityembedding"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/converter_factoranalysis"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/serialization_basic_tests"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/serialization_multiclass_labels"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/kernel_machine_train_locked"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/statistics"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/transfer_multitaskleastsquaresregression"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/transfer_multitasklogisticregression"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/statistics_quadratic_time_mmd"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/statistics_linear_time_mmd"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/statistics_mmd_kernel_selection"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/statistics_hsic"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/classifier_featureblocklogisticregression"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/so_multiclass_BMRM"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/balanced_conditional_probability_tree"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/classifier_multiclass_ecoc"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/classifier_multiclass_ecoc_discriminant"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/classifier_multiclass_ecoc_random"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/streaming_from_dense"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/library_mldatahdf5"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/library_hdf5"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/library_serialization"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/classifier_svmlight_string_features_precomputed_kernel"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/classifier_mkl_svmlight_modelselection_bug"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/base_migration_new_buggy"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/regression_libsvr"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/classifier_multiclass_prob_heuristics"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/streaming_onlineliblinear"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/io_linereader"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/library_circularbuffer"
    "D:/Code/Common/shogun-3.0.0/build/examples/undocumented/libshogun/so_factorgraph"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "libshogun-examples")

