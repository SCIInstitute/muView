# In this example a kernel matrix is computed for a given real-valued data set.
# The kernel used is the Chi2 kernel which operates on real-valued vectors. It
# computes the chi-squared distance between sets of histograms. It is a very
# useful distance in image recognition (used to detect objects). The preprocessor
# NormOne, normalizes vectors to have norm 1.

library(shogun)

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))

#NormOne
print('NormOne')

feats_train <- RealFeatures(fm_train_real)
feats_test <- RealFeatures(fm_test_real)

preproc <- NormOne()
dump <- preproc$init(preproc, feats_train)
dump <- feats_train$add_preproc(feats_train, preproc)
dump <- feats_train$apply_preproc(feats_train)
dump <- feats_test$add_preproc(feats_test, preproc)
dump <- feats_test$apply_preproc(feats_test)

width <- 1.4
size_cache <- as.integer(10)

kernel <- Chi2Kernel(feats_train, feats_train, width, size_cache)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(kernel, feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()
