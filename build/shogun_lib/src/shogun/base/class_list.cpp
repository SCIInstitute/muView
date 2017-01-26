/*
 * THIS IS A GENERATED FILE!  DO NOT CHANGE THIS FILE!  CHANGE THE
 * CORRESPONDING TEMPLAT FILE, PLEASE!
-e  */

/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "base/class_list.h"

#include <string.h>

#include "kernel/Kernel.h"

#include "classifier/AveragedPerceptron.h"
#include "classifier/mkl/MKLClassification.h"
#include "classifier/mkl/MKLMulticlass.h"
#include "classifier/mkl/MKLOneClass.h"
#include "classifier/NearestCentroid.h"
#include "classifier/Perceptron.h"
#include "classifier/PluginEstimate.h"
#include "classifier/svm/GNPPLib.h"
#include "classifier/svm/GNPPSVM.h"
#include "classifier/svm/GPBTSVM.h"
#include "classifier/svm/LibSVM.h"
#include "classifier/svm/LibSVMOneClass.h"
#include "classifier/svm/MPDSVM.h"
#include "classifier/svm/OnlineLibLinear.h"
#include "classifier/svm/OnlineSVMSGD.h"
#include "classifier/svm/QPBSVMLib.h"
#include "classifier/svm/SGDQN.h"
#include "classifier/svm/SubGradientSVM.h"
#include "classifier/svm/SVM.h"
#include "classifier/svm/SVMLight.h"
#include "classifier/svm/SVMLightOneClass.h"
#include "classifier/svm/SVMLin.h"
#include "classifier/svm/SVMOcas.h"
#include "classifier/svm/SVMSGD.h"
#include "classifier/svm/WDSVMOcas.h"
#include "classifier/vw/cache/VwNativeCacheReader.h"
#include "classifier/vw/cache/VwNativeCacheWriter.h"
#include "classifier/vw/learners/VwAdaptiveLearner.h"
#include "classifier/vw/learners/VwNonAdaptiveLearner.h"
#include "classifier/vw/VowpalWabbit.h"
#include "classifier/vw/VwEnvironment.h"
#include "classifier/vw/VwParser.h"
#include "classifier/vw/VwRegressor.h"
#include "clustering/Hierarchical.h"
#include "clustering/KMeans.h"
#include "distance/AttenuatedEuclidianDistance.h"
#include "distance/BrayCurtisDistance.h"
#include "distance/CanberraMetric.h"
#include "distance/CanberraWordDistance.h"
#include "distance/ChebyshewMetric.h"
#include "distance/ChiSquareDistance.h"
#include "distance/CosineDistance.h"
#include "distance/CustomDistance.h"
#include "distance/EuclidianDistance.h"
#include "distance/GeodesicMetric.h"
#include "distance/HammingWordDistance.h"
#include "distance/JensenMetric.h"
#include "distance/KernelDistance.h"
#include "distance/ManhattanMetric.h"
#include "distance/ManhattanWordDistance.h"
#include "distance/MinkowskiMetric.h"
#include "distance/SparseEuclidianDistance.h"
#include "distance/TanimotoDistance.h"
#include "distributions/GHMM.h"
#include "distributions/Histogram.h"
#include "distributions/HMM.h"
#include "distributions/LinearHMM.h"
#include "distributions/PositionalPWM.h"
#include "evaluation/ClusteringAccuracy.h"
#include "evaluation/ClusteringMutualInformation.h"
#include "evaluation/ContingencyTableEvaluation.h"
#include "evaluation/CrossValidation.h"
#include "evaluation/CrossValidationSplitting.h"
#include "evaluation/MeanAbsoluteError.h"
#include "evaluation/MeanSquaredError.h"
#include "evaluation/MeanSquaredLogError.h"
#include "evaluation/MulticlassAccuracy.h"
#include "evaluation/PRCEvaluation.h"
#include "evaluation/ROCEvaluation.h"
#include "evaluation/StratifiedCrossValidationSplitting.h"
#include "features/Alphabet.h"
#include "features/BinnedDotFeatures.h"
#include "features/CombinedDotFeatures.h"
#include "features/CombinedFeatures.h"
#include "features/DenseFeatures.h"
#include "features/DummyFeatures.h"
#include "features/ExplicitSpecFeatures.h"
#include "features/FKFeatures.h"
#include "features/HashedWDFeatures.h"
#include "features/HashedWDFeaturesTransposed.h"
#include "features/ImplicitWeightedSpecFeatures.h"
#include "features/LBPPyrDotFeatures.h"
#include "features/PolyFeatures.h"
#include "features/RealFileFeatures.h"
#include "features/SNPFeatures.h"
#include "features/SparseFeatures.h"
#include "features/SparsePolyFeatures.h"
#include "features/StreamingDenseFeatures.h"
#include "features/StreamingSparseFeatures.h"
#include "features/StreamingStringFeatures.h"
#include "features/StreamingVwFeatures.h"
#include "features/StringFeatures.h"
#include "features/StringFileFeatures.h"
#include "features/Subset.h"
#include "features/SubsetStack.h"
#include "features/TOPFeatures.h"
#include "features/WDFeatures.h"
#include "io/AsciiFile.h"
#include "io/BinaryFile.h"
#include "io/BinaryStream.h"
#include "io/IOBuffer.h"
#include "io/MemoryMappedFile.h"
#include "io/ParseBuffer.h"
#include "io/SerializableAsciiFile.h"
#include "io/SimpleFile.h"
#include "io/StreamingAsciiFile.h"
#include "io/StreamingFile.h"
#include "io/StreamingFileFromDenseFeatures.h"
#include "io/StreamingFileFromFeatures.h"
#include "io/StreamingFileFromSparseFeatures.h"
#include "io/StreamingFileFromStringFeatures.h"
#include "io/StreamingVwCacheFile.h"
#include "io/StreamingVwFile.h"
#include "kernel/ANOVAKernel.h"
#include "kernel/AUCKernel.h"
#include "kernel/BesselKernel.h"
#include "kernel/CauchyKernel.h"
#include "kernel/Chi2Kernel.h"
#include "kernel/CircularKernel.h"
#include "kernel/CombinedKernel.h"
#include "kernel/CommUlongStringKernel.h"
#include "kernel/CommWordStringKernel.h"
#include "kernel/ConstKernel.h"
#include "kernel/CustomKernel.h"
#include "kernel/DiagKernel.h"
#include "kernel/DistanceKernel.h"
#include "kernel/DistantSegmentsKernel.h"
#include "kernel/ExponentialKernel.h"
#include "kernel/FixedDegreeStringKernel.h"
#include "kernel/GaussianKernel.h"
#include "kernel/GaussianMatchStringKernel.h"
#include "kernel/GaussianShiftKernel.h"
#include "kernel/GaussianShortRealKernel.h"
#include "kernel/HistogramIntersectionKernel.h"
#include "kernel/HistogramWordStringKernel.h"
#include "kernel/InverseMultiQuadricKernel.h"
#include "kernel/JensenShannonKernel.h"
#include "kernel/LinearKernel.h"
#include "kernel/LinearStringKernel.h"
#include "kernel/LocalAlignmentStringKernel.h"
#include "kernel/LocalityImprovedStringKernel.h"
#include "kernel/LogKernel.h"
#include "kernel/MatchWordStringKernel.h"
#include "kernel/MultiquadricKernel.h"
#include "kernel/normalizer/AvgDiagKernelNormalizer.h"
#include "kernel/normalizer/DiceKernelNormalizer.h"
#include "kernel/normalizer/FirstElementKernelNormalizer.h"
#include "kernel/normalizer/IdentityKernelNormalizer.h"
#include "kernel/normalizer/RidgeKernelNormalizer.h"
#include "kernel/normalizer/ScatterKernelNormalizer.h"
#include "kernel/normalizer/SqrtDiagKernelNormalizer.h"
#include "kernel/normalizer/TanimotoKernelNormalizer.h"
#include "kernel/normalizer/VarianceKernelNormalizer.h"
#include "kernel/normalizer/ZeroMeanCenterKernelNormalizer.h"
#include "kernel/OligoStringKernel.h"
#include "kernel/PolyKernel.h"
#include "kernel/PolyMatchStringKernel.h"
#include "kernel/PolyMatchWordStringKernel.h"
#include "kernel/PowerKernel.h"
#include "kernel/ProductKernel.h"
#include "kernel/PyramidChi2.h"
#include "kernel/RationalQuadraticKernel.h"
#include "kernel/RegulatoryModulesStringKernel.h"
#include "kernel/SalzbergWordStringKernel.h"
#include "kernel/SigmoidKernel.h"
#include "kernel/SimpleLocalityImprovedStringKernel.h"
#include "kernel/SNPStringKernel.h"
#include "kernel/SparseSpatialSampleStringKernel.h"
#include "kernel/SpectrumMismatchRBFKernel.h"
#include "kernel/SpectrumRBFKernel.h"
#include "kernel/SphericalKernel.h"
#include "kernel/SplineKernel.h"
#include "kernel/TensorProductPairKernel.h"
#include "kernel/TStudentKernel.h"
#include "kernel/WaveKernel.h"
#include "kernel/WaveletKernel.h"
#include "kernel/WeightedCommWordStringKernel.h"
#include "kernel/WeightedDegreePositionStringKernel.h"
#include "kernel/WeightedDegreeRBFKernel.h"
#include "kernel/WeightedDegreeStringKernel.h"
#include "labels/BinaryLabels.h"
#include "labels/MulticlassLabels.h"
#include "labels/RegressionLabels.h"
#include "labels/StructuredLabels.h"
#include "lib/BitString.h"
#include "lib/Cache.h"
#include "lib/Compressor.h"
#include "lib/DynamicArray.h"
#include "lib/DynamicObjectArray.h"
#include "lib/FibonacciHeap.h"
#include "lib/Hash.h"
#include "lib/List.h"
#include "lib/Set.h"
#include "lib/Signal.h"
#include "lib/StructuredData.h"
#include "lib/Time.h"
#include "loss/HingeLoss.h"
#include "loss/LogLoss.h"
#include "loss/LogLossMargin.h"
#include "loss/SmoothHingeLoss.h"
#include "loss/SquaredHingeLoss.h"
#include "loss/SquaredLoss.h"
#include "machine/BaseMulticlassMachine.h"
#include "machine/DistanceMachine.h"
#include "machine/KernelMachine.h"
#include "machine/KernelMulticlassMachine.h"
#include "machine/KernelStructuredOutputMachine.h"
#include "machine/LinearMachine.h"
#include "machine/LinearMulticlassMachine.h"
#include "machine/LinearStructuredOutputMachine.h"
#include "machine/Machine.h"
#include "machine/NativeMulticlassMachine.h"
#include "machine/OnlineLinearMachine.h"
#include "machine/SLEPMachine.h"
#include "machine/StructuredOutputMachine.h"
#include "mathematics/Math.h"
#include "mathematics/SparseInverseCovariance.h"
#include "mathematics/Statistics.h"
#include "modelselection/GridSearchModelSelection.h"
#include "modelselection/ModelSelectionParameters.h"
#include "modelselection/ParameterCombination.h"
#include "multiclass/ecoc/ECOCAEDDecoder.h"
#include "multiclass/ecoc/ECOCDiscriminantEncoder.h"
#include "multiclass/ecoc/ECOCEDDecoder.h"
#include "multiclass/ecoc/ECOCForestEncoder.h"
#include "multiclass/ecoc/ECOCHDDecoder.h"
#include "multiclass/ecoc/ECOCLLBDecoder.h"
#include "multiclass/ecoc/ECOCOVOEncoder.h"
#include "multiclass/ecoc/ECOCOVREncoder.h"
#include "multiclass/ecoc/ECOCRandomDenseEncoder.h"
#include "multiclass/ecoc/ECOCRandomSparseEncoder.h"
#include "multiclass/ecoc/ECOCStrategy.h"
#include "multiclass/GaussianNaiveBayes.h"
#include "multiclass/GMNPLib.h"
#include "multiclass/GMNPSVM.h"
#include "multiclass/KNN.h"
#include "multiclass/LaRank.h"
#include "multiclass/MulticlassLibSVM.h"
#include "multiclass/MulticlassOCAS.h"
#include "multiclass/MulticlassOneVsOneStrategy.h"
#include "multiclass/MulticlassOneVsRestStrategy.h"
#include "multiclass/MulticlassSVM.h"
#include "multiclass/ScatterSVM.h"
#include "multiclass/tree/BalancedConditionalProbabilityTree.h"
#include "multiclass/tree/RandomConditionalProbabilityTree.h"
#include "preprocessor/DecompressString.h"
#include "preprocessor/DimensionReductionPreprocessor.h"
#include "preprocessor/HomogeneousKernelMap.h"
#include "preprocessor/LogPlusOne.h"
#include "preprocessor/NormOne.h"
#include "preprocessor/PNorm.h"
#include "preprocessor/PruneVarSubMean.h"
#include "preprocessor/RandomFourierGaussPreproc.h"
#include "preprocessor/SortUlongString.h"
#include "preprocessor/SortWordString.h"
#include "preprocessor/SumOne.h"
#include "regression/gp/ExactInferenceMethod.h"
#include "regression/gp/GaussianLikelihood.h"
#include "regression/gp/ZeroMean.h"
#include "regression/svr/LibSVR.h"
#include "regression/svr/MKLRegression.h"
#include "regression/svr/SVRLight.h"
#include "statistics/LinearTimeMMD.h"
#include "statistics/QuadraticTimeMMD.h"
#include "statistics/StatisticalTest.h"
#include "structure/DualLibQPBMSOSVM.h"
#include "structure/DynProg.h"
#include "structure/IntronList.h"
#include "structure/MulticlassModel.h"
#include "structure/MulticlassRiskFunction.h"
#include "structure/MulticlassSOLabels.h"
#include "structure/Plif.h"
#include "structure/PlifArray.h"
#include "structure/PlifMatrix.h"
#include "structure/SegmentLoss.h"
#include "transfer/domain_adaptation/DomainAdaptationSVM.h"
#include "transfer/multitask/MultitaskKernelMaskNormalizer.h"
#include "transfer/multitask/MultitaskKernelMaskPairNormalizer.h"
#include "transfer/multitask/MultitaskKernelNormalizer.h"
#include "transfer/multitask/MultitaskKernelPlifNormalizer.h"
#include "transfer/multitask/MultitaskKernelTreeNormalizer.h"
#include "transfer/multitask/MultitaskLSRegression.h"
#include "transfer/multitask/Task.h"
#include "transfer/multitask/TaskGroup.h"
#include "transfer/multitask/TaskTree.h"
#include "ui/GUIClassifier.h"
#include "ui/GUIConverter.h"
#include "ui/GUIDistance.h"
#include "ui/GUIFeatures.h"
#include "ui/GUIHMM.h"
#include "ui/GUIKernel.h"
#include "ui/GUILabels.h"
#include "ui/GUIMath.h"
#include "ui/GUIPluginEstimate.h"
#include "ui/GUIPreprocessor.h"
#include "ui/GUIStructure.h"
#include "ui/GUITime.h"
using namespace shogun;

static CSGObject* __new_CAveragedPerceptron(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAveragedPerceptron(): NULL; }
static CSGObject* __new_CMKLClassification(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLClassification(): NULL; }
static CSGObject* __new_CMKLMulticlass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLMulticlass(): NULL; }
static CSGObject* __new_CMKLOneClass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLOneClass(): NULL; }
static CSGObject* __new_CNearestCentroid(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNearestCentroid(): NULL; }
static CSGObject* __new_CPerceptron(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPerceptron(): NULL; }
static CSGObject* __new_CPluginEstimate(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPluginEstimate(): NULL; }
static CSGObject* __new_CGNPPLib(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGNPPLib(): NULL; }
static CSGObject* __new_CGNPPSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGNPPSVM(): NULL; }
static CSGObject* __new_CGPBTSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGPBTSVM(): NULL; }
static CSGObject* __new_CLibSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVM(): NULL; }
static CSGObject* __new_CLibSVMOneClass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVMOneClass(): NULL; }
static CSGObject* __new_CMPDSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMPDSVM(): NULL; }
static CSGObject* __new_COnlineLibLinear(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COnlineLibLinear(): NULL; }
static CSGObject* __new_COnlineSVMSGD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COnlineSVMSGD(): NULL; }
static CSGObject* __new_CQPBSVMLib(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CQPBSVMLib(): NULL; }
static CSGObject* __new_CSGDQN(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSGDQN(): NULL; }
static CSGObject* __new_CSubGradientSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSubGradientSVM(): NULL; }
static CSGObject* __new_CSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVM(): NULL; }
static CSGObject* __new_CSVMLight(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMLight(): NULL; }
static CSGObject* __new_CSVMLightOneClass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMLightOneClass(): NULL; }
static CSGObject* __new_CSVMLin(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMLin(): NULL; }
static CSGObject* __new_CSVMOcas(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMOcas(): NULL; }
static CSGObject* __new_CSVMSGD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMSGD(): NULL; }
static CSGObject* __new_CWDSVMOcas(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWDSVMOcas(): NULL; }
static CSGObject* __new_CVwNativeCacheReader(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwNativeCacheReader(): NULL; }
static CSGObject* __new_CVwNativeCacheWriter(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwNativeCacheWriter(): NULL; }
static CSGObject* __new_CVwAdaptiveLearner(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwAdaptiveLearner(): NULL; }
static CSGObject* __new_CVwNonAdaptiveLearner(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwNonAdaptiveLearner(): NULL; }
static CSGObject* __new_CVowpalWabbit(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVowpalWabbit(): NULL; }
static CSGObject* __new_CVwEnvironment(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwEnvironment(): NULL; }
static CSGObject* __new_CVwParser(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwParser(): NULL; }
static CSGObject* __new_CVwRegressor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwRegressor(): NULL; }
static CSGObject* __new_CHierarchical(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHierarchical(): NULL; }
static CSGObject* __new_CKMeans(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKMeans(): NULL; }
static CSGObject* __new_CAttenuatedEuclidianDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAttenuatedEuclidianDistance(): NULL; }
static CSGObject* __new_CBrayCurtisDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBrayCurtisDistance(): NULL; }
static CSGObject* __new_CCanberraMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCanberraMetric(): NULL; }
static CSGObject* __new_CCanberraWordDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCanberraWordDistance(): NULL; }
static CSGObject* __new_CChebyshewMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CChebyshewMetric(): NULL; }
static CSGObject* __new_CChiSquareDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CChiSquareDistance(): NULL; }
static CSGObject* __new_CCosineDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCosineDistance(): NULL; }
static CSGObject* __new_CCustomDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCustomDistance(): NULL; }
static CSGObject* __new_CEuclidianDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CEuclidianDistance(): NULL; }
static CSGObject* __new_CGeodesicMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGeodesicMetric(): NULL; }
static CSGObject* __new_CHammingWordDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHammingWordDistance(): NULL; }
static CSGObject* __new_CJensenMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJensenMetric(): NULL; }
static CSGObject* __new_CKernelDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelDistance(): NULL; }
static CSGObject* __new_CManhattanMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CManhattanMetric(): NULL; }
static CSGObject* __new_CManhattanWordDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CManhattanWordDistance(): NULL; }
static CSGObject* __new_CMinkowskiMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMinkowskiMetric(): NULL; }
static CSGObject* __new_CSparseEuclidianDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparseEuclidianDistance(): NULL; }
static CSGObject* __new_CTanimotoDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTanimotoDistance(): NULL; }
static CSGObject* __new_CGHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGHMM(): NULL; }
static CSGObject* __new_CHistogram(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHistogram(): NULL; }
static CSGObject* __new_CHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHMM(): NULL; }
static CSGObject* __new_CLinearHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearHMM(): NULL; }
static CSGObject* __new_CPositionalPWM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPositionalPWM(): NULL; }
static CSGObject* __new_CClusteringAccuracy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CClusteringAccuracy(): NULL; }
static CSGObject* __new_CClusteringMutualInformation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CClusteringMutualInformation(): NULL; }
static CSGObject* __new_CContingencyTableEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CContingencyTableEvaluation(): NULL; }
static CSGObject* __new_CAccuracyMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAccuracyMeasure(): NULL; }
static CSGObject* __new_CErrorRateMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CErrorRateMeasure(): NULL; }
static CSGObject* __new_CBALMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBALMeasure(): NULL; }
static CSGObject* __new_CWRACCMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWRACCMeasure(): NULL; }
static CSGObject* __new_CF1Measure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CF1Measure(): NULL; }
static CSGObject* __new_CCrossCorrelationMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossCorrelationMeasure(): NULL; }
static CSGObject* __new_CRecallMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRecallMeasure(): NULL; }
static CSGObject* __new_CPrecisionMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPrecisionMeasure(): NULL; }
static CSGObject* __new_CSpecificityMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSpecificityMeasure(): NULL; }
static CSGObject* __new_CCrossValidation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidation(): NULL; }
static CSGObject* __new_CCrossValidationSplitting(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationSplitting(): NULL; }
static CSGObject* __new_CMeanAbsoluteError(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanAbsoluteError(): NULL; }
static CSGObject* __new_CMeanSquaredError(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanSquaredError(): NULL; }
static CSGObject* __new_CMeanSquaredLogError(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanSquaredLogError(): NULL; }
static CSGObject* __new_CMulticlassAccuracy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassAccuracy(): NULL; }
static CSGObject* __new_CPRCEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPRCEvaluation(): NULL; }
static CSGObject* __new_CROCEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CROCEvaluation(): NULL; }
static CSGObject* __new_CStratifiedCrossValidationSplitting(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStratifiedCrossValidationSplitting(): NULL; }
static CSGObject* __new_CAlphabet(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAlphabet(): NULL; }
static CSGObject* __new_CBinnedDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBinnedDotFeatures(): NULL; }
static CSGObject* __new_CCombinedDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCombinedDotFeatures(): NULL; }
static CSGObject* __new_CCombinedFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCombinedFeatures(): NULL; }
static CSGObject* __new_CDummyFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDummyFeatures(): NULL; }
static CSGObject* __new_CExplicitSpecFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CExplicitSpecFeatures(): NULL; }
static CSGObject* __new_CFKFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFKFeatures(): NULL; }
static CSGObject* __new_CHashedWDFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedWDFeatures(): NULL; }
static CSGObject* __new_CHashedWDFeaturesTransposed(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedWDFeaturesTransposed(): NULL; }
static CSGObject* __new_CImplicitWeightedSpecFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CImplicitWeightedSpecFeatures(): NULL; }
static CSGObject* __new_CLBPPyrDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLBPPyrDotFeatures(): NULL; }
static CSGObject* __new_CPolyFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyFeatures(): NULL; }
static CSGObject* __new_CRealFileFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRealFileFeatures(): NULL; }
static CSGObject* __new_CSNPFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSNPFeatures(): NULL; }
static CSGObject* __new_CSparsePolyFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparsePolyFeatures(): NULL; }
static CSGObject* __new_CStreamingVwFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingVwFeatures(): NULL; }
static CSGObject* __new_CSubset(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSubset(): NULL; }
static CSGObject* __new_CSubsetStack(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSubsetStack(): NULL; }
static CSGObject* __new_CTOPFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTOPFeatures(): NULL; }
static CSGObject* __new_CWDFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWDFeatures(): NULL; }
static CSGObject* __new_CAsciiFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAsciiFile(): NULL; }
static CSGObject* __new_CBinaryFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBinaryFile(): NULL; }
static CSGObject* __new_CIOBuffer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIOBuffer(): NULL; }
static CSGObject* __new_CSerializableAsciiFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSerializableAsciiFile(): NULL; }
static CSGObject* __new_CStreamingAsciiFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingAsciiFile(): NULL; }
static CSGObject* __new_CStreamingFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingFile(): NULL; }
static CSGObject* __new_CStreamingFileFromFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingFileFromFeatures(): NULL; }
static CSGObject* __new_CStreamingVwCacheFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingVwCacheFile(): NULL; }
static CSGObject* __new_CStreamingVwFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingVwFile(): NULL; }
static CSGObject* __new_CANOVAKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CANOVAKernel(): NULL; }
static CSGObject* __new_CAUCKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAUCKernel(): NULL; }
static CSGObject* __new_CBesselKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBesselKernel(): NULL; }
static CSGObject* __new_CCauchyKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCauchyKernel(): NULL; }
static CSGObject* __new_CChi2Kernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CChi2Kernel(): NULL; }
static CSGObject* __new_CCircularKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCircularKernel(): NULL; }
static CSGObject* __new_CCombinedKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCombinedKernel(): NULL; }
static CSGObject* __new_CCommUlongStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCommUlongStringKernel(): NULL; }
static CSGObject* __new_CCommWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCommWordStringKernel(): NULL; }
static CSGObject* __new_CConstKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CConstKernel(): NULL; }
static CSGObject* __new_CCustomKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCustomKernel(): NULL; }
static CSGObject* __new_CDiagKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDiagKernel(): NULL; }
static CSGObject* __new_CDistanceKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDistanceKernel(): NULL; }
static CSGObject* __new_CDistantSegmentsKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDistantSegmentsKernel(): NULL; }
static CSGObject* __new_CExponentialKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CExponentialKernel(): NULL; }
static CSGObject* __new_CFixedDegreeStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFixedDegreeStringKernel(): NULL; }
static CSGObject* __new_CGaussianKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianKernel(): NULL; }
static CSGObject* __new_CGaussianMatchStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianMatchStringKernel(): NULL; }
static CSGObject* __new_CGaussianShiftKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianShiftKernel(): NULL; }
static CSGObject* __new_CGaussianShortRealKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianShortRealKernel(): NULL; }
static CSGObject* __new_CHistogramIntersectionKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHistogramIntersectionKernel(): NULL; }
static CSGObject* __new_CHistogramWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHistogramWordStringKernel(): NULL; }
static CSGObject* __new_CInverseMultiQuadricKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CInverseMultiQuadricKernel(): NULL; }
static CSGObject* __new_CJensenShannonKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJensenShannonKernel(): NULL; }
static CSGObject* __new_CLinearKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearKernel(): NULL; }
static CSGObject* __new_CLinearStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearStringKernel(): NULL; }
static CSGObject* __new_CLocalAlignmentStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLocalAlignmentStringKernel(): NULL; }
static CSGObject* __new_CLocalityImprovedStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLocalityImprovedStringKernel(): NULL; }
static CSGObject* __new_CLogKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogKernel(): NULL; }
static CSGObject* __new_CMatchWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMatchWordStringKernel(): NULL; }
static CSGObject* __new_CMultiquadricKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultiquadricKernel(): NULL; }
static CSGObject* __new_CAvgDiagKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAvgDiagKernelNormalizer(): NULL; }
static CSGObject* __new_CDiceKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDiceKernelNormalizer(): NULL; }
static CSGObject* __new_CFirstElementKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFirstElementKernelNormalizer(): NULL; }
static CSGObject* __new_CIdentityKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIdentityKernelNormalizer(): NULL; }
static CSGObject* __new_CRidgeKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRidgeKernelNormalizer(): NULL; }
static CSGObject* __new_CScatterKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CScatterKernelNormalizer(): NULL; }
static CSGObject* __new_CSqrtDiagKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSqrtDiagKernelNormalizer(): NULL; }
static CSGObject* __new_CTanimotoKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTanimotoKernelNormalizer(): NULL; }
static CSGObject* __new_CVarianceKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVarianceKernelNormalizer(): NULL; }
static CSGObject* __new_CZeroMeanCenterKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CZeroMeanCenterKernelNormalizer(): NULL; }
static CSGObject* __new_COligoStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COligoStringKernel(): NULL; }
static CSGObject* __new_CPolyKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyKernel(): NULL; }
static CSGObject* __new_CPolyMatchStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyMatchStringKernel(): NULL; }
static CSGObject* __new_CPolyMatchWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyMatchWordStringKernel(): NULL; }
static CSGObject* __new_CPowerKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPowerKernel(): NULL; }
static CSGObject* __new_CProductKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CProductKernel(): NULL; }
static CSGObject* __new_CPyramidChi2(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPyramidChi2(): NULL; }
static CSGObject* __new_CRationalQuadraticKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRationalQuadraticKernel(): NULL; }
static CSGObject* __new_CRegulatoryModulesStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRegulatoryModulesStringKernel(): NULL; }
static CSGObject* __new_CSalzbergWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSalzbergWordStringKernel(): NULL; }
static CSGObject* __new_CSigmoidKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSigmoidKernel(): NULL; }
static CSGObject* __new_CSimpleLocalityImprovedStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSimpleLocalityImprovedStringKernel(): NULL; }
static CSGObject* __new_CSNPStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSNPStringKernel(): NULL; }
static CSGObject* __new_CSparseSpatialSampleStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparseSpatialSampleStringKernel(): NULL; }
static CSGObject* __new_CSpectrumMismatchRBFKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSpectrumMismatchRBFKernel(): NULL; }
static CSGObject* __new_CSpectrumRBFKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSpectrumRBFKernel(): NULL; }
static CSGObject* __new_CSphericalKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSphericalKernel(): NULL; }
static CSGObject* __new_CSplineKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSplineKernel(): NULL; }
static CSGObject* __new_CTensorProductPairKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTensorProductPairKernel(): NULL; }
static CSGObject* __new_CTStudentKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTStudentKernel(): NULL; }
static CSGObject* __new_CWaveKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWaveKernel(): NULL; }
static CSGObject* __new_CWaveletKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWaveletKernel(): NULL; }
static CSGObject* __new_CWeightedCommWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedCommWordStringKernel(): NULL; }
static CSGObject* __new_CWeightedDegreePositionStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedDegreePositionStringKernel(): NULL; }
static CSGObject* __new_CWeightedDegreeRBFKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedDegreeRBFKernel(): NULL; }
static CSGObject* __new_CWeightedDegreeStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedDegreeStringKernel(): NULL; }
static CSGObject* __new_CBinaryLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBinaryLabels(): NULL; }
static CSGObject* __new_CMulticlassLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassLabels(): NULL; }
static CSGObject* __new_CRegressionLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRegressionLabels(): NULL; }
static CSGObject* __new_CStructuredLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredLabels(): NULL; }
static CSGObject* __new_CBitString(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBitString(): NULL; }
static CSGObject* __new_CCompressor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCompressor(): NULL; }
static CSGObject* __new_CDynamicObjectArray(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDynamicObjectArray(): NULL; }
static CSGObject* __new_CFibonacciHeap(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFibonacciHeap(): NULL; }
static CSGObject* __new_CHash(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHash(): NULL; }
static CSGObject* __new_CListElement(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CListElement(): NULL; }
static CSGObject* __new_CList(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CList(): NULL; }
static CSGObject* __new_CSignal(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSignal(): NULL; }
static CSGObject* __new_CStructuredData(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredData(): NULL; }
static CSGObject* __new_CTime(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTime(): NULL; }
static CSGObject* __new_CHingeLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHingeLoss(): NULL; }
static CSGObject* __new_CLogLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogLoss(): NULL; }
static CSGObject* __new_CLogLossMargin(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogLossMargin(): NULL; }
static CSGObject* __new_CSmoothHingeLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSmoothHingeLoss(): NULL; }
static CSGObject* __new_CSquaredHingeLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSquaredHingeLoss(): NULL; }
static CSGObject* __new_CSquaredLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSquaredLoss(): NULL; }
static CSGObject* __new_CBaseMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBaseMulticlassMachine(): NULL; }
static CSGObject* __new_CDistanceMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDistanceMachine(): NULL; }
static CSGObject* __new_CKernelMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelMachine(): NULL; }
static CSGObject* __new_CKernelMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelMulticlassMachine(): NULL; }
static CSGObject* __new_CKernelStructuredOutputMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelStructuredOutputMachine(): NULL; }
static CSGObject* __new_CLinearMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearMachine(): NULL; }
static CSGObject* __new_CLinearMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearMulticlassMachine(): NULL; }
static CSGObject* __new_CLinearStructuredOutputMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearStructuredOutputMachine(): NULL; }
static CSGObject* __new_CMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMachine(): NULL; }
static CSGObject* __new_CNativeMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNativeMulticlassMachine(): NULL; }
static CSGObject* __new_COnlineLinearMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COnlineLinearMachine(): NULL; }
static CSGObject* __new_CSLEPMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSLEPMachine(): NULL; }
static CSGObject* __new_CStructuredOutputMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredOutputMachine(): NULL; }
static CSGObject* __new_CMath(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMath(): NULL; }
static CSGObject* __new_CSparseInverseCovariance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparseInverseCovariance(): NULL; }
static CSGObject* __new_CStatistics(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStatistics(): NULL; }
static CSGObject* __new_CGridSearchModelSelection(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGridSearchModelSelection(): NULL; }
static CSGObject* __new_CModelSelectionParameters(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CModelSelectionParameters(): NULL; }
static CSGObject* __new_CParameterCombination(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CParameterCombination(): NULL; }
static CSGObject* __new_CECOCAEDDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCAEDDecoder(): NULL; }
static CSGObject* __new_CECOCDiscriminantEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCDiscriminantEncoder(): NULL; }
static CSGObject* __new_CECOCEDDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCEDDecoder(): NULL; }
static CSGObject* __new_CECOCForestEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCForestEncoder(): NULL; }
static CSGObject* __new_CECOCHDDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCHDDecoder(): NULL; }
static CSGObject* __new_CECOCLLBDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCLLBDecoder(): NULL; }
static CSGObject* __new_CECOCOVOEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCOVOEncoder(): NULL; }
static CSGObject* __new_CECOCOVREncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCOVREncoder(): NULL; }
static CSGObject* __new_CECOCRandomDenseEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCRandomDenseEncoder(): NULL; }
static CSGObject* __new_CECOCRandomSparseEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCRandomSparseEncoder(): NULL; }
static CSGObject* __new_CECOCStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCStrategy(): NULL; }
static CSGObject* __new_CGaussianNaiveBayes(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianNaiveBayes(): NULL; }
static CSGObject* __new_CGMNPLib(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGMNPLib(): NULL; }
static CSGObject* __new_CGMNPSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGMNPSVM(): NULL; }
static CSGObject* __new_CKNN(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKNN(): NULL; }
static CSGObject* __new_CLaRank(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLaRank(): NULL; }
static CSGObject* __new_CMulticlassLibSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassLibSVM(): NULL; }
static CSGObject* __new_CMulticlassOCAS(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOCAS(): NULL; }
static CSGObject* __new_CMulticlassOneVsOneStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOneVsOneStrategy(): NULL; }
static CSGObject* __new_CMulticlassOneVsRestStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOneVsRestStrategy(): NULL; }
static CSGObject* __new_CMulticlassSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassSVM(): NULL; }
static CSGObject* __new_CThresholdRejectionStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CThresholdRejectionStrategy(): NULL; }
static CSGObject* __new_CDixonQTestRejectionStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDixonQTestRejectionStrategy(): NULL; }
static CSGObject* __new_CScatterSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CScatterSVM(): NULL; }
static CSGObject* __new_CBalancedConditionalProbabilityTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBalancedConditionalProbabilityTree(): NULL; }
static CSGObject* __new_CRandomConditionalProbabilityTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomConditionalProbabilityTree(): NULL; }
static CSGObject* __new_CDimensionReductionPreprocessor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDimensionReductionPreprocessor(): NULL; }
static CSGObject* __new_CHomogeneousKernelMap(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHomogeneousKernelMap(): NULL; }
static CSGObject* __new_CLogPlusOne(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogPlusOne(): NULL; }
static CSGObject* __new_CNormOne(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNormOne(): NULL; }
static CSGObject* __new_CPNorm(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPNorm(): NULL; }
static CSGObject* __new_CPruneVarSubMean(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPruneVarSubMean(): NULL; }
static CSGObject* __new_CRandomFourierGaussPreproc(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomFourierGaussPreproc(): NULL; }
static CSGObject* __new_CSortUlongString(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSortUlongString(): NULL; }
static CSGObject* __new_CSortWordString(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSortWordString(): NULL; }
static CSGObject* __new_CSumOne(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSumOne(): NULL; }
static CSGObject* __new_CExactInferenceMethod(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CExactInferenceMethod(): NULL; }
static CSGObject* __new_CGaussianLikelihood(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianLikelihood(): NULL; }
static CSGObject* __new_CZeroMean(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CZeroMean(): NULL; }
static CSGObject* __new_CLibSVR(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVR(): NULL; }
static CSGObject* __new_CMKLRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLRegression(): NULL; }
static CSGObject* __new_CSVRLight(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVRLight(): NULL; }
static CSGObject* __new_CLinearTimeMMD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearTimeMMD(): NULL; }
static CSGObject* __new_CQuadraticTimeMMD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CQuadraticTimeMMD(): NULL; }
static CSGObject* __new_CStatisticalTest(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStatisticalTest(): NULL; }
static CSGObject* __new_CDualLibQPBMSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDualLibQPBMSOSVM(): NULL; }
static CSGObject* __new_CDynProg(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDynProg(): NULL; }
static CSGObject* __new_CIntronList(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIntronList(): NULL; }
static CSGObject* __new_CMulticlassModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassModel(): NULL; }
static CSGObject* __new_CMulticlassRiskFunction(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassRiskFunction(): NULL; }
static CSGObject* __new_CMulticlassSOLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassSOLabels(): NULL; }
static CSGObject* __new_CPlif(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPlif(): NULL; }
static CSGObject* __new_CPlifArray(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPlifArray(): NULL; }
static CSGObject* __new_CPlifMatrix(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPlifMatrix(): NULL; }
static CSGObject* __new_CSegmentLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSegmentLoss(): NULL; }
static CSGObject* __new_CDomainAdaptationSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDomainAdaptationSVM(): NULL; }
static CSGObject* __new_CMultitaskKernelMaskNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelMaskNormalizer(): NULL; }
static CSGObject* __new_CMultitaskKernelMaskPairNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelMaskPairNormalizer(): NULL; }
static CSGObject* __new_CMultitaskKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelNormalizer(): NULL; }
static CSGObject* __new_CMultitaskKernelPlifNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelPlifNormalizer(): NULL; }
static CSGObject* __new_CNode(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNode(): NULL; }
static CSGObject* __new_CTaxonomy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTaxonomy(): NULL; }
static CSGObject* __new_CMultitaskKernelTreeNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelTreeNormalizer(): NULL; }
static CSGObject* __new_CMultitaskLSRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskLSRegression(): NULL; }
static CSGObject* __new_CTask(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTask(): NULL; }
static CSGObject* __new_CTaskGroup(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTaskGroup(): NULL; }
static CSGObject* __new_CTaskTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTaskTree(): NULL; }
static CSGObject* __new_CGUIClassifier(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIClassifier(): NULL; }
static CSGObject* __new_CGUIConverter(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIConverter(): NULL; }
static CSGObject* __new_CGUIDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIDistance(): NULL; }
static CSGObject* __new_CGUIFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIFeatures(): NULL; }
static CSGObject* __new_CGUIHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIHMM(): NULL; }
static CSGObject* __new_CGUIKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIKernel(): NULL; }
static CSGObject* __new_CGUILabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUILabels(): NULL; }
static CSGObject* __new_CGUIMath(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIMath(): NULL; }
static CSGObject* __new_CGUIPluginEstimate(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIPluginEstimate(): NULL; }
static CSGObject* __new_CGUIPreprocessor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIPreprocessor(): NULL; }
static CSGObject* __new_CGUIStructure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIStructure(): NULL; }
static CSGObject* __new_CGUITime(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUITime(): NULL; }
static CSGObject* __new_CDenseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CDenseFeatures<bool>();
		case PT_CHAR: return new CDenseFeatures<char>();
		case PT_INT8: return new CDenseFeatures<int8_t>();
		case PT_UINT8: return new CDenseFeatures<uint8_t>();
		case PT_INT16: return new CDenseFeatures<int16_t>();
		case PT_UINT16: return new CDenseFeatures<uint16_t>();
		case PT_INT32: return new CDenseFeatures<int32_t>();
		case PT_UINT32: return new CDenseFeatures<uint32_t>();
		case PT_INT64: return new CDenseFeatures<int64_t>();
		case PT_UINT64: return new CDenseFeatures<uint64_t>();
		case PT_FLOAT32: return new CDenseFeatures<float32_t>();
		case PT_FLOAT64: return new CDenseFeatures<float64_t>();
		case PT_FLOATMAX: return new CDenseFeatures<floatmax_t>();
		case PT_SGOBJECT: return NULL;
	}
	return NULL;
}
static CSGObject* __new_CSparseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CSparseFeatures<bool>();
		case PT_CHAR: return new CSparseFeatures<char>();
		case PT_INT8: return new CSparseFeatures<int8_t>();
		case PT_UINT8: return new CSparseFeatures<uint8_t>();
		case PT_INT16: return new CSparseFeatures<int16_t>();
		case PT_UINT16: return new CSparseFeatures<uint16_t>();
		case PT_INT32: return new CSparseFeatures<int32_t>();
		case PT_UINT32: return new CSparseFeatures<uint32_t>();
		case PT_INT64: return new CSparseFeatures<int64_t>();
		case PT_UINT64: return new CSparseFeatures<uint64_t>();
		case PT_FLOAT32: return new CSparseFeatures<float32_t>();
		case PT_FLOAT64: return new CSparseFeatures<float64_t>();
		case PT_FLOATMAX: return new CSparseFeatures<floatmax_t>();
		case PT_SGOBJECT: return NULL;
	}
	return NULL;
}
static CSGObject* __new_CStreamingDenseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStreamingDenseFeatures<bool>();
		case PT_CHAR: return new CStreamingDenseFeatures<char>();
		case PT_INT8: return new CStreamingDenseFeatures<int8_t>();
		case PT_UINT8: return new CStreamingDenseFeatures<uint8_t>();
		case PT_INT16: return new CStreamingDenseFeatures<int16_t>();
		case PT_UINT16: return new CStreamingDenseFeatures<uint16_t>();
		case PT_INT32: return new CStreamingDenseFeatures<int32_t>();
		case PT_UINT32: return new CStreamingDenseFeatures<uint32_t>();
		case PT_INT64: return new CStreamingDenseFeatures<int64_t>();
		case PT_UINT64: return new CStreamingDenseFeatures<uint64_t>();
		case PT_FLOAT32: return new CStreamingDenseFeatures<float32_t>();
		case PT_FLOAT64: return new CStreamingDenseFeatures<float64_t>();
		case PT_FLOATMAX: return new CStreamingDenseFeatures<floatmax_t>();
		case PT_SGOBJECT: return NULL;
	}
	return NULL;
}
static CSGObject* __new_CStreamingSparseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStreamingSparseFeatures<bool>();
		case PT_CHAR: return new CStreamingSparseFeatures<char>();
		case PT_INT8: return new CStreamingSparseFeatures<int8_t>();
		case PT_UINT8: return new CStreamingSparseFeatures<uint8_t>();
		case PT_INT16: return new CStreamingSparseFeatures<int16_t>();
		case PT_UINT16: return new CStreamingSparseFeatures<uint16_t>();
		case PT_INT32: return new CStreamingSparseFeatures<int32_t>();
		case PT_UINT32: return new CStreamingSparseFeatures<uint32_t>();
		case PT_INT64: return new CStreamingSparseFeatures<int64_t>();
		case PT_UINT64: return new CStreamingSparseFeatures<uint64_t>();
		case PT_FLOAT32: return new CStreamingSparseFeatures<float32_t>();
		case PT_FLOAT64: return new CStreamingSparseFeatures<float64_t>();
		case PT_FLOATMAX: return new CStreamingSparseFeatures<floatmax_t>();
		case PT_SGOBJECT: return NULL;
	}
	return NULL;
}
static CSGObject* __new_CStreamingStringFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStreamingStringFeatures<bool>();
		case PT_CHAR: return new CStreamingStringFeatures<char>();
		case PT_INT8: return new CStreamingStringFeatures<int8_t>();
		case PT_UINT8: return new CStreamingStringFeatures<uint8_t>();
		case PT_INT16: return new CStreamingStringFeatures<int16_t>();
		case PT_UINT16: return new CStreamingStringFeatures<uint16_t>();
		case PT_INT32: return new CStreamingStringFeatures<int32_t>();
		case PT_UINT32: return new CStreamingStringFeatures<uint32_t>();
		case PT_INT64: return new CStreamingStringFeatures<int64_t>();
		case PT_UINT64: return new CStreamingStringFeatures<uint64_t>();
		case PT_FLOAT32: return new CStreamingStringFeatures<float32_t>();
		case PT_FLOAT64: return new CStreamingStringFeatures<float64_t>();
		case PT_FLOATMAX: return new CStreamingStringFeatures<floatmax_t>();
		case PT_SGOBJECT: return NULL;
	}
	return NULL;
}
static CSGObject* __new_CStringFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStringFeatures<bool>();
		case PT_CHAR: return new CStringFeatures<char>();
		case PT_INT8: return new CStringFeatures<int8_t>();
		case PT_UINT8: return new CStringFeatures<uint8_t>();
		case PT_INT16: return new CStringFeatures<int16_t>();
		case PT_UINT16: return new CStringFeatures<uint16_t>();
		case PT_INT32: return new CStringFeatures<int32_t>();
		case PT_UINT32: return new CStringFeatures<uint32_t>();
		case PT_INT64: return new CStringFeatures<int64_t>();
		case PT_UINT64: return new CStringFeatures<uint64_t>();
		case PT_FLOAT32: return new CStringFeatures<float32_t>();
		case PT_FLOAT64: return new CStringFeatures<float64_t>();
		case PT_FLOATMAX: return new CStringFeatures<floatmax_t>();
		case PT_SGOBJECT: return NULL;
	}
	return NULL;
}
static CSGObject* __new_CStringFileFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStringFileFeatures<bool>();
		case PT_CHAR: return new CStringFileFeatures<char>();
		case PT_INT8: return new CStringFileFeatures<int8_t>();
		case PT_UINT8: return new CStringFileFeatures<uint8_t>();
		case PT_INT16: return new CStringFileFeatures<int16_t>();
		case PT_UINT16: return new CStringFileFeatures<uint16_t>();
		case PT_INT32: return new CStringFileFeatures<int32_t>();
		case PT_UINT32: return new CStringFileFeatures<uint32_t>();
		case PT_INT64: return new CStringFileFeatures<int64_t>();
		case PT_UINT64: return new CStringFileFeatures<uint64_t>();
		case PT_FLOAT32: return new CStringFileFeatures<float32_t>();
		case PT_FLOAT64: return new CStringFileFeatures<float64_t>();
		case PT_FLOATMAX: return new CStringFileFeatures<floatmax_t>();
		case PT_SGOBJECT: return NULL;
	}
	return NULL;
}
static CSGObject* __new_CBinaryStream(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CBinaryStream<bool>();
		case PT_CHAR: return new CBinaryStream<char>();
		case PT_INT8: return new CBinaryStream<int8_t>();
		case PT_UINT8: return new CBinaryStream<uint8_t>();
		case PT_INT16: return new CBinaryStream<int16_t>();
		case PT_UINT16: return new CBinaryStream<uint16_t>();
		case PT_INT32: return new CBinaryStream<int32_t>();
		case PT_UINT32: return new CBinaryStream<uint32_t>();
		case PT_INT64: return new CBinaryStream<int64_t>();
		case PT_UINT64: return new CBinaryStream<uint64_t>();
		case PT_FLOAT32: return new CBinaryStream<float32_t>();
		case PT_FLOAT64: return new CBinaryStream<float64_t>();
		case PT_FLOATMAX: return new CBinaryStream<floatmax_t>();
		case PT_SGOBJECT: return NULL;
	}
	return NULL;
}
static CSGObject* __new_CMemoryMappedFile(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CMemoryMappedFile<bool>();
		case PT_CHAR: return new CMemoryMappedFile<char>();
		case PT_INT8: return new CMemoryMappedFile<int8_t>();
		case PT_UINT8: return new CMemoryMappedFile<uint8_t>();
		case PT_INT16: return new CMemoryMappedFile<int16_t>();
		case PT_UINT16: return new CMemoryMappedFile<uint16_t>();
		case PT_INT32: return new CMemoryMappedFile<int32_t>();
		case PT_UINT32: return new CMemoryMappedFile<uint32_t>();
		case PT_INT64: return new CMemoryMappedFile<int64_t>();
		case PT_UINT64: return new CMemoryMappedFile<uint64_t>();
		case PT_FLOAT32: return new CMemoryMappedFile<float32_t>();
		case PT_FLOAT64: return new CMemoryMappedFile<float64_t>();
		case PT_FLOATMAX: return new CMemoryMappedFile<floatmax_t>();
		case PT_SGOBJECT: return NULL;
	}
	return NULL;
}
static CSGObject* __new_CParseBuffer(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CParseBuffer<bool>();
		case PT_CHAR: return new CParseBuffer<char>();
		case PT_INT8: return new CParseBuffer<int8_t>();
		case PT_UINT8: return new CParseBuffer<uint8_t>();
		case PT_INT16: return new CParseBuffer<int16_t>();
		case PT_UINT16: return new CParseBuffer<uint16_t>();
		case PT_INT32: return new CParseBuffer<int32_t>();
		case PT_UINT32: return new CParseBuffer<uint32_t>();
		case PT_INT64: return new CParseBuffer<int64_t>();
		case PT_UINT64: return new CParseBuffer<uint64_t>();
		case PT_FLOAT32: return new CParseBuffer<float32_t>();
		case PT_FLOAT64: return new CParseBuffer<float64_t>();
		case PT_FLOATMAX: return new CParseBuffer<floatmax_t>();
		case PT_SGOBJECT: return NULL;
	}
	return NULL;
}
static CSGObject* __new_CSimpleFile(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CSimpleFile<bool>();
		case PT_CHAR: return new CSimpleFile<char>();
		case PT_INT8: return new CSimpleFile<int8_t>();
		case PT_UINT8: return new CSimpleFile<uint8_t>();
		case PT_INT16: return new CSimpleFile<int16_t>();
		case PT_UINT16: return new CSimpleFile<uint16_t>();
		case PT_INT32: return new CSimpleFile<int32_t>();
		case PT_UINT32: return new CSimpleFile<uint32_t>();
		case PT_INT64: return new CSimpleFile<int64_t>();
		case PT_UINT64: return new CSimpleFile<uint64_t>();
		case PT_FLOAT32: return new CSimpleFile<float32_t>();
		case PT_FLOAT64: return new CSimpleFile<float64_t>();
		case PT_FLOATMAX: return new CSimpleFile<floatmax_t>();
		case PT_SGOBJECT: return NULL;
	}
	return NULL;
}
static CSGObject* __new_CStreamingFileFromDenseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStreamingFileFromDenseFeatures<bool>();
		case PT_CHAR: return new CStreamingFileFromDenseFeatures<char>();
		case PT_INT8: return new CStreamingFileFromDenseFeatures<int8_t>();
		case PT_UINT8: return new CStreamingFileFromDenseFeatures<uint8_t>();
		case PT_INT16: return new CStreamingFileFromDenseFeatures<int16_t>();
		case PT_UINT16: return new CStreamingFileFromDenseFeatures<uint16_t>();
		case PT_INT32: return new CStreamingFileFromDenseFeatures<int32_t>();
		case PT_UINT32: return new CStreamingFileFromDenseFeatures<uint32_t>();
		case PT_INT64: return new CStreamingFileFromDenseFeatures<int64_t>();
		case PT_UINT64: return new CStreamingFileFromDenseFeatures<uint64_t>();
		case PT_FLOAT32: return new CStreamingFileFromDenseFeatures<float32_t>();
		case PT_FLOAT64: return new CStreamingFileFromDenseFeatures<float64_t>();
		case PT_FLOATMAX: return new CStreamingFileFromDenseFeatures<floatmax_t>();
		case PT_SGOBJECT: return NULL;
	}
	return NULL;
}
static CSGObject* __new_CStreamingFileFromSparseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStreamingFileFromSparseFeatures<bool>();
		case PT_CHAR: return new CStreamingFileFromSparseFeatures<char>();
		case PT_INT8: return new CStreamingFileFromSparseFeatures<int8_t>();
		case PT_UINT8: return new CStreamingFileFromSparseFeatures<uint8_t>();
		case PT_INT16: return new CStreamingFileFromSparseFeatures<int16_t>();
		case PT_UINT16: return new CStreamingFileFromSparseFeatures<uint16_t>();
		case PT_INT32: return new CStreamingFileFromSparseFeatures<int32_t>();
		case PT_UINT32: return new CStreamingFileFromSparseFeatures<uint32_t>();
		case PT_INT64: return new CStreamingFileFromSparseFeatures<int64_t>();
		case PT_UINT64: return new CStreamingFileFromSparseFeatures<uint64_t>();
		case PT_FLOAT32: return new CStreamingFileFromSparseFeatures<float32_t>();
		case PT_FLOAT64: return new CStreamingFileFromSparseFeatures<float64_t>();
		case PT_FLOATMAX: return new CStreamingFileFromSparseFeatures<floatmax_t>();
		case PT_SGOBJECT: return NULL;
	}
	return NULL;
}
static CSGObject* __new_CStreamingFileFromStringFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStreamingFileFromStringFeatures<bool>();
		case PT_CHAR: return new CStreamingFileFromStringFeatures<char>();
		case PT_INT8: return new CStreamingFileFromStringFeatures<int8_t>();
		case PT_UINT8: return new CStreamingFileFromStringFeatures<uint8_t>();
		case PT_INT16: return new CStreamingFileFromStringFeatures<int16_t>();
		case PT_UINT16: return new CStreamingFileFromStringFeatures<uint16_t>();
		case PT_INT32: return new CStreamingFileFromStringFeatures<int32_t>();
		case PT_UINT32: return new CStreamingFileFromStringFeatures<uint32_t>();
		case PT_INT64: return new CStreamingFileFromStringFeatures<int64_t>();
		case PT_UINT64: return new CStreamingFileFromStringFeatures<uint64_t>();
		case PT_FLOAT32: return new CStreamingFileFromStringFeatures<float32_t>();
		case PT_FLOAT64: return new CStreamingFileFromStringFeatures<float64_t>();
		case PT_FLOATMAX: return new CStreamingFileFromStringFeatures<floatmax_t>();
		case PT_SGOBJECT: return NULL;
	}
	return NULL;
}
static CSGObject* __new_CCache(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CCache<bool>();
		case PT_CHAR: return new CCache<char>();
		case PT_INT8: return new CCache<int8_t>();
		case PT_UINT8: return new CCache<uint8_t>();
		case PT_INT16: return new CCache<int16_t>();
		case PT_UINT16: return new CCache<uint16_t>();
		case PT_INT32: return new CCache<int32_t>();
		case PT_UINT32: return new CCache<uint32_t>();
		case PT_INT64: return new CCache<int64_t>();
		case PT_UINT64: return new CCache<uint64_t>();
		case PT_FLOAT32: return new CCache<float32_t>();
		case PT_FLOAT64: return new CCache<float64_t>();
		case PT_FLOATMAX: return new CCache<floatmax_t>();
		case PT_SGOBJECT: return NULL;
	}
	return NULL;
}
static CSGObject* __new_CDynamicArray(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CDynamicArray<bool>();
		case PT_CHAR: return new CDynamicArray<char>();
		case PT_INT8: return new CDynamicArray<int8_t>();
		case PT_UINT8: return new CDynamicArray<uint8_t>();
		case PT_INT16: return new CDynamicArray<int16_t>();
		case PT_UINT16: return new CDynamicArray<uint16_t>();
		case PT_INT32: return new CDynamicArray<int32_t>();
		case PT_UINT32: return new CDynamicArray<uint32_t>();
		case PT_INT64: return new CDynamicArray<int64_t>();
		case PT_UINT64: return new CDynamicArray<uint64_t>();
		case PT_FLOAT32: return new CDynamicArray<float32_t>();
		case PT_FLOAT64: return new CDynamicArray<float64_t>();
		case PT_FLOATMAX: return new CDynamicArray<floatmax_t>();
		case PT_SGOBJECT: return NULL;
	}
	return NULL;
}
static CSGObject* __new_CSet(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CSet<bool>();
		case PT_CHAR: return new CSet<char>();
		case PT_INT8: return new CSet<int8_t>();
		case PT_UINT8: return new CSet<uint8_t>();
		case PT_INT16: return new CSet<int16_t>();
		case PT_UINT16: return new CSet<uint16_t>();
		case PT_INT32: return new CSet<int32_t>();
		case PT_UINT32: return new CSet<uint32_t>();
		case PT_INT64: return new CSet<int64_t>();
		case PT_UINT64: return new CSet<uint64_t>();
		case PT_FLOAT32: return new CSet<float32_t>();
		case PT_FLOAT64: return new CSet<float64_t>();
		case PT_FLOATMAX: return new CSet<floatmax_t>();
		case PT_SGOBJECT: return NULL;
	}
	return NULL;
}
static CSGObject* __new_CDecompressString(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CDecompressString<bool>();
		case PT_CHAR: return new CDecompressString<char>();
		case PT_INT8: return new CDecompressString<int8_t>();
		case PT_UINT8: return new CDecompressString<uint8_t>();
		case PT_INT16: return new CDecompressString<int16_t>();
		case PT_UINT16: return new CDecompressString<uint16_t>();
		case PT_INT32: return new CDecompressString<int32_t>();
		case PT_UINT32: return new CDecompressString<uint32_t>();
		case PT_INT64: return new CDecompressString<int64_t>();
		case PT_UINT64: return new CDecompressString<uint64_t>();
		case PT_FLOAT32: return new CDecompressString<float32_t>();
		case PT_FLOAT64: return new CDecompressString<float64_t>();
		case PT_FLOATMAX: return new CDecompressString<floatmax_t>();
		case PT_SGOBJECT: return NULL;
	}
	return NULL;
}
typedef CSGObject* (*new_sgserializable_t)(EPrimitiveType generic);
#ifndef DOXYGEN_SHOULD_SKIP_THIS
typedef struct
{
	const char* m_class_name;
	new_sgserializable_t m_new_sgserializable;
} class_list_entry_t;
#endif

static class_list_entry_t class_list[] = {
{"AveragedPerceptron", __new_CAveragedPerceptron},
{"MKLClassification", __new_CMKLClassification},
{"MKLMulticlass", __new_CMKLMulticlass},
{"MKLOneClass", __new_CMKLOneClass},
{"NearestCentroid", __new_CNearestCentroid},
{"Perceptron", __new_CPerceptron},
{"PluginEstimate", __new_CPluginEstimate},
{"GNPPLib", __new_CGNPPLib},
{"GNPPSVM", __new_CGNPPSVM},
{"GPBTSVM", __new_CGPBTSVM},
{"LibSVM", __new_CLibSVM},
{"LibSVMOneClass", __new_CLibSVMOneClass},
{"MPDSVM", __new_CMPDSVM},
{"OnlineLibLinear", __new_COnlineLibLinear},
{"OnlineSVMSGD", __new_COnlineSVMSGD},
{"QPBSVMLib", __new_CQPBSVMLib},
{"SGDQN", __new_CSGDQN},
{"SubGradientSVM", __new_CSubGradientSVM},
{"SVM", __new_CSVM},
{"SVMLight", __new_CSVMLight},
{"SVMLightOneClass", __new_CSVMLightOneClass},
{"SVMLin", __new_CSVMLin},
{"SVMOcas", __new_CSVMOcas},
{"SVMSGD", __new_CSVMSGD},
{"WDSVMOcas", __new_CWDSVMOcas},
{"VwNativeCacheReader", __new_CVwNativeCacheReader},
{"VwNativeCacheWriter", __new_CVwNativeCacheWriter},
{"VwAdaptiveLearner", __new_CVwAdaptiveLearner},
{"VwNonAdaptiveLearner", __new_CVwNonAdaptiveLearner},
{"VowpalWabbit", __new_CVowpalWabbit},
{"VwEnvironment", __new_CVwEnvironment},
{"VwParser", __new_CVwParser},
{"VwRegressor", __new_CVwRegressor},
{"Hierarchical", __new_CHierarchical},
{"KMeans", __new_CKMeans},
{"AttenuatedEuclidianDistance", __new_CAttenuatedEuclidianDistance},
{"BrayCurtisDistance", __new_CBrayCurtisDistance},
{"CanberraMetric", __new_CCanberraMetric},
{"CanberraWordDistance", __new_CCanberraWordDistance},
{"ChebyshewMetric", __new_CChebyshewMetric},
{"ChiSquareDistance", __new_CChiSquareDistance},
{"CosineDistance", __new_CCosineDistance},
{"CustomDistance", __new_CCustomDistance},
{"EuclidianDistance", __new_CEuclidianDistance},
{"GeodesicMetric", __new_CGeodesicMetric},
{"HammingWordDistance", __new_CHammingWordDistance},
{"JensenMetric", __new_CJensenMetric},
{"KernelDistance", __new_CKernelDistance},
{"ManhattanMetric", __new_CManhattanMetric},
{"ManhattanWordDistance", __new_CManhattanWordDistance},
{"MinkowskiMetric", __new_CMinkowskiMetric},
{"SparseEuclidianDistance", __new_CSparseEuclidianDistance},
{"TanimotoDistance", __new_CTanimotoDistance},
{"GHMM", __new_CGHMM},
{"Histogram", __new_CHistogram},
{"HMM", __new_CHMM},
{"LinearHMM", __new_CLinearHMM},
{"PositionalPWM", __new_CPositionalPWM},
{"ClusteringAccuracy", __new_CClusteringAccuracy},
{"ClusteringMutualInformation", __new_CClusteringMutualInformation},
{"ContingencyTableEvaluation", __new_CContingencyTableEvaluation},
{"AccuracyMeasure", __new_CAccuracyMeasure},
{"ErrorRateMeasure", __new_CErrorRateMeasure},
{"BALMeasure", __new_CBALMeasure},
{"WRACCMeasure", __new_CWRACCMeasure},
{"F1Measure", __new_CF1Measure},
{"CrossCorrelationMeasure", __new_CCrossCorrelationMeasure},
{"RecallMeasure", __new_CRecallMeasure},
{"PrecisionMeasure", __new_CPrecisionMeasure},
{"SpecificityMeasure", __new_CSpecificityMeasure},
{"CrossValidation", __new_CCrossValidation},
{"CrossValidationSplitting", __new_CCrossValidationSplitting},
{"MeanAbsoluteError", __new_CMeanAbsoluteError},
{"MeanSquaredError", __new_CMeanSquaredError},
{"MeanSquaredLogError", __new_CMeanSquaredLogError},
{"MulticlassAccuracy", __new_CMulticlassAccuracy},
{"PRCEvaluation", __new_CPRCEvaluation},
{"ROCEvaluation", __new_CROCEvaluation},
{"StratifiedCrossValidationSplitting", __new_CStratifiedCrossValidationSplitting},
{"Alphabet", __new_CAlphabet},
{"BinnedDotFeatures", __new_CBinnedDotFeatures},
{"CombinedDotFeatures", __new_CCombinedDotFeatures},
{"CombinedFeatures", __new_CCombinedFeatures},
{"DummyFeatures", __new_CDummyFeatures},
{"ExplicitSpecFeatures", __new_CExplicitSpecFeatures},
{"FKFeatures", __new_CFKFeatures},
{"HashedWDFeatures", __new_CHashedWDFeatures},
{"HashedWDFeaturesTransposed", __new_CHashedWDFeaturesTransposed},
{"ImplicitWeightedSpecFeatures", __new_CImplicitWeightedSpecFeatures},
{"LBPPyrDotFeatures", __new_CLBPPyrDotFeatures},
{"PolyFeatures", __new_CPolyFeatures},
{"RealFileFeatures", __new_CRealFileFeatures},
{"SNPFeatures", __new_CSNPFeatures},
{"SparsePolyFeatures", __new_CSparsePolyFeatures},
{"StreamingVwFeatures", __new_CStreamingVwFeatures},
{"Subset", __new_CSubset},
{"SubsetStack", __new_CSubsetStack},
{"TOPFeatures", __new_CTOPFeatures},
{"WDFeatures", __new_CWDFeatures},
{"AsciiFile", __new_CAsciiFile},
{"BinaryFile", __new_CBinaryFile},
{"IOBuffer", __new_CIOBuffer},
{"SerializableAsciiFile", __new_CSerializableAsciiFile},
{"StreamingAsciiFile", __new_CStreamingAsciiFile},
{"StreamingFile", __new_CStreamingFile},
{"StreamingFileFromFeatures", __new_CStreamingFileFromFeatures},
{"StreamingVwCacheFile", __new_CStreamingVwCacheFile},
{"StreamingVwFile", __new_CStreamingVwFile},
{"ANOVAKernel", __new_CANOVAKernel},
{"AUCKernel", __new_CAUCKernel},
{"BesselKernel", __new_CBesselKernel},
{"CauchyKernel", __new_CCauchyKernel},
{"Chi2Kernel", __new_CChi2Kernel},
{"CircularKernel", __new_CCircularKernel},
{"CombinedKernel", __new_CCombinedKernel},
{"CommUlongStringKernel", __new_CCommUlongStringKernel},
{"CommWordStringKernel", __new_CCommWordStringKernel},
{"ConstKernel", __new_CConstKernel},
{"CustomKernel", __new_CCustomKernel},
{"DiagKernel", __new_CDiagKernel},
{"DistanceKernel", __new_CDistanceKernel},
{"DistantSegmentsKernel", __new_CDistantSegmentsKernel},
{"ExponentialKernel", __new_CExponentialKernel},
{"FixedDegreeStringKernel", __new_CFixedDegreeStringKernel},
{"GaussianKernel", __new_CGaussianKernel},
{"GaussianMatchStringKernel", __new_CGaussianMatchStringKernel},
{"GaussianShiftKernel", __new_CGaussianShiftKernel},
{"GaussianShortRealKernel", __new_CGaussianShortRealKernel},
{"HistogramIntersectionKernel", __new_CHistogramIntersectionKernel},
{"HistogramWordStringKernel", __new_CHistogramWordStringKernel},
{"InverseMultiQuadricKernel", __new_CInverseMultiQuadricKernel},
{"JensenShannonKernel", __new_CJensenShannonKernel},
{"LinearKernel", __new_CLinearKernel},
{"LinearStringKernel", __new_CLinearStringKernel},
{"LocalAlignmentStringKernel", __new_CLocalAlignmentStringKernel},
{"LocalityImprovedStringKernel", __new_CLocalityImprovedStringKernel},
{"LogKernel", __new_CLogKernel},
{"MatchWordStringKernel", __new_CMatchWordStringKernel},
{"MultiquadricKernel", __new_CMultiquadricKernel},
{"AvgDiagKernelNormalizer", __new_CAvgDiagKernelNormalizer},
{"DiceKernelNormalizer", __new_CDiceKernelNormalizer},
{"FirstElementKernelNormalizer", __new_CFirstElementKernelNormalizer},
{"IdentityKernelNormalizer", __new_CIdentityKernelNormalizer},
{"RidgeKernelNormalizer", __new_CRidgeKernelNormalizer},
{"ScatterKernelNormalizer", __new_CScatterKernelNormalizer},
{"SqrtDiagKernelNormalizer", __new_CSqrtDiagKernelNormalizer},
{"TanimotoKernelNormalizer", __new_CTanimotoKernelNormalizer},
{"VarianceKernelNormalizer", __new_CVarianceKernelNormalizer},
{"ZeroMeanCenterKernelNormalizer", __new_CZeroMeanCenterKernelNormalizer},
{"OligoStringKernel", __new_COligoStringKernel},
{"PolyKernel", __new_CPolyKernel},
{"PolyMatchStringKernel", __new_CPolyMatchStringKernel},
{"PolyMatchWordStringKernel", __new_CPolyMatchWordStringKernel},
{"PowerKernel", __new_CPowerKernel},
{"ProductKernel", __new_CProductKernel},
{"PyramidChi2", __new_CPyramidChi2},
{"RationalQuadraticKernel", __new_CRationalQuadraticKernel},
{"RegulatoryModulesStringKernel", __new_CRegulatoryModulesStringKernel},
{"SalzbergWordStringKernel", __new_CSalzbergWordStringKernel},
{"SigmoidKernel", __new_CSigmoidKernel},
{"SimpleLocalityImprovedStringKernel", __new_CSimpleLocalityImprovedStringKernel},
{"SNPStringKernel", __new_CSNPStringKernel},
{"SparseSpatialSampleStringKernel", __new_CSparseSpatialSampleStringKernel},
{"SpectrumMismatchRBFKernel", __new_CSpectrumMismatchRBFKernel},
{"SpectrumRBFKernel", __new_CSpectrumRBFKernel},
{"SphericalKernel", __new_CSphericalKernel},
{"SplineKernel", __new_CSplineKernel},
{"TensorProductPairKernel", __new_CTensorProductPairKernel},
{"TStudentKernel", __new_CTStudentKernel},
{"WaveKernel", __new_CWaveKernel},
{"WaveletKernel", __new_CWaveletKernel},
{"WeightedCommWordStringKernel", __new_CWeightedCommWordStringKernel},
{"WeightedDegreePositionStringKernel", __new_CWeightedDegreePositionStringKernel},
{"WeightedDegreeRBFKernel", __new_CWeightedDegreeRBFKernel},
{"WeightedDegreeStringKernel", __new_CWeightedDegreeStringKernel},
{"BinaryLabels", __new_CBinaryLabels},
{"MulticlassLabels", __new_CMulticlassLabels},
{"RegressionLabels", __new_CRegressionLabels},
{"StructuredLabels", __new_CStructuredLabels},
{"BitString", __new_CBitString},
{"Compressor", __new_CCompressor},
{"DynamicObjectArray", __new_CDynamicObjectArray},
{"FibonacciHeap", __new_CFibonacciHeap},
{"Hash", __new_CHash},
{"ListElement", __new_CListElement},
{"List", __new_CList},
{"Signal", __new_CSignal},
{"StructuredData", __new_CStructuredData},
{"Time", __new_CTime},
{"HingeLoss", __new_CHingeLoss},
{"LogLoss", __new_CLogLoss},
{"LogLossMargin", __new_CLogLossMargin},
{"SmoothHingeLoss", __new_CSmoothHingeLoss},
{"SquaredHingeLoss", __new_CSquaredHingeLoss},
{"SquaredLoss", __new_CSquaredLoss},
{"BaseMulticlassMachine", __new_CBaseMulticlassMachine},
{"DistanceMachine", __new_CDistanceMachine},
{"KernelMachine", __new_CKernelMachine},
{"KernelMulticlassMachine", __new_CKernelMulticlassMachine},
{"KernelStructuredOutputMachine", __new_CKernelStructuredOutputMachine},
{"LinearMachine", __new_CLinearMachine},
{"LinearMulticlassMachine", __new_CLinearMulticlassMachine},
{"LinearStructuredOutputMachine", __new_CLinearStructuredOutputMachine},
{"Machine", __new_CMachine},
{"NativeMulticlassMachine", __new_CNativeMulticlassMachine},
{"OnlineLinearMachine", __new_COnlineLinearMachine},
{"SLEPMachine", __new_CSLEPMachine},
{"StructuredOutputMachine", __new_CStructuredOutputMachine},
{"Math", __new_CMath},
{"SparseInverseCovariance", __new_CSparseInverseCovariance},
{"Statistics", __new_CStatistics},
{"GridSearchModelSelection", __new_CGridSearchModelSelection},
{"ModelSelectionParameters", __new_CModelSelectionParameters},
{"ParameterCombination", __new_CParameterCombination},
{"ECOCAEDDecoder", __new_CECOCAEDDecoder},
{"ECOCDiscriminantEncoder", __new_CECOCDiscriminantEncoder},
{"ECOCEDDecoder", __new_CECOCEDDecoder},
{"ECOCForestEncoder", __new_CECOCForestEncoder},
{"ECOCHDDecoder", __new_CECOCHDDecoder},
{"ECOCLLBDecoder", __new_CECOCLLBDecoder},
{"ECOCOVOEncoder", __new_CECOCOVOEncoder},
{"ECOCOVREncoder", __new_CECOCOVREncoder},
{"ECOCRandomDenseEncoder", __new_CECOCRandomDenseEncoder},
{"ECOCRandomSparseEncoder", __new_CECOCRandomSparseEncoder},
{"ECOCStrategy", __new_CECOCStrategy},
{"GaussianNaiveBayes", __new_CGaussianNaiveBayes},
{"GMNPLib", __new_CGMNPLib},
{"GMNPSVM", __new_CGMNPSVM},
{"KNN", __new_CKNN},
{"LaRank", __new_CLaRank},
{"MulticlassLibSVM", __new_CMulticlassLibSVM},
{"MulticlassOCAS", __new_CMulticlassOCAS},
{"MulticlassOneVsOneStrategy", __new_CMulticlassOneVsOneStrategy},
{"MulticlassOneVsRestStrategy", __new_CMulticlassOneVsRestStrategy},
{"MulticlassSVM", __new_CMulticlassSVM},
{"ThresholdRejectionStrategy", __new_CThresholdRejectionStrategy},
{"DixonQTestRejectionStrategy", __new_CDixonQTestRejectionStrategy},
{"ScatterSVM", __new_CScatterSVM},
{"BalancedConditionalProbabilityTree", __new_CBalancedConditionalProbabilityTree},
{"RandomConditionalProbabilityTree", __new_CRandomConditionalProbabilityTree},
{"DimensionReductionPreprocessor", __new_CDimensionReductionPreprocessor},
{"HomogeneousKernelMap", __new_CHomogeneousKernelMap},
{"LogPlusOne", __new_CLogPlusOne},
{"NormOne", __new_CNormOne},
{"PNorm", __new_CPNorm},
{"PruneVarSubMean", __new_CPruneVarSubMean},
{"RandomFourierGaussPreproc", __new_CRandomFourierGaussPreproc},
{"SortUlongString", __new_CSortUlongString},
{"SortWordString", __new_CSortWordString},
{"SumOne", __new_CSumOne},
{"ExactInferenceMethod", __new_CExactInferenceMethod},
{"GaussianLikelihood", __new_CGaussianLikelihood},
{"ZeroMean", __new_CZeroMean},
{"LibSVR", __new_CLibSVR},
{"MKLRegression", __new_CMKLRegression},
{"SVRLight", __new_CSVRLight},
{"LinearTimeMMD", __new_CLinearTimeMMD},
{"QuadraticTimeMMD", __new_CQuadraticTimeMMD},
{"StatisticalTest", __new_CStatisticalTest},
{"DualLibQPBMSOSVM", __new_CDualLibQPBMSOSVM},
{"DynProg", __new_CDynProg},
{"IntronList", __new_CIntronList},
{"MulticlassModel", __new_CMulticlassModel},
{"MulticlassRiskFunction", __new_CMulticlassRiskFunction},
{"MulticlassSOLabels", __new_CMulticlassSOLabels},
{"Plif", __new_CPlif},
{"PlifArray", __new_CPlifArray},
{"PlifMatrix", __new_CPlifMatrix},
{"SegmentLoss", __new_CSegmentLoss},
{"DomainAdaptationSVM", __new_CDomainAdaptationSVM},
{"MultitaskKernelMaskNormalizer", __new_CMultitaskKernelMaskNormalizer},
{"MultitaskKernelMaskPairNormalizer", __new_CMultitaskKernelMaskPairNormalizer},
{"MultitaskKernelNormalizer", __new_CMultitaskKernelNormalizer},
{"MultitaskKernelPlifNormalizer", __new_CMultitaskKernelPlifNormalizer},
{"Node", __new_CNode},
{"Taxonomy", __new_CTaxonomy},
{"MultitaskKernelTreeNormalizer", __new_CMultitaskKernelTreeNormalizer},
{"MultitaskLSRegression", __new_CMultitaskLSRegression},
{"Task", __new_CTask},
{"TaskGroup", __new_CTaskGroup},
{"TaskTree", __new_CTaskTree},
{"GUIClassifier", __new_CGUIClassifier},
{"GUIConverter", __new_CGUIConverter},
{"GUIDistance", __new_CGUIDistance},
{"GUIFeatures", __new_CGUIFeatures},
{"GUIHMM", __new_CGUIHMM},
{"GUIKernel", __new_CGUIKernel},
{"GUILabels", __new_CGUILabels},
{"GUIMath", __new_CGUIMath},
{"GUIPluginEstimate", __new_CGUIPluginEstimate},
{"GUIPreprocessor", __new_CGUIPreprocessor},
{"GUIStructure", __new_CGUIStructure},
{"GUITime", __new_CGUITime},
{"DenseFeatures", __new_CDenseFeatures},
{"SparseFeatures", __new_CSparseFeatures},
{"StreamingDenseFeatures", __new_CStreamingDenseFeatures},
{"StreamingSparseFeatures", __new_CStreamingSparseFeatures},
{"StreamingStringFeatures", __new_CStreamingStringFeatures},
{"StringFeatures", __new_CStringFeatures},
{"StringFileFeatures", __new_CStringFileFeatures},
{"BinaryStream", __new_CBinaryStream},
{"MemoryMappedFile", __new_CMemoryMappedFile},
{"ParseBuffer", __new_CParseBuffer},
{"SimpleFile", __new_CSimpleFile},
{"StreamingFileFromDenseFeatures", __new_CStreamingFileFromDenseFeatures},
{"StreamingFileFromSparseFeatures", __new_CStreamingFileFromSparseFeatures},
{"StreamingFileFromStringFeatures", __new_CStreamingFileFromStringFeatures},
{"Cache", __new_CCache},
{"DynamicArray", __new_CDynamicArray},
{"Set", __new_CSet},
{"DecompressString", __new_CDecompressString},	{NULL, NULL}
};

CSGObject* shogun::new_sgserializable(const char* sgserializable_name,
						   EPrimitiveType generic)
{
	for (class_list_entry_t* i=class_list; i->m_class_name != NULL;
		 i++)
	{
		if (strncmp(i->m_class_name, sgserializable_name, STRING_LEN) == 0)
			return i->m_new_sgserializable(generic);
	}

	return NULL;
}
