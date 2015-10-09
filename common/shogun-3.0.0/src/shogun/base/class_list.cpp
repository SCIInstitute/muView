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

#include "\classifier\AveragedPerceptron.h"
#include "\classifier\FeatureBlockLogisticRegression.h"
#include "\classifier\NearestCentroid.h"
#include "\classifier\Perceptron.h"
#include "\classifier\PluginEstimate.h"
#include "\classifier\mkl\MKLClassification.h"
#include "\classifier\mkl\MKLMulticlass.h"
#include "\classifier\mkl\MKLOneClass.h"
#include "\classifier\svm\GNPPLib.h"
#include "\classifier\svm\GNPPSVM.h"
#include "\classifier\svm\GPBTSVM.h"
#include "\classifier\svm\LibLinear.h"
#include "\classifier\svm\LibSVM.h"
#include "\classifier\svm\LibSVMOneClass.h"
#include "\classifier\svm\MPDSVM.h"
#include "\classifier\svm\OnlineLibLinear.h"
#include "\classifier\svm\OnlineSVMSGD.h"
#include "\classifier\svm\QPBSVMLib.h"
#include "\classifier\svm\SGDQN.h"
#include "\classifier\svm\SVM.h"
#include "\classifier\svm\SVMLight.h"
#include "\classifier\svm\SVMLightOneClass.h"
#include "\classifier\svm\SVMLin.h"
#include "\classifier\svm\SVMOcas.h"
#include "\classifier\svm\SVMSGD.h"
#include "\classifier\svm\WDSVMOcas.h"
#include "\classifier\vw\VowpalWabbit.h"
#include "\classifier\vw\VwEnvironment.h"
#include "\classifier\vw\VwParser.h"
#include "\classifier\vw\VwRegressor.h"
#include "\classifier\vw\cache\VwNativeCacheReader.h"
#include "\classifier\vw\cache\VwNativeCacheWriter.h"
#include "\classifier\vw\learners\VwAdaptiveLearner.h"
#include "\classifier\vw\learners\VwNonAdaptiveLearner.h"
#include "\clustering\Hierarchical.h"
#include "\clustering\KMeans.h"
#include "\converter\HashedDocConverter.h"
#include "\distance\AttenuatedEuclideanDistance.h"
#include "\distance\BrayCurtisDistance.h"
#include "\distance\CanberraMetric.h"
#include "\distance\CanberraWordDistance.h"
#include "\distance\ChebyshewMetric.h"
#include "\distance\ChiSquareDistance.h"
#include "\distance\CosineDistance.h"
#include "\distance\CustomDistance.h"
#include "\distance\EuclideanDistance.h"
#include "\distance\GeodesicMetric.h"
#include "\distance\HammingWordDistance.h"
#include "\distance\JensenMetric.h"
#include "\distance\KernelDistance.h"
#include "\distance\ManhattanMetric.h"
#include "\distance\ManhattanWordDistance.h"
#include "\distance\MinkowskiMetric.h"
#include "\distance\SparseEuclideanDistance.h"
#include "\distance\TanimotoDistance.h"
#include "\distributions\GHMM.h"
#include "\distributions\Histogram.h"
#include "\distributions\HMM.h"
#include "\distributions\LinearHMM.h"
#include "\distributions\PositionalPWM.h"
#include "\ensemble\MajorityVote.h"
#include "\ensemble\MeanRule.h"
#include "\ensemble\WeightedMajorityVote.h"
#include "\evaluation\ClusteringAccuracy.h"
#include "\evaluation\ClusteringMutualInformation.h"
#include "\evaluation\ContingencyTableEvaluation.h"
#include "\evaluation\CrossValidation.h"
#include "\evaluation\CrossValidationMKLStorage.h"
#include "\evaluation\CrossValidationMulticlassStorage.h"
#include "\evaluation\CrossValidationPrintOutput.h"
#include "\evaluation\CrossValidationSplitting.h"
#include "\evaluation\GradientCriterion.h"
#include "\evaluation\GradientEvaluation.h"
#include "\evaluation\GradientResult.h"
#include "\evaluation\MeanAbsoluteError.h"
#include "\evaluation\MeanSquaredError.h"
#include "\evaluation\MeanSquaredLogError.h"
#include "\evaluation\MulticlassAccuracy.h"
#include "\evaluation\MulticlassOVREvaluation.h"
#include "\evaluation\PRCEvaluation.h"
#include "\evaluation\ROCEvaluation.h"
#include "\evaluation\StratifiedCrossValidationSplitting.h"
#include "\evaluation\StructuredAccuracy.h"
#include "\features\Alphabet.h"
#include "\features\BinnedDotFeatures.h"
#include "\features\CombinedDotFeatures.h"
#include "\features\CombinedFeatures.h"
#include "\features\DataGenerator.h"
#include "\features\DenseFeatures.h"
#include "\features\DenseSubsetFeatures.h"
#include "\features\DummyFeatures.h"
#include "\features\ExplicitSpecFeatures.h"
#include "\features\FactorGraphFeatures.h"
#include "\features\FKFeatures.h"
#include "\features\HashedDenseFeatures.h"
#include "\features\HashedDocDotFeatures.h"
#include "\features\HashedSparseFeatures.h"
#include "\features\HashedWDFeatures.h"
#include "\features\HashedWDFeaturesTransposed.h"
#include "\features\ImplicitWeightedSpecFeatures.h"
#include "\features\LatentFeatures.h"
#include "\features\LBPPyrDotFeatures.h"
#include "\features\MatrixFeatures.h"
#include "\features\PolyFeatures.h"
#include "\features\RandomFourierDotFeatures.h"
#include "\features\RealFileFeatures.h"
#include "\features\SNPFeatures.h"
#include "\features\SparseFeatures.h"
#include "\features\SparsePolyFeatures.h"
#include "\features\StringFeatures.h"
#include "\features\StringFileFeatures.h"
#include "\features\Subset.h"
#include "\features\SubsetStack.h"
#include "\features\TOPFeatures.h"
#include "\features\WDFeatures.h"
#include "\features\streaming\StreamingDenseFeatures.h"
#include "\features\streaming\StreamingHashedDenseFeatures.h"
#include "\features\streaming\StreamingHashedDocDotFeatures.h"
#include "\features\streaming\StreamingHashedSparseFeatures.h"
#include "\features\streaming\StreamingSparseFeatures.h"
#include "\features\streaming\StreamingStringFeatures.h"
#include "\features\streaming\StreamingVwFeatures.h"
#include "\features\streaming\generators\GaussianBlobsDataGenerator.h"
#include "\features\streaming\generators\MeanShiftDataGenerator.h"
#include "\io\BinaryFile.h"
#include "\io\BinaryStream.h"
#include "\io\CSVFile.h"
#include "\io\IOBuffer.h"
#include "\io\LibSVMFile.h"
#include "\io\LineReader.h"
#include "\io\MemoryMappedFile.h"
#include "\io\Parser.h"
#include "\io\SerializableAsciiFile.h"
#include "\io\SimpleFile.h"
#include "\io\streaming\ParseBuffer.h"
#include "\io\streaming\StreamingAsciiFile.h"
#include "\io\streaming\StreamingFile.h"
#include "\io\streaming\StreamingFileFromDenseFeatures.h"
#include "\io\streaming\StreamingFileFromFeatures.h"
#include "\io\streaming\StreamingFileFromSparseFeatures.h"
#include "\io\streaming\StreamingFileFromStringFeatures.h"
#include "\io\streaming\StreamingVwCacheFile.h"
#include "\io\streaming\StreamingVwFile.h"
#include "\kernel\ANOVAKernel.h"
#include "\kernel\AUCKernel.h"
#include "\kernel\BesselKernel.h"
#include "\kernel\CauchyKernel.h"
#include "\kernel\Chi2Kernel.h"
#include "\kernel\CircularKernel.h"
#include "\kernel\CombinedKernel.h"
#include "\kernel\ConstKernel.h"
#include "\kernel\CustomKernel.h"
#include "\kernel\DiagKernel.h"
#include "\kernel\DistanceKernel.h"
#include "\kernel\ExponentialKernel.h"
#include "\kernel\GaussianARDKernel.h"
#include "\kernel\GaussianKernel.h"
#include "\kernel\GaussianShiftKernel.h"
#include "\kernel\GaussianShortRealKernel.h"
#include "\kernel\HistogramIntersectionKernel.h"
#include "\kernel\InverseMultiQuadricKernel.h"
#include "\kernel\JensenShannonKernel.h"
#include "\kernel\LinearARDKernel.h"
#include "\kernel\LinearKernel.h"
#include "\kernel\LogKernel.h"
#include "\kernel\MultiquadricKernel.h"
#include "\kernel\PolyKernel.h"
#include "\kernel\PowerKernel.h"
#include "\kernel\ProductKernel.h"
#include "\kernel\PyramidChi2.h"
#include "\kernel\RationalQuadraticKernel.h"
#include "\kernel\SigmoidKernel.h"
#include "\kernel\SphericalKernel.h"
#include "\kernel\SplineKernel.h"
#include "\kernel\TensorProductPairKernel.h"
#include "\kernel\TStudentKernel.h"
#include "\kernel\WaveKernel.h"
#include "\kernel\WaveletKernel.h"
#include "\kernel\WeightedDegreeRBFKernel.h"
#include "\kernel\normalizer\AvgDiagKernelNormalizer.h"
#include "\kernel\normalizer\DiceKernelNormalizer.h"
#include "\kernel\normalizer\FirstElementKernelNormalizer.h"
#include "\kernel\normalizer\IdentityKernelNormalizer.h"
#include "\kernel\normalizer\RidgeKernelNormalizer.h"
#include "\kernel\normalizer\ScatterKernelNormalizer.h"
#include "\kernel\normalizer\SqrtDiagKernelNormalizer.h"
#include "\kernel\normalizer\TanimotoKernelNormalizer.h"
#include "\kernel\normalizer\VarianceKernelNormalizer.h"
#include "\kernel\normalizer\ZeroMeanCenterKernelNormalizer.h"
#include "\kernel\string\CommUlongStringKernel.h"
#include "\kernel\string\CommWordStringKernel.h"
#include "\kernel\string\DistantSegmentsKernel.h"
#include "\kernel\string\FixedDegreeStringKernel.h"
#include "\kernel\string\GaussianMatchStringKernel.h"
#include "\kernel\string\HistogramWordStringKernel.h"
#include "\kernel\string\LinearStringKernel.h"
#include "\kernel\string\LocalAlignmentStringKernel.h"
#include "\kernel\string\LocalityImprovedStringKernel.h"
#include "\kernel\string\MatchWordStringKernel.h"
#include "\kernel\string\OligoStringKernel.h"
#include "\kernel\string\PolyMatchStringKernel.h"
#include "\kernel\string\PolyMatchWordStringKernel.h"
#include "\kernel\string\RegulatoryModulesStringKernel.h"
#include "\kernel\string\SalzbergWordStringKernel.h"
#include "\kernel\string\SimpleLocalityImprovedStringKernel.h"
#include "\kernel\string\SNPStringKernel.h"
#include "\kernel\string\SparseSpatialSampleStringKernel.h"
#include "\kernel\string\SpectrumMismatchRBFKernel.h"
#include "\kernel\string\SpectrumRBFKernel.h"
#include "\kernel\string\WeightedCommWordStringKernel.h"
#include "\kernel\string\WeightedDegreePositionStringKernel.h"
#include "\kernel\string\WeightedDegreeStringKernel.h"
#include "\labels\BinaryLabels.h"
#include "\labels\FactorGraphLabels.h"
#include "\labels\LabelsFactory.h"
#include "\labels\LatentLabels.h"
#include "\labels\MulticlassLabels.h"
#include "\labels\MulticlassMultipleOutputLabels.h"
#include "\labels\RegressionLabels.h"
#include "\labels\StructuredLabels.h"
#include "\latent\LatentSOSVM.h"
#include "\latent\LatentSVM.h"
#include "\lib\BitString.h"
#include "\lib\Cache.h"
#include "\lib\CircularBuffer.h"
#include "\lib\Compressor.h"
#include "\lib\Data.h"
#include "\lib\DelimiterTokenizer.h"
#include "\lib\DynamicArray.h"
#include "\lib\DynamicObjectArray.h"
#include "\lib\Hash.h"
#include "\lib\IndexBlock.h"
#include "\lib\IndexBlockGroup.h"
#include "\lib\IndexBlockTree.h"
#include "\lib\List.h"
#include "\lib\NGramTokenizer.h"
#include "\lib\Set.h"
#include "\lib\Signal.h"
#include "\lib\StructuredData.h"
#include "\lib\Time.h"
#include "\lib\computation\aggregator\StoreScalarAggregator.h"
#include "\lib\computation\engine\SerialComputationEngine.h"
#include "\lib\computation\jobresult\JobResult.h"
#include "\lib\computation\jobresult\ScalarResult.h"
#include "\lib\computation\jobresult\VectorResult.h"
#include "\loss\HingeLoss.h"
#include "\loss\LogLoss.h"
#include "\loss\LogLossMargin.h"
#include "\loss\SmoothHingeLoss.h"
#include "\loss\SquaredHingeLoss.h"
#include "\loss\SquaredLoss.h"
#include "\machine\BaggingMachine.h"
#include "\machine\BaseMulticlassMachine.h"
#include "\machine\DistanceMachine.h"
#include "\machine\KernelMachine.h"
#include "\machine\KernelMulticlassMachine.h"
#include "\machine\KernelStructuredOutputMachine.h"
#include "\machine\LinearMachine.h"
#include "\machine\LinearMulticlassMachine.h"
#include "\machine\LinearStructuredOutputMachine.h"
#include "\machine\Machine.h"
#include "\machine\NativeMulticlassMachine.h"
#include "\machine\OnlineLinearMachine.h"
#include "\machine\StructuredOutputMachine.h"
#include "\machine\gp\ZeroMean.h"
#include "\mathematics\JacobiEllipticFunctions.h"
#include "\mathematics\Math.h"
#include "\mathematics\Random.h"
#include "\mathematics\SparseInverseCovariance.h"
#include "\mathematics\Statistics.h"
#include "\mathematics\linalg\linop\SparseMatrixOperator.h"
#include "\mathematics\linalg\ratapprox\logdet\LogDetEstimator.h"
#include "\mathematics\linalg\ratapprox\tracesampler\NormalSampler.h"
#include "\modelselection\GridSearchModelSelection.h"
#include "\modelselection\ModelSelectionParameters.h"
#include "\modelselection\ParameterCombination.h"
#include "\modelselection\RandomSearchModelSelection.h"
#include "\multiclass\GaussianNaiveBayes.h"
#include "\multiclass\GMNPLib.h"
#include "\multiclass\GMNPSVM.h"
#include "\multiclass\KNN.h"
#include "\multiclass\LaRank.h"
#include "\multiclass\MulticlassLibLinear.h"
#include "\multiclass\MulticlassLibSVM.h"
#include "\multiclass\MulticlassOCAS.h"
#include "\multiclass\MulticlassOneVsOneStrategy.h"
#include "\multiclass\MulticlassOneVsRestStrategy.h"
#include "\multiclass\MulticlassSVM.h"
#include "\multiclass\ScatterSVM.h"
#include "\multiclass\ShareBoost.h"
#include "\multiclass\ecoc\ECOCAEDDecoder.h"
#include "\multiclass\ecoc\ECOCDiscriminantEncoder.h"
#include "\multiclass\ecoc\ECOCEDDecoder.h"
#include "\multiclass\ecoc\ECOCForestEncoder.h"
#include "\multiclass\ecoc\ECOCHDDecoder.h"
#include "\multiclass\ecoc\ECOCLLBDecoder.h"
#include "\multiclass\ecoc\ECOCOVOEncoder.h"
#include "\multiclass\ecoc\ECOCOVREncoder.h"
#include "\multiclass\ecoc\ECOCRandomDenseEncoder.h"
#include "\multiclass\ecoc\ECOCRandomSparseEncoder.h"
#include "\multiclass\ecoc\ECOCStrategy.h"
#include "\multiclass\tree\BalancedConditionalProbabilityTree.h"
#include "\multiclass\tree\RandomConditionalProbabilityTree.h"
#include "\multiclass\tree\RelaxedTree.h"
#include "\multiclass\tree\TreeMachine.h"
#include "\preprocessor\DecompressString.h"
#include "\preprocessor\DimensionReductionPreprocessor.h"
#include "\preprocessor\HomogeneousKernelMap.h"
#include "\preprocessor\LogPlusOne.h"
#include "\preprocessor\NormOne.h"
#include "\preprocessor\PNorm.h"
#include "\preprocessor\PruneVarSubMean.h"
#include "\preprocessor\RandomFourierGaussPreproc.h"
#include "\preprocessor\RescaleFeatures.h"
#include "\preprocessor\SortUlongString.h"
#include "\preprocessor\SortWordString.h"
#include "\preprocessor\SumOne.h"
#include "\regression\svr\LibSVR.h"
#include "\regression\svr\MKLRegression.h"
#include "\regression\svr\SVRLight.h"
#include "\statistics\HSIC.h"
#include "\statistics\KernelMeanMatching.h"
#include "\statistics\LinearTimeMMD.h"
#include "\statistics\MMDKernelSelectionCombMaxL2.h"
#include "\statistics\MMDKernelSelectionCombOpt.h"
#include "\statistics\MMDKernelSelectionMax.h"
#include "\statistics\MMDKernelSelectionMedian.h"
#include "\statistics\MMDKernelSelectionOpt.h"
#include "\statistics\QuadraticTimeMMD.h"
#include "\structure\CCSOSVM.h"
#include "\structure\DisjointSet.h"
#include "\structure\DualLibQPBMSOSVM.h"
#include "\structure\DynProg.h"
#include "\structure\Factor.h"
#include "\structure\FactorGraph.h"
#include "\structure\FactorGraphModel.h"
#include "\structure\FactorType.h"
#include "\structure\HMSVMModel.h"
#include "\structure\IntronList.h"
#include "\structure\MAPInference.h"
#include "\structure\MulticlassModel.h"
#include "\structure\MulticlassSOLabels.h"
#include "\structure\Plif.h"
#include "\structure\PlifArray.h"
#include "\structure\PlifMatrix.h"
#include "\structure\SegmentLoss.h"
#include "\structure\SequenceLabels.h"
#include "\structure\SOSVMHelper.h"
#include "\structure\StochasticSOSVM.h"
#include "\structure\TwoStateModel.h"
#include "\transfer\domain_adaptation\DomainAdaptationSVM.h"
#include "\transfer\multitask\MultitaskClusteredLogisticRegression.h"
#include "\transfer\multitask\MultitaskKernelMaskNormalizer.h"
#include "\transfer\multitask\MultitaskKernelMaskPairNormalizer.h"
#include "\transfer\multitask\MultitaskKernelNormalizer.h"
#include "\transfer\multitask\MultitaskKernelPlifNormalizer.h"
#include "\transfer\multitask\MultitaskKernelTreeNormalizer.h"
#include "\transfer\multitask\MultitaskL12LogisticRegression.h"
#include "\transfer\multitask\MultitaskLeastSquaresRegression.h"
#include "\transfer\multitask\MultitaskLinearMachine.h"
#include "\transfer\multitask\MultitaskLogisticRegression.h"
#include "\transfer\multitask\MultitaskROCEvaluation.h"
#include "\transfer\multitask\MultitaskTraceLogisticRegression.h"
#include "\transfer\multitask\Task.h"
#include "\transfer\multitask\TaskGroup.h"
#include "\transfer\multitask\TaskTree.h"
#include "\ui\GUIClassifier.h"
#include "\ui\GUIConverter.h"
#include "\ui\GUIDistance.h"
#include "\ui\GUIFeatures.h"
#include "\ui\GUIHMM.h"
#include "\ui\GUIKernel.h"
#include "\ui\GUILabels.h"
#include "\ui\GUIMath.h"
#include "\ui\GUIPluginEstimate.h"
#include "\ui\GUIPreprocessor.h"
#include "\ui\GUIStructure.h"
#include "\ui\GUITime.h"
using namespace shogun;

#define SHOGUN_TEMPLATE_CLASS
#define SHOGUN_BASIC_CLASS
static SHOGUN_BASIC_CLASS CSGObject* __new_CAveragedPerceptron(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAveragedPerceptron(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFeatureBlockLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFeatureBlockLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLClassification(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLClassification(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLMulticlass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLMulticlass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLOneClass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLOneClass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNearestCentroid(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNearestCentroid(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPerceptron(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPerceptron(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPluginEstimate(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPluginEstimate(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGNPPLib(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGNPPLib(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGNPPSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGNPPSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGPBTSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGPBTSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibLinear(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibLinear(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVMOneClass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVMOneClass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMPDSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMPDSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COnlineLibLinear(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COnlineLibLinear(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COnlineSVMSGD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COnlineSVMSGD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CQPBSVMLib(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CQPBSVMLib(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSGDQN(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSGDQN(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMLight(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMLight(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMLightOneClass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMLightOneClass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMLin(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMLin(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMOcas(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMOcas(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMSGD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMSGD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWDSVMOcas(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWDSVMOcas(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwNativeCacheReader(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwNativeCacheReader(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwNativeCacheWriter(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwNativeCacheWriter(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwAdaptiveLearner(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwAdaptiveLearner(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwNonAdaptiveLearner(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwNonAdaptiveLearner(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVowpalWabbit(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVowpalWabbit(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwEnvironment(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwEnvironment(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwParser(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwParser(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwRegressor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwRegressor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHierarchical(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHierarchical(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKMeans(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKMeans(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedDocConverter(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedDocConverter(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAttenuatedEuclideanDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAttenuatedEuclideanDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBrayCurtisDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBrayCurtisDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCanberraMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCanberraMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCanberraWordDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCanberraWordDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CChebyshewMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CChebyshewMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CChiSquareDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CChiSquareDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCosineDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCosineDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCustomDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCustomDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CEuclideanDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CEuclideanDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGeodesicMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGeodesicMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHammingWordDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHammingWordDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJensenMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJensenMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CManhattanMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CManhattanMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CManhattanWordDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CManhattanWordDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMinkowskiMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMinkowskiMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparseEuclideanDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparseEuclideanDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTanimotoDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTanimotoDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHistogram(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHistogram(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPositionalPWM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPositionalPWM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMajorityVote(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMajorityVote(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanRule(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanRule(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedMajorityVote(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedMajorityVote(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CClusteringAccuracy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CClusteringAccuracy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CClusteringMutualInformation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CClusteringMutualInformation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CContingencyTableEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CContingencyTableEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAccuracyMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAccuracyMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CErrorRateMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CErrorRateMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBALMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBALMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWRACCMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWRACCMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CF1Measure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CF1Measure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossCorrelationMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossCorrelationMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRecallMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRecallMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPrecisionMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPrecisionMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSpecificityMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSpecificityMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationResult(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationResult(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationMKLStorage(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationMKLStorage(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationMulticlassStorage(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationMulticlassStorage(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationPrintOutput(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationPrintOutput(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationSplitting(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationSplitting(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGradientCriterion(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGradientCriterion(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGradientEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGradientEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGradientResult(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGradientResult(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanAbsoluteError(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanAbsoluteError(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanSquaredError(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanSquaredError(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanSquaredLogError(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanSquaredLogError(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassAccuracy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassAccuracy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOVREvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOVREvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPRCEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPRCEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CROCEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CROCEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStratifiedCrossValidationSplitting(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStratifiedCrossValidationSplitting(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredAccuracy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredAccuracy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAlphabet(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAlphabet(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBinnedDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBinnedDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCombinedDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCombinedDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCombinedFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCombinedFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDataGenerator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDataGenerator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDummyFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDummyFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CExplicitSpecFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CExplicitSpecFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFKFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFKFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedDocDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedDocDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedWDFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedWDFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedWDFeaturesTransposed(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedWDFeaturesTransposed(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CImplicitWeightedSpecFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CImplicitWeightedSpecFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLBPPyrDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLBPPyrDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomFourierDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomFourierDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRealFileFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRealFileFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSNPFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSNPFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparsePolyFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparsePolyFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianBlobsDataGenerator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianBlobsDataGenerator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanShiftDataGenerator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanShiftDataGenerator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingHashedDocDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingHashedDocDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingVwFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingVwFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSubset(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSubset(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSubsetStack(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSubsetStack(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTOPFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTOPFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWDFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWDFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBinaryFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBinaryFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCSVFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCSVFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIOBuffer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIOBuffer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVMFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVMFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLineReader(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLineReader(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CParser(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CParser(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSerializableAsciiFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSerializableAsciiFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingAsciiFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingAsciiFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingFileFromFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingFileFromFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingVwCacheFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingVwCacheFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingVwFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingVwFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CANOVAKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CANOVAKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAUCKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAUCKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBesselKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBesselKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCauchyKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCauchyKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CChi2Kernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CChi2Kernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCircularKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCircularKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCombinedKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCombinedKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CConstKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CConstKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCustomKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCustomKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDiagKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDiagKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDistanceKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDistanceKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CExponentialKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CExponentialKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianARDKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianARDKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianShiftKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianShiftKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianShortRealKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianShortRealKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHistogramIntersectionKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHistogramIntersectionKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CInverseMultiQuadricKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CInverseMultiQuadricKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJensenShannonKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJensenShannonKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearARDKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearARDKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultiquadricKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultiquadricKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAvgDiagKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAvgDiagKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDiceKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDiceKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFirstElementKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFirstElementKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIdentityKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIdentityKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRidgeKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRidgeKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CScatterKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CScatterKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSqrtDiagKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSqrtDiagKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTanimotoKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTanimotoKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVarianceKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVarianceKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CZeroMeanCenterKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CZeroMeanCenterKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPowerKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPowerKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CProductKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CProductKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPyramidChi2(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPyramidChi2(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRationalQuadraticKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRationalQuadraticKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSigmoidKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSigmoidKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSphericalKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSphericalKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSplineKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSplineKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCommUlongStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCommUlongStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCommWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCommWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDistantSegmentsKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDistantSegmentsKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFixedDegreeStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFixedDegreeStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianMatchStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianMatchStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHistogramWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHistogramWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLocalAlignmentStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLocalAlignmentStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLocalityImprovedStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLocalityImprovedStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMatchWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMatchWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COligoStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COligoStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyMatchStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyMatchStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyMatchWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyMatchWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRegulatoryModulesStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRegulatoryModulesStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSalzbergWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSalzbergWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSimpleLocalityImprovedStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSimpleLocalityImprovedStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSNPStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSNPStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparseSpatialSampleStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparseSpatialSampleStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSpectrumMismatchRBFKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSpectrumMismatchRBFKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSpectrumRBFKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSpectrumRBFKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedCommWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedCommWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedDegreePositionStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedDegreePositionStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedDegreeStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedDegreeStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTensorProductPairKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTensorProductPairKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTStudentKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTStudentKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWaveKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWaveKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWaveletKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWaveletKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedDegreeRBFKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedDegreeRBFKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBinaryLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBinaryLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphObservation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphObservation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLabelsFactory(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLabelsFactory(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassMultipleOutputLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassMultipleOutputLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRegressionLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRegressionLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBitString(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBitString(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCircularBuffer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCircularBuffer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCompressor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCompressor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSerialComputationEngine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSerialComputationEngine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJobResult(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJobResult(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CData(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CData(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDelimiterTokenizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDelimiterTokenizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDynamicObjectArray(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDynamicObjectArray(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHash(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHash(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIndexBlock(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIndexBlock(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIndexBlockGroup(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIndexBlockGroup(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIndexBlockTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIndexBlockTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CListElement(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CListElement(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CList(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CList(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNGramTokenizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNGramTokenizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSignal(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSignal(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredData(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredData(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTime(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTime(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHingeLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHingeLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogLossMargin(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogLossMargin(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSmoothHingeLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSmoothHingeLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSquaredHingeLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSquaredHingeLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSquaredLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSquaredLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBaggingMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBaggingMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBaseMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBaseMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDistanceMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDistanceMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CZeroMean(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CZeroMean(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelStructuredOutputMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelStructuredOutputMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearStructuredOutputMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearStructuredOutputMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNativeMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNativeMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COnlineLinearMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COnlineLinearMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredOutputMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredOutputMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJacobiEllipticFunctions(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJacobiEllipticFunctions(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogDetEstimator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogDetEstimator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNormalSampler(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNormalSampler(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMath(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMath(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandom(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandom(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparseInverseCovariance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparseInverseCovariance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStatistics(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStatistics(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGridSearchModelSelection(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGridSearchModelSelection(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CModelSelectionParameters(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CModelSelectionParameters(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CParameterCombination(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CParameterCombination(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomSearchModelSelection(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomSearchModelSelection(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCAEDDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCAEDDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCDiscriminantEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCDiscriminantEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCEDDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCEDDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCForestEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCForestEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCHDDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCHDDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCLLBDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCLLBDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCOVOEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCOVOEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCOVREncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCOVREncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCRandomDenseEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCRandomDenseEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCRandomSparseEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCRandomSparseEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianNaiveBayes(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianNaiveBayes(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGMNPLib(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGMNPLib(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGMNPSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGMNPSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKNN(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKNN(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLaRank(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLaRank(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassLibLinear(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassLibLinear(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassLibSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassLibSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOCAS(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOCAS(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOneVsOneStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOneVsOneStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOneVsRestStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOneVsRestStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CThresholdRejectionStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CThresholdRejectionStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDixonQTestRejectionStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDixonQTestRejectionStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CScatterSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CScatterSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CShareBoost(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CShareBoost(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBalancedConditionalProbabilityTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBalancedConditionalProbabilityTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomConditionalProbabilityTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomConditionalProbabilityTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRelaxedTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRelaxedTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTron(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTron(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDimensionReductionPreprocessor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDimensionReductionPreprocessor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHomogeneousKernelMap(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHomogeneousKernelMap(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogPlusOne(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogPlusOne(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNormOne(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNormOne(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPNorm(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPNorm(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPruneVarSubMean(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPruneVarSubMean(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomFourierGaussPreproc(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomFourierGaussPreproc(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRescaleFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRescaleFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSortUlongString(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSortUlongString(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSortWordString(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSortWordString(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSumOne(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSumOne(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVR(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVR(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVRLight(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVRLight(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHSIC(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHSIC(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelMeanMatching(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelMeanMatching(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearTimeMMD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearTimeMMD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionCombMaxL2(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionCombMaxL2(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionCombOpt(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionCombOpt(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionMax(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionMax(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionMedian(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionMedian(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionOpt(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionOpt(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CQuadraticTimeMMD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CQuadraticTimeMMD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCCSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCCSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDisjointSet(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDisjointSet(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDualLibQPBMSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDualLibQPBMSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDynProg(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDynProg(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorDataSource(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorDataSource(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraph(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraph(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorType(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorType(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTableFactorType(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTableFactorType(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHMSVMModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHMSVMModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIntronList(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIntronList(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMAPInference(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMAPInference(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassSOLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassSOLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPlif(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPlif(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPlifArray(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPlifArray(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPlifMatrix(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPlifMatrix(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSegmentLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSegmentLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSequence(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSequence(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSequenceLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSequenceLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSOSVMHelper(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSOSVMHelper(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStochasticSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStochasticSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTwoStateModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTwoStateModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDomainAdaptationSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDomainAdaptationSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskClusteredLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskClusteredLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelMaskNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelMaskNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelMaskPairNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelMaskPairNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelPlifNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelPlifNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNode(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNode(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTaxonomy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTaxonomy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelTreeNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelTreeNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskL12LogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskL12LogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskLeastSquaresRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskLeastSquaresRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskLinearMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskLinearMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskROCEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskROCEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskTraceLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskTraceLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTask(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTask(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTaskGroup(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTaskGroup(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTaskTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTaskTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIClassifier(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIClassifier(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIConverter(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIConverter(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUILabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUILabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIMath(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIMath(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIPluginEstimate(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIPluginEstimate(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIPreprocessor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIPreprocessor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIStructure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIStructure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUITime(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUITime(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAveragedPerceptron(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAveragedPerceptron(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFeatureBlockLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFeatureBlockLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLClassification(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLClassification(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLMulticlass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLMulticlass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLOneClass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLOneClass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNearestCentroid(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNearestCentroid(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPerceptron(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPerceptron(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPluginEstimate(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPluginEstimate(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGNPPLib(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGNPPLib(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGNPPSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGNPPSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGPBTSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGPBTSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibLinear(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibLinear(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVMOneClass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVMOneClass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMPDSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMPDSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COnlineLibLinear(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COnlineLibLinear(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COnlineSVMSGD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COnlineSVMSGD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CQPBSVMLib(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CQPBSVMLib(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSGDQN(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSGDQN(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMLight(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMLight(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMLightOneClass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMLightOneClass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMLin(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMLin(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMOcas(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMOcas(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMSGD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMSGD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWDSVMOcas(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWDSVMOcas(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwNativeCacheReader(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwNativeCacheReader(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwNativeCacheWriter(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwNativeCacheWriter(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwAdaptiveLearner(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwAdaptiveLearner(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwNonAdaptiveLearner(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwNonAdaptiveLearner(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVowpalWabbit(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVowpalWabbit(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwEnvironment(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwEnvironment(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwParser(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwParser(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwRegressor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwRegressor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHierarchical(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHierarchical(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKMeans(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKMeans(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedDocConverter(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedDocConverter(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAttenuatedEuclideanDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAttenuatedEuclideanDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBrayCurtisDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBrayCurtisDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCanberraMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCanberraMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCanberraWordDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCanberraWordDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CChebyshewMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CChebyshewMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CChiSquareDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CChiSquareDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCosineDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCosineDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCustomDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCustomDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CEuclideanDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CEuclideanDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGeodesicMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGeodesicMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHammingWordDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHammingWordDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJensenMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJensenMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CManhattanMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CManhattanMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CManhattanWordDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CManhattanWordDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMinkowskiMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMinkowskiMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparseEuclideanDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparseEuclideanDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTanimotoDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTanimotoDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHistogram(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHistogram(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPositionalPWM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPositionalPWM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMajorityVote(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMajorityVote(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanRule(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanRule(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedMajorityVote(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedMajorityVote(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CClusteringAccuracy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CClusteringAccuracy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CClusteringMutualInformation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CClusteringMutualInformation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CContingencyTableEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CContingencyTableEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAccuracyMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAccuracyMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CErrorRateMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CErrorRateMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBALMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBALMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWRACCMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWRACCMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CF1Measure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CF1Measure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossCorrelationMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossCorrelationMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRecallMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRecallMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPrecisionMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPrecisionMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSpecificityMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSpecificityMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationResult(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationResult(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationMKLStorage(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationMKLStorage(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationMulticlassStorage(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationMulticlassStorage(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationPrintOutput(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationPrintOutput(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationSplitting(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationSplitting(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGradientCriterion(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGradientCriterion(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGradientEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGradientEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGradientResult(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGradientResult(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanAbsoluteError(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanAbsoluteError(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanSquaredError(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanSquaredError(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanSquaredLogError(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanSquaredLogError(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassAccuracy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassAccuracy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOVREvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOVREvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPRCEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPRCEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CROCEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CROCEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStratifiedCrossValidationSplitting(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStratifiedCrossValidationSplitting(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredAccuracy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredAccuracy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAlphabet(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAlphabet(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBinnedDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBinnedDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCombinedDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCombinedDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCombinedFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCombinedFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDataGenerator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDataGenerator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDummyFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDummyFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CExplicitSpecFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CExplicitSpecFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFKFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFKFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedDocDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedDocDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedWDFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedWDFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedWDFeaturesTransposed(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedWDFeaturesTransposed(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CImplicitWeightedSpecFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CImplicitWeightedSpecFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLBPPyrDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLBPPyrDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomFourierDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomFourierDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRealFileFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRealFileFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSNPFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSNPFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparsePolyFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparsePolyFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianBlobsDataGenerator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianBlobsDataGenerator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanShiftDataGenerator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanShiftDataGenerator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingHashedDocDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingHashedDocDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingVwFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingVwFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSubset(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSubset(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSubsetStack(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSubsetStack(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTOPFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTOPFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWDFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWDFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBinaryFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBinaryFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCSVFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCSVFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIOBuffer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIOBuffer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVMFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVMFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLineReader(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLineReader(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CParser(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CParser(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSerializableAsciiFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSerializableAsciiFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingAsciiFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingAsciiFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingFileFromFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingFileFromFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingVwCacheFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingVwCacheFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingVwFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingVwFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CANOVAKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CANOVAKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAUCKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAUCKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBesselKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBesselKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCauchyKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCauchyKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CChi2Kernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CChi2Kernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCircularKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCircularKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCombinedKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCombinedKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CConstKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CConstKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCustomKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCustomKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDiagKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDiagKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDistanceKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDistanceKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CExponentialKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CExponentialKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianARDKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianARDKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianShiftKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianShiftKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianShortRealKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianShortRealKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHistogramIntersectionKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHistogramIntersectionKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CInverseMultiQuadricKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CInverseMultiQuadricKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJensenShannonKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJensenShannonKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearARDKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearARDKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultiquadricKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultiquadricKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAvgDiagKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAvgDiagKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDiceKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDiceKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFirstElementKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFirstElementKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIdentityKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIdentityKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRidgeKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRidgeKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CScatterKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CScatterKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSqrtDiagKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSqrtDiagKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTanimotoKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTanimotoKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVarianceKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVarianceKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CZeroMeanCenterKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CZeroMeanCenterKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPowerKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPowerKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CProductKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CProductKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPyramidChi2(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPyramidChi2(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRationalQuadraticKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRationalQuadraticKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSigmoidKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSigmoidKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSphericalKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSphericalKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSplineKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSplineKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCommUlongStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCommUlongStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCommWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCommWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDistantSegmentsKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDistantSegmentsKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFixedDegreeStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFixedDegreeStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianMatchStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianMatchStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHistogramWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHistogramWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLocalAlignmentStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLocalAlignmentStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLocalityImprovedStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLocalityImprovedStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMatchWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMatchWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COligoStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COligoStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyMatchStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyMatchStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyMatchWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyMatchWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRegulatoryModulesStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRegulatoryModulesStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSalzbergWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSalzbergWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSimpleLocalityImprovedStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSimpleLocalityImprovedStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSNPStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSNPStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparseSpatialSampleStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparseSpatialSampleStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSpectrumMismatchRBFKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSpectrumMismatchRBFKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSpectrumRBFKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSpectrumRBFKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedCommWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedCommWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedDegreePositionStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedDegreePositionStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedDegreeStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedDegreeStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTensorProductPairKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTensorProductPairKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTStudentKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTStudentKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWaveKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWaveKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWaveletKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWaveletKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedDegreeRBFKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedDegreeRBFKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBinaryLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBinaryLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphObservation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphObservation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLabelsFactory(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLabelsFactory(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassMultipleOutputLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassMultipleOutputLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRegressionLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRegressionLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBitString(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBitString(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCircularBuffer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCircularBuffer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCompressor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCompressor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSerialComputationEngine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSerialComputationEngine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJobResult(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJobResult(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CData(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CData(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDelimiterTokenizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDelimiterTokenizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDynamicObjectArray(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDynamicObjectArray(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHash(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHash(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIndexBlock(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIndexBlock(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIndexBlockGroup(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIndexBlockGroup(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIndexBlockTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIndexBlockTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CListElement(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CListElement(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CList(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CList(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNGramTokenizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNGramTokenizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSignal(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSignal(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredData(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredData(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTime(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTime(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHingeLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHingeLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogLossMargin(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogLossMargin(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSmoothHingeLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSmoothHingeLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSquaredHingeLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSquaredHingeLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSquaredLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSquaredLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBaggingMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBaggingMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBaseMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBaseMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDistanceMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDistanceMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CZeroMean(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CZeroMean(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelStructuredOutputMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelStructuredOutputMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearStructuredOutputMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearStructuredOutputMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNativeMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNativeMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COnlineLinearMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COnlineLinearMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredOutputMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredOutputMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJacobiEllipticFunctions(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJacobiEllipticFunctions(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogDetEstimator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogDetEstimator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNormalSampler(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNormalSampler(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMath(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMath(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandom(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandom(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparseInverseCovariance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparseInverseCovariance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStatistics(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStatistics(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGridSearchModelSelection(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGridSearchModelSelection(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CModelSelectionParameters(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CModelSelectionParameters(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CParameterCombination(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CParameterCombination(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomSearchModelSelection(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomSearchModelSelection(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCAEDDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCAEDDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCDiscriminantEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCDiscriminantEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCEDDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCEDDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCForestEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCForestEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCHDDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCHDDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCLLBDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCLLBDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCOVOEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCOVOEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCOVREncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCOVREncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCRandomDenseEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCRandomDenseEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCRandomSparseEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCRandomSparseEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianNaiveBayes(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianNaiveBayes(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGMNPLib(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGMNPLib(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGMNPSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGMNPSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKNN(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKNN(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLaRank(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLaRank(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassLibLinear(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassLibLinear(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassLibSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassLibSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOCAS(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOCAS(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOneVsOneStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOneVsOneStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOneVsRestStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOneVsRestStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CThresholdRejectionStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CThresholdRejectionStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDixonQTestRejectionStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDixonQTestRejectionStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CScatterSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CScatterSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CShareBoost(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CShareBoost(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBalancedConditionalProbabilityTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBalancedConditionalProbabilityTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomConditionalProbabilityTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomConditionalProbabilityTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRelaxedTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRelaxedTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTron(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTron(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDimensionReductionPreprocessor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDimensionReductionPreprocessor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHomogeneousKernelMap(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHomogeneousKernelMap(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogPlusOne(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogPlusOne(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNormOne(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNormOne(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPNorm(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPNorm(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPruneVarSubMean(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPruneVarSubMean(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomFourierGaussPreproc(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomFourierGaussPreproc(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRescaleFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRescaleFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSortUlongString(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSortUlongString(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSortWordString(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSortWordString(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSumOne(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSumOne(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVR(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVR(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVRLight(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVRLight(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHSIC(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHSIC(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelMeanMatching(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelMeanMatching(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearTimeMMD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearTimeMMD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionCombMaxL2(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionCombMaxL2(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionCombOpt(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionCombOpt(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionMax(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionMax(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionMedian(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionMedian(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionOpt(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionOpt(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CQuadraticTimeMMD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CQuadraticTimeMMD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCCSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCCSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDisjointSet(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDisjointSet(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDualLibQPBMSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDualLibQPBMSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDynProg(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDynProg(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorDataSource(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorDataSource(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraph(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraph(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorType(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorType(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTableFactorType(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTableFactorType(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHMSVMModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHMSVMModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIntronList(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIntronList(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMAPInference(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMAPInference(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassSOLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassSOLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPlif(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPlif(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPlifArray(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPlifArray(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPlifMatrix(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPlifMatrix(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSegmentLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSegmentLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSequence(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSequence(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSequenceLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSequenceLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSOSVMHelper(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSOSVMHelper(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStochasticSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStochasticSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTwoStateModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTwoStateModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDomainAdaptationSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDomainAdaptationSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskClusteredLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskClusteredLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelMaskNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelMaskNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelMaskPairNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelMaskPairNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelPlifNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelPlifNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNode(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNode(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTaxonomy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTaxonomy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelTreeNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelTreeNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskL12LogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskL12LogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskLeastSquaresRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskLeastSquaresRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskLinearMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskLinearMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskROCEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskROCEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskTraceLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskTraceLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTask(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTask(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTaskGroup(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTaskGroup(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTaskTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTaskTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIClassifier(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIClassifier(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIConverter(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIConverter(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUILabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUILabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIMath(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIMath(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIPluginEstimate(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIPluginEstimate(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIPreprocessor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIPreprocessor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIStructure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIStructure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUITime(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUITime(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAveragedPerceptron(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAveragedPerceptron(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFeatureBlockLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFeatureBlockLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLClassification(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLClassification(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLMulticlass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLMulticlass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLOneClass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLOneClass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNearestCentroid(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNearestCentroid(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPerceptron(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPerceptron(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPluginEstimate(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPluginEstimate(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGNPPLib(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGNPPLib(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGNPPSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGNPPSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGPBTSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGPBTSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibLinear(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibLinear(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVMOneClass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVMOneClass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMPDSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMPDSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COnlineLibLinear(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COnlineLibLinear(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COnlineSVMSGD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COnlineSVMSGD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CQPBSVMLib(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CQPBSVMLib(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSGDQN(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSGDQN(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMLight(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMLight(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMLightOneClass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMLightOneClass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMLin(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMLin(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMOcas(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMOcas(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMSGD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMSGD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWDSVMOcas(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWDSVMOcas(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwNativeCacheReader(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwNativeCacheReader(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwNativeCacheWriter(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwNativeCacheWriter(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwAdaptiveLearner(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwAdaptiveLearner(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwNonAdaptiveLearner(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwNonAdaptiveLearner(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVowpalWabbit(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVowpalWabbit(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwEnvironment(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwEnvironment(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwParser(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwParser(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwRegressor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwRegressor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHierarchical(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHierarchical(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKMeans(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKMeans(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedDocConverter(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedDocConverter(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAttenuatedEuclideanDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAttenuatedEuclideanDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBrayCurtisDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBrayCurtisDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCanberraMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCanberraMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCanberraWordDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCanberraWordDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CChebyshewMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CChebyshewMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CChiSquareDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CChiSquareDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCosineDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCosineDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCustomDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCustomDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CEuclideanDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CEuclideanDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGeodesicMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGeodesicMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHammingWordDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHammingWordDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJensenMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJensenMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CManhattanMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CManhattanMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CManhattanWordDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CManhattanWordDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMinkowskiMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMinkowskiMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparseEuclideanDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparseEuclideanDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTanimotoDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTanimotoDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHistogram(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHistogram(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPositionalPWM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPositionalPWM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMajorityVote(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMajorityVote(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanRule(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanRule(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedMajorityVote(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedMajorityVote(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CClusteringAccuracy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CClusteringAccuracy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CClusteringMutualInformation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CClusteringMutualInformation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CContingencyTableEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CContingencyTableEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAccuracyMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAccuracyMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CErrorRateMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CErrorRateMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBALMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBALMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWRACCMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWRACCMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CF1Measure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CF1Measure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossCorrelationMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossCorrelationMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRecallMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRecallMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPrecisionMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPrecisionMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSpecificityMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSpecificityMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationResult(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationResult(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationMKLStorage(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationMKLStorage(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationMulticlassStorage(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationMulticlassStorage(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationPrintOutput(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationPrintOutput(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationSplitting(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationSplitting(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGradientCriterion(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGradientCriterion(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGradientEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGradientEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGradientResult(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGradientResult(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanAbsoluteError(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanAbsoluteError(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanSquaredError(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanSquaredError(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanSquaredLogError(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanSquaredLogError(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassAccuracy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassAccuracy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOVREvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOVREvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPRCEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPRCEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CROCEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CROCEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStratifiedCrossValidationSplitting(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStratifiedCrossValidationSplitting(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredAccuracy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredAccuracy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAlphabet(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAlphabet(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBinnedDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBinnedDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCombinedDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCombinedDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCombinedFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCombinedFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDataGenerator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDataGenerator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDummyFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDummyFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CExplicitSpecFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CExplicitSpecFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFKFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFKFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedDocDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedDocDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedWDFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedWDFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedWDFeaturesTransposed(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedWDFeaturesTransposed(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CImplicitWeightedSpecFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CImplicitWeightedSpecFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLBPPyrDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLBPPyrDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomFourierDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomFourierDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRealFileFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRealFileFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSNPFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSNPFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparsePolyFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparsePolyFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianBlobsDataGenerator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianBlobsDataGenerator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanShiftDataGenerator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanShiftDataGenerator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingHashedDocDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingHashedDocDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingVwFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingVwFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSubset(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSubset(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSubsetStack(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSubsetStack(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTOPFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTOPFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWDFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWDFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBinaryFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBinaryFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCSVFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCSVFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIOBuffer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIOBuffer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVMFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVMFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLineReader(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLineReader(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CParser(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CParser(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSerializableAsciiFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSerializableAsciiFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingAsciiFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingAsciiFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingFileFromFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingFileFromFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingVwCacheFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingVwCacheFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingVwFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingVwFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CANOVAKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CANOVAKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAUCKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAUCKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBesselKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBesselKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCauchyKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCauchyKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CChi2Kernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CChi2Kernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCircularKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCircularKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCombinedKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCombinedKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CConstKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CConstKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCustomKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCustomKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDiagKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDiagKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDistanceKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDistanceKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CExponentialKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CExponentialKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianARDKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianARDKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianShiftKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianShiftKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianShortRealKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianShortRealKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHistogramIntersectionKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHistogramIntersectionKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CInverseMultiQuadricKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CInverseMultiQuadricKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJensenShannonKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJensenShannonKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearARDKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearARDKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultiquadricKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultiquadricKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAvgDiagKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAvgDiagKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDiceKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDiceKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFirstElementKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFirstElementKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIdentityKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIdentityKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRidgeKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRidgeKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CScatterKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CScatterKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSqrtDiagKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSqrtDiagKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTanimotoKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTanimotoKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVarianceKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVarianceKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CZeroMeanCenterKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CZeroMeanCenterKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPowerKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPowerKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CProductKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CProductKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPyramidChi2(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPyramidChi2(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRationalQuadraticKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRationalQuadraticKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSigmoidKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSigmoidKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSphericalKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSphericalKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSplineKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSplineKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCommUlongStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCommUlongStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCommWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCommWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDistantSegmentsKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDistantSegmentsKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFixedDegreeStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFixedDegreeStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianMatchStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianMatchStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHistogramWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHistogramWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLocalAlignmentStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLocalAlignmentStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLocalityImprovedStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLocalityImprovedStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMatchWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMatchWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COligoStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COligoStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyMatchStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyMatchStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyMatchWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyMatchWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRegulatoryModulesStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRegulatoryModulesStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSalzbergWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSalzbergWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSimpleLocalityImprovedStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSimpleLocalityImprovedStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSNPStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSNPStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparseSpatialSampleStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparseSpatialSampleStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSpectrumMismatchRBFKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSpectrumMismatchRBFKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSpectrumRBFKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSpectrumRBFKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedCommWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedCommWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedDegreePositionStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedDegreePositionStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedDegreeStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedDegreeStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTensorProductPairKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTensorProductPairKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTStudentKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTStudentKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWaveKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWaveKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWaveletKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWaveletKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedDegreeRBFKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedDegreeRBFKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBinaryLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBinaryLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphObservation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphObservation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLabelsFactory(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLabelsFactory(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassMultipleOutputLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassMultipleOutputLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRegressionLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRegressionLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBitString(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBitString(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCircularBuffer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCircularBuffer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCompressor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCompressor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSerialComputationEngine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSerialComputationEngine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJobResult(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJobResult(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CData(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CData(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDelimiterTokenizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDelimiterTokenizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDynamicObjectArray(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDynamicObjectArray(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHash(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHash(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIndexBlock(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIndexBlock(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIndexBlockGroup(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIndexBlockGroup(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIndexBlockTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIndexBlockTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CListElement(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CListElement(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CList(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CList(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNGramTokenizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNGramTokenizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSignal(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSignal(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredData(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredData(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTime(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTime(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHingeLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHingeLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogLossMargin(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogLossMargin(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSmoothHingeLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSmoothHingeLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSquaredHingeLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSquaredHingeLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSquaredLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSquaredLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBaggingMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBaggingMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBaseMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBaseMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDistanceMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDistanceMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CZeroMean(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CZeroMean(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelStructuredOutputMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelStructuredOutputMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearStructuredOutputMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearStructuredOutputMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNativeMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNativeMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COnlineLinearMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COnlineLinearMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredOutputMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredOutputMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJacobiEllipticFunctions(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJacobiEllipticFunctions(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogDetEstimator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogDetEstimator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNormalSampler(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNormalSampler(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMath(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMath(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandom(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandom(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparseInverseCovariance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparseInverseCovariance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStatistics(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStatistics(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGridSearchModelSelection(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGridSearchModelSelection(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CModelSelectionParameters(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CModelSelectionParameters(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CParameterCombination(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CParameterCombination(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomSearchModelSelection(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomSearchModelSelection(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCAEDDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCAEDDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCDiscriminantEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCDiscriminantEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCEDDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCEDDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCForestEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCForestEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCHDDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCHDDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCLLBDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCLLBDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCOVOEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCOVOEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCOVREncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCOVREncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCRandomDenseEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCRandomDenseEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCRandomSparseEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCRandomSparseEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianNaiveBayes(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianNaiveBayes(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGMNPLib(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGMNPLib(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGMNPSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGMNPSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKNN(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKNN(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLaRank(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLaRank(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassLibLinear(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassLibLinear(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassLibSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassLibSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOCAS(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOCAS(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOneVsOneStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOneVsOneStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOneVsRestStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOneVsRestStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CThresholdRejectionStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CThresholdRejectionStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDixonQTestRejectionStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDixonQTestRejectionStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CScatterSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CScatterSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CShareBoost(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CShareBoost(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBalancedConditionalProbabilityTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBalancedConditionalProbabilityTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomConditionalProbabilityTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomConditionalProbabilityTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRelaxedTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRelaxedTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTron(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTron(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDimensionReductionPreprocessor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDimensionReductionPreprocessor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHomogeneousKernelMap(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHomogeneousKernelMap(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogPlusOne(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogPlusOne(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNormOne(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNormOne(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPNorm(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPNorm(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPruneVarSubMean(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPruneVarSubMean(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomFourierGaussPreproc(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomFourierGaussPreproc(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRescaleFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRescaleFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSortUlongString(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSortUlongString(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSortWordString(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSortWordString(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSumOne(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSumOne(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVR(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVR(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVRLight(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVRLight(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHSIC(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHSIC(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelMeanMatching(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelMeanMatching(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearTimeMMD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearTimeMMD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionCombMaxL2(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionCombMaxL2(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionCombOpt(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionCombOpt(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionMax(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionMax(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionMedian(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionMedian(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionOpt(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionOpt(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CQuadraticTimeMMD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CQuadraticTimeMMD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCCSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCCSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDisjointSet(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDisjointSet(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDualLibQPBMSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDualLibQPBMSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDynProg(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDynProg(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorDataSource(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorDataSource(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraph(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraph(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorType(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorType(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTableFactorType(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTableFactorType(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHMSVMModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHMSVMModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIntronList(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIntronList(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMAPInference(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMAPInference(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassSOLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassSOLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPlif(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPlif(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPlifArray(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPlifArray(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPlifMatrix(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPlifMatrix(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSegmentLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSegmentLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSequence(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSequence(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSequenceLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSequenceLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSOSVMHelper(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSOSVMHelper(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStochasticSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStochasticSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTwoStateModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTwoStateModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDomainAdaptationSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDomainAdaptationSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskClusteredLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskClusteredLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelMaskNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelMaskNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelMaskPairNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelMaskPairNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelPlifNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelPlifNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNode(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNode(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTaxonomy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTaxonomy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelTreeNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelTreeNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskL12LogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskL12LogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskLeastSquaresRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskLeastSquaresRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskLinearMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskLinearMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskROCEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskROCEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskTraceLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskTraceLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTask(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTask(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTaskGroup(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTaskGroup(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTaskTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTaskTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIClassifier(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIClassifier(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIConverter(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIConverter(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUILabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUILabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIMath(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIMath(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIPluginEstimate(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIPluginEstimate(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIPreprocessor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIPreprocessor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIStructure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIStructure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUITime(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUITime(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAveragedPerceptron(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAveragedPerceptron(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFeatureBlockLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFeatureBlockLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLClassification(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLClassification(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLMulticlass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLMulticlass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLOneClass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLOneClass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNearestCentroid(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNearestCentroid(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPerceptron(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPerceptron(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPluginEstimate(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPluginEstimate(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGNPPLib(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGNPPLib(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGNPPSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGNPPSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGPBTSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGPBTSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibLinear(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibLinear(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVMOneClass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVMOneClass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMPDSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMPDSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COnlineLibLinear(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COnlineLibLinear(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COnlineSVMSGD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COnlineSVMSGD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CQPBSVMLib(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CQPBSVMLib(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSGDQN(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSGDQN(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMLight(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMLight(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMLightOneClass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMLightOneClass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMLin(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMLin(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMOcas(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMOcas(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMSGD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMSGD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWDSVMOcas(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWDSVMOcas(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwNativeCacheReader(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwNativeCacheReader(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwNativeCacheWriter(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwNativeCacheWriter(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwAdaptiveLearner(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwAdaptiveLearner(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwNonAdaptiveLearner(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwNonAdaptiveLearner(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVowpalWabbit(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVowpalWabbit(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwEnvironment(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwEnvironment(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwParser(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwParser(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwRegressor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwRegressor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHierarchical(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHierarchical(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKMeans(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKMeans(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedDocConverter(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedDocConverter(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAttenuatedEuclideanDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAttenuatedEuclideanDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBrayCurtisDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBrayCurtisDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCanberraMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCanberraMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCanberraWordDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCanberraWordDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CChebyshewMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CChebyshewMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CChiSquareDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CChiSquareDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCosineDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCosineDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCustomDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCustomDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CEuclideanDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CEuclideanDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGeodesicMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGeodesicMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHammingWordDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHammingWordDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJensenMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJensenMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CManhattanMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CManhattanMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CManhattanWordDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CManhattanWordDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMinkowskiMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMinkowskiMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparseEuclideanDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparseEuclideanDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTanimotoDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTanimotoDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHistogram(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHistogram(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPositionalPWM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPositionalPWM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMajorityVote(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMajorityVote(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanRule(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanRule(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedMajorityVote(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedMajorityVote(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CClusteringAccuracy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CClusteringAccuracy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CClusteringMutualInformation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CClusteringMutualInformation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CContingencyTableEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CContingencyTableEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAccuracyMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAccuracyMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CErrorRateMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CErrorRateMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBALMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBALMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWRACCMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWRACCMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CF1Measure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CF1Measure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossCorrelationMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossCorrelationMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRecallMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRecallMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPrecisionMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPrecisionMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSpecificityMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSpecificityMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationResult(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationResult(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationMKLStorage(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationMKLStorage(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationMulticlassStorage(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationMulticlassStorage(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationPrintOutput(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationPrintOutput(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationSplitting(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationSplitting(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGradientCriterion(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGradientCriterion(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGradientEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGradientEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGradientResult(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGradientResult(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanAbsoluteError(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanAbsoluteError(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanSquaredError(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanSquaredError(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanSquaredLogError(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanSquaredLogError(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassAccuracy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassAccuracy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOVREvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOVREvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPRCEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPRCEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CROCEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CROCEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStratifiedCrossValidationSplitting(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStratifiedCrossValidationSplitting(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredAccuracy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredAccuracy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAlphabet(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAlphabet(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBinnedDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBinnedDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCombinedDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCombinedDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCombinedFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCombinedFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDataGenerator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDataGenerator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDummyFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDummyFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CExplicitSpecFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CExplicitSpecFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFKFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFKFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedDocDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedDocDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedWDFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedWDFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedWDFeaturesTransposed(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedWDFeaturesTransposed(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CImplicitWeightedSpecFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CImplicitWeightedSpecFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLBPPyrDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLBPPyrDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomFourierDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomFourierDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRealFileFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRealFileFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSNPFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSNPFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparsePolyFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparsePolyFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianBlobsDataGenerator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianBlobsDataGenerator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanShiftDataGenerator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanShiftDataGenerator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingHashedDocDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingHashedDocDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingVwFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingVwFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSubset(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSubset(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSubsetStack(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSubsetStack(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTOPFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTOPFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWDFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWDFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBinaryFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBinaryFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCSVFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCSVFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIOBuffer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIOBuffer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVMFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVMFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLineReader(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLineReader(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CParser(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CParser(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSerializableAsciiFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSerializableAsciiFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingAsciiFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingAsciiFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingFileFromFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingFileFromFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingVwCacheFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingVwCacheFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingVwFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingVwFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CANOVAKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CANOVAKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAUCKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAUCKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBesselKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBesselKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCauchyKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCauchyKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CChi2Kernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CChi2Kernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCircularKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCircularKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCombinedKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCombinedKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CConstKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CConstKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCustomKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCustomKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDiagKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDiagKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDistanceKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDistanceKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CExponentialKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CExponentialKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianARDKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianARDKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianShiftKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianShiftKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianShortRealKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianShortRealKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHistogramIntersectionKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHistogramIntersectionKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CInverseMultiQuadricKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CInverseMultiQuadricKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJensenShannonKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJensenShannonKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearARDKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearARDKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultiquadricKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultiquadricKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAvgDiagKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAvgDiagKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDiceKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDiceKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFirstElementKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFirstElementKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIdentityKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIdentityKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRidgeKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRidgeKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CScatterKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CScatterKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSqrtDiagKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSqrtDiagKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTanimotoKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTanimotoKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVarianceKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVarianceKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CZeroMeanCenterKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CZeroMeanCenterKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPowerKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPowerKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CProductKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CProductKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPyramidChi2(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPyramidChi2(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRationalQuadraticKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRationalQuadraticKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSigmoidKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSigmoidKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSphericalKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSphericalKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSplineKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSplineKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCommUlongStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCommUlongStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCommWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCommWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDistantSegmentsKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDistantSegmentsKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFixedDegreeStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFixedDegreeStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianMatchStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianMatchStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHistogramWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHistogramWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLocalAlignmentStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLocalAlignmentStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLocalityImprovedStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLocalityImprovedStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMatchWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMatchWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COligoStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COligoStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyMatchStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyMatchStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyMatchWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyMatchWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRegulatoryModulesStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRegulatoryModulesStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSalzbergWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSalzbergWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSimpleLocalityImprovedStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSimpleLocalityImprovedStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSNPStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSNPStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparseSpatialSampleStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparseSpatialSampleStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSpectrumMismatchRBFKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSpectrumMismatchRBFKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSpectrumRBFKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSpectrumRBFKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedCommWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedCommWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedDegreePositionStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedDegreePositionStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedDegreeStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedDegreeStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTensorProductPairKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTensorProductPairKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTStudentKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTStudentKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWaveKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWaveKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWaveletKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWaveletKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedDegreeRBFKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedDegreeRBFKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBinaryLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBinaryLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphObservation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphObservation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLabelsFactory(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLabelsFactory(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassMultipleOutputLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassMultipleOutputLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRegressionLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRegressionLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBitString(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBitString(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCircularBuffer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCircularBuffer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCompressor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCompressor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSerialComputationEngine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSerialComputationEngine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJobResult(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJobResult(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CData(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CData(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDelimiterTokenizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDelimiterTokenizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDynamicObjectArray(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDynamicObjectArray(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHash(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHash(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIndexBlock(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIndexBlock(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIndexBlockGroup(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIndexBlockGroup(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIndexBlockTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIndexBlockTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CListElement(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CListElement(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CList(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CList(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNGramTokenizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNGramTokenizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSignal(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSignal(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredData(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredData(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTime(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTime(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHingeLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHingeLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogLossMargin(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogLossMargin(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSmoothHingeLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSmoothHingeLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSquaredHingeLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSquaredHingeLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSquaredLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSquaredLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBaggingMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBaggingMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBaseMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBaseMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDistanceMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDistanceMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CZeroMean(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CZeroMean(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelStructuredOutputMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelStructuredOutputMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearStructuredOutputMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearStructuredOutputMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNativeMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNativeMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COnlineLinearMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COnlineLinearMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredOutputMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredOutputMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJacobiEllipticFunctions(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJacobiEllipticFunctions(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogDetEstimator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogDetEstimator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNormalSampler(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNormalSampler(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMath(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMath(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandom(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandom(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparseInverseCovariance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparseInverseCovariance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStatistics(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStatistics(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGridSearchModelSelection(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGridSearchModelSelection(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CModelSelectionParameters(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CModelSelectionParameters(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CParameterCombination(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CParameterCombination(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomSearchModelSelection(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomSearchModelSelection(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCAEDDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCAEDDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCDiscriminantEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCDiscriminantEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCEDDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCEDDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCForestEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCForestEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCHDDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCHDDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCLLBDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCLLBDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCOVOEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCOVOEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCOVREncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCOVREncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCRandomDenseEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCRandomDenseEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCRandomSparseEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCRandomSparseEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianNaiveBayes(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianNaiveBayes(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGMNPLib(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGMNPLib(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGMNPSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGMNPSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKNN(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKNN(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLaRank(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLaRank(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassLibLinear(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassLibLinear(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassLibSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassLibSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOCAS(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOCAS(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOneVsOneStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOneVsOneStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOneVsRestStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOneVsRestStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CThresholdRejectionStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CThresholdRejectionStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDixonQTestRejectionStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDixonQTestRejectionStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CScatterSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CScatterSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CShareBoost(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CShareBoost(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBalancedConditionalProbabilityTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBalancedConditionalProbabilityTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomConditionalProbabilityTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomConditionalProbabilityTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRelaxedTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRelaxedTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTron(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTron(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDimensionReductionPreprocessor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDimensionReductionPreprocessor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHomogeneousKernelMap(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHomogeneousKernelMap(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogPlusOne(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogPlusOne(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNormOne(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNormOne(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPNorm(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPNorm(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPruneVarSubMean(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPruneVarSubMean(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomFourierGaussPreproc(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomFourierGaussPreproc(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRescaleFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRescaleFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSortUlongString(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSortUlongString(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSortWordString(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSortWordString(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSumOne(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSumOne(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVR(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVR(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVRLight(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVRLight(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHSIC(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHSIC(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelMeanMatching(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelMeanMatching(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearTimeMMD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearTimeMMD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionCombMaxL2(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionCombMaxL2(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionCombOpt(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionCombOpt(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionMax(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionMax(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionMedian(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionMedian(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionOpt(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionOpt(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CQuadraticTimeMMD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CQuadraticTimeMMD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCCSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCCSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDisjointSet(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDisjointSet(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDualLibQPBMSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDualLibQPBMSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDynProg(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDynProg(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorDataSource(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorDataSource(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraph(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraph(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorType(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorType(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTableFactorType(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTableFactorType(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHMSVMModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHMSVMModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIntronList(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIntronList(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMAPInference(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMAPInference(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassSOLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassSOLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPlif(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPlif(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPlifArray(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPlifArray(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPlifMatrix(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPlifMatrix(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSegmentLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSegmentLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSequence(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSequence(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSequenceLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSequenceLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSOSVMHelper(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSOSVMHelper(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStochasticSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStochasticSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTwoStateModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTwoStateModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDomainAdaptationSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDomainAdaptationSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskClusteredLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskClusteredLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelMaskNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelMaskNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelMaskPairNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelMaskPairNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelPlifNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelPlifNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNode(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNode(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTaxonomy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTaxonomy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelTreeNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelTreeNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskL12LogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskL12LogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskLeastSquaresRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskLeastSquaresRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskLinearMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskLinearMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskROCEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskROCEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskTraceLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskTraceLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTask(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTask(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTaskGroup(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTaskGroup(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTaskTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTaskTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIClassifier(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIClassifier(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIConverter(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIConverter(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUILabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUILabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIMath(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIMath(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIPluginEstimate(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIPluginEstimate(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIPreprocessor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIPreprocessor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIStructure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIStructure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUITime(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUITime(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAveragedPerceptron(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAveragedPerceptron(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFeatureBlockLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFeatureBlockLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLClassification(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLClassification(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLMulticlass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLMulticlass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLOneClass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLOneClass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNearestCentroid(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNearestCentroid(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPerceptron(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPerceptron(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPluginEstimate(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPluginEstimate(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGNPPLib(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGNPPLib(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGNPPSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGNPPSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGPBTSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGPBTSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibLinear(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibLinear(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVMOneClass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVMOneClass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMPDSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMPDSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COnlineLibLinear(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COnlineLibLinear(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COnlineSVMSGD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COnlineSVMSGD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CQPBSVMLib(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CQPBSVMLib(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSGDQN(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSGDQN(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMLight(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMLight(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMLightOneClass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMLightOneClass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMLin(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMLin(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMOcas(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMOcas(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMSGD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMSGD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWDSVMOcas(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWDSVMOcas(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwNativeCacheReader(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwNativeCacheReader(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwNativeCacheWriter(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwNativeCacheWriter(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwAdaptiveLearner(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwAdaptiveLearner(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwNonAdaptiveLearner(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwNonAdaptiveLearner(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVowpalWabbit(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVowpalWabbit(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwEnvironment(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwEnvironment(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwParser(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwParser(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwRegressor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwRegressor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHierarchical(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHierarchical(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKMeans(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKMeans(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedDocConverter(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedDocConverter(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAttenuatedEuclideanDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAttenuatedEuclideanDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBrayCurtisDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBrayCurtisDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCanberraMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCanberraMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCanberraWordDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCanberraWordDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CChebyshewMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CChebyshewMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CChiSquareDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CChiSquareDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCosineDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCosineDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCustomDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCustomDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CEuclideanDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CEuclideanDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGeodesicMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGeodesicMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHammingWordDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHammingWordDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJensenMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJensenMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CManhattanMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CManhattanMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CManhattanWordDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CManhattanWordDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMinkowskiMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMinkowskiMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparseEuclideanDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparseEuclideanDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTanimotoDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTanimotoDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHistogram(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHistogram(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPositionalPWM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPositionalPWM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMajorityVote(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMajorityVote(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanRule(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanRule(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedMajorityVote(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedMajorityVote(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CClusteringAccuracy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CClusteringAccuracy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CClusteringMutualInformation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CClusteringMutualInformation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CContingencyTableEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CContingencyTableEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAccuracyMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAccuracyMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CErrorRateMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CErrorRateMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBALMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBALMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWRACCMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWRACCMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CF1Measure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CF1Measure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossCorrelationMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossCorrelationMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRecallMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRecallMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPrecisionMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPrecisionMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSpecificityMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSpecificityMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationResult(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationResult(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationMKLStorage(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationMKLStorage(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationMulticlassStorage(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationMulticlassStorage(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationPrintOutput(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationPrintOutput(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationSplitting(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationSplitting(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGradientCriterion(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGradientCriterion(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGradientEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGradientEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGradientResult(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGradientResult(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanAbsoluteError(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanAbsoluteError(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanSquaredError(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanSquaredError(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanSquaredLogError(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanSquaredLogError(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassAccuracy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassAccuracy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOVREvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOVREvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPRCEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPRCEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CROCEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CROCEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStratifiedCrossValidationSplitting(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStratifiedCrossValidationSplitting(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredAccuracy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredAccuracy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAlphabet(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAlphabet(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBinnedDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBinnedDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCombinedDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCombinedDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCombinedFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCombinedFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDataGenerator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDataGenerator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDummyFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDummyFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CExplicitSpecFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CExplicitSpecFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFKFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFKFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedDocDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedDocDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedWDFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedWDFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedWDFeaturesTransposed(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedWDFeaturesTransposed(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CImplicitWeightedSpecFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CImplicitWeightedSpecFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLBPPyrDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLBPPyrDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomFourierDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomFourierDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRealFileFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRealFileFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSNPFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSNPFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparsePolyFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparsePolyFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianBlobsDataGenerator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianBlobsDataGenerator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanShiftDataGenerator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanShiftDataGenerator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingHashedDocDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingHashedDocDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingVwFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingVwFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSubset(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSubset(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSubsetStack(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSubsetStack(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTOPFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTOPFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWDFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWDFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBinaryFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBinaryFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCSVFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCSVFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIOBuffer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIOBuffer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVMFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVMFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLineReader(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLineReader(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CParser(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CParser(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSerializableAsciiFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSerializableAsciiFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingAsciiFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingAsciiFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingFileFromFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingFileFromFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingVwCacheFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingVwCacheFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingVwFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingVwFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CANOVAKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CANOVAKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAUCKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAUCKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBesselKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBesselKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCauchyKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCauchyKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CChi2Kernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CChi2Kernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCircularKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCircularKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCombinedKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCombinedKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CConstKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CConstKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCustomKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCustomKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDiagKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDiagKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDistanceKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDistanceKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CExponentialKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CExponentialKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianARDKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianARDKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianShiftKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianShiftKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianShortRealKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianShortRealKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHistogramIntersectionKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHistogramIntersectionKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CInverseMultiQuadricKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CInverseMultiQuadricKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJensenShannonKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJensenShannonKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearARDKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearARDKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultiquadricKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultiquadricKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAvgDiagKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAvgDiagKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDiceKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDiceKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFirstElementKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFirstElementKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIdentityKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIdentityKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRidgeKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRidgeKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CScatterKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CScatterKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSqrtDiagKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSqrtDiagKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTanimotoKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTanimotoKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVarianceKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVarianceKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CZeroMeanCenterKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CZeroMeanCenterKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPowerKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPowerKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CProductKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CProductKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPyramidChi2(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPyramidChi2(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRationalQuadraticKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRationalQuadraticKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSigmoidKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSigmoidKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSphericalKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSphericalKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSplineKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSplineKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCommUlongStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCommUlongStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCommWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCommWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDistantSegmentsKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDistantSegmentsKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFixedDegreeStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFixedDegreeStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianMatchStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianMatchStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHistogramWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHistogramWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLocalAlignmentStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLocalAlignmentStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLocalityImprovedStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLocalityImprovedStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMatchWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMatchWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COligoStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COligoStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyMatchStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyMatchStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyMatchWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyMatchWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRegulatoryModulesStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRegulatoryModulesStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSalzbergWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSalzbergWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSimpleLocalityImprovedStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSimpleLocalityImprovedStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSNPStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSNPStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparseSpatialSampleStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparseSpatialSampleStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSpectrumMismatchRBFKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSpectrumMismatchRBFKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSpectrumRBFKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSpectrumRBFKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedCommWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedCommWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedDegreePositionStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedDegreePositionStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedDegreeStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedDegreeStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTensorProductPairKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTensorProductPairKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTStudentKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTStudentKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWaveKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWaveKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWaveletKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWaveletKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedDegreeRBFKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedDegreeRBFKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBinaryLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBinaryLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphObservation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphObservation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLabelsFactory(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLabelsFactory(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassMultipleOutputLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassMultipleOutputLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRegressionLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRegressionLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBitString(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBitString(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCircularBuffer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCircularBuffer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCompressor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCompressor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSerialComputationEngine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSerialComputationEngine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJobResult(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJobResult(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CData(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CData(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDelimiterTokenizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDelimiterTokenizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDynamicObjectArray(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDynamicObjectArray(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHash(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHash(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIndexBlock(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIndexBlock(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIndexBlockGroup(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIndexBlockGroup(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIndexBlockTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIndexBlockTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CListElement(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CListElement(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CList(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CList(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNGramTokenizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNGramTokenizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSignal(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSignal(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredData(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredData(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTime(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTime(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHingeLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHingeLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogLossMargin(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogLossMargin(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSmoothHingeLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSmoothHingeLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSquaredHingeLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSquaredHingeLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSquaredLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSquaredLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBaggingMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBaggingMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBaseMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBaseMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDistanceMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDistanceMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CZeroMean(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CZeroMean(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelStructuredOutputMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelStructuredOutputMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearStructuredOutputMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearStructuredOutputMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNativeMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNativeMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COnlineLinearMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COnlineLinearMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredOutputMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredOutputMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJacobiEllipticFunctions(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJacobiEllipticFunctions(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogDetEstimator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogDetEstimator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNormalSampler(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNormalSampler(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMath(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMath(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandom(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandom(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparseInverseCovariance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparseInverseCovariance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStatistics(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStatistics(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGridSearchModelSelection(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGridSearchModelSelection(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CModelSelectionParameters(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CModelSelectionParameters(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CParameterCombination(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CParameterCombination(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomSearchModelSelection(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomSearchModelSelection(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCAEDDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCAEDDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCDiscriminantEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCDiscriminantEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCEDDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCEDDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCForestEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCForestEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCHDDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCHDDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCLLBDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCLLBDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCOVOEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCOVOEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCOVREncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCOVREncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCRandomDenseEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCRandomDenseEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCRandomSparseEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCRandomSparseEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianNaiveBayes(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianNaiveBayes(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGMNPLib(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGMNPLib(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGMNPSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGMNPSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKNN(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKNN(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLaRank(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLaRank(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassLibLinear(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassLibLinear(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassLibSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassLibSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOCAS(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOCAS(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOneVsOneStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOneVsOneStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOneVsRestStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOneVsRestStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CThresholdRejectionStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CThresholdRejectionStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDixonQTestRejectionStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDixonQTestRejectionStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CScatterSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CScatterSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CShareBoost(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CShareBoost(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBalancedConditionalProbabilityTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBalancedConditionalProbabilityTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomConditionalProbabilityTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomConditionalProbabilityTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRelaxedTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRelaxedTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTron(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTron(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDimensionReductionPreprocessor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDimensionReductionPreprocessor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHomogeneousKernelMap(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHomogeneousKernelMap(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogPlusOne(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogPlusOne(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNormOne(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNormOne(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPNorm(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPNorm(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPruneVarSubMean(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPruneVarSubMean(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomFourierGaussPreproc(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomFourierGaussPreproc(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRescaleFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRescaleFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSortUlongString(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSortUlongString(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSortWordString(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSortWordString(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSumOne(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSumOne(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVR(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVR(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVRLight(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVRLight(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHSIC(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHSIC(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelMeanMatching(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelMeanMatching(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearTimeMMD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearTimeMMD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionCombMaxL2(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionCombMaxL2(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionCombOpt(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionCombOpt(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionMax(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionMax(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionMedian(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionMedian(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionOpt(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionOpt(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CQuadraticTimeMMD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CQuadraticTimeMMD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCCSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCCSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDisjointSet(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDisjointSet(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDualLibQPBMSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDualLibQPBMSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDynProg(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDynProg(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorDataSource(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorDataSource(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraph(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraph(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorType(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorType(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTableFactorType(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTableFactorType(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHMSVMModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHMSVMModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIntronList(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIntronList(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMAPInference(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMAPInference(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassSOLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassSOLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPlif(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPlif(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPlifArray(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPlifArray(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPlifMatrix(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPlifMatrix(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSegmentLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSegmentLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSequence(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSequence(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSequenceLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSequenceLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSOSVMHelper(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSOSVMHelper(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStochasticSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStochasticSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTwoStateModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTwoStateModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDomainAdaptationSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDomainAdaptationSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskClusteredLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskClusteredLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelMaskNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelMaskNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelMaskPairNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelMaskPairNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelPlifNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelPlifNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNode(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNode(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTaxonomy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTaxonomy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelTreeNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelTreeNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskL12LogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskL12LogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskLeastSquaresRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskLeastSquaresRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskLinearMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskLinearMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskROCEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskROCEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskTraceLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskTraceLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTask(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTask(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTaskGroup(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTaskGroup(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTaskTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTaskTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIClassifier(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIClassifier(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIConverter(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIConverter(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUILabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUILabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIMath(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIMath(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIPluginEstimate(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIPluginEstimate(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIPreprocessor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIPreprocessor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIStructure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIStructure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUITime(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUITime(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAveragedPerceptron(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAveragedPerceptron(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFeatureBlockLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFeatureBlockLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLClassification(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLClassification(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLMulticlass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLMulticlass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLOneClass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLOneClass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNearestCentroid(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNearestCentroid(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPerceptron(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPerceptron(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPluginEstimate(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPluginEstimate(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGNPPLib(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGNPPLib(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGNPPSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGNPPSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGPBTSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGPBTSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibLinear(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibLinear(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVMOneClass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVMOneClass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMPDSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMPDSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COnlineLibLinear(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COnlineLibLinear(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COnlineSVMSGD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COnlineSVMSGD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CQPBSVMLib(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CQPBSVMLib(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSGDQN(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSGDQN(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMLight(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMLight(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMLightOneClass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMLightOneClass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMLin(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMLin(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMOcas(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMOcas(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMSGD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMSGD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWDSVMOcas(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWDSVMOcas(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwNativeCacheReader(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwNativeCacheReader(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwNativeCacheWriter(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwNativeCacheWriter(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwAdaptiveLearner(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwAdaptiveLearner(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwNonAdaptiveLearner(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwNonAdaptiveLearner(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVowpalWabbit(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVowpalWabbit(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwEnvironment(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwEnvironment(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwParser(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwParser(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwRegressor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwRegressor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHierarchical(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHierarchical(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKMeans(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKMeans(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedDocConverter(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedDocConverter(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAttenuatedEuclideanDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAttenuatedEuclideanDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBrayCurtisDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBrayCurtisDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCanberraMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCanberraMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCanberraWordDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCanberraWordDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CChebyshewMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CChebyshewMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CChiSquareDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CChiSquareDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCosineDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCosineDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCustomDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCustomDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CEuclideanDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CEuclideanDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGeodesicMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGeodesicMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHammingWordDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHammingWordDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJensenMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJensenMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CManhattanMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CManhattanMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CManhattanWordDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CManhattanWordDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMinkowskiMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMinkowskiMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparseEuclideanDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparseEuclideanDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTanimotoDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTanimotoDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHistogram(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHistogram(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPositionalPWM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPositionalPWM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMajorityVote(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMajorityVote(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanRule(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanRule(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedMajorityVote(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedMajorityVote(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CClusteringAccuracy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CClusteringAccuracy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CClusteringMutualInformation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CClusteringMutualInformation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CContingencyTableEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CContingencyTableEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAccuracyMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAccuracyMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CErrorRateMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CErrorRateMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBALMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBALMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWRACCMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWRACCMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CF1Measure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CF1Measure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossCorrelationMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossCorrelationMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRecallMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRecallMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPrecisionMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPrecisionMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSpecificityMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSpecificityMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationResult(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationResult(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationMKLStorage(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationMKLStorage(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationMulticlassStorage(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationMulticlassStorage(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationPrintOutput(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationPrintOutput(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationSplitting(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationSplitting(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGradientCriterion(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGradientCriterion(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGradientEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGradientEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGradientResult(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGradientResult(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanAbsoluteError(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanAbsoluteError(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanSquaredError(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanSquaredError(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanSquaredLogError(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanSquaredLogError(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassAccuracy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassAccuracy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOVREvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOVREvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPRCEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPRCEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CROCEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CROCEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStratifiedCrossValidationSplitting(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStratifiedCrossValidationSplitting(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredAccuracy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredAccuracy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAlphabet(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAlphabet(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBinnedDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBinnedDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCombinedDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCombinedDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCombinedFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCombinedFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDataGenerator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDataGenerator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDummyFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDummyFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CExplicitSpecFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CExplicitSpecFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFKFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFKFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedDocDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedDocDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedWDFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedWDFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedWDFeaturesTransposed(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedWDFeaturesTransposed(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CImplicitWeightedSpecFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CImplicitWeightedSpecFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLBPPyrDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLBPPyrDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomFourierDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomFourierDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRealFileFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRealFileFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSNPFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSNPFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparsePolyFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparsePolyFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianBlobsDataGenerator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianBlobsDataGenerator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanShiftDataGenerator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanShiftDataGenerator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingHashedDocDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingHashedDocDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingVwFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingVwFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSubset(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSubset(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSubsetStack(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSubsetStack(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTOPFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTOPFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWDFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWDFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBinaryFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBinaryFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCSVFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCSVFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIOBuffer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIOBuffer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVMFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVMFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLineReader(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLineReader(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CParser(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CParser(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSerializableAsciiFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSerializableAsciiFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingAsciiFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingAsciiFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingFileFromFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingFileFromFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingVwCacheFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingVwCacheFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingVwFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingVwFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CANOVAKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CANOVAKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAUCKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAUCKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBesselKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBesselKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCauchyKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCauchyKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CChi2Kernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CChi2Kernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCircularKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCircularKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCombinedKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCombinedKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CConstKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CConstKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCustomKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCustomKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDiagKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDiagKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDistanceKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDistanceKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CExponentialKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CExponentialKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianARDKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianARDKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianShiftKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianShiftKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianShortRealKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianShortRealKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHistogramIntersectionKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHistogramIntersectionKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CInverseMultiQuadricKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CInverseMultiQuadricKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJensenShannonKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJensenShannonKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearARDKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearARDKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultiquadricKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultiquadricKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAvgDiagKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAvgDiagKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDiceKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDiceKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFirstElementKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFirstElementKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIdentityKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIdentityKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRidgeKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRidgeKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CScatterKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CScatterKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSqrtDiagKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSqrtDiagKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTanimotoKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTanimotoKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVarianceKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVarianceKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CZeroMeanCenterKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CZeroMeanCenterKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPowerKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPowerKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CProductKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CProductKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPyramidChi2(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPyramidChi2(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRationalQuadraticKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRationalQuadraticKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSigmoidKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSigmoidKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSphericalKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSphericalKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSplineKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSplineKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCommUlongStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCommUlongStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCommWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCommWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDistantSegmentsKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDistantSegmentsKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFixedDegreeStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFixedDegreeStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianMatchStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianMatchStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHistogramWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHistogramWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLocalAlignmentStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLocalAlignmentStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLocalityImprovedStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLocalityImprovedStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMatchWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMatchWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COligoStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COligoStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyMatchStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyMatchStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyMatchWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyMatchWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRegulatoryModulesStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRegulatoryModulesStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSalzbergWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSalzbergWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSimpleLocalityImprovedStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSimpleLocalityImprovedStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSNPStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSNPStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparseSpatialSampleStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparseSpatialSampleStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSpectrumMismatchRBFKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSpectrumMismatchRBFKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSpectrumRBFKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSpectrumRBFKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedCommWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedCommWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedDegreePositionStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedDegreePositionStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedDegreeStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedDegreeStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTensorProductPairKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTensorProductPairKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTStudentKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTStudentKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWaveKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWaveKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWaveletKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWaveletKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedDegreeRBFKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedDegreeRBFKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBinaryLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBinaryLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphObservation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphObservation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLabelsFactory(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLabelsFactory(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassMultipleOutputLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassMultipleOutputLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRegressionLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRegressionLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBitString(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBitString(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCircularBuffer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCircularBuffer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCompressor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCompressor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSerialComputationEngine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSerialComputationEngine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJobResult(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJobResult(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CData(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CData(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDelimiterTokenizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDelimiterTokenizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDynamicObjectArray(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDynamicObjectArray(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHash(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHash(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIndexBlock(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIndexBlock(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIndexBlockGroup(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIndexBlockGroup(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIndexBlockTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIndexBlockTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CListElement(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CListElement(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CList(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CList(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNGramTokenizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNGramTokenizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSignal(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSignal(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredData(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredData(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTime(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTime(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHingeLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHingeLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogLossMargin(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogLossMargin(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSmoothHingeLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSmoothHingeLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSquaredHingeLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSquaredHingeLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSquaredLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSquaredLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBaggingMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBaggingMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBaseMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBaseMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDistanceMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDistanceMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CZeroMean(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CZeroMean(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelStructuredOutputMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelStructuredOutputMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearStructuredOutputMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearStructuredOutputMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNativeMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNativeMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COnlineLinearMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COnlineLinearMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredOutputMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredOutputMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJacobiEllipticFunctions(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJacobiEllipticFunctions(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogDetEstimator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogDetEstimator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNormalSampler(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNormalSampler(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMath(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMath(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandom(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandom(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparseInverseCovariance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparseInverseCovariance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStatistics(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStatistics(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGridSearchModelSelection(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGridSearchModelSelection(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CModelSelectionParameters(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CModelSelectionParameters(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CParameterCombination(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CParameterCombination(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomSearchModelSelection(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomSearchModelSelection(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCAEDDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCAEDDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCDiscriminantEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCDiscriminantEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCEDDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCEDDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCForestEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCForestEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCHDDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCHDDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCLLBDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCLLBDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCOVOEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCOVOEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCOVREncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCOVREncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCRandomDenseEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCRandomDenseEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCRandomSparseEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCRandomSparseEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianNaiveBayes(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianNaiveBayes(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGMNPLib(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGMNPLib(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGMNPSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGMNPSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKNN(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKNN(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLaRank(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLaRank(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassLibLinear(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassLibLinear(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassLibSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassLibSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOCAS(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOCAS(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOneVsOneStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOneVsOneStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOneVsRestStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOneVsRestStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CThresholdRejectionStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CThresholdRejectionStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDixonQTestRejectionStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDixonQTestRejectionStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CScatterSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CScatterSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CShareBoost(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CShareBoost(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBalancedConditionalProbabilityTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBalancedConditionalProbabilityTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomConditionalProbabilityTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomConditionalProbabilityTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRelaxedTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRelaxedTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTron(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTron(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDimensionReductionPreprocessor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDimensionReductionPreprocessor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHomogeneousKernelMap(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHomogeneousKernelMap(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogPlusOne(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogPlusOne(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNormOne(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNormOne(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPNorm(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPNorm(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPruneVarSubMean(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPruneVarSubMean(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomFourierGaussPreproc(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomFourierGaussPreproc(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRescaleFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRescaleFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSortUlongString(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSortUlongString(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSortWordString(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSortWordString(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSumOne(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSumOne(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVR(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVR(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVRLight(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVRLight(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHSIC(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHSIC(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelMeanMatching(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelMeanMatching(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearTimeMMD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearTimeMMD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionCombMaxL2(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionCombMaxL2(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionCombOpt(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionCombOpt(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionMax(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionMax(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionMedian(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionMedian(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionOpt(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionOpt(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CQuadraticTimeMMD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CQuadraticTimeMMD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCCSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCCSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDisjointSet(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDisjointSet(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDualLibQPBMSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDualLibQPBMSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDynProg(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDynProg(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorDataSource(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorDataSource(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraph(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraph(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorType(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorType(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTableFactorType(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTableFactorType(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHMSVMModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHMSVMModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIntronList(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIntronList(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMAPInference(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMAPInference(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassSOLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassSOLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPlif(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPlif(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPlifArray(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPlifArray(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPlifMatrix(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPlifMatrix(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSegmentLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSegmentLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSequence(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSequence(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSequenceLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSequenceLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSOSVMHelper(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSOSVMHelper(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStochasticSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStochasticSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTwoStateModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTwoStateModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDomainAdaptationSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDomainAdaptationSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskClusteredLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskClusteredLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelMaskNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelMaskNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelMaskPairNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelMaskPairNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelPlifNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelPlifNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNode(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNode(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTaxonomy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTaxonomy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelTreeNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelTreeNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskL12LogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskL12LogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskLeastSquaresRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskLeastSquaresRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskLinearMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskLinearMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskROCEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskROCEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskTraceLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskTraceLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTask(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTask(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTaskGroup(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTaskGroup(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTaskTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTaskTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIClassifier(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIClassifier(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIConverter(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIConverter(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUILabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUILabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIMath(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIMath(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIPluginEstimate(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIPluginEstimate(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIPreprocessor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIPreprocessor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIStructure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIStructure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUITime(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUITime(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAveragedPerceptron(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAveragedPerceptron(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFeatureBlockLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFeatureBlockLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLClassification(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLClassification(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLMulticlass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLMulticlass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLOneClass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLOneClass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNearestCentroid(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNearestCentroid(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPerceptron(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPerceptron(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPluginEstimate(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPluginEstimate(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGNPPLib(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGNPPLib(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGNPPSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGNPPSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGPBTSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGPBTSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibLinear(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibLinear(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVMOneClass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVMOneClass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMPDSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMPDSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COnlineLibLinear(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COnlineLibLinear(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COnlineSVMSGD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COnlineSVMSGD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CQPBSVMLib(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CQPBSVMLib(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSGDQN(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSGDQN(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMLight(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMLight(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMLightOneClass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMLightOneClass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMLin(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMLin(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMOcas(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMOcas(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMSGD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMSGD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWDSVMOcas(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWDSVMOcas(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwNativeCacheReader(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwNativeCacheReader(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwNativeCacheWriter(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwNativeCacheWriter(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwAdaptiveLearner(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwAdaptiveLearner(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwNonAdaptiveLearner(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwNonAdaptiveLearner(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVowpalWabbit(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVowpalWabbit(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwEnvironment(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwEnvironment(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwParser(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwParser(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwRegressor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwRegressor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHierarchical(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHierarchical(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKMeans(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKMeans(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedDocConverter(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedDocConverter(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAttenuatedEuclideanDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAttenuatedEuclideanDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBrayCurtisDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBrayCurtisDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCanberraMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCanberraMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCanberraWordDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCanberraWordDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CChebyshewMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CChebyshewMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CChiSquareDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CChiSquareDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCosineDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCosineDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCustomDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCustomDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CEuclideanDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CEuclideanDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGeodesicMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGeodesicMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHammingWordDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHammingWordDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJensenMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJensenMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CManhattanMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CManhattanMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CManhattanWordDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CManhattanWordDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMinkowskiMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMinkowskiMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparseEuclideanDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparseEuclideanDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTanimotoDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTanimotoDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHistogram(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHistogram(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPositionalPWM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPositionalPWM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMajorityVote(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMajorityVote(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanRule(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanRule(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedMajorityVote(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedMajorityVote(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CClusteringAccuracy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CClusteringAccuracy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CClusteringMutualInformation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CClusteringMutualInformation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CContingencyTableEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CContingencyTableEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAccuracyMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAccuracyMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CErrorRateMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CErrorRateMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBALMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBALMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWRACCMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWRACCMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CF1Measure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CF1Measure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossCorrelationMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossCorrelationMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRecallMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRecallMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPrecisionMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPrecisionMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSpecificityMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSpecificityMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationResult(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationResult(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationMKLStorage(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationMKLStorage(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationMulticlassStorage(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationMulticlassStorage(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationPrintOutput(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationPrintOutput(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationSplitting(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationSplitting(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGradientCriterion(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGradientCriterion(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGradientEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGradientEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGradientResult(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGradientResult(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanAbsoluteError(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanAbsoluteError(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanSquaredError(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanSquaredError(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanSquaredLogError(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanSquaredLogError(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassAccuracy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassAccuracy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOVREvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOVREvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPRCEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPRCEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CROCEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CROCEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStratifiedCrossValidationSplitting(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStratifiedCrossValidationSplitting(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredAccuracy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredAccuracy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAlphabet(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAlphabet(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBinnedDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBinnedDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCombinedDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCombinedDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCombinedFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCombinedFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDataGenerator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDataGenerator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDummyFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDummyFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CExplicitSpecFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CExplicitSpecFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFKFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFKFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedDocDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedDocDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedWDFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedWDFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedWDFeaturesTransposed(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedWDFeaturesTransposed(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CImplicitWeightedSpecFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CImplicitWeightedSpecFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLBPPyrDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLBPPyrDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomFourierDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomFourierDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRealFileFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRealFileFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSNPFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSNPFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparsePolyFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparsePolyFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianBlobsDataGenerator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianBlobsDataGenerator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanShiftDataGenerator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanShiftDataGenerator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingHashedDocDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingHashedDocDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingVwFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingVwFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSubset(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSubset(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSubsetStack(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSubsetStack(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTOPFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTOPFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWDFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWDFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBinaryFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBinaryFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCSVFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCSVFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIOBuffer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIOBuffer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVMFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVMFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLineReader(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLineReader(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CParser(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CParser(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSerializableAsciiFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSerializableAsciiFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingAsciiFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingAsciiFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingFileFromFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingFileFromFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingVwCacheFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingVwCacheFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingVwFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingVwFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CANOVAKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CANOVAKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAUCKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAUCKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBesselKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBesselKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCauchyKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCauchyKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CChi2Kernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CChi2Kernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCircularKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCircularKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCombinedKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCombinedKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CConstKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CConstKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCustomKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCustomKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDiagKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDiagKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDistanceKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDistanceKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CExponentialKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CExponentialKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianARDKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianARDKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianShiftKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianShiftKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianShortRealKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianShortRealKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHistogramIntersectionKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHistogramIntersectionKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CInverseMultiQuadricKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CInverseMultiQuadricKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJensenShannonKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJensenShannonKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearARDKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearARDKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultiquadricKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultiquadricKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAvgDiagKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAvgDiagKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDiceKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDiceKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFirstElementKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFirstElementKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIdentityKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIdentityKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRidgeKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRidgeKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CScatterKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CScatterKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSqrtDiagKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSqrtDiagKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTanimotoKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTanimotoKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVarianceKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVarianceKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CZeroMeanCenterKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CZeroMeanCenterKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPowerKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPowerKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CProductKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CProductKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPyramidChi2(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPyramidChi2(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRationalQuadraticKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRationalQuadraticKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSigmoidKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSigmoidKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSphericalKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSphericalKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSplineKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSplineKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCommUlongStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCommUlongStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCommWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCommWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDistantSegmentsKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDistantSegmentsKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFixedDegreeStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFixedDegreeStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianMatchStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianMatchStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHistogramWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHistogramWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLocalAlignmentStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLocalAlignmentStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLocalityImprovedStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLocalityImprovedStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMatchWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMatchWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COligoStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COligoStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyMatchStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyMatchStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyMatchWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyMatchWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRegulatoryModulesStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRegulatoryModulesStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSalzbergWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSalzbergWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSimpleLocalityImprovedStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSimpleLocalityImprovedStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSNPStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSNPStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparseSpatialSampleStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparseSpatialSampleStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSpectrumMismatchRBFKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSpectrumMismatchRBFKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSpectrumRBFKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSpectrumRBFKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedCommWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedCommWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedDegreePositionStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedDegreePositionStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedDegreeStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedDegreeStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTensorProductPairKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTensorProductPairKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTStudentKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTStudentKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWaveKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWaveKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWaveletKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWaveletKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedDegreeRBFKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedDegreeRBFKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBinaryLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBinaryLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphObservation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphObservation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLabelsFactory(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLabelsFactory(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassMultipleOutputLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassMultipleOutputLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRegressionLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRegressionLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBitString(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBitString(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCircularBuffer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCircularBuffer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCompressor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCompressor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSerialComputationEngine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSerialComputationEngine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJobResult(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJobResult(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CData(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CData(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDelimiterTokenizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDelimiterTokenizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDynamicObjectArray(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDynamicObjectArray(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHash(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHash(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIndexBlock(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIndexBlock(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIndexBlockGroup(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIndexBlockGroup(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIndexBlockTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIndexBlockTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CListElement(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CListElement(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CList(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CList(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNGramTokenizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNGramTokenizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSignal(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSignal(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredData(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredData(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTime(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTime(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHingeLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHingeLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogLossMargin(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogLossMargin(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSmoothHingeLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSmoothHingeLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSquaredHingeLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSquaredHingeLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSquaredLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSquaredLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBaggingMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBaggingMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBaseMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBaseMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDistanceMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDistanceMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CZeroMean(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CZeroMean(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelStructuredOutputMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelStructuredOutputMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearStructuredOutputMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearStructuredOutputMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNativeMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNativeMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COnlineLinearMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COnlineLinearMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredOutputMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredOutputMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJacobiEllipticFunctions(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJacobiEllipticFunctions(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogDetEstimator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogDetEstimator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNormalSampler(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNormalSampler(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMath(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMath(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandom(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandom(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparseInverseCovariance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparseInverseCovariance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStatistics(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStatistics(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGridSearchModelSelection(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGridSearchModelSelection(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CModelSelectionParameters(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CModelSelectionParameters(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CParameterCombination(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CParameterCombination(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomSearchModelSelection(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomSearchModelSelection(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCAEDDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCAEDDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCDiscriminantEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCDiscriminantEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCEDDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCEDDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCForestEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCForestEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCHDDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCHDDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCLLBDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCLLBDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCOVOEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCOVOEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCOVREncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCOVREncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCRandomDenseEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCRandomDenseEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCRandomSparseEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCRandomSparseEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianNaiveBayes(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianNaiveBayes(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGMNPLib(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGMNPLib(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGMNPSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGMNPSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKNN(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKNN(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLaRank(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLaRank(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassLibLinear(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassLibLinear(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassLibSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassLibSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOCAS(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOCAS(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOneVsOneStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOneVsOneStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOneVsRestStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOneVsRestStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CThresholdRejectionStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CThresholdRejectionStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDixonQTestRejectionStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDixonQTestRejectionStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CScatterSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CScatterSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CShareBoost(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CShareBoost(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBalancedConditionalProbabilityTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBalancedConditionalProbabilityTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomConditionalProbabilityTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomConditionalProbabilityTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRelaxedTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRelaxedTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTron(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTron(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDimensionReductionPreprocessor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDimensionReductionPreprocessor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHomogeneousKernelMap(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHomogeneousKernelMap(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogPlusOne(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogPlusOne(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNormOne(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNormOne(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPNorm(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPNorm(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPruneVarSubMean(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPruneVarSubMean(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomFourierGaussPreproc(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomFourierGaussPreproc(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRescaleFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRescaleFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSortUlongString(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSortUlongString(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSortWordString(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSortWordString(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSumOne(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSumOne(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVR(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVR(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVRLight(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVRLight(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHSIC(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHSIC(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelMeanMatching(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelMeanMatching(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearTimeMMD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearTimeMMD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionCombMaxL2(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionCombMaxL2(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionCombOpt(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionCombOpt(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionMax(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionMax(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionMedian(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionMedian(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionOpt(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionOpt(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CQuadraticTimeMMD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CQuadraticTimeMMD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCCSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCCSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDisjointSet(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDisjointSet(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDualLibQPBMSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDualLibQPBMSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDynProg(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDynProg(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorDataSource(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorDataSource(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraph(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraph(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorType(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorType(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTableFactorType(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTableFactorType(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHMSVMModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHMSVMModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIntronList(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIntronList(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMAPInference(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMAPInference(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassSOLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassSOLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPlif(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPlif(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPlifArray(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPlifArray(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPlifMatrix(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPlifMatrix(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSegmentLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSegmentLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSequence(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSequence(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSequenceLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSequenceLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSOSVMHelper(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSOSVMHelper(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStochasticSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStochasticSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTwoStateModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTwoStateModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDomainAdaptationSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDomainAdaptationSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskClusteredLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskClusteredLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelMaskNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelMaskNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelMaskPairNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelMaskPairNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelPlifNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelPlifNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNode(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNode(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTaxonomy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTaxonomy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelTreeNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelTreeNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskL12LogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskL12LogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskLeastSquaresRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskLeastSquaresRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskLinearMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskLinearMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskROCEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskROCEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskTraceLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskTraceLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTask(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTask(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTaskGroup(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTaskGroup(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTaskTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTaskTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIClassifier(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIClassifier(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIConverter(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIConverter(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUILabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUILabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIMath(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIMath(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIPluginEstimate(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIPluginEstimate(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIPreprocessor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIPreprocessor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIStructure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIStructure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUITime(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUITime(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAveragedPerceptron(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAveragedPerceptron(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFeatureBlockLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFeatureBlockLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLClassification(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLClassification(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLMulticlass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLMulticlass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLOneClass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLOneClass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNearestCentroid(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNearestCentroid(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPerceptron(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPerceptron(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPluginEstimate(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPluginEstimate(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGNPPLib(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGNPPLib(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGNPPSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGNPPSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGPBTSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGPBTSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibLinear(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibLinear(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVMOneClass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVMOneClass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMPDSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMPDSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COnlineLibLinear(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COnlineLibLinear(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COnlineSVMSGD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COnlineSVMSGD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CQPBSVMLib(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CQPBSVMLib(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSGDQN(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSGDQN(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMLight(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMLight(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMLightOneClass(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMLightOneClass(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMLin(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMLin(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMOcas(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMOcas(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVMSGD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVMSGD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWDSVMOcas(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWDSVMOcas(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwNativeCacheReader(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwNativeCacheReader(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwNativeCacheWriter(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwNativeCacheWriter(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwAdaptiveLearner(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwAdaptiveLearner(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwNonAdaptiveLearner(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwNonAdaptiveLearner(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVowpalWabbit(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVowpalWabbit(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwEnvironment(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwEnvironment(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwParser(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwParser(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVwRegressor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVwRegressor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHierarchical(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHierarchical(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKMeans(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKMeans(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedDocConverter(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedDocConverter(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAttenuatedEuclideanDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAttenuatedEuclideanDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBrayCurtisDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBrayCurtisDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCanberraMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCanberraMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCanberraWordDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCanberraWordDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CChebyshewMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CChebyshewMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CChiSquareDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CChiSquareDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCosineDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCosineDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCustomDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCustomDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CEuclideanDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CEuclideanDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGeodesicMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGeodesicMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHammingWordDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHammingWordDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJensenMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJensenMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CManhattanMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CManhattanMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CManhattanWordDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CManhattanWordDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMinkowskiMetric(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMinkowskiMetric(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparseEuclideanDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparseEuclideanDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTanimotoDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTanimotoDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHistogram(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHistogram(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPositionalPWM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPositionalPWM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMajorityVote(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMajorityVote(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanRule(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanRule(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedMajorityVote(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedMajorityVote(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CClusteringAccuracy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CClusteringAccuracy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CClusteringMutualInformation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CClusteringMutualInformation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CContingencyTableEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CContingencyTableEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAccuracyMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAccuracyMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CErrorRateMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CErrorRateMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBALMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBALMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWRACCMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWRACCMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CF1Measure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CF1Measure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossCorrelationMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossCorrelationMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRecallMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRecallMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPrecisionMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPrecisionMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSpecificityMeasure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSpecificityMeasure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationResult(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationResult(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationMKLStorage(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationMKLStorage(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationMulticlassStorage(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationMulticlassStorage(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationPrintOutput(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationPrintOutput(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCrossValidationSplitting(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCrossValidationSplitting(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGradientCriterion(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGradientCriterion(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGradientEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGradientEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGradientResult(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGradientResult(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanAbsoluteError(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanAbsoluteError(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanSquaredError(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanSquaredError(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanSquaredLogError(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanSquaredLogError(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassAccuracy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassAccuracy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOVREvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOVREvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPRCEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPRCEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CROCEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CROCEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStratifiedCrossValidationSplitting(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStratifiedCrossValidationSplitting(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredAccuracy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredAccuracy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAlphabet(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAlphabet(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBinnedDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBinnedDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCombinedDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCombinedDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCombinedFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCombinedFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDataGenerator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDataGenerator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDummyFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDummyFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CExplicitSpecFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CExplicitSpecFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFKFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFKFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedDocDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedDocDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedWDFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedWDFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHashedWDFeaturesTransposed(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHashedWDFeaturesTransposed(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CImplicitWeightedSpecFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CImplicitWeightedSpecFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLBPPyrDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLBPPyrDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomFourierDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomFourierDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRealFileFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRealFileFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSNPFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSNPFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparsePolyFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparsePolyFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianBlobsDataGenerator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianBlobsDataGenerator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMeanShiftDataGenerator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMeanShiftDataGenerator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingHashedDocDotFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingHashedDocDotFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingVwFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingVwFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSubset(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSubset(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSubsetStack(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSubsetStack(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTOPFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTOPFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWDFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWDFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBinaryFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBinaryFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCSVFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCSVFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIOBuffer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIOBuffer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVMFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVMFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLineReader(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLineReader(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CParser(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CParser(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSerializableAsciiFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSerializableAsciiFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingAsciiFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingAsciiFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingFileFromFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingFileFromFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingVwCacheFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingVwCacheFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStreamingVwFile(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStreamingVwFile(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CANOVAKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CANOVAKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAUCKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAUCKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBesselKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBesselKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCauchyKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCauchyKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CChi2Kernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CChi2Kernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCircularKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCircularKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCombinedKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCombinedKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CConstKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CConstKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCustomKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCustomKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDiagKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDiagKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDistanceKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDistanceKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CExponentialKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CExponentialKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianARDKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianARDKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianShiftKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianShiftKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianShortRealKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianShortRealKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHistogramIntersectionKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHistogramIntersectionKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CInverseMultiQuadricKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CInverseMultiQuadricKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJensenShannonKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJensenShannonKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearARDKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearARDKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultiquadricKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultiquadricKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CAvgDiagKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CAvgDiagKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDiceKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDiceKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFirstElementKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFirstElementKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIdentityKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIdentityKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRidgeKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRidgeKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CScatterKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CScatterKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSqrtDiagKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSqrtDiagKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTanimotoKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTanimotoKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CVarianceKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CVarianceKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CZeroMeanCenterKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CZeroMeanCenterKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPowerKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPowerKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CProductKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CProductKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPyramidChi2(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPyramidChi2(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRationalQuadraticKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRationalQuadraticKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSigmoidKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSigmoidKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSphericalKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSphericalKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSplineKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSplineKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCommUlongStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCommUlongStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCommWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCommWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDistantSegmentsKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDistantSegmentsKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFixedDegreeStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFixedDegreeStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianMatchStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianMatchStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHistogramWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHistogramWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLocalAlignmentStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLocalAlignmentStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLocalityImprovedStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLocalityImprovedStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMatchWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMatchWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COligoStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COligoStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyMatchStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyMatchStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPolyMatchWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPolyMatchWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRegulatoryModulesStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRegulatoryModulesStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSalzbergWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSalzbergWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSimpleLocalityImprovedStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSimpleLocalityImprovedStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSNPStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSNPStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparseSpatialSampleStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparseSpatialSampleStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSpectrumMismatchRBFKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSpectrumMismatchRBFKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSpectrumRBFKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSpectrumRBFKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedCommWordStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedCommWordStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedDegreePositionStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedDegreePositionStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedDegreeStringKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedDegreeStringKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTensorProductPairKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTensorProductPairKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTStudentKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTStudentKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWaveKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWaveKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWaveletKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWaveletKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CWeightedDegreeRBFKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CWeightedDegreeRBFKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBinaryLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBinaryLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphObservation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphObservation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLabelsFactory(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLabelsFactory(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassMultipleOutputLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassMultipleOutputLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRegressionLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRegressionLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLatentSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLatentSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBitString(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBitString(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCircularBuffer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCircularBuffer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCompressor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCompressor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSerialComputationEngine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSerialComputationEngine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJobResult(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJobResult(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CData(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CData(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDelimiterTokenizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDelimiterTokenizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDynamicObjectArray(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDynamicObjectArray(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHash(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHash(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIndexBlock(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIndexBlock(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIndexBlockGroup(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIndexBlockGroup(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIndexBlockTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIndexBlockTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CListElement(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CListElement(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CList(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CList(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNGramTokenizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNGramTokenizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSignal(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSignal(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredData(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredData(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTime(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTime(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHingeLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHingeLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogLossMargin(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogLossMargin(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSmoothHingeLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSmoothHingeLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSquaredHingeLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSquaredHingeLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSquaredLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSquaredLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBaggingMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBaggingMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBaseMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBaseMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDistanceMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDistanceMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CZeroMean(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CZeroMean(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelStructuredOutputMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelStructuredOutputMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearStructuredOutputMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearStructuredOutputMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNativeMulticlassMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNativeMulticlassMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_COnlineLinearMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new COnlineLinearMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStructuredOutputMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStructuredOutputMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CJacobiEllipticFunctions(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CJacobiEllipticFunctions(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogDetEstimator(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogDetEstimator(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNormalSampler(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNormalSampler(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMath(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMath(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandom(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandom(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSparseInverseCovariance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSparseInverseCovariance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStatistics(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStatistics(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGridSearchModelSelection(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGridSearchModelSelection(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CModelSelectionParameters(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CModelSelectionParameters(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CParameterCombination(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CParameterCombination(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomSearchModelSelection(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomSearchModelSelection(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCAEDDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCAEDDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCDiscriminantEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCDiscriminantEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCEDDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCEDDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCForestEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCForestEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCHDDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCHDDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCLLBDecoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCLLBDecoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCOVOEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCOVOEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCOVREncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCOVREncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCRandomDenseEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCRandomDenseEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCRandomSparseEncoder(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCRandomSparseEncoder(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CECOCStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CECOCStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGaussianNaiveBayes(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGaussianNaiveBayes(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGMNPLib(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGMNPLib(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGMNPSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGMNPSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKNN(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKNN(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLaRank(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLaRank(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassLibLinear(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassLibLinear(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassLibSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassLibSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOCAS(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOCAS(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOneVsOneStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOneVsOneStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassOneVsRestStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassOneVsRestStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CThresholdRejectionStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CThresholdRejectionStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDixonQTestRejectionStrategy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDixonQTestRejectionStrategy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CScatterSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CScatterSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CShareBoost(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CShareBoost(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CBalancedConditionalProbabilityTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CBalancedConditionalProbabilityTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomConditionalProbabilityTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomConditionalProbabilityTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRelaxedTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRelaxedTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTron(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTron(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDimensionReductionPreprocessor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDimensionReductionPreprocessor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHomogeneousKernelMap(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHomogeneousKernelMap(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLogPlusOne(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLogPlusOne(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNormOne(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNormOne(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPNorm(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPNorm(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPruneVarSubMean(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPruneVarSubMean(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRandomFourierGaussPreproc(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRandomFourierGaussPreproc(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CRescaleFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CRescaleFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSortUlongString(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSortUlongString(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSortWordString(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSortWordString(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSumOne(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSumOne(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLibSVR(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLibSVR(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMKLRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMKLRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSVRLight(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSVRLight(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHSIC(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHSIC(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CKernelMeanMatching(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CKernelMeanMatching(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CLinearTimeMMD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CLinearTimeMMD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionCombMaxL2(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionCombMaxL2(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionCombOpt(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionCombOpt(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionMax(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionMax(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionMedian(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionMedian(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMMDKernelSelectionOpt(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMMDKernelSelectionOpt(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CQuadraticTimeMMD(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CQuadraticTimeMMD(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CCCSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CCCSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDisjointSet(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDisjointSet(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDualLibQPBMSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDualLibQPBMSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDynProg(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDynProg(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorDataSource(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorDataSource(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraph(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraph(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorGraphModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorGraphModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CFactorType(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CFactorType(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTableFactorType(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTableFactorType(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CHMSVMModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CHMSVMModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CIntronList(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CIntronList(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMAPInference(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMAPInference(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMulticlassSOLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMulticlassSOLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPlif(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPlif(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPlifArray(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPlifArray(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CPlifMatrix(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CPlifMatrix(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSegmentLoss(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSegmentLoss(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSequence(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSequence(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSequenceLabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSequenceLabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CSOSVMHelper(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CSOSVMHelper(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CStochasticSOSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CStochasticSOSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTwoStateModel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTwoStateModel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CDomainAdaptationSVM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CDomainAdaptationSVM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskClusteredLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskClusteredLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelMaskNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelMaskNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelMaskPairNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelMaskPairNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelPlifNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelPlifNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CNode(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CNode(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTaxonomy(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTaxonomy(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskKernelTreeNormalizer(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskKernelTreeNormalizer(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskL12LogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskL12LogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskLeastSquaresRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskLeastSquaresRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskLinearMachine(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskLinearMachine(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskROCEvaluation(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskROCEvaluation(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CMultitaskTraceLogisticRegression(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CMultitaskTraceLogisticRegression(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTask(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTask(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTaskGroup(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTaskGroup(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CTaskTree(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CTaskTree(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIClassifier(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIClassifier(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIConverter(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIConverter(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIDistance(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIDistance(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIFeatures(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIFeatures(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIHMM(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIHMM(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIKernel(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIKernel(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUILabels(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUILabels(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIMath(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIMath(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIPluginEstimate(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIPluginEstimate(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIPreprocessor(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIPreprocessor(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUIStructure(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUIStructure(): NULL; }
static SHOGUN_BASIC_CLASS CSGObject* __new_CGUITime(EPrimitiveType g) { return g == PT_NOT_GENERIC? new CGUITime(): NULL; }
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDenseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDenseSubsetFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CDenseSubsetFeatures<bool>();
		case PT_CHAR: return new CDenseSubsetFeatures<char>();
		case PT_INT8: return new CDenseSubsetFeatures<int8_t>();
		case PT_UINT8: return new CDenseSubsetFeatures<uint8_t>();
		case PT_INT16: return new CDenseSubsetFeatures<int16_t>();
		case PT_UINT16: return new CDenseSubsetFeatures<uint16_t>();
		case PT_INT32: return new CDenseSubsetFeatures<int32_t>();
		case PT_UINT32: return new CDenseSubsetFeatures<uint32_t>();
		case PT_INT64: return new CDenseSubsetFeatures<int64_t>();
		case PT_UINT64: return new CDenseSubsetFeatures<uint64_t>();
		case PT_FLOAT32: return new CDenseSubsetFeatures<float32_t>();
		case PT_FLOAT64: return new CDenseSubsetFeatures<float64_t>();
		case PT_FLOATMAX: return new CDenseSubsetFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CHashedDenseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CHashedDenseFeatures<bool>();
		case PT_CHAR: return new CHashedDenseFeatures<char>();
		case PT_INT8: return new CHashedDenseFeatures<int8_t>();
		case PT_UINT8: return new CHashedDenseFeatures<uint8_t>();
		case PT_INT16: return new CHashedDenseFeatures<int16_t>();
		case PT_UINT16: return new CHashedDenseFeatures<uint16_t>();
		case PT_INT32: return new CHashedDenseFeatures<int32_t>();
		case PT_UINT32: return new CHashedDenseFeatures<uint32_t>();
		case PT_INT64: return new CHashedDenseFeatures<int64_t>();
		case PT_UINT64: return new CHashedDenseFeatures<uint64_t>();
		case PT_FLOAT32: return new CHashedDenseFeatures<float32_t>();
		case PT_FLOAT64: return new CHashedDenseFeatures<float64_t>();
		case PT_FLOATMAX: return new CHashedDenseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CHashedSparseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CHashedSparseFeatures<bool>();
		case PT_CHAR: return new CHashedSparseFeatures<char>();
		case PT_INT8: return new CHashedSparseFeatures<int8_t>();
		case PT_UINT8: return new CHashedSparseFeatures<uint8_t>();
		case PT_INT16: return new CHashedSparseFeatures<int16_t>();
		case PT_UINT16: return new CHashedSparseFeatures<uint16_t>();
		case PT_INT32: return new CHashedSparseFeatures<int32_t>();
		case PT_UINT32: return new CHashedSparseFeatures<uint32_t>();
		case PT_INT64: return new CHashedSparseFeatures<int64_t>();
		case PT_UINT64: return new CHashedSparseFeatures<uint64_t>();
		case PT_FLOAT32: return new CHashedSparseFeatures<float32_t>();
		case PT_FLOAT64: return new CHashedSparseFeatures<float64_t>();
		case PT_FLOATMAX: return new CHashedSparseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CMatrixFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CMatrixFeatures<bool>();
		case PT_CHAR: return new CMatrixFeatures<char>();
		case PT_INT8: return new CMatrixFeatures<int8_t>();
		case PT_UINT8: return new CMatrixFeatures<uint8_t>();
		case PT_INT16: return new CMatrixFeatures<int16_t>();
		case PT_UINT16: return new CMatrixFeatures<uint16_t>();
		case PT_INT32: return new CMatrixFeatures<int32_t>();
		case PT_UINT32: return new CMatrixFeatures<uint32_t>();
		case PT_INT64: return new CMatrixFeatures<int64_t>();
		case PT_UINT64: return new CMatrixFeatures<uint64_t>();
		case PT_FLOAT32: return new CMatrixFeatures<float32_t>();
		case PT_FLOAT64: return new CMatrixFeatures<float64_t>();
		case PT_FLOATMAX: return new CMatrixFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSparseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingDenseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingHashedDenseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStreamingHashedDenseFeatures<bool>();
		case PT_CHAR: return new CStreamingHashedDenseFeatures<char>();
		case PT_INT8: return new CStreamingHashedDenseFeatures<int8_t>();
		case PT_UINT8: return new CStreamingHashedDenseFeatures<uint8_t>();
		case PT_INT16: return new CStreamingHashedDenseFeatures<int16_t>();
		case PT_UINT16: return new CStreamingHashedDenseFeatures<uint16_t>();
		case PT_INT32: return new CStreamingHashedDenseFeatures<int32_t>();
		case PT_UINT32: return new CStreamingHashedDenseFeatures<uint32_t>();
		case PT_INT64: return new CStreamingHashedDenseFeatures<int64_t>();
		case PT_UINT64: return new CStreamingHashedDenseFeatures<uint64_t>();
		case PT_FLOAT32: return new CStreamingHashedDenseFeatures<float32_t>();
		case PT_FLOAT64: return new CStreamingHashedDenseFeatures<float64_t>();
		case PT_FLOATMAX: return new CStreamingHashedDenseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingHashedSparseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStreamingHashedSparseFeatures<bool>();
		case PT_CHAR: return new CStreamingHashedSparseFeatures<char>();
		case PT_INT8: return new CStreamingHashedSparseFeatures<int8_t>();
		case PT_UINT8: return new CStreamingHashedSparseFeatures<uint8_t>();
		case PT_INT16: return new CStreamingHashedSparseFeatures<int16_t>();
		case PT_UINT16: return new CStreamingHashedSparseFeatures<uint16_t>();
		case PT_INT32: return new CStreamingHashedSparseFeatures<int32_t>();
		case PT_UINT32: return new CStreamingHashedSparseFeatures<uint32_t>();
		case PT_INT64: return new CStreamingHashedSparseFeatures<int64_t>();
		case PT_UINT64: return new CStreamingHashedSparseFeatures<uint64_t>();
		case PT_FLOAT32: return new CStreamingHashedSparseFeatures<float32_t>();
		case PT_FLOAT64: return new CStreamingHashedSparseFeatures<float64_t>();
		case PT_FLOATMAX: return new CStreamingHashedSparseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingSparseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingStringFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStringFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStringFileFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CBinaryStream(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CMemoryMappedFile(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSimpleFile(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CParseBuffer(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingFileFromDenseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingFileFromSparseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingFileFromStringFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CCache(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDynamicArray(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSet(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CTreeMachine(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CTreeMachine<bool>();
		case PT_CHAR: return new CTreeMachine<char>();
		case PT_INT8: return new CTreeMachine<int8_t>();
		case PT_UINT8: return new CTreeMachine<uint8_t>();
		case PT_INT16: return new CTreeMachine<int16_t>();
		case PT_UINT16: return new CTreeMachine<uint16_t>();
		case PT_INT32: return new CTreeMachine<int32_t>();
		case PT_UINT32: return new CTreeMachine<uint32_t>();
		case PT_INT64: return new CTreeMachine<int64_t>();
		case PT_UINT64: return new CTreeMachine<uint64_t>();
		case PT_FLOAT32: return new CTreeMachine<float32_t>();
		case PT_FLOAT64: return new CTreeMachine<float64_t>();
		case PT_FLOATMAX: return new CTreeMachine<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDecompressString(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDenseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDenseSubsetFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CDenseSubsetFeatures<bool>();
		case PT_CHAR: return new CDenseSubsetFeatures<char>();
		case PT_INT8: return new CDenseSubsetFeatures<int8_t>();
		case PT_UINT8: return new CDenseSubsetFeatures<uint8_t>();
		case PT_INT16: return new CDenseSubsetFeatures<int16_t>();
		case PT_UINT16: return new CDenseSubsetFeatures<uint16_t>();
		case PT_INT32: return new CDenseSubsetFeatures<int32_t>();
		case PT_UINT32: return new CDenseSubsetFeatures<uint32_t>();
		case PT_INT64: return new CDenseSubsetFeatures<int64_t>();
		case PT_UINT64: return new CDenseSubsetFeatures<uint64_t>();
		case PT_FLOAT32: return new CDenseSubsetFeatures<float32_t>();
		case PT_FLOAT64: return new CDenseSubsetFeatures<float64_t>();
		case PT_FLOATMAX: return new CDenseSubsetFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CHashedDenseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CHashedDenseFeatures<bool>();
		case PT_CHAR: return new CHashedDenseFeatures<char>();
		case PT_INT8: return new CHashedDenseFeatures<int8_t>();
		case PT_UINT8: return new CHashedDenseFeatures<uint8_t>();
		case PT_INT16: return new CHashedDenseFeatures<int16_t>();
		case PT_UINT16: return new CHashedDenseFeatures<uint16_t>();
		case PT_INT32: return new CHashedDenseFeatures<int32_t>();
		case PT_UINT32: return new CHashedDenseFeatures<uint32_t>();
		case PT_INT64: return new CHashedDenseFeatures<int64_t>();
		case PT_UINT64: return new CHashedDenseFeatures<uint64_t>();
		case PT_FLOAT32: return new CHashedDenseFeatures<float32_t>();
		case PT_FLOAT64: return new CHashedDenseFeatures<float64_t>();
		case PT_FLOATMAX: return new CHashedDenseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CHashedSparseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CHashedSparseFeatures<bool>();
		case PT_CHAR: return new CHashedSparseFeatures<char>();
		case PT_INT8: return new CHashedSparseFeatures<int8_t>();
		case PT_UINT8: return new CHashedSparseFeatures<uint8_t>();
		case PT_INT16: return new CHashedSparseFeatures<int16_t>();
		case PT_UINT16: return new CHashedSparseFeatures<uint16_t>();
		case PT_INT32: return new CHashedSparseFeatures<int32_t>();
		case PT_UINT32: return new CHashedSparseFeatures<uint32_t>();
		case PT_INT64: return new CHashedSparseFeatures<int64_t>();
		case PT_UINT64: return new CHashedSparseFeatures<uint64_t>();
		case PT_FLOAT32: return new CHashedSparseFeatures<float32_t>();
		case PT_FLOAT64: return new CHashedSparseFeatures<float64_t>();
		case PT_FLOATMAX: return new CHashedSparseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CMatrixFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CMatrixFeatures<bool>();
		case PT_CHAR: return new CMatrixFeatures<char>();
		case PT_INT8: return new CMatrixFeatures<int8_t>();
		case PT_UINT8: return new CMatrixFeatures<uint8_t>();
		case PT_INT16: return new CMatrixFeatures<int16_t>();
		case PT_UINT16: return new CMatrixFeatures<uint16_t>();
		case PT_INT32: return new CMatrixFeatures<int32_t>();
		case PT_UINT32: return new CMatrixFeatures<uint32_t>();
		case PT_INT64: return new CMatrixFeatures<int64_t>();
		case PT_UINT64: return new CMatrixFeatures<uint64_t>();
		case PT_FLOAT32: return new CMatrixFeatures<float32_t>();
		case PT_FLOAT64: return new CMatrixFeatures<float64_t>();
		case PT_FLOATMAX: return new CMatrixFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSparseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingDenseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingHashedDenseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStreamingHashedDenseFeatures<bool>();
		case PT_CHAR: return new CStreamingHashedDenseFeatures<char>();
		case PT_INT8: return new CStreamingHashedDenseFeatures<int8_t>();
		case PT_UINT8: return new CStreamingHashedDenseFeatures<uint8_t>();
		case PT_INT16: return new CStreamingHashedDenseFeatures<int16_t>();
		case PT_UINT16: return new CStreamingHashedDenseFeatures<uint16_t>();
		case PT_INT32: return new CStreamingHashedDenseFeatures<int32_t>();
		case PT_UINT32: return new CStreamingHashedDenseFeatures<uint32_t>();
		case PT_INT64: return new CStreamingHashedDenseFeatures<int64_t>();
		case PT_UINT64: return new CStreamingHashedDenseFeatures<uint64_t>();
		case PT_FLOAT32: return new CStreamingHashedDenseFeatures<float32_t>();
		case PT_FLOAT64: return new CStreamingHashedDenseFeatures<float64_t>();
		case PT_FLOATMAX: return new CStreamingHashedDenseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingHashedSparseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStreamingHashedSparseFeatures<bool>();
		case PT_CHAR: return new CStreamingHashedSparseFeatures<char>();
		case PT_INT8: return new CStreamingHashedSparseFeatures<int8_t>();
		case PT_UINT8: return new CStreamingHashedSparseFeatures<uint8_t>();
		case PT_INT16: return new CStreamingHashedSparseFeatures<int16_t>();
		case PT_UINT16: return new CStreamingHashedSparseFeatures<uint16_t>();
		case PT_INT32: return new CStreamingHashedSparseFeatures<int32_t>();
		case PT_UINT32: return new CStreamingHashedSparseFeatures<uint32_t>();
		case PT_INT64: return new CStreamingHashedSparseFeatures<int64_t>();
		case PT_UINT64: return new CStreamingHashedSparseFeatures<uint64_t>();
		case PT_FLOAT32: return new CStreamingHashedSparseFeatures<float32_t>();
		case PT_FLOAT64: return new CStreamingHashedSparseFeatures<float64_t>();
		case PT_FLOATMAX: return new CStreamingHashedSparseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingSparseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingStringFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStringFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStringFileFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CBinaryStream(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CMemoryMappedFile(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSimpleFile(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CParseBuffer(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingFileFromDenseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingFileFromSparseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingFileFromStringFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CCache(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDynamicArray(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSet(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CTreeMachine(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CTreeMachine<bool>();
		case PT_CHAR: return new CTreeMachine<char>();
		case PT_INT8: return new CTreeMachine<int8_t>();
		case PT_UINT8: return new CTreeMachine<uint8_t>();
		case PT_INT16: return new CTreeMachine<int16_t>();
		case PT_UINT16: return new CTreeMachine<uint16_t>();
		case PT_INT32: return new CTreeMachine<int32_t>();
		case PT_UINT32: return new CTreeMachine<uint32_t>();
		case PT_INT64: return new CTreeMachine<int64_t>();
		case PT_UINT64: return new CTreeMachine<uint64_t>();
		case PT_FLOAT32: return new CTreeMachine<float32_t>();
		case PT_FLOAT64: return new CTreeMachine<float64_t>();
		case PT_FLOATMAX: return new CTreeMachine<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDecompressString(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDenseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDenseSubsetFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CDenseSubsetFeatures<bool>();
		case PT_CHAR: return new CDenseSubsetFeatures<char>();
		case PT_INT8: return new CDenseSubsetFeatures<int8_t>();
		case PT_UINT8: return new CDenseSubsetFeatures<uint8_t>();
		case PT_INT16: return new CDenseSubsetFeatures<int16_t>();
		case PT_UINT16: return new CDenseSubsetFeatures<uint16_t>();
		case PT_INT32: return new CDenseSubsetFeatures<int32_t>();
		case PT_UINT32: return new CDenseSubsetFeatures<uint32_t>();
		case PT_INT64: return new CDenseSubsetFeatures<int64_t>();
		case PT_UINT64: return new CDenseSubsetFeatures<uint64_t>();
		case PT_FLOAT32: return new CDenseSubsetFeatures<float32_t>();
		case PT_FLOAT64: return new CDenseSubsetFeatures<float64_t>();
		case PT_FLOATMAX: return new CDenseSubsetFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CHashedDenseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CHashedDenseFeatures<bool>();
		case PT_CHAR: return new CHashedDenseFeatures<char>();
		case PT_INT8: return new CHashedDenseFeatures<int8_t>();
		case PT_UINT8: return new CHashedDenseFeatures<uint8_t>();
		case PT_INT16: return new CHashedDenseFeatures<int16_t>();
		case PT_UINT16: return new CHashedDenseFeatures<uint16_t>();
		case PT_INT32: return new CHashedDenseFeatures<int32_t>();
		case PT_UINT32: return new CHashedDenseFeatures<uint32_t>();
		case PT_INT64: return new CHashedDenseFeatures<int64_t>();
		case PT_UINT64: return new CHashedDenseFeatures<uint64_t>();
		case PT_FLOAT32: return new CHashedDenseFeatures<float32_t>();
		case PT_FLOAT64: return new CHashedDenseFeatures<float64_t>();
		case PT_FLOATMAX: return new CHashedDenseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CHashedSparseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CHashedSparseFeatures<bool>();
		case PT_CHAR: return new CHashedSparseFeatures<char>();
		case PT_INT8: return new CHashedSparseFeatures<int8_t>();
		case PT_UINT8: return new CHashedSparseFeatures<uint8_t>();
		case PT_INT16: return new CHashedSparseFeatures<int16_t>();
		case PT_UINT16: return new CHashedSparseFeatures<uint16_t>();
		case PT_INT32: return new CHashedSparseFeatures<int32_t>();
		case PT_UINT32: return new CHashedSparseFeatures<uint32_t>();
		case PT_INT64: return new CHashedSparseFeatures<int64_t>();
		case PT_UINT64: return new CHashedSparseFeatures<uint64_t>();
		case PT_FLOAT32: return new CHashedSparseFeatures<float32_t>();
		case PT_FLOAT64: return new CHashedSparseFeatures<float64_t>();
		case PT_FLOATMAX: return new CHashedSparseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CMatrixFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CMatrixFeatures<bool>();
		case PT_CHAR: return new CMatrixFeatures<char>();
		case PT_INT8: return new CMatrixFeatures<int8_t>();
		case PT_UINT8: return new CMatrixFeatures<uint8_t>();
		case PT_INT16: return new CMatrixFeatures<int16_t>();
		case PT_UINT16: return new CMatrixFeatures<uint16_t>();
		case PT_INT32: return new CMatrixFeatures<int32_t>();
		case PT_UINT32: return new CMatrixFeatures<uint32_t>();
		case PT_INT64: return new CMatrixFeatures<int64_t>();
		case PT_UINT64: return new CMatrixFeatures<uint64_t>();
		case PT_FLOAT32: return new CMatrixFeatures<float32_t>();
		case PT_FLOAT64: return new CMatrixFeatures<float64_t>();
		case PT_FLOATMAX: return new CMatrixFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSparseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingDenseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingHashedDenseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStreamingHashedDenseFeatures<bool>();
		case PT_CHAR: return new CStreamingHashedDenseFeatures<char>();
		case PT_INT8: return new CStreamingHashedDenseFeatures<int8_t>();
		case PT_UINT8: return new CStreamingHashedDenseFeatures<uint8_t>();
		case PT_INT16: return new CStreamingHashedDenseFeatures<int16_t>();
		case PT_UINT16: return new CStreamingHashedDenseFeatures<uint16_t>();
		case PT_INT32: return new CStreamingHashedDenseFeatures<int32_t>();
		case PT_UINT32: return new CStreamingHashedDenseFeatures<uint32_t>();
		case PT_INT64: return new CStreamingHashedDenseFeatures<int64_t>();
		case PT_UINT64: return new CStreamingHashedDenseFeatures<uint64_t>();
		case PT_FLOAT32: return new CStreamingHashedDenseFeatures<float32_t>();
		case PT_FLOAT64: return new CStreamingHashedDenseFeatures<float64_t>();
		case PT_FLOATMAX: return new CStreamingHashedDenseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingHashedSparseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStreamingHashedSparseFeatures<bool>();
		case PT_CHAR: return new CStreamingHashedSparseFeatures<char>();
		case PT_INT8: return new CStreamingHashedSparseFeatures<int8_t>();
		case PT_UINT8: return new CStreamingHashedSparseFeatures<uint8_t>();
		case PT_INT16: return new CStreamingHashedSparseFeatures<int16_t>();
		case PT_UINT16: return new CStreamingHashedSparseFeatures<uint16_t>();
		case PT_INT32: return new CStreamingHashedSparseFeatures<int32_t>();
		case PT_UINT32: return new CStreamingHashedSparseFeatures<uint32_t>();
		case PT_INT64: return new CStreamingHashedSparseFeatures<int64_t>();
		case PT_UINT64: return new CStreamingHashedSparseFeatures<uint64_t>();
		case PT_FLOAT32: return new CStreamingHashedSparseFeatures<float32_t>();
		case PT_FLOAT64: return new CStreamingHashedSparseFeatures<float64_t>();
		case PT_FLOATMAX: return new CStreamingHashedSparseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingSparseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingStringFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStringFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStringFileFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CBinaryStream(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CMemoryMappedFile(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSimpleFile(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CParseBuffer(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingFileFromDenseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingFileFromSparseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingFileFromStringFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CCache(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDynamicArray(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSet(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CTreeMachine(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CTreeMachine<bool>();
		case PT_CHAR: return new CTreeMachine<char>();
		case PT_INT8: return new CTreeMachine<int8_t>();
		case PT_UINT8: return new CTreeMachine<uint8_t>();
		case PT_INT16: return new CTreeMachine<int16_t>();
		case PT_UINT16: return new CTreeMachine<uint16_t>();
		case PT_INT32: return new CTreeMachine<int32_t>();
		case PT_UINT32: return new CTreeMachine<uint32_t>();
		case PT_INT64: return new CTreeMachine<int64_t>();
		case PT_UINT64: return new CTreeMachine<uint64_t>();
		case PT_FLOAT32: return new CTreeMachine<float32_t>();
		case PT_FLOAT64: return new CTreeMachine<float64_t>();
		case PT_FLOATMAX: return new CTreeMachine<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDecompressString(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDenseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDenseSubsetFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CDenseSubsetFeatures<bool>();
		case PT_CHAR: return new CDenseSubsetFeatures<char>();
		case PT_INT8: return new CDenseSubsetFeatures<int8_t>();
		case PT_UINT8: return new CDenseSubsetFeatures<uint8_t>();
		case PT_INT16: return new CDenseSubsetFeatures<int16_t>();
		case PT_UINT16: return new CDenseSubsetFeatures<uint16_t>();
		case PT_INT32: return new CDenseSubsetFeatures<int32_t>();
		case PT_UINT32: return new CDenseSubsetFeatures<uint32_t>();
		case PT_INT64: return new CDenseSubsetFeatures<int64_t>();
		case PT_UINT64: return new CDenseSubsetFeatures<uint64_t>();
		case PT_FLOAT32: return new CDenseSubsetFeatures<float32_t>();
		case PT_FLOAT64: return new CDenseSubsetFeatures<float64_t>();
		case PT_FLOATMAX: return new CDenseSubsetFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CHashedDenseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CHashedDenseFeatures<bool>();
		case PT_CHAR: return new CHashedDenseFeatures<char>();
		case PT_INT8: return new CHashedDenseFeatures<int8_t>();
		case PT_UINT8: return new CHashedDenseFeatures<uint8_t>();
		case PT_INT16: return new CHashedDenseFeatures<int16_t>();
		case PT_UINT16: return new CHashedDenseFeatures<uint16_t>();
		case PT_INT32: return new CHashedDenseFeatures<int32_t>();
		case PT_UINT32: return new CHashedDenseFeatures<uint32_t>();
		case PT_INT64: return new CHashedDenseFeatures<int64_t>();
		case PT_UINT64: return new CHashedDenseFeatures<uint64_t>();
		case PT_FLOAT32: return new CHashedDenseFeatures<float32_t>();
		case PT_FLOAT64: return new CHashedDenseFeatures<float64_t>();
		case PT_FLOATMAX: return new CHashedDenseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CHashedSparseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CHashedSparseFeatures<bool>();
		case PT_CHAR: return new CHashedSparseFeatures<char>();
		case PT_INT8: return new CHashedSparseFeatures<int8_t>();
		case PT_UINT8: return new CHashedSparseFeatures<uint8_t>();
		case PT_INT16: return new CHashedSparseFeatures<int16_t>();
		case PT_UINT16: return new CHashedSparseFeatures<uint16_t>();
		case PT_INT32: return new CHashedSparseFeatures<int32_t>();
		case PT_UINT32: return new CHashedSparseFeatures<uint32_t>();
		case PT_INT64: return new CHashedSparseFeatures<int64_t>();
		case PT_UINT64: return new CHashedSparseFeatures<uint64_t>();
		case PT_FLOAT32: return new CHashedSparseFeatures<float32_t>();
		case PT_FLOAT64: return new CHashedSparseFeatures<float64_t>();
		case PT_FLOATMAX: return new CHashedSparseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CMatrixFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CMatrixFeatures<bool>();
		case PT_CHAR: return new CMatrixFeatures<char>();
		case PT_INT8: return new CMatrixFeatures<int8_t>();
		case PT_UINT8: return new CMatrixFeatures<uint8_t>();
		case PT_INT16: return new CMatrixFeatures<int16_t>();
		case PT_UINT16: return new CMatrixFeatures<uint16_t>();
		case PT_INT32: return new CMatrixFeatures<int32_t>();
		case PT_UINT32: return new CMatrixFeatures<uint32_t>();
		case PT_INT64: return new CMatrixFeatures<int64_t>();
		case PT_UINT64: return new CMatrixFeatures<uint64_t>();
		case PT_FLOAT32: return new CMatrixFeatures<float32_t>();
		case PT_FLOAT64: return new CMatrixFeatures<float64_t>();
		case PT_FLOATMAX: return new CMatrixFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSparseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingDenseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingHashedDenseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStreamingHashedDenseFeatures<bool>();
		case PT_CHAR: return new CStreamingHashedDenseFeatures<char>();
		case PT_INT8: return new CStreamingHashedDenseFeatures<int8_t>();
		case PT_UINT8: return new CStreamingHashedDenseFeatures<uint8_t>();
		case PT_INT16: return new CStreamingHashedDenseFeatures<int16_t>();
		case PT_UINT16: return new CStreamingHashedDenseFeatures<uint16_t>();
		case PT_INT32: return new CStreamingHashedDenseFeatures<int32_t>();
		case PT_UINT32: return new CStreamingHashedDenseFeatures<uint32_t>();
		case PT_INT64: return new CStreamingHashedDenseFeatures<int64_t>();
		case PT_UINT64: return new CStreamingHashedDenseFeatures<uint64_t>();
		case PT_FLOAT32: return new CStreamingHashedDenseFeatures<float32_t>();
		case PT_FLOAT64: return new CStreamingHashedDenseFeatures<float64_t>();
		case PT_FLOATMAX: return new CStreamingHashedDenseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingHashedSparseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStreamingHashedSparseFeatures<bool>();
		case PT_CHAR: return new CStreamingHashedSparseFeatures<char>();
		case PT_INT8: return new CStreamingHashedSparseFeatures<int8_t>();
		case PT_UINT8: return new CStreamingHashedSparseFeatures<uint8_t>();
		case PT_INT16: return new CStreamingHashedSparseFeatures<int16_t>();
		case PT_UINT16: return new CStreamingHashedSparseFeatures<uint16_t>();
		case PT_INT32: return new CStreamingHashedSparseFeatures<int32_t>();
		case PT_UINT32: return new CStreamingHashedSparseFeatures<uint32_t>();
		case PT_INT64: return new CStreamingHashedSparseFeatures<int64_t>();
		case PT_UINT64: return new CStreamingHashedSparseFeatures<uint64_t>();
		case PT_FLOAT32: return new CStreamingHashedSparseFeatures<float32_t>();
		case PT_FLOAT64: return new CStreamingHashedSparseFeatures<float64_t>();
		case PT_FLOATMAX: return new CStreamingHashedSparseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingSparseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingStringFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStringFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStringFileFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CBinaryStream(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CMemoryMappedFile(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSimpleFile(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CParseBuffer(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingFileFromDenseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingFileFromSparseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingFileFromStringFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CCache(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDynamicArray(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSet(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CTreeMachine(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CTreeMachine<bool>();
		case PT_CHAR: return new CTreeMachine<char>();
		case PT_INT8: return new CTreeMachine<int8_t>();
		case PT_UINT8: return new CTreeMachine<uint8_t>();
		case PT_INT16: return new CTreeMachine<int16_t>();
		case PT_UINT16: return new CTreeMachine<uint16_t>();
		case PT_INT32: return new CTreeMachine<int32_t>();
		case PT_UINT32: return new CTreeMachine<uint32_t>();
		case PT_INT64: return new CTreeMachine<int64_t>();
		case PT_UINT64: return new CTreeMachine<uint64_t>();
		case PT_FLOAT32: return new CTreeMachine<float32_t>();
		case PT_FLOAT64: return new CTreeMachine<float64_t>();
		case PT_FLOATMAX: return new CTreeMachine<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDecompressString(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDenseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDenseSubsetFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CDenseSubsetFeatures<bool>();
		case PT_CHAR: return new CDenseSubsetFeatures<char>();
		case PT_INT8: return new CDenseSubsetFeatures<int8_t>();
		case PT_UINT8: return new CDenseSubsetFeatures<uint8_t>();
		case PT_INT16: return new CDenseSubsetFeatures<int16_t>();
		case PT_UINT16: return new CDenseSubsetFeatures<uint16_t>();
		case PT_INT32: return new CDenseSubsetFeatures<int32_t>();
		case PT_UINT32: return new CDenseSubsetFeatures<uint32_t>();
		case PT_INT64: return new CDenseSubsetFeatures<int64_t>();
		case PT_UINT64: return new CDenseSubsetFeatures<uint64_t>();
		case PT_FLOAT32: return new CDenseSubsetFeatures<float32_t>();
		case PT_FLOAT64: return new CDenseSubsetFeatures<float64_t>();
		case PT_FLOATMAX: return new CDenseSubsetFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CHashedDenseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CHashedDenseFeatures<bool>();
		case PT_CHAR: return new CHashedDenseFeatures<char>();
		case PT_INT8: return new CHashedDenseFeatures<int8_t>();
		case PT_UINT8: return new CHashedDenseFeatures<uint8_t>();
		case PT_INT16: return new CHashedDenseFeatures<int16_t>();
		case PT_UINT16: return new CHashedDenseFeatures<uint16_t>();
		case PT_INT32: return new CHashedDenseFeatures<int32_t>();
		case PT_UINT32: return new CHashedDenseFeatures<uint32_t>();
		case PT_INT64: return new CHashedDenseFeatures<int64_t>();
		case PT_UINT64: return new CHashedDenseFeatures<uint64_t>();
		case PT_FLOAT32: return new CHashedDenseFeatures<float32_t>();
		case PT_FLOAT64: return new CHashedDenseFeatures<float64_t>();
		case PT_FLOATMAX: return new CHashedDenseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CHashedSparseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CHashedSparseFeatures<bool>();
		case PT_CHAR: return new CHashedSparseFeatures<char>();
		case PT_INT8: return new CHashedSparseFeatures<int8_t>();
		case PT_UINT8: return new CHashedSparseFeatures<uint8_t>();
		case PT_INT16: return new CHashedSparseFeatures<int16_t>();
		case PT_UINT16: return new CHashedSparseFeatures<uint16_t>();
		case PT_INT32: return new CHashedSparseFeatures<int32_t>();
		case PT_UINT32: return new CHashedSparseFeatures<uint32_t>();
		case PT_INT64: return new CHashedSparseFeatures<int64_t>();
		case PT_UINT64: return new CHashedSparseFeatures<uint64_t>();
		case PT_FLOAT32: return new CHashedSparseFeatures<float32_t>();
		case PT_FLOAT64: return new CHashedSparseFeatures<float64_t>();
		case PT_FLOATMAX: return new CHashedSparseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CMatrixFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CMatrixFeatures<bool>();
		case PT_CHAR: return new CMatrixFeatures<char>();
		case PT_INT8: return new CMatrixFeatures<int8_t>();
		case PT_UINT8: return new CMatrixFeatures<uint8_t>();
		case PT_INT16: return new CMatrixFeatures<int16_t>();
		case PT_UINT16: return new CMatrixFeatures<uint16_t>();
		case PT_INT32: return new CMatrixFeatures<int32_t>();
		case PT_UINT32: return new CMatrixFeatures<uint32_t>();
		case PT_INT64: return new CMatrixFeatures<int64_t>();
		case PT_UINT64: return new CMatrixFeatures<uint64_t>();
		case PT_FLOAT32: return new CMatrixFeatures<float32_t>();
		case PT_FLOAT64: return new CMatrixFeatures<float64_t>();
		case PT_FLOATMAX: return new CMatrixFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSparseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingDenseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingHashedDenseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStreamingHashedDenseFeatures<bool>();
		case PT_CHAR: return new CStreamingHashedDenseFeatures<char>();
		case PT_INT8: return new CStreamingHashedDenseFeatures<int8_t>();
		case PT_UINT8: return new CStreamingHashedDenseFeatures<uint8_t>();
		case PT_INT16: return new CStreamingHashedDenseFeatures<int16_t>();
		case PT_UINT16: return new CStreamingHashedDenseFeatures<uint16_t>();
		case PT_INT32: return new CStreamingHashedDenseFeatures<int32_t>();
		case PT_UINT32: return new CStreamingHashedDenseFeatures<uint32_t>();
		case PT_INT64: return new CStreamingHashedDenseFeatures<int64_t>();
		case PT_UINT64: return new CStreamingHashedDenseFeatures<uint64_t>();
		case PT_FLOAT32: return new CStreamingHashedDenseFeatures<float32_t>();
		case PT_FLOAT64: return new CStreamingHashedDenseFeatures<float64_t>();
		case PT_FLOATMAX: return new CStreamingHashedDenseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingHashedSparseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStreamingHashedSparseFeatures<bool>();
		case PT_CHAR: return new CStreamingHashedSparseFeatures<char>();
		case PT_INT8: return new CStreamingHashedSparseFeatures<int8_t>();
		case PT_UINT8: return new CStreamingHashedSparseFeatures<uint8_t>();
		case PT_INT16: return new CStreamingHashedSparseFeatures<int16_t>();
		case PT_UINT16: return new CStreamingHashedSparseFeatures<uint16_t>();
		case PT_INT32: return new CStreamingHashedSparseFeatures<int32_t>();
		case PT_UINT32: return new CStreamingHashedSparseFeatures<uint32_t>();
		case PT_INT64: return new CStreamingHashedSparseFeatures<int64_t>();
		case PT_UINT64: return new CStreamingHashedSparseFeatures<uint64_t>();
		case PT_FLOAT32: return new CStreamingHashedSparseFeatures<float32_t>();
		case PT_FLOAT64: return new CStreamingHashedSparseFeatures<float64_t>();
		case PT_FLOATMAX: return new CStreamingHashedSparseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingSparseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingStringFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStringFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStringFileFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CBinaryStream(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CMemoryMappedFile(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSimpleFile(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CParseBuffer(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingFileFromDenseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingFileFromSparseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingFileFromStringFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CCache(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDynamicArray(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSet(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CTreeMachine(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CTreeMachine<bool>();
		case PT_CHAR: return new CTreeMachine<char>();
		case PT_INT8: return new CTreeMachine<int8_t>();
		case PT_UINT8: return new CTreeMachine<uint8_t>();
		case PT_INT16: return new CTreeMachine<int16_t>();
		case PT_UINT16: return new CTreeMachine<uint16_t>();
		case PT_INT32: return new CTreeMachine<int32_t>();
		case PT_UINT32: return new CTreeMachine<uint32_t>();
		case PT_INT64: return new CTreeMachine<int64_t>();
		case PT_UINT64: return new CTreeMachine<uint64_t>();
		case PT_FLOAT32: return new CTreeMachine<float32_t>();
		case PT_FLOAT64: return new CTreeMachine<float64_t>();
		case PT_FLOATMAX: return new CTreeMachine<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDecompressString(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDenseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDenseSubsetFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CDenseSubsetFeatures<bool>();
		case PT_CHAR: return new CDenseSubsetFeatures<char>();
		case PT_INT8: return new CDenseSubsetFeatures<int8_t>();
		case PT_UINT8: return new CDenseSubsetFeatures<uint8_t>();
		case PT_INT16: return new CDenseSubsetFeatures<int16_t>();
		case PT_UINT16: return new CDenseSubsetFeatures<uint16_t>();
		case PT_INT32: return new CDenseSubsetFeatures<int32_t>();
		case PT_UINT32: return new CDenseSubsetFeatures<uint32_t>();
		case PT_INT64: return new CDenseSubsetFeatures<int64_t>();
		case PT_UINT64: return new CDenseSubsetFeatures<uint64_t>();
		case PT_FLOAT32: return new CDenseSubsetFeatures<float32_t>();
		case PT_FLOAT64: return new CDenseSubsetFeatures<float64_t>();
		case PT_FLOATMAX: return new CDenseSubsetFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CHashedDenseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CHashedDenseFeatures<bool>();
		case PT_CHAR: return new CHashedDenseFeatures<char>();
		case PT_INT8: return new CHashedDenseFeatures<int8_t>();
		case PT_UINT8: return new CHashedDenseFeatures<uint8_t>();
		case PT_INT16: return new CHashedDenseFeatures<int16_t>();
		case PT_UINT16: return new CHashedDenseFeatures<uint16_t>();
		case PT_INT32: return new CHashedDenseFeatures<int32_t>();
		case PT_UINT32: return new CHashedDenseFeatures<uint32_t>();
		case PT_INT64: return new CHashedDenseFeatures<int64_t>();
		case PT_UINT64: return new CHashedDenseFeatures<uint64_t>();
		case PT_FLOAT32: return new CHashedDenseFeatures<float32_t>();
		case PT_FLOAT64: return new CHashedDenseFeatures<float64_t>();
		case PT_FLOATMAX: return new CHashedDenseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CHashedSparseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CHashedSparseFeatures<bool>();
		case PT_CHAR: return new CHashedSparseFeatures<char>();
		case PT_INT8: return new CHashedSparseFeatures<int8_t>();
		case PT_UINT8: return new CHashedSparseFeatures<uint8_t>();
		case PT_INT16: return new CHashedSparseFeatures<int16_t>();
		case PT_UINT16: return new CHashedSparseFeatures<uint16_t>();
		case PT_INT32: return new CHashedSparseFeatures<int32_t>();
		case PT_UINT32: return new CHashedSparseFeatures<uint32_t>();
		case PT_INT64: return new CHashedSparseFeatures<int64_t>();
		case PT_UINT64: return new CHashedSparseFeatures<uint64_t>();
		case PT_FLOAT32: return new CHashedSparseFeatures<float32_t>();
		case PT_FLOAT64: return new CHashedSparseFeatures<float64_t>();
		case PT_FLOATMAX: return new CHashedSparseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CMatrixFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CMatrixFeatures<bool>();
		case PT_CHAR: return new CMatrixFeatures<char>();
		case PT_INT8: return new CMatrixFeatures<int8_t>();
		case PT_UINT8: return new CMatrixFeatures<uint8_t>();
		case PT_INT16: return new CMatrixFeatures<int16_t>();
		case PT_UINT16: return new CMatrixFeatures<uint16_t>();
		case PT_INT32: return new CMatrixFeatures<int32_t>();
		case PT_UINT32: return new CMatrixFeatures<uint32_t>();
		case PT_INT64: return new CMatrixFeatures<int64_t>();
		case PT_UINT64: return new CMatrixFeatures<uint64_t>();
		case PT_FLOAT32: return new CMatrixFeatures<float32_t>();
		case PT_FLOAT64: return new CMatrixFeatures<float64_t>();
		case PT_FLOATMAX: return new CMatrixFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSparseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingDenseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingHashedDenseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStreamingHashedDenseFeatures<bool>();
		case PT_CHAR: return new CStreamingHashedDenseFeatures<char>();
		case PT_INT8: return new CStreamingHashedDenseFeatures<int8_t>();
		case PT_UINT8: return new CStreamingHashedDenseFeatures<uint8_t>();
		case PT_INT16: return new CStreamingHashedDenseFeatures<int16_t>();
		case PT_UINT16: return new CStreamingHashedDenseFeatures<uint16_t>();
		case PT_INT32: return new CStreamingHashedDenseFeatures<int32_t>();
		case PT_UINT32: return new CStreamingHashedDenseFeatures<uint32_t>();
		case PT_INT64: return new CStreamingHashedDenseFeatures<int64_t>();
		case PT_UINT64: return new CStreamingHashedDenseFeatures<uint64_t>();
		case PT_FLOAT32: return new CStreamingHashedDenseFeatures<float32_t>();
		case PT_FLOAT64: return new CStreamingHashedDenseFeatures<float64_t>();
		case PT_FLOATMAX: return new CStreamingHashedDenseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingHashedSparseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStreamingHashedSparseFeatures<bool>();
		case PT_CHAR: return new CStreamingHashedSparseFeatures<char>();
		case PT_INT8: return new CStreamingHashedSparseFeatures<int8_t>();
		case PT_UINT8: return new CStreamingHashedSparseFeatures<uint8_t>();
		case PT_INT16: return new CStreamingHashedSparseFeatures<int16_t>();
		case PT_UINT16: return new CStreamingHashedSparseFeatures<uint16_t>();
		case PT_INT32: return new CStreamingHashedSparseFeatures<int32_t>();
		case PT_UINT32: return new CStreamingHashedSparseFeatures<uint32_t>();
		case PT_INT64: return new CStreamingHashedSparseFeatures<int64_t>();
		case PT_UINT64: return new CStreamingHashedSparseFeatures<uint64_t>();
		case PT_FLOAT32: return new CStreamingHashedSparseFeatures<float32_t>();
		case PT_FLOAT64: return new CStreamingHashedSparseFeatures<float64_t>();
		case PT_FLOATMAX: return new CStreamingHashedSparseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingSparseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingStringFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStringFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStringFileFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CBinaryStream(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CMemoryMappedFile(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSimpleFile(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CParseBuffer(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingFileFromDenseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingFileFromSparseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingFileFromStringFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CCache(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDynamicArray(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSet(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CTreeMachine(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CTreeMachine<bool>();
		case PT_CHAR: return new CTreeMachine<char>();
		case PT_INT8: return new CTreeMachine<int8_t>();
		case PT_UINT8: return new CTreeMachine<uint8_t>();
		case PT_INT16: return new CTreeMachine<int16_t>();
		case PT_UINT16: return new CTreeMachine<uint16_t>();
		case PT_INT32: return new CTreeMachine<int32_t>();
		case PT_UINT32: return new CTreeMachine<uint32_t>();
		case PT_INT64: return new CTreeMachine<int64_t>();
		case PT_UINT64: return new CTreeMachine<uint64_t>();
		case PT_FLOAT32: return new CTreeMachine<float32_t>();
		case PT_FLOAT64: return new CTreeMachine<float64_t>();
		case PT_FLOATMAX: return new CTreeMachine<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDecompressString(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDenseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDenseSubsetFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CDenseSubsetFeatures<bool>();
		case PT_CHAR: return new CDenseSubsetFeatures<char>();
		case PT_INT8: return new CDenseSubsetFeatures<int8_t>();
		case PT_UINT8: return new CDenseSubsetFeatures<uint8_t>();
		case PT_INT16: return new CDenseSubsetFeatures<int16_t>();
		case PT_UINT16: return new CDenseSubsetFeatures<uint16_t>();
		case PT_INT32: return new CDenseSubsetFeatures<int32_t>();
		case PT_UINT32: return new CDenseSubsetFeatures<uint32_t>();
		case PT_INT64: return new CDenseSubsetFeatures<int64_t>();
		case PT_UINT64: return new CDenseSubsetFeatures<uint64_t>();
		case PT_FLOAT32: return new CDenseSubsetFeatures<float32_t>();
		case PT_FLOAT64: return new CDenseSubsetFeatures<float64_t>();
		case PT_FLOATMAX: return new CDenseSubsetFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CHashedDenseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CHashedDenseFeatures<bool>();
		case PT_CHAR: return new CHashedDenseFeatures<char>();
		case PT_INT8: return new CHashedDenseFeatures<int8_t>();
		case PT_UINT8: return new CHashedDenseFeatures<uint8_t>();
		case PT_INT16: return new CHashedDenseFeatures<int16_t>();
		case PT_UINT16: return new CHashedDenseFeatures<uint16_t>();
		case PT_INT32: return new CHashedDenseFeatures<int32_t>();
		case PT_UINT32: return new CHashedDenseFeatures<uint32_t>();
		case PT_INT64: return new CHashedDenseFeatures<int64_t>();
		case PT_UINT64: return new CHashedDenseFeatures<uint64_t>();
		case PT_FLOAT32: return new CHashedDenseFeatures<float32_t>();
		case PT_FLOAT64: return new CHashedDenseFeatures<float64_t>();
		case PT_FLOATMAX: return new CHashedDenseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CHashedSparseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CHashedSparseFeatures<bool>();
		case PT_CHAR: return new CHashedSparseFeatures<char>();
		case PT_INT8: return new CHashedSparseFeatures<int8_t>();
		case PT_UINT8: return new CHashedSparseFeatures<uint8_t>();
		case PT_INT16: return new CHashedSparseFeatures<int16_t>();
		case PT_UINT16: return new CHashedSparseFeatures<uint16_t>();
		case PT_INT32: return new CHashedSparseFeatures<int32_t>();
		case PT_UINT32: return new CHashedSparseFeatures<uint32_t>();
		case PT_INT64: return new CHashedSparseFeatures<int64_t>();
		case PT_UINT64: return new CHashedSparseFeatures<uint64_t>();
		case PT_FLOAT32: return new CHashedSparseFeatures<float32_t>();
		case PT_FLOAT64: return new CHashedSparseFeatures<float64_t>();
		case PT_FLOATMAX: return new CHashedSparseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CMatrixFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CMatrixFeatures<bool>();
		case PT_CHAR: return new CMatrixFeatures<char>();
		case PT_INT8: return new CMatrixFeatures<int8_t>();
		case PT_UINT8: return new CMatrixFeatures<uint8_t>();
		case PT_INT16: return new CMatrixFeatures<int16_t>();
		case PT_UINT16: return new CMatrixFeatures<uint16_t>();
		case PT_INT32: return new CMatrixFeatures<int32_t>();
		case PT_UINT32: return new CMatrixFeatures<uint32_t>();
		case PT_INT64: return new CMatrixFeatures<int64_t>();
		case PT_UINT64: return new CMatrixFeatures<uint64_t>();
		case PT_FLOAT32: return new CMatrixFeatures<float32_t>();
		case PT_FLOAT64: return new CMatrixFeatures<float64_t>();
		case PT_FLOATMAX: return new CMatrixFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSparseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingDenseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingHashedDenseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStreamingHashedDenseFeatures<bool>();
		case PT_CHAR: return new CStreamingHashedDenseFeatures<char>();
		case PT_INT8: return new CStreamingHashedDenseFeatures<int8_t>();
		case PT_UINT8: return new CStreamingHashedDenseFeatures<uint8_t>();
		case PT_INT16: return new CStreamingHashedDenseFeatures<int16_t>();
		case PT_UINT16: return new CStreamingHashedDenseFeatures<uint16_t>();
		case PT_INT32: return new CStreamingHashedDenseFeatures<int32_t>();
		case PT_UINT32: return new CStreamingHashedDenseFeatures<uint32_t>();
		case PT_INT64: return new CStreamingHashedDenseFeatures<int64_t>();
		case PT_UINT64: return new CStreamingHashedDenseFeatures<uint64_t>();
		case PT_FLOAT32: return new CStreamingHashedDenseFeatures<float32_t>();
		case PT_FLOAT64: return new CStreamingHashedDenseFeatures<float64_t>();
		case PT_FLOATMAX: return new CStreamingHashedDenseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingHashedSparseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStreamingHashedSparseFeatures<bool>();
		case PT_CHAR: return new CStreamingHashedSparseFeatures<char>();
		case PT_INT8: return new CStreamingHashedSparseFeatures<int8_t>();
		case PT_UINT8: return new CStreamingHashedSparseFeatures<uint8_t>();
		case PT_INT16: return new CStreamingHashedSparseFeatures<int16_t>();
		case PT_UINT16: return new CStreamingHashedSparseFeatures<uint16_t>();
		case PT_INT32: return new CStreamingHashedSparseFeatures<int32_t>();
		case PT_UINT32: return new CStreamingHashedSparseFeatures<uint32_t>();
		case PT_INT64: return new CStreamingHashedSparseFeatures<int64_t>();
		case PT_UINT64: return new CStreamingHashedSparseFeatures<uint64_t>();
		case PT_FLOAT32: return new CStreamingHashedSparseFeatures<float32_t>();
		case PT_FLOAT64: return new CStreamingHashedSparseFeatures<float64_t>();
		case PT_FLOATMAX: return new CStreamingHashedSparseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingSparseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingStringFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStringFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStringFileFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CBinaryStream(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CMemoryMappedFile(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSimpleFile(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CParseBuffer(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingFileFromDenseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingFileFromSparseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingFileFromStringFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CCache(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDynamicArray(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSet(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CTreeMachine(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CTreeMachine<bool>();
		case PT_CHAR: return new CTreeMachine<char>();
		case PT_INT8: return new CTreeMachine<int8_t>();
		case PT_UINT8: return new CTreeMachine<uint8_t>();
		case PT_INT16: return new CTreeMachine<int16_t>();
		case PT_UINT16: return new CTreeMachine<uint16_t>();
		case PT_INT32: return new CTreeMachine<int32_t>();
		case PT_UINT32: return new CTreeMachine<uint32_t>();
		case PT_INT64: return new CTreeMachine<int64_t>();
		case PT_UINT64: return new CTreeMachine<uint64_t>();
		case PT_FLOAT32: return new CTreeMachine<float32_t>();
		case PT_FLOAT64: return new CTreeMachine<float64_t>();
		case PT_FLOATMAX: return new CTreeMachine<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDecompressString(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDenseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDenseSubsetFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CDenseSubsetFeatures<bool>();
		case PT_CHAR: return new CDenseSubsetFeatures<char>();
		case PT_INT8: return new CDenseSubsetFeatures<int8_t>();
		case PT_UINT8: return new CDenseSubsetFeatures<uint8_t>();
		case PT_INT16: return new CDenseSubsetFeatures<int16_t>();
		case PT_UINT16: return new CDenseSubsetFeatures<uint16_t>();
		case PT_INT32: return new CDenseSubsetFeatures<int32_t>();
		case PT_UINT32: return new CDenseSubsetFeatures<uint32_t>();
		case PT_INT64: return new CDenseSubsetFeatures<int64_t>();
		case PT_UINT64: return new CDenseSubsetFeatures<uint64_t>();
		case PT_FLOAT32: return new CDenseSubsetFeatures<float32_t>();
		case PT_FLOAT64: return new CDenseSubsetFeatures<float64_t>();
		case PT_FLOATMAX: return new CDenseSubsetFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CHashedDenseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CHashedDenseFeatures<bool>();
		case PT_CHAR: return new CHashedDenseFeatures<char>();
		case PT_INT8: return new CHashedDenseFeatures<int8_t>();
		case PT_UINT8: return new CHashedDenseFeatures<uint8_t>();
		case PT_INT16: return new CHashedDenseFeatures<int16_t>();
		case PT_UINT16: return new CHashedDenseFeatures<uint16_t>();
		case PT_INT32: return new CHashedDenseFeatures<int32_t>();
		case PT_UINT32: return new CHashedDenseFeatures<uint32_t>();
		case PT_INT64: return new CHashedDenseFeatures<int64_t>();
		case PT_UINT64: return new CHashedDenseFeatures<uint64_t>();
		case PT_FLOAT32: return new CHashedDenseFeatures<float32_t>();
		case PT_FLOAT64: return new CHashedDenseFeatures<float64_t>();
		case PT_FLOATMAX: return new CHashedDenseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CHashedSparseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CHashedSparseFeatures<bool>();
		case PT_CHAR: return new CHashedSparseFeatures<char>();
		case PT_INT8: return new CHashedSparseFeatures<int8_t>();
		case PT_UINT8: return new CHashedSparseFeatures<uint8_t>();
		case PT_INT16: return new CHashedSparseFeatures<int16_t>();
		case PT_UINT16: return new CHashedSparseFeatures<uint16_t>();
		case PT_INT32: return new CHashedSparseFeatures<int32_t>();
		case PT_UINT32: return new CHashedSparseFeatures<uint32_t>();
		case PT_INT64: return new CHashedSparseFeatures<int64_t>();
		case PT_UINT64: return new CHashedSparseFeatures<uint64_t>();
		case PT_FLOAT32: return new CHashedSparseFeatures<float32_t>();
		case PT_FLOAT64: return new CHashedSparseFeatures<float64_t>();
		case PT_FLOATMAX: return new CHashedSparseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CMatrixFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CMatrixFeatures<bool>();
		case PT_CHAR: return new CMatrixFeatures<char>();
		case PT_INT8: return new CMatrixFeatures<int8_t>();
		case PT_UINT8: return new CMatrixFeatures<uint8_t>();
		case PT_INT16: return new CMatrixFeatures<int16_t>();
		case PT_UINT16: return new CMatrixFeatures<uint16_t>();
		case PT_INT32: return new CMatrixFeatures<int32_t>();
		case PT_UINT32: return new CMatrixFeatures<uint32_t>();
		case PT_INT64: return new CMatrixFeatures<int64_t>();
		case PT_UINT64: return new CMatrixFeatures<uint64_t>();
		case PT_FLOAT32: return new CMatrixFeatures<float32_t>();
		case PT_FLOAT64: return new CMatrixFeatures<float64_t>();
		case PT_FLOATMAX: return new CMatrixFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSparseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingDenseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingHashedDenseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStreamingHashedDenseFeatures<bool>();
		case PT_CHAR: return new CStreamingHashedDenseFeatures<char>();
		case PT_INT8: return new CStreamingHashedDenseFeatures<int8_t>();
		case PT_UINT8: return new CStreamingHashedDenseFeatures<uint8_t>();
		case PT_INT16: return new CStreamingHashedDenseFeatures<int16_t>();
		case PT_UINT16: return new CStreamingHashedDenseFeatures<uint16_t>();
		case PT_INT32: return new CStreamingHashedDenseFeatures<int32_t>();
		case PT_UINT32: return new CStreamingHashedDenseFeatures<uint32_t>();
		case PT_INT64: return new CStreamingHashedDenseFeatures<int64_t>();
		case PT_UINT64: return new CStreamingHashedDenseFeatures<uint64_t>();
		case PT_FLOAT32: return new CStreamingHashedDenseFeatures<float32_t>();
		case PT_FLOAT64: return new CStreamingHashedDenseFeatures<float64_t>();
		case PT_FLOATMAX: return new CStreamingHashedDenseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingHashedSparseFeatures(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStreamingHashedSparseFeatures<bool>();
		case PT_CHAR: return new CStreamingHashedSparseFeatures<char>();
		case PT_INT8: return new CStreamingHashedSparseFeatures<int8_t>();
		case PT_UINT8: return new CStreamingHashedSparseFeatures<uint8_t>();
		case PT_INT16: return new CStreamingHashedSparseFeatures<int16_t>();
		case PT_UINT16: return new CStreamingHashedSparseFeatures<uint16_t>();
		case PT_INT32: return new CStreamingHashedSparseFeatures<int32_t>();
		case PT_UINT32: return new CStreamingHashedSparseFeatures<uint32_t>();
		case PT_INT64: return new CStreamingHashedSparseFeatures<int64_t>();
		case PT_UINT64: return new CStreamingHashedSparseFeatures<uint64_t>();
		case PT_FLOAT32: return new CStreamingHashedSparseFeatures<float32_t>();
		case PT_FLOAT64: return new CStreamingHashedSparseFeatures<float64_t>();
		case PT_FLOATMAX: return new CStreamingHashedSparseFeatures<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingSparseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingStringFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStringFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStringFileFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CBinaryStream(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CMemoryMappedFile(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSimpleFile(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CParseBuffer(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingFileFromDenseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingFileFromSparseFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStreamingFileFromStringFeatures(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CCache(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDynamicArray(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSet(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CTreeMachine(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CTreeMachine<bool>();
		case PT_CHAR: return new CTreeMachine<char>();
		case PT_INT8: return new CTreeMachine<int8_t>();
		case PT_UINT8: return new CTreeMachine<uint8_t>();
		case PT_INT16: return new CTreeMachine<int16_t>();
		case PT_UINT16: return new CTreeMachine<uint16_t>();
		case PT_INT32: return new CTreeMachine<int32_t>();
		case PT_UINT32: return new CTreeMachine<uint32_t>();
		case PT_INT64: return new CTreeMachine<int64_t>();
		case PT_UINT64: return new CTreeMachine<uint64_t>();
		case PT_FLOAT32: return new CTreeMachine<float32_t>();
		case PT_FLOAT64: return new CTreeMachine<float64_t>();
		case PT_FLOATMAX: return new CTreeMachine<floatmax_t>();
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CDecompressString(EPrimitiveType g)
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
		case PT_COMPLEX128: return NULL;
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStoreScalarAggregator(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStoreScalarAggregator<bool>();
		case PT_CHAR: return new CStoreScalarAggregator<char>();
		case PT_INT8: return new CStoreScalarAggregator<int8_t>();
		case PT_UINT8: return new CStoreScalarAggregator<uint8_t>();
		case PT_INT16: return new CStoreScalarAggregator<int16_t>();
		case PT_UINT16: return new CStoreScalarAggregator<uint16_t>();
		case PT_INT32: return new CStoreScalarAggregator<int32_t>();
		case PT_UINT32: return new CStoreScalarAggregator<uint32_t>();
		case PT_INT64: return new CStoreScalarAggregator<int64_t>();
		case PT_UINT64: return new CStoreScalarAggregator<uint64_t>();
		case PT_FLOAT32: return new CStoreScalarAggregator<float32_t>();
		case PT_FLOAT64: return new CStoreScalarAggregator<float64_t>();
		case PT_FLOATMAX: return new CStoreScalarAggregator<floatmax_t>();
		case PT_COMPLEX128: return new CStoreScalarAggregator<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CScalarResult(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CScalarResult<bool>();
		case PT_CHAR: return new CScalarResult<char>();
		case PT_INT8: return new CScalarResult<int8_t>();
		case PT_UINT8: return new CScalarResult<uint8_t>();
		case PT_INT16: return new CScalarResult<int16_t>();
		case PT_UINT16: return new CScalarResult<uint16_t>();
		case PT_INT32: return new CScalarResult<int32_t>();
		case PT_UINT32: return new CScalarResult<uint32_t>();
		case PT_INT64: return new CScalarResult<int64_t>();
		case PT_UINT64: return new CScalarResult<uint64_t>();
		case PT_FLOAT32: return new CScalarResult<float32_t>();
		case PT_FLOAT64: return new CScalarResult<float64_t>();
		case PT_FLOATMAX: return new CScalarResult<floatmax_t>();
		case PT_COMPLEX128: return new CScalarResult<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CVectorResult(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CVectorResult<bool>();
		case PT_CHAR: return new CVectorResult<char>();
		case PT_INT8: return new CVectorResult<int8_t>();
		case PT_UINT8: return new CVectorResult<uint8_t>();
		case PT_INT16: return new CVectorResult<int16_t>();
		case PT_UINT16: return new CVectorResult<uint16_t>();
		case PT_INT32: return new CVectorResult<int32_t>();
		case PT_UINT32: return new CVectorResult<uint32_t>();
		case PT_INT64: return new CVectorResult<int64_t>();
		case PT_UINT64: return new CVectorResult<uint64_t>();
		case PT_FLOAT32: return new CVectorResult<float32_t>();
		case PT_FLOAT64: return new CVectorResult<float64_t>();
		case PT_FLOATMAX: return new CVectorResult<floatmax_t>();
		case PT_COMPLEX128: return new CVectorResult<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSparseMatrixOperator(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CSparseMatrixOperator<bool>();
		case PT_CHAR: return new CSparseMatrixOperator<char>();
		case PT_INT8: return new CSparseMatrixOperator<int8_t>();
		case PT_UINT8: return new CSparseMatrixOperator<uint8_t>();
		case PT_INT16: return new CSparseMatrixOperator<int16_t>();
		case PT_UINT16: return new CSparseMatrixOperator<uint16_t>();
		case PT_INT32: return new CSparseMatrixOperator<int32_t>();
		case PT_UINT32: return new CSparseMatrixOperator<uint32_t>();
		case PT_INT64: return new CSparseMatrixOperator<int64_t>();
		case PT_UINT64: return new CSparseMatrixOperator<uint64_t>();
		case PT_FLOAT32: return new CSparseMatrixOperator<float32_t>();
		case PT_FLOAT64: return new CSparseMatrixOperator<float64_t>();
		case PT_FLOATMAX: return new CSparseMatrixOperator<floatmax_t>();
		case PT_COMPLEX128: return new CSparseMatrixOperator<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStoreScalarAggregator(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStoreScalarAggregator<bool>();
		case PT_CHAR: return new CStoreScalarAggregator<char>();
		case PT_INT8: return new CStoreScalarAggregator<int8_t>();
		case PT_UINT8: return new CStoreScalarAggregator<uint8_t>();
		case PT_INT16: return new CStoreScalarAggregator<int16_t>();
		case PT_UINT16: return new CStoreScalarAggregator<uint16_t>();
		case PT_INT32: return new CStoreScalarAggregator<int32_t>();
		case PT_UINT32: return new CStoreScalarAggregator<uint32_t>();
		case PT_INT64: return new CStoreScalarAggregator<int64_t>();
		case PT_UINT64: return new CStoreScalarAggregator<uint64_t>();
		case PT_FLOAT32: return new CStoreScalarAggregator<float32_t>();
		case PT_FLOAT64: return new CStoreScalarAggregator<float64_t>();
		case PT_FLOATMAX: return new CStoreScalarAggregator<floatmax_t>();
		case PT_COMPLEX128: return new CStoreScalarAggregator<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CScalarResult(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CScalarResult<bool>();
		case PT_CHAR: return new CScalarResult<char>();
		case PT_INT8: return new CScalarResult<int8_t>();
		case PT_UINT8: return new CScalarResult<uint8_t>();
		case PT_INT16: return new CScalarResult<int16_t>();
		case PT_UINT16: return new CScalarResult<uint16_t>();
		case PT_INT32: return new CScalarResult<int32_t>();
		case PT_UINT32: return new CScalarResult<uint32_t>();
		case PT_INT64: return new CScalarResult<int64_t>();
		case PT_UINT64: return new CScalarResult<uint64_t>();
		case PT_FLOAT32: return new CScalarResult<float32_t>();
		case PT_FLOAT64: return new CScalarResult<float64_t>();
		case PT_FLOATMAX: return new CScalarResult<floatmax_t>();
		case PT_COMPLEX128: return new CScalarResult<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CVectorResult(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CVectorResult<bool>();
		case PT_CHAR: return new CVectorResult<char>();
		case PT_INT8: return new CVectorResult<int8_t>();
		case PT_UINT8: return new CVectorResult<uint8_t>();
		case PT_INT16: return new CVectorResult<int16_t>();
		case PT_UINT16: return new CVectorResult<uint16_t>();
		case PT_INT32: return new CVectorResult<int32_t>();
		case PT_UINT32: return new CVectorResult<uint32_t>();
		case PT_INT64: return new CVectorResult<int64_t>();
		case PT_UINT64: return new CVectorResult<uint64_t>();
		case PT_FLOAT32: return new CVectorResult<float32_t>();
		case PT_FLOAT64: return new CVectorResult<float64_t>();
		case PT_FLOATMAX: return new CVectorResult<floatmax_t>();
		case PT_COMPLEX128: return new CVectorResult<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSparseMatrixOperator(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CSparseMatrixOperator<bool>();
		case PT_CHAR: return new CSparseMatrixOperator<char>();
		case PT_INT8: return new CSparseMatrixOperator<int8_t>();
		case PT_UINT8: return new CSparseMatrixOperator<uint8_t>();
		case PT_INT16: return new CSparseMatrixOperator<int16_t>();
		case PT_UINT16: return new CSparseMatrixOperator<uint16_t>();
		case PT_INT32: return new CSparseMatrixOperator<int32_t>();
		case PT_UINT32: return new CSparseMatrixOperator<uint32_t>();
		case PT_INT64: return new CSparseMatrixOperator<int64_t>();
		case PT_UINT64: return new CSparseMatrixOperator<uint64_t>();
		case PT_FLOAT32: return new CSparseMatrixOperator<float32_t>();
		case PT_FLOAT64: return new CSparseMatrixOperator<float64_t>();
		case PT_FLOATMAX: return new CSparseMatrixOperator<floatmax_t>();
		case PT_COMPLEX128: return new CSparseMatrixOperator<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStoreScalarAggregator(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStoreScalarAggregator<bool>();
		case PT_CHAR: return new CStoreScalarAggregator<char>();
		case PT_INT8: return new CStoreScalarAggregator<int8_t>();
		case PT_UINT8: return new CStoreScalarAggregator<uint8_t>();
		case PT_INT16: return new CStoreScalarAggregator<int16_t>();
		case PT_UINT16: return new CStoreScalarAggregator<uint16_t>();
		case PT_INT32: return new CStoreScalarAggregator<int32_t>();
		case PT_UINT32: return new CStoreScalarAggregator<uint32_t>();
		case PT_INT64: return new CStoreScalarAggregator<int64_t>();
		case PT_UINT64: return new CStoreScalarAggregator<uint64_t>();
		case PT_FLOAT32: return new CStoreScalarAggregator<float32_t>();
		case PT_FLOAT64: return new CStoreScalarAggregator<float64_t>();
		case PT_FLOATMAX: return new CStoreScalarAggregator<floatmax_t>();
		case PT_COMPLEX128: return new CStoreScalarAggregator<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CScalarResult(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CScalarResult<bool>();
		case PT_CHAR: return new CScalarResult<char>();
		case PT_INT8: return new CScalarResult<int8_t>();
		case PT_UINT8: return new CScalarResult<uint8_t>();
		case PT_INT16: return new CScalarResult<int16_t>();
		case PT_UINT16: return new CScalarResult<uint16_t>();
		case PT_INT32: return new CScalarResult<int32_t>();
		case PT_UINT32: return new CScalarResult<uint32_t>();
		case PT_INT64: return new CScalarResult<int64_t>();
		case PT_UINT64: return new CScalarResult<uint64_t>();
		case PT_FLOAT32: return new CScalarResult<float32_t>();
		case PT_FLOAT64: return new CScalarResult<float64_t>();
		case PT_FLOATMAX: return new CScalarResult<floatmax_t>();
		case PT_COMPLEX128: return new CScalarResult<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CVectorResult(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CVectorResult<bool>();
		case PT_CHAR: return new CVectorResult<char>();
		case PT_INT8: return new CVectorResult<int8_t>();
		case PT_UINT8: return new CVectorResult<uint8_t>();
		case PT_INT16: return new CVectorResult<int16_t>();
		case PT_UINT16: return new CVectorResult<uint16_t>();
		case PT_INT32: return new CVectorResult<int32_t>();
		case PT_UINT32: return new CVectorResult<uint32_t>();
		case PT_INT64: return new CVectorResult<int64_t>();
		case PT_UINT64: return new CVectorResult<uint64_t>();
		case PT_FLOAT32: return new CVectorResult<float32_t>();
		case PT_FLOAT64: return new CVectorResult<float64_t>();
		case PT_FLOATMAX: return new CVectorResult<floatmax_t>();
		case PT_COMPLEX128: return new CVectorResult<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSparseMatrixOperator(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CSparseMatrixOperator<bool>();
		case PT_CHAR: return new CSparseMatrixOperator<char>();
		case PT_INT8: return new CSparseMatrixOperator<int8_t>();
		case PT_UINT8: return new CSparseMatrixOperator<uint8_t>();
		case PT_INT16: return new CSparseMatrixOperator<int16_t>();
		case PT_UINT16: return new CSparseMatrixOperator<uint16_t>();
		case PT_INT32: return new CSparseMatrixOperator<int32_t>();
		case PT_UINT32: return new CSparseMatrixOperator<uint32_t>();
		case PT_INT64: return new CSparseMatrixOperator<int64_t>();
		case PT_UINT64: return new CSparseMatrixOperator<uint64_t>();
		case PT_FLOAT32: return new CSparseMatrixOperator<float32_t>();
		case PT_FLOAT64: return new CSparseMatrixOperator<float64_t>();
		case PT_FLOATMAX: return new CSparseMatrixOperator<floatmax_t>();
		case PT_COMPLEX128: return new CSparseMatrixOperator<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStoreScalarAggregator(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStoreScalarAggregator<bool>();
		case PT_CHAR: return new CStoreScalarAggregator<char>();
		case PT_INT8: return new CStoreScalarAggregator<int8_t>();
		case PT_UINT8: return new CStoreScalarAggregator<uint8_t>();
		case PT_INT16: return new CStoreScalarAggregator<int16_t>();
		case PT_UINT16: return new CStoreScalarAggregator<uint16_t>();
		case PT_INT32: return new CStoreScalarAggregator<int32_t>();
		case PT_UINT32: return new CStoreScalarAggregator<uint32_t>();
		case PT_INT64: return new CStoreScalarAggregator<int64_t>();
		case PT_UINT64: return new CStoreScalarAggregator<uint64_t>();
		case PT_FLOAT32: return new CStoreScalarAggregator<float32_t>();
		case PT_FLOAT64: return new CStoreScalarAggregator<float64_t>();
		case PT_FLOATMAX: return new CStoreScalarAggregator<floatmax_t>();
		case PT_COMPLEX128: return new CStoreScalarAggregator<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CScalarResult(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CScalarResult<bool>();
		case PT_CHAR: return new CScalarResult<char>();
		case PT_INT8: return new CScalarResult<int8_t>();
		case PT_UINT8: return new CScalarResult<uint8_t>();
		case PT_INT16: return new CScalarResult<int16_t>();
		case PT_UINT16: return new CScalarResult<uint16_t>();
		case PT_INT32: return new CScalarResult<int32_t>();
		case PT_UINT32: return new CScalarResult<uint32_t>();
		case PT_INT64: return new CScalarResult<int64_t>();
		case PT_UINT64: return new CScalarResult<uint64_t>();
		case PT_FLOAT32: return new CScalarResult<float32_t>();
		case PT_FLOAT64: return new CScalarResult<float64_t>();
		case PT_FLOATMAX: return new CScalarResult<floatmax_t>();
		case PT_COMPLEX128: return new CScalarResult<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CVectorResult(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CVectorResult<bool>();
		case PT_CHAR: return new CVectorResult<char>();
		case PT_INT8: return new CVectorResult<int8_t>();
		case PT_UINT8: return new CVectorResult<uint8_t>();
		case PT_INT16: return new CVectorResult<int16_t>();
		case PT_UINT16: return new CVectorResult<uint16_t>();
		case PT_INT32: return new CVectorResult<int32_t>();
		case PT_UINT32: return new CVectorResult<uint32_t>();
		case PT_INT64: return new CVectorResult<int64_t>();
		case PT_UINT64: return new CVectorResult<uint64_t>();
		case PT_FLOAT32: return new CVectorResult<float32_t>();
		case PT_FLOAT64: return new CVectorResult<float64_t>();
		case PT_FLOATMAX: return new CVectorResult<floatmax_t>();
		case PT_COMPLEX128: return new CVectorResult<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSparseMatrixOperator(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CSparseMatrixOperator<bool>();
		case PT_CHAR: return new CSparseMatrixOperator<char>();
		case PT_INT8: return new CSparseMatrixOperator<int8_t>();
		case PT_UINT8: return new CSparseMatrixOperator<uint8_t>();
		case PT_INT16: return new CSparseMatrixOperator<int16_t>();
		case PT_UINT16: return new CSparseMatrixOperator<uint16_t>();
		case PT_INT32: return new CSparseMatrixOperator<int32_t>();
		case PT_UINT32: return new CSparseMatrixOperator<uint32_t>();
		case PT_INT64: return new CSparseMatrixOperator<int64_t>();
		case PT_UINT64: return new CSparseMatrixOperator<uint64_t>();
		case PT_FLOAT32: return new CSparseMatrixOperator<float32_t>();
		case PT_FLOAT64: return new CSparseMatrixOperator<float64_t>();
		case PT_FLOATMAX: return new CSparseMatrixOperator<floatmax_t>();
		case PT_COMPLEX128: return new CSparseMatrixOperator<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStoreScalarAggregator(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStoreScalarAggregator<bool>();
		case PT_CHAR: return new CStoreScalarAggregator<char>();
		case PT_INT8: return new CStoreScalarAggregator<int8_t>();
		case PT_UINT8: return new CStoreScalarAggregator<uint8_t>();
		case PT_INT16: return new CStoreScalarAggregator<int16_t>();
		case PT_UINT16: return new CStoreScalarAggregator<uint16_t>();
		case PT_INT32: return new CStoreScalarAggregator<int32_t>();
		case PT_UINT32: return new CStoreScalarAggregator<uint32_t>();
		case PT_INT64: return new CStoreScalarAggregator<int64_t>();
		case PT_UINT64: return new CStoreScalarAggregator<uint64_t>();
		case PT_FLOAT32: return new CStoreScalarAggregator<float32_t>();
		case PT_FLOAT64: return new CStoreScalarAggregator<float64_t>();
		case PT_FLOATMAX: return new CStoreScalarAggregator<floatmax_t>();
		case PT_COMPLEX128: return new CStoreScalarAggregator<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CScalarResult(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CScalarResult<bool>();
		case PT_CHAR: return new CScalarResult<char>();
		case PT_INT8: return new CScalarResult<int8_t>();
		case PT_UINT8: return new CScalarResult<uint8_t>();
		case PT_INT16: return new CScalarResult<int16_t>();
		case PT_UINT16: return new CScalarResult<uint16_t>();
		case PT_INT32: return new CScalarResult<int32_t>();
		case PT_UINT32: return new CScalarResult<uint32_t>();
		case PT_INT64: return new CScalarResult<int64_t>();
		case PT_UINT64: return new CScalarResult<uint64_t>();
		case PT_FLOAT32: return new CScalarResult<float32_t>();
		case PT_FLOAT64: return new CScalarResult<float64_t>();
		case PT_FLOATMAX: return new CScalarResult<floatmax_t>();
		case PT_COMPLEX128: return new CScalarResult<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CVectorResult(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CVectorResult<bool>();
		case PT_CHAR: return new CVectorResult<char>();
		case PT_INT8: return new CVectorResult<int8_t>();
		case PT_UINT8: return new CVectorResult<uint8_t>();
		case PT_INT16: return new CVectorResult<int16_t>();
		case PT_UINT16: return new CVectorResult<uint16_t>();
		case PT_INT32: return new CVectorResult<int32_t>();
		case PT_UINT32: return new CVectorResult<uint32_t>();
		case PT_INT64: return new CVectorResult<int64_t>();
		case PT_UINT64: return new CVectorResult<uint64_t>();
		case PT_FLOAT32: return new CVectorResult<float32_t>();
		case PT_FLOAT64: return new CVectorResult<float64_t>();
		case PT_FLOATMAX: return new CVectorResult<floatmax_t>();
		case PT_COMPLEX128: return new CVectorResult<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSparseMatrixOperator(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CSparseMatrixOperator<bool>();
		case PT_CHAR: return new CSparseMatrixOperator<char>();
		case PT_INT8: return new CSparseMatrixOperator<int8_t>();
		case PT_UINT8: return new CSparseMatrixOperator<uint8_t>();
		case PT_INT16: return new CSparseMatrixOperator<int16_t>();
		case PT_UINT16: return new CSparseMatrixOperator<uint16_t>();
		case PT_INT32: return new CSparseMatrixOperator<int32_t>();
		case PT_UINT32: return new CSparseMatrixOperator<uint32_t>();
		case PT_INT64: return new CSparseMatrixOperator<int64_t>();
		case PT_UINT64: return new CSparseMatrixOperator<uint64_t>();
		case PT_FLOAT32: return new CSparseMatrixOperator<float32_t>();
		case PT_FLOAT64: return new CSparseMatrixOperator<float64_t>();
		case PT_FLOATMAX: return new CSparseMatrixOperator<floatmax_t>();
		case PT_COMPLEX128: return new CSparseMatrixOperator<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStoreScalarAggregator(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStoreScalarAggregator<bool>();
		case PT_CHAR: return new CStoreScalarAggregator<char>();
		case PT_INT8: return new CStoreScalarAggregator<int8_t>();
		case PT_UINT8: return new CStoreScalarAggregator<uint8_t>();
		case PT_INT16: return new CStoreScalarAggregator<int16_t>();
		case PT_UINT16: return new CStoreScalarAggregator<uint16_t>();
		case PT_INT32: return new CStoreScalarAggregator<int32_t>();
		case PT_UINT32: return new CStoreScalarAggregator<uint32_t>();
		case PT_INT64: return new CStoreScalarAggregator<int64_t>();
		case PT_UINT64: return new CStoreScalarAggregator<uint64_t>();
		case PT_FLOAT32: return new CStoreScalarAggregator<float32_t>();
		case PT_FLOAT64: return new CStoreScalarAggregator<float64_t>();
		case PT_FLOATMAX: return new CStoreScalarAggregator<floatmax_t>();
		case PT_COMPLEX128: return new CStoreScalarAggregator<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CScalarResult(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CScalarResult<bool>();
		case PT_CHAR: return new CScalarResult<char>();
		case PT_INT8: return new CScalarResult<int8_t>();
		case PT_UINT8: return new CScalarResult<uint8_t>();
		case PT_INT16: return new CScalarResult<int16_t>();
		case PT_UINT16: return new CScalarResult<uint16_t>();
		case PT_INT32: return new CScalarResult<int32_t>();
		case PT_UINT32: return new CScalarResult<uint32_t>();
		case PT_INT64: return new CScalarResult<int64_t>();
		case PT_UINT64: return new CScalarResult<uint64_t>();
		case PT_FLOAT32: return new CScalarResult<float32_t>();
		case PT_FLOAT64: return new CScalarResult<float64_t>();
		case PT_FLOATMAX: return new CScalarResult<floatmax_t>();
		case PT_COMPLEX128: return new CScalarResult<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CVectorResult(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CVectorResult<bool>();
		case PT_CHAR: return new CVectorResult<char>();
		case PT_INT8: return new CVectorResult<int8_t>();
		case PT_UINT8: return new CVectorResult<uint8_t>();
		case PT_INT16: return new CVectorResult<int16_t>();
		case PT_UINT16: return new CVectorResult<uint16_t>();
		case PT_INT32: return new CVectorResult<int32_t>();
		case PT_UINT32: return new CVectorResult<uint32_t>();
		case PT_INT64: return new CVectorResult<int64_t>();
		case PT_UINT64: return new CVectorResult<uint64_t>();
		case PT_FLOAT32: return new CVectorResult<float32_t>();
		case PT_FLOAT64: return new CVectorResult<float64_t>();
		case PT_FLOATMAX: return new CVectorResult<floatmax_t>();
		case PT_COMPLEX128: return new CVectorResult<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSparseMatrixOperator(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CSparseMatrixOperator<bool>();
		case PT_CHAR: return new CSparseMatrixOperator<char>();
		case PT_INT8: return new CSparseMatrixOperator<int8_t>();
		case PT_UINT8: return new CSparseMatrixOperator<uint8_t>();
		case PT_INT16: return new CSparseMatrixOperator<int16_t>();
		case PT_UINT16: return new CSparseMatrixOperator<uint16_t>();
		case PT_INT32: return new CSparseMatrixOperator<int32_t>();
		case PT_UINT32: return new CSparseMatrixOperator<uint32_t>();
		case PT_INT64: return new CSparseMatrixOperator<int64_t>();
		case PT_UINT64: return new CSparseMatrixOperator<uint64_t>();
		case PT_FLOAT32: return new CSparseMatrixOperator<float32_t>();
		case PT_FLOAT64: return new CSparseMatrixOperator<float64_t>();
		case PT_FLOATMAX: return new CSparseMatrixOperator<floatmax_t>();
		case PT_COMPLEX128: return new CSparseMatrixOperator<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStoreScalarAggregator(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStoreScalarAggregator<bool>();
		case PT_CHAR: return new CStoreScalarAggregator<char>();
		case PT_INT8: return new CStoreScalarAggregator<int8_t>();
		case PT_UINT8: return new CStoreScalarAggregator<uint8_t>();
		case PT_INT16: return new CStoreScalarAggregator<int16_t>();
		case PT_UINT16: return new CStoreScalarAggregator<uint16_t>();
		case PT_INT32: return new CStoreScalarAggregator<int32_t>();
		case PT_UINT32: return new CStoreScalarAggregator<uint32_t>();
		case PT_INT64: return new CStoreScalarAggregator<int64_t>();
		case PT_UINT64: return new CStoreScalarAggregator<uint64_t>();
		case PT_FLOAT32: return new CStoreScalarAggregator<float32_t>();
		case PT_FLOAT64: return new CStoreScalarAggregator<float64_t>();
		case PT_FLOATMAX: return new CStoreScalarAggregator<floatmax_t>();
		case PT_COMPLEX128: return new CStoreScalarAggregator<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CScalarResult(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CScalarResult<bool>();
		case PT_CHAR: return new CScalarResult<char>();
		case PT_INT8: return new CScalarResult<int8_t>();
		case PT_UINT8: return new CScalarResult<uint8_t>();
		case PT_INT16: return new CScalarResult<int16_t>();
		case PT_UINT16: return new CScalarResult<uint16_t>();
		case PT_INT32: return new CScalarResult<int32_t>();
		case PT_UINT32: return new CScalarResult<uint32_t>();
		case PT_INT64: return new CScalarResult<int64_t>();
		case PT_UINT64: return new CScalarResult<uint64_t>();
		case PT_FLOAT32: return new CScalarResult<float32_t>();
		case PT_FLOAT64: return new CScalarResult<float64_t>();
		case PT_FLOATMAX: return new CScalarResult<floatmax_t>();
		case PT_COMPLEX128: return new CScalarResult<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CVectorResult(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CVectorResult<bool>();
		case PT_CHAR: return new CVectorResult<char>();
		case PT_INT8: return new CVectorResult<int8_t>();
		case PT_UINT8: return new CVectorResult<uint8_t>();
		case PT_INT16: return new CVectorResult<int16_t>();
		case PT_UINT16: return new CVectorResult<uint16_t>();
		case PT_INT32: return new CVectorResult<int32_t>();
		case PT_UINT32: return new CVectorResult<uint32_t>();
		case PT_INT64: return new CVectorResult<int64_t>();
		case PT_UINT64: return new CVectorResult<uint64_t>();
		case PT_FLOAT32: return new CVectorResult<float32_t>();
		case PT_FLOAT64: return new CVectorResult<float64_t>();
		case PT_FLOATMAX: return new CVectorResult<floatmax_t>();
		case PT_COMPLEX128: return new CVectorResult<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSparseMatrixOperator(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CSparseMatrixOperator<bool>();
		case PT_CHAR: return new CSparseMatrixOperator<char>();
		case PT_INT8: return new CSparseMatrixOperator<int8_t>();
		case PT_UINT8: return new CSparseMatrixOperator<uint8_t>();
		case PT_INT16: return new CSparseMatrixOperator<int16_t>();
		case PT_UINT16: return new CSparseMatrixOperator<uint16_t>();
		case PT_INT32: return new CSparseMatrixOperator<int32_t>();
		case PT_UINT32: return new CSparseMatrixOperator<uint32_t>();
		case PT_INT64: return new CSparseMatrixOperator<int64_t>();
		case PT_UINT64: return new CSparseMatrixOperator<uint64_t>();
		case PT_FLOAT32: return new CSparseMatrixOperator<float32_t>();
		case PT_FLOAT64: return new CSparseMatrixOperator<float64_t>();
		case PT_FLOATMAX: return new CSparseMatrixOperator<floatmax_t>();
		case PT_COMPLEX128: return new CSparseMatrixOperator<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CStoreScalarAggregator(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CStoreScalarAggregator<bool>();
		case PT_CHAR: return new CStoreScalarAggregator<char>();
		case PT_INT8: return new CStoreScalarAggregator<int8_t>();
		case PT_UINT8: return new CStoreScalarAggregator<uint8_t>();
		case PT_INT16: return new CStoreScalarAggregator<int16_t>();
		case PT_UINT16: return new CStoreScalarAggregator<uint16_t>();
		case PT_INT32: return new CStoreScalarAggregator<int32_t>();
		case PT_UINT32: return new CStoreScalarAggregator<uint32_t>();
		case PT_INT64: return new CStoreScalarAggregator<int64_t>();
		case PT_UINT64: return new CStoreScalarAggregator<uint64_t>();
		case PT_FLOAT32: return new CStoreScalarAggregator<float32_t>();
		case PT_FLOAT64: return new CStoreScalarAggregator<float64_t>();
		case PT_FLOATMAX: return new CStoreScalarAggregator<floatmax_t>();
		case PT_COMPLEX128: return new CStoreScalarAggregator<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CScalarResult(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CScalarResult<bool>();
		case PT_CHAR: return new CScalarResult<char>();
		case PT_INT8: return new CScalarResult<int8_t>();
		case PT_UINT8: return new CScalarResult<uint8_t>();
		case PT_INT16: return new CScalarResult<int16_t>();
		case PT_UINT16: return new CScalarResult<uint16_t>();
		case PT_INT32: return new CScalarResult<int32_t>();
		case PT_UINT32: return new CScalarResult<uint32_t>();
		case PT_INT64: return new CScalarResult<int64_t>();
		case PT_UINT64: return new CScalarResult<uint64_t>();
		case PT_FLOAT32: return new CScalarResult<float32_t>();
		case PT_FLOAT64: return new CScalarResult<float64_t>();
		case PT_FLOATMAX: return new CScalarResult<floatmax_t>();
		case PT_COMPLEX128: return new CScalarResult<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CVectorResult(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CVectorResult<bool>();
		case PT_CHAR: return new CVectorResult<char>();
		case PT_INT8: return new CVectorResult<int8_t>();
		case PT_UINT8: return new CVectorResult<uint8_t>();
		case PT_INT16: return new CVectorResult<int16_t>();
		case PT_UINT16: return new CVectorResult<uint16_t>();
		case PT_INT32: return new CVectorResult<int32_t>();
		case PT_UINT32: return new CVectorResult<uint32_t>();
		case PT_INT64: return new CVectorResult<int64_t>();
		case PT_UINT64: return new CVectorResult<uint64_t>();
		case PT_FLOAT32: return new CVectorResult<float32_t>();
		case PT_FLOAT64: return new CVectorResult<float64_t>();
		case PT_FLOATMAX: return new CVectorResult<floatmax_t>();
		case PT_COMPLEX128: return new CVectorResult<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
	}
	return NULL;
}
static SHOGUN_TEMPLATE_CLASS CSGObject* __new_CSparseMatrixOperator(EPrimitiveType g)
{
	switch (g)
	{
		case PT_BOOL: return new CSparseMatrixOperator<bool>();
		case PT_CHAR: return new CSparseMatrixOperator<char>();
		case PT_INT8: return new CSparseMatrixOperator<int8_t>();
		case PT_UINT8: return new CSparseMatrixOperator<uint8_t>();
		case PT_INT16: return new CSparseMatrixOperator<int16_t>();
		case PT_UINT16: return new CSparseMatrixOperator<uint16_t>();
		case PT_INT32: return new CSparseMatrixOperator<int32_t>();
		case PT_UINT32: return new CSparseMatrixOperator<uint32_t>();
		case PT_INT64: return new CSparseMatrixOperator<int64_t>();
		case PT_UINT64: return new CSparseMatrixOperator<uint64_t>();
		case PT_FLOAT32: return new CSparseMatrixOperator<float32_t>();
		case PT_FLOAT64: return new CSparseMatrixOperator<float64_t>();
		case PT_FLOATMAX: return new CSparseMatrixOperator<floatmax_t>();
		case PT_COMPLEX128: return new CSparseMatrixOperator<complex128_t>();
		case PT_SGOBJECT:
		case PT_UNDEFINED: return NULL;
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
{"AveragedPerceptron", SHOGUN_BASIC_CLASS __new_CAveragedPerceptron},
{"FeatureBlockLogisticRegression", SHOGUN_BASIC_CLASS __new_CFeatureBlockLogisticRegression},
{"MKLClassification", SHOGUN_BASIC_CLASS __new_CMKLClassification},
{"MKLMulticlass", SHOGUN_BASIC_CLASS __new_CMKLMulticlass},
{"MKLOneClass", SHOGUN_BASIC_CLASS __new_CMKLOneClass},
{"NearestCentroid", SHOGUN_BASIC_CLASS __new_CNearestCentroid},
{"Perceptron", SHOGUN_BASIC_CLASS __new_CPerceptron},
{"PluginEstimate", SHOGUN_BASIC_CLASS __new_CPluginEstimate},
{"GNPPLib", SHOGUN_BASIC_CLASS __new_CGNPPLib},
{"GNPPSVM", SHOGUN_BASIC_CLASS __new_CGNPPSVM},
{"GPBTSVM", SHOGUN_BASIC_CLASS __new_CGPBTSVM},
{"LibLinear", SHOGUN_BASIC_CLASS __new_CLibLinear},
{"LibSVM", SHOGUN_BASIC_CLASS __new_CLibSVM},
{"LibSVMOneClass", SHOGUN_BASIC_CLASS __new_CLibSVMOneClass},
{"MPDSVM", SHOGUN_BASIC_CLASS __new_CMPDSVM},
{"OnlineLibLinear", SHOGUN_BASIC_CLASS __new_COnlineLibLinear},
{"OnlineSVMSGD", SHOGUN_BASIC_CLASS __new_COnlineSVMSGD},
{"QPBSVMLib", SHOGUN_BASIC_CLASS __new_CQPBSVMLib},
{"SGDQN", SHOGUN_BASIC_CLASS __new_CSGDQN},
{"SVM", SHOGUN_BASIC_CLASS __new_CSVM},
{"SVMLight", SHOGUN_BASIC_CLASS __new_CSVMLight},
{"SVMLightOneClass", SHOGUN_BASIC_CLASS __new_CSVMLightOneClass},
{"SVMLin", SHOGUN_BASIC_CLASS __new_CSVMLin},
{"SVMOcas", SHOGUN_BASIC_CLASS __new_CSVMOcas},
{"SVMSGD", SHOGUN_BASIC_CLASS __new_CSVMSGD},
{"WDSVMOcas", SHOGUN_BASIC_CLASS __new_CWDSVMOcas},
{"VwNativeCacheReader", SHOGUN_BASIC_CLASS __new_CVwNativeCacheReader},
{"VwNativeCacheWriter", SHOGUN_BASIC_CLASS __new_CVwNativeCacheWriter},
{"VwAdaptiveLearner", SHOGUN_BASIC_CLASS __new_CVwAdaptiveLearner},
{"VwNonAdaptiveLearner", SHOGUN_BASIC_CLASS __new_CVwNonAdaptiveLearner},
{"VowpalWabbit", SHOGUN_BASIC_CLASS __new_CVowpalWabbit},
{"VwEnvironment", SHOGUN_BASIC_CLASS __new_CVwEnvironment},
{"VwParser", SHOGUN_BASIC_CLASS __new_CVwParser},
{"VwRegressor", SHOGUN_BASIC_CLASS __new_CVwRegressor},
{"Hierarchical", SHOGUN_BASIC_CLASS __new_CHierarchical},
{"KMeans", SHOGUN_BASIC_CLASS __new_CKMeans},
{"HashedDocConverter", SHOGUN_BASIC_CLASS __new_CHashedDocConverter},
{"AttenuatedEuclideanDistance", SHOGUN_BASIC_CLASS __new_CAttenuatedEuclideanDistance},
{"BrayCurtisDistance", SHOGUN_BASIC_CLASS __new_CBrayCurtisDistance},
{"CanberraMetric", SHOGUN_BASIC_CLASS __new_CCanberraMetric},
{"CanberraWordDistance", SHOGUN_BASIC_CLASS __new_CCanberraWordDistance},
{"ChebyshewMetric", SHOGUN_BASIC_CLASS __new_CChebyshewMetric},
{"ChiSquareDistance", SHOGUN_BASIC_CLASS __new_CChiSquareDistance},
{"CosineDistance", SHOGUN_BASIC_CLASS __new_CCosineDistance},
{"CustomDistance", SHOGUN_BASIC_CLASS __new_CCustomDistance},
{"EuclideanDistance", SHOGUN_BASIC_CLASS __new_CEuclideanDistance},
{"GeodesicMetric", SHOGUN_BASIC_CLASS __new_CGeodesicMetric},
{"HammingWordDistance", SHOGUN_BASIC_CLASS __new_CHammingWordDistance},
{"JensenMetric", SHOGUN_BASIC_CLASS __new_CJensenMetric},
{"KernelDistance", SHOGUN_BASIC_CLASS __new_CKernelDistance},
{"ManhattanMetric", SHOGUN_BASIC_CLASS __new_CManhattanMetric},
{"ManhattanWordDistance", SHOGUN_BASIC_CLASS __new_CManhattanWordDistance},
{"MinkowskiMetric", SHOGUN_BASIC_CLASS __new_CMinkowskiMetric},
{"SparseEuclideanDistance", SHOGUN_BASIC_CLASS __new_CSparseEuclideanDistance},
{"TanimotoDistance", SHOGUN_BASIC_CLASS __new_CTanimotoDistance},
{"GHMM", SHOGUN_BASIC_CLASS __new_CGHMM},
{"Histogram", SHOGUN_BASIC_CLASS __new_CHistogram},
{"HMM", SHOGUN_BASIC_CLASS __new_CHMM},
{"LinearHMM", SHOGUN_BASIC_CLASS __new_CLinearHMM},
{"PositionalPWM", SHOGUN_BASIC_CLASS __new_CPositionalPWM},
{"MajorityVote", SHOGUN_BASIC_CLASS __new_CMajorityVote},
{"MeanRule", SHOGUN_BASIC_CLASS __new_CMeanRule},
{"WeightedMajorityVote", SHOGUN_BASIC_CLASS __new_CWeightedMajorityVote},
{"ClusteringAccuracy", SHOGUN_BASIC_CLASS __new_CClusteringAccuracy},
{"ClusteringMutualInformation", SHOGUN_BASIC_CLASS __new_CClusteringMutualInformation},
{"ContingencyTableEvaluation", SHOGUN_BASIC_CLASS __new_CContingencyTableEvaluation},
{"AccuracyMeasure", SHOGUN_BASIC_CLASS __new_CAccuracyMeasure},
{"ErrorRateMeasure", SHOGUN_BASIC_CLASS __new_CErrorRateMeasure},
{"BALMeasure", SHOGUN_BASIC_CLASS __new_CBALMeasure},
{"WRACCMeasure", SHOGUN_BASIC_CLASS __new_CWRACCMeasure},
{"F1Measure", SHOGUN_BASIC_CLASS __new_CF1Measure},
{"CrossCorrelationMeasure", SHOGUN_BASIC_CLASS __new_CCrossCorrelationMeasure},
{"RecallMeasure", SHOGUN_BASIC_CLASS __new_CRecallMeasure},
{"PrecisionMeasure", SHOGUN_BASIC_CLASS __new_CPrecisionMeasure},
{"SpecificityMeasure", SHOGUN_BASIC_CLASS __new_CSpecificityMeasure},
{"CrossValidationResult", SHOGUN_BASIC_CLASS __new_CCrossValidationResult},
{"CrossValidation", SHOGUN_BASIC_CLASS __new_CCrossValidation},
{"CrossValidationMKLStorage", SHOGUN_BASIC_CLASS __new_CCrossValidationMKLStorage},
{"CrossValidationMulticlassStorage", SHOGUN_BASIC_CLASS __new_CCrossValidationMulticlassStorage},
{"CrossValidationPrintOutput", SHOGUN_BASIC_CLASS __new_CCrossValidationPrintOutput},
{"CrossValidationSplitting", SHOGUN_BASIC_CLASS __new_CCrossValidationSplitting},
{"GradientCriterion", SHOGUN_BASIC_CLASS __new_CGradientCriterion},
{"GradientEvaluation", SHOGUN_BASIC_CLASS __new_CGradientEvaluation},
{"GradientResult", SHOGUN_BASIC_CLASS __new_CGradientResult},
{"MeanAbsoluteError", SHOGUN_BASIC_CLASS __new_CMeanAbsoluteError},
{"MeanSquaredError", SHOGUN_BASIC_CLASS __new_CMeanSquaredError},
{"MeanSquaredLogError", SHOGUN_BASIC_CLASS __new_CMeanSquaredLogError},
{"MulticlassAccuracy", SHOGUN_BASIC_CLASS __new_CMulticlassAccuracy},
{"MulticlassOVREvaluation", SHOGUN_BASIC_CLASS __new_CMulticlassOVREvaluation},
{"PRCEvaluation", SHOGUN_BASIC_CLASS __new_CPRCEvaluation},
{"ROCEvaluation", SHOGUN_BASIC_CLASS __new_CROCEvaluation},
{"StratifiedCrossValidationSplitting", SHOGUN_BASIC_CLASS __new_CStratifiedCrossValidationSplitting},
{"StructuredAccuracy", SHOGUN_BASIC_CLASS __new_CStructuredAccuracy},
{"Alphabet", SHOGUN_BASIC_CLASS __new_CAlphabet},
{"BinnedDotFeatures", SHOGUN_BASIC_CLASS __new_CBinnedDotFeatures},
{"CombinedDotFeatures", SHOGUN_BASIC_CLASS __new_CCombinedDotFeatures},
{"CombinedFeatures", SHOGUN_BASIC_CLASS __new_CCombinedFeatures},
{"DataGenerator", SHOGUN_BASIC_CLASS __new_CDataGenerator},
{"DummyFeatures", SHOGUN_BASIC_CLASS __new_CDummyFeatures},
{"ExplicitSpecFeatures", SHOGUN_BASIC_CLASS __new_CExplicitSpecFeatures},
{"FactorGraphFeatures", SHOGUN_BASIC_CLASS __new_CFactorGraphFeatures},
{"FKFeatures", SHOGUN_BASIC_CLASS __new_CFKFeatures},
{"HashedDocDotFeatures", SHOGUN_BASIC_CLASS __new_CHashedDocDotFeatures},
{"HashedWDFeatures", SHOGUN_BASIC_CLASS __new_CHashedWDFeatures},
{"HashedWDFeaturesTransposed", SHOGUN_BASIC_CLASS __new_CHashedWDFeaturesTransposed},
{"ImplicitWeightedSpecFeatures", SHOGUN_BASIC_CLASS __new_CImplicitWeightedSpecFeatures},
{"LatentFeatures", SHOGUN_BASIC_CLASS __new_CLatentFeatures},
{"LBPPyrDotFeatures", SHOGUN_BASIC_CLASS __new_CLBPPyrDotFeatures},
{"PolyFeatures", SHOGUN_BASIC_CLASS __new_CPolyFeatures},
{"RandomFourierDotFeatures", SHOGUN_BASIC_CLASS __new_CRandomFourierDotFeatures},
{"RealFileFeatures", SHOGUN_BASIC_CLASS __new_CRealFileFeatures},
{"SNPFeatures", SHOGUN_BASIC_CLASS __new_CSNPFeatures},
{"SparsePolyFeatures", SHOGUN_BASIC_CLASS __new_CSparsePolyFeatures},
{"GaussianBlobsDataGenerator", SHOGUN_BASIC_CLASS __new_CGaussianBlobsDataGenerator},
{"MeanShiftDataGenerator", SHOGUN_BASIC_CLASS __new_CMeanShiftDataGenerator},
{"StreamingHashedDocDotFeatures", SHOGUN_BASIC_CLASS __new_CStreamingHashedDocDotFeatures},
{"StreamingVwFeatures", SHOGUN_BASIC_CLASS __new_CStreamingVwFeatures},
{"Subset", SHOGUN_BASIC_CLASS __new_CSubset},
{"SubsetStack", SHOGUN_BASIC_CLASS __new_CSubsetStack},
{"TOPFeatures", SHOGUN_BASIC_CLASS __new_CTOPFeatures},
{"WDFeatures", SHOGUN_BASIC_CLASS __new_CWDFeatures},
{"BinaryFile", SHOGUN_BASIC_CLASS __new_CBinaryFile},
{"CSVFile", SHOGUN_BASIC_CLASS __new_CCSVFile},
{"IOBuffer", SHOGUN_BASIC_CLASS __new_CIOBuffer},
{"LibSVMFile", SHOGUN_BASIC_CLASS __new_CLibSVMFile},
{"LineReader", SHOGUN_BASIC_CLASS __new_CLineReader},
{"Parser", SHOGUN_BASIC_CLASS __new_CParser},
{"SerializableAsciiFile", SHOGUN_BASIC_CLASS __new_CSerializableAsciiFile},
{"StreamingAsciiFile", SHOGUN_BASIC_CLASS __new_CStreamingAsciiFile},
{"StreamingFile", SHOGUN_BASIC_CLASS __new_CStreamingFile},
{"StreamingFileFromFeatures", SHOGUN_BASIC_CLASS __new_CStreamingFileFromFeatures},
{"StreamingVwCacheFile", SHOGUN_BASIC_CLASS __new_CStreamingVwCacheFile},
{"StreamingVwFile", SHOGUN_BASIC_CLASS __new_CStreamingVwFile},
{"ANOVAKernel", SHOGUN_BASIC_CLASS __new_CANOVAKernel},
{"AUCKernel", SHOGUN_BASIC_CLASS __new_CAUCKernel},
{"BesselKernel", SHOGUN_BASIC_CLASS __new_CBesselKernel},
{"CauchyKernel", SHOGUN_BASIC_CLASS __new_CCauchyKernel},
{"Chi2Kernel", SHOGUN_BASIC_CLASS __new_CChi2Kernel},
{"CircularKernel", SHOGUN_BASIC_CLASS __new_CCircularKernel},
{"CombinedKernel", SHOGUN_BASIC_CLASS __new_CCombinedKernel},
{"ConstKernel", SHOGUN_BASIC_CLASS __new_CConstKernel},
{"CustomKernel", SHOGUN_BASIC_CLASS __new_CCustomKernel},
{"DiagKernel", SHOGUN_BASIC_CLASS __new_CDiagKernel},
{"DistanceKernel", SHOGUN_BASIC_CLASS __new_CDistanceKernel},
{"ExponentialKernel", SHOGUN_BASIC_CLASS __new_CExponentialKernel},
{"GaussianARDKernel", SHOGUN_BASIC_CLASS __new_CGaussianARDKernel},
{"GaussianKernel", SHOGUN_BASIC_CLASS __new_CGaussianKernel},
{"GaussianShiftKernel", SHOGUN_BASIC_CLASS __new_CGaussianShiftKernel},
{"GaussianShortRealKernel", SHOGUN_BASIC_CLASS __new_CGaussianShortRealKernel},
{"HistogramIntersectionKernel", SHOGUN_BASIC_CLASS __new_CHistogramIntersectionKernel},
{"InverseMultiQuadricKernel", SHOGUN_BASIC_CLASS __new_CInverseMultiQuadricKernel},
{"JensenShannonKernel", SHOGUN_BASIC_CLASS __new_CJensenShannonKernel},
{"LinearARDKernel", SHOGUN_BASIC_CLASS __new_CLinearARDKernel},
{"LinearKernel", SHOGUN_BASIC_CLASS __new_CLinearKernel},
{"LogKernel", SHOGUN_BASIC_CLASS __new_CLogKernel},
{"MultiquadricKernel", SHOGUN_BASIC_CLASS __new_CMultiquadricKernel},
{"AvgDiagKernelNormalizer", SHOGUN_BASIC_CLASS __new_CAvgDiagKernelNormalizer},
{"DiceKernelNormalizer", SHOGUN_BASIC_CLASS __new_CDiceKernelNormalizer},
{"FirstElementKernelNormalizer", SHOGUN_BASIC_CLASS __new_CFirstElementKernelNormalizer},
{"IdentityKernelNormalizer", SHOGUN_BASIC_CLASS __new_CIdentityKernelNormalizer},
{"RidgeKernelNormalizer", SHOGUN_BASIC_CLASS __new_CRidgeKernelNormalizer},
{"ScatterKernelNormalizer", SHOGUN_BASIC_CLASS __new_CScatterKernelNormalizer},
{"SqrtDiagKernelNormalizer", SHOGUN_BASIC_CLASS __new_CSqrtDiagKernelNormalizer},
{"TanimotoKernelNormalizer", SHOGUN_BASIC_CLASS __new_CTanimotoKernelNormalizer},
{"VarianceKernelNormalizer", SHOGUN_BASIC_CLASS __new_CVarianceKernelNormalizer},
{"ZeroMeanCenterKernelNormalizer", SHOGUN_BASIC_CLASS __new_CZeroMeanCenterKernelNormalizer},
{"PolyKernel", SHOGUN_BASIC_CLASS __new_CPolyKernel},
{"PowerKernel", SHOGUN_BASIC_CLASS __new_CPowerKernel},
{"ProductKernel", SHOGUN_BASIC_CLASS __new_CProductKernel},
{"PyramidChi2", SHOGUN_BASIC_CLASS __new_CPyramidChi2},
{"RationalQuadraticKernel", SHOGUN_BASIC_CLASS __new_CRationalQuadraticKernel},
{"SigmoidKernel", SHOGUN_BASIC_CLASS __new_CSigmoidKernel},
{"SphericalKernel", SHOGUN_BASIC_CLASS __new_CSphericalKernel},
{"SplineKernel", SHOGUN_BASIC_CLASS __new_CSplineKernel},
{"CommUlongStringKernel", SHOGUN_BASIC_CLASS __new_CCommUlongStringKernel},
{"CommWordStringKernel", SHOGUN_BASIC_CLASS __new_CCommWordStringKernel},
{"DistantSegmentsKernel", SHOGUN_BASIC_CLASS __new_CDistantSegmentsKernel},
{"FixedDegreeStringKernel", SHOGUN_BASIC_CLASS __new_CFixedDegreeStringKernel},
{"GaussianMatchStringKernel", SHOGUN_BASIC_CLASS __new_CGaussianMatchStringKernel},
{"HistogramWordStringKernel", SHOGUN_BASIC_CLASS __new_CHistogramWordStringKernel},
{"LinearStringKernel", SHOGUN_BASIC_CLASS __new_CLinearStringKernel},
{"LocalAlignmentStringKernel", SHOGUN_BASIC_CLASS __new_CLocalAlignmentStringKernel},
{"LocalityImprovedStringKernel", SHOGUN_BASIC_CLASS __new_CLocalityImprovedStringKernel},
{"MatchWordStringKernel", SHOGUN_BASIC_CLASS __new_CMatchWordStringKernel},
{"OligoStringKernel", SHOGUN_BASIC_CLASS __new_COligoStringKernel},
{"PolyMatchStringKernel", SHOGUN_BASIC_CLASS __new_CPolyMatchStringKernel},
{"PolyMatchWordStringKernel", SHOGUN_BASIC_CLASS __new_CPolyMatchWordStringKernel},
{"RegulatoryModulesStringKernel", SHOGUN_BASIC_CLASS __new_CRegulatoryModulesStringKernel},
{"SalzbergWordStringKernel", SHOGUN_BASIC_CLASS __new_CSalzbergWordStringKernel},
{"SimpleLocalityImprovedStringKernel", SHOGUN_BASIC_CLASS __new_CSimpleLocalityImprovedStringKernel},
{"SNPStringKernel", SHOGUN_BASIC_CLASS __new_CSNPStringKernel},
{"SparseSpatialSampleStringKernel", SHOGUN_BASIC_CLASS __new_CSparseSpatialSampleStringKernel},
{"SpectrumMismatchRBFKernel", SHOGUN_BASIC_CLASS __new_CSpectrumMismatchRBFKernel},
{"SpectrumRBFKernel", SHOGUN_BASIC_CLASS __new_CSpectrumRBFKernel},
{"WeightedCommWordStringKernel", SHOGUN_BASIC_CLASS __new_CWeightedCommWordStringKernel},
{"WeightedDegreePositionStringKernel", SHOGUN_BASIC_CLASS __new_CWeightedDegreePositionStringKernel},
{"WeightedDegreeStringKernel", SHOGUN_BASIC_CLASS __new_CWeightedDegreeStringKernel},
{"TensorProductPairKernel", SHOGUN_BASIC_CLASS __new_CTensorProductPairKernel},
{"TStudentKernel", SHOGUN_BASIC_CLASS __new_CTStudentKernel},
{"WaveKernel", SHOGUN_BASIC_CLASS __new_CWaveKernel},
{"WaveletKernel", SHOGUN_BASIC_CLASS __new_CWaveletKernel},
{"WeightedDegreeRBFKernel", SHOGUN_BASIC_CLASS __new_CWeightedDegreeRBFKernel},
{"BinaryLabels", SHOGUN_BASIC_CLASS __new_CBinaryLabels},
{"FactorGraphObservation", SHOGUN_BASIC_CLASS __new_CFactorGraphObservation},
{"FactorGraphLabels", SHOGUN_BASIC_CLASS __new_CFactorGraphLabels},
{"LabelsFactory", SHOGUN_BASIC_CLASS __new_CLabelsFactory},
{"LatentLabels", SHOGUN_BASIC_CLASS __new_CLatentLabels},
{"MulticlassLabels", SHOGUN_BASIC_CLASS __new_CMulticlassLabels},
{"MulticlassMultipleOutputLabels", SHOGUN_BASIC_CLASS __new_CMulticlassMultipleOutputLabels},
{"RegressionLabels", SHOGUN_BASIC_CLASS __new_CRegressionLabels},
{"StructuredLabels", SHOGUN_BASIC_CLASS __new_CStructuredLabels},
{"LatentSOSVM", SHOGUN_BASIC_CLASS __new_CLatentSOSVM},
{"LatentSVM", SHOGUN_BASIC_CLASS __new_CLatentSVM},
{"BitString", SHOGUN_BASIC_CLASS __new_CBitString},
{"CircularBuffer", SHOGUN_BASIC_CLASS __new_CCircularBuffer},
{"Compressor", SHOGUN_BASIC_CLASS __new_CCompressor},
{"SerialComputationEngine", SHOGUN_BASIC_CLASS __new_CSerialComputationEngine},
{"JobResult", SHOGUN_BASIC_CLASS __new_CJobResult},
{"Data", SHOGUN_BASIC_CLASS __new_CData},
{"DelimiterTokenizer", SHOGUN_BASIC_CLASS __new_CDelimiterTokenizer},
{"DynamicObjectArray", SHOGUN_BASIC_CLASS __new_CDynamicObjectArray},
{"Hash", SHOGUN_BASIC_CLASS __new_CHash},
{"IndexBlock", SHOGUN_BASIC_CLASS __new_CIndexBlock},
{"IndexBlockGroup", SHOGUN_BASIC_CLASS __new_CIndexBlockGroup},
{"IndexBlockTree", SHOGUN_BASIC_CLASS __new_CIndexBlockTree},
{"ListElement", SHOGUN_BASIC_CLASS __new_CListElement},
{"List", SHOGUN_BASIC_CLASS __new_CList},
{"NGramTokenizer", SHOGUN_BASIC_CLASS __new_CNGramTokenizer},
{"Signal", SHOGUN_BASIC_CLASS __new_CSignal},
{"StructuredData", SHOGUN_BASIC_CLASS __new_CStructuredData},
{"Time", SHOGUN_BASIC_CLASS __new_CTime},
{"HingeLoss", SHOGUN_BASIC_CLASS __new_CHingeLoss},
{"LogLoss", SHOGUN_BASIC_CLASS __new_CLogLoss},
{"LogLossMargin", SHOGUN_BASIC_CLASS __new_CLogLossMargin},
{"SmoothHingeLoss", SHOGUN_BASIC_CLASS __new_CSmoothHingeLoss},
{"SquaredHingeLoss", SHOGUN_BASIC_CLASS __new_CSquaredHingeLoss},
{"SquaredLoss", SHOGUN_BASIC_CLASS __new_CSquaredLoss},
{"BaggingMachine", SHOGUN_BASIC_CLASS __new_CBaggingMachine},
{"BaseMulticlassMachine", SHOGUN_BASIC_CLASS __new_CBaseMulticlassMachine},
{"DistanceMachine", SHOGUN_BASIC_CLASS __new_CDistanceMachine},
{"ZeroMean", SHOGUN_BASIC_CLASS __new_CZeroMean},
{"KernelMachine", SHOGUN_BASIC_CLASS __new_CKernelMachine},
{"KernelMulticlassMachine", SHOGUN_BASIC_CLASS __new_CKernelMulticlassMachine},
{"KernelStructuredOutputMachine", SHOGUN_BASIC_CLASS __new_CKernelStructuredOutputMachine},
{"LinearMachine", SHOGUN_BASIC_CLASS __new_CLinearMachine},
{"LinearMulticlassMachine", SHOGUN_BASIC_CLASS __new_CLinearMulticlassMachine},
{"LinearStructuredOutputMachine", SHOGUN_BASIC_CLASS __new_CLinearStructuredOutputMachine},
{"Machine", SHOGUN_BASIC_CLASS __new_CMachine},
{"NativeMulticlassMachine", SHOGUN_BASIC_CLASS __new_CNativeMulticlassMachine},
{"OnlineLinearMachine", SHOGUN_BASIC_CLASS __new_COnlineLinearMachine},
{"StructuredOutputMachine", SHOGUN_BASIC_CLASS __new_CStructuredOutputMachine},
{"JacobiEllipticFunctions", SHOGUN_BASIC_CLASS __new_CJacobiEllipticFunctions},
{"LogDetEstimator", SHOGUN_BASIC_CLASS __new_CLogDetEstimator},
{"NormalSampler", SHOGUN_BASIC_CLASS __new_CNormalSampler},
{"Math", SHOGUN_BASIC_CLASS __new_CMath},
{"Random", SHOGUN_BASIC_CLASS __new_CRandom},
{"SparseInverseCovariance", SHOGUN_BASIC_CLASS __new_CSparseInverseCovariance},
{"Statistics", SHOGUN_BASIC_CLASS __new_CStatistics},
{"GridSearchModelSelection", SHOGUN_BASIC_CLASS __new_CGridSearchModelSelection},
{"ModelSelectionParameters", SHOGUN_BASIC_CLASS __new_CModelSelectionParameters},
{"ParameterCombination", SHOGUN_BASIC_CLASS __new_CParameterCombination},
{"RandomSearchModelSelection", SHOGUN_BASIC_CLASS __new_CRandomSearchModelSelection},
{"ECOCAEDDecoder", SHOGUN_BASIC_CLASS __new_CECOCAEDDecoder},
{"ECOCDiscriminantEncoder", SHOGUN_BASIC_CLASS __new_CECOCDiscriminantEncoder},
{"ECOCEDDecoder", SHOGUN_BASIC_CLASS __new_CECOCEDDecoder},
{"ECOCForestEncoder", SHOGUN_BASIC_CLASS __new_CECOCForestEncoder},
{"ECOCHDDecoder", SHOGUN_BASIC_CLASS __new_CECOCHDDecoder},
{"ECOCLLBDecoder", SHOGUN_BASIC_CLASS __new_CECOCLLBDecoder},
{"ECOCOVOEncoder", SHOGUN_BASIC_CLASS __new_CECOCOVOEncoder},
{"ECOCOVREncoder", SHOGUN_BASIC_CLASS __new_CECOCOVREncoder},
{"ECOCRandomDenseEncoder", SHOGUN_BASIC_CLASS __new_CECOCRandomDenseEncoder},
{"ECOCRandomSparseEncoder", SHOGUN_BASIC_CLASS __new_CECOCRandomSparseEncoder},
{"ECOCStrategy", SHOGUN_BASIC_CLASS __new_CECOCStrategy},
{"GaussianNaiveBayes", SHOGUN_BASIC_CLASS __new_CGaussianNaiveBayes},
{"GMNPLib", SHOGUN_BASIC_CLASS __new_CGMNPLib},
{"GMNPSVM", SHOGUN_BASIC_CLASS __new_CGMNPSVM},
{"KNN", SHOGUN_BASIC_CLASS __new_CKNN},
{"LaRank", SHOGUN_BASIC_CLASS __new_CLaRank},
{"MulticlassLibLinear", SHOGUN_BASIC_CLASS __new_CMulticlassLibLinear},
{"MulticlassLibSVM", SHOGUN_BASIC_CLASS __new_CMulticlassLibSVM},
{"MulticlassOCAS", SHOGUN_BASIC_CLASS __new_CMulticlassOCAS},
{"MulticlassOneVsOneStrategy", SHOGUN_BASIC_CLASS __new_CMulticlassOneVsOneStrategy},
{"MulticlassOneVsRestStrategy", SHOGUN_BASIC_CLASS __new_CMulticlassOneVsRestStrategy},
{"MulticlassSVM", SHOGUN_BASIC_CLASS __new_CMulticlassSVM},
{"ThresholdRejectionStrategy", SHOGUN_BASIC_CLASS __new_CThresholdRejectionStrategy},
{"DixonQTestRejectionStrategy", SHOGUN_BASIC_CLASS __new_CDixonQTestRejectionStrategy},
{"ScatterSVM", SHOGUN_BASIC_CLASS __new_CScatterSVM},
{"ShareBoost", SHOGUN_BASIC_CLASS __new_CShareBoost},
{"BalancedConditionalProbabilityTree", SHOGUN_BASIC_CLASS __new_CBalancedConditionalProbabilityTree},
{"RandomConditionalProbabilityTree", SHOGUN_BASIC_CLASS __new_CRandomConditionalProbabilityTree},
{"RelaxedTree", SHOGUN_BASIC_CLASS __new_CRelaxedTree},
{"Tron", SHOGUN_BASIC_CLASS __new_CTron},
{"DimensionReductionPreprocessor", SHOGUN_BASIC_CLASS __new_CDimensionReductionPreprocessor},
{"HomogeneousKernelMap", SHOGUN_BASIC_CLASS __new_CHomogeneousKernelMap},
{"LogPlusOne", SHOGUN_BASIC_CLASS __new_CLogPlusOne},
{"NormOne", SHOGUN_BASIC_CLASS __new_CNormOne},
{"PNorm", SHOGUN_BASIC_CLASS __new_CPNorm},
{"PruneVarSubMean", SHOGUN_BASIC_CLASS __new_CPruneVarSubMean},
{"RandomFourierGaussPreproc", SHOGUN_BASIC_CLASS __new_CRandomFourierGaussPreproc},
{"RescaleFeatures", SHOGUN_BASIC_CLASS __new_CRescaleFeatures},
{"SortUlongString", SHOGUN_BASIC_CLASS __new_CSortUlongString},
{"SortWordString", SHOGUN_BASIC_CLASS __new_CSortWordString},
{"SumOne", SHOGUN_BASIC_CLASS __new_CSumOne},
{"LibSVR", SHOGUN_BASIC_CLASS __new_CLibSVR},
{"MKLRegression", SHOGUN_BASIC_CLASS __new_CMKLRegression},
{"SVRLight", SHOGUN_BASIC_CLASS __new_CSVRLight},
{"HSIC", SHOGUN_BASIC_CLASS __new_CHSIC},
{"KernelMeanMatching", SHOGUN_BASIC_CLASS __new_CKernelMeanMatching},
{"LinearTimeMMD", SHOGUN_BASIC_CLASS __new_CLinearTimeMMD},
{"MMDKernelSelectionCombMaxL2", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionCombMaxL2},
{"MMDKernelSelectionCombOpt", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionCombOpt},
{"MMDKernelSelectionMax", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionMax},
{"MMDKernelSelectionMedian", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionMedian},
{"MMDKernelSelectionOpt", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionOpt},
{"QuadraticTimeMMD", SHOGUN_BASIC_CLASS __new_CQuadraticTimeMMD},
{"CCSOSVM", SHOGUN_BASIC_CLASS __new_CCCSOSVM},
{"DisjointSet", SHOGUN_BASIC_CLASS __new_CDisjointSet},
{"DualLibQPBMSOSVM", SHOGUN_BASIC_CLASS __new_CDualLibQPBMSOSVM},
{"DynProg", SHOGUN_BASIC_CLASS __new_CDynProg},
{"FactorDataSource", SHOGUN_BASIC_CLASS __new_CFactorDataSource},
{"Factor", SHOGUN_BASIC_CLASS __new_CFactor},
{"FactorGraph", SHOGUN_BASIC_CLASS __new_CFactorGraph},
{"FactorGraphModel", SHOGUN_BASIC_CLASS __new_CFactorGraphModel},
{"FactorType", SHOGUN_BASIC_CLASS __new_CFactorType},
{"TableFactorType", SHOGUN_BASIC_CLASS __new_CTableFactorType},
{"HMSVMModel", SHOGUN_BASIC_CLASS __new_CHMSVMModel},
{"IntronList", SHOGUN_BASIC_CLASS __new_CIntronList},
{"MAPInference", SHOGUN_BASIC_CLASS __new_CMAPInference},
{"MulticlassModel", SHOGUN_BASIC_CLASS __new_CMulticlassModel},
{"MulticlassSOLabels", SHOGUN_BASIC_CLASS __new_CMulticlassSOLabels},
{"Plif", SHOGUN_BASIC_CLASS __new_CPlif},
{"PlifArray", SHOGUN_BASIC_CLASS __new_CPlifArray},
{"PlifMatrix", SHOGUN_BASIC_CLASS __new_CPlifMatrix},
{"SegmentLoss", SHOGUN_BASIC_CLASS __new_CSegmentLoss},
{"Sequence", SHOGUN_BASIC_CLASS __new_CSequence},
{"SequenceLabels", SHOGUN_BASIC_CLASS __new_CSequenceLabels},
{"SOSVMHelper", SHOGUN_BASIC_CLASS __new_CSOSVMHelper},
{"StochasticSOSVM", SHOGUN_BASIC_CLASS __new_CStochasticSOSVM},
{"TwoStateModel", SHOGUN_BASIC_CLASS __new_CTwoStateModel},
{"DomainAdaptationSVM", SHOGUN_BASIC_CLASS __new_CDomainAdaptationSVM},
{"MultitaskClusteredLogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskClusteredLogisticRegression},
{"MultitaskKernelMaskNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelMaskNormalizer},
{"MultitaskKernelMaskPairNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelMaskPairNormalizer},
{"MultitaskKernelNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelNormalizer},
{"MultitaskKernelPlifNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelPlifNormalizer},
{"Node", SHOGUN_BASIC_CLASS __new_CNode},
{"Taxonomy", SHOGUN_BASIC_CLASS __new_CTaxonomy},
{"MultitaskKernelTreeNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelTreeNormalizer},
{"MultitaskL12LogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskL12LogisticRegression},
{"MultitaskLeastSquaresRegression", SHOGUN_BASIC_CLASS __new_CMultitaskLeastSquaresRegression},
{"MultitaskLinearMachine", SHOGUN_BASIC_CLASS __new_CMultitaskLinearMachine},
{"MultitaskLogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskLogisticRegression},
{"MultitaskROCEvaluation", SHOGUN_BASIC_CLASS __new_CMultitaskROCEvaluation},
{"MultitaskTraceLogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskTraceLogisticRegression},
{"Task", SHOGUN_BASIC_CLASS __new_CTask},
{"TaskGroup", SHOGUN_BASIC_CLASS __new_CTaskGroup},
{"TaskTree", SHOGUN_BASIC_CLASS __new_CTaskTree},
{"GUIClassifier", SHOGUN_BASIC_CLASS __new_CGUIClassifier},
{"GUIConverter", SHOGUN_BASIC_CLASS __new_CGUIConverter},
{"GUIDistance", SHOGUN_BASIC_CLASS __new_CGUIDistance},
{"GUIFeatures", SHOGUN_BASIC_CLASS __new_CGUIFeatures},
{"GUIHMM", SHOGUN_BASIC_CLASS __new_CGUIHMM},
{"GUIKernel", SHOGUN_BASIC_CLASS __new_CGUIKernel},
{"GUILabels", SHOGUN_BASIC_CLASS __new_CGUILabels},
{"GUIMath", SHOGUN_BASIC_CLASS __new_CGUIMath},
{"GUIPluginEstimate", SHOGUN_BASIC_CLASS __new_CGUIPluginEstimate},
{"GUIPreprocessor", SHOGUN_BASIC_CLASS __new_CGUIPreprocessor},
{"GUIStructure", SHOGUN_BASIC_CLASS __new_CGUIStructure},
{"GUITime", SHOGUN_BASIC_CLASS __new_CGUITime},
{"AveragedPerceptron", SHOGUN_BASIC_CLASS __new_CAveragedPerceptron},
{"FeatureBlockLogisticRegression", SHOGUN_BASIC_CLASS __new_CFeatureBlockLogisticRegression},
{"MKLClassification", SHOGUN_BASIC_CLASS __new_CMKLClassification},
{"MKLMulticlass", SHOGUN_BASIC_CLASS __new_CMKLMulticlass},
{"MKLOneClass", SHOGUN_BASIC_CLASS __new_CMKLOneClass},
{"NearestCentroid", SHOGUN_BASIC_CLASS __new_CNearestCentroid},
{"Perceptron", SHOGUN_BASIC_CLASS __new_CPerceptron},
{"PluginEstimate", SHOGUN_BASIC_CLASS __new_CPluginEstimate},
{"GNPPLib", SHOGUN_BASIC_CLASS __new_CGNPPLib},
{"GNPPSVM", SHOGUN_BASIC_CLASS __new_CGNPPSVM},
{"GPBTSVM", SHOGUN_BASIC_CLASS __new_CGPBTSVM},
{"LibLinear", SHOGUN_BASIC_CLASS __new_CLibLinear},
{"LibSVM", SHOGUN_BASIC_CLASS __new_CLibSVM},
{"LibSVMOneClass", SHOGUN_BASIC_CLASS __new_CLibSVMOneClass},
{"MPDSVM", SHOGUN_BASIC_CLASS __new_CMPDSVM},
{"OnlineLibLinear", SHOGUN_BASIC_CLASS __new_COnlineLibLinear},
{"OnlineSVMSGD", SHOGUN_BASIC_CLASS __new_COnlineSVMSGD},
{"QPBSVMLib", SHOGUN_BASIC_CLASS __new_CQPBSVMLib},
{"SGDQN", SHOGUN_BASIC_CLASS __new_CSGDQN},
{"SVM", SHOGUN_BASIC_CLASS __new_CSVM},
{"SVMLight", SHOGUN_BASIC_CLASS __new_CSVMLight},
{"SVMLightOneClass", SHOGUN_BASIC_CLASS __new_CSVMLightOneClass},
{"SVMLin", SHOGUN_BASIC_CLASS __new_CSVMLin},
{"SVMOcas", SHOGUN_BASIC_CLASS __new_CSVMOcas},
{"SVMSGD", SHOGUN_BASIC_CLASS __new_CSVMSGD},
{"WDSVMOcas", SHOGUN_BASIC_CLASS __new_CWDSVMOcas},
{"VwNativeCacheReader", SHOGUN_BASIC_CLASS __new_CVwNativeCacheReader},
{"VwNativeCacheWriter", SHOGUN_BASIC_CLASS __new_CVwNativeCacheWriter},
{"VwAdaptiveLearner", SHOGUN_BASIC_CLASS __new_CVwAdaptiveLearner},
{"VwNonAdaptiveLearner", SHOGUN_BASIC_CLASS __new_CVwNonAdaptiveLearner},
{"VowpalWabbit", SHOGUN_BASIC_CLASS __new_CVowpalWabbit},
{"VwEnvironment", SHOGUN_BASIC_CLASS __new_CVwEnvironment},
{"VwParser", SHOGUN_BASIC_CLASS __new_CVwParser},
{"VwRegressor", SHOGUN_BASIC_CLASS __new_CVwRegressor},
{"Hierarchical", SHOGUN_BASIC_CLASS __new_CHierarchical},
{"KMeans", SHOGUN_BASIC_CLASS __new_CKMeans},
{"HashedDocConverter", SHOGUN_BASIC_CLASS __new_CHashedDocConverter},
{"AttenuatedEuclideanDistance", SHOGUN_BASIC_CLASS __new_CAttenuatedEuclideanDistance},
{"BrayCurtisDistance", SHOGUN_BASIC_CLASS __new_CBrayCurtisDistance},
{"CanberraMetric", SHOGUN_BASIC_CLASS __new_CCanberraMetric},
{"CanberraWordDistance", SHOGUN_BASIC_CLASS __new_CCanberraWordDistance},
{"ChebyshewMetric", SHOGUN_BASIC_CLASS __new_CChebyshewMetric},
{"ChiSquareDistance", SHOGUN_BASIC_CLASS __new_CChiSquareDistance},
{"CosineDistance", SHOGUN_BASIC_CLASS __new_CCosineDistance},
{"CustomDistance", SHOGUN_BASIC_CLASS __new_CCustomDistance},
{"EuclideanDistance", SHOGUN_BASIC_CLASS __new_CEuclideanDistance},
{"GeodesicMetric", SHOGUN_BASIC_CLASS __new_CGeodesicMetric},
{"HammingWordDistance", SHOGUN_BASIC_CLASS __new_CHammingWordDistance},
{"JensenMetric", SHOGUN_BASIC_CLASS __new_CJensenMetric},
{"KernelDistance", SHOGUN_BASIC_CLASS __new_CKernelDistance},
{"ManhattanMetric", SHOGUN_BASIC_CLASS __new_CManhattanMetric},
{"ManhattanWordDistance", SHOGUN_BASIC_CLASS __new_CManhattanWordDistance},
{"MinkowskiMetric", SHOGUN_BASIC_CLASS __new_CMinkowskiMetric},
{"SparseEuclideanDistance", SHOGUN_BASIC_CLASS __new_CSparseEuclideanDistance},
{"TanimotoDistance", SHOGUN_BASIC_CLASS __new_CTanimotoDistance},
{"GHMM", SHOGUN_BASIC_CLASS __new_CGHMM},
{"Histogram", SHOGUN_BASIC_CLASS __new_CHistogram},
{"HMM", SHOGUN_BASIC_CLASS __new_CHMM},
{"LinearHMM", SHOGUN_BASIC_CLASS __new_CLinearHMM},
{"PositionalPWM", SHOGUN_BASIC_CLASS __new_CPositionalPWM},
{"MajorityVote", SHOGUN_BASIC_CLASS __new_CMajorityVote},
{"MeanRule", SHOGUN_BASIC_CLASS __new_CMeanRule},
{"WeightedMajorityVote", SHOGUN_BASIC_CLASS __new_CWeightedMajorityVote},
{"ClusteringAccuracy", SHOGUN_BASIC_CLASS __new_CClusteringAccuracy},
{"ClusteringMutualInformation", SHOGUN_BASIC_CLASS __new_CClusteringMutualInformation},
{"ContingencyTableEvaluation", SHOGUN_BASIC_CLASS __new_CContingencyTableEvaluation},
{"AccuracyMeasure", SHOGUN_BASIC_CLASS __new_CAccuracyMeasure},
{"ErrorRateMeasure", SHOGUN_BASIC_CLASS __new_CErrorRateMeasure},
{"BALMeasure", SHOGUN_BASIC_CLASS __new_CBALMeasure},
{"WRACCMeasure", SHOGUN_BASIC_CLASS __new_CWRACCMeasure},
{"F1Measure", SHOGUN_BASIC_CLASS __new_CF1Measure},
{"CrossCorrelationMeasure", SHOGUN_BASIC_CLASS __new_CCrossCorrelationMeasure},
{"RecallMeasure", SHOGUN_BASIC_CLASS __new_CRecallMeasure},
{"PrecisionMeasure", SHOGUN_BASIC_CLASS __new_CPrecisionMeasure},
{"SpecificityMeasure", SHOGUN_BASIC_CLASS __new_CSpecificityMeasure},
{"CrossValidationResult", SHOGUN_BASIC_CLASS __new_CCrossValidationResult},
{"CrossValidation", SHOGUN_BASIC_CLASS __new_CCrossValidation},
{"CrossValidationMKLStorage", SHOGUN_BASIC_CLASS __new_CCrossValidationMKLStorage},
{"CrossValidationMulticlassStorage", SHOGUN_BASIC_CLASS __new_CCrossValidationMulticlassStorage},
{"CrossValidationPrintOutput", SHOGUN_BASIC_CLASS __new_CCrossValidationPrintOutput},
{"CrossValidationSplitting", SHOGUN_BASIC_CLASS __new_CCrossValidationSplitting},
{"GradientCriterion", SHOGUN_BASIC_CLASS __new_CGradientCriterion},
{"GradientEvaluation", SHOGUN_BASIC_CLASS __new_CGradientEvaluation},
{"GradientResult", SHOGUN_BASIC_CLASS __new_CGradientResult},
{"MeanAbsoluteError", SHOGUN_BASIC_CLASS __new_CMeanAbsoluteError},
{"MeanSquaredError", SHOGUN_BASIC_CLASS __new_CMeanSquaredError},
{"MeanSquaredLogError", SHOGUN_BASIC_CLASS __new_CMeanSquaredLogError},
{"MulticlassAccuracy", SHOGUN_BASIC_CLASS __new_CMulticlassAccuracy},
{"MulticlassOVREvaluation", SHOGUN_BASIC_CLASS __new_CMulticlassOVREvaluation},
{"PRCEvaluation", SHOGUN_BASIC_CLASS __new_CPRCEvaluation},
{"ROCEvaluation", SHOGUN_BASIC_CLASS __new_CROCEvaluation},
{"StratifiedCrossValidationSplitting", SHOGUN_BASIC_CLASS __new_CStratifiedCrossValidationSplitting},
{"StructuredAccuracy", SHOGUN_BASIC_CLASS __new_CStructuredAccuracy},
{"Alphabet", SHOGUN_BASIC_CLASS __new_CAlphabet},
{"BinnedDotFeatures", SHOGUN_BASIC_CLASS __new_CBinnedDotFeatures},
{"CombinedDotFeatures", SHOGUN_BASIC_CLASS __new_CCombinedDotFeatures},
{"CombinedFeatures", SHOGUN_BASIC_CLASS __new_CCombinedFeatures},
{"DataGenerator", SHOGUN_BASIC_CLASS __new_CDataGenerator},
{"DummyFeatures", SHOGUN_BASIC_CLASS __new_CDummyFeatures},
{"ExplicitSpecFeatures", SHOGUN_BASIC_CLASS __new_CExplicitSpecFeatures},
{"FactorGraphFeatures", SHOGUN_BASIC_CLASS __new_CFactorGraphFeatures},
{"FKFeatures", SHOGUN_BASIC_CLASS __new_CFKFeatures},
{"HashedDocDotFeatures", SHOGUN_BASIC_CLASS __new_CHashedDocDotFeatures},
{"HashedWDFeatures", SHOGUN_BASIC_CLASS __new_CHashedWDFeatures},
{"HashedWDFeaturesTransposed", SHOGUN_BASIC_CLASS __new_CHashedWDFeaturesTransposed},
{"ImplicitWeightedSpecFeatures", SHOGUN_BASIC_CLASS __new_CImplicitWeightedSpecFeatures},
{"LatentFeatures", SHOGUN_BASIC_CLASS __new_CLatentFeatures},
{"LBPPyrDotFeatures", SHOGUN_BASIC_CLASS __new_CLBPPyrDotFeatures},
{"PolyFeatures", SHOGUN_BASIC_CLASS __new_CPolyFeatures},
{"RandomFourierDotFeatures", SHOGUN_BASIC_CLASS __new_CRandomFourierDotFeatures},
{"RealFileFeatures", SHOGUN_BASIC_CLASS __new_CRealFileFeatures},
{"SNPFeatures", SHOGUN_BASIC_CLASS __new_CSNPFeatures},
{"SparsePolyFeatures", SHOGUN_BASIC_CLASS __new_CSparsePolyFeatures},
{"GaussianBlobsDataGenerator", SHOGUN_BASIC_CLASS __new_CGaussianBlobsDataGenerator},
{"MeanShiftDataGenerator", SHOGUN_BASIC_CLASS __new_CMeanShiftDataGenerator},
{"StreamingHashedDocDotFeatures", SHOGUN_BASIC_CLASS __new_CStreamingHashedDocDotFeatures},
{"StreamingVwFeatures", SHOGUN_BASIC_CLASS __new_CStreamingVwFeatures},
{"Subset", SHOGUN_BASIC_CLASS __new_CSubset},
{"SubsetStack", SHOGUN_BASIC_CLASS __new_CSubsetStack},
{"TOPFeatures", SHOGUN_BASIC_CLASS __new_CTOPFeatures},
{"WDFeatures", SHOGUN_BASIC_CLASS __new_CWDFeatures},
{"BinaryFile", SHOGUN_BASIC_CLASS __new_CBinaryFile},
{"CSVFile", SHOGUN_BASIC_CLASS __new_CCSVFile},
{"IOBuffer", SHOGUN_BASIC_CLASS __new_CIOBuffer},
{"LibSVMFile", SHOGUN_BASIC_CLASS __new_CLibSVMFile},
{"LineReader", SHOGUN_BASIC_CLASS __new_CLineReader},
{"Parser", SHOGUN_BASIC_CLASS __new_CParser},
{"SerializableAsciiFile", SHOGUN_BASIC_CLASS __new_CSerializableAsciiFile},
{"StreamingAsciiFile", SHOGUN_BASIC_CLASS __new_CStreamingAsciiFile},
{"StreamingFile", SHOGUN_BASIC_CLASS __new_CStreamingFile},
{"StreamingFileFromFeatures", SHOGUN_BASIC_CLASS __new_CStreamingFileFromFeatures},
{"StreamingVwCacheFile", SHOGUN_BASIC_CLASS __new_CStreamingVwCacheFile},
{"StreamingVwFile", SHOGUN_BASIC_CLASS __new_CStreamingVwFile},
{"ANOVAKernel", SHOGUN_BASIC_CLASS __new_CANOVAKernel},
{"AUCKernel", SHOGUN_BASIC_CLASS __new_CAUCKernel},
{"BesselKernel", SHOGUN_BASIC_CLASS __new_CBesselKernel},
{"CauchyKernel", SHOGUN_BASIC_CLASS __new_CCauchyKernel},
{"Chi2Kernel", SHOGUN_BASIC_CLASS __new_CChi2Kernel},
{"CircularKernel", SHOGUN_BASIC_CLASS __new_CCircularKernel},
{"CombinedKernel", SHOGUN_BASIC_CLASS __new_CCombinedKernel},
{"ConstKernel", SHOGUN_BASIC_CLASS __new_CConstKernel},
{"CustomKernel", SHOGUN_BASIC_CLASS __new_CCustomKernel},
{"DiagKernel", SHOGUN_BASIC_CLASS __new_CDiagKernel},
{"DistanceKernel", SHOGUN_BASIC_CLASS __new_CDistanceKernel},
{"ExponentialKernel", SHOGUN_BASIC_CLASS __new_CExponentialKernel},
{"GaussianARDKernel", SHOGUN_BASIC_CLASS __new_CGaussianARDKernel},
{"GaussianKernel", SHOGUN_BASIC_CLASS __new_CGaussianKernel},
{"GaussianShiftKernel", SHOGUN_BASIC_CLASS __new_CGaussianShiftKernel},
{"GaussianShortRealKernel", SHOGUN_BASIC_CLASS __new_CGaussianShortRealKernel},
{"HistogramIntersectionKernel", SHOGUN_BASIC_CLASS __new_CHistogramIntersectionKernel},
{"InverseMultiQuadricKernel", SHOGUN_BASIC_CLASS __new_CInverseMultiQuadricKernel},
{"JensenShannonKernel", SHOGUN_BASIC_CLASS __new_CJensenShannonKernel},
{"LinearARDKernel", SHOGUN_BASIC_CLASS __new_CLinearARDKernel},
{"LinearKernel", SHOGUN_BASIC_CLASS __new_CLinearKernel},
{"LogKernel", SHOGUN_BASIC_CLASS __new_CLogKernel},
{"MultiquadricKernel", SHOGUN_BASIC_CLASS __new_CMultiquadricKernel},
{"AvgDiagKernelNormalizer", SHOGUN_BASIC_CLASS __new_CAvgDiagKernelNormalizer},
{"DiceKernelNormalizer", SHOGUN_BASIC_CLASS __new_CDiceKernelNormalizer},
{"FirstElementKernelNormalizer", SHOGUN_BASIC_CLASS __new_CFirstElementKernelNormalizer},
{"IdentityKernelNormalizer", SHOGUN_BASIC_CLASS __new_CIdentityKernelNormalizer},
{"RidgeKernelNormalizer", SHOGUN_BASIC_CLASS __new_CRidgeKernelNormalizer},
{"ScatterKernelNormalizer", SHOGUN_BASIC_CLASS __new_CScatterKernelNormalizer},
{"SqrtDiagKernelNormalizer", SHOGUN_BASIC_CLASS __new_CSqrtDiagKernelNormalizer},
{"TanimotoKernelNormalizer", SHOGUN_BASIC_CLASS __new_CTanimotoKernelNormalizer},
{"VarianceKernelNormalizer", SHOGUN_BASIC_CLASS __new_CVarianceKernelNormalizer},
{"ZeroMeanCenterKernelNormalizer", SHOGUN_BASIC_CLASS __new_CZeroMeanCenterKernelNormalizer},
{"PolyKernel", SHOGUN_BASIC_CLASS __new_CPolyKernel},
{"PowerKernel", SHOGUN_BASIC_CLASS __new_CPowerKernel},
{"ProductKernel", SHOGUN_BASIC_CLASS __new_CProductKernel},
{"PyramidChi2", SHOGUN_BASIC_CLASS __new_CPyramidChi2},
{"RationalQuadraticKernel", SHOGUN_BASIC_CLASS __new_CRationalQuadraticKernel},
{"SigmoidKernel", SHOGUN_BASIC_CLASS __new_CSigmoidKernel},
{"SphericalKernel", SHOGUN_BASIC_CLASS __new_CSphericalKernel},
{"SplineKernel", SHOGUN_BASIC_CLASS __new_CSplineKernel},
{"CommUlongStringKernel", SHOGUN_BASIC_CLASS __new_CCommUlongStringKernel},
{"CommWordStringKernel", SHOGUN_BASIC_CLASS __new_CCommWordStringKernel},
{"DistantSegmentsKernel", SHOGUN_BASIC_CLASS __new_CDistantSegmentsKernel},
{"FixedDegreeStringKernel", SHOGUN_BASIC_CLASS __new_CFixedDegreeStringKernel},
{"GaussianMatchStringKernel", SHOGUN_BASIC_CLASS __new_CGaussianMatchStringKernel},
{"HistogramWordStringKernel", SHOGUN_BASIC_CLASS __new_CHistogramWordStringKernel},
{"LinearStringKernel", SHOGUN_BASIC_CLASS __new_CLinearStringKernel},
{"LocalAlignmentStringKernel", SHOGUN_BASIC_CLASS __new_CLocalAlignmentStringKernel},
{"LocalityImprovedStringKernel", SHOGUN_BASIC_CLASS __new_CLocalityImprovedStringKernel},
{"MatchWordStringKernel", SHOGUN_BASIC_CLASS __new_CMatchWordStringKernel},
{"OligoStringKernel", SHOGUN_BASIC_CLASS __new_COligoStringKernel},
{"PolyMatchStringKernel", SHOGUN_BASIC_CLASS __new_CPolyMatchStringKernel},
{"PolyMatchWordStringKernel", SHOGUN_BASIC_CLASS __new_CPolyMatchWordStringKernel},
{"RegulatoryModulesStringKernel", SHOGUN_BASIC_CLASS __new_CRegulatoryModulesStringKernel},
{"SalzbergWordStringKernel", SHOGUN_BASIC_CLASS __new_CSalzbergWordStringKernel},
{"SimpleLocalityImprovedStringKernel", SHOGUN_BASIC_CLASS __new_CSimpleLocalityImprovedStringKernel},
{"SNPStringKernel", SHOGUN_BASIC_CLASS __new_CSNPStringKernel},
{"SparseSpatialSampleStringKernel", SHOGUN_BASIC_CLASS __new_CSparseSpatialSampleStringKernel},
{"SpectrumMismatchRBFKernel", SHOGUN_BASIC_CLASS __new_CSpectrumMismatchRBFKernel},
{"SpectrumRBFKernel", SHOGUN_BASIC_CLASS __new_CSpectrumRBFKernel},
{"WeightedCommWordStringKernel", SHOGUN_BASIC_CLASS __new_CWeightedCommWordStringKernel},
{"WeightedDegreePositionStringKernel", SHOGUN_BASIC_CLASS __new_CWeightedDegreePositionStringKernel},
{"WeightedDegreeStringKernel", SHOGUN_BASIC_CLASS __new_CWeightedDegreeStringKernel},
{"TensorProductPairKernel", SHOGUN_BASIC_CLASS __new_CTensorProductPairKernel},
{"TStudentKernel", SHOGUN_BASIC_CLASS __new_CTStudentKernel},
{"WaveKernel", SHOGUN_BASIC_CLASS __new_CWaveKernel},
{"WaveletKernel", SHOGUN_BASIC_CLASS __new_CWaveletKernel},
{"WeightedDegreeRBFKernel", SHOGUN_BASIC_CLASS __new_CWeightedDegreeRBFKernel},
{"BinaryLabels", SHOGUN_BASIC_CLASS __new_CBinaryLabels},
{"FactorGraphObservation", SHOGUN_BASIC_CLASS __new_CFactorGraphObservation},
{"FactorGraphLabels", SHOGUN_BASIC_CLASS __new_CFactorGraphLabels},
{"LabelsFactory", SHOGUN_BASIC_CLASS __new_CLabelsFactory},
{"LatentLabels", SHOGUN_BASIC_CLASS __new_CLatentLabels},
{"MulticlassLabels", SHOGUN_BASIC_CLASS __new_CMulticlassLabels},
{"MulticlassMultipleOutputLabels", SHOGUN_BASIC_CLASS __new_CMulticlassMultipleOutputLabels},
{"RegressionLabels", SHOGUN_BASIC_CLASS __new_CRegressionLabels},
{"StructuredLabels", SHOGUN_BASIC_CLASS __new_CStructuredLabels},
{"LatentSOSVM", SHOGUN_BASIC_CLASS __new_CLatentSOSVM},
{"LatentSVM", SHOGUN_BASIC_CLASS __new_CLatentSVM},
{"BitString", SHOGUN_BASIC_CLASS __new_CBitString},
{"CircularBuffer", SHOGUN_BASIC_CLASS __new_CCircularBuffer},
{"Compressor", SHOGUN_BASIC_CLASS __new_CCompressor},
{"SerialComputationEngine", SHOGUN_BASIC_CLASS __new_CSerialComputationEngine},
{"JobResult", SHOGUN_BASIC_CLASS __new_CJobResult},
{"Data", SHOGUN_BASIC_CLASS __new_CData},
{"DelimiterTokenizer", SHOGUN_BASIC_CLASS __new_CDelimiterTokenizer},
{"DynamicObjectArray", SHOGUN_BASIC_CLASS __new_CDynamicObjectArray},
{"Hash", SHOGUN_BASIC_CLASS __new_CHash},
{"IndexBlock", SHOGUN_BASIC_CLASS __new_CIndexBlock},
{"IndexBlockGroup", SHOGUN_BASIC_CLASS __new_CIndexBlockGroup},
{"IndexBlockTree", SHOGUN_BASIC_CLASS __new_CIndexBlockTree},
{"ListElement", SHOGUN_BASIC_CLASS __new_CListElement},
{"List", SHOGUN_BASIC_CLASS __new_CList},
{"NGramTokenizer", SHOGUN_BASIC_CLASS __new_CNGramTokenizer},
{"Signal", SHOGUN_BASIC_CLASS __new_CSignal},
{"StructuredData", SHOGUN_BASIC_CLASS __new_CStructuredData},
{"Time", SHOGUN_BASIC_CLASS __new_CTime},
{"HingeLoss", SHOGUN_BASIC_CLASS __new_CHingeLoss},
{"LogLoss", SHOGUN_BASIC_CLASS __new_CLogLoss},
{"LogLossMargin", SHOGUN_BASIC_CLASS __new_CLogLossMargin},
{"SmoothHingeLoss", SHOGUN_BASIC_CLASS __new_CSmoothHingeLoss},
{"SquaredHingeLoss", SHOGUN_BASIC_CLASS __new_CSquaredHingeLoss},
{"SquaredLoss", SHOGUN_BASIC_CLASS __new_CSquaredLoss},
{"BaggingMachine", SHOGUN_BASIC_CLASS __new_CBaggingMachine},
{"BaseMulticlassMachine", SHOGUN_BASIC_CLASS __new_CBaseMulticlassMachine},
{"DistanceMachine", SHOGUN_BASIC_CLASS __new_CDistanceMachine},
{"ZeroMean", SHOGUN_BASIC_CLASS __new_CZeroMean},
{"KernelMachine", SHOGUN_BASIC_CLASS __new_CKernelMachine},
{"KernelMulticlassMachine", SHOGUN_BASIC_CLASS __new_CKernelMulticlassMachine},
{"KernelStructuredOutputMachine", SHOGUN_BASIC_CLASS __new_CKernelStructuredOutputMachine},
{"LinearMachine", SHOGUN_BASIC_CLASS __new_CLinearMachine},
{"LinearMulticlassMachine", SHOGUN_BASIC_CLASS __new_CLinearMulticlassMachine},
{"LinearStructuredOutputMachine", SHOGUN_BASIC_CLASS __new_CLinearStructuredOutputMachine},
{"Machine", SHOGUN_BASIC_CLASS __new_CMachine},
{"NativeMulticlassMachine", SHOGUN_BASIC_CLASS __new_CNativeMulticlassMachine},
{"OnlineLinearMachine", SHOGUN_BASIC_CLASS __new_COnlineLinearMachine},
{"StructuredOutputMachine", SHOGUN_BASIC_CLASS __new_CStructuredOutputMachine},
{"JacobiEllipticFunctions", SHOGUN_BASIC_CLASS __new_CJacobiEllipticFunctions},
{"LogDetEstimator", SHOGUN_BASIC_CLASS __new_CLogDetEstimator},
{"NormalSampler", SHOGUN_BASIC_CLASS __new_CNormalSampler},
{"Math", SHOGUN_BASIC_CLASS __new_CMath},
{"Random", SHOGUN_BASIC_CLASS __new_CRandom},
{"SparseInverseCovariance", SHOGUN_BASIC_CLASS __new_CSparseInverseCovariance},
{"Statistics", SHOGUN_BASIC_CLASS __new_CStatistics},
{"GridSearchModelSelection", SHOGUN_BASIC_CLASS __new_CGridSearchModelSelection},
{"ModelSelectionParameters", SHOGUN_BASIC_CLASS __new_CModelSelectionParameters},
{"ParameterCombination", SHOGUN_BASIC_CLASS __new_CParameterCombination},
{"RandomSearchModelSelection", SHOGUN_BASIC_CLASS __new_CRandomSearchModelSelection},
{"ECOCAEDDecoder", SHOGUN_BASIC_CLASS __new_CECOCAEDDecoder},
{"ECOCDiscriminantEncoder", SHOGUN_BASIC_CLASS __new_CECOCDiscriminantEncoder},
{"ECOCEDDecoder", SHOGUN_BASIC_CLASS __new_CECOCEDDecoder},
{"ECOCForestEncoder", SHOGUN_BASIC_CLASS __new_CECOCForestEncoder},
{"ECOCHDDecoder", SHOGUN_BASIC_CLASS __new_CECOCHDDecoder},
{"ECOCLLBDecoder", SHOGUN_BASIC_CLASS __new_CECOCLLBDecoder},
{"ECOCOVOEncoder", SHOGUN_BASIC_CLASS __new_CECOCOVOEncoder},
{"ECOCOVREncoder", SHOGUN_BASIC_CLASS __new_CECOCOVREncoder},
{"ECOCRandomDenseEncoder", SHOGUN_BASIC_CLASS __new_CECOCRandomDenseEncoder},
{"ECOCRandomSparseEncoder", SHOGUN_BASIC_CLASS __new_CECOCRandomSparseEncoder},
{"ECOCStrategy", SHOGUN_BASIC_CLASS __new_CECOCStrategy},
{"GaussianNaiveBayes", SHOGUN_BASIC_CLASS __new_CGaussianNaiveBayes},
{"GMNPLib", SHOGUN_BASIC_CLASS __new_CGMNPLib},
{"GMNPSVM", SHOGUN_BASIC_CLASS __new_CGMNPSVM},
{"KNN", SHOGUN_BASIC_CLASS __new_CKNN},
{"LaRank", SHOGUN_BASIC_CLASS __new_CLaRank},
{"MulticlassLibLinear", SHOGUN_BASIC_CLASS __new_CMulticlassLibLinear},
{"MulticlassLibSVM", SHOGUN_BASIC_CLASS __new_CMulticlassLibSVM},
{"MulticlassOCAS", SHOGUN_BASIC_CLASS __new_CMulticlassOCAS},
{"MulticlassOneVsOneStrategy", SHOGUN_BASIC_CLASS __new_CMulticlassOneVsOneStrategy},
{"MulticlassOneVsRestStrategy", SHOGUN_BASIC_CLASS __new_CMulticlassOneVsRestStrategy},
{"MulticlassSVM", SHOGUN_BASIC_CLASS __new_CMulticlassSVM},
{"ThresholdRejectionStrategy", SHOGUN_BASIC_CLASS __new_CThresholdRejectionStrategy},
{"DixonQTestRejectionStrategy", SHOGUN_BASIC_CLASS __new_CDixonQTestRejectionStrategy},
{"ScatterSVM", SHOGUN_BASIC_CLASS __new_CScatterSVM},
{"ShareBoost", SHOGUN_BASIC_CLASS __new_CShareBoost},
{"BalancedConditionalProbabilityTree", SHOGUN_BASIC_CLASS __new_CBalancedConditionalProbabilityTree},
{"RandomConditionalProbabilityTree", SHOGUN_BASIC_CLASS __new_CRandomConditionalProbabilityTree},
{"RelaxedTree", SHOGUN_BASIC_CLASS __new_CRelaxedTree},
{"Tron", SHOGUN_BASIC_CLASS __new_CTron},
{"DimensionReductionPreprocessor", SHOGUN_BASIC_CLASS __new_CDimensionReductionPreprocessor},
{"HomogeneousKernelMap", SHOGUN_BASIC_CLASS __new_CHomogeneousKernelMap},
{"LogPlusOne", SHOGUN_BASIC_CLASS __new_CLogPlusOne},
{"NormOne", SHOGUN_BASIC_CLASS __new_CNormOne},
{"PNorm", SHOGUN_BASIC_CLASS __new_CPNorm},
{"PruneVarSubMean", SHOGUN_BASIC_CLASS __new_CPruneVarSubMean},
{"RandomFourierGaussPreproc", SHOGUN_BASIC_CLASS __new_CRandomFourierGaussPreproc},
{"RescaleFeatures", SHOGUN_BASIC_CLASS __new_CRescaleFeatures},
{"SortUlongString", SHOGUN_BASIC_CLASS __new_CSortUlongString},
{"SortWordString", SHOGUN_BASIC_CLASS __new_CSortWordString},
{"SumOne", SHOGUN_BASIC_CLASS __new_CSumOne},
{"LibSVR", SHOGUN_BASIC_CLASS __new_CLibSVR},
{"MKLRegression", SHOGUN_BASIC_CLASS __new_CMKLRegression},
{"SVRLight", SHOGUN_BASIC_CLASS __new_CSVRLight},
{"HSIC", SHOGUN_BASIC_CLASS __new_CHSIC},
{"KernelMeanMatching", SHOGUN_BASIC_CLASS __new_CKernelMeanMatching},
{"LinearTimeMMD", SHOGUN_BASIC_CLASS __new_CLinearTimeMMD},
{"MMDKernelSelectionCombMaxL2", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionCombMaxL2},
{"MMDKernelSelectionCombOpt", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionCombOpt},
{"MMDKernelSelectionMax", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionMax},
{"MMDKernelSelectionMedian", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionMedian},
{"MMDKernelSelectionOpt", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionOpt},
{"QuadraticTimeMMD", SHOGUN_BASIC_CLASS __new_CQuadraticTimeMMD},
{"CCSOSVM", SHOGUN_BASIC_CLASS __new_CCCSOSVM},
{"DisjointSet", SHOGUN_BASIC_CLASS __new_CDisjointSet},
{"DualLibQPBMSOSVM", SHOGUN_BASIC_CLASS __new_CDualLibQPBMSOSVM},
{"DynProg", SHOGUN_BASIC_CLASS __new_CDynProg},
{"FactorDataSource", SHOGUN_BASIC_CLASS __new_CFactorDataSource},
{"Factor", SHOGUN_BASIC_CLASS __new_CFactor},
{"FactorGraph", SHOGUN_BASIC_CLASS __new_CFactorGraph},
{"FactorGraphModel", SHOGUN_BASIC_CLASS __new_CFactorGraphModel},
{"FactorType", SHOGUN_BASIC_CLASS __new_CFactorType},
{"TableFactorType", SHOGUN_BASIC_CLASS __new_CTableFactorType},
{"HMSVMModel", SHOGUN_BASIC_CLASS __new_CHMSVMModel},
{"IntronList", SHOGUN_BASIC_CLASS __new_CIntronList},
{"MAPInference", SHOGUN_BASIC_CLASS __new_CMAPInference},
{"MulticlassModel", SHOGUN_BASIC_CLASS __new_CMulticlassModel},
{"MulticlassSOLabels", SHOGUN_BASIC_CLASS __new_CMulticlassSOLabels},
{"Plif", SHOGUN_BASIC_CLASS __new_CPlif},
{"PlifArray", SHOGUN_BASIC_CLASS __new_CPlifArray},
{"PlifMatrix", SHOGUN_BASIC_CLASS __new_CPlifMatrix},
{"SegmentLoss", SHOGUN_BASIC_CLASS __new_CSegmentLoss},
{"Sequence", SHOGUN_BASIC_CLASS __new_CSequence},
{"SequenceLabels", SHOGUN_BASIC_CLASS __new_CSequenceLabels},
{"SOSVMHelper", SHOGUN_BASIC_CLASS __new_CSOSVMHelper},
{"StochasticSOSVM", SHOGUN_BASIC_CLASS __new_CStochasticSOSVM},
{"TwoStateModel", SHOGUN_BASIC_CLASS __new_CTwoStateModel},
{"DomainAdaptationSVM", SHOGUN_BASIC_CLASS __new_CDomainAdaptationSVM},
{"MultitaskClusteredLogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskClusteredLogisticRegression},
{"MultitaskKernelMaskNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelMaskNormalizer},
{"MultitaskKernelMaskPairNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelMaskPairNormalizer},
{"MultitaskKernelNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelNormalizer},
{"MultitaskKernelPlifNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelPlifNormalizer},
{"Node", SHOGUN_BASIC_CLASS __new_CNode},
{"Taxonomy", SHOGUN_BASIC_CLASS __new_CTaxonomy},
{"MultitaskKernelTreeNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelTreeNormalizer},
{"MultitaskL12LogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskL12LogisticRegression},
{"MultitaskLeastSquaresRegression", SHOGUN_BASIC_CLASS __new_CMultitaskLeastSquaresRegression},
{"MultitaskLinearMachine", SHOGUN_BASIC_CLASS __new_CMultitaskLinearMachine},
{"MultitaskLogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskLogisticRegression},
{"MultitaskROCEvaluation", SHOGUN_BASIC_CLASS __new_CMultitaskROCEvaluation},
{"MultitaskTraceLogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskTraceLogisticRegression},
{"Task", SHOGUN_BASIC_CLASS __new_CTask},
{"TaskGroup", SHOGUN_BASIC_CLASS __new_CTaskGroup},
{"TaskTree", SHOGUN_BASIC_CLASS __new_CTaskTree},
{"GUIClassifier", SHOGUN_BASIC_CLASS __new_CGUIClassifier},
{"GUIConverter", SHOGUN_BASIC_CLASS __new_CGUIConverter},
{"GUIDistance", SHOGUN_BASIC_CLASS __new_CGUIDistance},
{"GUIFeatures", SHOGUN_BASIC_CLASS __new_CGUIFeatures},
{"GUIHMM", SHOGUN_BASIC_CLASS __new_CGUIHMM},
{"GUIKernel", SHOGUN_BASIC_CLASS __new_CGUIKernel},
{"GUILabels", SHOGUN_BASIC_CLASS __new_CGUILabels},
{"GUIMath", SHOGUN_BASIC_CLASS __new_CGUIMath},
{"GUIPluginEstimate", SHOGUN_BASIC_CLASS __new_CGUIPluginEstimate},
{"GUIPreprocessor", SHOGUN_BASIC_CLASS __new_CGUIPreprocessor},
{"GUIStructure", SHOGUN_BASIC_CLASS __new_CGUIStructure},
{"GUITime", SHOGUN_BASIC_CLASS __new_CGUITime},
{"AveragedPerceptron", SHOGUN_BASIC_CLASS __new_CAveragedPerceptron},
{"FeatureBlockLogisticRegression", SHOGUN_BASIC_CLASS __new_CFeatureBlockLogisticRegression},
{"MKLClassification", SHOGUN_BASIC_CLASS __new_CMKLClassification},
{"MKLMulticlass", SHOGUN_BASIC_CLASS __new_CMKLMulticlass},
{"MKLOneClass", SHOGUN_BASIC_CLASS __new_CMKLOneClass},
{"NearestCentroid", SHOGUN_BASIC_CLASS __new_CNearestCentroid},
{"Perceptron", SHOGUN_BASIC_CLASS __new_CPerceptron},
{"PluginEstimate", SHOGUN_BASIC_CLASS __new_CPluginEstimate},
{"GNPPLib", SHOGUN_BASIC_CLASS __new_CGNPPLib},
{"GNPPSVM", SHOGUN_BASIC_CLASS __new_CGNPPSVM},
{"GPBTSVM", SHOGUN_BASIC_CLASS __new_CGPBTSVM},
{"LibLinear", SHOGUN_BASIC_CLASS __new_CLibLinear},
{"LibSVM", SHOGUN_BASIC_CLASS __new_CLibSVM},
{"LibSVMOneClass", SHOGUN_BASIC_CLASS __new_CLibSVMOneClass},
{"MPDSVM", SHOGUN_BASIC_CLASS __new_CMPDSVM},
{"OnlineLibLinear", SHOGUN_BASIC_CLASS __new_COnlineLibLinear},
{"OnlineSVMSGD", SHOGUN_BASIC_CLASS __new_COnlineSVMSGD},
{"QPBSVMLib", SHOGUN_BASIC_CLASS __new_CQPBSVMLib},
{"SGDQN", SHOGUN_BASIC_CLASS __new_CSGDQN},
{"SVM", SHOGUN_BASIC_CLASS __new_CSVM},
{"SVMLight", SHOGUN_BASIC_CLASS __new_CSVMLight},
{"SVMLightOneClass", SHOGUN_BASIC_CLASS __new_CSVMLightOneClass},
{"SVMLin", SHOGUN_BASIC_CLASS __new_CSVMLin},
{"SVMOcas", SHOGUN_BASIC_CLASS __new_CSVMOcas},
{"SVMSGD", SHOGUN_BASIC_CLASS __new_CSVMSGD},
{"WDSVMOcas", SHOGUN_BASIC_CLASS __new_CWDSVMOcas},
{"VwNativeCacheReader", SHOGUN_BASIC_CLASS __new_CVwNativeCacheReader},
{"VwNativeCacheWriter", SHOGUN_BASIC_CLASS __new_CVwNativeCacheWriter},
{"VwAdaptiveLearner", SHOGUN_BASIC_CLASS __new_CVwAdaptiveLearner},
{"VwNonAdaptiveLearner", SHOGUN_BASIC_CLASS __new_CVwNonAdaptiveLearner},
{"VowpalWabbit", SHOGUN_BASIC_CLASS __new_CVowpalWabbit},
{"VwEnvironment", SHOGUN_BASIC_CLASS __new_CVwEnvironment},
{"VwParser", SHOGUN_BASIC_CLASS __new_CVwParser},
{"VwRegressor", SHOGUN_BASIC_CLASS __new_CVwRegressor},
{"Hierarchical", SHOGUN_BASIC_CLASS __new_CHierarchical},
{"KMeans", SHOGUN_BASIC_CLASS __new_CKMeans},
{"HashedDocConverter", SHOGUN_BASIC_CLASS __new_CHashedDocConverter},
{"AttenuatedEuclideanDistance", SHOGUN_BASIC_CLASS __new_CAttenuatedEuclideanDistance},
{"BrayCurtisDistance", SHOGUN_BASIC_CLASS __new_CBrayCurtisDistance},
{"CanberraMetric", SHOGUN_BASIC_CLASS __new_CCanberraMetric},
{"CanberraWordDistance", SHOGUN_BASIC_CLASS __new_CCanberraWordDistance},
{"ChebyshewMetric", SHOGUN_BASIC_CLASS __new_CChebyshewMetric},
{"ChiSquareDistance", SHOGUN_BASIC_CLASS __new_CChiSquareDistance},
{"CosineDistance", SHOGUN_BASIC_CLASS __new_CCosineDistance},
{"CustomDistance", SHOGUN_BASIC_CLASS __new_CCustomDistance},
{"EuclideanDistance", SHOGUN_BASIC_CLASS __new_CEuclideanDistance},
{"GeodesicMetric", SHOGUN_BASIC_CLASS __new_CGeodesicMetric},
{"HammingWordDistance", SHOGUN_BASIC_CLASS __new_CHammingWordDistance},
{"JensenMetric", SHOGUN_BASIC_CLASS __new_CJensenMetric},
{"KernelDistance", SHOGUN_BASIC_CLASS __new_CKernelDistance},
{"ManhattanMetric", SHOGUN_BASIC_CLASS __new_CManhattanMetric},
{"ManhattanWordDistance", SHOGUN_BASIC_CLASS __new_CManhattanWordDistance},
{"MinkowskiMetric", SHOGUN_BASIC_CLASS __new_CMinkowskiMetric},
{"SparseEuclideanDistance", SHOGUN_BASIC_CLASS __new_CSparseEuclideanDistance},
{"TanimotoDistance", SHOGUN_BASIC_CLASS __new_CTanimotoDistance},
{"GHMM", SHOGUN_BASIC_CLASS __new_CGHMM},
{"Histogram", SHOGUN_BASIC_CLASS __new_CHistogram},
{"HMM", SHOGUN_BASIC_CLASS __new_CHMM},
{"LinearHMM", SHOGUN_BASIC_CLASS __new_CLinearHMM},
{"PositionalPWM", SHOGUN_BASIC_CLASS __new_CPositionalPWM},
{"MajorityVote", SHOGUN_BASIC_CLASS __new_CMajorityVote},
{"MeanRule", SHOGUN_BASIC_CLASS __new_CMeanRule},
{"WeightedMajorityVote", SHOGUN_BASIC_CLASS __new_CWeightedMajorityVote},
{"ClusteringAccuracy", SHOGUN_BASIC_CLASS __new_CClusteringAccuracy},
{"ClusteringMutualInformation", SHOGUN_BASIC_CLASS __new_CClusteringMutualInformation},
{"ContingencyTableEvaluation", SHOGUN_BASIC_CLASS __new_CContingencyTableEvaluation},
{"AccuracyMeasure", SHOGUN_BASIC_CLASS __new_CAccuracyMeasure},
{"ErrorRateMeasure", SHOGUN_BASIC_CLASS __new_CErrorRateMeasure},
{"BALMeasure", SHOGUN_BASIC_CLASS __new_CBALMeasure},
{"WRACCMeasure", SHOGUN_BASIC_CLASS __new_CWRACCMeasure},
{"F1Measure", SHOGUN_BASIC_CLASS __new_CF1Measure},
{"CrossCorrelationMeasure", SHOGUN_BASIC_CLASS __new_CCrossCorrelationMeasure},
{"RecallMeasure", SHOGUN_BASIC_CLASS __new_CRecallMeasure},
{"PrecisionMeasure", SHOGUN_BASIC_CLASS __new_CPrecisionMeasure},
{"SpecificityMeasure", SHOGUN_BASIC_CLASS __new_CSpecificityMeasure},
{"CrossValidationResult", SHOGUN_BASIC_CLASS __new_CCrossValidationResult},
{"CrossValidation", SHOGUN_BASIC_CLASS __new_CCrossValidation},
{"CrossValidationMKLStorage", SHOGUN_BASIC_CLASS __new_CCrossValidationMKLStorage},
{"CrossValidationMulticlassStorage", SHOGUN_BASIC_CLASS __new_CCrossValidationMulticlassStorage},
{"CrossValidationPrintOutput", SHOGUN_BASIC_CLASS __new_CCrossValidationPrintOutput},
{"CrossValidationSplitting", SHOGUN_BASIC_CLASS __new_CCrossValidationSplitting},
{"GradientCriterion", SHOGUN_BASIC_CLASS __new_CGradientCriterion},
{"GradientEvaluation", SHOGUN_BASIC_CLASS __new_CGradientEvaluation},
{"GradientResult", SHOGUN_BASIC_CLASS __new_CGradientResult},
{"MeanAbsoluteError", SHOGUN_BASIC_CLASS __new_CMeanAbsoluteError},
{"MeanSquaredError", SHOGUN_BASIC_CLASS __new_CMeanSquaredError},
{"MeanSquaredLogError", SHOGUN_BASIC_CLASS __new_CMeanSquaredLogError},
{"MulticlassAccuracy", SHOGUN_BASIC_CLASS __new_CMulticlassAccuracy},
{"MulticlassOVREvaluation", SHOGUN_BASIC_CLASS __new_CMulticlassOVREvaluation},
{"PRCEvaluation", SHOGUN_BASIC_CLASS __new_CPRCEvaluation},
{"ROCEvaluation", SHOGUN_BASIC_CLASS __new_CROCEvaluation},
{"StratifiedCrossValidationSplitting", SHOGUN_BASIC_CLASS __new_CStratifiedCrossValidationSplitting},
{"StructuredAccuracy", SHOGUN_BASIC_CLASS __new_CStructuredAccuracy},
{"Alphabet", SHOGUN_BASIC_CLASS __new_CAlphabet},
{"BinnedDotFeatures", SHOGUN_BASIC_CLASS __new_CBinnedDotFeatures},
{"CombinedDotFeatures", SHOGUN_BASIC_CLASS __new_CCombinedDotFeatures},
{"CombinedFeatures", SHOGUN_BASIC_CLASS __new_CCombinedFeatures},
{"DataGenerator", SHOGUN_BASIC_CLASS __new_CDataGenerator},
{"DummyFeatures", SHOGUN_BASIC_CLASS __new_CDummyFeatures},
{"ExplicitSpecFeatures", SHOGUN_BASIC_CLASS __new_CExplicitSpecFeatures},
{"FactorGraphFeatures", SHOGUN_BASIC_CLASS __new_CFactorGraphFeatures},
{"FKFeatures", SHOGUN_BASIC_CLASS __new_CFKFeatures},
{"HashedDocDotFeatures", SHOGUN_BASIC_CLASS __new_CHashedDocDotFeatures},
{"HashedWDFeatures", SHOGUN_BASIC_CLASS __new_CHashedWDFeatures},
{"HashedWDFeaturesTransposed", SHOGUN_BASIC_CLASS __new_CHashedWDFeaturesTransposed},
{"ImplicitWeightedSpecFeatures", SHOGUN_BASIC_CLASS __new_CImplicitWeightedSpecFeatures},
{"LatentFeatures", SHOGUN_BASIC_CLASS __new_CLatentFeatures},
{"LBPPyrDotFeatures", SHOGUN_BASIC_CLASS __new_CLBPPyrDotFeatures},
{"PolyFeatures", SHOGUN_BASIC_CLASS __new_CPolyFeatures},
{"RandomFourierDotFeatures", SHOGUN_BASIC_CLASS __new_CRandomFourierDotFeatures},
{"RealFileFeatures", SHOGUN_BASIC_CLASS __new_CRealFileFeatures},
{"SNPFeatures", SHOGUN_BASIC_CLASS __new_CSNPFeatures},
{"SparsePolyFeatures", SHOGUN_BASIC_CLASS __new_CSparsePolyFeatures},
{"GaussianBlobsDataGenerator", SHOGUN_BASIC_CLASS __new_CGaussianBlobsDataGenerator},
{"MeanShiftDataGenerator", SHOGUN_BASIC_CLASS __new_CMeanShiftDataGenerator},
{"StreamingHashedDocDotFeatures", SHOGUN_BASIC_CLASS __new_CStreamingHashedDocDotFeatures},
{"StreamingVwFeatures", SHOGUN_BASIC_CLASS __new_CStreamingVwFeatures},
{"Subset", SHOGUN_BASIC_CLASS __new_CSubset},
{"SubsetStack", SHOGUN_BASIC_CLASS __new_CSubsetStack},
{"TOPFeatures", SHOGUN_BASIC_CLASS __new_CTOPFeatures},
{"WDFeatures", SHOGUN_BASIC_CLASS __new_CWDFeatures},
{"BinaryFile", SHOGUN_BASIC_CLASS __new_CBinaryFile},
{"CSVFile", SHOGUN_BASIC_CLASS __new_CCSVFile},
{"IOBuffer", SHOGUN_BASIC_CLASS __new_CIOBuffer},
{"LibSVMFile", SHOGUN_BASIC_CLASS __new_CLibSVMFile},
{"LineReader", SHOGUN_BASIC_CLASS __new_CLineReader},
{"Parser", SHOGUN_BASIC_CLASS __new_CParser},
{"SerializableAsciiFile", SHOGUN_BASIC_CLASS __new_CSerializableAsciiFile},
{"StreamingAsciiFile", SHOGUN_BASIC_CLASS __new_CStreamingAsciiFile},
{"StreamingFile", SHOGUN_BASIC_CLASS __new_CStreamingFile},
{"StreamingFileFromFeatures", SHOGUN_BASIC_CLASS __new_CStreamingFileFromFeatures},
{"StreamingVwCacheFile", SHOGUN_BASIC_CLASS __new_CStreamingVwCacheFile},
{"StreamingVwFile", SHOGUN_BASIC_CLASS __new_CStreamingVwFile},
{"ANOVAKernel", SHOGUN_BASIC_CLASS __new_CANOVAKernel},
{"AUCKernel", SHOGUN_BASIC_CLASS __new_CAUCKernel},
{"BesselKernel", SHOGUN_BASIC_CLASS __new_CBesselKernel},
{"CauchyKernel", SHOGUN_BASIC_CLASS __new_CCauchyKernel},
{"Chi2Kernel", SHOGUN_BASIC_CLASS __new_CChi2Kernel},
{"CircularKernel", SHOGUN_BASIC_CLASS __new_CCircularKernel},
{"CombinedKernel", SHOGUN_BASIC_CLASS __new_CCombinedKernel},
{"ConstKernel", SHOGUN_BASIC_CLASS __new_CConstKernel},
{"CustomKernel", SHOGUN_BASIC_CLASS __new_CCustomKernel},
{"DiagKernel", SHOGUN_BASIC_CLASS __new_CDiagKernel},
{"DistanceKernel", SHOGUN_BASIC_CLASS __new_CDistanceKernel},
{"ExponentialKernel", SHOGUN_BASIC_CLASS __new_CExponentialKernel},
{"GaussianARDKernel", SHOGUN_BASIC_CLASS __new_CGaussianARDKernel},
{"GaussianKernel", SHOGUN_BASIC_CLASS __new_CGaussianKernel},
{"GaussianShiftKernel", SHOGUN_BASIC_CLASS __new_CGaussianShiftKernel},
{"GaussianShortRealKernel", SHOGUN_BASIC_CLASS __new_CGaussianShortRealKernel},
{"HistogramIntersectionKernel", SHOGUN_BASIC_CLASS __new_CHistogramIntersectionKernel},
{"InverseMultiQuadricKernel", SHOGUN_BASIC_CLASS __new_CInverseMultiQuadricKernel},
{"JensenShannonKernel", SHOGUN_BASIC_CLASS __new_CJensenShannonKernel},
{"LinearARDKernel", SHOGUN_BASIC_CLASS __new_CLinearARDKernel},
{"LinearKernel", SHOGUN_BASIC_CLASS __new_CLinearKernel},
{"LogKernel", SHOGUN_BASIC_CLASS __new_CLogKernel},
{"MultiquadricKernel", SHOGUN_BASIC_CLASS __new_CMultiquadricKernel},
{"AvgDiagKernelNormalizer", SHOGUN_BASIC_CLASS __new_CAvgDiagKernelNormalizer},
{"DiceKernelNormalizer", SHOGUN_BASIC_CLASS __new_CDiceKernelNormalizer},
{"FirstElementKernelNormalizer", SHOGUN_BASIC_CLASS __new_CFirstElementKernelNormalizer},
{"IdentityKernelNormalizer", SHOGUN_BASIC_CLASS __new_CIdentityKernelNormalizer},
{"RidgeKernelNormalizer", SHOGUN_BASIC_CLASS __new_CRidgeKernelNormalizer},
{"ScatterKernelNormalizer", SHOGUN_BASIC_CLASS __new_CScatterKernelNormalizer},
{"SqrtDiagKernelNormalizer", SHOGUN_BASIC_CLASS __new_CSqrtDiagKernelNormalizer},
{"TanimotoKernelNormalizer", SHOGUN_BASIC_CLASS __new_CTanimotoKernelNormalizer},
{"VarianceKernelNormalizer", SHOGUN_BASIC_CLASS __new_CVarianceKernelNormalizer},
{"ZeroMeanCenterKernelNormalizer", SHOGUN_BASIC_CLASS __new_CZeroMeanCenterKernelNormalizer},
{"PolyKernel", SHOGUN_BASIC_CLASS __new_CPolyKernel},
{"PowerKernel", SHOGUN_BASIC_CLASS __new_CPowerKernel},
{"ProductKernel", SHOGUN_BASIC_CLASS __new_CProductKernel},
{"PyramidChi2", SHOGUN_BASIC_CLASS __new_CPyramidChi2},
{"RationalQuadraticKernel", SHOGUN_BASIC_CLASS __new_CRationalQuadraticKernel},
{"SigmoidKernel", SHOGUN_BASIC_CLASS __new_CSigmoidKernel},
{"SphericalKernel", SHOGUN_BASIC_CLASS __new_CSphericalKernel},
{"SplineKernel", SHOGUN_BASIC_CLASS __new_CSplineKernel},
{"CommUlongStringKernel", SHOGUN_BASIC_CLASS __new_CCommUlongStringKernel},
{"CommWordStringKernel", SHOGUN_BASIC_CLASS __new_CCommWordStringKernel},
{"DistantSegmentsKernel", SHOGUN_BASIC_CLASS __new_CDistantSegmentsKernel},
{"FixedDegreeStringKernel", SHOGUN_BASIC_CLASS __new_CFixedDegreeStringKernel},
{"GaussianMatchStringKernel", SHOGUN_BASIC_CLASS __new_CGaussianMatchStringKernel},
{"HistogramWordStringKernel", SHOGUN_BASIC_CLASS __new_CHistogramWordStringKernel},
{"LinearStringKernel", SHOGUN_BASIC_CLASS __new_CLinearStringKernel},
{"LocalAlignmentStringKernel", SHOGUN_BASIC_CLASS __new_CLocalAlignmentStringKernel},
{"LocalityImprovedStringKernel", SHOGUN_BASIC_CLASS __new_CLocalityImprovedStringKernel},
{"MatchWordStringKernel", SHOGUN_BASIC_CLASS __new_CMatchWordStringKernel},
{"OligoStringKernel", SHOGUN_BASIC_CLASS __new_COligoStringKernel},
{"PolyMatchStringKernel", SHOGUN_BASIC_CLASS __new_CPolyMatchStringKernel},
{"PolyMatchWordStringKernel", SHOGUN_BASIC_CLASS __new_CPolyMatchWordStringKernel},
{"RegulatoryModulesStringKernel", SHOGUN_BASIC_CLASS __new_CRegulatoryModulesStringKernel},
{"SalzbergWordStringKernel", SHOGUN_BASIC_CLASS __new_CSalzbergWordStringKernel},
{"SimpleLocalityImprovedStringKernel", SHOGUN_BASIC_CLASS __new_CSimpleLocalityImprovedStringKernel},
{"SNPStringKernel", SHOGUN_BASIC_CLASS __new_CSNPStringKernel},
{"SparseSpatialSampleStringKernel", SHOGUN_BASIC_CLASS __new_CSparseSpatialSampleStringKernel},
{"SpectrumMismatchRBFKernel", SHOGUN_BASIC_CLASS __new_CSpectrumMismatchRBFKernel},
{"SpectrumRBFKernel", SHOGUN_BASIC_CLASS __new_CSpectrumRBFKernel},
{"WeightedCommWordStringKernel", SHOGUN_BASIC_CLASS __new_CWeightedCommWordStringKernel},
{"WeightedDegreePositionStringKernel", SHOGUN_BASIC_CLASS __new_CWeightedDegreePositionStringKernel},
{"WeightedDegreeStringKernel", SHOGUN_BASIC_CLASS __new_CWeightedDegreeStringKernel},
{"TensorProductPairKernel", SHOGUN_BASIC_CLASS __new_CTensorProductPairKernel},
{"TStudentKernel", SHOGUN_BASIC_CLASS __new_CTStudentKernel},
{"WaveKernel", SHOGUN_BASIC_CLASS __new_CWaveKernel},
{"WaveletKernel", SHOGUN_BASIC_CLASS __new_CWaveletKernel},
{"WeightedDegreeRBFKernel", SHOGUN_BASIC_CLASS __new_CWeightedDegreeRBFKernel},
{"BinaryLabels", SHOGUN_BASIC_CLASS __new_CBinaryLabels},
{"FactorGraphObservation", SHOGUN_BASIC_CLASS __new_CFactorGraphObservation},
{"FactorGraphLabels", SHOGUN_BASIC_CLASS __new_CFactorGraphLabels},
{"LabelsFactory", SHOGUN_BASIC_CLASS __new_CLabelsFactory},
{"LatentLabels", SHOGUN_BASIC_CLASS __new_CLatentLabels},
{"MulticlassLabels", SHOGUN_BASIC_CLASS __new_CMulticlassLabels},
{"MulticlassMultipleOutputLabels", SHOGUN_BASIC_CLASS __new_CMulticlassMultipleOutputLabels},
{"RegressionLabels", SHOGUN_BASIC_CLASS __new_CRegressionLabels},
{"StructuredLabels", SHOGUN_BASIC_CLASS __new_CStructuredLabels},
{"LatentSOSVM", SHOGUN_BASIC_CLASS __new_CLatentSOSVM},
{"LatentSVM", SHOGUN_BASIC_CLASS __new_CLatentSVM},
{"BitString", SHOGUN_BASIC_CLASS __new_CBitString},
{"CircularBuffer", SHOGUN_BASIC_CLASS __new_CCircularBuffer},
{"Compressor", SHOGUN_BASIC_CLASS __new_CCompressor},
{"SerialComputationEngine", SHOGUN_BASIC_CLASS __new_CSerialComputationEngine},
{"JobResult", SHOGUN_BASIC_CLASS __new_CJobResult},
{"Data", SHOGUN_BASIC_CLASS __new_CData},
{"DelimiterTokenizer", SHOGUN_BASIC_CLASS __new_CDelimiterTokenizer},
{"DynamicObjectArray", SHOGUN_BASIC_CLASS __new_CDynamicObjectArray},
{"Hash", SHOGUN_BASIC_CLASS __new_CHash},
{"IndexBlock", SHOGUN_BASIC_CLASS __new_CIndexBlock},
{"IndexBlockGroup", SHOGUN_BASIC_CLASS __new_CIndexBlockGroup},
{"IndexBlockTree", SHOGUN_BASIC_CLASS __new_CIndexBlockTree},
{"ListElement", SHOGUN_BASIC_CLASS __new_CListElement},
{"List", SHOGUN_BASIC_CLASS __new_CList},
{"NGramTokenizer", SHOGUN_BASIC_CLASS __new_CNGramTokenizer},
{"Signal", SHOGUN_BASIC_CLASS __new_CSignal},
{"StructuredData", SHOGUN_BASIC_CLASS __new_CStructuredData},
{"Time", SHOGUN_BASIC_CLASS __new_CTime},
{"HingeLoss", SHOGUN_BASIC_CLASS __new_CHingeLoss},
{"LogLoss", SHOGUN_BASIC_CLASS __new_CLogLoss},
{"LogLossMargin", SHOGUN_BASIC_CLASS __new_CLogLossMargin},
{"SmoothHingeLoss", SHOGUN_BASIC_CLASS __new_CSmoothHingeLoss},
{"SquaredHingeLoss", SHOGUN_BASIC_CLASS __new_CSquaredHingeLoss},
{"SquaredLoss", SHOGUN_BASIC_CLASS __new_CSquaredLoss},
{"BaggingMachine", SHOGUN_BASIC_CLASS __new_CBaggingMachine},
{"BaseMulticlassMachine", SHOGUN_BASIC_CLASS __new_CBaseMulticlassMachine},
{"DistanceMachine", SHOGUN_BASIC_CLASS __new_CDistanceMachine},
{"ZeroMean", SHOGUN_BASIC_CLASS __new_CZeroMean},
{"KernelMachine", SHOGUN_BASIC_CLASS __new_CKernelMachine},
{"KernelMulticlassMachine", SHOGUN_BASIC_CLASS __new_CKernelMulticlassMachine},
{"KernelStructuredOutputMachine", SHOGUN_BASIC_CLASS __new_CKernelStructuredOutputMachine},
{"LinearMachine", SHOGUN_BASIC_CLASS __new_CLinearMachine},
{"LinearMulticlassMachine", SHOGUN_BASIC_CLASS __new_CLinearMulticlassMachine},
{"LinearStructuredOutputMachine", SHOGUN_BASIC_CLASS __new_CLinearStructuredOutputMachine},
{"Machine", SHOGUN_BASIC_CLASS __new_CMachine},
{"NativeMulticlassMachine", SHOGUN_BASIC_CLASS __new_CNativeMulticlassMachine},
{"OnlineLinearMachine", SHOGUN_BASIC_CLASS __new_COnlineLinearMachine},
{"StructuredOutputMachine", SHOGUN_BASIC_CLASS __new_CStructuredOutputMachine},
{"JacobiEllipticFunctions", SHOGUN_BASIC_CLASS __new_CJacobiEllipticFunctions},
{"LogDetEstimator", SHOGUN_BASIC_CLASS __new_CLogDetEstimator},
{"NormalSampler", SHOGUN_BASIC_CLASS __new_CNormalSampler},
{"Math", SHOGUN_BASIC_CLASS __new_CMath},
{"Random", SHOGUN_BASIC_CLASS __new_CRandom},
{"SparseInverseCovariance", SHOGUN_BASIC_CLASS __new_CSparseInverseCovariance},
{"Statistics", SHOGUN_BASIC_CLASS __new_CStatistics},
{"GridSearchModelSelection", SHOGUN_BASIC_CLASS __new_CGridSearchModelSelection},
{"ModelSelectionParameters", SHOGUN_BASIC_CLASS __new_CModelSelectionParameters},
{"ParameterCombination", SHOGUN_BASIC_CLASS __new_CParameterCombination},
{"RandomSearchModelSelection", SHOGUN_BASIC_CLASS __new_CRandomSearchModelSelection},
{"ECOCAEDDecoder", SHOGUN_BASIC_CLASS __new_CECOCAEDDecoder},
{"ECOCDiscriminantEncoder", SHOGUN_BASIC_CLASS __new_CECOCDiscriminantEncoder},
{"ECOCEDDecoder", SHOGUN_BASIC_CLASS __new_CECOCEDDecoder},
{"ECOCForestEncoder", SHOGUN_BASIC_CLASS __new_CECOCForestEncoder},
{"ECOCHDDecoder", SHOGUN_BASIC_CLASS __new_CECOCHDDecoder},
{"ECOCLLBDecoder", SHOGUN_BASIC_CLASS __new_CECOCLLBDecoder},
{"ECOCOVOEncoder", SHOGUN_BASIC_CLASS __new_CECOCOVOEncoder},
{"ECOCOVREncoder", SHOGUN_BASIC_CLASS __new_CECOCOVREncoder},
{"ECOCRandomDenseEncoder", SHOGUN_BASIC_CLASS __new_CECOCRandomDenseEncoder},
{"ECOCRandomSparseEncoder", SHOGUN_BASIC_CLASS __new_CECOCRandomSparseEncoder},
{"ECOCStrategy", SHOGUN_BASIC_CLASS __new_CECOCStrategy},
{"GaussianNaiveBayes", SHOGUN_BASIC_CLASS __new_CGaussianNaiveBayes},
{"GMNPLib", SHOGUN_BASIC_CLASS __new_CGMNPLib},
{"GMNPSVM", SHOGUN_BASIC_CLASS __new_CGMNPSVM},
{"KNN", SHOGUN_BASIC_CLASS __new_CKNN},
{"LaRank", SHOGUN_BASIC_CLASS __new_CLaRank},
{"MulticlassLibLinear", SHOGUN_BASIC_CLASS __new_CMulticlassLibLinear},
{"MulticlassLibSVM", SHOGUN_BASIC_CLASS __new_CMulticlassLibSVM},
{"MulticlassOCAS", SHOGUN_BASIC_CLASS __new_CMulticlassOCAS},
{"MulticlassOneVsOneStrategy", SHOGUN_BASIC_CLASS __new_CMulticlassOneVsOneStrategy},
{"MulticlassOneVsRestStrategy", SHOGUN_BASIC_CLASS __new_CMulticlassOneVsRestStrategy},
{"MulticlassSVM", SHOGUN_BASIC_CLASS __new_CMulticlassSVM},
{"ThresholdRejectionStrategy", SHOGUN_BASIC_CLASS __new_CThresholdRejectionStrategy},
{"DixonQTestRejectionStrategy", SHOGUN_BASIC_CLASS __new_CDixonQTestRejectionStrategy},
{"ScatterSVM", SHOGUN_BASIC_CLASS __new_CScatterSVM},
{"ShareBoost", SHOGUN_BASIC_CLASS __new_CShareBoost},
{"BalancedConditionalProbabilityTree", SHOGUN_BASIC_CLASS __new_CBalancedConditionalProbabilityTree},
{"RandomConditionalProbabilityTree", SHOGUN_BASIC_CLASS __new_CRandomConditionalProbabilityTree},
{"RelaxedTree", SHOGUN_BASIC_CLASS __new_CRelaxedTree},
{"Tron", SHOGUN_BASIC_CLASS __new_CTron},
{"DimensionReductionPreprocessor", SHOGUN_BASIC_CLASS __new_CDimensionReductionPreprocessor},
{"HomogeneousKernelMap", SHOGUN_BASIC_CLASS __new_CHomogeneousKernelMap},
{"LogPlusOne", SHOGUN_BASIC_CLASS __new_CLogPlusOne},
{"NormOne", SHOGUN_BASIC_CLASS __new_CNormOne},
{"PNorm", SHOGUN_BASIC_CLASS __new_CPNorm},
{"PruneVarSubMean", SHOGUN_BASIC_CLASS __new_CPruneVarSubMean},
{"RandomFourierGaussPreproc", SHOGUN_BASIC_CLASS __new_CRandomFourierGaussPreproc},
{"RescaleFeatures", SHOGUN_BASIC_CLASS __new_CRescaleFeatures},
{"SortUlongString", SHOGUN_BASIC_CLASS __new_CSortUlongString},
{"SortWordString", SHOGUN_BASIC_CLASS __new_CSortWordString},
{"SumOne", SHOGUN_BASIC_CLASS __new_CSumOne},
{"LibSVR", SHOGUN_BASIC_CLASS __new_CLibSVR},
{"MKLRegression", SHOGUN_BASIC_CLASS __new_CMKLRegression},
{"SVRLight", SHOGUN_BASIC_CLASS __new_CSVRLight},
{"HSIC", SHOGUN_BASIC_CLASS __new_CHSIC},
{"KernelMeanMatching", SHOGUN_BASIC_CLASS __new_CKernelMeanMatching},
{"LinearTimeMMD", SHOGUN_BASIC_CLASS __new_CLinearTimeMMD},
{"MMDKernelSelectionCombMaxL2", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionCombMaxL2},
{"MMDKernelSelectionCombOpt", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionCombOpt},
{"MMDKernelSelectionMax", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionMax},
{"MMDKernelSelectionMedian", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionMedian},
{"MMDKernelSelectionOpt", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionOpt},
{"QuadraticTimeMMD", SHOGUN_BASIC_CLASS __new_CQuadraticTimeMMD},
{"CCSOSVM", SHOGUN_BASIC_CLASS __new_CCCSOSVM},
{"DisjointSet", SHOGUN_BASIC_CLASS __new_CDisjointSet},
{"DualLibQPBMSOSVM", SHOGUN_BASIC_CLASS __new_CDualLibQPBMSOSVM},
{"DynProg", SHOGUN_BASIC_CLASS __new_CDynProg},
{"FactorDataSource", SHOGUN_BASIC_CLASS __new_CFactorDataSource},
{"Factor", SHOGUN_BASIC_CLASS __new_CFactor},
{"FactorGraph", SHOGUN_BASIC_CLASS __new_CFactorGraph},
{"FactorGraphModel", SHOGUN_BASIC_CLASS __new_CFactorGraphModel},
{"FactorType", SHOGUN_BASIC_CLASS __new_CFactorType},
{"TableFactorType", SHOGUN_BASIC_CLASS __new_CTableFactorType},
{"HMSVMModel", SHOGUN_BASIC_CLASS __new_CHMSVMModel},
{"IntronList", SHOGUN_BASIC_CLASS __new_CIntronList},
{"MAPInference", SHOGUN_BASIC_CLASS __new_CMAPInference},
{"MulticlassModel", SHOGUN_BASIC_CLASS __new_CMulticlassModel},
{"MulticlassSOLabels", SHOGUN_BASIC_CLASS __new_CMulticlassSOLabels},
{"Plif", SHOGUN_BASIC_CLASS __new_CPlif},
{"PlifArray", SHOGUN_BASIC_CLASS __new_CPlifArray},
{"PlifMatrix", SHOGUN_BASIC_CLASS __new_CPlifMatrix},
{"SegmentLoss", SHOGUN_BASIC_CLASS __new_CSegmentLoss},
{"Sequence", SHOGUN_BASIC_CLASS __new_CSequence},
{"SequenceLabels", SHOGUN_BASIC_CLASS __new_CSequenceLabels},
{"SOSVMHelper", SHOGUN_BASIC_CLASS __new_CSOSVMHelper},
{"StochasticSOSVM", SHOGUN_BASIC_CLASS __new_CStochasticSOSVM},
{"TwoStateModel", SHOGUN_BASIC_CLASS __new_CTwoStateModel},
{"DomainAdaptationSVM", SHOGUN_BASIC_CLASS __new_CDomainAdaptationSVM},
{"MultitaskClusteredLogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskClusteredLogisticRegression},
{"MultitaskKernelMaskNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelMaskNormalizer},
{"MultitaskKernelMaskPairNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelMaskPairNormalizer},
{"MultitaskKernelNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelNormalizer},
{"MultitaskKernelPlifNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelPlifNormalizer},
{"Node", SHOGUN_BASIC_CLASS __new_CNode},
{"Taxonomy", SHOGUN_BASIC_CLASS __new_CTaxonomy},
{"MultitaskKernelTreeNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelTreeNormalizer},
{"MultitaskL12LogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskL12LogisticRegression},
{"MultitaskLeastSquaresRegression", SHOGUN_BASIC_CLASS __new_CMultitaskLeastSquaresRegression},
{"MultitaskLinearMachine", SHOGUN_BASIC_CLASS __new_CMultitaskLinearMachine},
{"MultitaskLogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskLogisticRegression},
{"MultitaskROCEvaluation", SHOGUN_BASIC_CLASS __new_CMultitaskROCEvaluation},
{"MultitaskTraceLogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskTraceLogisticRegression},
{"Task", SHOGUN_BASIC_CLASS __new_CTask},
{"TaskGroup", SHOGUN_BASIC_CLASS __new_CTaskGroup},
{"TaskTree", SHOGUN_BASIC_CLASS __new_CTaskTree},
{"GUIClassifier", SHOGUN_BASIC_CLASS __new_CGUIClassifier},
{"GUIConverter", SHOGUN_BASIC_CLASS __new_CGUIConverter},
{"GUIDistance", SHOGUN_BASIC_CLASS __new_CGUIDistance},
{"GUIFeatures", SHOGUN_BASIC_CLASS __new_CGUIFeatures},
{"GUIHMM", SHOGUN_BASIC_CLASS __new_CGUIHMM},
{"GUIKernel", SHOGUN_BASIC_CLASS __new_CGUIKernel},
{"GUILabels", SHOGUN_BASIC_CLASS __new_CGUILabels},
{"GUIMath", SHOGUN_BASIC_CLASS __new_CGUIMath},
{"GUIPluginEstimate", SHOGUN_BASIC_CLASS __new_CGUIPluginEstimate},
{"GUIPreprocessor", SHOGUN_BASIC_CLASS __new_CGUIPreprocessor},
{"GUIStructure", SHOGUN_BASIC_CLASS __new_CGUIStructure},
{"GUITime", SHOGUN_BASIC_CLASS __new_CGUITime},
{"AveragedPerceptron", SHOGUN_BASIC_CLASS __new_CAveragedPerceptron},
{"FeatureBlockLogisticRegression", SHOGUN_BASIC_CLASS __new_CFeatureBlockLogisticRegression},
{"MKLClassification", SHOGUN_BASIC_CLASS __new_CMKLClassification},
{"MKLMulticlass", SHOGUN_BASIC_CLASS __new_CMKLMulticlass},
{"MKLOneClass", SHOGUN_BASIC_CLASS __new_CMKLOneClass},
{"NearestCentroid", SHOGUN_BASIC_CLASS __new_CNearestCentroid},
{"Perceptron", SHOGUN_BASIC_CLASS __new_CPerceptron},
{"PluginEstimate", SHOGUN_BASIC_CLASS __new_CPluginEstimate},
{"GNPPLib", SHOGUN_BASIC_CLASS __new_CGNPPLib},
{"GNPPSVM", SHOGUN_BASIC_CLASS __new_CGNPPSVM},
{"GPBTSVM", SHOGUN_BASIC_CLASS __new_CGPBTSVM},
{"LibLinear", SHOGUN_BASIC_CLASS __new_CLibLinear},
{"LibSVM", SHOGUN_BASIC_CLASS __new_CLibSVM},
{"LibSVMOneClass", SHOGUN_BASIC_CLASS __new_CLibSVMOneClass},
{"MPDSVM", SHOGUN_BASIC_CLASS __new_CMPDSVM},
{"OnlineLibLinear", SHOGUN_BASIC_CLASS __new_COnlineLibLinear},
{"OnlineSVMSGD", SHOGUN_BASIC_CLASS __new_COnlineSVMSGD},
{"QPBSVMLib", SHOGUN_BASIC_CLASS __new_CQPBSVMLib},
{"SGDQN", SHOGUN_BASIC_CLASS __new_CSGDQN},
{"SVM", SHOGUN_BASIC_CLASS __new_CSVM},
{"SVMLight", SHOGUN_BASIC_CLASS __new_CSVMLight},
{"SVMLightOneClass", SHOGUN_BASIC_CLASS __new_CSVMLightOneClass},
{"SVMLin", SHOGUN_BASIC_CLASS __new_CSVMLin},
{"SVMOcas", SHOGUN_BASIC_CLASS __new_CSVMOcas},
{"SVMSGD", SHOGUN_BASIC_CLASS __new_CSVMSGD},
{"WDSVMOcas", SHOGUN_BASIC_CLASS __new_CWDSVMOcas},
{"VwNativeCacheReader", SHOGUN_BASIC_CLASS __new_CVwNativeCacheReader},
{"VwNativeCacheWriter", SHOGUN_BASIC_CLASS __new_CVwNativeCacheWriter},
{"VwAdaptiveLearner", SHOGUN_BASIC_CLASS __new_CVwAdaptiveLearner},
{"VwNonAdaptiveLearner", SHOGUN_BASIC_CLASS __new_CVwNonAdaptiveLearner},
{"VowpalWabbit", SHOGUN_BASIC_CLASS __new_CVowpalWabbit},
{"VwEnvironment", SHOGUN_BASIC_CLASS __new_CVwEnvironment},
{"VwParser", SHOGUN_BASIC_CLASS __new_CVwParser},
{"VwRegressor", SHOGUN_BASIC_CLASS __new_CVwRegressor},
{"Hierarchical", SHOGUN_BASIC_CLASS __new_CHierarchical},
{"KMeans", SHOGUN_BASIC_CLASS __new_CKMeans},
{"HashedDocConverter", SHOGUN_BASIC_CLASS __new_CHashedDocConverter},
{"AttenuatedEuclideanDistance", SHOGUN_BASIC_CLASS __new_CAttenuatedEuclideanDistance},
{"BrayCurtisDistance", SHOGUN_BASIC_CLASS __new_CBrayCurtisDistance},
{"CanberraMetric", SHOGUN_BASIC_CLASS __new_CCanberraMetric},
{"CanberraWordDistance", SHOGUN_BASIC_CLASS __new_CCanberraWordDistance},
{"ChebyshewMetric", SHOGUN_BASIC_CLASS __new_CChebyshewMetric},
{"ChiSquareDistance", SHOGUN_BASIC_CLASS __new_CChiSquareDistance},
{"CosineDistance", SHOGUN_BASIC_CLASS __new_CCosineDistance},
{"CustomDistance", SHOGUN_BASIC_CLASS __new_CCustomDistance},
{"EuclideanDistance", SHOGUN_BASIC_CLASS __new_CEuclideanDistance},
{"GeodesicMetric", SHOGUN_BASIC_CLASS __new_CGeodesicMetric},
{"HammingWordDistance", SHOGUN_BASIC_CLASS __new_CHammingWordDistance},
{"JensenMetric", SHOGUN_BASIC_CLASS __new_CJensenMetric},
{"KernelDistance", SHOGUN_BASIC_CLASS __new_CKernelDistance},
{"ManhattanMetric", SHOGUN_BASIC_CLASS __new_CManhattanMetric},
{"ManhattanWordDistance", SHOGUN_BASIC_CLASS __new_CManhattanWordDistance},
{"MinkowskiMetric", SHOGUN_BASIC_CLASS __new_CMinkowskiMetric},
{"SparseEuclideanDistance", SHOGUN_BASIC_CLASS __new_CSparseEuclideanDistance},
{"TanimotoDistance", SHOGUN_BASIC_CLASS __new_CTanimotoDistance},
{"GHMM", SHOGUN_BASIC_CLASS __new_CGHMM},
{"Histogram", SHOGUN_BASIC_CLASS __new_CHistogram},
{"HMM", SHOGUN_BASIC_CLASS __new_CHMM},
{"LinearHMM", SHOGUN_BASIC_CLASS __new_CLinearHMM},
{"PositionalPWM", SHOGUN_BASIC_CLASS __new_CPositionalPWM},
{"MajorityVote", SHOGUN_BASIC_CLASS __new_CMajorityVote},
{"MeanRule", SHOGUN_BASIC_CLASS __new_CMeanRule},
{"WeightedMajorityVote", SHOGUN_BASIC_CLASS __new_CWeightedMajorityVote},
{"ClusteringAccuracy", SHOGUN_BASIC_CLASS __new_CClusteringAccuracy},
{"ClusteringMutualInformation", SHOGUN_BASIC_CLASS __new_CClusteringMutualInformation},
{"ContingencyTableEvaluation", SHOGUN_BASIC_CLASS __new_CContingencyTableEvaluation},
{"AccuracyMeasure", SHOGUN_BASIC_CLASS __new_CAccuracyMeasure},
{"ErrorRateMeasure", SHOGUN_BASIC_CLASS __new_CErrorRateMeasure},
{"BALMeasure", SHOGUN_BASIC_CLASS __new_CBALMeasure},
{"WRACCMeasure", SHOGUN_BASIC_CLASS __new_CWRACCMeasure},
{"F1Measure", SHOGUN_BASIC_CLASS __new_CF1Measure},
{"CrossCorrelationMeasure", SHOGUN_BASIC_CLASS __new_CCrossCorrelationMeasure},
{"RecallMeasure", SHOGUN_BASIC_CLASS __new_CRecallMeasure},
{"PrecisionMeasure", SHOGUN_BASIC_CLASS __new_CPrecisionMeasure},
{"SpecificityMeasure", SHOGUN_BASIC_CLASS __new_CSpecificityMeasure},
{"CrossValidationResult", SHOGUN_BASIC_CLASS __new_CCrossValidationResult},
{"CrossValidation", SHOGUN_BASIC_CLASS __new_CCrossValidation},
{"CrossValidationMKLStorage", SHOGUN_BASIC_CLASS __new_CCrossValidationMKLStorage},
{"CrossValidationMulticlassStorage", SHOGUN_BASIC_CLASS __new_CCrossValidationMulticlassStorage},
{"CrossValidationPrintOutput", SHOGUN_BASIC_CLASS __new_CCrossValidationPrintOutput},
{"CrossValidationSplitting", SHOGUN_BASIC_CLASS __new_CCrossValidationSplitting},
{"GradientCriterion", SHOGUN_BASIC_CLASS __new_CGradientCriterion},
{"GradientEvaluation", SHOGUN_BASIC_CLASS __new_CGradientEvaluation},
{"GradientResult", SHOGUN_BASIC_CLASS __new_CGradientResult},
{"MeanAbsoluteError", SHOGUN_BASIC_CLASS __new_CMeanAbsoluteError},
{"MeanSquaredError", SHOGUN_BASIC_CLASS __new_CMeanSquaredError},
{"MeanSquaredLogError", SHOGUN_BASIC_CLASS __new_CMeanSquaredLogError},
{"MulticlassAccuracy", SHOGUN_BASIC_CLASS __new_CMulticlassAccuracy},
{"MulticlassOVREvaluation", SHOGUN_BASIC_CLASS __new_CMulticlassOVREvaluation},
{"PRCEvaluation", SHOGUN_BASIC_CLASS __new_CPRCEvaluation},
{"ROCEvaluation", SHOGUN_BASIC_CLASS __new_CROCEvaluation},
{"StratifiedCrossValidationSplitting", SHOGUN_BASIC_CLASS __new_CStratifiedCrossValidationSplitting},
{"StructuredAccuracy", SHOGUN_BASIC_CLASS __new_CStructuredAccuracy},
{"Alphabet", SHOGUN_BASIC_CLASS __new_CAlphabet},
{"BinnedDotFeatures", SHOGUN_BASIC_CLASS __new_CBinnedDotFeatures},
{"CombinedDotFeatures", SHOGUN_BASIC_CLASS __new_CCombinedDotFeatures},
{"CombinedFeatures", SHOGUN_BASIC_CLASS __new_CCombinedFeatures},
{"DataGenerator", SHOGUN_BASIC_CLASS __new_CDataGenerator},
{"DummyFeatures", SHOGUN_BASIC_CLASS __new_CDummyFeatures},
{"ExplicitSpecFeatures", SHOGUN_BASIC_CLASS __new_CExplicitSpecFeatures},
{"FactorGraphFeatures", SHOGUN_BASIC_CLASS __new_CFactorGraphFeatures},
{"FKFeatures", SHOGUN_BASIC_CLASS __new_CFKFeatures},
{"HashedDocDotFeatures", SHOGUN_BASIC_CLASS __new_CHashedDocDotFeatures},
{"HashedWDFeatures", SHOGUN_BASIC_CLASS __new_CHashedWDFeatures},
{"HashedWDFeaturesTransposed", SHOGUN_BASIC_CLASS __new_CHashedWDFeaturesTransposed},
{"ImplicitWeightedSpecFeatures", SHOGUN_BASIC_CLASS __new_CImplicitWeightedSpecFeatures},
{"LatentFeatures", SHOGUN_BASIC_CLASS __new_CLatentFeatures},
{"LBPPyrDotFeatures", SHOGUN_BASIC_CLASS __new_CLBPPyrDotFeatures},
{"PolyFeatures", SHOGUN_BASIC_CLASS __new_CPolyFeatures},
{"RandomFourierDotFeatures", SHOGUN_BASIC_CLASS __new_CRandomFourierDotFeatures},
{"RealFileFeatures", SHOGUN_BASIC_CLASS __new_CRealFileFeatures},
{"SNPFeatures", SHOGUN_BASIC_CLASS __new_CSNPFeatures},
{"SparsePolyFeatures", SHOGUN_BASIC_CLASS __new_CSparsePolyFeatures},
{"GaussianBlobsDataGenerator", SHOGUN_BASIC_CLASS __new_CGaussianBlobsDataGenerator},
{"MeanShiftDataGenerator", SHOGUN_BASIC_CLASS __new_CMeanShiftDataGenerator},
{"StreamingHashedDocDotFeatures", SHOGUN_BASIC_CLASS __new_CStreamingHashedDocDotFeatures},
{"StreamingVwFeatures", SHOGUN_BASIC_CLASS __new_CStreamingVwFeatures},
{"Subset", SHOGUN_BASIC_CLASS __new_CSubset},
{"SubsetStack", SHOGUN_BASIC_CLASS __new_CSubsetStack},
{"TOPFeatures", SHOGUN_BASIC_CLASS __new_CTOPFeatures},
{"WDFeatures", SHOGUN_BASIC_CLASS __new_CWDFeatures},
{"BinaryFile", SHOGUN_BASIC_CLASS __new_CBinaryFile},
{"CSVFile", SHOGUN_BASIC_CLASS __new_CCSVFile},
{"IOBuffer", SHOGUN_BASIC_CLASS __new_CIOBuffer},
{"LibSVMFile", SHOGUN_BASIC_CLASS __new_CLibSVMFile},
{"LineReader", SHOGUN_BASIC_CLASS __new_CLineReader},
{"Parser", SHOGUN_BASIC_CLASS __new_CParser},
{"SerializableAsciiFile", SHOGUN_BASIC_CLASS __new_CSerializableAsciiFile},
{"StreamingAsciiFile", SHOGUN_BASIC_CLASS __new_CStreamingAsciiFile},
{"StreamingFile", SHOGUN_BASIC_CLASS __new_CStreamingFile},
{"StreamingFileFromFeatures", SHOGUN_BASIC_CLASS __new_CStreamingFileFromFeatures},
{"StreamingVwCacheFile", SHOGUN_BASIC_CLASS __new_CStreamingVwCacheFile},
{"StreamingVwFile", SHOGUN_BASIC_CLASS __new_CStreamingVwFile},
{"ANOVAKernel", SHOGUN_BASIC_CLASS __new_CANOVAKernel},
{"AUCKernel", SHOGUN_BASIC_CLASS __new_CAUCKernel},
{"BesselKernel", SHOGUN_BASIC_CLASS __new_CBesselKernel},
{"CauchyKernel", SHOGUN_BASIC_CLASS __new_CCauchyKernel},
{"Chi2Kernel", SHOGUN_BASIC_CLASS __new_CChi2Kernel},
{"CircularKernel", SHOGUN_BASIC_CLASS __new_CCircularKernel},
{"CombinedKernel", SHOGUN_BASIC_CLASS __new_CCombinedKernel},
{"ConstKernel", SHOGUN_BASIC_CLASS __new_CConstKernel},
{"CustomKernel", SHOGUN_BASIC_CLASS __new_CCustomKernel},
{"DiagKernel", SHOGUN_BASIC_CLASS __new_CDiagKernel},
{"DistanceKernel", SHOGUN_BASIC_CLASS __new_CDistanceKernel},
{"ExponentialKernel", SHOGUN_BASIC_CLASS __new_CExponentialKernel},
{"GaussianARDKernel", SHOGUN_BASIC_CLASS __new_CGaussianARDKernel},
{"GaussianKernel", SHOGUN_BASIC_CLASS __new_CGaussianKernel},
{"GaussianShiftKernel", SHOGUN_BASIC_CLASS __new_CGaussianShiftKernel},
{"GaussianShortRealKernel", SHOGUN_BASIC_CLASS __new_CGaussianShortRealKernel},
{"HistogramIntersectionKernel", SHOGUN_BASIC_CLASS __new_CHistogramIntersectionKernel},
{"InverseMultiQuadricKernel", SHOGUN_BASIC_CLASS __new_CInverseMultiQuadricKernel},
{"JensenShannonKernel", SHOGUN_BASIC_CLASS __new_CJensenShannonKernel},
{"LinearARDKernel", SHOGUN_BASIC_CLASS __new_CLinearARDKernel},
{"LinearKernel", SHOGUN_BASIC_CLASS __new_CLinearKernel},
{"LogKernel", SHOGUN_BASIC_CLASS __new_CLogKernel},
{"MultiquadricKernel", SHOGUN_BASIC_CLASS __new_CMultiquadricKernel},
{"AvgDiagKernelNormalizer", SHOGUN_BASIC_CLASS __new_CAvgDiagKernelNormalizer},
{"DiceKernelNormalizer", SHOGUN_BASIC_CLASS __new_CDiceKernelNormalizer},
{"FirstElementKernelNormalizer", SHOGUN_BASIC_CLASS __new_CFirstElementKernelNormalizer},
{"IdentityKernelNormalizer", SHOGUN_BASIC_CLASS __new_CIdentityKernelNormalizer},
{"RidgeKernelNormalizer", SHOGUN_BASIC_CLASS __new_CRidgeKernelNormalizer},
{"ScatterKernelNormalizer", SHOGUN_BASIC_CLASS __new_CScatterKernelNormalizer},
{"SqrtDiagKernelNormalizer", SHOGUN_BASIC_CLASS __new_CSqrtDiagKernelNormalizer},
{"TanimotoKernelNormalizer", SHOGUN_BASIC_CLASS __new_CTanimotoKernelNormalizer},
{"VarianceKernelNormalizer", SHOGUN_BASIC_CLASS __new_CVarianceKernelNormalizer},
{"ZeroMeanCenterKernelNormalizer", SHOGUN_BASIC_CLASS __new_CZeroMeanCenterKernelNormalizer},
{"PolyKernel", SHOGUN_BASIC_CLASS __new_CPolyKernel},
{"PowerKernel", SHOGUN_BASIC_CLASS __new_CPowerKernel},
{"ProductKernel", SHOGUN_BASIC_CLASS __new_CProductKernel},
{"PyramidChi2", SHOGUN_BASIC_CLASS __new_CPyramidChi2},
{"RationalQuadraticKernel", SHOGUN_BASIC_CLASS __new_CRationalQuadraticKernel},
{"SigmoidKernel", SHOGUN_BASIC_CLASS __new_CSigmoidKernel},
{"SphericalKernel", SHOGUN_BASIC_CLASS __new_CSphericalKernel},
{"SplineKernel", SHOGUN_BASIC_CLASS __new_CSplineKernel},
{"CommUlongStringKernel", SHOGUN_BASIC_CLASS __new_CCommUlongStringKernel},
{"CommWordStringKernel", SHOGUN_BASIC_CLASS __new_CCommWordStringKernel},
{"DistantSegmentsKernel", SHOGUN_BASIC_CLASS __new_CDistantSegmentsKernel},
{"FixedDegreeStringKernel", SHOGUN_BASIC_CLASS __new_CFixedDegreeStringKernel},
{"GaussianMatchStringKernel", SHOGUN_BASIC_CLASS __new_CGaussianMatchStringKernel},
{"HistogramWordStringKernel", SHOGUN_BASIC_CLASS __new_CHistogramWordStringKernel},
{"LinearStringKernel", SHOGUN_BASIC_CLASS __new_CLinearStringKernel},
{"LocalAlignmentStringKernel", SHOGUN_BASIC_CLASS __new_CLocalAlignmentStringKernel},
{"LocalityImprovedStringKernel", SHOGUN_BASIC_CLASS __new_CLocalityImprovedStringKernel},
{"MatchWordStringKernel", SHOGUN_BASIC_CLASS __new_CMatchWordStringKernel},
{"OligoStringKernel", SHOGUN_BASIC_CLASS __new_COligoStringKernel},
{"PolyMatchStringKernel", SHOGUN_BASIC_CLASS __new_CPolyMatchStringKernel},
{"PolyMatchWordStringKernel", SHOGUN_BASIC_CLASS __new_CPolyMatchWordStringKernel},
{"RegulatoryModulesStringKernel", SHOGUN_BASIC_CLASS __new_CRegulatoryModulesStringKernel},
{"SalzbergWordStringKernel", SHOGUN_BASIC_CLASS __new_CSalzbergWordStringKernel},
{"SimpleLocalityImprovedStringKernel", SHOGUN_BASIC_CLASS __new_CSimpleLocalityImprovedStringKernel},
{"SNPStringKernel", SHOGUN_BASIC_CLASS __new_CSNPStringKernel},
{"SparseSpatialSampleStringKernel", SHOGUN_BASIC_CLASS __new_CSparseSpatialSampleStringKernel},
{"SpectrumMismatchRBFKernel", SHOGUN_BASIC_CLASS __new_CSpectrumMismatchRBFKernel},
{"SpectrumRBFKernel", SHOGUN_BASIC_CLASS __new_CSpectrumRBFKernel},
{"WeightedCommWordStringKernel", SHOGUN_BASIC_CLASS __new_CWeightedCommWordStringKernel},
{"WeightedDegreePositionStringKernel", SHOGUN_BASIC_CLASS __new_CWeightedDegreePositionStringKernel},
{"WeightedDegreeStringKernel", SHOGUN_BASIC_CLASS __new_CWeightedDegreeStringKernel},
{"TensorProductPairKernel", SHOGUN_BASIC_CLASS __new_CTensorProductPairKernel},
{"TStudentKernel", SHOGUN_BASIC_CLASS __new_CTStudentKernel},
{"WaveKernel", SHOGUN_BASIC_CLASS __new_CWaveKernel},
{"WaveletKernel", SHOGUN_BASIC_CLASS __new_CWaveletKernel},
{"WeightedDegreeRBFKernel", SHOGUN_BASIC_CLASS __new_CWeightedDegreeRBFKernel},
{"BinaryLabels", SHOGUN_BASIC_CLASS __new_CBinaryLabels},
{"FactorGraphObservation", SHOGUN_BASIC_CLASS __new_CFactorGraphObservation},
{"FactorGraphLabels", SHOGUN_BASIC_CLASS __new_CFactorGraphLabels},
{"LabelsFactory", SHOGUN_BASIC_CLASS __new_CLabelsFactory},
{"LatentLabels", SHOGUN_BASIC_CLASS __new_CLatentLabels},
{"MulticlassLabels", SHOGUN_BASIC_CLASS __new_CMulticlassLabels},
{"MulticlassMultipleOutputLabels", SHOGUN_BASIC_CLASS __new_CMulticlassMultipleOutputLabels},
{"RegressionLabels", SHOGUN_BASIC_CLASS __new_CRegressionLabels},
{"StructuredLabels", SHOGUN_BASIC_CLASS __new_CStructuredLabels},
{"LatentSOSVM", SHOGUN_BASIC_CLASS __new_CLatentSOSVM},
{"LatentSVM", SHOGUN_BASIC_CLASS __new_CLatentSVM},
{"BitString", SHOGUN_BASIC_CLASS __new_CBitString},
{"CircularBuffer", SHOGUN_BASIC_CLASS __new_CCircularBuffer},
{"Compressor", SHOGUN_BASIC_CLASS __new_CCompressor},
{"SerialComputationEngine", SHOGUN_BASIC_CLASS __new_CSerialComputationEngine},
{"JobResult", SHOGUN_BASIC_CLASS __new_CJobResult},
{"Data", SHOGUN_BASIC_CLASS __new_CData},
{"DelimiterTokenizer", SHOGUN_BASIC_CLASS __new_CDelimiterTokenizer},
{"DynamicObjectArray", SHOGUN_BASIC_CLASS __new_CDynamicObjectArray},
{"Hash", SHOGUN_BASIC_CLASS __new_CHash},
{"IndexBlock", SHOGUN_BASIC_CLASS __new_CIndexBlock},
{"IndexBlockGroup", SHOGUN_BASIC_CLASS __new_CIndexBlockGroup},
{"IndexBlockTree", SHOGUN_BASIC_CLASS __new_CIndexBlockTree},
{"ListElement", SHOGUN_BASIC_CLASS __new_CListElement},
{"List", SHOGUN_BASIC_CLASS __new_CList},
{"NGramTokenizer", SHOGUN_BASIC_CLASS __new_CNGramTokenizer},
{"Signal", SHOGUN_BASIC_CLASS __new_CSignal},
{"StructuredData", SHOGUN_BASIC_CLASS __new_CStructuredData},
{"Time", SHOGUN_BASIC_CLASS __new_CTime},
{"HingeLoss", SHOGUN_BASIC_CLASS __new_CHingeLoss},
{"LogLoss", SHOGUN_BASIC_CLASS __new_CLogLoss},
{"LogLossMargin", SHOGUN_BASIC_CLASS __new_CLogLossMargin},
{"SmoothHingeLoss", SHOGUN_BASIC_CLASS __new_CSmoothHingeLoss},
{"SquaredHingeLoss", SHOGUN_BASIC_CLASS __new_CSquaredHingeLoss},
{"SquaredLoss", SHOGUN_BASIC_CLASS __new_CSquaredLoss},
{"BaggingMachine", SHOGUN_BASIC_CLASS __new_CBaggingMachine},
{"BaseMulticlassMachine", SHOGUN_BASIC_CLASS __new_CBaseMulticlassMachine},
{"DistanceMachine", SHOGUN_BASIC_CLASS __new_CDistanceMachine},
{"ZeroMean", SHOGUN_BASIC_CLASS __new_CZeroMean},
{"KernelMachine", SHOGUN_BASIC_CLASS __new_CKernelMachine},
{"KernelMulticlassMachine", SHOGUN_BASIC_CLASS __new_CKernelMulticlassMachine},
{"KernelStructuredOutputMachine", SHOGUN_BASIC_CLASS __new_CKernelStructuredOutputMachine},
{"LinearMachine", SHOGUN_BASIC_CLASS __new_CLinearMachine},
{"LinearMulticlassMachine", SHOGUN_BASIC_CLASS __new_CLinearMulticlassMachine},
{"LinearStructuredOutputMachine", SHOGUN_BASIC_CLASS __new_CLinearStructuredOutputMachine},
{"Machine", SHOGUN_BASIC_CLASS __new_CMachine},
{"NativeMulticlassMachine", SHOGUN_BASIC_CLASS __new_CNativeMulticlassMachine},
{"OnlineLinearMachine", SHOGUN_BASIC_CLASS __new_COnlineLinearMachine},
{"StructuredOutputMachine", SHOGUN_BASIC_CLASS __new_CStructuredOutputMachine},
{"JacobiEllipticFunctions", SHOGUN_BASIC_CLASS __new_CJacobiEllipticFunctions},
{"LogDetEstimator", SHOGUN_BASIC_CLASS __new_CLogDetEstimator},
{"NormalSampler", SHOGUN_BASIC_CLASS __new_CNormalSampler},
{"Math", SHOGUN_BASIC_CLASS __new_CMath},
{"Random", SHOGUN_BASIC_CLASS __new_CRandom},
{"SparseInverseCovariance", SHOGUN_BASIC_CLASS __new_CSparseInverseCovariance},
{"Statistics", SHOGUN_BASIC_CLASS __new_CStatistics},
{"GridSearchModelSelection", SHOGUN_BASIC_CLASS __new_CGridSearchModelSelection},
{"ModelSelectionParameters", SHOGUN_BASIC_CLASS __new_CModelSelectionParameters},
{"ParameterCombination", SHOGUN_BASIC_CLASS __new_CParameterCombination},
{"RandomSearchModelSelection", SHOGUN_BASIC_CLASS __new_CRandomSearchModelSelection},
{"ECOCAEDDecoder", SHOGUN_BASIC_CLASS __new_CECOCAEDDecoder},
{"ECOCDiscriminantEncoder", SHOGUN_BASIC_CLASS __new_CECOCDiscriminantEncoder},
{"ECOCEDDecoder", SHOGUN_BASIC_CLASS __new_CECOCEDDecoder},
{"ECOCForestEncoder", SHOGUN_BASIC_CLASS __new_CECOCForestEncoder},
{"ECOCHDDecoder", SHOGUN_BASIC_CLASS __new_CECOCHDDecoder},
{"ECOCLLBDecoder", SHOGUN_BASIC_CLASS __new_CECOCLLBDecoder},
{"ECOCOVOEncoder", SHOGUN_BASIC_CLASS __new_CECOCOVOEncoder},
{"ECOCOVREncoder", SHOGUN_BASIC_CLASS __new_CECOCOVREncoder},
{"ECOCRandomDenseEncoder", SHOGUN_BASIC_CLASS __new_CECOCRandomDenseEncoder},
{"ECOCRandomSparseEncoder", SHOGUN_BASIC_CLASS __new_CECOCRandomSparseEncoder},
{"ECOCStrategy", SHOGUN_BASIC_CLASS __new_CECOCStrategy},
{"GaussianNaiveBayes", SHOGUN_BASIC_CLASS __new_CGaussianNaiveBayes},
{"GMNPLib", SHOGUN_BASIC_CLASS __new_CGMNPLib},
{"GMNPSVM", SHOGUN_BASIC_CLASS __new_CGMNPSVM},
{"KNN", SHOGUN_BASIC_CLASS __new_CKNN},
{"LaRank", SHOGUN_BASIC_CLASS __new_CLaRank},
{"MulticlassLibLinear", SHOGUN_BASIC_CLASS __new_CMulticlassLibLinear},
{"MulticlassLibSVM", SHOGUN_BASIC_CLASS __new_CMulticlassLibSVM},
{"MulticlassOCAS", SHOGUN_BASIC_CLASS __new_CMulticlassOCAS},
{"MulticlassOneVsOneStrategy", SHOGUN_BASIC_CLASS __new_CMulticlassOneVsOneStrategy},
{"MulticlassOneVsRestStrategy", SHOGUN_BASIC_CLASS __new_CMulticlassOneVsRestStrategy},
{"MulticlassSVM", SHOGUN_BASIC_CLASS __new_CMulticlassSVM},
{"ThresholdRejectionStrategy", SHOGUN_BASIC_CLASS __new_CThresholdRejectionStrategy},
{"DixonQTestRejectionStrategy", SHOGUN_BASIC_CLASS __new_CDixonQTestRejectionStrategy},
{"ScatterSVM", SHOGUN_BASIC_CLASS __new_CScatterSVM},
{"ShareBoost", SHOGUN_BASIC_CLASS __new_CShareBoost},
{"BalancedConditionalProbabilityTree", SHOGUN_BASIC_CLASS __new_CBalancedConditionalProbabilityTree},
{"RandomConditionalProbabilityTree", SHOGUN_BASIC_CLASS __new_CRandomConditionalProbabilityTree},
{"RelaxedTree", SHOGUN_BASIC_CLASS __new_CRelaxedTree},
{"Tron", SHOGUN_BASIC_CLASS __new_CTron},
{"DimensionReductionPreprocessor", SHOGUN_BASIC_CLASS __new_CDimensionReductionPreprocessor},
{"HomogeneousKernelMap", SHOGUN_BASIC_CLASS __new_CHomogeneousKernelMap},
{"LogPlusOne", SHOGUN_BASIC_CLASS __new_CLogPlusOne},
{"NormOne", SHOGUN_BASIC_CLASS __new_CNormOne},
{"PNorm", SHOGUN_BASIC_CLASS __new_CPNorm},
{"PruneVarSubMean", SHOGUN_BASIC_CLASS __new_CPruneVarSubMean},
{"RandomFourierGaussPreproc", SHOGUN_BASIC_CLASS __new_CRandomFourierGaussPreproc},
{"RescaleFeatures", SHOGUN_BASIC_CLASS __new_CRescaleFeatures},
{"SortUlongString", SHOGUN_BASIC_CLASS __new_CSortUlongString},
{"SortWordString", SHOGUN_BASIC_CLASS __new_CSortWordString},
{"SumOne", SHOGUN_BASIC_CLASS __new_CSumOne},
{"LibSVR", SHOGUN_BASIC_CLASS __new_CLibSVR},
{"MKLRegression", SHOGUN_BASIC_CLASS __new_CMKLRegression},
{"SVRLight", SHOGUN_BASIC_CLASS __new_CSVRLight},
{"HSIC", SHOGUN_BASIC_CLASS __new_CHSIC},
{"KernelMeanMatching", SHOGUN_BASIC_CLASS __new_CKernelMeanMatching},
{"LinearTimeMMD", SHOGUN_BASIC_CLASS __new_CLinearTimeMMD},
{"MMDKernelSelectionCombMaxL2", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionCombMaxL2},
{"MMDKernelSelectionCombOpt", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionCombOpt},
{"MMDKernelSelectionMax", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionMax},
{"MMDKernelSelectionMedian", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionMedian},
{"MMDKernelSelectionOpt", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionOpt},
{"QuadraticTimeMMD", SHOGUN_BASIC_CLASS __new_CQuadraticTimeMMD},
{"CCSOSVM", SHOGUN_BASIC_CLASS __new_CCCSOSVM},
{"DisjointSet", SHOGUN_BASIC_CLASS __new_CDisjointSet},
{"DualLibQPBMSOSVM", SHOGUN_BASIC_CLASS __new_CDualLibQPBMSOSVM},
{"DynProg", SHOGUN_BASIC_CLASS __new_CDynProg},
{"FactorDataSource", SHOGUN_BASIC_CLASS __new_CFactorDataSource},
{"Factor", SHOGUN_BASIC_CLASS __new_CFactor},
{"FactorGraph", SHOGUN_BASIC_CLASS __new_CFactorGraph},
{"FactorGraphModel", SHOGUN_BASIC_CLASS __new_CFactorGraphModel},
{"FactorType", SHOGUN_BASIC_CLASS __new_CFactorType},
{"TableFactorType", SHOGUN_BASIC_CLASS __new_CTableFactorType},
{"HMSVMModel", SHOGUN_BASIC_CLASS __new_CHMSVMModel},
{"IntronList", SHOGUN_BASIC_CLASS __new_CIntronList},
{"MAPInference", SHOGUN_BASIC_CLASS __new_CMAPInference},
{"MulticlassModel", SHOGUN_BASIC_CLASS __new_CMulticlassModel},
{"MulticlassSOLabels", SHOGUN_BASIC_CLASS __new_CMulticlassSOLabels},
{"Plif", SHOGUN_BASIC_CLASS __new_CPlif},
{"PlifArray", SHOGUN_BASIC_CLASS __new_CPlifArray},
{"PlifMatrix", SHOGUN_BASIC_CLASS __new_CPlifMatrix},
{"SegmentLoss", SHOGUN_BASIC_CLASS __new_CSegmentLoss},
{"Sequence", SHOGUN_BASIC_CLASS __new_CSequence},
{"SequenceLabels", SHOGUN_BASIC_CLASS __new_CSequenceLabels},
{"SOSVMHelper", SHOGUN_BASIC_CLASS __new_CSOSVMHelper},
{"StochasticSOSVM", SHOGUN_BASIC_CLASS __new_CStochasticSOSVM},
{"TwoStateModel", SHOGUN_BASIC_CLASS __new_CTwoStateModel},
{"DomainAdaptationSVM", SHOGUN_BASIC_CLASS __new_CDomainAdaptationSVM},
{"MultitaskClusteredLogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskClusteredLogisticRegression},
{"MultitaskKernelMaskNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelMaskNormalizer},
{"MultitaskKernelMaskPairNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelMaskPairNormalizer},
{"MultitaskKernelNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelNormalizer},
{"MultitaskKernelPlifNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelPlifNormalizer},
{"Node", SHOGUN_BASIC_CLASS __new_CNode},
{"Taxonomy", SHOGUN_BASIC_CLASS __new_CTaxonomy},
{"MultitaskKernelTreeNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelTreeNormalizer},
{"MultitaskL12LogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskL12LogisticRegression},
{"MultitaskLeastSquaresRegression", SHOGUN_BASIC_CLASS __new_CMultitaskLeastSquaresRegression},
{"MultitaskLinearMachine", SHOGUN_BASIC_CLASS __new_CMultitaskLinearMachine},
{"MultitaskLogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskLogisticRegression},
{"MultitaskROCEvaluation", SHOGUN_BASIC_CLASS __new_CMultitaskROCEvaluation},
{"MultitaskTraceLogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskTraceLogisticRegression},
{"Task", SHOGUN_BASIC_CLASS __new_CTask},
{"TaskGroup", SHOGUN_BASIC_CLASS __new_CTaskGroup},
{"TaskTree", SHOGUN_BASIC_CLASS __new_CTaskTree},
{"GUIClassifier", SHOGUN_BASIC_CLASS __new_CGUIClassifier},
{"GUIConverter", SHOGUN_BASIC_CLASS __new_CGUIConverter},
{"GUIDistance", SHOGUN_BASIC_CLASS __new_CGUIDistance},
{"GUIFeatures", SHOGUN_BASIC_CLASS __new_CGUIFeatures},
{"GUIHMM", SHOGUN_BASIC_CLASS __new_CGUIHMM},
{"GUIKernel", SHOGUN_BASIC_CLASS __new_CGUIKernel},
{"GUILabels", SHOGUN_BASIC_CLASS __new_CGUILabels},
{"GUIMath", SHOGUN_BASIC_CLASS __new_CGUIMath},
{"GUIPluginEstimate", SHOGUN_BASIC_CLASS __new_CGUIPluginEstimate},
{"GUIPreprocessor", SHOGUN_BASIC_CLASS __new_CGUIPreprocessor},
{"GUIStructure", SHOGUN_BASIC_CLASS __new_CGUIStructure},
{"GUITime", SHOGUN_BASIC_CLASS __new_CGUITime},
{"AveragedPerceptron", SHOGUN_BASIC_CLASS __new_CAveragedPerceptron},
{"FeatureBlockLogisticRegression", SHOGUN_BASIC_CLASS __new_CFeatureBlockLogisticRegression},
{"MKLClassification", SHOGUN_BASIC_CLASS __new_CMKLClassification},
{"MKLMulticlass", SHOGUN_BASIC_CLASS __new_CMKLMulticlass},
{"MKLOneClass", SHOGUN_BASIC_CLASS __new_CMKLOneClass},
{"NearestCentroid", SHOGUN_BASIC_CLASS __new_CNearestCentroid},
{"Perceptron", SHOGUN_BASIC_CLASS __new_CPerceptron},
{"PluginEstimate", SHOGUN_BASIC_CLASS __new_CPluginEstimate},
{"GNPPLib", SHOGUN_BASIC_CLASS __new_CGNPPLib},
{"GNPPSVM", SHOGUN_BASIC_CLASS __new_CGNPPSVM},
{"GPBTSVM", SHOGUN_BASIC_CLASS __new_CGPBTSVM},
{"LibLinear", SHOGUN_BASIC_CLASS __new_CLibLinear},
{"LibSVM", SHOGUN_BASIC_CLASS __new_CLibSVM},
{"LibSVMOneClass", SHOGUN_BASIC_CLASS __new_CLibSVMOneClass},
{"MPDSVM", SHOGUN_BASIC_CLASS __new_CMPDSVM},
{"OnlineLibLinear", SHOGUN_BASIC_CLASS __new_COnlineLibLinear},
{"OnlineSVMSGD", SHOGUN_BASIC_CLASS __new_COnlineSVMSGD},
{"QPBSVMLib", SHOGUN_BASIC_CLASS __new_CQPBSVMLib},
{"SGDQN", SHOGUN_BASIC_CLASS __new_CSGDQN},
{"SVM", SHOGUN_BASIC_CLASS __new_CSVM},
{"SVMLight", SHOGUN_BASIC_CLASS __new_CSVMLight},
{"SVMLightOneClass", SHOGUN_BASIC_CLASS __new_CSVMLightOneClass},
{"SVMLin", SHOGUN_BASIC_CLASS __new_CSVMLin},
{"SVMOcas", SHOGUN_BASIC_CLASS __new_CSVMOcas},
{"SVMSGD", SHOGUN_BASIC_CLASS __new_CSVMSGD},
{"WDSVMOcas", SHOGUN_BASIC_CLASS __new_CWDSVMOcas},
{"VwNativeCacheReader", SHOGUN_BASIC_CLASS __new_CVwNativeCacheReader},
{"VwNativeCacheWriter", SHOGUN_BASIC_CLASS __new_CVwNativeCacheWriter},
{"VwAdaptiveLearner", SHOGUN_BASIC_CLASS __new_CVwAdaptiveLearner},
{"VwNonAdaptiveLearner", SHOGUN_BASIC_CLASS __new_CVwNonAdaptiveLearner},
{"VowpalWabbit", SHOGUN_BASIC_CLASS __new_CVowpalWabbit},
{"VwEnvironment", SHOGUN_BASIC_CLASS __new_CVwEnvironment},
{"VwParser", SHOGUN_BASIC_CLASS __new_CVwParser},
{"VwRegressor", SHOGUN_BASIC_CLASS __new_CVwRegressor},
{"Hierarchical", SHOGUN_BASIC_CLASS __new_CHierarchical},
{"KMeans", SHOGUN_BASIC_CLASS __new_CKMeans},
{"HashedDocConverter", SHOGUN_BASIC_CLASS __new_CHashedDocConverter},
{"AttenuatedEuclideanDistance", SHOGUN_BASIC_CLASS __new_CAttenuatedEuclideanDistance},
{"BrayCurtisDistance", SHOGUN_BASIC_CLASS __new_CBrayCurtisDistance},
{"CanberraMetric", SHOGUN_BASIC_CLASS __new_CCanberraMetric},
{"CanberraWordDistance", SHOGUN_BASIC_CLASS __new_CCanberraWordDistance},
{"ChebyshewMetric", SHOGUN_BASIC_CLASS __new_CChebyshewMetric},
{"ChiSquareDistance", SHOGUN_BASIC_CLASS __new_CChiSquareDistance},
{"CosineDistance", SHOGUN_BASIC_CLASS __new_CCosineDistance},
{"CustomDistance", SHOGUN_BASIC_CLASS __new_CCustomDistance},
{"EuclideanDistance", SHOGUN_BASIC_CLASS __new_CEuclideanDistance},
{"GeodesicMetric", SHOGUN_BASIC_CLASS __new_CGeodesicMetric},
{"HammingWordDistance", SHOGUN_BASIC_CLASS __new_CHammingWordDistance},
{"JensenMetric", SHOGUN_BASIC_CLASS __new_CJensenMetric},
{"KernelDistance", SHOGUN_BASIC_CLASS __new_CKernelDistance},
{"ManhattanMetric", SHOGUN_BASIC_CLASS __new_CManhattanMetric},
{"ManhattanWordDistance", SHOGUN_BASIC_CLASS __new_CManhattanWordDistance},
{"MinkowskiMetric", SHOGUN_BASIC_CLASS __new_CMinkowskiMetric},
{"SparseEuclideanDistance", SHOGUN_BASIC_CLASS __new_CSparseEuclideanDistance},
{"TanimotoDistance", SHOGUN_BASIC_CLASS __new_CTanimotoDistance},
{"GHMM", SHOGUN_BASIC_CLASS __new_CGHMM},
{"Histogram", SHOGUN_BASIC_CLASS __new_CHistogram},
{"HMM", SHOGUN_BASIC_CLASS __new_CHMM},
{"LinearHMM", SHOGUN_BASIC_CLASS __new_CLinearHMM},
{"PositionalPWM", SHOGUN_BASIC_CLASS __new_CPositionalPWM},
{"MajorityVote", SHOGUN_BASIC_CLASS __new_CMajorityVote},
{"MeanRule", SHOGUN_BASIC_CLASS __new_CMeanRule},
{"WeightedMajorityVote", SHOGUN_BASIC_CLASS __new_CWeightedMajorityVote},
{"ClusteringAccuracy", SHOGUN_BASIC_CLASS __new_CClusteringAccuracy},
{"ClusteringMutualInformation", SHOGUN_BASIC_CLASS __new_CClusteringMutualInformation},
{"ContingencyTableEvaluation", SHOGUN_BASIC_CLASS __new_CContingencyTableEvaluation},
{"AccuracyMeasure", SHOGUN_BASIC_CLASS __new_CAccuracyMeasure},
{"ErrorRateMeasure", SHOGUN_BASIC_CLASS __new_CErrorRateMeasure},
{"BALMeasure", SHOGUN_BASIC_CLASS __new_CBALMeasure},
{"WRACCMeasure", SHOGUN_BASIC_CLASS __new_CWRACCMeasure},
{"F1Measure", SHOGUN_BASIC_CLASS __new_CF1Measure},
{"CrossCorrelationMeasure", SHOGUN_BASIC_CLASS __new_CCrossCorrelationMeasure},
{"RecallMeasure", SHOGUN_BASIC_CLASS __new_CRecallMeasure},
{"PrecisionMeasure", SHOGUN_BASIC_CLASS __new_CPrecisionMeasure},
{"SpecificityMeasure", SHOGUN_BASIC_CLASS __new_CSpecificityMeasure},
{"CrossValidationResult", SHOGUN_BASIC_CLASS __new_CCrossValidationResult},
{"CrossValidation", SHOGUN_BASIC_CLASS __new_CCrossValidation},
{"CrossValidationMKLStorage", SHOGUN_BASIC_CLASS __new_CCrossValidationMKLStorage},
{"CrossValidationMulticlassStorage", SHOGUN_BASIC_CLASS __new_CCrossValidationMulticlassStorage},
{"CrossValidationPrintOutput", SHOGUN_BASIC_CLASS __new_CCrossValidationPrintOutput},
{"CrossValidationSplitting", SHOGUN_BASIC_CLASS __new_CCrossValidationSplitting},
{"GradientCriterion", SHOGUN_BASIC_CLASS __new_CGradientCriterion},
{"GradientEvaluation", SHOGUN_BASIC_CLASS __new_CGradientEvaluation},
{"GradientResult", SHOGUN_BASIC_CLASS __new_CGradientResult},
{"MeanAbsoluteError", SHOGUN_BASIC_CLASS __new_CMeanAbsoluteError},
{"MeanSquaredError", SHOGUN_BASIC_CLASS __new_CMeanSquaredError},
{"MeanSquaredLogError", SHOGUN_BASIC_CLASS __new_CMeanSquaredLogError},
{"MulticlassAccuracy", SHOGUN_BASIC_CLASS __new_CMulticlassAccuracy},
{"MulticlassOVREvaluation", SHOGUN_BASIC_CLASS __new_CMulticlassOVREvaluation},
{"PRCEvaluation", SHOGUN_BASIC_CLASS __new_CPRCEvaluation},
{"ROCEvaluation", SHOGUN_BASIC_CLASS __new_CROCEvaluation},
{"StratifiedCrossValidationSplitting", SHOGUN_BASIC_CLASS __new_CStratifiedCrossValidationSplitting},
{"StructuredAccuracy", SHOGUN_BASIC_CLASS __new_CStructuredAccuracy},
{"Alphabet", SHOGUN_BASIC_CLASS __new_CAlphabet},
{"BinnedDotFeatures", SHOGUN_BASIC_CLASS __new_CBinnedDotFeatures},
{"CombinedDotFeatures", SHOGUN_BASIC_CLASS __new_CCombinedDotFeatures},
{"CombinedFeatures", SHOGUN_BASIC_CLASS __new_CCombinedFeatures},
{"DataGenerator", SHOGUN_BASIC_CLASS __new_CDataGenerator},
{"DummyFeatures", SHOGUN_BASIC_CLASS __new_CDummyFeatures},
{"ExplicitSpecFeatures", SHOGUN_BASIC_CLASS __new_CExplicitSpecFeatures},
{"FactorGraphFeatures", SHOGUN_BASIC_CLASS __new_CFactorGraphFeatures},
{"FKFeatures", SHOGUN_BASIC_CLASS __new_CFKFeatures},
{"HashedDocDotFeatures", SHOGUN_BASIC_CLASS __new_CHashedDocDotFeatures},
{"HashedWDFeatures", SHOGUN_BASIC_CLASS __new_CHashedWDFeatures},
{"HashedWDFeaturesTransposed", SHOGUN_BASIC_CLASS __new_CHashedWDFeaturesTransposed},
{"ImplicitWeightedSpecFeatures", SHOGUN_BASIC_CLASS __new_CImplicitWeightedSpecFeatures},
{"LatentFeatures", SHOGUN_BASIC_CLASS __new_CLatentFeatures},
{"LBPPyrDotFeatures", SHOGUN_BASIC_CLASS __new_CLBPPyrDotFeatures},
{"PolyFeatures", SHOGUN_BASIC_CLASS __new_CPolyFeatures},
{"RandomFourierDotFeatures", SHOGUN_BASIC_CLASS __new_CRandomFourierDotFeatures},
{"RealFileFeatures", SHOGUN_BASIC_CLASS __new_CRealFileFeatures},
{"SNPFeatures", SHOGUN_BASIC_CLASS __new_CSNPFeatures},
{"SparsePolyFeatures", SHOGUN_BASIC_CLASS __new_CSparsePolyFeatures},
{"GaussianBlobsDataGenerator", SHOGUN_BASIC_CLASS __new_CGaussianBlobsDataGenerator},
{"MeanShiftDataGenerator", SHOGUN_BASIC_CLASS __new_CMeanShiftDataGenerator},
{"StreamingHashedDocDotFeatures", SHOGUN_BASIC_CLASS __new_CStreamingHashedDocDotFeatures},
{"StreamingVwFeatures", SHOGUN_BASIC_CLASS __new_CStreamingVwFeatures},
{"Subset", SHOGUN_BASIC_CLASS __new_CSubset},
{"SubsetStack", SHOGUN_BASIC_CLASS __new_CSubsetStack},
{"TOPFeatures", SHOGUN_BASIC_CLASS __new_CTOPFeatures},
{"WDFeatures", SHOGUN_BASIC_CLASS __new_CWDFeatures},
{"BinaryFile", SHOGUN_BASIC_CLASS __new_CBinaryFile},
{"CSVFile", SHOGUN_BASIC_CLASS __new_CCSVFile},
{"IOBuffer", SHOGUN_BASIC_CLASS __new_CIOBuffer},
{"LibSVMFile", SHOGUN_BASIC_CLASS __new_CLibSVMFile},
{"LineReader", SHOGUN_BASIC_CLASS __new_CLineReader},
{"Parser", SHOGUN_BASIC_CLASS __new_CParser},
{"SerializableAsciiFile", SHOGUN_BASIC_CLASS __new_CSerializableAsciiFile},
{"StreamingAsciiFile", SHOGUN_BASIC_CLASS __new_CStreamingAsciiFile},
{"StreamingFile", SHOGUN_BASIC_CLASS __new_CStreamingFile},
{"StreamingFileFromFeatures", SHOGUN_BASIC_CLASS __new_CStreamingFileFromFeatures},
{"StreamingVwCacheFile", SHOGUN_BASIC_CLASS __new_CStreamingVwCacheFile},
{"StreamingVwFile", SHOGUN_BASIC_CLASS __new_CStreamingVwFile},
{"ANOVAKernel", SHOGUN_BASIC_CLASS __new_CANOVAKernel},
{"AUCKernel", SHOGUN_BASIC_CLASS __new_CAUCKernel},
{"BesselKernel", SHOGUN_BASIC_CLASS __new_CBesselKernel},
{"CauchyKernel", SHOGUN_BASIC_CLASS __new_CCauchyKernel},
{"Chi2Kernel", SHOGUN_BASIC_CLASS __new_CChi2Kernel},
{"CircularKernel", SHOGUN_BASIC_CLASS __new_CCircularKernel},
{"CombinedKernel", SHOGUN_BASIC_CLASS __new_CCombinedKernel},
{"ConstKernel", SHOGUN_BASIC_CLASS __new_CConstKernel},
{"CustomKernel", SHOGUN_BASIC_CLASS __new_CCustomKernel},
{"DiagKernel", SHOGUN_BASIC_CLASS __new_CDiagKernel},
{"DistanceKernel", SHOGUN_BASIC_CLASS __new_CDistanceKernel},
{"ExponentialKernel", SHOGUN_BASIC_CLASS __new_CExponentialKernel},
{"GaussianARDKernel", SHOGUN_BASIC_CLASS __new_CGaussianARDKernel},
{"GaussianKernel", SHOGUN_BASIC_CLASS __new_CGaussianKernel},
{"GaussianShiftKernel", SHOGUN_BASIC_CLASS __new_CGaussianShiftKernel},
{"GaussianShortRealKernel", SHOGUN_BASIC_CLASS __new_CGaussianShortRealKernel},
{"HistogramIntersectionKernel", SHOGUN_BASIC_CLASS __new_CHistogramIntersectionKernel},
{"InverseMultiQuadricKernel", SHOGUN_BASIC_CLASS __new_CInverseMultiQuadricKernel},
{"JensenShannonKernel", SHOGUN_BASIC_CLASS __new_CJensenShannonKernel},
{"LinearARDKernel", SHOGUN_BASIC_CLASS __new_CLinearARDKernel},
{"LinearKernel", SHOGUN_BASIC_CLASS __new_CLinearKernel},
{"LogKernel", SHOGUN_BASIC_CLASS __new_CLogKernel},
{"MultiquadricKernel", SHOGUN_BASIC_CLASS __new_CMultiquadricKernel},
{"AvgDiagKernelNormalizer", SHOGUN_BASIC_CLASS __new_CAvgDiagKernelNormalizer},
{"DiceKernelNormalizer", SHOGUN_BASIC_CLASS __new_CDiceKernelNormalizer},
{"FirstElementKernelNormalizer", SHOGUN_BASIC_CLASS __new_CFirstElementKernelNormalizer},
{"IdentityKernelNormalizer", SHOGUN_BASIC_CLASS __new_CIdentityKernelNormalizer},
{"RidgeKernelNormalizer", SHOGUN_BASIC_CLASS __new_CRidgeKernelNormalizer},
{"ScatterKernelNormalizer", SHOGUN_BASIC_CLASS __new_CScatterKernelNormalizer},
{"SqrtDiagKernelNormalizer", SHOGUN_BASIC_CLASS __new_CSqrtDiagKernelNormalizer},
{"TanimotoKernelNormalizer", SHOGUN_BASIC_CLASS __new_CTanimotoKernelNormalizer},
{"VarianceKernelNormalizer", SHOGUN_BASIC_CLASS __new_CVarianceKernelNormalizer},
{"ZeroMeanCenterKernelNormalizer", SHOGUN_BASIC_CLASS __new_CZeroMeanCenterKernelNormalizer},
{"PolyKernel", SHOGUN_BASIC_CLASS __new_CPolyKernel},
{"PowerKernel", SHOGUN_BASIC_CLASS __new_CPowerKernel},
{"ProductKernel", SHOGUN_BASIC_CLASS __new_CProductKernel},
{"PyramidChi2", SHOGUN_BASIC_CLASS __new_CPyramidChi2},
{"RationalQuadraticKernel", SHOGUN_BASIC_CLASS __new_CRationalQuadraticKernel},
{"SigmoidKernel", SHOGUN_BASIC_CLASS __new_CSigmoidKernel},
{"SphericalKernel", SHOGUN_BASIC_CLASS __new_CSphericalKernel},
{"SplineKernel", SHOGUN_BASIC_CLASS __new_CSplineKernel},
{"CommUlongStringKernel", SHOGUN_BASIC_CLASS __new_CCommUlongStringKernel},
{"CommWordStringKernel", SHOGUN_BASIC_CLASS __new_CCommWordStringKernel},
{"DistantSegmentsKernel", SHOGUN_BASIC_CLASS __new_CDistantSegmentsKernel},
{"FixedDegreeStringKernel", SHOGUN_BASIC_CLASS __new_CFixedDegreeStringKernel},
{"GaussianMatchStringKernel", SHOGUN_BASIC_CLASS __new_CGaussianMatchStringKernel},
{"HistogramWordStringKernel", SHOGUN_BASIC_CLASS __new_CHistogramWordStringKernel},
{"LinearStringKernel", SHOGUN_BASIC_CLASS __new_CLinearStringKernel},
{"LocalAlignmentStringKernel", SHOGUN_BASIC_CLASS __new_CLocalAlignmentStringKernel},
{"LocalityImprovedStringKernel", SHOGUN_BASIC_CLASS __new_CLocalityImprovedStringKernel},
{"MatchWordStringKernel", SHOGUN_BASIC_CLASS __new_CMatchWordStringKernel},
{"OligoStringKernel", SHOGUN_BASIC_CLASS __new_COligoStringKernel},
{"PolyMatchStringKernel", SHOGUN_BASIC_CLASS __new_CPolyMatchStringKernel},
{"PolyMatchWordStringKernel", SHOGUN_BASIC_CLASS __new_CPolyMatchWordStringKernel},
{"RegulatoryModulesStringKernel", SHOGUN_BASIC_CLASS __new_CRegulatoryModulesStringKernel},
{"SalzbergWordStringKernel", SHOGUN_BASIC_CLASS __new_CSalzbergWordStringKernel},
{"SimpleLocalityImprovedStringKernel", SHOGUN_BASIC_CLASS __new_CSimpleLocalityImprovedStringKernel},
{"SNPStringKernel", SHOGUN_BASIC_CLASS __new_CSNPStringKernel},
{"SparseSpatialSampleStringKernel", SHOGUN_BASIC_CLASS __new_CSparseSpatialSampleStringKernel},
{"SpectrumMismatchRBFKernel", SHOGUN_BASIC_CLASS __new_CSpectrumMismatchRBFKernel},
{"SpectrumRBFKernel", SHOGUN_BASIC_CLASS __new_CSpectrumRBFKernel},
{"WeightedCommWordStringKernel", SHOGUN_BASIC_CLASS __new_CWeightedCommWordStringKernel},
{"WeightedDegreePositionStringKernel", SHOGUN_BASIC_CLASS __new_CWeightedDegreePositionStringKernel},
{"WeightedDegreeStringKernel", SHOGUN_BASIC_CLASS __new_CWeightedDegreeStringKernel},
{"TensorProductPairKernel", SHOGUN_BASIC_CLASS __new_CTensorProductPairKernel},
{"TStudentKernel", SHOGUN_BASIC_CLASS __new_CTStudentKernel},
{"WaveKernel", SHOGUN_BASIC_CLASS __new_CWaveKernel},
{"WaveletKernel", SHOGUN_BASIC_CLASS __new_CWaveletKernel},
{"WeightedDegreeRBFKernel", SHOGUN_BASIC_CLASS __new_CWeightedDegreeRBFKernel},
{"BinaryLabels", SHOGUN_BASIC_CLASS __new_CBinaryLabels},
{"FactorGraphObservation", SHOGUN_BASIC_CLASS __new_CFactorGraphObservation},
{"FactorGraphLabels", SHOGUN_BASIC_CLASS __new_CFactorGraphLabels},
{"LabelsFactory", SHOGUN_BASIC_CLASS __new_CLabelsFactory},
{"LatentLabels", SHOGUN_BASIC_CLASS __new_CLatentLabels},
{"MulticlassLabels", SHOGUN_BASIC_CLASS __new_CMulticlassLabels},
{"MulticlassMultipleOutputLabels", SHOGUN_BASIC_CLASS __new_CMulticlassMultipleOutputLabels},
{"RegressionLabels", SHOGUN_BASIC_CLASS __new_CRegressionLabels},
{"StructuredLabels", SHOGUN_BASIC_CLASS __new_CStructuredLabels},
{"LatentSOSVM", SHOGUN_BASIC_CLASS __new_CLatentSOSVM},
{"LatentSVM", SHOGUN_BASIC_CLASS __new_CLatentSVM},
{"BitString", SHOGUN_BASIC_CLASS __new_CBitString},
{"CircularBuffer", SHOGUN_BASIC_CLASS __new_CCircularBuffer},
{"Compressor", SHOGUN_BASIC_CLASS __new_CCompressor},
{"SerialComputationEngine", SHOGUN_BASIC_CLASS __new_CSerialComputationEngine},
{"JobResult", SHOGUN_BASIC_CLASS __new_CJobResult},
{"Data", SHOGUN_BASIC_CLASS __new_CData},
{"DelimiterTokenizer", SHOGUN_BASIC_CLASS __new_CDelimiterTokenizer},
{"DynamicObjectArray", SHOGUN_BASIC_CLASS __new_CDynamicObjectArray},
{"Hash", SHOGUN_BASIC_CLASS __new_CHash},
{"IndexBlock", SHOGUN_BASIC_CLASS __new_CIndexBlock},
{"IndexBlockGroup", SHOGUN_BASIC_CLASS __new_CIndexBlockGroup},
{"IndexBlockTree", SHOGUN_BASIC_CLASS __new_CIndexBlockTree},
{"ListElement", SHOGUN_BASIC_CLASS __new_CListElement},
{"List", SHOGUN_BASIC_CLASS __new_CList},
{"NGramTokenizer", SHOGUN_BASIC_CLASS __new_CNGramTokenizer},
{"Signal", SHOGUN_BASIC_CLASS __new_CSignal},
{"StructuredData", SHOGUN_BASIC_CLASS __new_CStructuredData},
{"Time", SHOGUN_BASIC_CLASS __new_CTime},
{"HingeLoss", SHOGUN_BASIC_CLASS __new_CHingeLoss},
{"LogLoss", SHOGUN_BASIC_CLASS __new_CLogLoss},
{"LogLossMargin", SHOGUN_BASIC_CLASS __new_CLogLossMargin},
{"SmoothHingeLoss", SHOGUN_BASIC_CLASS __new_CSmoothHingeLoss},
{"SquaredHingeLoss", SHOGUN_BASIC_CLASS __new_CSquaredHingeLoss},
{"SquaredLoss", SHOGUN_BASIC_CLASS __new_CSquaredLoss},
{"BaggingMachine", SHOGUN_BASIC_CLASS __new_CBaggingMachine},
{"BaseMulticlassMachine", SHOGUN_BASIC_CLASS __new_CBaseMulticlassMachine},
{"DistanceMachine", SHOGUN_BASIC_CLASS __new_CDistanceMachine},
{"ZeroMean", SHOGUN_BASIC_CLASS __new_CZeroMean},
{"KernelMachine", SHOGUN_BASIC_CLASS __new_CKernelMachine},
{"KernelMulticlassMachine", SHOGUN_BASIC_CLASS __new_CKernelMulticlassMachine},
{"KernelStructuredOutputMachine", SHOGUN_BASIC_CLASS __new_CKernelStructuredOutputMachine},
{"LinearMachine", SHOGUN_BASIC_CLASS __new_CLinearMachine},
{"LinearMulticlassMachine", SHOGUN_BASIC_CLASS __new_CLinearMulticlassMachine},
{"LinearStructuredOutputMachine", SHOGUN_BASIC_CLASS __new_CLinearStructuredOutputMachine},
{"Machine", SHOGUN_BASIC_CLASS __new_CMachine},
{"NativeMulticlassMachine", SHOGUN_BASIC_CLASS __new_CNativeMulticlassMachine},
{"OnlineLinearMachine", SHOGUN_BASIC_CLASS __new_COnlineLinearMachine},
{"StructuredOutputMachine", SHOGUN_BASIC_CLASS __new_CStructuredOutputMachine},
{"JacobiEllipticFunctions", SHOGUN_BASIC_CLASS __new_CJacobiEllipticFunctions},
{"LogDetEstimator", SHOGUN_BASIC_CLASS __new_CLogDetEstimator},
{"NormalSampler", SHOGUN_BASIC_CLASS __new_CNormalSampler},
{"Math", SHOGUN_BASIC_CLASS __new_CMath},
{"Random", SHOGUN_BASIC_CLASS __new_CRandom},
{"SparseInverseCovariance", SHOGUN_BASIC_CLASS __new_CSparseInverseCovariance},
{"Statistics", SHOGUN_BASIC_CLASS __new_CStatistics},
{"GridSearchModelSelection", SHOGUN_BASIC_CLASS __new_CGridSearchModelSelection},
{"ModelSelectionParameters", SHOGUN_BASIC_CLASS __new_CModelSelectionParameters},
{"ParameterCombination", SHOGUN_BASIC_CLASS __new_CParameterCombination},
{"RandomSearchModelSelection", SHOGUN_BASIC_CLASS __new_CRandomSearchModelSelection},
{"ECOCAEDDecoder", SHOGUN_BASIC_CLASS __new_CECOCAEDDecoder},
{"ECOCDiscriminantEncoder", SHOGUN_BASIC_CLASS __new_CECOCDiscriminantEncoder},
{"ECOCEDDecoder", SHOGUN_BASIC_CLASS __new_CECOCEDDecoder},
{"ECOCForestEncoder", SHOGUN_BASIC_CLASS __new_CECOCForestEncoder},
{"ECOCHDDecoder", SHOGUN_BASIC_CLASS __new_CECOCHDDecoder},
{"ECOCLLBDecoder", SHOGUN_BASIC_CLASS __new_CECOCLLBDecoder},
{"ECOCOVOEncoder", SHOGUN_BASIC_CLASS __new_CECOCOVOEncoder},
{"ECOCOVREncoder", SHOGUN_BASIC_CLASS __new_CECOCOVREncoder},
{"ECOCRandomDenseEncoder", SHOGUN_BASIC_CLASS __new_CECOCRandomDenseEncoder},
{"ECOCRandomSparseEncoder", SHOGUN_BASIC_CLASS __new_CECOCRandomSparseEncoder},
{"ECOCStrategy", SHOGUN_BASIC_CLASS __new_CECOCStrategy},
{"GaussianNaiveBayes", SHOGUN_BASIC_CLASS __new_CGaussianNaiveBayes},
{"GMNPLib", SHOGUN_BASIC_CLASS __new_CGMNPLib},
{"GMNPSVM", SHOGUN_BASIC_CLASS __new_CGMNPSVM},
{"KNN", SHOGUN_BASIC_CLASS __new_CKNN},
{"LaRank", SHOGUN_BASIC_CLASS __new_CLaRank},
{"MulticlassLibLinear", SHOGUN_BASIC_CLASS __new_CMulticlassLibLinear},
{"MulticlassLibSVM", SHOGUN_BASIC_CLASS __new_CMulticlassLibSVM},
{"MulticlassOCAS", SHOGUN_BASIC_CLASS __new_CMulticlassOCAS},
{"MulticlassOneVsOneStrategy", SHOGUN_BASIC_CLASS __new_CMulticlassOneVsOneStrategy},
{"MulticlassOneVsRestStrategy", SHOGUN_BASIC_CLASS __new_CMulticlassOneVsRestStrategy},
{"MulticlassSVM", SHOGUN_BASIC_CLASS __new_CMulticlassSVM},
{"ThresholdRejectionStrategy", SHOGUN_BASIC_CLASS __new_CThresholdRejectionStrategy},
{"DixonQTestRejectionStrategy", SHOGUN_BASIC_CLASS __new_CDixonQTestRejectionStrategy},
{"ScatterSVM", SHOGUN_BASIC_CLASS __new_CScatterSVM},
{"ShareBoost", SHOGUN_BASIC_CLASS __new_CShareBoost},
{"BalancedConditionalProbabilityTree", SHOGUN_BASIC_CLASS __new_CBalancedConditionalProbabilityTree},
{"RandomConditionalProbabilityTree", SHOGUN_BASIC_CLASS __new_CRandomConditionalProbabilityTree},
{"RelaxedTree", SHOGUN_BASIC_CLASS __new_CRelaxedTree},
{"Tron", SHOGUN_BASIC_CLASS __new_CTron},
{"DimensionReductionPreprocessor", SHOGUN_BASIC_CLASS __new_CDimensionReductionPreprocessor},
{"HomogeneousKernelMap", SHOGUN_BASIC_CLASS __new_CHomogeneousKernelMap},
{"LogPlusOne", SHOGUN_BASIC_CLASS __new_CLogPlusOne},
{"NormOne", SHOGUN_BASIC_CLASS __new_CNormOne},
{"PNorm", SHOGUN_BASIC_CLASS __new_CPNorm},
{"PruneVarSubMean", SHOGUN_BASIC_CLASS __new_CPruneVarSubMean},
{"RandomFourierGaussPreproc", SHOGUN_BASIC_CLASS __new_CRandomFourierGaussPreproc},
{"RescaleFeatures", SHOGUN_BASIC_CLASS __new_CRescaleFeatures},
{"SortUlongString", SHOGUN_BASIC_CLASS __new_CSortUlongString},
{"SortWordString", SHOGUN_BASIC_CLASS __new_CSortWordString},
{"SumOne", SHOGUN_BASIC_CLASS __new_CSumOne},
{"LibSVR", SHOGUN_BASIC_CLASS __new_CLibSVR},
{"MKLRegression", SHOGUN_BASIC_CLASS __new_CMKLRegression},
{"SVRLight", SHOGUN_BASIC_CLASS __new_CSVRLight},
{"HSIC", SHOGUN_BASIC_CLASS __new_CHSIC},
{"KernelMeanMatching", SHOGUN_BASIC_CLASS __new_CKernelMeanMatching},
{"LinearTimeMMD", SHOGUN_BASIC_CLASS __new_CLinearTimeMMD},
{"MMDKernelSelectionCombMaxL2", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionCombMaxL2},
{"MMDKernelSelectionCombOpt", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionCombOpt},
{"MMDKernelSelectionMax", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionMax},
{"MMDKernelSelectionMedian", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionMedian},
{"MMDKernelSelectionOpt", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionOpt},
{"QuadraticTimeMMD", SHOGUN_BASIC_CLASS __new_CQuadraticTimeMMD},
{"CCSOSVM", SHOGUN_BASIC_CLASS __new_CCCSOSVM},
{"DisjointSet", SHOGUN_BASIC_CLASS __new_CDisjointSet},
{"DualLibQPBMSOSVM", SHOGUN_BASIC_CLASS __new_CDualLibQPBMSOSVM},
{"DynProg", SHOGUN_BASIC_CLASS __new_CDynProg},
{"FactorDataSource", SHOGUN_BASIC_CLASS __new_CFactorDataSource},
{"Factor", SHOGUN_BASIC_CLASS __new_CFactor},
{"FactorGraph", SHOGUN_BASIC_CLASS __new_CFactorGraph},
{"FactorGraphModel", SHOGUN_BASIC_CLASS __new_CFactorGraphModel},
{"FactorType", SHOGUN_BASIC_CLASS __new_CFactorType},
{"TableFactorType", SHOGUN_BASIC_CLASS __new_CTableFactorType},
{"HMSVMModel", SHOGUN_BASIC_CLASS __new_CHMSVMModel},
{"IntronList", SHOGUN_BASIC_CLASS __new_CIntronList},
{"MAPInference", SHOGUN_BASIC_CLASS __new_CMAPInference},
{"MulticlassModel", SHOGUN_BASIC_CLASS __new_CMulticlassModel},
{"MulticlassSOLabels", SHOGUN_BASIC_CLASS __new_CMulticlassSOLabels},
{"Plif", SHOGUN_BASIC_CLASS __new_CPlif},
{"PlifArray", SHOGUN_BASIC_CLASS __new_CPlifArray},
{"PlifMatrix", SHOGUN_BASIC_CLASS __new_CPlifMatrix},
{"SegmentLoss", SHOGUN_BASIC_CLASS __new_CSegmentLoss},
{"Sequence", SHOGUN_BASIC_CLASS __new_CSequence},
{"SequenceLabels", SHOGUN_BASIC_CLASS __new_CSequenceLabels},
{"SOSVMHelper", SHOGUN_BASIC_CLASS __new_CSOSVMHelper},
{"StochasticSOSVM", SHOGUN_BASIC_CLASS __new_CStochasticSOSVM},
{"TwoStateModel", SHOGUN_BASIC_CLASS __new_CTwoStateModel},
{"DomainAdaptationSVM", SHOGUN_BASIC_CLASS __new_CDomainAdaptationSVM},
{"MultitaskClusteredLogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskClusteredLogisticRegression},
{"MultitaskKernelMaskNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelMaskNormalizer},
{"MultitaskKernelMaskPairNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelMaskPairNormalizer},
{"MultitaskKernelNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelNormalizer},
{"MultitaskKernelPlifNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelPlifNormalizer},
{"Node", SHOGUN_BASIC_CLASS __new_CNode},
{"Taxonomy", SHOGUN_BASIC_CLASS __new_CTaxonomy},
{"MultitaskKernelTreeNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelTreeNormalizer},
{"MultitaskL12LogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskL12LogisticRegression},
{"MultitaskLeastSquaresRegression", SHOGUN_BASIC_CLASS __new_CMultitaskLeastSquaresRegression},
{"MultitaskLinearMachine", SHOGUN_BASIC_CLASS __new_CMultitaskLinearMachine},
{"MultitaskLogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskLogisticRegression},
{"MultitaskROCEvaluation", SHOGUN_BASIC_CLASS __new_CMultitaskROCEvaluation},
{"MultitaskTraceLogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskTraceLogisticRegression},
{"Task", SHOGUN_BASIC_CLASS __new_CTask},
{"TaskGroup", SHOGUN_BASIC_CLASS __new_CTaskGroup},
{"TaskTree", SHOGUN_BASIC_CLASS __new_CTaskTree},
{"GUIClassifier", SHOGUN_BASIC_CLASS __new_CGUIClassifier},
{"GUIConverter", SHOGUN_BASIC_CLASS __new_CGUIConverter},
{"GUIDistance", SHOGUN_BASIC_CLASS __new_CGUIDistance},
{"GUIFeatures", SHOGUN_BASIC_CLASS __new_CGUIFeatures},
{"GUIHMM", SHOGUN_BASIC_CLASS __new_CGUIHMM},
{"GUIKernel", SHOGUN_BASIC_CLASS __new_CGUIKernel},
{"GUILabels", SHOGUN_BASIC_CLASS __new_CGUILabels},
{"GUIMath", SHOGUN_BASIC_CLASS __new_CGUIMath},
{"GUIPluginEstimate", SHOGUN_BASIC_CLASS __new_CGUIPluginEstimate},
{"GUIPreprocessor", SHOGUN_BASIC_CLASS __new_CGUIPreprocessor},
{"GUIStructure", SHOGUN_BASIC_CLASS __new_CGUIStructure},
{"GUITime", SHOGUN_BASIC_CLASS __new_CGUITime},
{"AveragedPerceptron", SHOGUN_BASIC_CLASS __new_CAveragedPerceptron},
{"FeatureBlockLogisticRegression", SHOGUN_BASIC_CLASS __new_CFeatureBlockLogisticRegression},
{"MKLClassification", SHOGUN_BASIC_CLASS __new_CMKLClassification},
{"MKLMulticlass", SHOGUN_BASIC_CLASS __new_CMKLMulticlass},
{"MKLOneClass", SHOGUN_BASIC_CLASS __new_CMKLOneClass},
{"NearestCentroid", SHOGUN_BASIC_CLASS __new_CNearestCentroid},
{"Perceptron", SHOGUN_BASIC_CLASS __new_CPerceptron},
{"PluginEstimate", SHOGUN_BASIC_CLASS __new_CPluginEstimate},
{"GNPPLib", SHOGUN_BASIC_CLASS __new_CGNPPLib},
{"GNPPSVM", SHOGUN_BASIC_CLASS __new_CGNPPSVM},
{"GPBTSVM", SHOGUN_BASIC_CLASS __new_CGPBTSVM},
{"LibLinear", SHOGUN_BASIC_CLASS __new_CLibLinear},
{"LibSVM", SHOGUN_BASIC_CLASS __new_CLibSVM},
{"LibSVMOneClass", SHOGUN_BASIC_CLASS __new_CLibSVMOneClass},
{"MPDSVM", SHOGUN_BASIC_CLASS __new_CMPDSVM},
{"OnlineLibLinear", SHOGUN_BASIC_CLASS __new_COnlineLibLinear},
{"OnlineSVMSGD", SHOGUN_BASIC_CLASS __new_COnlineSVMSGD},
{"QPBSVMLib", SHOGUN_BASIC_CLASS __new_CQPBSVMLib},
{"SGDQN", SHOGUN_BASIC_CLASS __new_CSGDQN},
{"SVM", SHOGUN_BASIC_CLASS __new_CSVM},
{"SVMLight", SHOGUN_BASIC_CLASS __new_CSVMLight},
{"SVMLightOneClass", SHOGUN_BASIC_CLASS __new_CSVMLightOneClass},
{"SVMLin", SHOGUN_BASIC_CLASS __new_CSVMLin},
{"SVMOcas", SHOGUN_BASIC_CLASS __new_CSVMOcas},
{"SVMSGD", SHOGUN_BASIC_CLASS __new_CSVMSGD},
{"WDSVMOcas", SHOGUN_BASIC_CLASS __new_CWDSVMOcas},
{"VwNativeCacheReader", SHOGUN_BASIC_CLASS __new_CVwNativeCacheReader},
{"VwNativeCacheWriter", SHOGUN_BASIC_CLASS __new_CVwNativeCacheWriter},
{"VwAdaptiveLearner", SHOGUN_BASIC_CLASS __new_CVwAdaptiveLearner},
{"VwNonAdaptiveLearner", SHOGUN_BASIC_CLASS __new_CVwNonAdaptiveLearner},
{"VowpalWabbit", SHOGUN_BASIC_CLASS __new_CVowpalWabbit},
{"VwEnvironment", SHOGUN_BASIC_CLASS __new_CVwEnvironment},
{"VwParser", SHOGUN_BASIC_CLASS __new_CVwParser},
{"VwRegressor", SHOGUN_BASIC_CLASS __new_CVwRegressor},
{"Hierarchical", SHOGUN_BASIC_CLASS __new_CHierarchical},
{"KMeans", SHOGUN_BASIC_CLASS __new_CKMeans},
{"HashedDocConverter", SHOGUN_BASIC_CLASS __new_CHashedDocConverter},
{"AttenuatedEuclideanDistance", SHOGUN_BASIC_CLASS __new_CAttenuatedEuclideanDistance},
{"BrayCurtisDistance", SHOGUN_BASIC_CLASS __new_CBrayCurtisDistance},
{"CanberraMetric", SHOGUN_BASIC_CLASS __new_CCanberraMetric},
{"CanberraWordDistance", SHOGUN_BASIC_CLASS __new_CCanberraWordDistance},
{"ChebyshewMetric", SHOGUN_BASIC_CLASS __new_CChebyshewMetric},
{"ChiSquareDistance", SHOGUN_BASIC_CLASS __new_CChiSquareDistance},
{"CosineDistance", SHOGUN_BASIC_CLASS __new_CCosineDistance},
{"CustomDistance", SHOGUN_BASIC_CLASS __new_CCustomDistance},
{"EuclideanDistance", SHOGUN_BASIC_CLASS __new_CEuclideanDistance},
{"GeodesicMetric", SHOGUN_BASIC_CLASS __new_CGeodesicMetric},
{"HammingWordDistance", SHOGUN_BASIC_CLASS __new_CHammingWordDistance},
{"JensenMetric", SHOGUN_BASIC_CLASS __new_CJensenMetric},
{"KernelDistance", SHOGUN_BASIC_CLASS __new_CKernelDistance},
{"ManhattanMetric", SHOGUN_BASIC_CLASS __new_CManhattanMetric},
{"ManhattanWordDistance", SHOGUN_BASIC_CLASS __new_CManhattanWordDistance},
{"MinkowskiMetric", SHOGUN_BASIC_CLASS __new_CMinkowskiMetric},
{"SparseEuclideanDistance", SHOGUN_BASIC_CLASS __new_CSparseEuclideanDistance},
{"TanimotoDistance", SHOGUN_BASIC_CLASS __new_CTanimotoDistance},
{"GHMM", SHOGUN_BASIC_CLASS __new_CGHMM},
{"Histogram", SHOGUN_BASIC_CLASS __new_CHistogram},
{"HMM", SHOGUN_BASIC_CLASS __new_CHMM},
{"LinearHMM", SHOGUN_BASIC_CLASS __new_CLinearHMM},
{"PositionalPWM", SHOGUN_BASIC_CLASS __new_CPositionalPWM},
{"MajorityVote", SHOGUN_BASIC_CLASS __new_CMajorityVote},
{"MeanRule", SHOGUN_BASIC_CLASS __new_CMeanRule},
{"WeightedMajorityVote", SHOGUN_BASIC_CLASS __new_CWeightedMajorityVote},
{"ClusteringAccuracy", SHOGUN_BASIC_CLASS __new_CClusteringAccuracy},
{"ClusteringMutualInformation", SHOGUN_BASIC_CLASS __new_CClusteringMutualInformation},
{"ContingencyTableEvaluation", SHOGUN_BASIC_CLASS __new_CContingencyTableEvaluation},
{"AccuracyMeasure", SHOGUN_BASIC_CLASS __new_CAccuracyMeasure},
{"ErrorRateMeasure", SHOGUN_BASIC_CLASS __new_CErrorRateMeasure},
{"BALMeasure", SHOGUN_BASIC_CLASS __new_CBALMeasure},
{"WRACCMeasure", SHOGUN_BASIC_CLASS __new_CWRACCMeasure},
{"F1Measure", SHOGUN_BASIC_CLASS __new_CF1Measure},
{"CrossCorrelationMeasure", SHOGUN_BASIC_CLASS __new_CCrossCorrelationMeasure},
{"RecallMeasure", SHOGUN_BASIC_CLASS __new_CRecallMeasure},
{"PrecisionMeasure", SHOGUN_BASIC_CLASS __new_CPrecisionMeasure},
{"SpecificityMeasure", SHOGUN_BASIC_CLASS __new_CSpecificityMeasure},
{"CrossValidationResult", SHOGUN_BASIC_CLASS __new_CCrossValidationResult},
{"CrossValidation", SHOGUN_BASIC_CLASS __new_CCrossValidation},
{"CrossValidationMKLStorage", SHOGUN_BASIC_CLASS __new_CCrossValidationMKLStorage},
{"CrossValidationMulticlassStorage", SHOGUN_BASIC_CLASS __new_CCrossValidationMulticlassStorage},
{"CrossValidationPrintOutput", SHOGUN_BASIC_CLASS __new_CCrossValidationPrintOutput},
{"CrossValidationSplitting", SHOGUN_BASIC_CLASS __new_CCrossValidationSplitting},
{"GradientCriterion", SHOGUN_BASIC_CLASS __new_CGradientCriterion},
{"GradientEvaluation", SHOGUN_BASIC_CLASS __new_CGradientEvaluation},
{"GradientResult", SHOGUN_BASIC_CLASS __new_CGradientResult},
{"MeanAbsoluteError", SHOGUN_BASIC_CLASS __new_CMeanAbsoluteError},
{"MeanSquaredError", SHOGUN_BASIC_CLASS __new_CMeanSquaredError},
{"MeanSquaredLogError", SHOGUN_BASIC_CLASS __new_CMeanSquaredLogError},
{"MulticlassAccuracy", SHOGUN_BASIC_CLASS __new_CMulticlassAccuracy},
{"MulticlassOVREvaluation", SHOGUN_BASIC_CLASS __new_CMulticlassOVREvaluation},
{"PRCEvaluation", SHOGUN_BASIC_CLASS __new_CPRCEvaluation},
{"ROCEvaluation", SHOGUN_BASIC_CLASS __new_CROCEvaluation},
{"StratifiedCrossValidationSplitting", SHOGUN_BASIC_CLASS __new_CStratifiedCrossValidationSplitting},
{"StructuredAccuracy", SHOGUN_BASIC_CLASS __new_CStructuredAccuracy},
{"Alphabet", SHOGUN_BASIC_CLASS __new_CAlphabet},
{"BinnedDotFeatures", SHOGUN_BASIC_CLASS __new_CBinnedDotFeatures},
{"CombinedDotFeatures", SHOGUN_BASIC_CLASS __new_CCombinedDotFeatures},
{"CombinedFeatures", SHOGUN_BASIC_CLASS __new_CCombinedFeatures},
{"DataGenerator", SHOGUN_BASIC_CLASS __new_CDataGenerator},
{"DummyFeatures", SHOGUN_BASIC_CLASS __new_CDummyFeatures},
{"ExplicitSpecFeatures", SHOGUN_BASIC_CLASS __new_CExplicitSpecFeatures},
{"FactorGraphFeatures", SHOGUN_BASIC_CLASS __new_CFactorGraphFeatures},
{"FKFeatures", SHOGUN_BASIC_CLASS __new_CFKFeatures},
{"HashedDocDotFeatures", SHOGUN_BASIC_CLASS __new_CHashedDocDotFeatures},
{"HashedWDFeatures", SHOGUN_BASIC_CLASS __new_CHashedWDFeatures},
{"HashedWDFeaturesTransposed", SHOGUN_BASIC_CLASS __new_CHashedWDFeaturesTransposed},
{"ImplicitWeightedSpecFeatures", SHOGUN_BASIC_CLASS __new_CImplicitWeightedSpecFeatures},
{"LatentFeatures", SHOGUN_BASIC_CLASS __new_CLatentFeatures},
{"LBPPyrDotFeatures", SHOGUN_BASIC_CLASS __new_CLBPPyrDotFeatures},
{"PolyFeatures", SHOGUN_BASIC_CLASS __new_CPolyFeatures},
{"RandomFourierDotFeatures", SHOGUN_BASIC_CLASS __new_CRandomFourierDotFeatures},
{"RealFileFeatures", SHOGUN_BASIC_CLASS __new_CRealFileFeatures},
{"SNPFeatures", SHOGUN_BASIC_CLASS __new_CSNPFeatures},
{"SparsePolyFeatures", SHOGUN_BASIC_CLASS __new_CSparsePolyFeatures},
{"GaussianBlobsDataGenerator", SHOGUN_BASIC_CLASS __new_CGaussianBlobsDataGenerator},
{"MeanShiftDataGenerator", SHOGUN_BASIC_CLASS __new_CMeanShiftDataGenerator},
{"StreamingHashedDocDotFeatures", SHOGUN_BASIC_CLASS __new_CStreamingHashedDocDotFeatures},
{"StreamingVwFeatures", SHOGUN_BASIC_CLASS __new_CStreamingVwFeatures},
{"Subset", SHOGUN_BASIC_CLASS __new_CSubset},
{"SubsetStack", SHOGUN_BASIC_CLASS __new_CSubsetStack},
{"TOPFeatures", SHOGUN_BASIC_CLASS __new_CTOPFeatures},
{"WDFeatures", SHOGUN_BASIC_CLASS __new_CWDFeatures},
{"BinaryFile", SHOGUN_BASIC_CLASS __new_CBinaryFile},
{"CSVFile", SHOGUN_BASIC_CLASS __new_CCSVFile},
{"IOBuffer", SHOGUN_BASIC_CLASS __new_CIOBuffer},
{"LibSVMFile", SHOGUN_BASIC_CLASS __new_CLibSVMFile},
{"LineReader", SHOGUN_BASIC_CLASS __new_CLineReader},
{"Parser", SHOGUN_BASIC_CLASS __new_CParser},
{"SerializableAsciiFile", SHOGUN_BASIC_CLASS __new_CSerializableAsciiFile},
{"StreamingAsciiFile", SHOGUN_BASIC_CLASS __new_CStreamingAsciiFile},
{"StreamingFile", SHOGUN_BASIC_CLASS __new_CStreamingFile},
{"StreamingFileFromFeatures", SHOGUN_BASIC_CLASS __new_CStreamingFileFromFeatures},
{"StreamingVwCacheFile", SHOGUN_BASIC_CLASS __new_CStreamingVwCacheFile},
{"StreamingVwFile", SHOGUN_BASIC_CLASS __new_CStreamingVwFile},
{"ANOVAKernel", SHOGUN_BASIC_CLASS __new_CANOVAKernel},
{"AUCKernel", SHOGUN_BASIC_CLASS __new_CAUCKernel},
{"BesselKernel", SHOGUN_BASIC_CLASS __new_CBesselKernel},
{"CauchyKernel", SHOGUN_BASIC_CLASS __new_CCauchyKernel},
{"Chi2Kernel", SHOGUN_BASIC_CLASS __new_CChi2Kernel},
{"CircularKernel", SHOGUN_BASIC_CLASS __new_CCircularKernel},
{"CombinedKernel", SHOGUN_BASIC_CLASS __new_CCombinedKernel},
{"ConstKernel", SHOGUN_BASIC_CLASS __new_CConstKernel},
{"CustomKernel", SHOGUN_BASIC_CLASS __new_CCustomKernel},
{"DiagKernel", SHOGUN_BASIC_CLASS __new_CDiagKernel},
{"DistanceKernel", SHOGUN_BASIC_CLASS __new_CDistanceKernel},
{"ExponentialKernel", SHOGUN_BASIC_CLASS __new_CExponentialKernel},
{"GaussianARDKernel", SHOGUN_BASIC_CLASS __new_CGaussianARDKernel},
{"GaussianKernel", SHOGUN_BASIC_CLASS __new_CGaussianKernel},
{"GaussianShiftKernel", SHOGUN_BASIC_CLASS __new_CGaussianShiftKernel},
{"GaussianShortRealKernel", SHOGUN_BASIC_CLASS __new_CGaussianShortRealKernel},
{"HistogramIntersectionKernel", SHOGUN_BASIC_CLASS __new_CHistogramIntersectionKernel},
{"InverseMultiQuadricKernel", SHOGUN_BASIC_CLASS __new_CInverseMultiQuadricKernel},
{"JensenShannonKernel", SHOGUN_BASIC_CLASS __new_CJensenShannonKernel},
{"LinearARDKernel", SHOGUN_BASIC_CLASS __new_CLinearARDKernel},
{"LinearKernel", SHOGUN_BASIC_CLASS __new_CLinearKernel},
{"LogKernel", SHOGUN_BASIC_CLASS __new_CLogKernel},
{"MultiquadricKernel", SHOGUN_BASIC_CLASS __new_CMultiquadricKernel},
{"AvgDiagKernelNormalizer", SHOGUN_BASIC_CLASS __new_CAvgDiagKernelNormalizer},
{"DiceKernelNormalizer", SHOGUN_BASIC_CLASS __new_CDiceKernelNormalizer},
{"FirstElementKernelNormalizer", SHOGUN_BASIC_CLASS __new_CFirstElementKernelNormalizer},
{"IdentityKernelNormalizer", SHOGUN_BASIC_CLASS __new_CIdentityKernelNormalizer},
{"RidgeKernelNormalizer", SHOGUN_BASIC_CLASS __new_CRidgeKernelNormalizer},
{"ScatterKernelNormalizer", SHOGUN_BASIC_CLASS __new_CScatterKernelNormalizer},
{"SqrtDiagKernelNormalizer", SHOGUN_BASIC_CLASS __new_CSqrtDiagKernelNormalizer},
{"TanimotoKernelNormalizer", SHOGUN_BASIC_CLASS __new_CTanimotoKernelNormalizer},
{"VarianceKernelNormalizer", SHOGUN_BASIC_CLASS __new_CVarianceKernelNormalizer},
{"ZeroMeanCenterKernelNormalizer", SHOGUN_BASIC_CLASS __new_CZeroMeanCenterKernelNormalizer},
{"PolyKernel", SHOGUN_BASIC_CLASS __new_CPolyKernel},
{"PowerKernel", SHOGUN_BASIC_CLASS __new_CPowerKernel},
{"ProductKernel", SHOGUN_BASIC_CLASS __new_CProductKernel},
{"PyramidChi2", SHOGUN_BASIC_CLASS __new_CPyramidChi2},
{"RationalQuadraticKernel", SHOGUN_BASIC_CLASS __new_CRationalQuadraticKernel},
{"SigmoidKernel", SHOGUN_BASIC_CLASS __new_CSigmoidKernel},
{"SphericalKernel", SHOGUN_BASIC_CLASS __new_CSphericalKernel},
{"SplineKernel", SHOGUN_BASIC_CLASS __new_CSplineKernel},
{"CommUlongStringKernel", SHOGUN_BASIC_CLASS __new_CCommUlongStringKernel},
{"CommWordStringKernel", SHOGUN_BASIC_CLASS __new_CCommWordStringKernel},
{"DistantSegmentsKernel", SHOGUN_BASIC_CLASS __new_CDistantSegmentsKernel},
{"FixedDegreeStringKernel", SHOGUN_BASIC_CLASS __new_CFixedDegreeStringKernel},
{"GaussianMatchStringKernel", SHOGUN_BASIC_CLASS __new_CGaussianMatchStringKernel},
{"HistogramWordStringKernel", SHOGUN_BASIC_CLASS __new_CHistogramWordStringKernel},
{"LinearStringKernel", SHOGUN_BASIC_CLASS __new_CLinearStringKernel},
{"LocalAlignmentStringKernel", SHOGUN_BASIC_CLASS __new_CLocalAlignmentStringKernel},
{"LocalityImprovedStringKernel", SHOGUN_BASIC_CLASS __new_CLocalityImprovedStringKernel},
{"MatchWordStringKernel", SHOGUN_BASIC_CLASS __new_CMatchWordStringKernel},
{"OligoStringKernel", SHOGUN_BASIC_CLASS __new_COligoStringKernel},
{"PolyMatchStringKernel", SHOGUN_BASIC_CLASS __new_CPolyMatchStringKernel},
{"PolyMatchWordStringKernel", SHOGUN_BASIC_CLASS __new_CPolyMatchWordStringKernel},
{"RegulatoryModulesStringKernel", SHOGUN_BASIC_CLASS __new_CRegulatoryModulesStringKernel},
{"SalzbergWordStringKernel", SHOGUN_BASIC_CLASS __new_CSalzbergWordStringKernel},
{"SimpleLocalityImprovedStringKernel", SHOGUN_BASIC_CLASS __new_CSimpleLocalityImprovedStringKernel},
{"SNPStringKernel", SHOGUN_BASIC_CLASS __new_CSNPStringKernel},
{"SparseSpatialSampleStringKernel", SHOGUN_BASIC_CLASS __new_CSparseSpatialSampleStringKernel},
{"SpectrumMismatchRBFKernel", SHOGUN_BASIC_CLASS __new_CSpectrumMismatchRBFKernel},
{"SpectrumRBFKernel", SHOGUN_BASIC_CLASS __new_CSpectrumRBFKernel},
{"WeightedCommWordStringKernel", SHOGUN_BASIC_CLASS __new_CWeightedCommWordStringKernel},
{"WeightedDegreePositionStringKernel", SHOGUN_BASIC_CLASS __new_CWeightedDegreePositionStringKernel},
{"WeightedDegreeStringKernel", SHOGUN_BASIC_CLASS __new_CWeightedDegreeStringKernel},
{"TensorProductPairKernel", SHOGUN_BASIC_CLASS __new_CTensorProductPairKernel},
{"TStudentKernel", SHOGUN_BASIC_CLASS __new_CTStudentKernel},
{"WaveKernel", SHOGUN_BASIC_CLASS __new_CWaveKernel},
{"WaveletKernel", SHOGUN_BASIC_CLASS __new_CWaveletKernel},
{"WeightedDegreeRBFKernel", SHOGUN_BASIC_CLASS __new_CWeightedDegreeRBFKernel},
{"BinaryLabels", SHOGUN_BASIC_CLASS __new_CBinaryLabels},
{"FactorGraphObservation", SHOGUN_BASIC_CLASS __new_CFactorGraphObservation},
{"FactorGraphLabels", SHOGUN_BASIC_CLASS __new_CFactorGraphLabels},
{"LabelsFactory", SHOGUN_BASIC_CLASS __new_CLabelsFactory},
{"LatentLabels", SHOGUN_BASIC_CLASS __new_CLatentLabels},
{"MulticlassLabels", SHOGUN_BASIC_CLASS __new_CMulticlassLabels},
{"MulticlassMultipleOutputLabels", SHOGUN_BASIC_CLASS __new_CMulticlassMultipleOutputLabels},
{"RegressionLabels", SHOGUN_BASIC_CLASS __new_CRegressionLabels},
{"StructuredLabels", SHOGUN_BASIC_CLASS __new_CStructuredLabels},
{"LatentSOSVM", SHOGUN_BASIC_CLASS __new_CLatentSOSVM},
{"LatentSVM", SHOGUN_BASIC_CLASS __new_CLatentSVM},
{"BitString", SHOGUN_BASIC_CLASS __new_CBitString},
{"CircularBuffer", SHOGUN_BASIC_CLASS __new_CCircularBuffer},
{"Compressor", SHOGUN_BASIC_CLASS __new_CCompressor},
{"SerialComputationEngine", SHOGUN_BASIC_CLASS __new_CSerialComputationEngine},
{"JobResult", SHOGUN_BASIC_CLASS __new_CJobResult},
{"Data", SHOGUN_BASIC_CLASS __new_CData},
{"DelimiterTokenizer", SHOGUN_BASIC_CLASS __new_CDelimiterTokenizer},
{"DynamicObjectArray", SHOGUN_BASIC_CLASS __new_CDynamicObjectArray},
{"Hash", SHOGUN_BASIC_CLASS __new_CHash},
{"IndexBlock", SHOGUN_BASIC_CLASS __new_CIndexBlock},
{"IndexBlockGroup", SHOGUN_BASIC_CLASS __new_CIndexBlockGroup},
{"IndexBlockTree", SHOGUN_BASIC_CLASS __new_CIndexBlockTree},
{"ListElement", SHOGUN_BASIC_CLASS __new_CListElement},
{"List", SHOGUN_BASIC_CLASS __new_CList},
{"NGramTokenizer", SHOGUN_BASIC_CLASS __new_CNGramTokenizer},
{"Signal", SHOGUN_BASIC_CLASS __new_CSignal},
{"StructuredData", SHOGUN_BASIC_CLASS __new_CStructuredData},
{"Time", SHOGUN_BASIC_CLASS __new_CTime},
{"HingeLoss", SHOGUN_BASIC_CLASS __new_CHingeLoss},
{"LogLoss", SHOGUN_BASIC_CLASS __new_CLogLoss},
{"LogLossMargin", SHOGUN_BASIC_CLASS __new_CLogLossMargin},
{"SmoothHingeLoss", SHOGUN_BASIC_CLASS __new_CSmoothHingeLoss},
{"SquaredHingeLoss", SHOGUN_BASIC_CLASS __new_CSquaredHingeLoss},
{"SquaredLoss", SHOGUN_BASIC_CLASS __new_CSquaredLoss},
{"BaggingMachine", SHOGUN_BASIC_CLASS __new_CBaggingMachine},
{"BaseMulticlassMachine", SHOGUN_BASIC_CLASS __new_CBaseMulticlassMachine},
{"DistanceMachine", SHOGUN_BASIC_CLASS __new_CDistanceMachine},
{"ZeroMean", SHOGUN_BASIC_CLASS __new_CZeroMean},
{"KernelMachine", SHOGUN_BASIC_CLASS __new_CKernelMachine},
{"KernelMulticlassMachine", SHOGUN_BASIC_CLASS __new_CKernelMulticlassMachine},
{"KernelStructuredOutputMachine", SHOGUN_BASIC_CLASS __new_CKernelStructuredOutputMachine},
{"LinearMachine", SHOGUN_BASIC_CLASS __new_CLinearMachine},
{"LinearMulticlassMachine", SHOGUN_BASIC_CLASS __new_CLinearMulticlassMachine},
{"LinearStructuredOutputMachine", SHOGUN_BASIC_CLASS __new_CLinearStructuredOutputMachine},
{"Machine", SHOGUN_BASIC_CLASS __new_CMachine},
{"NativeMulticlassMachine", SHOGUN_BASIC_CLASS __new_CNativeMulticlassMachine},
{"OnlineLinearMachine", SHOGUN_BASIC_CLASS __new_COnlineLinearMachine},
{"StructuredOutputMachine", SHOGUN_BASIC_CLASS __new_CStructuredOutputMachine},
{"JacobiEllipticFunctions", SHOGUN_BASIC_CLASS __new_CJacobiEllipticFunctions},
{"LogDetEstimator", SHOGUN_BASIC_CLASS __new_CLogDetEstimator},
{"NormalSampler", SHOGUN_BASIC_CLASS __new_CNormalSampler},
{"Math", SHOGUN_BASIC_CLASS __new_CMath},
{"Random", SHOGUN_BASIC_CLASS __new_CRandom},
{"SparseInverseCovariance", SHOGUN_BASIC_CLASS __new_CSparseInverseCovariance},
{"Statistics", SHOGUN_BASIC_CLASS __new_CStatistics},
{"GridSearchModelSelection", SHOGUN_BASIC_CLASS __new_CGridSearchModelSelection},
{"ModelSelectionParameters", SHOGUN_BASIC_CLASS __new_CModelSelectionParameters},
{"ParameterCombination", SHOGUN_BASIC_CLASS __new_CParameterCombination},
{"RandomSearchModelSelection", SHOGUN_BASIC_CLASS __new_CRandomSearchModelSelection},
{"ECOCAEDDecoder", SHOGUN_BASIC_CLASS __new_CECOCAEDDecoder},
{"ECOCDiscriminantEncoder", SHOGUN_BASIC_CLASS __new_CECOCDiscriminantEncoder},
{"ECOCEDDecoder", SHOGUN_BASIC_CLASS __new_CECOCEDDecoder},
{"ECOCForestEncoder", SHOGUN_BASIC_CLASS __new_CECOCForestEncoder},
{"ECOCHDDecoder", SHOGUN_BASIC_CLASS __new_CECOCHDDecoder},
{"ECOCLLBDecoder", SHOGUN_BASIC_CLASS __new_CECOCLLBDecoder},
{"ECOCOVOEncoder", SHOGUN_BASIC_CLASS __new_CECOCOVOEncoder},
{"ECOCOVREncoder", SHOGUN_BASIC_CLASS __new_CECOCOVREncoder},
{"ECOCRandomDenseEncoder", SHOGUN_BASIC_CLASS __new_CECOCRandomDenseEncoder},
{"ECOCRandomSparseEncoder", SHOGUN_BASIC_CLASS __new_CECOCRandomSparseEncoder},
{"ECOCStrategy", SHOGUN_BASIC_CLASS __new_CECOCStrategy},
{"GaussianNaiveBayes", SHOGUN_BASIC_CLASS __new_CGaussianNaiveBayes},
{"GMNPLib", SHOGUN_BASIC_CLASS __new_CGMNPLib},
{"GMNPSVM", SHOGUN_BASIC_CLASS __new_CGMNPSVM},
{"KNN", SHOGUN_BASIC_CLASS __new_CKNN},
{"LaRank", SHOGUN_BASIC_CLASS __new_CLaRank},
{"MulticlassLibLinear", SHOGUN_BASIC_CLASS __new_CMulticlassLibLinear},
{"MulticlassLibSVM", SHOGUN_BASIC_CLASS __new_CMulticlassLibSVM},
{"MulticlassOCAS", SHOGUN_BASIC_CLASS __new_CMulticlassOCAS},
{"MulticlassOneVsOneStrategy", SHOGUN_BASIC_CLASS __new_CMulticlassOneVsOneStrategy},
{"MulticlassOneVsRestStrategy", SHOGUN_BASIC_CLASS __new_CMulticlassOneVsRestStrategy},
{"MulticlassSVM", SHOGUN_BASIC_CLASS __new_CMulticlassSVM},
{"ThresholdRejectionStrategy", SHOGUN_BASIC_CLASS __new_CThresholdRejectionStrategy},
{"DixonQTestRejectionStrategy", SHOGUN_BASIC_CLASS __new_CDixonQTestRejectionStrategy},
{"ScatterSVM", SHOGUN_BASIC_CLASS __new_CScatterSVM},
{"ShareBoost", SHOGUN_BASIC_CLASS __new_CShareBoost},
{"BalancedConditionalProbabilityTree", SHOGUN_BASIC_CLASS __new_CBalancedConditionalProbabilityTree},
{"RandomConditionalProbabilityTree", SHOGUN_BASIC_CLASS __new_CRandomConditionalProbabilityTree},
{"RelaxedTree", SHOGUN_BASIC_CLASS __new_CRelaxedTree},
{"Tron", SHOGUN_BASIC_CLASS __new_CTron},
{"DimensionReductionPreprocessor", SHOGUN_BASIC_CLASS __new_CDimensionReductionPreprocessor},
{"HomogeneousKernelMap", SHOGUN_BASIC_CLASS __new_CHomogeneousKernelMap},
{"LogPlusOne", SHOGUN_BASIC_CLASS __new_CLogPlusOne},
{"NormOne", SHOGUN_BASIC_CLASS __new_CNormOne},
{"PNorm", SHOGUN_BASIC_CLASS __new_CPNorm},
{"PruneVarSubMean", SHOGUN_BASIC_CLASS __new_CPruneVarSubMean},
{"RandomFourierGaussPreproc", SHOGUN_BASIC_CLASS __new_CRandomFourierGaussPreproc},
{"RescaleFeatures", SHOGUN_BASIC_CLASS __new_CRescaleFeatures},
{"SortUlongString", SHOGUN_BASIC_CLASS __new_CSortUlongString},
{"SortWordString", SHOGUN_BASIC_CLASS __new_CSortWordString},
{"SumOne", SHOGUN_BASIC_CLASS __new_CSumOne},
{"LibSVR", SHOGUN_BASIC_CLASS __new_CLibSVR},
{"MKLRegression", SHOGUN_BASIC_CLASS __new_CMKLRegression},
{"SVRLight", SHOGUN_BASIC_CLASS __new_CSVRLight},
{"HSIC", SHOGUN_BASIC_CLASS __new_CHSIC},
{"KernelMeanMatching", SHOGUN_BASIC_CLASS __new_CKernelMeanMatching},
{"LinearTimeMMD", SHOGUN_BASIC_CLASS __new_CLinearTimeMMD},
{"MMDKernelSelectionCombMaxL2", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionCombMaxL2},
{"MMDKernelSelectionCombOpt", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionCombOpt},
{"MMDKernelSelectionMax", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionMax},
{"MMDKernelSelectionMedian", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionMedian},
{"MMDKernelSelectionOpt", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionOpt},
{"QuadraticTimeMMD", SHOGUN_BASIC_CLASS __new_CQuadraticTimeMMD},
{"CCSOSVM", SHOGUN_BASIC_CLASS __new_CCCSOSVM},
{"DisjointSet", SHOGUN_BASIC_CLASS __new_CDisjointSet},
{"DualLibQPBMSOSVM", SHOGUN_BASIC_CLASS __new_CDualLibQPBMSOSVM},
{"DynProg", SHOGUN_BASIC_CLASS __new_CDynProg},
{"FactorDataSource", SHOGUN_BASIC_CLASS __new_CFactorDataSource},
{"Factor", SHOGUN_BASIC_CLASS __new_CFactor},
{"FactorGraph", SHOGUN_BASIC_CLASS __new_CFactorGraph},
{"FactorGraphModel", SHOGUN_BASIC_CLASS __new_CFactorGraphModel},
{"FactorType", SHOGUN_BASIC_CLASS __new_CFactorType},
{"TableFactorType", SHOGUN_BASIC_CLASS __new_CTableFactorType},
{"HMSVMModel", SHOGUN_BASIC_CLASS __new_CHMSVMModel},
{"IntronList", SHOGUN_BASIC_CLASS __new_CIntronList},
{"MAPInference", SHOGUN_BASIC_CLASS __new_CMAPInference},
{"MulticlassModel", SHOGUN_BASIC_CLASS __new_CMulticlassModel},
{"MulticlassSOLabels", SHOGUN_BASIC_CLASS __new_CMulticlassSOLabels},
{"Plif", SHOGUN_BASIC_CLASS __new_CPlif},
{"PlifArray", SHOGUN_BASIC_CLASS __new_CPlifArray},
{"PlifMatrix", SHOGUN_BASIC_CLASS __new_CPlifMatrix},
{"SegmentLoss", SHOGUN_BASIC_CLASS __new_CSegmentLoss},
{"Sequence", SHOGUN_BASIC_CLASS __new_CSequence},
{"SequenceLabels", SHOGUN_BASIC_CLASS __new_CSequenceLabels},
{"SOSVMHelper", SHOGUN_BASIC_CLASS __new_CSOSVMHelper},
{"StochasticSOSVM", SHOGUN_BASIC_CLASS __new_CStochasticSOSVM},
{"TwoStateModel", SHOGUN_BASIC_CLASS __new_CTwoStateModel},
{"DomainAdaptationSVM", SHOGUN_BASIC_CLASS __new_CDomainAdaptationSVM},
{"MultitaskClusteredLogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskClusteredLogisticRegression},
{"MultitaskKernelMaskNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelMaskNormalizer},
{"MultitaskKernelMaskPairNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelMaskPairNormalizer},
{"MultitaskKernelNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelNormalizer},
{"MultitaskKernelPlifNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelPlifNormalizer},
{"Node", SHOGUN_BASIC_CLASS __new_CNode},
{"Taxonomy", SHOGUN_BASIC_CLASS __new_CTaxonomy},
{"MultitaskKernelTreeNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelTreeNormalizer},
{"MultitaskL12LogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskL12LogisticRegression},
{"MultitaskLeastSquaresRegression", SHOGUN_BASIC_CLASS __new_CMultitaskLeastSquaresRegression},
{"MultitaskLinearMachine", SHOGUN_BASIC_CLASS __new_CMultitaskLinearMachine},
{"MultitaskLogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskLogisticRegression},
{"MultitaskROCEvaluation", SHOGUN_BASIC_CLASS __new_CMultitaskROCEvaluation},
{"MultitaskTraceLogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskTraceLogisticRegression},
{"Task", SHOGUN_BASIC_CLASS __new_CTask},
{"TaskGroup", SHOGUN_BASIC_CLASS __new_CTaskGroup},
{"TaskTree", SHOGUN_BASIC_CLASS __new_CTaskTree},
{"GUIClassifier", SHOGUN_BASIC_CLASS __new_CGUIClassifier},
{"GUIConverter", SHOGUN_BASIC_CLASS __new_CGUIConverter},
{"GUIDistance", SHOGUN_BASIC_CLASS __new_CGUIDistance},
{"GUIFeatures", SHOGUN_BASIC_CLASS __new_CGUIFeatures},
{"GUIHMM", SHOGUN_BASIC_CLASS __new_CGUIHMM},
{"GUIKernel", SHOGUN_BASIC_CLASS __new_CGUIKernel},
{"GUILabels", SHOGUN_BASIC_CLASS __new_CGUILabels},
{"GUIMath", SHOGUN_BASIC_CLASS __new_CGUIMath},
{"GUIPluginEstimate", SHOGUN_BASIC_CLASS __new_CGUIPluginEstimate},
{"GUIPreprocessor", SHOGUN_BASIC_CLASS __new_CGUIPreprocessor},
{"GUIStructure", SHOGUN_BASIC_CLASS __new_CGUIStructure},
{"GUITime", SHOGUN_BASIC_CLASS __new_CGUITime},
{"AveragedPerceptron", SHOGUN_BASIC_CLASS __new_CAveragedPerceptron},
{"FeatureBlockLogisticRegression", SHOGUN_BASIC_CLASS __new_CFeatureBlockLogisticRegression},
{"MKLClassification", SHOGUN_BASIC_CLASS __new_CMKLClassification},
{"MKLMulticlass", SHOGUN_BASIC_CLASS __new_CMKLMulticlass},
{"MKLOneClass", SHOGUN_BASIC_CLASS __new_CMKLOneClass},
{"NearestCentroid", SHOGUN_BASIC_CLASS __new_CNearestCentroid},
{"Perceptron", SHOGUN_BASIC_CLASS __new_CPerceptron},
{"PluginEstimate", SHOGUN_BASIC_CLASS __new_CPluginEstimate},
{"GNPPLib", SHOGUN_BASIC_CLASS __new_CGNPPLib},
{"GNPPSVM", SHOGUN_BASIC_CLASS __new_CGNPPSVM},
{"GPBTSVM", SHOGUN_BASIC_CLASS __new_CGPBTSVM},
{"LibLinear", SHOGUN_BASIC_CLASS __new_CLibLinear},
{"LibSVM", SHOGUN_BASIC_CLASS __new_CLibSVM},
{"LibSVMOneClass", SHOGUN_BASIC_CLASS __new_CLibSVMOneClass},
{"MPDSVM", SHOGUN_BASIC_CLASS __new_CMPDSVM},
{"OnlineLibLinear", SHOGUN_BASIC_CLASS __new_COnlineLibLinear},
{"OnlineSVMSGD", SHOGUN_BASIC_CLASS __new_COnlineSVMSGD},
{"QPBSVMLib", SHOGUN_BASIC_CLASS __new_CQPBSVMLib},
{"SGDQN", SHOGUN_BASIC_CLASS __new_CSGDQN},
{"SVM", SHOGUN_BASIC_CLASS __new_CSVM},
{"SVMLight", SHOGUN_BASIC_CLASS __new_CSVMLight},
{"SVMLightOneClass", SHOGUN_BASIC_CLASS __new_CSVMLightOneClass},
{"SVMLin", SHOGUN_BASIC_CLASS __new_CSVMLin},
{"SVMOcas", SHOGUN_BASIC_CLASS __new_CSVMOcas},
{"SVMSGD", SHOGUN_BASIC_CLASS __new_CSVMSGD},
{"WDSVMOcas", SHOGUN_BASIC_CLASS __new_CWDSVMOcas},
{"VwNativeCacheReader", SHOGUN_BASIC_CLASS __new_CVwNativeCacheReader},
{"VwNativeCacheWriter", SHOGUN_BASIC_CLASS __new_CVwNativeCacheWriter},
{"VwAdaptiveLearner", SHOGUN_BASIC_CLASS __new_CVwAdaptiveLearner},
{"VwNonAdaptiveLearner", SHOGUN_BASIC_CLASS __new_CVwNonAdaptiveLearner},
{"VowpalWabbit", SHOGUN_BASIC_CLASS __new_CVowpalWabbit},
{"VwEnvironment", SHOGUN_BASIC_CLASS __new_CVwEnvironment},
{"VwParser", SHOGUN_BASIC_CLASS __new_CVwParser},
{"VwRegressor", SHOGUN_BASIC_CLASS __new_CVwRegressor},
{"Hierarchical", SHOGUN_BASIC_CLASS __new_CHierarchical},
{"KMeans", SHOGUN_BASIC_CLASS __new_CKMeans},
{"HashedDocConverter", SHOGUN_BASIC_CLASS __new_CHashedDocConverter},
{"AttenuatedEuclideanDistance", SHOGUN_BASIC_CLASS __new_CAttenuatedEuclideanDistance},
{"BrayCurtisDistance", SHOGUN_BASIC_CLASS __new_CBrayCurtisDistance},
{"CanberraMetric", SHOGUN_BASIC_CLASS __new_CCanberraMetric},
{"CanberraWordDistance", SHOGUN_BASIC_CLASS __new_CCanberraWordDistance},
{"ChebyshewMetric", SHOGUN_BASIC_CLASS __new_CChebyshewMetric},
{"ChiSquareDistance", SHOGUN_BASIC_CLASS __new_CChiSquareDistance},
{"CosineDistance", SHOGUN_BASIC_CLASS __new_CCosineDistance},
{"CustomDistance", SHOGUN_BASIC_CLASS __new_CCustomDistance},
{"EuclideanDistance", SHOGUN_BASIC_CLASS __new_CEuclideanDistance},
{"GeodesicMetric", SHOGUN_BASIC_CLASS __new_CGeodesicMetric},
{"HammingWordDistance", SHOGUN_BASIC_CLASS __new_CHammingWordDistance},
{"JensenMetric", SHOGUN_BASIC_CLASS __new_CJensenMetric},
{"KernelDistance", SHOGUN_BASIC_CLASS __new_CKernelDistance},
{"ManhattanMetric", SHOGUN_BASIC_CLASS __new_CManhattanMetric},
{"ManhattanWordDistance", SHOGUN_BASIC_CLASS __new_CManhattanWordDistance},
{"MinkowskiMetric", SHOGUN_BASIC_CLASS __new_CMinkowskiMetric},
{"SparseEuclideanDistance", SHOGUN_BASIC_CLASS __new_CSparseEuclideanDistance},
{"TanimotoDistance", SHOGUN_BASIC_CLASS __new_CTanimotoDistance},
{"GHMM", SHOGUN_BASIC_CLASS __new_CGHMM},
{"Histogram", SHOGUN_BASIC_CLASS __new_CHistogram},
{"HMM", SHOGUN_BASIC_CLASS __new_CHMM},
{"LinearHMM", SHOGUN_BASIC_CLASS __new_CLinearHMM},
{"PositionalPWM", SHOGUN_BASIC_CLASS __new_CPositionalPWM},
{"MajorityVote", SHOGUN_BASIC_CLASS __new_CMajorityVote},
{"MeanRule", SHOGUN_BASIC_CLASS __new_CMeanRule},
{"WeightedMajorityVote", SHOGUN_BASIC_CLASS __new_CWeightedMajorityVote},
{"ClusteringAccuracy", SHOGUN_BASIC_CLASS __new_CClusteringAccuracy},
{"ClusteringMutualInformation", SHOGUN_BASIC_CLASS __new_CClusteringMutualInformation},
{"ContingencyTableEvaluation", SHOGUN_BASIC_CLASS __new_CContingencyTableEvaluation},
{"AccuracyMeasure", SHOGUN_BASIC_CLASS __new_CAccuracyMeasure},
{"ErrorRateMeasure", SHOGUN_BASIC_CLASS __new_CErrorRateMeasure},
{"BALMeasure", SHOGUN_BASIC_CLASS __new_CBALMeasure},
{"WRACCMeasure", SHOGUN_BASIC_CLASS __new_CWRACCMeasure},
{"F1Measure", SHOGUN_BASIC_CLASS __new_CF1Measure},
{"CrossCorrelationMeasure", SHOGUN_BASIC_CLASS __new_CCrossCorrelationMeasure},
{"RecallMeasure", SHOGUN_BASIC_CLASS __new_CRecallMeasure},
{"PrecisionMeasure", SHOGUN_BASIC_CLASS __new_CPrecisionMeasure},
{"SpecificityMeasure", SHOGUN_BASIC_CLASS __new_CSpecificityMeasure},
{"CrossValidationResult", SHOGUN_BASIC_CLASS __new_CCrossValidationResult},
{"CrossValidation", SHOGUN_BASIC_CLASS __new_CCrossValidation},
{"CrossValidationMKLStorage", SHOGUN_BASIC_CLASS __new_CCrossValidationMKLStorage},
{"CrossValidationMulticlassStorage", SHOGUN_BASIC_CLASS __new_CCrossValidationMulticlassStorage},
{"CrossValidationPrintOutput", SHOGUN_BASIC_CLASS __new_CCrossValidationPrintOutput},
{"CrossValidationSplitting", SHOGUN_BASIC_CLASS __new_CCrossValidationSplitting},
{"GradientCriterion", SHOGUN_BASIC_CLASS __new_CGradientCriterion},
{"GradientEvaluation", SHOGUN_BASIC_CLASS __new_CGradientEvaluation},
{"GradientResult", SHOGUN_BASIC_CLASS __new_CGradientResult},
{"MeanAbsoluteError", SHOGUN_BASIC_CLASS __new_CMeanAbsoluteError},
{"MeanSquaredError", SHOGUN_BASIC_CLASS __new_CMeanSquaredError},
{"MeanSquaredLogError", SHOGUN_BASIC_CLASS __new_CMeanSquaredLogError},
{"MulticlassAccuracy", SHOGUN_BASIC_CLASS __new_CMulticlassAccuracy},
{"MulticlassOVREvaluation", SHOGUN_BASIC_CLASS __new_CMulticlassOVREvaluation},
{"PRCEvaluation", SHOGUN_BASIC_CLASS __new_CPRCEvaluation},
{"ROCEvaluation", SHOGUN_BASIC_CLASS __new_CROCEvaluation},
{"StratifiedCrossValidationSplitting", SHOGUN_BASIC_CLASS __new_CStratifiedCrossValidationSplitting},
{"StructuredAccuracy", SHOGUN_BASIC_CLASS __new_CStructuredAccuracy},
{"Alphabet", SHOGUN_BASIC_CLASS __new_CAlphabet},
{"BinnedDotFeatures", SHOGUN_BASIC_CLASS __new_CBinnedDotFeatures},
{"CombinedDotFeatures", SHOGUN_BASIC_CLASS __new_CCombinedDotFeatures},
{"CombinedFeatures", SHOGUN_BASIC_CLASS __new_CCombinedFeatures},
{"DataGenerator", SHOGUN_BASIC_CLASS __new_CDataGenerator},
{"DummyFeatures", SHOGUN_BASIC_CLASS __new_CDummyFeatures},
{"ExplicitSpecFeatures", SHOGUN_BASIC_CLASS __new_CExplicitSpecFeatures},
{"FactorGraphFeatures", SHOGUN_BASIC_CLASS __new_CFactorGraphFeatures},
{"FKFeatures", SHOGUN_BASIC_CLASS __new_CFKFeatures},
{"HashedDocDotFeatures", SHOGUN_BASIC_CLASS __new_CHashedDocDotFeatures},
{"HashedWDFeatures", SHOGUN_BASIC_CLASS __new_CHashedWDFeatures},
{"HashedWDFeaturesTransposed", SHOGUN_BASIC_CLASS __new_CHashedWDFeaturesTransposed},
{"ImplicitWeightedSpecFeatures", SHOGUN_BASIC_CLASS __new_CImplicitWeightedSpecFeatures},
{"LatentFeatures", SHOGUN_BASIC_CLASS __new_CLatentFeatures},
{"LBPPyrDotFeatures", SHOGUN_BASIC_CLASS __new_CLBPPyrDotFeatures},
{"PolyFeatures", SHOGUN_BASIC_CLASS __new_CPolyFeatures},
{"RandomFourierDotFeatures", SHOGUN_BASIC_CLASS __new_CRandomFourierDotFeatures},
{"RealFileFeatures", SHOGUN_BASIC_CLASS __new_CRealFileFeatures},
{"SNPFeatures", SHOGUN_BASIC_CLASS __new_CSNPFeatures},
{"SparsePolyFeatures", SHOGUN_BASIC_CLASS __new_CSparsePolyFeatures},
{"GaussianBlobsDataGenerator", SHOGUN_BASIC_CLASS __new_CGaussianBlobsDataGenerator},
{"MeanShiftDataGenerator", SHOGUN_BASIC_CLASS __new_CMeanShiftDataGenerator},
{"StreamingHashedDocDotFeatures", SHOGUN_BASIC_CLASS __new_CStreamingHashedDocDotFeatures},
{"StreamingVwFeatures", SHOGUN_BASIC_CLASS __new_CStreamingVwFeatures},
{"Subset", SHOGUN_BASIC_CLASS __new_CSubset},
{"SubsetStack", SHOGUN_BASIC_CLASS __new_CSubsetStack},
{"TOPFeatures", SHOGUN_BASIC_CLASS __new_CTOPFeatures},
{"WDFeatures", SHOGUN_BASIC_CLASS __new_CWDFeatures},
{"BinaryFile", SHOGUN_BASIC_CLASS __new_CBinaryFile},
{"CSVFile", SHOGUN_BASIC_CLASS __new_CCSVFile},
{"IOBuffer", SHOGUN_BASIC_CLASS __new_CIOBuffer},
{"LibSVMFile", SHOGUN_BASIC_CLASS __new_CLibSVMFile},
{"LineReader", SHOGUN_BASIC_CLASS __new_CLineReader},
{"Parser", SHOGUN_BASIC_CLASS __new_CParser},
{"SerializableAsciiFile", SHOGUN_BASIC_CLASS __new_CSerializableAsciiFile},
{"StreamingAsciiFile", SHOGUN_BASIC_CLASS __new_CStreamingAsciiFile},
{"StreamingFile", SHOGUN_BASIC_CLASS __new_CStreamingFile},
{"StreamingFileFromFeatures", SHOGUN_BASIC_CLASS __new_CStreamingFileFromFeatures},
{"StreamingVwCacheFile", SHOGUN_BASIC_CLASS __new_CStreamingVwCacheFile},
{"StreamingVwFile", SHOGUN_BASIC_CLASS __new_CStreamingVwFile},
{"ANOVAKernel", SHOGUN_BASIC_CLASS __new_CANOVAKernel},
{"AUCKernel", SHOGUN_BASIC_CLASS __new_CAUCKernel},
{"BesselKernel", SHOGUN_BASIC_CLASS __new_CBesselKernel},
{"CauchyKernel", SHOGUN_BASIC_CLASS __new_CCauchyKernel},
{"Chi2Kernel", SHOGUN_BASIC_CLASS __new_CChi2Kernel},
{"CircularKernel", SHOGUN_BASIC_CLASS __new_CCircularKernel},
{"CombinedKernel", SHOGUN_BASIC_CLASS __new_CCombinedKernel},
{"ConstKernel", SHOGUN_BASIC_CLASS __new_CConstKernel},
{"CustomKernel", SHOGUN_BASIC_CLASS __new_CCustomKernel},
{"DiagKernel", SHOGUN_BASIC_CLASS __new_CDiagKernel},
{"DistanceKernel", SHOGUN_BASIC_CLASS __new_CDistanceKernel},
{"ExponentialKernel", SHOGUN_BASIC_CLASS __new_CExponentialKernel},
{"GaussianARDKernel", SHOGUN_BASIC_CLASS __new_CGaussianARDKernel},
{"GaussianKernel", SHOGUN_BASIC_CLASS __new_CGaussianKernel},
{"GaussianShiftKernel", SHOGUN_BASIC_CLASS __new_CGaussianShiftKernel},
{"GaussianShortRealKernel", SHOGUN_BASIC_CLASS __new_CGaussianShortRealKernel},
{"HistogramIntersectionKernel", SHOGUN_BASIC_CLASS __new_CHistogramIntersectionKernel},
{"InverseMultiQuadricKernel", SHOGUN_BASIC_CLASS __new_CInverseMultiQuadricKernel},
{"JensenShannonKernel", SHOGUN_BASIC_CLASS __new_CJensenShannonKernel},
{"LinearARDKernel", SHOGUN_BASIC_CLASS __new_CLinearARDKernel},
{"LinearKernel", SHOGUN_BASIC_CLASS __new_CLinearKernel},
{"LogKernel", SHOGUN_BASIC_CLASS __new_CLogKernel},
{"MultiquadricKernel", SHOGUN_BASIC_CLASS __new_CMultiquadricKernel},
{"AvgDiagKernelNormalizer", SHOGUN_BASIC_CLASS __new_CAvgDiagKernelNormalizer},
{"DiceKernelNormalizer", SHOGUN_BASIC_CLASS __new_CDiceKernelNormalizer},
{"FirstElementKernelNormalizer", SHOGUN_BASIC_CLASS __new_CFirstElementKernelNormalizer},
{"IdentityKernelNormalizer", SHOGUN_BASIC_CLASS __new_CIdentityKernelNormalizer},
{"RidgeKernelNormalizer", SHOGUN_BASIC_CLASS __new_CRidgeKernelNormalizer},
{"ScatterKernelNormalizer", SHOGUN_BASIC_CLASS __new_CScatterKernelNormalizer},
{"SqrtDiagKernelNormalizer", SHOGUN_BASIC_CLASS __new_CSqrtDiagKernelNormalizer},
{"TanimotoKernelNormalizer", SHOGUN_BASIC_CLASS __new_CTanimotoKernelNormalizer},
{"VarianceKernelNormalizer", SHOGUN_BASIC_CLASS __new_CVarianceKernelNormalizer},
{"ZeroMeanCenterKernelNormalizer", SHOGUN_BASIC_CLASS __new_CZeroMeanCenterKernelNormalizer},
{"PolyKernel", SHOGUN_BASIC_CLASS __new_CPolyKernel},
{"PowerKernel", SHOGUN_BASIC_CLASS __new_CPowerKernel},
{"ProductKernel", SHOGUN_BASIC_CLASS __new_CProductKernel},
{"PyramidChi2", SHOGUN_BASIC_CLASS __new_CPyramidChi2},
{"RationalQuadraticKernel", SHOGUN_BASIC_CLASS __new_CRationalQuadraticKernel},
{"SigmoidKernel", SHOGUN_BASIC_CLASS __new_CSigmoidKernel},
{"SphericalKernel", SHOGUN_BASIC_CLASS __new_CSphericalKernel},
{"SplineKernel", SHOGUN_BASIC_CLASS __new_CSplineKernel},
{"CommUlongStringKernel", SHOGUN_BASIC_CLASS __new_CCommUlongStringKernel},
{"CommWordStringKernel", SHOGUN_BASIC_CLASS __new_CCommWordStringKernel},
{"DistantSegmentsKernel", SHOGUN_BASIC_CLASS __new_CDistantSegmentsKernel},
{"FixedDegreeStringKernel", SHOGUN_BASIC_CLASS __new_CFixedDegreeStringKernel},
{"GaussianMatchStringKernel", SHOGUN_BASIC_CLASS __new_CGaussianMatchStringKernel},
{"HistogramWordStringKernel", SHOGUN_BASIC_CLASS __new_CHistogramWordStringKernel},
{"LinearStringKernel", SHOGUN_BASIC_CLASS __new_CLinearStringKernel},
{"LocalAlignmentStringKernel", SHOGUN_BASIC_CLASS __new_CLocalAlignmentStringKernel},
{"LocalityImprovedStringKernel", SHOGUN_BASIC_CLASS __new_CLocalityImprovedStringKernel},
{"MatchWordStringKernel", SHOGUN_BASIC_CLASS __new_CMatchWordStringKernel},
{"OligoStringKernel", SHOGUN_BASIC_CLASS __new_COligoStringKernel},
{"PolyMatchStringKernel", SHOGUN_BASIC_CLASS __new_CPolyMatchStringKernel},
{"PolyMatchWordStringKernel", SHOGUN_BASIC_CLASS __new_CPolyMatchWordStringKernel},
{"RegulatoryModulesStringKernel", SHOGUN_BASIC_CLASS __new_CRegulatoryModulesStringKernel},
{"SalzbergWordStringKernel", SHOGUN_BASIC_CLASS __new_CSalzbergWordStringKernel},
{"SimpleLocalityImprovedStringKernel", SHOGUN_BASIC_CLASS __new_CSimpleLocalityImprovedStringKernel},
{"SNPStringKernel", SHOGUN_BASIC_CLASS __new_CSNPStringKernel},
{"SparseSpatialSampleStringKernel", SHOGUN_BASIC_CLASS __new_CSparseSpatialSampleStringKernel},
{"SpectrumMismatchRBFKernel", SHOGUN_BASIC_CLASS __new_CSpectrumMismatchRBFKernel},
{"SpectrumRBFKernel", SHOGUN_BASIC_CLASS __new_CSpectrumRBFKernel},
{"WeightedCommWordStringKernel", SHOGUN_BASIC_CLASS __new_CWeightedCommWordStringKernel},
{"WeightedDegreePositionStringKernel", SHOGUN_BASIC_CLASS __new_CWeightedDegreePositionStringKernel},
{"WeightedDegreeStringKernel", SHOGUN_BASIC_CLASS __new_CWeightedDegreeStringKernel},
{"TensorProductPairKernel", SHOGUN_BASIC_CLASS __new_CTensorProductPairKernel},
{"TStudentKernel", SHOGUN_BASIC_CLASS __new_CTStudentKernel},
{"WaveKernel", SHOGUN_BASIC_CLASS __new_CWaveKernel},
{"WaveletKernel", SHOGUN_BASIC_CLASS __new_CWaveletKernel},
{"WeightedDegreeRBFKernel", SHOGUN_BASIC_CLASS __new_CWeightedDegreeRBFKernel},
{"BinaryLabels", SHOGUN_BASIC_CLASS __new_CBinaryLabels},
{"FactorGraphObservation", SHOGUN_BASIC_CLASS __new_CFactorGraphObservation},
{"FactorGraphLabels", SHOGUN_BASIC_CLASS __new_CFactorGraphLabels},
{"LabelsFactory", SHOGUN_BASIC_CLASS __new_CLabelsFactory},
{"LatentLabels", SHOGUN_BASIC_CLASS __new_CLatentLabels},
{"MulticlassLabels", SHOGUN_BASIC_CLASS __new_CMulticlassLabels},
{"MulticlassMultipleOutputLabels", SHOGUN_BASIC_CLASS __new_CMulticlassMultipleOutputLabels},
{"RegressionLabels", SHOGUN_BASIC_CLASS __new_CRegressionLabels},
{"StructuredLabels", SHOGUN_BASIC_CLASS __new_CStructuredLabels},
{"LatentSOSVM", SHOGUN_BASIC_CLASS __new_CLatentSOSVM},
{"LatentSVM", SHOGUN_BASIC_CLASS __new_CLatentSVM},
{"BitString", SHOGUN_BASIC_CLASS __new_CBitString},
{"CircularBuffer", SHOGUN_BASIC_CLASS __new_CCircularBuffer},
{"Compressor", SHOGUN_BASIC_CLASS __new_CCompressor},
{"SerialComputationEngine", SHOGUN_BASIC_CLASS __new_CSerialComputationEngine},
{"JobResult", SHOGUN_BASIC_CLASS __new_CJobResult},
{"Data", SHOGUN_BASIC_CLASS __new_CData},
{"DelimiterTokenizer", SHOGUN_BASIC_CLASS __new_CDelimiterTokenizer},
{"DynamicObjectArray", SHOGUN_BASIC_CLASS __new_CDynamicObjectArray},
{"Hash", SHOGUN_BASIC_CLASS __new_CHash},
{"IndexBlock", SHOGUN_BASIC_CLASS __new_CIndexBlock},
{"IndexBlockGroup", SHOGUN_BASIC_CLASS __new_CIndexBlockGroup},
{"IndexBlockTree", SHOGUN_BASIC_CLASS __new_CIndexBlockTree},
{"ListElement", SHOGUN_BASIC_CLASS __new_CListElement},
{"List", SHOGUN_BASIC_CLASS __new_CList},
{"NGramTokenizer", SHOGUN_BASIC_CLASS __new_CNGramTokenizer},
{"Signal", SHOGUN_BASIC_CLASS __new_CSignal},
{"StructuredData", SHOGUN_BASIC_CLASS __new_CStructuredData},
{"Time", SHOGUN_BASIC_CLASS __new_CTime},
{"HingeLoss", SHOGUN_BASIC_CLASS __new_CHingeLoss},
{"LogLoss", SHOGUN_BASIC_CLASS __new_CLogLoss},
{"LogLossMargin", SHOGUN_BASIC_CLASS __new_CLogLossMargin},
{"SmoothHingeLoss", SHOGUN_BASIC_CLASS __new_CSmoothHingeLoss},
{"SquaredHingeLoss", SHOGUN_BASIC_CLASS __new_CSquaredHingeLoss},
{"SquaredLoss", SHOGUN_BASIC_CLASS __new_CSquaredLoss},
{"BaggingMachine", SHOGUN_BASIC_CLASS __new_CBaggingMachine},
{"BaseMulticlassMachine", SHOGUN_BASIC_CLASS __new_CBaseMulticlassMachine},
{"DistanceMachine", SHOGUN_BASIC_CLASS __new_CDistanceMachine},
{"ZeroMean", SHOGUN_BASIC_CLASS __new_CZeroMean},
{"KernelMachine", SHOGUN_BASIC_CLASS __new_CKernelMachine},
{"KernelMulticlassMachine", SHOGUN_BASIC_CLASS __new_CKernelMulticlassMachine},
{"KernelStructuredOutputMachine", SHOGUN_BASIC_CLASS __new_CKernelStructuredOutputMachine},
{"LinearMachine", SHOGUN_BASIC_CLASS __new_CLinearMachine},
{"LinearMulticlassMachine", SHOGUN_BASIC_CLASS __new_CLinearMulticlassMachine},
{"LinearStructuredOutputMachine", SHOGUN_BASIC_CLASS __new_CLinearStructuredOutputMachine},
{"Machine", SHOGUN_BASIC_CLASS __new_CMachine},
{"NativeMulticlassMachine", SHOGUN_BASIC_CLASS __new_CNativeMulticlassMachine},
{"OnlineLinearMachine", SHOGUN_BASIC_CLASS __new_COnlineLinearMachine},
{"StructuredOutputMachine", SHOGUN_BASIC_CLASS __new_CStructuredOutputMachine},
{"JacobiEllipticFunctions", SHOGUN_BASIC_CLASS __new_CJacobiEllipticFunctions},
{"LogDetEstimator", SHOGUN_BASIC_CLASS __new_CLogDetEstimator},
{"NormalSampler", SHOGUN_BASIC_CLASS __new_CNormalSampler},
{"Math", SHOGUN_BASIC_CLASS __new_CMath},
{"Random", SHOGUN_BASIC_CLASS __new_CRandom},
{"SparseInverseCovariance", SHOGUN_BASIC_CLASS __new_CSparseInverseCovariance},
{"Statistics", SHOGUN_BASIC_CLASS __new_CStatistics},
{"GridSearchModelSelection", SHOGUN_BASIC_CLASS __new_CGridSearchModelSelection},
{"ModelSelectionParameters", SHOGUN_BASIC_CLASS __new_CModelSelectionParameters},
{"ParameterCombination", SHOGUN_BASIC_CLASS __new_CParameterCombination},
{"RandomSearchModelSelection", SHOGUN_BASIC_CLASS __new_CRandomSearchModelSelection},
{"ECOCAEDDecoder", SHOGUN_BASIC_CLASS __new_CECOCAEDDecoder},
{"ECOCDiscriminantEncoder", SHOGUN_BASIC_CLASS __new_CECOCDiscriminantEncoder},
{"ECOCEDDecoder", SHOGUN_BASIC_CLASS __new_CECOCEDDecoder},
{"ECOCForestEncoder", SHOGUN_BASIC_CLASS __new_CECOCForestEncoder},
{"ECOCHDDecoder", SHOGUN_BASIC_CLASS __new_CECOCHDDecoder},
{"ECOCLLBDecoder", SHOGUN_BASIC_CLASS __new_CECOCLLBDecoder},
{"ECOCOVOEncoder", SHOGUN_BASIC_CLASS __new_CECOCOVOEncoder},
{"ECOCOVREncoder", SHOGUN_BASIC_CLASS __new_CECOCOVREncoder},
{"ECOCRandomDenseEncoder", SHOGUN_BASIC_CLASS __new_CECOCRandomDenseEncoder},
{"ECOCRandomSparseEncoder", SHOGUN_BASIC_CLASS __new_CECOCRandomSparseEncoder},
{"ECOCStrategy", SHOGUN_BASIC_CLASS __new_CECOCStrategy},
{"GaussianNaiveBayes", SHOGUN_BASIC_CLASS __new_CGaussianNaiveBayes},
{"GMNPLib", SHOGUN_BASIC_CLASS __new_CGMNPLib},
{"GMNPSVM", SHOGUN_BASIC_CLASS __new_CGMNPSVM},
{"KNN", SHOGUN_BASIC_CLASS __new_CKNN},
{"LaRank", SHOGUN_BASIC_CLASS __new_CLaRank},
{"MulticlassLibLinear", SHOGUN_BASIC_CLASS __new_CMulticlassLibLinear},
{"MulticlassLibSVM", SHOGUN_BASIC_CLASS __new_CMulticlassLibSVM},
{"MulticlassOCAS", SHOGUN_BASIC_CLASS __new_CMulticlassOCAS},
{"MulticlassOneVsOneStrategy", SHOGUN_BASIC_CLASS __new_CMulticlassOneVsOneStrategy},
{"MulticlassOneVsRestStrategy", SHOGUN_BASIC_CLASS __new_CMulticlassOneVsRestStrategy},
{"MulticlassSVM", SHOGUN_BASIC_CLASS __new_CMulticlassSVM},
{"ThresholdRejectionStrategy", SHOGUN_BASIC_CLASS __new_CThresholdRejectionStrategy},
{"DixonQTestRejectionStrategy", SHOGUN_BASIC_CLASS __new_CDixonQTestRejectionStrategy},
{"ScatterSVM", SHOGUN_BASIC_CLASS __new_CScatterSVM},
{"ShareBoost", SHOGUN_BASIC_CLASS __new_CShareBoost},
{"BalancedConditionalProbabilityTree", SHOGUN_BASIC_CLASS __new_CBalancedConditionalProbabilityTree},
{"RandomConditionalProbabilityTree", SHOGUN_BASIC_CLASS __new_CRandomConditionalProbabilityTree},
{"RelaxedTree", SHOGUN_BASIC_CLASS __new_CRelaxedTree},
{"Tron", SHOGUN_BASIC_CLASS __new_CTron},
{"DimensionReductionPreprocessor", SHOGUN_BASIC_CLASS __new_CDimensionReductionPreprocessor},
{"HomogeneousKernelMap", SHOGUN_BASIC_CLASS __new_CHomogeneousKernelMap},
{"LogPlusOne", SHOGUN_BASIC_CLASS __new_CLogPlusOne},
{"NormOne", SHOGUN_BASIC_CLASS __new_CNormOne},
{"PNorm", SHOGUN_BASIC_CLASS __new_CPNorm},
{"PruneVarSubMean", SHOGUN_BASIC_CLASS __new_CPruneVarSubMean},
{"RandomFourierGaussPreproc", SHOGUN_BASIC_CLASS __new_CRandomFourierGaussPreproc},
{"RescaleFeatures", SHOGUN_BASIC_CLASS __new_CRescaleFeatures},
{"SortUlongString", SHOGUN_BASIC_CLASS __new_CSortUlongString},
{"SortWordString", SHOGUN_BASIC_CLASS __new_CSortWordString},
{"SumOne", SHOGUN_BASIC_CLASS __new_CSumOne},
{"LibSVR", SHOGUN_BASIC_CLASS __new_CLibSVR},
{"MKLRegression", SHOGUN_BASIC_CLASS __new_CMKLRegression},
{"SVRLight", SHOGUN_BASIC_CLASS __new_CSVRLight},
{"HSIC", SHOGUN_BASIC_CLASS __new_CHSIC},
{"KernelMeanMatching", SHOGUN_BASIC_CLASS __new_CKernelMeanMatching},
{"LinearTimeMMD", SHOGUN_BASIC_CLASS __new_CLinearTimeMMD},
{"MMDKernelSelectionCombMaxL2", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionCombMaxL2},
{"MMDKernelSelectionCombOpt", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionCombOpt},
{"MMDKernelSelectionMax", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionMax},
{"MMDKernelSelectionMedian", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionMedian},
{"MMDKernelSelectionOpt", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionOpt},
{"QuadraticTimeMMD", SHOGUN_BASIC_CLASS __new_CQuadraticTimeMMD},
{"CCSOSVM", SHOGUN_BASIC_CLASS __new_CCCSOSVM},
{"DisjointSet", SHOGUN_BASIC_CLASS __new_CDisjointSet},
{"DualLibQPBMSOSVM", SHOGUN_BASIC_CLASS __new_CDualLibQPBMSOSVM},
{"DynProg", SHOGUN_BASIC_CLASS __new_CDynProg},
{"FactorDataSource", SHOGUN_BASIC_CLASS __new_CFactorDataSource},
{"Factor", SHOGUN_BASIC_CLASS __new_CFactor},
{"FactorGraph", SHOGUN_BASIC_CLASS __new_CFactorGraph},
{"FactorGraphModel", SHOGUN_BASIC_CLASS __new_CFactorGraphModel},
{"FactorType", SHOGUN_BASIC_CLASS __new_CFactorType},
{"TableFactorType", SHOGUN_BASIC_CLASS __new_CTableFactorType},
{"HMSVMModel", SHOGUN_BASIC_CLASS __new_CHMSVMModel},
{"IntronList", SHOGUN_BASIC_CLASS __new_CIntronList},
{"MAPInference", SHOGUN_BASIC_CLASS __new_CMAPInference},
{"MulticlassModel", SHOGUN_BASIC_CLASS __new_CMulticlassModel},
{"MulticlassSOLabels", SHOGUN_BASIC_CLASS __new_CMulticlassSOLabels},
{"Plif", SHOGUN_BASIC_CLASS __new_CPlif},
{"PlifArray", SHOGUN_BASIC_CLASS __new_CPlifArray},
{"PlifMatrix", SHOGUN_BASIC_CLASS __new_CPlifMatrix},
{"SegmentLoss", SHOGUN_BASIC_CLASS __new_CSegmentLoss},
{"Sequence", SHOGUN_BASIC_CLASS __new_CSequence},
{"SequenceLabels", SHOGUN_BASIC_CLASS __new_CSequenceLabels},
{"SOSVMHelper", SHOGUN_BASIC_CLASS __new_CSOSVMHelper},
{"StochasticSOSVM", SHOGUN_BASIC_CLASS __new_CStochasticSOSVM},
{"TwoStateModel", SHOGUN_BASIC_CLASS __new_CTwoStateModel},
{"DomainAdaptationSVM", SHOGUN_BASIC_CLASS __new_CDomainAdaptationSVM},
{"MultitaskClusteredLogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskClusteredLogisticRegression},
{"MultitaskKernelMaskNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelMaskNormalizer},
{"MultitaskKernelMaskPairNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelMaskPairNormalizer},
{"MultitaskKernelNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelNormalizer},
{"MultitaskKernelPlifNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelPlifNormalizer},
{"Node", SHOGUN_BASIC_CLASS __new_CNode},
{"Taxonomy", SHOGUN_BASIC_CLASS __new_CTaxonomy},
{"MultitaskKernelTreeNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelTreeNormalizer},
{"MultitaskL12LogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskL12LogisticRegression},
{"MultitaskLeastSquaresRegression", SHOGUN_BASIC_CLASS __new_CMultitaskLeastSquaresRegression},
{"MultitaskLinearMachine", SHOGUN_BASIC_CLASS __new_CMultitaskLinearMachine},
{"MultitaskLogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskLogisticRegression},
{"MultitaskROCEvaluation", SHOGUN_BASIC_CLASS __new_CMultitaskROCEvaluation},
{"MultitaskTraceLogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskTraceLogisticRegression},
{"Task", SHOGUN_BASIC_CLASS __new_CTask},
{"TaskGroup", SHOGUN_BASIC_CLASS __new_CTaskGroup},
{"TaskTree", SHOGUN_BASIC_CLASS __new_CTaskTree},
{"GUIClassifier", SHOGUN_BASIC_CLASS __new_CGUIClassifier},
{"GUIConverter", SHOGUN_BASIC_CLASS __new_CGUIConverter},
{"GUIDistance", SHOGUN_BASIC_CLASS __new_CGUIDistance},
{"GUIFeatures", SHOGUN_BASIC_CLASS __new_CGUIFeatures},
{"GUIHMM", SHOGUN_BASIC_CLASS __new_CGUIHMM},
{"GUIKernel", SHOGUN_BASIC_CLASS __new_CGUIKernel},
{"GUILabels", SHOGUN_BASIC_CLASS __new_CGUILabels},
{"GUIMath", SHOGUN_BASIC_CLASS __new_CGUIMath},
{"GUIPluginEstimate", SHOGUN_BASIC_CLASS __new_CGUIPluginEstimate},
{"GUIPreprocessor", SHOGUN_BASIC_CLASS __new_CGUIPreprocessor},
{"GUIStructure", SHOGUN_BASIC_CLASS __new_CGUIStructure},
{"GUITime", SHOGUN_BASIC_CLASS __new_CGUITime},
{"AveragedPerceptron", SHOGUN_BASIC_CLASS __new_CAveragedPerceptron},
{"FeatureBlockLogisticRegression", SHOGUN_BASIC_CLASS __new_CFeatureBlockLogisticRegression},
{"MKLClassification", SHOGUN_BASIC_CLASS __new_CMKLClassification},
{"MKLMulticlass", SHOGUN_BASIC_CLASS __new_CMKLMulticlass},
{"MKLOneClass", SHOGUN_BASIC_CLASS __new_CMKLOneClass},
{"NearestCentroid", SHOGUN_BASIC_CLASS __new_CNearestCentroid},
{"Perceptron", SHOGUN_BASIC_CLASS __new_CPerceptron},
{"PluginEstimate", SHOGUN_BASIC_CLASS __new_CPluginEstimate},
{"GNPPLib", SHOGUN_BASIC_CLASS __new_CGNPPLib},
{"GNPPSVM", SHOGUN_BASIC_CLASS __new_CGNPPSVM},
{"GPBTSVM", SHOGUN_BASIC_CLASS __new_CGPBTSVM},
{"LibLinear", SHOGUN_BASIC_CLASS __new_CLibLinear},
{"LibSVM", SHOGUN_BASIC_CLASS __new_CLibSVM},
{"LibSVMOneClass", SHOGUN_BASIC_CLASS __new_CLibSVMOneClass},
{"MPDSVM", SHOGUN_BASIC_CLASS __new_CMPDSVM},
{"OnlineLibLinear", SHOGUN_BASIC_CLASS __new_COnlineLibLinear},
{"OnlineSVMSGD", SHOGUN_BASIC_CLASS __new_COnlineSVMSGD},
{"QPBSVMLib", SHOGUN_BASIC_CLASS __new_CQPBSVMLib},
{"SGDQN", SHOGUN_BASIC_CLASS __new_CSGDQN},
{"SVM", SHOGUN_BASIC_CLASS __new_CSVM},
{"SVMLight", SHOGUN_BASIC_CLASS __new_CSVMLight},
{"SVMLightOneClass", SHOGUN_BASIC_CLASS __new_CSVMLightOneClass},
{"SVMLin", SHOGUN_BASIC_CLASS __new_CSVMLin},
{"SVMOcas", SHOGUN_BASIC_CLASS __new_CSVMOcas},
{"SVMSGD", SHOGUN_BASIC_CLASS __new_CSVMSGD},
{"WDSVMOcas", SHOGUN_BASIC_CLASS __new_CWDSVMOcas},
{"VwNativeCacheReader", SHOGUN_BASIC_CLASS __new_CVwNativeCacheReader},
{"VwNativeCacheWriter", SHOGUN_BASIC_CLASS __new_CVwNativeCacheWriter},
{"VwAdaptiveLearner", SHOGUN_BASIC_CLASS __new_CVwAdaptiveLearner},
{"VwNonAdaptiveLearner", SHOGUN_BASIC_CLASS __new_CVwNonAdaptiveLearner},
{"VowpalWabbit", SHOGUN_BASIC_CLASS __new_CVowpalWabbit},
{"VwEnvironment", SHOGUN_BASIC_CLASS __new_CVwEnvironment},
{"VwParser", SHOGUN_BASIC_CLASS __new_CVwParser},
{"VwRegressor", SHOGUN_BASIC_CLASS __new_CVwRegressor},
{"Hierarchical", SHOGUN_BASIC_CLASS __new_CHierarchical},
{"KMeans", SHOGUN_BASIC_CLASS __new_CKMeans},
{"HashedDocConverter", SHOGUN_BASIC_CLASS __new_CHashedDocConverter},
{"AttenuatedEuclideanDistance", SHOGUN_BASIC_CLASS __new_CAttenuatedEuclideanDistance},
{"BrayCurtisDistance", SHOGUN_BASIC_CLASS __new_CBrayCurtisDistance},
{"CanberraMetric", SHOGUN_BASIC_CLASS __new_CCanberraMetric},
{"CanberraWordDistance", SHOGUN_BASIC_CLASS __new_CCanberraWordDistance},
{"ChebyshewMetric", SHOGUN_BASIC_CLASS __new_CChebyshewMetric},
{"ChiSquareDistance", SHOGUN_BASIC_CLASS __new_CChiSquareDistance},
{"CosineDistance", SHOGUN_BASIC_CLASS __new_CCosineDistance},
{"CustomDistance", SHOGUN_BASIC_CLASS __new_CCustomDistance},
{"EuclideanDistance", SHOGUN_BASIC_CLASS __new_CEuclideanDistance},
{"GeodesicMetric", SHOGUN_BASIC_CLASS __new_CGeodesicMetric},
{"HammingWordDistance", SHOGUN_BASIC_CLASS __new_CHammingWordDistance},
{"JensenMetric", SHOGUN_BASIC_CLASS __new_CJensenMetric},
{"KernelDistance", SHOGUN_BASIC_CLASS __new_CKernelDistance},
{"ManhattanMetric", SHOGUN_BASIC_CLASS __new_CManhattanMetric},
{"ManhattanWordDistance", SHOGUN_BASIC_CLASS __new_CManhattanWordDistance},
{"MinkowskiMetric", SHOGUN_BASIC_CLASS __new_CMinkowskiMetric},
{"SparseEuclideanDistance", SHOGUN_BASIC_CLASS __new_CSparseEuclideanDistance},
{"TanimotoDistance", SHOGUN_BASIC_CLASS __new_CTanimotoDistance},
{"GHMM", SHOGUN_BASIC_CLASS __new_CGHMM},
{"Histogram", SHOGUN_BASIC_CLASS __new_CHistogram},
{"HMM", SHOGUN_BASIC_CLASS __new_CHMM},
{"LinearHMM", SHOGUN_BASIC_CLASS __new_CLinearHMM},
{"PositionalPWM", SHOGUN_BASIC_CLASS __new_CPositionalPWM},
{"MajorityVote", SHOGUN_BASIC_CLASS __new_CMajorityVote},
{"MeanRule", SHOGUN_BASIC_CLASS __new_CMeanRule},
{"WeightedMajorityVote", SHOGUN_BASIC_CLASS __new_CWeightedMajorityVote},
{"ClusteringAccuracy", SHOGUN_BASIC_CLASS __new_CClusteringAccuracy},
{"ClusteringMutualInformation", SHOGUN_BASIC_CLASS __new_CClusteringMutualInformation},
{"ContingencyTableEvaluation", SHOGUN_BASIC_CLASS __new_CContingencyTableEvaluation},
{"AccuracyMeasure", SHOGUN_BASIC_CLASS __new_CAccuracyMeasure},
{"ErrorRateMeasure", SHOGUN_BASIC_CLASS __new_CErrorRateMeasure},
{"BALMeasure", SHOGUN_BASIC_CLASS __new_CBALMeasure},
{"WRACCMeasure", SHOGUN_BASIC_CLASS __new_CWRACCMeasure},
{"F1Measure", SHOGUN_BASIC_CLASS __new_CF1Measure},
{"CrossCorrelationMeasure", SHOGUN_BASIC_CLASS __new_CCrossCorrelationMeasure},
{"RecallMeasure", SHOGUN_BASIC_CLASS __new_CRecallMeasure},
{"PrecisionMeasure", SHOGUN_BASIC_CLASS __new_CPrecisionMeasure},
{"SpecificityMeasure", SHOGUN_BASIC_CLASS __new_CSpecificityMeasure},
{"CrossValidationResult", SHOGUN_BASIC_CLASS __new_CCrossValidationResult},
{"CrossValidation", SHOGUN_BASIC_CLASS __new_CCrossValidation},
{"CrossValidationMKLStorage", SHOGUN_BASIC_CLASS __new_CCrossValidationMKLStorage},
{"CrossValidationMulticlassStorage", SHOGUN_BASIC_CLASS __new_CCrossValidationMulticlassStorage},
{"CrossValidationPrintOutput", SHOGUN_BASIC_CLASS __new_CCrossValidationPrintOutput},
{"CrossValidationSplitting", SHOGUN_BASIC_CLASS __new_CCrossValidationSplitting},
{"GradientCriterion", SHOGUN_BASIC_CLASS __new_CGradientCriterion},
{"GradientEvaluation", SHOGUN_BASIC_CLASS __new_CGradientEvaluation},
{"GradientResult", SHOGUN_BASIC_CLASS __new_CGradientResult},
{"MeanAbsoluteError", SHOGUN_BASIC_CLASS __new_CMeanAbsoluteError},
{"MeanSquaredError", SHOGUN_BASIC_CLASS __new_CMeanSquaredError},
{"MeanSquaredLogError", SHOGUN_BASIC_CLASS __new_CMeanSquaredLogError},
{"MulticlassAccuracy", SHOGUN_BASIC_CLASS __new_CMulticlassAccuracy},
{"MulticlassOVREvaluation", SHOGUN_BASIC_CLASS __new_CMulticlassOVREvaluation},
{"PRCEvaluation", SHOGUN_BASIC_CLASS __new_CPRCEvaluation},
{"ROCEvaluation", SHOGUN_BASIC_CLASS __new_CROCEvaluation},
{"StratifiedCrossValidationSplitting", SHOGUN_BASIC_CLASS __new_CStratifiedCrossValidationSplitting},
{"StructuredAccuracy", SHOGUN_BASIC_CLASS __new_CStructuredAccuracy},
{"Alphabet", SHOGUN_BASIC_CLASS __new_CAlphabet},
{"BinnedDotFeatures", SHOGUN_BASIC_CLASS __new_CBinnedDotFeatures},
{"CombinedDotFeatures", SHOGUN_BASIC_CLASS __new_CCombinedDotFeatures},
{"CombinedFeatures", SHOGUN_BASIC_CLASS __new_CCombinedFeatures},
{"DataGenerator", SHOGUN_BASIC_CLASS __new_CDataGenerator},
{"DummyFeatures", SHOGUN_BASIC_CLASS __new_CDummyFeatures},
{"ExplicitSpecFeatures", SHOGUN_BASIC_CLASS __new_CExplicitSpecFeatures},
{"FactorGraphFeatures", SHOGUN_BASIC_CLASS __new_CFactorGraphFeatures},
{"FKFeatures", SHOGUN_BASIC_CLASS __new_CFKFeatures},
{"HashedDocDotFeatures", SHOGUN_BASIC_CLASS __new_CHashedDocDotFeatures},
{"HashedWDFeatures", SHOGUN_BASIC_CLASS __new_CHashedWDFeatures},
{"HashedWDFeaturesTransposed", SHOGUN_BASIC_CLASS __new_CHashedWDFeaturesTransposed},
{"ImplicitWeightedSpecFeatures", SHOGUN_BASIC_CLASS __new_CImplicitWeightedSpecFeatures},
{"LatentFeatures", SHOGUN_BASIC_CLASS __new_CLatentFeatures},
{"LBPPyrDotFeatures", SHOGUN_BASIC_CLASS __new_CLBPPyrDotFeatures},
{"PolyFeatures", SHOGUN_BASIC_CLASS __new_CPolyFeatures},
{"RandomFourierDotFeatures", SHOGUN_BASIC_CLASS __new_CRandomFourierDotFeatures},
{"RealFileFeatures", SHOGUN_BASIC_CLASS __new_CRealFileFeatures},
{"SNPFeatures", SHOGUN_BASIC_CLASS __new_CSNPFeatures},
{"SparsePolyFeatures", SHOGUN_BASIC_CLASS __new_CSparsePolyFeatures},
{"GaussianBlobsDataGenerator", SHOGUN_BASIC_CLASS __new_CGaussianBlobsDataGenerator},
{"MeanShiftDataGenerator", SHOGUN_BASIC_CLASS __new_CMeanShiftDataGenerator},
{"StreamingHashedDocDotFeatures", SHOGUN_BASIC_CLASS __new_CStreamingHashedDocDotFeatures},
{"StreamingVwFeatures", SHOGUN_BASIC_CLASS __new_CStreamingVwFeatures},
{"Subset", SHOGUN_BASIC_CLASS __new_CSubset},
{"SubsetStack", SHOGUN_BASIC_CLASS __new_CSubsetStack},
{"TOPFeatures", SHOGUN_BASIC_CLASS __new_CTOPFeatures},
{"WDFeatures", SHOGUN_BASIC_CLASS __new_CWDFeatures},
{"BinaryFile", SHOGUN_BASIC_CLASS __new_CBinaryFile},
{"CSVFile", SHOGUN_BASIC_CLASS __new_CCSVFile},
{"IOBuffer", SHOGUN_BASIC_CLASS __new_CIOBuffer},
{"LibSVMFile", SHOGUN_BASIC_CLASS __new_CLibSVMFile},
{"LineReader", SHOGUN_BASIC_CLASS __new_CLineReader},
{"Parser", SHOGUN_BASIC_CLASS __new_CParser},
{"SerializableAsciiFile", SHOGUN_BASIC_CLASS __new_CSerializableAsciiFile},
{"StreamingAsciiFile", SHOGUN_BASIC_CLASS __new_CStreamingAsciiFile},
{"StreamingFile", SHOGUN_BASIC_CLASS __new_CStreamingFile},
{"StreamingFileFromFeatures", SHOGUN_BASIC_CLASS __new_CStreamingFileFromFeatures},
{"StreamingVwCacheFile", SHOGUN_BASIC_CLASS __new_CStreamingVwCacheFile},
{"StreamingVwFile", SHOGUN_BASIC_CLASS __new_CStreamingVwFile},
{"ANOVAKernel", SHOGUN_BASIC_CLASS __new_CANOVAKernel},
{"AUCKernel", SHOGUN_BASIC_CLASS __new_CAUCKernel},
{"BesselKernel", SHOGUN_BASIC_CLASS __new_CBesselKernel},
{"CauchyKernel", SHOGUN_BASIC_CLASS __new_CCauchyKernel},
{"Chi2Kernel", SHOGUN_BASIC_CLASS __new_CChi2Kernel},
{"CircularKernel", SHOGUN_BASIC_CLASS __new_CCircularKernel},
{"CombinedKernel", SHOGUN_BASIC_CLASS __new_CCombinedKernel},
{"ConstKernel", SHOGUN_BASIC_CLASS __new_CConstKernel},
{"CustomKernel", SHOGUN_BASIC_CLASS __new_CCustomKernel},
{"DiagKernel", SHOGUN_BASIC_CLASS __new_CDiagKernel},
{"DistanceKernel", SHOGUN_BASIC_CLASS __new_CDistanceKernel},
{"ExponentialKernel", SHOGUN_BASIC_CLASS __new_CExponentialKernel},
{"GaussianARDKernel", SHOGUN_BASIC_CLASS __new_CGaussianARDKernel},
{"GaussianKernel", SHOGUN_BASIC_CLASS __new_CGaussianKernel},
{"GaussianShiftKernel", SHOGUN_BASIC_CLASS __new_CGaussianShiftKernel},
{"GaussianShortRealKernel", SHOGUN_BASIC_CLASS __new_CGaussianShortRealKernel},
{"HistogramIntersectionKernel", SHOGUN_BASIC_CLASS __new_CHistogramIntersectionKernel},
{"InverseMultiQuadricKernel", SHOGUN_BASIC_CLASS __new_CInverseMultiQuadricKernel},
{"JensenShannonKernel", SHOGUN_BASIC_CLASS __new_CJensenShannonKernel},
{"LinearARDKernel", SHOGUN_BASIC_CLASS __new_CLinearARDKernel},
{"LinearKernel", SHOGUN_BASIC_CLASS __new_CLinearKernel},
{"LogKernel", SHOGUN_BASIC_CLASS __new_CLogKernel},
{"MultiquadricKernel", SHOGUN_BASIC_CLASS __new_CMultiquadricKernel},
{"AvgDiagKernelNormalizer", SHOGUN_BASIC_CLASS __new_CAvgDiagKernelNormalizer},
{"DiceKernelNormalizer", SHOGUN_BASIC_CLASS __new_CDiceKernelNormalizer},
{"FirstElementKernelNormalizer", SHOGUN_BASIC_CLASS __new_CFirstElementKernelNormalizer},
{"IdentityKernelNormalizer", SHOGUN_BASIC_CLASS __new_CIdentityKernelNormalizer},
{"RidgeKernelNormalizer", SHOGUN_BASIC_CLASS __new_CRidgeKernelNormalizer},
{"ScatterKernelNormalizer", SHOGUN_BASIC_CLASS __new_CScatterKernelNormalizer},
{"SqrtDiagKernelNormalizer", SHOGUN_BASIC_CLASS __new_CSqrtDiagKernelNormalizer},
{"TanimotoKernelNormalizer", SHOGUN_BASIC_CLASS __new_CTanimotoKernelNormalizer},
{"VarianceKernelNormalizer", SHOGUN_BASIC_CLASS __new_CVarianceKernelNormalizer},
{"ZeroMeanCenterKernelNormalizer", SHOGUN_BASIC_CLASS __new_CZeroMeanCenterKernelNormalizer},
{"PolyKernel", SHOGUN_BASIC_CLASS __new_CPolyKernel},
{"PowerKernel", SHOGUN_BASIC_CLASS __new_CPowerKernel},
{"ProductKernel", SHOGUN_BASIC_CLASS __new_CProductKernel},
{"PyramidChi2", SHOGUN_BASIC_CLASS __new_CPyramidChi2},
{"RationalQuadraticKernel", SHOGUN_BASIC_CLASS __new_CRationalQuadraticKernel},
{"SigmoidKernel", SHOGUN_BASIC_CLASS __new_CSigmoidKernel},
{"SphericalKernel", SHOGUN_BASIC_CLASS __new_CSphericalKernel},
{"SplineKernel", SHOGUN_BASIC_CLASS __new_CSplineKernel},
{"CommUlongStringKernel", SHOGUN_BASIC_CLASS __new_CCommUlongStringKernel},
{"CommWordStringKernel", SHOGUN_BASIC_CLASS __new_CCommWordStringKernel},
{"DistantSegmentsKernel", SHOGUN_BASIC_CLASS __new_CDistantSegmentsKernel},
{"FixedDegreeStringKernel", SHOGUN_BASIC_CLASS __new_CFixedDegreeStringKernel},
{"GaussianMatchStringKernel", SHOGUN_BASIC_CLASS __new_CGaussianMatchStringKernel},
{"HistogramWordStringKernel", SHOGUN_BASIC_CLASS __new_CHistogramWordStringKernel},
{"LinearStringKernel", SHOGUN_BASIC_CLASS __new_CLinearStringKernel},
{"LocalAlignmentStringKernel", SHOGUN_BASIC_CLASS __new_CLocalAlignmentStringKernel},
{"LocalityImprovedStringKernel", SHOGUN_BASIC_CLASS __new_CLocalityImprovedStringKernel},
{"MatchWordStringKernel", SHOGUN_BASIC_CLASS __new_CMatchWordStringKernel},
{"OligoStringKernel", SHOGUN_BASIC_CLASS __new_COligoStringKernel},
{"PolyMatchStringKernel", SHOGUN_BASIC_CLASS __new_CPolyMatchStringKernel},
{"PolyMatchWordStringKernel", SHOGUN_BASIC_CLASS __new_CPolyMatchWordStringKernel},
{"RegulatoryModulesStringKernel", SHOGUN_BASIC_CLASS __new_CRegulatoryModulesStringKernel},
{"SalzbergWordStringKernel", SHOGUN_BASIC_CLASS __new_CSalzbergWordStringKernel},
{"SimpleLocalityImprovedStringKernel", SHOGUN_BASIC_CLASS __new_CSimpleLocalityImprovedStringKernel},
{"SNPStringKernel", SHOGUN_BASIC_CLASS __new_CSNPStringKernel},
{"SparseSpatialSampleStringKernel", SHOGUN_BASIC_CLASS __new_CSparseSpatialSampleStringKernel},
{"SpectrumMismatchRBFKernel", SHOGUN_BASIC_CLASS __new_CSpectrumMismatchRBFKernel},
{"SpectrumRBFKernel", SHOGUN_BASIC_CLASS __new_CSpectrumRBFKernel},
{"WeightedCommWordStringKernel", SHOGUN_BASIC_CLASS __new_CWeightedCommWordStringKernel},
{"WeightedDegreePositionStringKernel", SHOGUN_BASIC_CLASS __new_CWeightedDegreePositionStringKernel},
{"WeightedDegreeStringKernel", SHOGUN_BASIC_CLASS __new_CWeightedDegreeStringKernel},
{"TensorProductPairKernel", SHOGUN_BASIC_CLASS __new_CTensorProductPairKernel},
{"TStudentKernel", SHOGUN_BASIC_CLASS __new_CTStudentKernel},
{"WaveKernel", SHOGUN_BASIC_CLASS __new_CWaveKernel},
{"WaveletKernel", SHOGUN_BASIC_CLASS __new_CWaveletKernel},
{"WeightedDegreeRBFKernel", SHOGUN_BASIC_CLASS __new_CWeightedDegreeRBFKernel},
{"BinaryLabels", SHOGUN_BASIC_CLASS __new_CBinaryLabels},
{"FactorGraphObservation", SHOGUN_BASIC_CLASS __new_CFactorGraphObservation},
{"FactorGraphLabels", SHOGUN_BASIC_CLASS __new_CFactorGraphLabels},
{"LabelsFactory", SHOGUN_BASIC_CLASS __new_CLabelsFactory},
{"LatentLabels", SHOGUN_BASIC_CLASS __new_CLatentLabels},
{"MulticlassLabels", SHOGUN_BASIC_CLASS __new_CMulticlassLabels},
{"MulticlassMultipleOutputLabels", SHOGUN_BASIC_CLASS __new_CMulticlassMultipleOutputLabels},
{"RegressionLabels", SHOGUN_BASIC_CLASS __new_CRegressionLabels},
{"StructuredLabels", SHOGUN_BASIC_CLASS __new_CStructuredLabels},
{"LatentSOSVM", SHOGUN_BASIC_CLASS __new_CLatentSOSVM},
{"LatentSVM", SHOGUN_BASIC_CLASS __new_CLatentSVM},
{"BitString", SHOGUN_BASIC_CLASS __new_CBitString},
{"CircularBuffer", SHOGUN_BASIC_CLASS __new_CCircularBuffer},
{"Compressor", SHOGUN_BASIC_CLASS __new_CCompressor},
{"SerialComputationEngine", SHOGUN_BASIC_CLASS __new_CSerialComputationEngine},
{"JobResult", SHOGUN_BASIC_CLASS __new_CJobResult},
{"Data", SHOGUN_BASIC_CLASS __new_CData},
{"DelimiterTokenizer", SHOGUN_BASIC_CLASS __new_CDelimiterTokenizer},
{"DynamicObjectArray", SHOGUN_BASIC_CLASS __new_CDynamicObjectArray},
{"Hash", SHOGUN_BASIC_CLASS __new_CHash},
{"IndexBlock", SHOGUN_BASIC_CLASS __new_CIndexBlock},
{"IndexBlockGroup", SHOGUN_BASIC_CLASS __new_CIndexBlockGroup},
{"IndexBlockTree", SHOGUN_BASIC_CLASS __new_CIndexBlockTree},
{"ListElement", SHOGUN_BASIC_CLASS __new_CListElement},
{"List", SHOGUN_BASIC_CLASS __new_CList},
{"NGramTokenizer", SHOGUN_BASIC_CLASS __new_CNGramTokenizer},
{"Signal", SHOGUN_BASIC_CLASS __new_CSignal},
{"StructuredData", SHOGUN_BASIC_CLASS __new_CStructuredData},
{"Time", SHOGUN_BASIC_CLASS __new_CTime},
{"HingeLoss", SHOGUN_BASIC_CLASS __new_CHingeLoss},
{"LogLoss", SHOGUN_BASIC_CLASS __new_CLogLoss},
{"LogLossMargin", SHOGUN_BASIC_CLASS __new_CLogLossMargin},
{"SmoothHingeLoss", SHOGUN_BASIC_CLASS __new_CSmoothHingeLoss},
{"SquaredHingeLoss", SHOGUN_BASIC_CLASS __new_CSquaredHingeLoss},
{"SquaredLoss", SHOGUN_BASIC_CLASS __new_CSquaredLoss},
{"BaggingMachine", SHOGUN_BASIC_CLASS __new_CBaggingMachine},
{"BaseMulticlassMachine", SHOGUN_BASIC_CLASS __new_CBaseMulticlassMachine},
{"DistanceMachine", SHOGUN_BASIC_CLASS __new_CDistanceMachine},
{"ZeroMean", SHOGUN_BASIC_CLASS __new_CZeroMean},
{"KernelMachine", SHOGUN_BASIC_CLASS __new_CKernelMachine},
{"KernelMulticlassMachine", SHOGUN_BASIC_CLASS __new_CKernelMulticlassMachine},
{"KernelStructuredOutputMachine", SHOGUN_BASIC_CLASS __new_CKernelStructuredOutputMachine},
{"LinearMachine", SHOGUN_BASIC_CLASS __new_CLinearMachine},
{"LinearMulticlassMachine", SHOGUN_BASIC_CLASS __new_CLinearMulticlassMachine},
{"LinearStructuredOutputMachine", SHOGUN_BASIC_CLASS __new_CLinearStructuredOutputMachine},
{"Machine", SHOGUN_BASIC_CLASS __new_CMachine},
{"NativeMulticlassMachine", SHOGUN_BASIC_CLASS __new_CNativeMulticlassMachine},
{"OnlineLinearMachine", SHOGUN_BASIC_CLASS __new_COnlineLinearMachine},
{"StructuredOutputMachine", SHOGUN_BASIC_CLASS __new_CStructuredOutputMachine},
{"JacobiEllipticFunctions", SHOGUN_BASIC_CLASS __new_CJacobiEllipticFunctions},
{"LogDetEstimator", SHOGUN_BASIC_CLASS __new_CLogDetEstimator},
{"NormalSampler", SHOGUN_BASIC_CLASS __new_CNormalSampler},
{"Math", SHOGUN_BASIC_CLASS __new_CMath},
{"Random", SHOGUN_BASIC_CLASS __new_CRandom},
{"SparseInverseCovariance", SHOGUN_BASIC_CLASS __new_CSparseInverseCovariance},
{"Statistics", SHOGUN_BASIC_CLASS __new_CStatistics},
{"GridSearchModelSelection", SHOGUN_BASIC_CLASS __new_CGridSearchModelSelection},
{"ModelSelectionParameters", SHOGUN_BASIC_CLASS __new_CModelSelectionParameters},
{"ParameterCombination", SHOGUN_BASIC_CLASS __new_CParameterCombination},
{"RandomSearchModelSelection", SHOGUN_BASIC_CLASS __new_CRandomSearchModelSelection},
{"ECOCAEDDecoder", SHOGUN_BASIC_CLASS __new_CECOCAEDDecoder},
{"ECOCDiscriminantEncoder", SHOGUN_BASIC_CLASS __new_CECOCDiscriminantEncoder},
{"ECOCEDDecoder", SHOGUN_BASIC_CLASS __new_CECOCEDDecoder},
{"ECOCForestEncoder", SHOGUN_BASIC_CLASS __new_CECOCForestEncoder},
{"ECOCHDDecoder", SHOGUN_BASIC_CLASS __new_CECOCHDDecoder},
{"ECOCLLBDecoder", SHOGUN_BASIC_CLASS __new_CECOCLLBDecoder},
{"ECOCOVOEncoder", SHOGUN_BASIC_CLASS __new_CECOCOVOEncoder},
{"ECOCOVREncoder", SHOGUN_BASIC_CLASS __new_CECOCOVREncoder},
{"ECOCRandomDenseEncoder", SHOGUN_BASIC_CLASS __new_CECOCRandomDenseEncoder},
{"ECOCRandomSparseEncoder", SHOGUN_BASIC_CLASS __new_CECOCRandomSparseEncoder},
{"ECOCStrategy", SHOGUN_BASIC_CLASS __new_CECOCStrategy},
{"GaussianNaiveBayes", SHOGUN_BASIC_CLASS __new_CGaussianNaiveBayes},
{"GMNPLib", SHOGUN_BASIC_CLASS __new_CGMNPLib},
{"GMNPSVM", SHOGUN_BASIC_CLASS __new_CGMNPSVM},
{"KNN", SHOGUN_BASIC_CLASS __new_CKNN},
{"LaRank", SHOGUN_BASIC_CLASS __new_CLaRank},
{"MulticlassLibLinear", SHOGUN_BASIC_CLASS __new_CMulticlassLibLinear},
{"MulticlassLibSVM", SHOGUN_BASIC_CLASS __new_CMulticlassLibSVM},
{"MulticlassOCAS", SHOGUN_BASIC_CLASS __new_CMulticlassOCAS},
{"MulticlassOneVsOneStrategy", SHOGUN_BASIC_CLASS __new_CMulticlassOneVsOneStrategy},
{"MulticlassOneVsRestStrategy", SHOGUN_BASIC_CLASS __new_CMulticlassOneVsRestStrategy},
{"MulticlassSVM", SHOGUN_BASIC_CLASS __new_CMulticlassSVM},
{"ThresholdRejectionStrategy", SHOGUN_BASIC_CLASS __new_CThresholdRejectionStrategy},
{"DixonQTestRejectionStrategy", SHOGUN_BASIC_CLASS __new_CDixonQTestRejectionStrategy},
{"ScatterSVM", SHOGUN_BASIC_CLASS __new_CScatterSVM},
{"ShareBoost", SHOGUN_BASIC_CLASS __new_CShareBoost},
{"BalancedConditionalProbabilityTree", SHOGUN_BASIC_CLASS __new_CBalancedConditionalProbabilityTree},
{"RandomConditionalProbabilityTree", SHOGUN_BASIC_CLASS __new_CRandomConditionalProbabilityTree},
{"RelaxedTree", SHOGUN_BASIC_CLASS __new_CRelaxedTree},
{"Tron", SHOGUN_BASIC_CLASS __new_CTron},
{"DimensionReductionPreprocessor", SHOGUN_BASIC_CLASS __new_CDimensionReductionPreprocessor},
{"HomogeneousKernelMap", SHOGUN_BASIC_CLASS __new_CHomogeneousKernelMap},
{"LogPlusOne", SHOGUN_BASIC_CLASS __new_CLogPlusOne},
{"NormOne", SHOGUN_BASIC_CLASS __new_CNormOne},
{"PNorm", SHOGUN_BASIC_CLASS __new_CPNorm},
{"PruneVarSubMean", SHOGUN_BASIC_CLASS __new_CPruneVarSubMean},
{"RandomFourierGaussPreproc", SHOGUN_BASIC_CLASS __new_CRandomFourierGaussPreproc},
{"RescaleFeatures", SHOGUN_BASIC_CLASS __new_CRescaleFeatures},
{"SortUlongString", SHOGUN_BASIC_CLASS __new_CSortUlongString},
{"SortWordString", SHOGUN_BASIC_CLASS __new_CSortWordString},
{"SumOne", SHOGUN_BASIC_CLASS __new_CSumOne},
{"LibSVR", SHOGUN_BASIC_CLASS __new_CLibSVR},
{"MKLRegression", SHOGUN_BASIC_CLASS __new_CMKLRegression},
{"SVRLight", SHOGUN_BASIC_CLASS __new_CSVRLight},
{"HSIC", SHOGUN_BASIC_CLASS __new_CHSIC},
{"KernelMeanMatching", SHOGUN_BASIC_CLASS __new_CKernelMeanMatching},
{"LinearTimeMMD", SHOGUN_BASIC_CLASS __new_CLinearTimeMMD},
{"MMDKernelSelectionCombMaxL2", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionCombMaxL2},
{"MMDKernelSelectionCombOpt", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionCombOpt},
{"MMDKernelSelectionMax", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionMax},
{"MMDKernelSelectionMedian", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionMedian},
{"MMDKernelSelectionOpt", SHOGUN_BASIC_CLASS __new_CMMDKernelSelectionOpt},
{"QuadraticTimeMMD", SHOGUN_BASIC_CLASS __new_CQuadraticTimeMMD},
{"CCSOSVM", SHOGUN_BASIC_CLASS __new_CCCSOSVM},
{"DisjointSet", SHOGUN_BASIC_CLASS __new_CDisjointSet},
{"DualLibQPBMSOSVM", SHOGUN_BASIC_CLASS __new_CDualLibQPBMSOSVM},
{"DynProg", SHOGUN_BASIC_CLASS __new_CDynProg},
{"FactorDataSource", SHOGUN_BASIC_CLASS __new_CFactorDataSource},
{"Factor", SHOGUN_BASIC_CLASS __new_CFactor},
{"FactorGraph", SHOGUN_BASIC_CLASS __new_CFactorGraph},
{"FactorGraphModel", SHOGUN_BASIC_CLASS __new_CFactorGraphModel},
{"FactorType", SHOGUN_BASIC_CLASS __new_CFactorType},
{"TableFactorType", SHOGUN_BASIC_CLASS __new_CTableFactorType},
{"HMSVMModel", SHOGUN_BASIC_CLASS __new_CHMSVMModel},
{"IntronList", SHOGUN_BASIC_CLASS __new_CIntronList},
{"MAPInference", SHOGUN_BASIC_CLASS __new_CMAPInference},
{"MulticlassModel", SHOGUN_BASIC_CLASS __new_CMulticlassModel},
{"MulticlassSOLabels", SHOGUN_BASIC_CLASS __new_CMulticlassSOLabels},
{"Plif", SHOGUN_BASIC_CLASS __new_CPlif},
{"PlifArray", SHOGUN_BASIC_CLASS __new_CPlifArray},
{"PlifMatrix", SHOGUN_BASIC_CLASS __new_CPlifMatrix},
{"SegmentLoss", SHOGUN_BASIC_CLASS __new_CSegmentLoss},
{"Sequence", SHOGUN_BASIC_CLASS __new_CSequence},
{"SequenceLabels", SHOGUN_BASIC_CLASS __new_CSequenceLabels},
{"SOSVMHelper", SHOGUN_BASIC_CLASS __new_CSOSVMHelper},
{"StochasticSOSVM", SHOGUN_BASIC_CLASS __new_CStochasticSOSVM},
{"TwoStateModel", SHOGUN_BASIC_CLASS __new_CTwoStateModel},
{"DomainAdaptationSVM", SHOGUN_BASIC_CLASS __new_CDomainAdaptationSVM},
{"MultitaskClusteredLogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskClusteredLogisticRegression},
{"MultitaskKernelMaskNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelMaskNormalizer},
{"MultitaskKernelMaskPairNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelMaskPairNormalizer},
{"MultitaskKernelNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelNormalizer},
{"MultitaskKernelPlifNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelPlifNormalizer},
{"Node", SHOGUN_BASIC_CLASS __new_CNode},
{"Taxonomy", SHOGUN_BASIC_CLASS __new_CTaxonomy},
{"MultitaskKernelTreeNormalizer", SHOGUN_BASIC_CLASS __new_CMultitaskKernelTreeNormalizer},
{"MultitaskL12LogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskL12LogisticRegression},
{"MultitaskLeastSquaresRegression", SHOGUN_BASIC_CLASS __new_CMultitaskLeastSquaresRegression},
{"MultitaskLinearMachine", SHOGUN_BASIC_CLASS __new_CMultitaskLinearMachine},
{"MultitaskLogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskLogisticRegression},
{"MultitaskROCEvaluation", SHOGUN_BASIC_CLASS __new_CMultitaskROCEvaluation},
{"MultitaskTraceLogisticRegression", SHOGUN_BASIC_CLASS __new_CMultitaskTraceLogisticRegression},
{"Task", SHOGUN_BASIC_CLASS __new_CTask},
{"TaskGroup", SHOGUN_BASIC_CLASS __new_CTaskGroup},
{"TaskTree", SHOGUN_BASIC_CLASS __new_CTaskTree},
{"GUIClassifier", SHOGUN_BASIC_CLASS __new_CGUIClassifier},
{"GUIConverter", SHOGUN_BASIC_CLASS __new_CGUIConverter},
{"GUIDistance", SHOGUN_BASIC_CLASS __new_CGUIDistance},
{"GUIFeatures", SHOGUN_BASIC_CLASS __new_CGUIFeatures},
{"GUIHMM", SHOGUN_BASIC_CLASS __new_CGUIHMM},
{"GUIKernel", SHOGUN_BASIC_CLASS __new_CGUIKernel},
{"GUILabels", SHOGUN_BASIC_CLASS __new_CGUILabels},
{"GUIMath", SHOGUN_BASIC_CLASS __new_CGUIMath},
{"GUIPluginEstimate", SHOGUN_BASIC_CLASS __new_CGUIPluginEstimate},
{"GUIPreprocessor", SHOGUN_BASIC_CLASS __new_CGUIPreprocessor},
{"GUIStructure", SHOGUN_BASIC_CLASS __new_CGUIStructure},
{"GUITime", SHOGUN_BASIC_CLASS __new_CGUITime},
{"DenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CDenseFeatures},
{"DenseSubsetFeatures", SHOGUN_TEMPLATE_CLASS __new_CDenseSubsetFeatures},
{"HashedDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CHashedDenseFeatures},
{"HashedSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CHashedSparseFeatures},
{"MatrixFeatures", SHOGUN_TEMPLATE_CLASS __new_CMatrixFeatures},
{"SparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CSparseFeatures},
{"StreamingDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingDenseFeatures},
{"StreamingHashedDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingHashedDenseFeatures},
{"StreamingHashedSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingHashedSparseFeatures},
{"StreamingSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingSparseFeatures},
{"StreamingStringFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingStringFeatures},
{"StringFeatures", SHOGUN_TEMPLATE_CLASS __new_CStringFeatures},
{"StringFileFeatures", SHOGUN_TEMPLATE_CLASS __new_CStringFileFeatures},
{"BinaryStream", SHOGUN_TEMPLATE_CLASS __new_CBinaryStream},
{"MemoryMappedFile", SHOGUN_TEMPLATE_CLASS __new_CMemoryMappedFile},
{"SimpleFile", SHOGUN_TEMPLATE_CLASS __new_CSimpleFile},
{"ParseBuffer", SHOGUN_TEMPLATE_CLASS __new_CParseBuffer},
{"StreamingFileFromDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingFileFromDenseFeatures},
{"StreamingFileFromSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingFileFromSparseFeatures},
{"StreamingFileFromStringFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingFileFromStringFeatures},
{"Cache", SHOGUN_TEMPLATE_CLASS __new_CCache},
{"DynamicArray", SHOGUN_TEMPLATE_CLASS __new_CDynamicArray},
{"Set", SHOGUN_TEMPLATE_CLASS __new_CSet},
{"TreeMachine", SHOGUN_TEMPLATE_CLASS __new_CTreeMachine},
{"DecompressString", SHOGUN_TEMPLATE_CLASS __new_CDecompressString},
{"DenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CDenseFeatures},
{"DenseSubsetFeatures", SHOGUN_TEMPLATE_CLASS __new_CDenseSubsetFeatures},
{"HashedDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CHashedDenseFeatures},
{"HashedSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CHashedSparseFeatures},
{"MatrixFeatures", SHOGUN_TEMPLATE_CLASS __new_CMatrixFeatures},
{"SparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CSparseFeatures},
{"StreamingDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingDenseFeatures},
{"StreamingHashedDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingHashedDenseFeatures},
{"StreamingHashedSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingHashedSparseFeatures},
{"StreamingSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingSparseFeatures},
{"StreamingStringFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingStringFeatures},
{"StringFeatures", SHOGUN_TEMPLATE_CLASS __new_CStringFeatures},
{"StringFileFeatures", SHOGUN_TEMPLATE_CLASS __new_CStringFileFeatures},
{"BinaryStream", SHOGUN_TEMPLATE_CLASS __new_CBinaryStream},
{"MemoryMappedFile", SHOGUN_TEMPLATE_CLASS __new_CMemoryMappedFile},
{"SimpleFile", SHOGUN_TEMPLATE_CLASS __new_CSimpleFile},
{"ParseBuffer", SHOGUN_TEMPLATE_CLASS __new_CParseBuffer},
{"StreamingFileFromDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingFileFromDenseFeatures},
{"StreamingFileFromSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingFileFromSparseFeatures},
{"StreamingFileFromStringFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingFileFromStringFeatures},
{"Cache", SHOGUN_TEMPLATE_CLASS __new_CCache},
{"DynamicArray", SHOGUN_TEMPLATE_CLASS __new_CDynamicArray},
{"Set", SHOGUN_TEMPLATE_CLASS __new_CSet},
{"TreeMachine", SHOGUN_TEMPLATE_CLASS __new_CTreeMachine},
{"DecompressString", SHOGUN_TEMPLATE_CLASS __new_CDecompressString},
{"DenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CDenseFeatures},
{"DenseSubsetFeatures", SHOGUN_TEMPLATE_CLASS __new_CDenseSubsetFeatures},
{"HashedDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CHashedDenseFeatures},
{"HashedSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CHashedSparseFeatures},
{"MatrixFeatures", SHOGUN_TEMPLATE_CLASS __new_CMatrixFeatures},
{"SparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CSparseFeatures},
{"StreamingDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingDenseFeatures},
{"StreamingHashedDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingHashedDenseFeatures},
{"StreamingHashedSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingHashedSparseFeatures},
{"StreamingSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingSparseFeatures},
{"StreamingStringFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingStringFeatures},
{"StringFeatures", SHOGUN_TEMPLATE_CLASS __new_CStringFeatures},
{"StringFileFeatures", SHOGUN_TEMPLATE_CLASS __new_CStringFileFeatures},
{"BinaryStream", SHOGUN_TEMPLATE_CLASS __new_CBinaryStream},
{"MemoryMappedFile", SHOGUN_TEMPLATE_CLASS __new_CMemoryMappedFile},
{"SimpleFile", SHOGUN_TEMPLATE_CLASS __new_CSimpleFile},
{"ParseBuffer", SHOGUN_TEMPLATE_CLASS __new_CParseBuffer},
{"StreamingFileFromDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingFileFromDenseFeatures},
{"StreamingFileFromSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingFileFromSparseFeatures},
{"StreamingFileFromStringFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingFileFromStringFeatures},
{"Cache", SHOGUN_TEMPLATE_CLASS __new_CCache},
{"DynamicArray", SHOGUN_TEMPLATE_CLASS __new_CDynamicArray},
{"Set", SHOGUN_TEMPLATE_CLASS __new_CSet},
{"TreeMachine", SHOGUN_TEMPLATE_CLASS __new_CTreeMachine},
{"DecompressString", SHOGUN_TEMPLATE_CLASS __new_CDecompressString},
{"DenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CDenseFeatures},
{"DenseSubsetFeatures", SHOGUN_TEMPLATE_CLASS __new_CDenseSubsetFeatures},
{"HashedDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CHashedDenseFeatures},
{"HashedSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CHashedSparseFeatures},
{"MatrixFeatures", SHOGUN_TEMPLATE_CLASS __new_CMatrixFeatures},
{"SparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CSparseFeatures},
{"StreamingDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingDenseFeatures},
{"StreamingHashedDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingHashedDenseFeatures},
{"StreamingHashedSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingHashedSparseFeatures},
{"StreamingSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingSparseFeatures},
{"StreamingStringFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingStringFeatures},
{"StringFeatures", SHOGUN_TEMPLATE_CLASS __new_CStringFeatures},
{"StringFileFeatures", SHOGUN_TEMPLATE_CLASS __new_CStringFileFeatures},
{"BinaryStream", SHOGUN_TEMPLATE_CLASS __new_CBinaryStream},
{"MemoryMappedFile", SHOGUN_TEMPLATE_CLASS __new_CMemoryMappedFile},
{"SimpleFile", SHOGUN_TEMPLATE_CLASS __new_CSimpleFile},
{"ParseBuffer", SHOGUN_TEMPLATE_CLASS __new_CParseBuffer},
{"StreamingFileFromDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingFileFromDenseFeatures},
{"StreamingFileFromSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingFileFromSparseFeatures},
{"StreamingFileFromStringFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingFileFromStringFeatures},
{"Cache", SHOGUN_TEMPLATE_CLASS __new_CCache},
{"DynamicArray", SHOGUN_TEMPLATE_CLASS __new_CDynamicArray},
{"Set", SHOGUN_TEMPLATE_CLASS __new_CSet},
{"TreeMachine", SHOGUN_TEMPLATE_CLASS __new_CTreeMachine},
{"DecompressString", SHOGUN_TEMPLATE_CLASS __new_CDecompressString},
{"DenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CDenseFeatures},
{"DenseSubsetFeatures", SHOGUN_TEMPLATE_CLASS __new_CDenseSubsetFeatures},
{"HashedDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CHashedDenseFeatures},
{"HashedSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CHashedSparseFeatures},
{"MatrixFeatures", SHOGUN_TEMPLATE_CLASS __new_CMatrixFeatures},
{"SparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CSparseFeatures},
{"StreamingDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingDenseFeatures},
{"StreamingHashedDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingHashedDenseFeatures},
{"StreamingHashedSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingHashedSparseFeatures},
{"StreamingSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingSparseFeatures},
{"StreamingStringFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingStringFeatures},
{"StringFeatures", SHOGUN_TEMPLATE_CLASS __new_CStringFeatures},
{"StringFileFeatures", SHOGUN_TEMPLATE_CLASS __new_CStringFileFeatures},
{"BinaryStream", SHOGUN_TEMPLATE_CLASS __new_CBinaryStream},
{"MemoryMappedFile", SHOGUN_TEMPLATE_CLASS __new_CMemoryMappedFile},
{"SimpleFile", SHOGUN_TEMPLATE_CLASS __new_CSimpleFile},
{"ParseBuffer", SHOGUN_TEMPLATE_CLASS __new_CParseBuffer},
{"StreamingFileFromDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingFileFromDenseFeatures},
{"StreamingFileFromSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingFileFromSparseFeatures},
{"StreamingFileFromStringFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingFileFromStringFeatures},
{"Cache", SHOGUN_TEMPLATE_CLASS __new_CCache},
{"DynamicArray", SHOGUN_TEMPLATE_CLASS __new_CDynamicArray},
{"Set", SHOGUN_TEMPLATE_CLASS __new_CSet},
{"TreeMachine", SHOGUN_TEMPLATE_CLASS __new_CTreeMachine},
{"DecompressString", SHOGUN_TEMPLATE_CLASS __new_CDecompressString},
{"DenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CDenseFeatures},
{"DenseSubsetFeatures", SHOGUN_TEMPLATE_CLASS __new_CDenseSubsetFeatures},
{"HashedDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CHashedDenseFeatures},
{"HashedSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CHashedSparseFeatures},
{"MatrixFeatures", SHOGUN_TEMPLATE_CLASS __new_CMatrixFeatures},
{"SparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CSparseFeatures},
{"StreamingDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingDenseFeatures},
{"StreamingHashedDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingHashedDenseFeatures},
{"StreamingHashedSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingHashedSparseFeatures},
{"StreamingSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingSparseFeatures},
{"StreamingStringFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingStringFeatures},
{"StringFeatures", SHOGUN_TEMPLATE_CLASS __new_CStringFeatures},
{"StringFileFeatures", SHOGUN_TEMPLATE_CLASS __new_CStringFileFeatures},
{"BinaryStream", SHOGUN_TEMPLATE_CLASS __new_CBinaryStream},
{"MemoryMappedFile", SHOGUN_TEMPLATE_CLASS __new_CMemoryMappedFile},
{"SimpleFile", SHOGUN_TEMPLATE_CLASS __new_CSimpleFile},
{"ParseBuffer", SHOGUN_TEMPLATE_CLASS __new_CParseBuffer},
{"StreamingFileFromDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingFileFromDenseFeatures},
{"StreamingFileFromSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingFileFromSparseFeatures},
{"StreamingFileFromStringFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingFileFromStringFeatures},
{"Cache", SHOGUN_TEMPLATE_CLASS __new_CCache},
{"DynamicArray", SHOGUN_TEMPLATE_CLASS __new_CDynamicArray},
{"Set", SHOGUN_TEMPLATE_CLASS __new_CSet},
{"TreeMachine", SHOGUN_TEMPLATE_CLASS __new_CTreeMachine},
{"DecompressString", SHOGUN_TEMPLATE_CLASS __new_CDecompressString},
{"DenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CDenseFeatures},
{"DenseSubsetFeatures", SHOGUN_TEMPLATE_CLASS __new_CDenseSubsetFeatures},
{"HashedDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CHashedDenseFeatures},
{"HashedSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CHashedSparseFeatures},
{"MatrixFeatures", SHOGUN_TEMPLATE_CLASS __new_CMatrixFeatures},
{"SparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CSparseFeatures},
{"StreamingDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingDenseFeatures},
{"StreamingHashedDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingHashedDenseFeatures},
{"StreamingHashedSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingHashedSparseFeatures},
{"StreamingSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingSparseFeatures},
{"StreamingStringFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingStringFeatures},
{"StringFeatures", SHOGUN_TEMPLATE_CLASS __new_CStringFeatures},
{"StringFileFeatures", SHOGUN_TEMPLATE_CLASS __new_CStringFileFeatures},
{"BinaryStream", SHOGUN_TEMPLATE_CLASS __new_CBinaryStream},
{"MemoryMappedFile", SHOGUN_TEMPLATE_CLASS __new_CMemoryMappedFile},
{"SimpleFile", SHOGUN_TEMPLATE_CLASS __new_CSimpleFile},
{"ParseBuffer", SHOGUN_TEMPLATE_CLASS __new_CParseBuffer},
{"StreamingFileFromDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingFileFromDenseFeatures},
{"StreamingFileFromSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingFileFromSparseFeatures},
{"StreamingFileFromStringFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingFileFromStringFeatures},
{"Cache", SHOGUN_TEMPLATE_CLASS __new_CCache},
{"DynamicArray", SHOGUN_TEMPLATE_CLASS __new_CDynamicArray},
{"Set", SHOGUN_TEMPLATE_CLASS __new_CSet},
{"TreeMachine", SHOGUN_TEMPLATE_CLASS __new_CTreeMachine},
{"DecompressString", SHOGUN_TEMPLATE_CLASS __new_CDecompressString},
{"DenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CDenseFeatures},
{"DenseSubsetFeatures", SHOGUN_TEMPLATE_CLASS __new_CDenseSubsetFeatures},
{"HashedDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CHashedDenseFeatures},
{"HashedSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CHashedSparseFeatures},
{"MatrixFeatures", SHOGUN_TEMPLATE_CLASS __new_CMatrixFeatures},
{"SparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CSparseFeatures},
{"StreamingDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingDenseFeatures},
{"StreamingHashedDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingHashedDenseFeatures},
{"StreamingHashedSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingHashedSparseFeatures},
{"StreamingSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingSparseFeatures},
{"StreamingStringFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingStringFeatures},
{"StringFeatures", SHOGUN_TEMPLATE_CLASS __new_CStringFeatures},
{"StringFileFeatures", SHOGUN_TEMPLATE_CLASS __new_CStringFileFeatures},
{"BinaryStream", SHOGUN_TEMPLATE_CLASS __new_CBinaryStream},
{"MemoryMappedFile", SHOGUN_TEMPLATE_CLASS __new_CMemoryMappedFile},
{"SimpleFile", SHOGUN_TEMPLATE_CLASS __new_CSimpleFile},
{"ParseBuffer", SHOGUN_TEMPLATE_CLASS __new_CParseBuffer},
{"StreamingFileFromDenseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingFileFromDenseFeatures},
{"StreamingFileFromSparseFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingFileFromSparseFeatures},
{"StreamingFileFromStringFeatures", SHOGUN_TEMPLATE_CLASS __new_CStreamingFileFromStringFeatures},
{"Cache", SHOGUN_TEMPLATE_CLASS __new_CCache},
{"DynamicArray", SHOGUN_TEMPLATE_CLASS __new_CDynamicArray},
{"Set", SHOGUN_TEMPLATE_CLASS __new_CSet},
{"TreeMachine", SHOGUN_TEMPLATE_CLASS __new_CTreeMachine},
{"DecompressString", SHOGUN_TEMPLATE_CLASS __new_CDecompressString},
{"StoreScalarAggregator", SHOGUN_TEMPLATE_CLASS __new_CStoreScalarAggregator},
{"ScalarResult", SHOGUN_TEMPLATE_CLASS __new_CScalarResult},
{"VectorResult", SHOGUN_TEMPLATE_CLASS __new_CVectorResult},
{"SparseMatrixOperator", SHOGUN_TEMPLATE_CLASS __new_CSparseMatrixOperator},
{"StoreScalarAggregator", SHOGUN_TEMPLATE_CLASS __new_CStoreScalarAggregator},
{"ScalarResult", SHOGUN_TEMPLATE_CLASS __new_CScalarResult},
{"VectorResult", SHOGUN_TEMPLATE_CLASS __new_CVectorResult},
{"SparseMatrixOperator", SHOGUN_TEMPLATE_CLASS __new_CSparseMatrixOperator},
{"StoreScalarAggregator", SHOGUN_TEMPLATE_CLASS __new_CStoreScalarAggregator},
{"ScalarResult", SHOGUN_TEMPLATE_CLASS __new_CScalarResult},
{"VectorResult", SHOGUN_TEMPLATE_CLASS __new_CVectorResult},
{"SparseMatrixOperator", SHOGUN_TEMPLATE_CLASS __new_CSparseMatrixOperator},
{"StoreScalarAggregator", SHOGUN_TEMPLATE_CLASS __new_CStoreScalarAggregator},
{"ScalarResult", SHOGUN_TEMPLATE_CLASS __new_CScalarResult},
{"VectorResult", SHOGUN_TEMPLATE_CLASS __new_CVectorResult},
{"SparseMatrixOperator", SHOGUN_TEMPLATE_CLASS __new_CSparseMatrixOperator},
{"StoreScalarAggregator", SHOGUN_TEMPLATE_CLASS __new_CStoreScalarAggregator},
{"ScalarResult", SHOGUN_TEMPLATE_CLASS __new_CScalarResult},
{"VectorResult", SHOGUN_TEMPLATE_CLASS __new_CVectorResult},
{"SparseMatrixOperator", SHOGUN_TEMPLATE_CLASS __new_CSparseMatrixOperator},
{"StoreScalarAggregator", SHOGUN_TEMPLATE_CLASS __new_CStoreScalarAggregator},
{"ScalarResult", SHOGUN_TEMPLATE_CLASS __new_CScalarResult},
{"VectorResult", SHOGUN_TEMPLATE_CLASS __new_CVectorResult},
{"SparseMatrixOperator", SHOGUN_TEMPLATE_CLASS __new_CSparseMatrixOperator},
{"StoreScalarAggregator", SHOGUN_TEMPLATE_CLASS __new_CStoreScalarAggregator},
{"ScalarResult", SHOGUN_TEMPLATE_CLASS __new_CScalarResult},
{"VectorResult", SHOGUN_TEMPLATE_CLASS __new_CVectorResult},
{"SparseMatrixOperator", SHOGUN_TEMPLATE_CLASS __new_CSparseMatrixOperator},
{"StoreScalarAggregator", SHOGUN_TEMPLATE_CLASS __new_CStoreScalarAggregator},
{"ScalarResult", SHOGUN_TEMPLATE_CLASS __new_CScalarResult},
{"VectorResult", SHOGUN_TEMPLATE_CLASS __new_CVectorResult},
{"SparseMatrixOperator", SHOGUN_TEMPLATE_CLASS __new_CSparseMatrixOperator},	{NULL, NULL}
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
