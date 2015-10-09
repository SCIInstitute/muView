/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn
 */

/* Multitask renames */
%rename(MultitaskKernelNormalizer) CMultitaskKernelNormalizer;
%rename(MultitaskKernelMklNormalizer) CMultitaskKernelMklNormalizer;
%rename(MultitaskKernelTreeNormalizer) CMultitaskKernelTreeNormalizer;
%rename(MultitaskKernelMaskNormalizer) CMultitaskKernelMaskNormalizer;
%rename(MultitaskKernelMaskPairNormalizer) CMultitaskKernelMaskPairNormalizer;
%rename(MultitaskKernelPlifNormalizer) CMultitaskKernelPlifNormalizer;

%rename(Task) CTask;
%rename(TaskGroup) CTaskGroup;
%rename(TaskTree) CTaskTree;
%rename(MultitaskLSRegression) CMultitaskLSRegression;

%rename(LibLinearMTL) CLibLinearMTL;

/* Domain adaptation renames */
#ifdef USE_SVMLIGHT
%rename(DomainAdaptationSVM) CDomainAdaptationSVM;
#endif //USE_SVMLIGHT
%rename(DomainAdaptationSVMLinear) CDomainAdaptationSVMLinear;


/* Multitask includes */
%include <shogun/transfer/multitask/MultitaskKernelNormalizer.h>
%include <shogun/transfer/multitask/MultitaskKernelMklNormalizer.h>
%include <shogun/transfer/multitask/MultitaskKernelTreeNormalizer.h>
%include <shogun/transfer/multitask/MultitaskKernelMaskNormalizer.h>
%include <shogun/transfer/multitask/MultitaskKernelMaskPairNormalizer.h>
%include <shogun/transfer/multitask/MultitaskKernelPlifNormalizer.h>

%include <shogun/transfer/multitask/TaskRelation.h>
%include <shogun/transfer/multitask/Task.h>
%include <shogun/transfer/multitask/TaskGroup.h>
%include <shogun/transfer/multitask/TaskTree.h>
%include <shogun/transfer/multitask/MultitaskLSRegression.h>

%include <shogun/transfer/multitask/LibLinearMTL.h>

/* Domain adaptation includes */
#ifdef USE_SVMLIGHT
%include <shogun/transfer/domain_adaptation/DomainAdaptationSVM.h>
#endif // USE_SVMLIGHT
%include <shogun/transfer/domain_adaptation/DomainAdaptationSVMLinear.h>
