/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef  SLEP_OPTIONS_H_
#define  SLEP_OPTIONS_H_

#define IGNORE_IN_CLASSLIST

#include <stdlib.h>

namespace shogun
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
IGNORE_IN_CLASSLIST struct slep_options
{
	bool general;
	int termination;
	double tolerance;
	int max_iter;
	int restart_num;
	int n_nodes;
	int regularization;
	int* ind;
	double* ind_t;
	double* G;
	double* initial_w;
	double q;

	static slep_options default_options()
	{
		slep_options opts;
		opts.general = false;
		opts.termination = 2;
		opts.tolerance = 1e-3;
		opts.max_iter = 1000;
		opts.restart_num = 100;
		opts.regularization = 0;
		opts.q = 2.0;
		opts.initial_w = NULL;
		opts.ind = NULL;
		opts.ind_t = NULL;
		opts.G = NULL;
		return opts;
	}
};
#endif
}
#endif   /* ----- #ifndef SLEP_OPTIONS_H_  ----- */


