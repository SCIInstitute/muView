/*
 * Copyright (c) 2009 Yahoo! Inc.  All rights reserved.  The copyrights
 * embodied in the content of this file are licensed under the BSD
 * (revised) open source license.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Adaptation of Vowpal Wabbit v5.1.
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society.
 */

#include <shogun/classifier/vw/cache/VwCacheWriter.h>

using namespace shogun;

CVwCacheWriter::CVwCacheWriter()
	: CSGObject()
{
#ifdef WIN32
	fd = 0;
#else
	fd = -1;
#endif
	env = NULL;
}

CVwCacheWriter::CVwCacheWriter(char * fname, CVwEnvironment* env_to_use)
	: CSGObject()
{
#ifdef WIN32
	fd = fopen(fname, "rw");
#else
	fd = open(fname, O_CREAT | O_TRUNC | O_RDWR, 0666);
#endif

	if (fd < 0)
		SG_SERROR("Error opening the file %s for writing cache!\n")

	env = env_to_use;
	SG_REF(env);
}

#ifdef WIN32
CVwCacheWriter::CVwCacheWriter(FILE* f, CVwEnvironment* env_to_use)
: CSGObject()
{
	fd = f;
	env = env_to_use;
	SG_REF(env);
}
#else
CVwCacheWriter::CVwCacheWriter(int32_t f, CVwEnvironment* env_to_use)
: CSGObject()
{
	fd = f;
	env = env_to_use;
	SG_REF(env);
}
#endif


CVwCacheWriter::~CVwCacheWriter()
{
	if (env)
		SG_UNREF(env);
}


#ifdef WIN32
void CVwCacheWriter::set_file(FILE * f)
{
	fd = f;
}
#else
void CVwCacheWriter::set_file(int32_t f)
{
	fd = f;
}
#endif

void CVwCacheWriter::set_env(CVwEnvironment* env_to_use)
{
	env = env_to_use;
	SG_REF(env);
}

CVwEnvironment* CVwCacheWriter::get_env()
{
	SG_REF(env);
	return env;
}
