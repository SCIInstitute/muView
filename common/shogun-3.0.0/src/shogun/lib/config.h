#define MACHINE "Win32"
#define CONFIGURE_OPTIONS "TODO"
#define COMPFLAGS_CPP "/Wall -g  /DWIN32 /D_WINDOWS /W3 /GR /EHsc -march=native /openmp -O3 "
#define LINKFLAGS "-lpthread"

#define SFMT_MEXP 19937
#define DSFMT_MEXP 19937

/* #undef HAVE_HDF5 */
/* #undef HAVE_CURL */
/* #undef HAVE_JSON */
/* #undef HAVE_XML */
#define HAVE_LARGEFILE 1
/* #undef HAVE_DOXYGEN */
/* #undef HAVE_LAPACK */
/* #undef HAVE_MVEC */
/* #undef HAVE_PROTOBUF */

/* #undef HAVE_ARPACK */
/* #undef HAVE_EIGEN3 */
/* #undef HAVE_CATLAS */
/* #undef HAVE_ATLAS */
/* #undef HAVE_NLOPT */
/* #undef USE_LPSOLVE */
#define HAVE_PTHREAD 1
/* #undef USE_CPLEX */
/* #undef HAVE_COLPACK */
/* #undef HAVE_ARPREC */

/* #undef HAVE_POWL */
#define HAVE_LGAMMAL 1
/* #undef HAVE_SQRTL */
#define HAVE_LOG2 1
/* #undef USE_LOGCACHE */
/* #undef USE_LOGSUMARRAY */

/* #undef USE_SPINLOCKS */
#define USE_SHORTREAL_KERNELCACHE 1
#define USE_BIGSTATES 1

/* #undef USE_HMMDEBUG */
#define USE_HMMCACHE 1
/* #undef USE_HMMPARALLEL */
/* #undef USE_HMMPARALLEL_STRUCTURES */

/* #undef USE_PATHDEBUG */

#define USE_SVMLIGHT 1
/* #undef USE_MOSEK */

/* #undef USE_GLPK */
/* #undef USE_LZO */
/* #undef USE_GZIP */
/* #undef USE_BZIP2 */
/* #undef USE_LZMA */
#define USE_REFERENCE_COUNTING 1
/* #undef USE_SNAPPY */

#define HAVE_SSE2 1
#define HAVE_BUILTIN_VECTOR 1
/* #undef OCTAVE_APIVERSION */

/* #undef DARWIN */
/* #undef FREEBSD */
/* #undef LINUX */

/* #undef USE_SWIG_DIRECTORS */
/* #undef TRACE_MEMORY_ALLOCS */
/* #undef USE_JEMALLOC */

/* #undef NARRAY_LIB */

/* #undef HAVE_CXX0X */
/* #undef HAVE_CXX11 */
#define HAVE_CXX11_ATOMIC 1

#define HAVE_JBLAS 1
