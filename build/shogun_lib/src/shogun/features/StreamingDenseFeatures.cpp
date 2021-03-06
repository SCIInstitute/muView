#include <shogun/mathematics/Math.h>
#include <shogun/features/StreamingDenseFeatures.h>
#include <shogun/io/StreamingFileFromDenseFeatures.h>

namespace shogun
{
template <class T> CStreamingDenseFeatures<T>::CStreamingDenseFeatures() : CStreamingDotFeatures()
{
	set_read_functions();
	init();
	parser.set_free_vector_after_release(false);
}

template <class T> CStreamingDenseFeatures<T>::CStreamingDenseFeatures(CStreamingFile* file,
			 bool is_labelled,
			 int32_t size)
	: CStreamingDotFeatures()
{
	init(file, is_labelled, size);
	set_read_functions();
	parser.set_free_vector_after_release(false);
}

template <class T> CStreamingDenseFeatures<T>::CStreamingDenseFeatures(CDenseFeatures<T>* dense_features,
			 float64_t* lab)
	: CStreamingDotFeatures()
{
	CStreamingFileFromDenseFeatures<T>* file;
	bool is_labelled;
	int32_t size = 1024;

	if (lab)
	{
		is_labelled = true;
		file = new CStreamingFileFromDenseFeatures<T>(dense_features, lab);
	}
	else
	{
		is_labelled = false;
		file = new CStreamingFileFromDenseFeatures<T>(dense_features);
	}

	SG_REF(file);

	init(file, is_labelled, size);
	set_read_functions();
	parser.set_free_vector_after_release(false);
	parser.set_free_vectors_on_destruct(false);
	seekable=true;
}

template <class T> CStreamingDenseFeatures<T>::~CStreamingDenseFeatures()
{
	parser.end_parser();
}

template <class T> void CStreamingDenseFeatures<T>::reset_stream()
{
	if (seekable)
	{
		((CStreamingFileFromDenseFeatures<T>*) working_file)->reset_stream();
		parser.exit_parser();
		parser.init(working_file, has_labels, 1);
		parser.set_free_vector_after_release(false);
		parser.start_parser();
	}
}

template <class T> float32_t CStreamingDenseFeatures<T>::dense_dot(const float32_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len==current_length);
	float32_t result=0;

	for (int32_t i=0; i<current_length; i++)
		result+=current_vector[i]*vec2[i];

	return result;
}

template <class T> float64_t CStreamingDenseFeatures<T>::dense_dot(const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len==current_length);
	float64_t result=0;

	for (int32_t i=0; i<current_length; i++)
		result+=current_vector[i]*vec2[i];

	return result;
}

template <class T> void CStreamingDenseFeatures<T>::add_to_dense_vec(float32_t alpha, float32_t* vec2, int32_t vec2_len , bool abs_val)
{
	ASSERT(vec2_len==current_length);

	if (abs_val)
	{
		for (int32_t i=0; i<current_length; i++)
			vec2[i]+=alpha*CMath::abs(current_vector[i]);
	}
	else
	{
		for (int32_t i=0; i<current_length; i++)
			vec2[i]+=alpha*current_vector[i];
	}
}

template <class T> void CStreamingDenseFeatures<T>::add_to_dense_vec(float64_t alpha, float64_t* vec2, int32_t vec2_len , bool abs_val)
{
	ASSERT(vec2_len==current_length);

	if (abs_val)
	{
		for (int32_t i=0; i<current_length; i++)
			vec2[i]+=alpha*CMath::abs(current_vector[i]);
	}
	else
	{
		for (int32_t i=0; i<current_length; i++)
			vec2[i]+=alpha*current_vector[i];
	}
}

template <class T> int32_t CStreamingDenseFeatures<T>::get_nnz_features_for_vector()
{
	return current_length;
}

template <class T> CFeatures* CStreamingDenseFeatures<T>::duplicate() const
{
	return new CStreamingDenseFeatures<T>(*this);
}

template <class T> int32_t CStreamingDenseFeatures<T>::get_num_vectors() const
{
	if (current_vector)
		return 1;
	return 0;
}

template <class T> int32_t CStreamingDenseFeatures<T>::get_size() const
{
	return sizeof(T);
}

template <class T>
void CStreamingDenseFeatures<T>::set_vector_reader()
{
	parser.set_read_vector(&CStreamingFile::get_vector);
}

template <class T>
void CStreamingDenseFeatures<T>::set_vector_and_label_reader()
{
	parser.set_read_vector_and_label(&CStreamingFile::get_vector_and_label);
}

#define GET_FEATURE_TYPE(f_type, sg_type)				\
template<> EFeatureType CStreamingDenseFeatures<sg_type>::get_feature_type() const \
{									\
	return f_type;							\
}

GET_FEATURE_TYPE(F_BOOL, bool)
GET_FEATURE_TYPE(F_CHAR, char)
GET_FEATURE_TYPE(F_BYTE, uint8_t)
GET_FEATURE_TYPE(F_BYTE, int8_t)
GET_FEATURE_TYPE(F_SHORT, int16_t)
GET_FEATURE_TYPE(F_WORD, uint16_t)
GET_FEATURE_TYPE(F_INT, int32_t)
GET_FEATURE_TYPE(F_UINT, uint32_t)
GET_FEATURE_TYPE(F_LONG, int64_t)
GET_FEATURE_TYPE(F_ULONG, uint64_t)
GET_FEATURE_TYPE(F_SHORTREAL, float32_t)
GET_FEATURE_TYPE(F_DREAL, float64_t)
GET_FEATURE_TYPE(F_LONGREAL, floatmax_t)
#undef GET_FEATURE_TYPE


template <class T>
void CStreamingDenseFeatures<T>::init()
{
	working_file=NULL;
	current_vector=NULL;
	seekable=false;
	current_length=-1;
}

template <class T>
void CStreamingDenseFeatures<T>::init(CStreamingFile* file,
				    bool is_labelled,
				    int32_t size)
{
	init();
	has_labels = is_labelled;
	working_file = file;
	parser.init(file, is_labelled, size);
	seekable=false;
}

template <class T>
void CStreamingDenseFeatures<T>::start_parser()
{
	if (!parser.is_running())
		parser.start_parser();
}

template <class T>
void CStreamingDenseFeatures<T>::end_parser()
{
	parser.end_parser();
}

template <class T>
bool CStreamingDenseFeatures<T>::get_next_example()
{
	bool ret_value;
	ret_value = (bool) parser.get_next_example(current_vector,
						   current_length,
						   current_label);

	return ret_value;
}

template <class T>
SGVector<T> CStreamingDenseFeatures<T>::get_vector()
{
	current_sgvector.vector=current_vector;
	current_sgvector.vlen=current_length;

	return current_sgvector;
}

template <class T>
float64_t CStreamingDenseFeatures<T>::get_label()
{
	ASSERT(has_labels);

	return current_label;
}

template <class T>
void CStreamingDenseFeatures<T>::release_example()
{
	parser.finalize_example();
}

template <class T>
int32_t CStreamingDenseFeatures<T>::get_dim_feature_space() const
{
	return current_length;
}

template <class T>
	float32_t CStreamingDenseFeatures<T>::dot(CStreamingDotFeatures* df)
{
	ASSERT(df);
	ASSERT(df->get_feature_type() == get_feature_type());
	ASSERT(df->get_feature_class() == get_feature_class());
	CStreamingDenseFeatures<T>* sf = (CStreamingDenseFeatures<T>*) df;

	SGVector<T> other_vector=sf->get_vector();

	return SGVector<T>::dot(current_vector, other_vector.vector, current_length);
}

template <class T>
float32_t CStreamingDenseFeatures<T>::dot(SGVector<T> sgvec1)
{
	int32_t len1;
	len1=sgvec1.vlen;

	if (len1 != current_length)
		SG_ERROR("Lengths %d and %d not equal while computing dot product!\n", len1, current_length);

	return SGVector<T>::dot(current_vector, sgvec1.vector, len1);
}

template <class T>
int32_t CStreamingDenseFeatures<T>::get_num_features()
{
	return current_length;
}

template <class T>
EFeatureClass CStreamingDenseFeatures<T>::get_feature_class() const
{
	return C_STREAMING_DENSE;
}

template class CStreamingDenseFeatures<bool>;
template class CStreamingDenseFeatures<char>;
template class CStreamingDenseFeatures<int8_t>;
template class CStreamingDenseFeatures<uint8_t>;
template class CStreamingDenseFeatures<int16_t>;
template class CStreamingDenseFeatures<uint16_t>;
template class CStreamingDenseFeatures<int32_t>;
template class CStreamingDenseFeatures<uint32_t>;
template class CStreamingDenseFeatures<int64_t>;
template class CStreamingDenseFeatures<uint64_t>;
template class CStreamingDenseFeatures<float32_t>;
template class CStreamingDenseFeatures<float64_t>;
template class CStreamingDenseFeatures<floatmax_t>;
}
