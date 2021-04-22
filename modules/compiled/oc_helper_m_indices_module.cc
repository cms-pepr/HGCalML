#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

/*
 * Takes as helper input
 * y, idx, count = tf.unique_with_counts(x)
 *
 * y is the unique tensor
 * idx going to be ignored
 * count -> tf.max(count) -> nmax_per_unique
 *
 */

REGISTER_OP("MIndices")
    .Attr("calc_m_not: bool")
    .Input("asso_idxs: int32")
    .Input("unique_idxs: int32")
    .Input("nmax_per_unique: int32")
    .Output("sel_idxs: int32")
    .Output("m_not: float32");


