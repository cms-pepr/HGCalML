#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
/*
 *
 * Just creates indices
 * Each vertex gets assigned a condensation point index
 * or to noise (-1)
 *
 * "soft" is like softNMS
 *
 *
 */


REGISTER_OP("PseudoRowSplit")
    .Input("in_asso_idx: int32")
    .Input("unique_assos: int32")
    .Input("n_per_rs: int32")
    .Input("row_splits: int32")
    .Output("asso_idx: int32")
    .Output("pseudo_rs: int32");



