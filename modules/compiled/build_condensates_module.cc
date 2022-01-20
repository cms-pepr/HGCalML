#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
/*
 *
 * Just creates indices
 * Each vertex gets assigned a condensation point index
 * or to noise (-1)
 *
 * "soft" is a bit like softNMS
 *
 * summed output will have full vertex dimension.
 * To get summed properties, use summed[is_cpoint] and keep an eye to row splits.
 *
 * The sum is always weighted by exp(-x**2/(2*(distance*radius)**2))
 *
 */


REGISTER_OP("BuildCondensates")
    .Attr("radius: float")
    .Attr("min_beta: float")
    .Attr("soft: bool")
    .Attr("sum: bool")
    .Input("ccoords: float32")
    .Input("betas: float32")
    .Input("dist: float32")
    .Input("tosum: float32")
    .Input("row_splits: int32")
    .Output("asso_idx: int32")
    .Output("is_cpoint: int32")
    .Output("n_condensates: int32")
    .Output("summed: float32");



