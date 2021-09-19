
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "select_mod_knn_kernel.h"
#include "helpers.h"
#include <string> //size_t, just for helper function
#include <cmath>

#include <iostream> //remove later DEBUG FIXME

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

static float calculateDistance(size_t i_v, size_t j_v, const float * d_coord,
        const float * d_coord_mod,
        size_t n_coords){
    float distsq=0;
    if(i_v == j_v)
        return 0;
    for(size_t i=0;i<n_coords;i++){
        float modi = 0;
        float modj = 0;
        for(size_t j=0;j<n_coords;j++){ //contract over axis 1
            modi += d_coord_mod[I3D(i_v, j, i, n_coords, n_coords)] * d_coord[I2D(i_v,j,n_coords)];
            modj += d_coord_mod[I3D(i_v, j, i, n_coords, n_coords)] * d_coord[I2D(j_v,j,n_coords)];
        }
        float dist = modi-modj;
        distsq += dist*dist;
    }
    return distsq;
}


static int searchLargestDistance(int i_v, float* d_dist, int n_neigh, float& maxdist){

    maxdist=0;
    int maxidx=0;
    if(n_neigh < 2)
        return maxidx;
    for(size_t n=1;n<n_neigh;n++){ //0 is self
        float distsq = d_dist[I2D(i_v,n,n_neigh)];
        if(distsq > maxdist){
            maxdist = distsq;
            maxidx = n;
        }
    }
    return maxidx;
}

static void set_defaults(
        int *d_indices,
        float *d_dist,
        const bool tf_compat,
        const int n_vert,
        const int n_neigh
){
    for(size_t i_v =0 ; i_v < n_vert ; i_v++){
        for(size_t n = 0; n < n_neigh; n++){

            if(n){
                if(tf_compat)
                    d_indices[I2D(i_v,n,n_neigh)] = i_v;
                else
                    d_indices[I2D(i_v,n,n_neigh)] = -1;
            }
            else{
                d_indices[I2D(i_v,n,n_neigh)] = i_v;
            }
            d_dist[I2D(i_v,n,n_neigh)] = 0;

        }
    }
}

static void select_knn_kernel(
        const float *d_coord,
        const float *d_coord_mod,
        const int* d_row_splits,
        const int* d_mask,
        int *d_indices,
        float *d_dist,

        const int n_vert,
        const int n_neigh,
        const int n_coords,

        const int j_rs,
        const bool tf_compat,
        const float max_radius,
        selknn::mask_mode_en mask_mode,
        selknn::mask_logic_en mask_logic) {

    //really no buffering at all here

    const size_t start_vert = d_row_splits[j_rs];
    const size_t end_vert = d_row_splits[j_rs+1];

    for(size_t i_v = start_vert; i_v < end_vert; i_v ++){
        if(i_v>=n_vert)
            return;//this will be a problem with actual RS, just a safety net


        if(mask_mode != selknn::mm_none){
            if(mask_logic == selknn::ml_and){
                if(!d_mask[i_v])
                    continue;
            }
            else{
                if(mask_mode == selknn::mm_scat && d_mask[i_v])
                    continue;
                else if(mask_mode == selknn::mm_acc && !d_mask[i_v])
                    continue;
            }
        }

        //protection against n_vert<n_neigh
        size_t nvert_in_row = end_vert - start_vert;
        size_t max_neighbours = n_neigh;
        //set default to self
        if(nvert_in_row<n_neigh){
            max_neighbours=nvert_in_row;
        }


        size_t nfilled=1;
        size_t maxidx_local=0;
        float maxdistsq=0;

        for(size_t j_v=start_vert;j_v<end_vert;j_v++){
            if(i_v == j_v)
                continue;

            if(mask_mode != selknn::mm_none){
                if(mask_logic == selknn::ml_and){
                    if(!d_mask[j_v])
                        continue;
                }
                else{
                    if(mask_mode == selknn::mm_scat && !d_mask[j_v])
                        continue;
                    else if(mask_mode == selknn::mm_acc && d_mask[j_v])
                        continue;
                }
            }

            //fill up
            float distsq = calculateDistance(i_v,j_v,d_coord,d_coord_mod,n_coords);
            if(nfilled<max_neighbours && (max_radius<=0 || max_radius>=distsq)){
                d_indices[I2D(i_v,nfilled,n_neigh)] = j_v;
                d_dist[I2D(i_v,nfilled,n_neigh)] = distsq;
                if(distsq > maxdistsq){
                    maxdistsq = distsq;
                    maxidx_local = nfilled;
                }
                nfilled++;
                continue;
            }
            if(distsq < maxdistsq){// automatically applies to max radius
                //replace former max
                d_indices[I2D(i_v,maxidx_local,n_neigh)] = j_v;
                d_dist[I2D(i_v,maxidx_local,n_neigh)] = distsq;
                //search new max
                maxidx_local = searchLargestDistance(i_v,d_dist,n_neigh,maxdistsq);
            }
        }
    }

}

// CPU specialization
template<typename dummy>
struct SelectModKnnOpFunctor<CPUDevice, dummy> {
    void operator()(const CPUDevice &d,

            const float *d_coord,
            const float *d_coord_mod,
            const int* d_row_splits,
            const int* d_mask,
            int *d_indices,
            float *d_dist,

            const int n_vert,
            const int n_neigh,
            const int n_coords,

            const int n_rs,
            const bool tf_compat,
            const float max_radius,
            selknn::mask_mode_en mask_mode,
            selknn::mask_logic_en mask_logic) {


        set_defaults(d_indices,
                d_dist,
                tf_compat,
                n_vert,
                n_neigh);
        //really no buffering at all here

        for(size_t j_rs=0;j_rs<n_rs-1;j_rs++){
            select_knn_kernel(d_coord,
                    d_coord_mod,
                    d_row_splits,
                    d_mask,
                    d_indices,
                    d_dist,

                    n_vert,
                    n_neigh,
                    n_coords,

                    j_rs,
                    tf_compat,
                    max_radius,
                    mask_mode,
                    mask_logic);
        }
    }
};

template<typename Device>
class SelectModKnnOp : public OpKernel {
public:
    explicit SelectModKnnOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context,
                        context->GetAttr("n_neighbours", &K_));
        OP_REQUIRES_OK(context,
                        context->GetAttr("tf_compatible", &tf_compat_));
        OP_REQUIRES_OK(context,
                        context->GetAttr("max_radius", &max_radius_));

        int mm_ml_int=0;
        OP_REQUIRES_OK(context,
                        context->GetAttr("mask_mode", &mm_ml_int));

        if(max_radius_>0)
            max_radius_ *= max_radius_;//use squared

        mask_mode = selknn::mm_none;
        mask_logic = selknn::ml_xor;
        //printf("mm_ml_int: %d\n",mm_ml_int);
        if(mm_ml_int>0){
            if(mm_ml_int>19){
                mask_logic = selknn::ml_and;
                mm_ml_int-=20;
               // printf("ml_and mode %d\n",(int)mask_logic);
            }
            else{
                mask_logic = selknn::ml_xor;
                mm_ml_int-=10;
               // printf("ml_xor mode %d\n",(int)mask_logic);

            }
            if(mm_ml_int == 1){
                mask_mode = selknn::mm_acc;
               // printf("mm_acc mode %d\n",(int)mask_mode);
            }
            else if(mm_ml_int == 2){
                mask_mode = selknn::mm_scat;
               // printf("mm_scat mode %d\n",(int)mask_mode);
            }

        }
    }

    void Compute(OpKernelContext *context) override {

        const Tensor &d_coord_tensor = context->input(0);
        const Tensor &d_coord_mod_tensor = context->input(1);
        const Tensor &d_rs_tensor = context->input(2);
        const Tensor &d_mask_tensor = context->input(3);


        int n_vert = d_coord_tensor.dim_size(0);
        int n_coords = d_coord_tensor.dim_size(1);
        int n_rs = d_rs_tensor.dim_size(0);

        auto coorddimsok = d_coord_mod_tensor.dims() == 3
                && d_coord_mod_tensor.dim_size(0) == n_vert
                && d_coord_mod_tensor.dim_size(1) == n_coords
                && d_coord_mod_tensor.dim_size(2) == n_coords
                ? tensorflow::Status(): tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
               "Coordinate modifier tensor needs to have 3 dimensions (V x C x C)");
        OP_REQUIRES_OK(context,coorddimsok);

        TensorShape outputShape;
        outputShape.AddDim(n_vert);
        outputShape.AddDim(K_);

        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, outputShape, &output_tensor));

        Tensor *output_distances = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, outputShape, &output_distances));


        SelectModKnnOpFunctor<Device, int>()(
                context->eigen_device<Device>(),

                d_coord_tensor.flat<float>().data(),
                d_coord_mod_tensor.flat<float>().data(),
                d_rs_tensor.flat<int>().data(),
                d_mask_tensor.flat<int>().data(),
                output_tensor->flat<int>().data(),
                output_distances->flat<float>().data(),

                n_vert,
                K_,
                n_coords,

                n_rs,
                tf_compat_,
                max_radius_,
                mask_mode,
                mask_logic
        );



    }

private:
    int K_;
    bool tf_compat_;
    float max_radius_;
    selknn::mask_mode_en mask_mode;
    selknn::mask_logic_en mask_logic;
};

REGISTER_KERNEL_BUILDER(Name("SelectModKnn").Device(DEVICE_CPU), SelectModKnnOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct SelectModKnnOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("SelectModKnn").Device(DEVICE_GPU), SelectModKnnOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}//functor
}//tensorflow
