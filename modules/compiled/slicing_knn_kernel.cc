
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "slicing_knn_kernel.h"
#include "helpers.h"
#include <string> //size_t, just for helper function
#include <cmath>
#include <vector>

#include <iostream> //remove later DEBUG FIXME

using namespace std;

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

float calculateDistance(size_t i_v, size_t j_v, const float * d_coord, size_t n_coords){
    float distsq=0;
    if(i_v == j_v)
        return 0;
    for(size_t i=0;i<n_coords;i++){
        float dist = d_coord[I2D(i_v,i,n_coords)] - d_coord[I2D(j_v,i,n_coords)];
        distsq += dist*dist;
    }
    return distsq;
}

int searchLargestDistance(int i_v, float* d_dist, int n_neigh, float& maxdist){

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

void set_defaults(
        int *d_indices,
        float *d_dist,
        const int n_vert,
        const int n_neigh
){
    for(size_t i_v =0 ; i_v < n_vert ; i_v++){
        for(size_t n = 0; n < n_neigh; n++){

            if(n){
                d_indices[I2D(i_v,n,n_neigh)] = i_v;
            }
            else{
                d_indices[I2D(i_v,n,n_neigh)] = i_v;
            }
            d_dist[I2D(i_v,n,n_neigh)] = 0;

        }
    }
}

void select_knn_kernel(
        const float *d_coord,
        const int* d_row_splits,
        int *d_indices,
        float *d_dist,

        const int n_vert,
        const int n_neigh,
        const int n_coords,

        const int j_rs,
        const float max_radius
        ) {

    //really no buffering at all here

    const size_t start_vert = d_row_splits[j_rs];
    const size_t end_vert = d_row_splits[j_rs+1];

    for(size_t i_v = start_vert; i_v < end_vert; i_v ++){
        if(i_v>=n_vert)
            return;//this will be a problem with actual RS, just a safety net

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

            //fill up
            float distsq = calculateDistance(i_v,j_v,d_coord,n_coords);
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
struct SlicingKnnOpFunctor<CPUDevice, dummy> {
    void operator()(const CPUDevice &d,

            const float *d_coords, // accessible only on GPU!!!
            const int* d_row_splits, // accessible only on GPU!
            const int* d_n_bins, // accessible only on GPU!!!
            const float* d_coords_min, // accessible only on GPU!!!
            const float* d_coords_max, // accessible only on GPU!!!
            int *neigh_idx,
            float *neigh_dist,

            const int V, // # of vertices
            const int K, // # of neighbours to be found
            const int n_coords,
            const int n_rs,

            std::vector<int> features_to_bin_on
            ) 
    {
        // debug print message
        // printf("Running SlicingKnn CPU Kernel");

        int n_vert = V;
        int n_neigh = K;

        set_defaults(neigh_idx,
                neigh_dist,
                n_vert,
                n_neigh);
        //really no buffering at all here

        for(size_t j_rs=0;j_rs<n_rs-1;j_rs++){
            select_knn_kernel(
                    d_coords,
                    d_row_splits,
                    neigh_idx,
                    neigh_dist,

                    n_vert,
                    n_neigh,
                    n_coords,

                    j_rs,
                    -1.0
                    );
        }

    }
};

template<typename Device>
class SlicingKnnOp : public OpKernel {

private:
    int K_;
    std::vector<int> features_to_bin_on;

public:
    explicit SlicingKnnOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context,
                       context->GetAttr("n_neighbours", &K_));
        OP_REQUIRES_OK(context,
                        context->GetAttr("features_to_bin_on", &features_to_bin_on));
    }

    void Compute(OpKernelContext *context) override {

        const Tensor &d_coord_tensor = context->input(0);
        const Tensor &d_rs_tensor = context->input(1);
        const Tensor &d_n_bins = context->input(2);
        const Tensor &d_coords_min = context->input(3);
        const Tensor &d_coords_max = context->input(4);

        int n_vert = d_coord_tensor.dim_size(0);
        int n_coords = d_coord_tensor.dim_size(1);
        int n_rs = d_rs_tensor.dim_size(0);

        TensorShape outputShape;
        outputShape.AddDim(n_vert);
        outputShape.AddDim(K_);

        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, outputShape, &output_tensor));

        Tensor *output_distances = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, outputShape, &output_distances));

        SlicingKnnOpFunctor<Device, int>()(
                context->eigen_device<Device>(),

                d_coord_tensor.flat<float>().data(),
                d_rs_tensor.flat<int>().data(),
                d_n_bins.flat<int>().data(),
                d_coords_min.flat<float>().data(),
                d_coords_max.flat<float>().data(),
                output_tensor->flat<int>().data(),
                output_distances->flat<float>().data(),

                n_vert,
                K_,
                n_coords,
                n_rs,

                features_to_bin_on
        );
    }

};

REGISTER_KERNEL_BUILDER(Name("SlicingKnn").Device(DEVICE_CPU), SlicingKnnOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct SlicingKnnOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("SlicingKnn").Device(DEVICE_GPU), SlicingKnnOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}//functor
}//tensorflow
