
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "local_distance_kernel.h"
#include "helpers.h"
#include <string> //size_t, just for helper function
#include <cmath>

#include <iostream> //remove later DEBUG FIXME

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {


static float calculateDistance(size_t i_v, size_t j_v, const float * d_coord, size_t n_coords){
    float distsq=0;
    if(i_v == j_v)
        return 0;
    for(size_t i=0;i<n_coords;i++){
        float dist = d_coord[I2D(i_v,i,n_coords)] - d_coord[I2D(j_v,i,n_coords)];
        distsq += dist*dist;
    }
    return distsq;
}

static void set_defaults(
        float *d_dist,
        const int n_vert,
        const int n_neigh
){
    for(size_t i_v =0 ; i_v < n_vert ; i_v++){
        for(size_t n = 0; n < n_neigh; n++){
            d_dist[I2D(i_v,n,n_neigh)] = 0;
        }
    }
}
// CPU specialization
template<typename dummy>
struct LocalDistanceOpFunctor<CPUDevice,dummy> {
    void operator()(
            const CPUDevice &d,

            const int *d_neigh_idxs,
            const float *d_coords,

            float * d_distances,

            const int n_coords,
            const int n_in_vert,
            const int n_out_vert,
            const int n_neigh
    ){
        set_defaults(d_distances,n_out_vert,n_neigh);

        for(int i_v=0;i_v<n_out_vert; i_v++){
            for(int j_n=0;j_n<n_neigh; j_n++){
                int j_v = d_neigh_idxs[I2D(i_v,j_n,n_neigh)];
                if(j_v < 0)
                    continue;
                float distsq = calculateDistance(i_v,j_v,d_coords,n_coords);
                d_distances[I2D(i_v, j_n, n_neigh)] = distsq;
            }
        }

    }

};

template<typename Device>
class LocalDistanceOp : public OpKernel {
public:
    explicit LocalDistanceOp(OpKernelConstruction *context) : OpKernel(context) {
    }

    void Compute(OpKernelContext *context) override {

        /*
         *
    .Input("coordinates: float32")
    .Input("neighbour_idxs: int32")
    .Output("distances: float32");
         */

        const Tensor &t_coords = context->input(0);
        const Tensor &t_neigh_idxs = context->input(1);


        const int n_in_vert  = t_coords.dim_size(0); //same as hierarch idxs, but not as global idxs
        const int n_out_vert = t_neigh_idxs.dim_size(0);
        const int n_coords = t_coords.dim_size(1);
        const int n_neigh = t_neigh_idxs.dim_size(1);



        Tensor *t_distances = NULL;

        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({
            n_out_vert, n_neigh
        }), &t_distances));



        LocalDistanceOpFunctor<Device, int>()(
                context->eigen_device<Device>(),

                t_neigh_idxs.flat<int>().data(),
                t_coords.flat<float>().data(),

                t_distances->flat<float>().data(),

                n_coords,
                n_in_vert,
                n_out_vert,
                n_neigh

        );




    }

};

REGISTER_KERNEL_BUILDER(Name("LocalDistance").Device(DEVICE_CPU), LocalDistanceOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct LocalDistanceOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("LocalDistance").Device(DEVICE_GPU), LocalDistanceOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}//functor
}//tensorflow
