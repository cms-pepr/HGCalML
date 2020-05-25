
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "select_knn_kernel.h"
#include "helpers.h"
#include <string> //size_t, just for helper function
#include <cmath>

#include <iostream> //remove later DEBUG FIXME

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

//purely debug function
void coutVector(size_t i_v,  int* d_indices, int n_neigh, const float * d_coord, int n_coords){

    for(size_t n=0;n<n_neigh;n++){
        size_t gidx = d_indices[I2D(i_v,n,n_neigh)];
        float distsq = calculateDistance(i_v,gidx,d_coord,n_coords);
        std::cout << "("<< n << ": " << gidx << ": " << distsq << ")  ";
    }
    std::cout << std::endl;

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

// CPU specialization
template<typename dummy>
struct SelectKnnOpFunctor<CPUDevice, dummy> {
    void operator()(const CPUDevice &d,

            const float *d_coord,
            const int* d_row_splits,
            int *d_indices,
            float *d_dist,

            const int n_vert,
            const int n_neigh,
            const int n_coords,

            const int n_rs,
            const bool tf_compat,
            const float max_radius) {

        //really no buffering at all here

        for(size_t j_rs=0;j_rs<n_rs-1;j_rs++){
            const size_t start_vert = d_row_splits[j_rs];
            const size_t end_vert = d_row_splits[j_rs+1];

            if(end_vert-start_vert < n_neigh){
                throw std::runtime_error("SelectKnn: K > V"); //should be replaced by TF version of it
            }

            for(size_t i_v=start_vert;i_v<end_vert;i_v++){

                d_indices[I2D(i_v,0,n_neigh)] = i_v;
                if(!tf_compat){//allow for <K neighbours
                    for(size_t nn=1;nn<n_neigh;nn++)
                        d_indices[I2D(i_v,nn,n_neigh)] = -1;
                }
                else{
                    for(size_t nn=1;nn<n_neigh;nn++)
                        d_indices[I2D(i_v,nn,n_neigh)] = i_v;
                }
                size_t max_neighbours = n_neigh;
                //set default to self
                if(end_vert - start_vert<n_neigh){
                    max_neighbours=end_vert - start_vert;
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
                    if(distsq < maxdistsq){
                        //replace former max
                        d_indices[I2D(i_v,maxidx_local,n_neigh)] = j_v;
                        d_dist[I2D(i_v,maxidx_local,n_neigh)] = distsq;
                        //search new max
                        maxidx_local = searchLargestDistance(i_v,d_dist,n_neigh,maxdistsq);
                    }
                }
            }//vert
        }
    }
};

template<typename Device>
class SelectKnnOp : public OpKernel {
public:
    explicit SelectKnnOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context,
                        context->GetAttr("n_neighbours", &K_));
        OP_REQUIRES_OK(context,
                        context->GetAttr("tf_compatible", &tf_compat_));
        OP_REQUIRES_OK(context,
                        context->GetAttr("max_radius", &max_radius_));

        if(max_radius_>0)
            max_radius_ *= max_radius_;//use squared

    }

    void Compute(OpKernelContext *context) override {

        const Tensor &d_coord_tensor = context->input(0);
        const Tensor &d_rs_tensor = context->input(1);


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


        SelectKnnOpFunctor<Device, int>()(
                context->eigen_device<Device>(),

                d_coord_tensor.flat<float>().data(),
                d_rs_tensor.flat<int>().data(),
                output_tensor->flat<int>().data(),
                output_distances->flat<float>().data(),

                n_vert,
                K_,
                n_coords,

                n_rs,
                tf_compat_,
                max_radius_
        );



    }

private:
    int K_;
    bool tf_compat_;
    float max_radius_;
};

REGISTER_KERNEL_BUILDER(Name("SelectKnn").Device(DEVICE_CPU), SelectKnnOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct SelectKnnOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("SelectKnn").Device(DEVICE_GPU), SelectKnnOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}//functor
}//tensorflow
