

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif 

#include "tensorflow/core/framework/op_kernel.h"
#include "helpers.h"
#include "binned_select_knn_kernel.h"
#include "binstepper.h"

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

template<int N_binning_dims>
static void select_knn_kernel(

        const float * d_coord,
        const int * d_bin_idx,
        const int * d_direction,
        const int * d_dim_bin_idx,

        const int * d_bin_boundaries,
        const int * d_n_bins,

        const float* d_bin_width,

        int *d_indices,
        float *d_dist,

        const int n_vert,
        const int n_neigh,
        const int n_coords,
        const int n_bin_dim,

        const int n_bboundaries,
        bool use_direction) {

    //bin boundaries [i] [i+1] describe the scan ranges


    //really no buffering at all here

    for(int i_v = 0; i_v < n_vert; i_v ++){//parallelise in cu
        if(i_v>=n_vert)
            return;//safe guard

        // 0: can only be neighbour, 1: can only have neighbour, 2: neither
        if(use_direction &&
                (d_direction[i_v] == 0 || d_direction[i_v] == 2))
            continue;

        size_t nfilled=1;//self-reference from defaults
        size_t maxidx_local=0;
        float maxdistsq=0;

        int total_subbins = 1;
        for(int sbi=0;sbi<n_bin_dim;sbi++)
            total_subbins *= d_n_bins[sbi];

        int iv_bin = d_bin_idx[i_v];
        int gbin_offset = total_subbins*(iv_bin / total_subbins);
        int sb_flat_offset = iv_bin - gbin_offset;

        //fill with number of bins and global bin index of the point
        //(without row splits being the first dimension)
        binstepper<N_binning_dims> stepper(d_n_bins, &d_dim_bin_idx[I2D(i_v,1,n_bin_dim+1)]);

        bool continue_search = true;
        int distance = 0;
        while(continue_search){

            stepper.set_d(distance);

            continue_search=false;

            while(true){
                int idx = stepper.step();
                if(idx<0){//not valid
                    if(!continue_search && !distance){//this should not happen
                        printf("stopping search for vtx %d at distance %d\n",i_v,distance);
                    }
                    break;

                }

                idx+=gbin_offset;

                if(idx>=n_bboundaries-1){
                    printf("idx %d out of range, gb offset %d, distance %d, sb_flat_offset %d, nbb %d\n", idx, gbin_offset, distance, sb_flat_offset,n_bboundaries);
                    continue;
                }

                int start_vertex = d_bin_boundaries[idx];
                int end_vertex = d_bin_boundaries[idx+1];

                for(size_t j_v=start_vertex;j_v<end_vertex;j_v++){
                    if(i_v == j_v)
                        continue;

                    // 0: can only be neighbour, 1: can only have neighbour, 2: neither
                    if(use_direction &&
                            (d_direction[j_v] == 1 || d_direction[j_v] == 2))
                        continue;


                    //fill up
                    float distsq = calculateDistance(i_v,j_v,d_coord,n_coords);
                    if(nfilled< n_neigh){
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

                continue_search=true;//at least one was valid

            }

            if(nfilled==n_neigh && d_bin_width[0]*distance * d_bin_width[0]*distance > maxdistsq)
                break;//done

            distance++;
        }

    }//cu parallelised loop
}



//specify  dimensions
template __global__ void select_knn_kernel<2>(
        const float * d_coord,const int * d_bin_idx,const int * d_direction,const int * d_dim_bin_idx,const int * d_bin_boundaries,
        const int * d_n_bins,const float* d_bin_width,int *d_indices,float *d_dist,const int n_vert,
        const int n_neigh,const int n_coords, const int n_bin_dim,
        const int n_bboundaries, bool use_direction);
template __global__ void select_knn_kernel<3>(
        const float * d_coord,const int * d_bin_idx,const int * d_direction,const int * d_dim_bin_idx,const int * d_bin_boundaries,
        const int * d_n_bins,const float* d_bin_width,int *d_indices,float *d_dist,const int n_vert,
        const int n_neigh,const int n_coords, const int n_bin_dim,
        const int n_bboundaries, bool use_direction);
template __global__ void select_knn_kernel<4>(
        const float * d_coord,const int * d_bin_idx,const int * d_direction,const int * d_dim_bin_idx,const int * d_bin_boundaries,
        const int * d_n_bins,const float* d_bin_width,int *d_indices,float *d_dist,const int n_vert,
        const int n_neigh,const int n_coords, const int n_bin_dim,
        const int n_bboundaries, bool use_direction);
template __global__ void select_knn_kernel<5>(
        const float * d_coord,const int * d_bin_idx,const int * d_direction,const int * d_dim_bin_idx,const int * d_bin_boundaries,
        const int * d_n_bins,const float* d_bin_width,int *d_indices,float *d_dist,const int n_vert,
        const int n_neigh,const int n_coords, const int n_bin_dim,
        const int n_bboundaries, bool use_direction);


template<typename dummy>
struct BinnedSelectKnnOpFunctor<CPUDevice, dummy> { //just because access needs to be different for GPU and CPU
    void operator()(
            const CPUDevice &d,

            const float * d_coord,
            const int * d_bin_idx,
            const int * d_direction,
            const int * d_dim_bin_idx,

            const int * d_bin_boundaries,
            const int * d_n_bins,

            const float* d_bin_width,

            int *d_indices,
            float *d_dist,

            const int n_vert,
            const int n_neigh,
            const int n_coords,
            const int n_bin_dim,

            const int n_bboundaries,
            bool tf_compat,
            bool use_direction
    ){
        set_defaults(d_indices,
                d_dist,
                tf_compat,
                n_vert,
                n_neigh);
        //really no buffering at all here

        if(n_bin_dim==2)
            select_knn_kernel<2>(d_coord,d_bin_idx,d_direction,d_dim_bin_idx,
                d_bin_boundaries,d_n_bins,d_bin_width,
                d_indices,d_dist,
                n_vert,n_neigh,n_coords,n_bin_dim,n_bboundaries,use_direction);

        if(n_bin_dim==3)
            select_knn_kernel<3>(d_coord,d_bin_idx,d_direction,d_dim_bin_idx,
                d_bin_boundaries,d_n_bins,d_bin_width,
                d_indices,d_dist,
                n_vert,n_neigh,n_coords,n_bin_dim,n_bboundaries,use_direction);

        if(n_bin_dim==4)
            select_knn_kernel<4>(d_coord,d_bin_idx,d_direction,d_dim_bin_idx,
                d_bin_boundaries,d_n_bins,d_bin_width,
                d_indices,d_dist,
                n_vert,n_neigh,n_coords,n_bin_dim,n_bboundaries,use_direction);

        if(n_bin_dim==4)
            select_knn_kernel<5>(d_coord,d_bin_idx,d_direction,d_dim_bin_idx,
                d_bin_boundaries,d_n_bins,d_bin_width,
                d_indices,d_dist,
                n_vert,n_neigh,n_coords,n_bin_dim,n_bboundaries,use_direction);

    }
};





template<typename Device>
class BinnedSelectKnnOp : public OpKernel {
public:
    explicit BinnedSelectKnnOp(OpKernelConstruction *context) : OpKernel(context) {
    
        //replace with actual configuration attributes
        
        OP_REQUIRES_OK(context,
                context->GetAttr("n_neighbours", &K_));
        OP_REQUIRES_OK(context,
                        context->GetAttr("tf_compatible", &tf_compat_));
        OP_REQUIRES_OK(context,
                        context->GetAttr("use_direction", &use_direction_));


    }


    void Compute(OpKernelContext *context) override {
        /*
         *
    .Input("coords: float")
    .Input("bin_idx: int32")
    .Input("bin_boundaries: int32")
    .Input("n_bins: int32")
    .Input("bin_width: float")

    .Output("indices: int32")
    .Output("distances: float32");
         */


        const Tensor &t_coords = context->input(0);
        const Tensor &t_bin_idx = context->input(1);
        const Tensor &t_dim_bin_idx = context->input(2);
        const Tensor &t_bin_boundaries = context->input(3);
        const Tensor &t_n_bins = context->input(4);
        const Tensor &t_bin_width = context->input(5);
        const Tensor &t_direction = context->input(6);

        const int n_vert = t_coords.dim_size(0);
        const int n_coords = t_coords.dim_size(1);
        const int n_bboundaries = t_bin_boundaries.dim_size(0);
        const int n_bin_dims_withrs = t_dim_bin_idx.dim_size(1);
        const int n_bin_dims = t_n_bins.dim_size(0);

        //checks

        OP_REQUIRES(context, n_bin_dims>1,
                    errors::InvalidArgument("BinnedSelectKnnOp expects at least 2 binning dimensions."));
        OP_REQUIRES(context, n_bin_dims<6,
                    errors::InvalidArgument("BinnedSelectKnnOp expects maximum 5 binning dimensions."));
        OP_REQUIRES(context, n_bin_dims_withrs-1 == n_bin_dims,
                    errors::InvalidArgument("BinnedSelectKnnOp expects number of bin dimensions (including row splits) -1 == number of total bin dimensions."));
        OP_REQUIRES(context, n_coords>1,
                    errors::InvalidArgument("BinnedSelectKnnOp expects at least 2 dimensions."));
        OP_REQUIRES(context, n_vert == t_bin_idx.dim_size(0),
                    errors::InvalidArgument("BinnedSelectKnnOp expects same first dimension for bin idx and coordinates."));
        OP_REQUIRES(context, 1 == t_bin_width.dim_size(0),
                    errors::InvalidArgument("BinnedSelectKnnOp expects singleton (dim(1)) for bin width."));

        if(use_direction_){
            OP_REQUIRES(context, t_direction.dim_size(0) == n_vert,
                        errors::InvalidArgument("BinnedSelectKnnOp needs a direction tensor (0,1,2,other) with lendth nvert is use_direction=True."));
        }


        TensorShape outputShape;
        outputShape.AddDim(n_vert);
        outputShape.AddDim(K_);

        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, outputShape, &output_tensor));

        Tensor *output_distances = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, outputShape, &output_distances));


        BinnedSelectKnnOpFunctor<Device, int>() (
                context->eigen_device<Device>(),

                t_coords.flat<float>().data(),
                t_bin_idx.flat<int>().data(),
                t_direction.flat<int>().data(),
                t_dim_bin_idx.flat<int>().data(),

                t_bin_boundaries.flat<int>().data(),
                t_n_bins.flat<int>().data(),

                t_bin_width.flat<float>().data(),

                output_tensor->flat<int>().data(),
                output_distances->flat<float>().data(),

                n_vert,
                K_,
                n_coords,
                n_bin_dims,

                n_bboundaries,
                tf_compat_,
                use_direction_

        );



    }
private:
    int K_;
    bool tf_compat_;
    bool use_direction_;

};

REGISTER_KERNEL_BUILDER(Name("BinnedSelectKnn").Device(DEVICE_CPU), BinnedSelectKnnOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct BinnedSelectKnnOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("BinnedSelectKnn").Device(DEVICE_GPU), BinnedSelectKnnOp<GPUDevice>);
#endif  

}//functor
}//tensorflow
