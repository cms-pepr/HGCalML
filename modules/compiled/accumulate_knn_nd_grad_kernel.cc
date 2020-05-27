#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "accumulate_knn_nd_grad_kernel.h"
#include "helpers.h"
#include <string> //size_t, just for helper function
#include <cmath>

#include <iostream> //remove later DEBUG FIXME

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
namespace functor {


float distanceWeight(float distsq){
    if(!distsq)return 1;
    return exp(-1.*ACCUMULATE_KNN_EXPONENT* distsq);
}

float distWeightD(const float *d_coord, size_t i, size_t j, size_t n_coords){
    float distsq=0;
    for(size_t i_c=0;i_c<n_coords;i_c++){
        float xic = d_coord[I2D(i,i_c,n_coords)];
        float xkc = d_coord[I2D(j,  i_c,n_coords)];
        distsq += (xic-xkc)*(xic-xkc);
    }
    return distanceWeight(distsq);
}

float delta(int k, int m){
    if (k==m) return 1;
    return 0;
}

void acc_knn_nd_grad_features(
        const float *d_grad_from_out_features,
        const float *d_coord,
        const float *d_feat, // sum(V) x F
        const int *d_max_feat_indices,
        const int * d_neigh_indices,
        float *d_out_grad_coords,
        float *d_out_grad_features,
        const int n_vert,
        const int n_neigh,
        const int n_coords,
        const int n_feat,
        const int n_grad_from_out_feat,
        const int n_moments){

    for (size_t i_v = 0; i_v < n_vert; i_v++)
        for(size_t nu_f=0;nu_f<n_feat;nu_f++)
            d_out_grad_features[I2D(i_v, nu_f, n_feat)]=0;


    for (size_t i_v = 0; i_v < n_vert; i_v++){
        for(size_t nu_f=0;nu_f<n_feat;nu_f++){
            for(size_t i_c=0;i_c<n_coords;i_c++){


                float vic = d_coord[I2D(i_v,i_c,n_coords)];

                for(size_t i_i_n = 0; i_i_n < n_neigh; i_i_n++){

                    size_t m_v = d_neigh_indices[I2D(i_v, i_i_n, n_neigh)];

                    float vnc = d_coord[I2D(m_v,i_c,n_coords)];
                    float distsq_im = (vic-vnc)*(vic-vnc);

                    float weight_imnu = distanceWeight(distsq_im);

                    float contrib = d_grad_from_out_features[I3D(i_v, nu_f, i_c, n_grad_from_out_feat, n_coords)]
                                                             / (float)n_neigh  * weight_imnu;
                    float maxgrad = d_grad_from_out_features[I3D(i_v, nu_f+n_feat, i_c,n_grad_from_out_feat, n_coords)];
                    //from max
                    size_t max_for_iv = d_max_feat_indices[I3D(i_v,nu_f,i_c,n_feat,n_coords)];

                    if(m_v ==  max_for_iv){
                        contrib += maxgrad * weight_imnu;
                    }


                    d_out_grad_features[I2D(m_v, nu_f, n_feat)] += contrib;

                }
            }//coord
        }//feat
    }//vert

}

void acc_knn_nd_grad_coords(
        const float *d_grad_from_out_features,
        const float *d_coord,
        const float *d_feat, // sum(V) x F
        const int *d_max_feat_indices,
        const int * d_neigh_indices,
        float *d_out_grad_coords,
        float *d_out_grad_features,
        const int n_vert,
        const int n_neigh,
        const int n_coords,
        const int n_feat,
        const int n_grad_from_out_feat,
        const int n_moments){

    for (size_t i_v = 0; i_v < n_vert; i_v++)
        for(size_t nu_c=0;nu_c<n_coords;nu_c++)
            d_out_grad_coords[I2D(i_v, nu_c, n_coords)]=0;

    for (size_t i_v = 0; i_v < n_vert; i_v++){

        for(size_t i_i_n = 0; i_i_n < n_neigh; i_i_n++){
            size_t m_v = d_neigh_indices[I2D(i_v, i_i_n, n_neigh)];

            for(size_t nu_c=0;nu_c<n_coords;nu_c++){

                float mean_contrib = 0;
                float maxcontr = 0;

                for (size_t b_f = 0; b_f < n_feat; b_f++){

                    float gibnu = d_grad_from_out_features[I3D(i_v, b_f, nu_c, n_grad_from_out_feat,n_coords)];
                    float gilnu = d_grad_from_out_features[I3D(i_v, b_f+n_feat, nu_c, n_grad_from_out_feat,n_coords)];
                    size_t max_for_iv = d_max_feat_indices[I3D(i_v, b_f, nu_c, n_feat,n_coords)];

                    for(size_t ii_k =0; ii_k< n_neigh ; ii_k++){
                        size_t k = d_neigh_indices[I2D(i_v, ii_k, n_neigh)];

                        float ddelta = delta(m_v,k) - delta(m_v,i_v);

                        float diknu= d_coord[I2D(i_v,nu_c,n_coords)] - d_coord[I2D(k,  nu_c,n_coords)];

                        float wiknu = distanceWeight(diknu*diknu);

                        float fbk = d_feat[I2D(k, b_f, n_feat)];


                        mean_contrib += gibnu * wiknu * fbk * diknu * ddelta ;
                        if(k == max_for_iv){//or k ??? also wrong.. something with this index
                            maxcontr += gilnu * wiknu * fbk * diknu * ddelta ;
                        }

                    }

                }
                float add = 2. * ACCUMULATE_KNN_EXPONENT/(float) n_neigh * mean_contrib +
                        2 * ACCUMULATE_KNN_EXPONENT * maxcontr;
                //ATOMIC this is slow..
                d_out_grad_coords[I2D(m_v, nu_c, n_coords)] += add;
            }

        }//coords

    }//vert
}


// CPU specialization
template<typename dummy>
struct AccumulateKnnNdGradOpFunctor<CPUDevice, dummy> {
    void operator()(const CPUDevice &d,

            const float *d_grad_from_out_features, // sum(V) x Fopout
            const float *d_grad_from_sum_features,

            const float *d_coord, // sum(V) x S
            const float *d_feat, // sum(V) x S
            const float *d_orig_out_feat,
            const int *d_max_feat_indices, // sum(V) x Fopin
            const int * d_neigh_indices, // sum(V) x N

            float *d_out_grad_coords, //sum(V) x S
            float *d_out_grad_features, //sum(V) x Fopin

            const int n_vert,
            const int n_neigh,
            const int n_coords,
            const int n_feat,
            const int n_grad_from_out_feat,
            const int n_moments) {

        //CPU implementation


        acc_knn_nd_grad_features(
                d_grad_from_out_features,
                d_coord,
                d_feat,
                d_max_feat_indices,
                d_neigh_indices,
                d_out_grad_coords,
                d_out_grad_features,
                n_vert,
                n_neigh,
                n_coords,
                n_feat,
                n_grad_from_out_feat,
                n_moments);

        acc_knn_nd_grad_coords(
                d_grad_from_out_features,
                d_coord,
                d_feat,
                d_max_feat_indices,
                d_neigh_indices,
                d_out_grad_coords,
                d_out_grad_features,
                n_vert,
                n_neigh,
                n_coords,
                n_feat,
                n_grad_from_out_feat,
                n_moments);
        //and coordinates

    }
};

template<typename Device>
class AccumulateKnnNdGradOp : public OpKernel {
public:
    explicit AccumulateKnnNdGradOp(OpKernelConstruction *context) : OpKernel(context) {
    }

    void Compute(OpKernelContext *context) override {

        const Tensor &t_grad_from_out_features = context->input(0);
        const Tensor &t_grad_from_sum_features = context->input(1);
        const Tensor &t_coords = context->input(2);
        const Tensor &t_feat = context->input(3);
        const Tensor &t_neigh_indices = context->input(4);
        const Tensor &t_max_feat_indices = context->input(5);

        const Tensor &t_orig_op_outfeat = context->input(6);

        int n_in_grad_feat = t_grad_from_out_features.dim_size(1);

        int n_vert = t_grad_from_out_features.dim_size(0);

        int n_neigh = t_neigh_indices.dim_size(1);
        int n_feat = t_feat.dim_size(1);
        int n_moments = (n_in_grad_feat - n_feat*2) / n_feat ;
        int n_coords = t_coords.dim_size(1);

        TensorShape outputShapeCoords;
        outputShapeCoords.AddDim(n_vert);
        outputShapeCoords.AddDim(n_coords);


        TensorShape outputShapeFeat;
        outputShapeFeat.AddDim(n_vert);
        outputShapeFeat.AddDim(n_feat);

        Tensor *t_out_grad_coords = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, outputShapeCoords, &t_out_grad_coords));

        Tensor *t_out_grad_features = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, outputShapeFeat, &t_out_grad_features));


        AccumulateKnnNdGradOpFunctor<Device, int>()(

                context->eigen_device<Device>(),

                t_grad_from_out_features.flat<float>().data(),
                t_grad_from_sum_features.flat<float>().data(),
                t_coords.flat<float>().data(),
                t_feat.flat<float>().data(),
                t_orig_op_outfeat.flat<float>().data(),

                t_max_feat_indices.flat<int>().data(),
                t_neigh_indices.flat<int>().data(),

                t_out_grad_coords->flat<float>().data(),
                t_out_grad_features->flat<float>().data(),

                n_vert,
                n_neigh,
                n_coords,
                n_feat,

                n_in_grad_feat,
                n_moments
        );



    }

};

REGISTER_KERNEL_BUILDER(Name("AccumulateKnnNdGrad").Device(DEVICE_CPU), AccumulateKnnNdGradOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct AccumulateKnnNdGradOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("AccumulateKnnNdGrad").Device(DEVICE_GPU), AccumulateKnnNdGradOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}//functor
}//tensorflow



