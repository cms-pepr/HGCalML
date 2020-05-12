#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "accumulate_knn_grad_kernel.h"
#include "helpers.h"
#include <string> //size_t, just for helper function
#include <cmath>

#include <iostream> //remove later DEBUG FIXME

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
namespace functor {


float distanceWeight(float distsq, float expscaler=1.){
    if(!distsq)return 1;
    return exp(-expscaler* distsq);
}

float distWeightD(const float *d_coord, size_t i, size_t j, size_t n_coords, float expscaler){
    float distsq=0;
    for(size_t i_c=0;i_c<n_coords;i_c++){
        float xic = d_coord[I2D(i,i_c,n_coords)];
        float xkc = d_coord[I2D(j,  i_c,n_coords)];
        distsq += (xic-xkc)*(xic-xkc);
    }
    return distanceWeight(distsq,expscaler);
}

float delta(int k, int m){
    if (k==m) return 1;
    return 0;
}


// CPU specialization
template<typename dummy>
struct AccumulateKnnGradOpFunctor<CPUDevice, dummy> {
    void operator()(const CPUDevice &d,

            const float *d_grad_from_out_features, // sum(V) x Fopout
            const float *d_coord, // sum(V) x S
            const float *d_feat, // sum(V) x S
            const int *d_max_feat_indices, // sum(V) x Fopin
            const int * d_neigh_indices, // sum(V) x N

            float *d_out_grad_coords, //sum(V) x S
            float *d_out_grad_features, //sum(V) x Fopin

            int n_vert,
            int n_neigh,
            int n_coords,
            int n_feat,

            int n_grad_from_out_feat,

            int n_moments) {

        //CPU implementation

        float expscaler=1;  //FIXME

        //set zero
        for (size_t i_v = 0; i_v < n_vert; i_v++){
            for (size_t i_f = 0; i_f < n_feat; i_f++)
                d_out_grad_features[I2D(i_v, i_f, n_feat)] = 0;
            for(size_t i_c=0;i_c<n_coords;i_c++)
                d_out_grad_coords[I2D(i_v,i_c,n_coords)] = 0;
        }



        //look for neighbours, add gradient term to every *neighbour*

        for (size_t i_v = 0; i_v < n_vert; i_v++) {

            //these should be all vertices that have m as neighbour, not the other way around - here is the problem
            for(size_t i_i_n = 0; i_i_n < n_neigh; i_i_n++){

                size_t m_v = d_neigh_indices[I2D(i_v, i_i_n, n_neigh)];

                float distsq_im = 0;
                for(size_t i_c=0;i_c<n_coords;i_c++){
                    float vic = d_coord[I2D(i_v,i_c,n_coords)];
                    float vnc = d_coord[I2D(m_v,i_c,n_coords)];
                    distsq_im += (vic-vnc)*(vic-vnc);
                }
                float weight_im = distanceWeight(distsq_im,expscaler);

                for (size_t nu_f = 0; nu_f < n_feat; nu_f++){

                    float contrib=0;
                    //from mean
                    contrib +=d_grad_from_out_features[I2D(i_v, nu_f, n_grad_from_out_feat)]  / (float)n_neigh  * weight_im;

                    //from max
                    if(m_v == d_max_feat_indices[I2D(i_v,nu_f,n_feat)] ){
                        contrib += d_grad_from_out_features[I2D(i_v, nu_f+n_feat, n_grad_from_out_feat)] * weight_im;
                    }

                    d_out_grad_features[I2D(m_v, nu_f, n_feat)] += contrib;

                }
                for(size_t nu_c = 0; nu_c < n_coords; nu_c++){

                    float mean_contrib = 0;
                    float maxcontr = 0;

                    for (size_t b_f = 0; b_f < n_feat; b_f++){
                        float thisfeat_mean_contr = 0;
                        float thisfeat_max_contr = 0;
                        for(size_t ii_k =0; ii_k< n_neigh ; ii_k++){

                            size_t k = d_neigh_indices[I2D(i_v, ii_k, n_neigh)];

                            float wik = distWeightD(d_coord,i_v,k,n_coords,expscaler);

                            float distsq_ik=0;
                            float diknu= d_coord[I2D(i_v,nu_c,n_coords)] - d_coord[I2D(k,  nu_c,n_coords)];

                            //resolve delta here once works
                            thisfeat_mean_contr +=  wik * d_feat[I2D(k, b_f, n_feat)] * diknu
                                    * (delta(m_v,k) - delta(m_v,i_v));

                            if(k == d_max_feat_indices[I2D(i_v,b_f,n_feat)] ){
                                thisfeat_max_contr += wik * d_feat[I2D(k, b_f, n_feat)] * diknu
                                        * (delta(m_v,k) - delta(m_v,i_v));
                            }

                        }

                        mean_contrib +=  thisfeat_mean_contr *
                                d_grad_from_out_features[I2D(i_v, b_f, n_grad_from_out_feat)];

                        maxcontr += thisfeat_max_contr*
                                d_grad_from_out_features[I2D(i_v, b_f, n_grad_from_out_feat)];

                        //max part here? probably...

                    }
                    d_out_grad_coords[I2D(m_v, nu_c, n_coords)] += 2. * expscaler/(float) n_neigh * mean_contrib +
                            2 * expscaler * maxcontr;
                }
            }
        }

    }
};

template<typename Device>
class AccumulateKnnGradOp : public OpKernel {
public:
    explicit AccumulateKnnGradOp(OpKernelConstruction *context) : OpKernel(context) {
    }

    void Compute(OpKernelContext *context) override {

        const Tensor &t_grad_from_out_features = context->input(0);
        const Tensor &t_coords = context->input(1);
        const Tensor &t_feat = context->input(2);
        const Tensor &t_neigh_indices = context->input(3);
        const Tensor &t_max_feat_indices = context->input(4);

        int n_in_grad_feat = t_grad_from_out_features.dim_size(1);

        int n_vert = t_grad_from_out_features.dim_size(0);

        int n_neigh = t_neigh_indices.dim_size(1);
        int n_feat = t_feat.dim_size(1);
        int n_moments = n_feat*2 - n_in_grad_feat;
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


        AccumulateKnnGradOpFunctor<Device, int>()(

                context->eigen_device<Device>(),

                t_grad_from_out_features.flat<float>().data(),
                t_coords.flat<float>().data(),
                t_feat.flat<float>().data(),
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

REGISTER_KERNEL_BUILDER(Name("AccumulateKnnGrad").Device(DEVICE_CPU), AccumulateKnnGradOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct AccumulateKnnGradOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("AccumulateKnnGrad").Device(DEVICE_GPU), AccumulateKnnGradOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}//functor
}//tensorflow



