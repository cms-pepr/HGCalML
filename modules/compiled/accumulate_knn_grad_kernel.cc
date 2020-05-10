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



        float distanceWeight(float distsq){
            if(!distsq)return 1;
            return exp(-10.* distsq);
        }


        // CPU specialization
        template<typename dummy>
        struct AccumulateKnnGradOpFunctor<CPUDevice, dummy> {
            void operator()(const CPUDevice &d,

                    const float *d_grad_from_out_features, // sum(V) x Fopout
                    const float *d_coord, // sum(V) x S
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

                //set zero
                for (size_t i_v = 0; i_v < n_vert; i_v++)
                    for (size_t i_f = 0; i_f < n_feat; i_f++)
                        d_out_grad_features[I2D(i_v, i_f, n_feat)] = 0;

                //look for neighbours, add gradient term to every *neighbour*

                for (size_t i_v = 0; i_v < n_vert; i_v++) {

                    for (size_t i_n = 0; i_n < n_neigh; i_n++) {
                        size_t nidx = d_neigh_indices[I2D(i_v,i_n,n_neigh)];

                        //here should be the feature loop
                        for (size_t i_f = 0; i_f < n_feat; i_f++) {
                            float dydx = 0;

                            float distsq=0;
                            if(nidx != i_v){
                                for(size_t i_c=0;i_c<n_coords;i_c++){
                                    float vic = d_coord[I2D(i_v,i_c,n_coords)];
                                    float vnc = d_coord[I2D(nidx,i_c,n_coords)];
                                    distsq += (vic-vnc)*(vic-vnc);
                                }
                            }
                            float weight = distanceWeight(distsq); //actually calc it

                            float dydxmean = 1. / (float) n_neigh * weight;
                            dydxmean *= d_grad_from_out_features[I2D(i_v, i_f,
                                    n_grad_from_out_feat)];

                            size_t maxfidx = d_max_feat_indices[I2D(i_v, i_f, n_feat)];
                            float dydxmax = 1. ? maxfidx == i_n : 0.;
                            dydxmax *= weight;
                            dydxmax *= d_grad_from_out_features[I2D(i_v, 2 * i_f,
                                    n_grad_from_out_feat)];


                            d_out_grad_features[I2D(nidx, i_f, n_feat)] += dydxmean + dydxmax;

                        }
                    }
                }

            }
        };

        template<typename Device>
        class AccumulateKnnGradOp : public OpKernel {
        public:
            explicit AccumulateKnnGradOp(OpKernelConstruction *context) : OpKernel(context) {
                OP_REQUIRES_OK(context,
                               context->GetAttr("n_features", &n_features));
            }

            void Compute(OpKernelContext *context) override {

                const Tensor &t_grad_from_out_features = context->input(0);
                const Tensor &t_coords = context->input(1);
                const Tensor &t_neigh_indices = context->input(2);
                const Tensor &t_max_feat_indices = context->input(3);

                int n_in_grad_feat = t_grad_from_out_features.dim_size(1);

                int n_vert = t_grad_from_out_features.dim_size(0);

                int n_neigh = t_neigh_indices.dim_size(1);
                int n_feat = n_features;
                int n_moments = n_features*2 - n_in_grad_feat;
                int n_coords = t_coords.dim_size(1);

                TensorShape outputShapeCoords;
                outputShapeCoords.AddDim(n_vert);
                outputShapeCoords.AddDim(n_coords);


                TensorShape outputShapeFeat;
                outputShapeFeat.AddDim(n_vert);
                outputShapeFeat.AddDim(n_features);


                Tensor *t_out_grad_coords = NULL;
                OP_REQUIRES_OK(context, context->allocate_output(0, outputShapeCoords, &t_out_grad_coords));

                Tensor *t_out_grad_features = NULL;
                OP_REQUIRES_OK(context, context->allocate_output(1, outputShapeFeat, &t_out_grad_features));

                DEBUGCOUT(n_vert)   ;
                DEBUGCOUT(n_neigh)  ;
                DEBUGCOUT(n_coords) ;
                DEBUGCOUT(n_feat)   ;
                DEBUGCOUT(n_in_grad_feat)   ;
                DEBUGCOUT(n_moments)   ;

                AccumulateKnnGradOpFunctor<Device, int>()(

                        context->eigen_device<Device>(),

                        t_grad_from_out_features.flat<float>().data(),
                        t_coords.flat<float>().data(),
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

        private:
            int n_coords,n_features;
        };

REGISTER_KERNEL_BUILDER(Name("AccumulateKnnGrad").Device(DEVICE_CPU), AccumulateKnnGradOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct AccumulateKnnGradOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("AccumulateKnnGrad").Device(DEVICE_GPU), AccumulateKnnGradOp<GPUDevice>);
#endif  // GOOGLE_CUDA

    }
}



