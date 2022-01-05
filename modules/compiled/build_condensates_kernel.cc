
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "build_condensates_kernel.h"
#include "helpers.h"
#include <string> //size_t, just for helper function
#include <cmath>

#include <iostream> //remove later DEBUG FIXME

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

//helpers here

static void set_defaults(
        int *asso_idx,
        int * is_cpoint,
        const float * d_betas,
        float * temp_betas,
        const int start_vertex,
        const int end_vertex,
        const int n_vert){

    for(size_t i_v=start_vertex;i_v<end_vertex;i_v++){
        asso_idx[i_v] = -start_vertex-1;

        temp_betas[i_v] = d_betas[i_v];

        is_cpoint[i_v] = 0;//not needed on GPU?
    }
}

static void copy_to_sum_and_default(
        const float * ref,
        float * target,
        float * target_to_zero,
        const int n_vert,
        const int n_f
){
    for(int i_v=0;i_v<n_vert;i_v++){
        for(int i_f=0;i_f<n_f;i_f++){
            target[I2D(i_v,i_f,n_f)] = ref[I2D(i_v,i_f,n_f)];
            target_to_zero[I2D(i_v,i_f,n_f)]=0;
        }
    }
}


static void get_max_beta(
        const float* temp_betas,
        int *asso_idx,
        int * is_cpoint,
        int * maxidx,

        const int n_vert,
        const int start_vertex,
        const int end_vertex,
        const float min_beta){

    //this needs to be some smart algo here
    //it will be called N_condensate times at least
    int ref=-1;
    float max = min_beta;
    for(int i_v=start_vertex;i_v<end_vertex;i_v++){
        float biv = temp_betas[i_v];
        if(biv > max && asso_idx[i_v] < 0){
            max=biv;
            ref=i_v;
        }

    }

    //if none can be found set ref to -1
    *maxidx = ref;
    if(ref>=0){
        is_cpoint[ref]=1;
        asso_idx[ref]=ref;
    }
}


static float distancesq(const int v_a,
        const int v_b,
        const float *d_ccoords,
        const int n_ccoords){
    float distsq=0;
    for(size_t i=0;i<n_ccoords;i++){
        float xa = d_ccoords[I2D(v_a,i,n_ccoords)];
        float xb = d_ccoords[I2D(v_b,i,n_ccoords)];
        distsq += (xa-xb)*(xa-xb);
    }
    return distsq;
}

static void check_and_collect(

        const int ref_vertex,
        const float ref_beta,
        const float *d_ccoords,
        const float *d_betas,
        const float *d_dist,
        const float *d_tosum,

        int *asso_idx,
        float * temp_betas,
        float * temp_tosum,
        float * summed,

        const int n_vert,
        const int n_ccoords,
        const int n_sumf,

        const int start_vertex,
        const int end_vertex,
        const float radiussq,
        const float min_beta,
        const bool soft,
        const bool sum){

    float modradiussq = d_dist[ref_vertex];
    modradiussq *= modradiussq;// squared, as distsq and radius
    modradiussq *= radiussq;

    for(size_t i_v=start_vertex;i_v<end_vertex;i_v++){

        if(asso_idx[i_v] < 0 || i_v == ref_vertex){

            float distsq = distancesq(ref_vertex,i_v,d_ccoords,n_ccoords);
            float prob = std::exp(-distsq/(2.*modradiussq));//1 sigma at radius
            if(soft){
                float subtract =  prob * ref_beta;
                float prebeta = temp_betas[i_v];
                float newbeta = prebeta-subtract;
                temp_betas[i_v] = newbeta;
            }
            if(distsq <= modradiussq){
                asso_idx[i_v] = ref_vertex;
            }
            if(sum){
                for(int i_f=0;i_f<n_sumf;i_f++){
                    float tmpfeat = temp_tosum[I2D(i_v,i_f,n_sumf)];
                    float origfeat = d_tosum[I2D(i_v,i_f,n_sumf)];
                    if(tmpfeat > 0){
                        float contrib = prob*origfeat;
                        if(contrib>tmpfeat)//larger than what's left
                            contrib = tmpfeat;
                        summed[I2D(ref_vertex,i_f,n_sumf)] += contrib;
                        temp_tosum[I2D(i_v,i_f,n_sumf)] -= contrib;
                    }
                }
            }//sum
        }//asso
    }//for
}



// CPU specialization
template<typename dummy>
struct BuildCondensatesOpFunctor<CPUDevice, dummy> {
    void operator()(const CPUDevice &d,

            const float *d_ccoords,
            const float *d_betas,
            const float *d_dist,
            const float *d_tosum,
            const int *row_splits,


            int *asso_idx,
            int *is_cpoint,
            float * temp_betas,
            int *n_condensates,
            float *temp_tosum,
            float *summed,

            const int n_vert,
            const int n_ccoords,
            const int n_sumf,

            const int n_rs,

            const float radius,
            const float min_beta,
            const bool soft,
            const bool sum) {


        if(sum){
            copy_to_sum_and_default(d_tosum,temp_tosum,summed,n_vert,n_sumf);
        }

        for(size_t j_rs=0;j_rs<n_rs-1;j_rs++){
            const int start_vertex = row_splits[j_rs];
            const int end_vertex = row_splits[j_rs+1];

            set_defaults(asso_idx,is_cpoint,d_betas,temp_betas,start_vertex,end_vertex,n_vert);

            int ref;
            get_max_beta(temp_betas,asso_idx,is_cpoint,&ref,n_vert,start_vertex,end_vertex,min_beta);
            //copy ref back
            float ref_beta = d_betas[ref];
            int ncond=0;
            //copy ref and refBeta from GPU to CPU

            while(ref>=0){

               // if(asso_idx[ref] >=0) continue; //
               // if(temp_betas[ref] < min_beta)continue;
                 //probably better to copy here instead of accessing n_vert times in GPU mem
                ncond++;

                check_and_collect(
                        ref,
                        ref_beta,
                        d_ccoords,
                        d_betas,
                        d_dist,
                        d_tosum,
                        asso_idx,
                        temp_betas,
                        temp_tosum,
                        summed,
                        n_vert,
                        n_ccoords,
                        n_sumf,
                        start_vertex,
                        end_vertex,
                        radius,
                        min_beta,
                        soft,
                        sum);

                get_max_beta(temp_betas,asso_idx,is_cpoint,&ref,n_vert,start_vertex,end_vertex,min_beta);
                //copy ref and refBeta from GPU to CPU
                ref_beta = d_betas[ref];
            }

            n_condensates[j_rs] = ncond;


        }

    }
};

template<typename Device>
class BuildCondensatesOp : public OpKernel {
public:
    explicit BuildCondensatesOp(OpKernelConstruction *context) : OpKernel(context) {

        OP_REQUIRES_OK(context,
                context->GetAttr("min_beta", &min_beta_));

        OP_REQUIRES_OK(context,
                context->GetAttr("radius", &radiussq_));
        radiussq_*=radiussq_;

        OP_REQUIRES_OK(context,
                context->GetAttr("soft", &soft_));

        OP_REQUIRES_OK(context,
                context->GetAttr("sum", &sum_));
    }


    void Compute(OpKernelContext *context) override {

        /*
         *
         * .Attr("radius: float")

REGISTER_OP("BuildCondensates")
    .Attr("radius: float")
    .Attr("min_beta: float")
    .Attr("soft: bool")

    .Input("ccoords: float32")
    .Input("betas: float32")
    .Input("row_splits: int32")

    .Output("asso_idx: int32")
    .Output("is_cpoint: int32");



         */

        const Tensor &t_ccoords = context->input(0);
        const Tensor &t_betas = context->input(1);
        const Tensor &t_dist = context->input(2);
        const Tensor &t_tosum = context->input(3);
        const Tensor &t_row_splits = context->input(4);

        const int n_vert = t_ccoords.dim_size(0);
        const int n_ccoords = t_ccoords.dim_size(1);
        const int n_rs = t_row_splits.dim_size(0);
        int n_sumf = t_tosum.dim_size(1);
        if(!sum_)
            n_sumf=0;
        const int n_f_vert = t_tosum.dim_size(0);

        if(sum_){
            OP_REQUIRES(context, n_f_vert == n_vert,
                    errors::InvalidArgument("BuildCondensatesOp expects first dimensions of tosum to match number of vertices if sum is activated."));
        }

        TensorShape outputShape_idx;
        outputShape_idx.AddDim(n_vert);

        Tensor *t_asso_idx = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, outputShape_idx, &t_asso_idx));

        Tensor *t_is_cpoint = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, outputShape_idx, &t_is_cpoint));

        TensorShape rsmone;
        rsmone.AddDim(n_rs-1);
        Tensor *t_n_condensates = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(2, rsmone, &t_n_condensates));

        Tensor *t_summed = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(3, t_tosum.shape(), &t_summed));

        Tensor t_temp_betas;
        OP_REQUIRES_OK(context, context->allocate_temp( DT_FLOAT ,outputShape_idx,&t_temp_betas));

        Tensor t_temp_to_be_summed;
        OP_REQUIRES_OK(context, context->allocate_temp( DT_FLOAT ,t_tosum.shape(),&t_temp_to_be_summed));

        BuildCondensatesOpFunctor<Device, int>()
                (
                context->eigen_device<Device>(),                                //
                //                                                                                                     /
                t_ccoords.flat<float>().data(),                     //   const float *d_ccoords,
                t_betas.flat<float>().data(),                       //   const float *d_betas,
                t_dist.flat<float>().data(),
                t_tosum.flat<float>().data(),
                t_row_splits.flat<int>().data(),                    //   const int *row_splits,

                t_asso_idx->flat<int>().data(), //                  int *asso_idx,
                t_is_cpoint->flat<int>().data(),//                  int *is_cpoint,
                t_temp_betas.flat<float>().data(),
                t_n_condensates->flat<int>().data(),
                t_temp_to_be_summed.flat<float>().data(),
                t_summed->flat<float>().data(),

                n_vert,
                n_ccoords,
                n_sumf,
                n_rs,

                radiussq_,
                min_beta_ ,
                soft_,
                sum_
        );



    }

private:
    float min_beta_;
    float radiussq_;
    bool soft_,sum_;
};

REGISTER_KERNEL_BUILDER(Name("BuildCondensates").Device(DEVICE_CPU), BuildCondensatesOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct BuildCondensatesOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("BuildCondensates").Device(DEVICE_GPU), BuildCondensatesOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}//functor
}//tensorflow
