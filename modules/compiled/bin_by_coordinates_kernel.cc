
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif 

#include "tensorflow/core/framework/op_kernel.h"
#include "helpers.h"
#include "bin_by_coordinates_kernel.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {


template<typename dummy>
struct BinByCoordinatesNbinsHelperOpFunctor<CPUDevice, dummy> { //just because access needs to be different for GPU and CPU
    void operator()(
            const CPUDevice &d,
            const int * n_bins,
            int * out_tot_bins,
            int n_nbins,
            int nrs){
    int n=1;
    printf("n bins:");
    for(int i=0;i<n_nbins;i++){
        n*=n_bins[i];
        printf(" %d ,", n_bins[i]);
    }
    printf("\n");

    *out_tot_bins=n*(nrs-1);
    }
};


static void set_defaults(
        int * d_n_per_bin,
        const int n_total_bins
       ){
    for(int i=0;i<n_total_bins;i++)
        d_n_per_bin[i]=0;
}

static void calc(
        const float * d_coords,
        const int * d_rs,

        const float * d_binswidth, //singleton
        const int * n_bins,//singleton

        int * d_assigned_bin,
        int * d_flat_assigned_bin,
        int * d_n_per_bin,

        const int n_vert,
        const int n_coords,
        const int n_rs,
        const int n_total_bins,
        const bool calc_n_per_bin
){

    for(int iv=0; iv<n_vert; iv++){

        ///same for cu

        int mul = 1;
        int idx = 0;

        for (int ic = n_coords-1; ic > -1; ic--) {

            int cidx = d_coords[I2D(iv,ic,n_coords)] / d_binswidth[0];

            if(cidx < 0 || cidx >= n_bins[ic]){
                printf("index %d of coordinate %d exceeds n bins %d or below 0, coord %e\n",cidx,ic,n_bins[ic],d_coords[I2D(iv,ic,n_coords)]);
                cidx = 0; //stable, but will create bogus later
            }
            d_assigned_bin[I2D(iv,ic+1,n_coords+1)]=cidx;

            idx += cidx * mul;
            mul *= n_bins[ic];

        }

        //get row split index last
        int rsidx=0;
        for(int irs=1 ; irs < n_rs ; irs++){
            if(d_rs[irs] > iv){
                break;
            }
            rsidx++;
        }

        idx += rsidx * mul;

        if(idx>=n_total_bins){
            printf("global index larger than total bins\n");//DEBUG if you see this you're screwed
            continue;
        }

        d_assigned_bin[I2D(iv,0,n_coords+1)]=rsidx; //now this is c-style ordering with [rs, c_N, c_N-1, ..., c_0]
        d_flat_assigned_bin[iv]=idx;


        if(calc_n_per_bin){
            //atomic in parallel!
            d_n_per_bin[idx] += 1;

        }
        //end same for cu

    }//iv loop
}


template<typename dummy>
struct BinByCoordinatesOpFunctor<CPUDevice, dummy> { //just because access needs to be different for GPU and CPU
    void operator()(
            const CPUDevice &d,

            const float * d_coords,
            const int * d_rs,

            const float * d_binswidth, //singleton
            const int * n_bins,//singleton

            int * d_assigned_bin,
            int * d_flat_assigned_bin,
            int * d_n_per_bin,

            const int n_vert,
            const int n_coords,
            const int n_rs,
            const int n_total_bins,
            const bool calc_n_per_bin){

        set_defaults(d_n_per_bin,n_total_bins);

        calc(d_coords, d_rs, d_binswidth,n_bins,

                d_assigned_bin,
                d_flat_assigned_bin,
                d_n_per_bin,

                n_vert,
                n_coords,
                n_rs,
                n_total_bins,
                calc_n_per_bin);
        /////

        //this will be parallelisation dimension

    }
};


template<typename Device>
class BinByCoordinatesOp : public OpKernel {
public:
    explicit BinByCoordinatesOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("calc_n_per_bin", &calc_n_per_bin_));
    }


    void Compute(OpKernelContext *context) override {

        /*
         * .Attr("calc_n_per_bin: bool")
    .Input("coordinates: float")
    .Input("row_splits: int32")
    .Input("bin_width: float")
    .Input("nbins: int32")
    .Output("bin_assignment: int32") //non-flat
    .Output("flat_bin_assignment: int32") //flat
    .Output("nperbin: int32"); //corresponding to the flat assignment
         *
         */

        const Tensor &t_coords = context->input(0);
        const Tensor &t_rs = context->input(1);
        const Tensor &t_binwdith = context->input(2);
        const Tensor &t_nbins = context->input(3);

        const int n_vert = t_coords.dim_size(0);
        const int n_coords = t_coords.dim_size(1);
        const int n_rs = t_rs.dim_size(0);

        ///size checks

        OP_REQUIRES(context, 1 == t_binwdith.dim_size(0),
                    errors::InvalidArgument("BinByCoordinatesOp expects singleton (dim: 1) for bin width."));
        OP_REQUIRES(context, n_coords == t_nbins.dim_size(0),
                    errors::InvalidArgument("BinByCoordinatesOp expects coordinate dimension for number of bins."));

        const int n_nbins = n_coords;//just for clarity
        
        int n_tot_bins=0;
        BinByCoordinatesNbinsHelperOpFunctor<Device, int>() (
                context->eigen_device<Device>(),
                t_nbins.flat<int>().data(),
                &n_tot_bins,
                n_nbins,
                n_rs );


        Tensor *t_assigned_bin = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0,{n_vert,n_coords+1},&t_assigned_bin));

        Tensor *t_flat_assigned_bin = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1,{n_vert},&t_flat_assigned_bin));

        Tensor *t_nper_bin = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(2,{n_tot_bins},&t_nper_bin));

        BinByCoordinatesOpFunctor<Device, int>() (
                context->eigen_device<Device>(),

                t_coords.flat<float>().data(),
                t_rs.flat<int>().data(),

                t_binwdith.flat<float>().data(),
                t_nbins.flat<int>().data(),

                t_assigned_bin->flat<int>().data() ,
                t_flat_assigned_bin->flat<int>().data() ,
                t_nper_bin->flat<int>().data() ,

                n_vert,
                n_coords,
                n_rs,
                n_tot_bins,
                calc_n_per_bin_
        );

    }
private:
    bool calc_n_per_bin_;

};


REGISTER_KERNEL_BUILDER(Name("BinByCoordinates").Device(DEVICE_CPU), BinByCoordinatesOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct BinByCoordinatesOpFunctor<GPUDevice, int>;
extern template struct BinByCoordinatesNbinsHelperOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("BinByCoordinates").Device(DEVICE_GPU), BinByCoordinatesOp<GPUDevice>);
#endif  


}//functor
}//tensorflow
