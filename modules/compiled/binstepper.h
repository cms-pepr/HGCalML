/*
 * only use bare c++ so that it compiled with nvcc
 * assert might need to be removed/exchanged
 * template only
 * for complaints: jkiesele
 */

//to be removed
#include "cuda_runtime.h"

template<int N_dims>
class binstepper{
public:

    __host__
    __device__
    binstepper(const int * bins_per_dim, const int* glidxs){
        total_cap_=1;
        for(int i=0;i<N_dims;i++){
            total_cap_ *= bins_per_dim[i];
            total_bins_[i] = bins_per_dim[i];//just a copy to local memory
            glidxs_[i] = glidxs[i];
        }
        set_d(0);
    }


    __host__
    __device__
    void set_d(int distance){
        d_ = distance;
        cube_cap_ = 2*d_+1;
        for(int i=1;i<N_dims;i++)
            cube_cap_*=2*d_+1;
        step_no_=0;
    }

    __host__
    __device__
    int step(){
        if(step_no_==cube_cap_)
            return -1;

        //read to inner cube
        int mul=cube_cap_;
        int cidx = step_no_;
        for(int i=0;i<N_dims;i++){
            mul /= 2*d_+1;
            cubeidxs_[i] = cidx / mul;
            cidx -= cubeidxs_[i] * mul;
        }

        step_no_++;

        //check if it is a step on surface
        bool on_surface=false;
        for(int i=0;i<N_dims;i++){
            if(abs(cubeidxs_[i]-d_)==d_)
                on_surface = true; //any abs is d
        }

        for(int i=0;i<N_dims;i++){
            if(abs(cubeidxs_[i]-d_)>d_)
                on_surface = false; //any abs is larger d

        }
        if(!on_surface)
            return step();

        //valid local step in cube: ok

        //apply to global index and check
        mul = 1;
        int glidx=0;
        for(int i = N_dims-1; i != -1; i--){//go backwards and make flat index in situ
            int iidx = cubeidxs_[i] - d_ + glidxs_[i];
            if(iidx<0 || iidx>=total_bins_[i])
                return step();
            glidx+=iidx*mul;
            mul*=total_bins_[i];
        }
        if(glidx>=total_cap_ || glidx<0)
            return step();

        return glidx;
    }

private:

    int step_no_;
    int total_cap_;
    int cube_cap_;
    int d_;
    int cubeidxs_[N_dims]; //temp
    int glidxs_[N_dims];
    int total_bins_[N_dims];
};


//end debug




//debug

#include <iostream>

static void cout_array(const int* arr, int n, int offset=0){
    std::cout << "[";
    for(int i=0;i<n;i++)
        std::cout << arr[i]+offset << ", ";
    std::cout <<"]";
}
template<int N_dims>
static void cout_s(const binstepper<N_dims>& s){
    std::cout << "\n(step_no " << s.step_no_
              << "\n(total_cap "<< s.total_cap_
              << "\n(cube_cap " << s.cube_cap_
              << "\n(d " << s.d_
              << "\n(N_dims " << N_dims
              << std::endl;
}








