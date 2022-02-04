/*
 * only use bare c++ so that it compiled with nvcc
 * assert might need to be removed/exchanged
 * template only, some dirty hack.
 * for complaints: jkiesele
 */

//to be removed
#include <iostream>
#include "cuda_runtime.h"

struct binstepper{
    int step_no;
    int total_cap;
    int cube_cap;
    int d;
    int n_dims;
};

//debug

static void cout_array(const int* arr, int n, int offset=0){
    std::cout << "[";
    for(int i=0;i<n;i++)
        std::cout << arr[i]+offset << ", ";
    std::cout <<"]";
}

static void cout_s(const binstepper& s){
    std::cout << "\n(step_no " << s.step_no
              << "\n(total_cap "<< s.total_cap
              << "\n(cube_cap " << s.cube_cap
              << "\n(d " << s.d
              << "\n(n_dims " << s.n_dims
              << std::endl;
}

//end debug


__host__
__device__
static void s_init(binstepper& stpr, const int* total_bins,  int n_dims){
    stpr.n_dims=n_dims;
    stpr.total_cap=1;
    for(int i=0;i<stpr.n_dims;i++)
        stpr.total_cap *= total_bins[i];
}



__host__
__device__
static void s_set_d(binstepper& stpr,  int d){
    stpr.d = d;
    stpr.cube_cap = 2*d+1;
    for(int i=1;i<stpr.n_dims;i++)
        stpr.cube_cap*=2*d+1;
    stpr.step_no=0;
}


__host__
__device__
static int s_step(binstepper& stpr, int* cubeidxs, const int* glidxs, const int* total_bins){
    if(stpr.step_no==stpr.cube_cap)
        return -1;

    //read to inner cube
    int mul=stpr.cube_cap;
    int cidx = stpr.step_no;
    for(int i=0;i<stpr.n_dims;i++){
        mul /= 2*stpr.d+1;
        cubeidxs[i] = cidx / mul;
        cidx -= cubeidxs[i] * mul;
    }

    stpr.step_no++;

    //check if it is a step on surface
    bool on_surface=false;
    for(int i=0;i<stpr.n_dims;i++){
        if(abs(cubeidxs[i]-stpr.d)==stpr.d)
            on_surface = true; //any abs is d
    }

    for(int i=0;i<stpr.n_dims;i++){
        if(abs(cubeidxs[i]-stpr.d)>stpr.d)
            on_surface = false; //any abs is larger d

    }
    if(!on_surface)
        return s_step(stpr,cubeidxs,glidxs,total_bins);

    //valid local step in cube: ok

    //apply to global index and check
    mul = 1;
    int glidx=0;
    for(int i = stpr.n_dims-1; i != -1; i--){//go backwards and make flat index in situ
        int iidx = cubeidxs[i] - stpr.d + glidxs[i];
        if(iidx<0 || iidx>=total_bins[i])
            return s_step(stpr,cubeidxs,glidxs,total_bins);
        glidx+=iidx*mul;
        mul*=total_bins[i];
    }
    if(glidx>=stpr.total_cap || glidx<0)
        return s_step(stpr,cubeidxs,glidxs,total_bins);

    return glidx;
}











