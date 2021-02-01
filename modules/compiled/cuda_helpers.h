/*
 * cuda_helpers.h
 *
 *  Created on: 12 May 2020
 *      Author: jkiesele
 */

#ifndef HGCALML_MODULES_COMPILED_CUDA_HELPERS_H_
#define HGCALML_MODULES_COMPILED_CUDA_HELPERS_H_



#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

template <class T>
//from dlib!
// http://dlib.net/dlib/dnn/cuda_utils.h.html
// boost licence, check for other sources at some point or install lib.
// right now, for testing here
class _grid_stride_range
{
    /*!
                WHAT THIS OBJECT REPRESENTS
                    This is a tool for making a for loop that loops over an entire block of
                    memory inside a kernel, but doing so in a way that parallelizes
                    appropriately across all the threads in a kernel launch.  For example,
                    the following kernel would add the vector a to the vector b and store
                    the output in out (assuming all vectors are of dimension n):
                        __global__ void add_arrays(
                            const float* a,
                            const float* b,
                            float* out,
                            T n
                        )
                        {
                            for (auto i : _grid_stride_range(0, n))
                            {
                                out[i] = a[i]+b[i];
                            }
                        }
            !*/

public:
    __device__ _grid_stride_range(
            T ibegin_,
            T iend_
    ) :
    ibegin(ibegin_),
    iend(iend_)
    {}

    class iterator
    {
    public:
        __device__ iterator() {}
        __device__ iterator(T pos_) : pos(pos_) {}

        __device__ T operator*() const
        {
            return pos;
        }

        __device__ iterator& operator++()
                                                {
            pos += gridDim.x * blockDim.x;
            return *this;
                                                }

        __device__ bool operator!=(const iterator& item) const
                                                { return pos < item.pos; }

    private:
        T pos;
    };

    __device__ iterator begin() const
    {
        return iterator(ibegin+blockDim.x * blockIdx.x + threadIdx.x);
    }
    __device__ iterator end() const
    {
        return iterator(iend);
    }
private:

    T ibegin;
    T iend;
};

template <class T>
class _grid_stride_range_y
{
    /*!
                WHAT THIS OBJECT REPRESENTS
                    This object is just like grid_stride_range except that it looks at
                    CUDA's y thread index (e.g. threadIdx.y) instead of the x index.
                    Therefore, if you launch a cuda kernel with a statement like:
                        dim3 blocks(1,10);
                        dim3 threads(32,32);  // You need to have x and y not equal to 1 to get parallelism over both loops.
                        add_arrays<<<blocks,threads>>>(a,b,out,nr,nc);
                    You can perform a nested 2D parallel for loop rather than doing just a
                    1D for loop.

                    So the code in the kernel would look like this if you wanted to add two
                    2D matrices:
                        __global__ void add_arrays(
                            const float* a,
                            const float* b,
                            float* out,
                            T nr,
                            T nc
                        )
                        {
                            for (auto r : _grid_stride_range_y(0, nr))
                            {
                                for (auto c : grid_stride_range(0, nc))
                                {
                                    auto i = r*nc+c;
                                    out[i] = a[i]+b[i];
                                }
                            }
                        }
            !*/

public:
    __device__ _grid_stride_range_y(
            T ibegin_,
            T iend_
    ) :
    ibegin(ibegin_),
    iend(iend_)
    {}

    class iterator
    {
    public:
        __device__ iterator() {}
        __device__ iterator(T pos_) : pos(pos_) {}

        __device__ T operator*() const
        {
            return pos;
        }

        __device__ iterator& operator++()
                        {
            pos += gridDim.y * blockDim.y;
            return *this;
                        }

        __device__ bool operator!=(const iterator& item) const
                        { return pos < item.pos; }

    private:
        T pos;
    };

    __device__ iterator begin() const
    {
        return iterator(ibegin+blockDim.y * blockIdx.y + threadIdx.y);
    }
    __device__ iterator end() const
    {
        return iterator(iend);
    }
private:

    T ibegin;
    T iend;
};

template<class T>
class _grid_and_block {
    /*
     * This can include some hardware resource (SM) checking etc in a future version
     */
public:

    _grid_and_block(size_t nx,size_t xb,size_t ny=1,size_t yb=1,size_t nz=1,size_t zb=1){
        calcGB(nx, xb, gx_, bx_);
        calcGB(ny, yb, gy_, by_);
        calcGB(nz, zb, gz_, bz_);
    }

    dim3 grid()const{
        return dim3(gx_,gy_,gz_);
    }
    dim3 block()const{
        return dim3(bx_,by_,bz_);
    }

private:
    void calcGB(size_t n,size_t b, T& grid, T& block){
        grid = n/b;
        block=b;
        if(n%b)
            grid+=1;
        if(!grid)
            grid=1;
    }
    _grid_and_block():gx_(0),gy_(0),gz_(0),bx_(0),by_(0),bz_(0){}
    T gx_,gy_,gz_; //replace by T
    T bx_,by_,bz_;
};

template<class T>
class _lock_mutex{
public:

    _lock_mutex(){
        T state=0;
        cudaMalloc((void**)&mutex_, sizeof(T));
        cudaMemcpy(mutex_, &state, sizeof(T), cudaMemcpyHostToDevice);
    }
    ~_lock_mutex(){
        cudaFree(mutex_);
    }

    T* mutex(){return mutex_;}

private:
    T * mutex_;
};
template<class T>
class _lock{
public:
    __device__
    _lock( T * mutex):mutex_(mutex){
    }
    __device__
    void lock(){
        while(atomicCAS(mutex_,0,1) != 0){};
    }
    __device__
    void unlock(){
        atomicExch(mutex_,0);
    }
//private:
    T * mutex_;
private:
    __device__
        _lock(){}
};

//template<class T>
//class _lock{
//
//public:
//    __device__ _lock( void )
//    {
//        mutex = new T;
//        (*mutex) = 0;
//    }
//    __device__ ~_lock( void )
//    {
//        delete mutex;
//    }
//
//    __device__ void lock( void )
//    {
//        while( atomicCAS( mutex, 0, 1 ) != 0 );
//    }
//    __device__ void unlock( void )
//    {
//        atomicExch( mutex, 0 );
//    }
//private:
//    T *mutex;
//};
//
//



typedef _grid_stride_range<size_t> grid_stride_range;
typedef _grid_stride_range_y<size_t> grid_stride_range_y;
typedef _grid_and_block<size_t> grid_and_block;
typedef _lock<int> lock;
typedef _lock_mutex<int> lock_mutex;

#endif /* HGCALML_MODULES_COMPILED_CUDA_HELPERS_H_ */
