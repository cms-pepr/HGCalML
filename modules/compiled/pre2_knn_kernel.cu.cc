//#define GOOGLE_CUDA 1


#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "pre2_knn_kernel.h"
#include "helpers.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cuda_helpers.h"
#include <iostream>

namespace tensorflow {
namespace functor {

namespace cpu{

void create_bin_neighbours_list(
        const int n_bins_x,
        const int n_bins_y, 
        int* bin_neighbours)
{
    int tmp_arr[3] = {0,-1,1};

    for (size_t j = 0; j<n_bins_y; j+=1){
        for (size_t i = 0; i<n_bins_x; i+=1){
            int index = (i + n_bins_x*j)*9;
            int counter = 0;
            for (size_t ii = 0; ii<3; ii++){
                int i_s = tmp_arr[ii];
                for (size_t jj = 0; jj<3; jj++){
                    int j_s = tmp_arr[jj];
                    int tmp_index = i+i_s + n_bins_x*(j+j_s);
                    if ((tmp_index<0) || (tmp_index>=n_bins_x*n_bins_y) || ((i+i_s)<0) || ((j+j_s)<0) || ((i+i_s)>=n_bins_x) || ((j+j_s)>=n_bins_y)){
                        bin_neighbours[index+counter] = -1;
                    }
                    else{
                        bin_neighbours[index+counter] = tmp_index;
                    }
                    counter+=1;
                }
            }
        }
    }
}

void calculate_n_vtx_per_bin_cumulative(
        size_t n_bins,
        int* n_vtx_per_bin,
        int* n_vtx_per_bin_cumulative)
{
    // output: n_vtx_per_bin_cumulative - positions from which to start placing vertices which belongs to different bins
    // e.g. if n_vtx_per_bin = {5,3,2,4} --> n_vtx_per_bin_cumulative = {0,5,8,10}
    n_vtx_per_bin_cumulative[0] = 0;
    for(size_t i = 1 ; i < n_bins+1; i += 1){
        n_vtx_per_bin_cumulative[i] = n_vtx_per_bin_cumulative[i-1] + n_vtx_per_bin[i-1];
    }
}

template <typename T>
void print_array(
        const T *in_arr,
        const size_t start,
        const size_t end,
        bool convert_to_int = false
){
    for(size_t i = start ; i < end ; i += 1){
        float tmp_val = in_arr[i];
        if (convert_to_int == true)
            printf("i: %d;\t%d\n", i, (int)tmp_val);
        else
            printf("i: %d;\t%f\n", i, tmp_val);
    }
}

template <typename T>
void print_2d_matrix(
        const T *in_arr,
        const size_t stride,
        const size_t start,
        const size_t end,
        bool convert_to_int = false
){
    for(size_t i = start ; i < end ; i += 1){
        float tmp_val = in_arr[i];
        if (i % stride == 0){
            printf("\n");
            printf("i: %d: ", (int)(i/stride));
        }
        if (convert_to_int == true)
            printf("\t%d", (int)tmp_val);
        else
            printf("\t%f", tmp_val);
    }
    printf("\n");
}

}// cpu namespace

namespace gpu{

__global__
void calculate_n_vtx_per_bin_cumulative(
        const int n_bins,
        int* n_vtx_per_bin,
        int* n_vtx_per_bin_cumulative)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // if (index==0)
    //     n_vtx_per_bin_cumulative[0] = 0;
    // else{
    //     for(size_t i = index ; i < n_bins+1; i += stride){
    //         n_vtx_per_bin_cumulative[i] = n_vtx_per_bin_cumulative[i-1] + n_vtx_per_bin[i-1];
    //     }
    // }
    
    for(size_t i = index ; i < n_bins+1; i += stride){
        n_vtx_per_bin_cumulative[i] = n_vtx_per_bin_cumulative[i-1] + n_vtx_per_bin[i-1];
    }
}

__device__
float calculateDistance(size_t i_v, size_t j_v, const float * d_coords, size_t n_coords){
    float distsq=0;
    if(i_v == j_v)
        return 0;
    for(size_t i=0;i<n_coords;i++){
        float dist = d_coords[I2D(i_v,i,n_coords)] - d_coords[I2D(j_v,i,n_coords)];
        distsq += dist*dist;
    }
    return distsq;
}

__global__
void find_min_max_2d_init(
    const float *d_coords,
    float* coord_min,
    float* coord_max
    )
{
        coord_min[0] = d_coords[0];
        coord_max[0] = d_coords[0];
        coord_min[1] = d_coords[1];
        coord_max[1] = d_coords[1];
}
__global__
void find_min_max_2d(
    const float *d_coords,
    const int n_coords,
    const int start_vert,
    const int end_vert,
    float* coord_min,
    float* coord_max
    )
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("index: ",index);
    float coord;
    int dim = index%2;
    for(size_t i=start_vert; i<end_vert; i+= 1){
        coord = d_coords[i*n_coords+dim];
        if(coord_min[dim] > coord)
            coord_min[dim] = coord;
        if(coord_max[dim] < coord)
            coord_max[dim] = coord;
    }
}

__device__
int searchLargestDistance(int i_v, float* neigh_dist, int K, float& maxdist){
    maxdist=0;
    int maxidx=0;
    for(size_t n=0;n<K;n++){
        float distsq = neigh_dist[I2D(i_v,n,K)];
        if(distsq > maxdist){
            maxdist = distsq;
            maxidx = n;
        }
    }
    return maxidx;
}

__device__
float calculate2dDistanceToThePoint(float *pointCoord, size_t i_v, const float* d_coords, size_t n_coords){
    float distsq=0;
    for(size_t i=0;i<2;i++){
        float dist = d_coords[I2D(i_v,i,n_coords)] - pointCoord[i];
        distsq += dist*dist;
    }
    return distsq;
}

__device__
int getGlobalIdx_2D_2D(){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y)
         + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}



template <typename T>
__global__
void set_defaults(
        T *in_arr,
        const size_t arr_size,
        const T def_val
){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(size_t i = index ; i < arr_size ; i += stride){
        in_arr[i] = def_val;
    }
}

template <typename T>
__global__
void insert_into_array(
        T *arr_to_copy,
        T *arr_to_insert,
        const size_t start_pos,
        const size_t end_pos
){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(size_t i = index ; i < (end_pos-start_pos) ; i += stride){
        arr_to_insert[start_pos+i] = arr_to_copy[i];
    }
}

template <typename T>
__global__
void insert_value_into_array(
        const T val_to_copy,
        T *arr_to_insert,
        const size_t pos
){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(size_t i = index ; i < 1 ; i += stride){
        arr_to_insert[pos] = val_to_copy;
    }
}


template <typename T>__global__
void print_array(
        const T *in_arr,
        const size_t start,
        const size_t end
){
    int index = blockIdx.x * blockDim.x + threadIdx.x + start;
    int stride = blockDim.x * gridDim.x;
    for(size_t i = index ; i < end ; i += stride){
        printf("i: %d;\t%f\n", i, in_arr[i]);
    }
}

template <typename T>__global__
void print_2d_matrix(
        const T *in_arr,
        const size_t stride_in,
        const size_t start,
        const size_t end,
        bool convert_to_int = false
){
    int index = blockIdx.x * blockDim.x + threadIdx.x + start;
    int stride = blockDim.x * gridDim.x;
    for(size_t i = index ; i < end ; i += stride){
        float tmp_val = in_arr[i];
        if (i % stride_in == 0){
            printf("\n");
            printf("i: %d:", i/stride_in);
        }
        if (convert_to_int == true)
            printf("\t%d", (int)tmp_val);
        else
            printf("\t%f", tmp_val);
    }
    printf("\n");
}

__global__
void get_bin_coords(float* coord_min, float* coord_max, float *coords_2d_bins, const size_t n_bins_x, const size_t n_bins_y){

    const size_t iBinX =  blockIdx.x * blockDim.x + threadIdx.x;
    if(iBinX >= n_bins_x)
        return;
    const size_t iBinY =  blockIdx.y * blockDim.y + threadIdx.y;
    if(iBinY >= n_bins_y)
        return;

    // define phase-space bin edges
    size_t bin_index = I2D(iBinY, iBinX, n_bins_x);
    coords_2d_bins[4*bin_index] = coord_min[0] + iBinX*(coord_max[0] - coord_min[0])/n_bins_x; // x1
    coords_2d_bins[4*bin_index+1] = coord_min[1] + iBinY*(coord_max[1] - coord_min[1])/n_bins_y; // y1
    coords_2d_bins[4*bin_index+2] = coord_min[0] + (iBinX+1)*(coord_max[0] - coord_min[0])/n_bins_x; // x2
    coords_2d_bins[4*bin_index+3] = coord_min[1] + (iBinY+1)*(coord_max[1] - coord_min[1])/n_bins_y; // y2
}

// calculate min and max of X and Y coordinates among all vertices
// make bins with widths: (x_max-x_min)/n_bins and (y_max-y_min)/n_bins
__global__
void constructPhaseSpaceBins(const float *d_coords, size_t n_coords, size_t start_vert,
                             size_t end_vert, int* n_bins,
                             float* coord_min, float* coord_max, int *n_vtx_per_bin,
                             int* bin_idx){

    // define which vertices belong to which bin
    int index = blockIdx.x * blockDim.x + threadIdx.x + start_vert;
    int stride = blockDim.x * gridDim.x;
    for(size_t i_v = index; i_v < end_vert; i_v += stride){
        
        size_t iDim = 0;
        float coord = d_coords[I2D(i_v,iDim,n_coords)];
        size_t indx_1 = (size_t)((coord - coord_min[iDim])/((coord_max[iDim]-coord_min[iDim])/n_bins[iDim]));
        iDim = 1;
        coord = d_coords[I2D(i_v,iDim,n_coords)];
        size_t indx_2 = (size_t)((coord - coord_min[iDim])/((coord_max[iDim]-coord_min[iDim])/n_bins[iDim]));

        size_t bin_index = I2D(indx_2, indx_1, n_bins[0]);
        atomicAdd(&n_vtx_per_bin[bin_index], 1); // to avoid race condition
        bin_idx[i_v] = bin_index;
    }
    // if (index==end_vert-2){
    //     printf("i: %d: %d\n",0, n_vtx_per_bin[0]);
    //     printf("i: %d: %d\n",1, n_vtx_per_bin[1]);
    // }

}


__global__
void prepare_translation_matrix(const size_t start_vert, const size_t end_vert, const int* n_vtx_per_bin_cumulative, const int* bin_idx, int* zero_counters, int* forward_translation_matrix){
    
    size_t index = blockIdx.x * blockDim.x + threadIdx.x + start_vert;
    size_t stride = blockDim.x * gridDim.x;
    for(size_t i_v = index; i_v < end_vert; i_v += stride){
        int bin_index = bin_idx[i_v];
        int index_to_fill = atomicAdd(&zero_counters[bin_index],1) + n_vtx_per_bin_cumulative[bin_index];
        forward_translation_matrix[index_to_fill] = i_v;
    }
}

__global__
void prepare_backward_translation_matrix(const size_t start_vert, const size_t end_vert, const int* forward_translation_matrix, int* backward_translation_matrix){
    
    size_t index = blockIdx.x * blockDim.x + threadIdx.x + start_vert;
    size_t stride = blockDim.x * gridDim.x;
    for(size_t i_v = index; i_v < end_vert; i_v += stride){
        backward_translation_matrix[forward_translation_matrix[i_v]] = i_v;
    }
}

template <typename T>
__global__
void translate_2d_matrix(size_t matrix_height, size_t matrix_width, const int* translation_matrix, const T *in_mattrix, T* out_matrix){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for(size_t i_counter = index; i_counter < matrix_height; i_counter += stride){
        size_t real_index = translation_matrix[i_counter];
        for(size_t i_column = 0; i_column < matrix_width; i_column += 1)
            out_matrix[matrix_width*i_counter+i_column] = in_mattrix[matrix_width*real_index+i_column];
    }
}

__global__
void translate_content_of_2d_matrix(size_t n_element_in_matrix, const int* translation_matrix, const int *in_mattrix, int* out_matrix){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for(size_t i_counter = index; i_counter < n_element_in_matrix; i_counter += stride){
        int real_index = in_mattrix[i_counter];
        int translated_index = -1;
        if (real_index>=0)
            translated_index = translation_matrix[real_index];
        out_matrix[i_counter] = translated_index;
    }
}



__global__
void init_incremented_values(const int start_vert, const int end_vert, int* in_arr){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x + start_vert;
    size_t stride = blockDim.x * gridDim.x;
    for(size_t i_v = index; i_v < end_vert-start_vert; i_v += stride){
        in_arr[i_v] = start_vert + i_v;
    }
}

} // gpu namespace

typedef Eigen::GpuDevice GPUDevice;


template <typename dummy>
struct Pre2KnnOpFunctor<GPUDevice, dummy> {
    void operator()(const GPUDevice& d,

            const float *d_coords, // accessible only on GPU!!!
            float *d_coords_sorted,
            // int *n_vtx_per_bin,
            // int *n_vtx_per_bin_cumulative,
            int *auxaliry_knn_arrays,
            const int V, // # of vertices
            const int n_coords,
            const int n_bins_x,
            const int n_bins_y) {
        
        // ******************************************
        // STEP 1: memory allocation and set defaults
        // ******************************************
        // printf("\n\n*** STEP 1: memory allocation and set defaults ***\n");
        int blockSize = 256;
        int numBlocks_V = (V + blockSize - 1) / blockSize;
        int numBlocks_n_bins = (n_bins_x*n_bins_y + blockSize - 1) / blockSize;
        const size_t start_vert = 0;
        const size_t end_vert = V;

        float* coords_2d_bins; 
        cudaMallocManaged(&coords_2d_bins, 4*n_bins_x*n_bins_y*sizeof(float));

        int *bin_idx;
        cudaMallocManaged(&bin_idx, V*sizeof(int));

        int *help_arr;
        cudaMallocManaged(&help_arr, V*sizeof(int));

        int *bin_idx_sorted ;
        cudaMallocManaged(&bin_idx_sorted , V*sizeof(int));

        int *vtx_idx_translation_matrix;
        cudaMallocManaged(&vtx_idx_translation_matrix, V*sizeof(int));

        // int *backward_vtx_idx_translation_matrix;
        // cudaMallocManaged(&backward_vtx_idx_translation_matrix, V*sizeof(int));

        int *n_bins;
        cudaMallocManaged(&n_bins, 2*sizeof(int));
        n_bins[0] = n_bins_x;
        n_bins[1] = n_bins_y;
        
        float* coord_min;
        float* coord_max;
        cudaMallocManaged(&coord_max, 2*sizeof(float));
        cudaMallocManaged(&coord_min, 2*sizeof(float));

        int* zero_counters;
        cudaMallocManaged(&zero_counters, n_bins_x*n_bins_y*sizeof(int));

        int* bin_neighbours;
        cudaMallocManaged(&bin_neighbours, 9*n_bins_x*n_bins_y*sizeof(int));

        int* n_vtx_per_bin;
        cudaMallocManaged(&n_vtx_per_bin, n_bins_x*n_bins_y*sizeof(int));

        int* n_vtx_per_bin_cumulative;
        cudaMallocManaged(&n_vtx_per_bin_cumulative, (n_bins_x*n_bins_y+1)*sizeof(int));

        // ********************************
        // STEP 2: fill in auxiliary arrays
        // ********************************
        // printf("\n\n*** STEP 2: fill in auxiliary arrays ***\n");

        // set default values
        gpu::set_defaults<<<numBlocks_n_bins ,blockSize>>>(n_vtx_per_bin, n_bins_x*n_bins_y, 0);
        gpu::set_defaults<<<numBlocks_n_bins ,blockSize>>>(zero_counters, n_bins_x*n_bins_y, 0);
        gpu::set_defaults<<<numBlocks_n_bins ,blockSize>>>(n_vtx_per_bin_cumulative, n_bins_x*n_bins_y+1, 0);
        gpu::set_defaults<<<numBlocks_n_bins ,blockSize>>>(auxaliry_knn_arrays,1+1+(n_bins_x*n_bins_y+1)+(9*n_bins_x*n_bins_y)+V, 0);
        cudaDeviceSynchronize();
        // printf("\n*** Check point #1 ***\n");

        // find min/max of 1st and 2nd coords
        gpu::find_min_max_2d_init<<<1,1>>>(d_coords,coord_min,coord_max);
        cudaDeviceSynchronize();
        gpu::find_min_max_2d<<<1,2>>>(d_coords,n_coords,start_vert,end_vert,coord_min,coord_max);
        cudaDeviceSynchronize();
        coord_min[0] -= (coord_max[0] - coord_min[0])*0.00001;
        coord_max[0] += (coord_max[0] - coord_min[0])*0.00001;
        coord_min[1] -= (coord_max[1] - coord_min[1])*0.00001;
        coord_max[1] += (coord_max[1] - coord_min[1])*0.00001;

        // printf("\n*** coord_min: ***\n");
        // cpu::print_array(coord_min, 0, 2);
        // printf("\n*** coord_max: ***\n");
        // cpu::print_array(coord_max, 0, 2);

        // printf("\n*** Check point #2 ***\n");
        dim3 numblocks_2d(n_bins_x/32+1,n_bins_y/32+1);
        dim3 threadsperblock_2d(32,32);
        gpu::get_bin_coords<<<numblocks_2d,threadsperblock_2d,0,d.stream()>>>(coord_min, coord_max, coords_2d_bins, n_bins_x, n_bins_y);
        cudaDeviceSynchronize();
        // printf("\n*** coords_2d_bins ***\n");
        // cpu::print_array(coords_2d_bins,0,2*n_bins_x*n_bins_y);
       
        // printf("\n*** Check point #3 ***\n");
        gpu::constructPhaseSpaceBins<<<numBlocks_V, blockSize>>>(
                d_coords, n_coords, start_vert, end_vert, n_bins, coord_min, coord_max,
                n_vtx_per_bin, bin_idx);
                // zero_counters, bin_idx);
        cudaDeviceSynchronize();
        // printf("\n*** n_vtx_per_bin ***\n");
        // gpu::print_array<<<1,1>>>(n_vtx_per_bin,0,4);
        // cudaDeviceSynchronize();
        // cpu::print_array(n_vtx_per_bin,0,4);
        // gpu::print_array<<<1,1>>>(zero_counters,0,4);
        // cpu::print_array(zero_counters,0,10);
     
        // printf("\n*** Check point #4 ***\n");
        // calculate n_vtx_per_bin_cumulative
        gpu::calculate_n_vtx_per_bin_cumulative<<<1,1>>>(n_bins_x*n_bins_y, n_vtx_per_bin, n_vtx_per_bin_cumulative);
        cudaDeviceSynchronize();
        // printf("\n*** n_vtx_per_bin_cumulative ***\n");
        // gpu::print_array<<<1,1>>>(n_vtx_per_bin_cumulative,0,4+1);
        // cudaDeviceSynchronize();
        // cpu::print_array(n_vtx_per_bin_cumulative,0,4+1);

        // printf("\n*** Check point #5 ***\n");
        // printf("\nn_vtx_per_bin_cumulative:\n");

        gpu::prepare_translation_matrix<<<numBlocks_V, blockSize>>>(start_vert, end_vert, n_vtx_per_bin_cumulative, bin_idx, zero_counters, vtx_idx_translation_matrix);
        cudaDeviceSynchronize();
        // printf("\n*** vtx_idx_translation_matrix ***\n");
        // gpu::print_array<<<1,1>>>(vtx_idx_translation_matrix,0,end_vert);
        // cpu::print_array(vtx_idx_translation_matrix,0,end_vert);
        // gpu::prepare_backward_translation_matrix<<<numBlocks_V, blockSize>>>(start_vert, end_vert, vtx_idx_translation_matrix, backward_vtx_idx_translation_matrix);
        // cudaDeviceSynchronize();

        // printf("\n*** Check point #6 ***\n");
        // resort coordinates
        gpu::translate_2d_matrix<<<numBlocks_V,blockSize>>>(end_vert-start_vert, n_coords, vtx_idx_translation_matrix, d_coords, d_coords_sorted);
        // gpu::translate_2d_matrix<<<numBlocks_V,blockSize>>>(end_vert-start_vert, 1, vtx_idx_translation_matrix, bin_idx, bin_idx_sorted);
        // // synchronize threads
        cudaDeviceSynchronize();

        // printf("\n*** Check point #7 ***\n");
        cpu::create_bin_neighbours_list(n_bins_x,n_bins_y,bin_neighbours);
        // printf("\n*** bin_neighbours ***\n");
        // cpu::print_array(bin_neighbours,0,9*n_bins_x*n_bins_y);

        // printf("\n*** Check point #8 ***\n");
        // prepare output matrix
        gpu::insert_value_into_array<<<1,1>>>(n_bins_x,auxaliry_knn_arrays,0);
        gpu::insert_value_into_array<<<1,1>>>(n_bins_y,auxaliry_knn_arrays,1);
        // auxaliry_knn_arrays[0] = n_bins_x;
        // auxaliry_knn_arrays[1] = n_bins_y;
        // printf("\n*** Check point #9 ***\n");
        gpu::insert_into_array<<<numBlocks_V,blockSize>>>(n_vtx_per_bin_cumulative,auxaliry_knn_arrays,2,2+n_bins_x*n_bins_y+1);
        cudaDeviceSynchronize();
        // printf("\n*** Check point #10 ***\n");
        gpu::insert_into_array<<<numBlocks_V,blockSize>>>(bin_neighbours,auxaliry_knn_arrays,2+n_bins_x*n_bins_y+1,2+n_bins_x*n_bins_y+1+9*n_bins_x*n_bins_y);
        cudaDeviceSynchronize();

        // printf("\n*** auxaliry_knn_arrays before last insertion ***\n");
        // gpu::print_array<<<1,1>>>(auxaliry_knn_arrays,0,2+n_bins_x*n_bins_y+1+9*n_bins_x*n_bins_y);

        // write bin_vtx_assoc array
        for (int i=0; i<n_bins_x*n_bins_y; i++){
            gpu::set_defaults<<<numBlocks_n_bins ,blockSize>>>(help_arr, n_vtx_per_bin[i], i);
            cudaDeviceSynchronize();
            int start_index = 2+n_bins_x*n_bins_y+1+9*n_bins_x*n_bins_y+n_vtx_per_bin_cumulative[i];
            int end_index = start_index+n_vtx_per_bin_cumulative[i+1];
            gpu::insert_into_array<<<numBlocks_V,blockSize>>>(help_arr,auxaliry_knn_arrays,start_index,end_index);
            cudaDeviceSynchronize();
        }

        // Free memory
        cudaFree(coords_2d_bins);
        cudaFree(n_vtx_per_bin);
        cudaFree(n_vtx_per_bin_cumulative);
        cudaFree(bin_idx);
        cudaFree(bin_idx_sorted);
        cudaFree(vtx_idx_translation_matrix);
        // cudaFree(backward_vtx_idx_translation_matrix);
        cudaFree(n_bins);
        cudaFree(coord_max);
        cudaFree(coord_min);
        cudaFree(zero_counters);
        cudaFree(bin_neighbours);
        cudaFree(help_arr);
    }
};


template struct Pre2KnnOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA
