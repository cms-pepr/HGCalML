//#define GOOGLE_CUDA 1

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "slicing_knn_kernel.h"
#include "helpers.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cuda_helpers.h"
#include <iostream>
#include <vector>

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
        const size_t n_bins,
        const int* n_vtx_per_bin,
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

void set_defaults(
        int *neigh_idx,
        float *neigh_dist,
        const int V,
        const int K
){
    for(size_t i = 0 ; i < V*K ; i += 1){
        neigh_idx[i] = -1;
        neigh_dist[i] = 0;
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
void print_gpu_array(
        const T *in_arr,
        const size_t start,
        const size_t end,
        bool convert_to_int = false
){
    int arr_size = end-start;
    std::vector<T> tmp_arr(arr_size);
    cudaMemcpy(&tmp_arr.at(0),in_arr,arr_size*sizeof(T),cudaMemcpyDeviceToHost);

    for(size_t i = start ; i < end ; i += 1){
        float tmp_val = tmp_arr.at(i);
        if (convert_to_int == true)
            printf("i: %d;\t%d\n", i, (int)tmp_val);
        else
            printf("i: %d;\t%f\n", i, tmp_val);
    }
}

}// cpu namespace

namespace gpu{

__global__
void calculate_n_vtx_per_bin_cumulative(
        const int n_bins,
        const int* n_vtx_per_bin,
        int* n_vtx_per_bin_cumulative)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for(size_t i = index ; i < n_bins+1; i += stride){
        n_vtx_per_bin_cumulative[i] = n_vtx_per_bin_cumulative[i-1] + n_vtx_per_bin[i-1];
    }
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


// calculate min and max of X and Y coordinates among all vertices
// make bins with widths: (x_max-x_min)/n_bins and (y_max-y_min)/n_bins
__global__
void constructPhaseSpaceBins(const float *d_coords, const size_t n_coords, const size_t start_vert,
                             const size_t end_vert, const int* n_bins,
                             const float* coord_min, const float* coord_max, 
                             const int* features_to_bin_on, int *n_vtx_per_bin,
                             int* bin_idx){

    // define which vertices belong to which bin
    int index = blockIdx.x * blockDim.x + threadIdx.x + start_vert;
    int stride = blockDim.x * gridDim.x;
    for(size_t i_v = index; i_v < end_vert; i_v += stride){
        
        size_t iDim = features_to_bin_on[0];
        float coord = d_coords[I2D(i_v,iDim,n_coords)];
        size_t indx_1 = (size_t)((coord - coord_min[iDim])/((coord_max[iDim]-coord_min[iDim])/n_bins[iDim]));
        iDim = features_to_bin_on[1];
        coord = d_coords[I2D(i_v,iDim,n_coords)];
        size_t indx_2 = (size_t)((coord - coord_min[iDim])/((coord_max[iDim]-coord_min[iDim])/n_bins[iDim]));

        size_t bin_index = I2D(indx_2, indx_1, n_bins[0]);
        atomicAdd(&n_vtx_per_bin[bin_index], 1); // to avoid race condition
        bin_idx[i_v] = bin_index;
    }
}


__global__
void prepare_translation_matrix(const size_t start_vert, const size_t end_vert, const int* n_vtx_per_bin_cumulative, const int* bin_idx, int* zero_counters, int* forward_translation_matrix){
    
    size_t index = blockIdx.x * blockDim.x + threadIdx.x + start_vert;
    size_t stride = blockDim.x * gridDim.x;
    for(size_t i_v = index; i_v < end_vert; i_v += stride){
        int bin_index = bin_idx[i_v];
        int index_to_fill = atomicAdd(&zero_counters[bin_index],1) + n_vtx_per_bin_cumulative[bin_index] + start_vert;
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
void translate_2d_matrix(const size_t start_vert, const size_t end_vert, const size_t matrix_width, const int* translation_matrix, const T *in_mattrix, T* out_matrix){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x + start_vert;
    size_t stride = blockDim.x * gridDim.x;
    for(size_t i_counter = index; i_counter < end_vert; i_counter += stride){
        size_t real_index = translation_matrix[i_counter];
        for(size_t i_column = 0; i_column < matrix_width; i_column += 1)
            out_matrix[matrix_width*i_counter+i_column] = in_mattrix[matrix_width*real_index+i_column];
    }
}

template <typename T>
__global__
void translate_ind_matrix(const size_t start_vert, const size_t end_vert, const size_t matrix_width, const int* translation_matrix, T *in_matrix){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x + start_vert;
    size_t stride = blockDim.x * gridDim.x;
    for(size_t i_counter = index; i_counter < end_vert; i_counter += stride){
        for(size_t i_column = 0; i_column < matrix_width; i_column += 1){
            size_t final_index = matrix_width*i_counter+i_column;
            in_matrix[final_index] = translation_matrix[in_matrix[final_index]];
        }
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

__global__
void set_defaults(
        int *neigh_idx,
        float *neigh_dist,
        const int V,
        const int K
){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(size_t i = index ; i < V*K ; i += stride){
        neigh_idx[i] = -1;
        neigh_dist[i] = 0;
    }
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

__global__
void print_neighbours(
        const size_t i_v,
        int *neigh_idx,
        float *neigh_dist,
        const size_t K
){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(size_t i = index ; i < K ; i += stride){
        printf(" %d (%f)", neigh_idx[I2D(i_v,i,K)], neigh_dist[I2D(i_v,i,K)]);
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


__global__
void findNeighbours(const int* indices_of_vert_to_find_new_neigh, // vertices for which we want to find neighbours in the targe phase-space bin
                    const size_t n_vertices_to_loop, // size of the first input array
                    const size_t indx_bin_to_use, // index of the newly added bin
                    const int* index_map_to_bins, // index_map_to_bins[i_v] -> bin number to which vertex belong
                    const int* n_vtx_per_bin,
                    const float *d_coords,
                    size_t start_vert,
                    size_t end_vert,
                    size_t n_coords, // number of dimentions
                    size_t K, // number of neighbours
                    float* neigh_dist, // distance matrix
                    int* neigh_idx, // indices matrix which corresponds to distance one
                    float max_radius = -1.0 // max. radius to search for neighbours
                    ){
    
    // loop to assign indices and distances to other vertices
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(size_t i = index; i <  n_vertices_to_loop; i += stride){
    // for(size_t i = 0; i < n_vertices_to_loop; i++){
        size_t i_v = indices_of_vert_to_find_new_neigh[i];

        //protection against V<K
        size_t max_neighbours = K;

        size_t nfilled=0;
        int running_index = max_neighbours - 1;

        while (running_index>=0){
            if (neigh_idx[I2D(i_v,running_index,K)] == -1) // default init value
                running_index -= 1;
            else{
                nfilled = running_index+1;
                break;
            }
        }
        
        //set default to self
        if((n_vtx_per_bin[indx_bin_to_use]+nfilled)<K){
            max_neighbours=(n_vtx_per_bin[indx_bin_to_use]+nfilled);
        }
        
        float maxdistsq = 0;
        size_t maxidx_local = 0;
        if (nfilled>0){
            maxidx_local = searchLargestDistance(i_v,neigh_dist,K,maxdistsq);
        }
        
        // assigning loop - searching neighbouth for i_v
        for(size_t j_v=start_vert;j_v<end_vert;j_v++){
            if(index_map_to_bins[j_v]!=indx_bin_to_use)
                continue;
            //fill up
            float distsq = calculateDistance(i_v,j_v,d_coords,n_coords);
          
            if(nfilled<max_neighbours && (max_radius<=0 || max_radius>=distsq)){
                // filling in distances until we reach max_neighbours
                neigh_idx[I2D(i_v,nfilled,K)] = j_v;
                neigh_dist[I2D(i_v,nfilled,K)] = distsq;
                
                if(distsq > maxdistsq){
                    maxdistsq = distsq;
                    maxidx_local = nfilled;
                }
                nfilled++;
                continue;
            }
            
            // if we already filled max_neighbours distances, compare each new distance
            // with the current maximum. if distance is smaller - threw away current maximum,
            // fill in new distance and find new maximum
            if(distsq < maxdistsq){// automatically applies to max radius
                //replace former max
                neigh_idx[I2D(i_v,maxidx_local,K)] = j_v;
                neigh_dist[I2D(i_v,maxidx_local,K)] = distsq;

                //search new max
                maxidx_local = searchLargestDistance(i_v,neigh_dist,K,maxdistsq);
            }
        }// loop through vertices
    }// loop through vertices
}

__global__
void perform_kNN_search(
    const int start_vert,
    const int end_vert,
    const float* d_coords_sorted,
    const int n_coords,
    const int K,
    const int bin_index_to_use,
    const int* bin_neighbours, // n_bins*9
    const int* n_vtx_per_bin_cumulative,
    const int* vtx_bin_assoc,
    int* farthest_neighbour,
    int* neigh_idx,
    float* neigh_dist
    ){

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for(size_t i = index + start_vert; i < end_vert; i += stride){
        int i_v = i;
        
        // protection against n_vert<K
        size_t max_neighbours = K;

        size_t nfilled=0; // number of already found neighbours in the matrix neigh_idx
        int running_index = max_neighbours - 1;

        while (running_index>=0){
            if (neigh_idx[I2D(i_v,running_index,K)] == -1) // default init value
                running_index -= 1;
            else{
                nfilled = running_index+1;
                break;
            }
        }

        int vtx_bin = vtx_bin_assoc[i_v];
        int target_bin = bin_neighbours[bin_index_to_use + 9*vtx_bin];
        if (target_bin<0) // can happen when bin has less than 8 neighbour bins
            continue;
        int target_bin_start_vert = n_vtx_per_bin_cumulative[target_bin] + start_vert;
        int target_bin_end_vert  = n_vtx_per_bin_cumulative[target_bin+1] + start_vert;
        
        if((target_bin_end_vert -target_bin_start_vert +nfilled)<K){
            max_neighbours=(target_bin_end_vert -target_bin_start_vert +nfilled); // K or less
        }
        
        float maxdistsq = 0;
        
        if (nfilled>1){ // if nfilled==1 - neighbour is the vtx itself
            maxdistsq = neigh_dist[I2D(i_v,farthest_neighbour[i_v],K)];
        }
        
        for(size_t j_v=target_bin_start_vert;j_v<target_bin_end_vert;j_v++){
            float distsq = calculateDistance(i_v,j_v,d_coords_sorted,n_coords);

            if(nfilled<max_neighbours){
                // filling in distances until we reach max_neighbours
                neigh_idx[I2D(i_v,nfilled,K)] = j_v;
                neigh_dist[I2D(i_v,nfilled,K)] = distsq;

                if(distsq > maxdistsq){
                    maxdistsq = distsq;
                    farthest_neighbour[i_v] = nfilled;
                }
                nfilled++;
                continue;
            }

            // if we already filled max_neighbours distances, compare each new distance
            // with the current maximum. if distance is smaller - threw away current maximum,
            // fill in new distance and find new maximum
            if(distsq < maxdistsq){// automatically applies to max radius
                //replace former max
                neigh_idx[I2D(i_v,farthest_neighbour[i_v],K)] = j_v;
                neigh_dist[I2D(i_v,farthest_neighbour[i_v],K)] = distsq;

                //search new max
                farthest_neighbour[i_v] = searchLargestDistance(i_v,neigh_dist,K,maxdistsq);
            }
        }// loop through target bin vertices
    }// loop thorugh all vertices
}



} // gpu namespace

typedef Eigen::GpuDevice GPUDevice;


template <typename dummy>
struct SlicingKnnOpFunctor<GPUDevice, dummy> {
    void operator()(const GPUDevice& d,

            const float *d_coords, // accessible only on GPU!
            const int* d_row_splits, // accessible only on GPU!
            int *neigh_idx,
            float *neigh_dist,

            const int V, // # of vertices
            const int K, // # of neighbours to be found
            const int n_coords,
            std::vector<float> phase_space_bin_boundary,
            const int n_rs,
            std::vector<int> n_bins,
            std::vector<int> features_to_bin_on
            ) {
        
        // ******************************************
        // STEP 1: memory allocation and set defaults
        // ******************************************
        int blockSize = 256;
        const int n_bins_x = n_bins[0];
        const int n_bins_y = n_bins[1];
        int numBlocks_V = (V + blockSize - 1) / blockSize;
        int numBlocks_n_bins = (n_bins_x*n_bins_y + blockSize - 1) / blockSize;

        int* n_bins_at_gpu;
        cudaMallocManaged(&n_bins_at_gpu, 2*sizeof(int));
        n_bins_at_gpu[0] = n_bins[0];
        n_bins_at_gpu[1] = n_bins[1];

        int* features_to_bin_on_at_gpu; 
        cudaMallocManaged(&features_to_bin_on_at_gpu, 2*sizeof(int));
        features_to_bin_on_at_gpu[0] = features_to_bin_on[0];
        features_to_bin_on_at_gpu[1] = features_to_bin_on[1];

        float* d_coords_sorted;
        cudaMallocManaged(&d_coords_sorted, V*n_coords*sizeof(float));

        int* tmp_neigh_idx;
        cudaMallocManaged(&tmp_neigh_idx, V*K*sizeof(int));

        float* tmp_neigh_dist;
        cudaMallocManaged(&tmp_neigh_dist, V*K*sizeof(float));

        int *bin_idx;
        cudaMallocManaged(&bin_idx, V*sizeof(int));

        int *help_arr;
        cudaMallocManaged(&help_arr, V*sizeof(int));

        int *vtx_bin_assoc;
        cudaMallocManaged(&vtx_bin_assoc, V*sizeof(int));

        int *bin_idx_sorted ;
        cudaMallocManaged(&bin_idx_sorted, V*sizeof(int));

        int *vtx_idx_translation_matrix;
        cudaMallocManaged(&vtx_idx_translation_matrix, V*sizeof(int));

        int *backward_vtx_idx_translation_matrix;
        cudaMallocManaged(&backward_vtx_idx_translation_matrix, V*sizeof(int));

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

        int *farthest_neighbour;
        cudaMallocManaged(&farthest_neighbour, V*sizeof(int));

        // set default values global
        gpu::set_defaults<<<numBlocks_V,blockSize>>>(tmp_neigh_idx, tmp_neigh_dist, V, K);
        gpu::set_defaults<<<numBlocks_V,blockSize>>>(farthest_neighbour, V, -1);

        dim3 numblocks_2d(n_bins_x/32+1,n_bins_y/32+1);
        dim3 threadsperblock_2d(32,32);

        cpu::create_bin_neighbours_list(n_bins_x,n_bins_y,bin_neighbours);

        // needed since d_row_splits is only accessible in GPU memory
        std::vector<int> cpu_rowsplits(n_rs);
        cudaMemcpy(&cpu_rowsplits.at(0),d_row_splits,n_rs*sizeof(int),cudaMemcpyDeviceToHost);

        for(size_t j_rs=0;j_rs<n_rs-1;j_rs++){ //n_rs-1 important!
            const int nvert_rs = cpu_rowsplits.at(j_rs+1) - cpu_rowsplits.at(j_rs);

            const int numBlocks_Vrs = (nvert_rs + blockSize - 1) / blockSize;
            const size_t start_vert = cpu_rowsplits.at(j_rs);
            const size_t end_vert = cpu_rowsplits.at(j_rs+1);

            // set default values per row split
            gpu::set_defaults<<<numBlocks_n_bins ,blockSize>>>(n_vtx_per_bin, n_bins_x*n_bins_y, 0);
            gpu::set_defaults<<<numBlocks_n_bins ,blockSize>>>(zero_counters, n_bins_x*n_bins_y, 0);
            gpu::set_defaults<<<numBlocks_n_bins ,blockSize>>>(n_vtx_per_bin_cumulative, n_bins_x*n_bins_y+1, 0);
        
            // ********************************
            // STEP 2: fill in auxiliary arrays
            // ********************************
            coord_min[0] = phase_space_bin_boundary[0+4*j_rs];
            coord_max[0] = phase_space_bin_boundary[1+4*j_rs];
            coord_min[1] = phase_space_bin_boundary[2+4*j_rs];
            coord_max[1] = phase_space_bin_boundary[3+4*j_rs];

            gpu::constructPhaseSpaceBins<<<numBlocks_Vrs, blockSize>>>(
                    d_coords, n_coords, start_vert, end_vert, n_bins_at_gpu, coord_min, coord_max,
                    features_to_bin_on_at_gpu, n_vtx_per_bin, bin_idx);
            cudaDeviceSynchronize();

            gpu::calculate_n_vtx_per_bin_cumulative<<<1,1>>>(n_bins_x*n_bins_y, n_vtx_per_bin, n_vtx_per_bin_cumulative);
            cudaDeviceSynchronize();

            gpu::prepare_translation_matrix<<<numBlocks_Vrs, blockSize>>>(start_vert, end_vert, n_vtx_per_bin_cumulative, bin_idx, zero_counters, vtx_idx_translation_matrix);
            cudaDeviceSynchronize();

            gpu::prepare_backward_translation_matrix<<<numBlocks_Vrs, blockSize>>>(start_vert, end_vert, vtx_idx_translation_matrix, backward_vtx_idx_translation_matrix);
            cudaDeviceSynchronize();

            gpu::translate_2d_matrix<<<numBlocks_Vrs,blockSize>>>(start_vert, end_vert, n_coords, vtx_idx_translation_matrix, d_coords, d_coords_sorted);
            cudaDeviceSynchronize();

            // write bin_vtx_assoc array
            for (int i=0; i<n_bins_x*n_bins_y; i++){
                int numBlocks_tmp = (n_vtx_per_bin[i] + blockSize - 1) / blockSize;
                gpu::set_defaults<<<numBlocks_tmp ,blockSize>>>(help_arr, n_vtx_per_bin[i], i);
                cudaDeviceSynchronize();
                int start_index = n_vtx_per_bin_cumulative[i]+start_vert;
                int end_index = start_index+n_vtx_per_bin_cumulative[i+1]+start_vert;
                numBlocks_tmp = (end_index-start_index + blockSize - 1) / blockSize;
                gpu::insert_into_array<<<numBlocks_V,blockSize>>>(help_arr,vtx_bin_assoc,start_index,end_index);
                cudaDeviceSynchronize();
            }

            // ***********************
            // STEP 3: find neighbours
            // ***********************

            int counter = 0;
            int N_MAX_BINS_TO_LOOP = 9;
            while (counter<N_MAX_BINS_TO_LOOP){

                gpu::perform_kNN_search<<<numBlocks_V,blockSize>>>(
                    start_vert,
                    end_vert,
                    d_coords_sorted,
                    n_coords,
                    K,
                    counter,
                    bin_neighbours,
                    n_vtx_per_bin_cumulative,
                    vtx_bin_assoc,
                    farthest_neighbour,
                    tmp_neigh_idx,
                    tmp_neigh_dist
                    );
                cudaDeviceSynchronize();
                counter++;
            }
        }

        // *******************************************************
        // STEP 4: backward-propagate cordinate neighbour matrices
        // *******************************************************
        gpu::translate_2d_matrix<<<numBlocks_V,blockSize>>>(0, V, K, backward_vtx_idx_translation_matrix, tmp_neigh_idx, neigh_idx);
        cudaDeviceSynchronize();
        gpu::translate_ind_matrix<<<numBlocks_V,blockSize>>>(0, V, K, vtx_idx_translation_matrix, neigh_idx);
        gpu::translate_2d_matrix<<<numBlocks_V,blockSize>>>(0, V, K, backward_vtx_idx_translation_matrix, tmp_neigh_dist, neigh_dist);
        cudaDeviceSynchronize();

        // *****************************
        // STEP 5: free allocated memory
        // *****************************
        // Free memory
        cudaFree(d_coords_sorted);
        cudaFree(n_vtx_per_bin);
        cudaFree(n_vtx_per_bin_cumulative);
        cudaFree(bin_idx);
        cudaFree(bin_idx_sorted);
        cudaFree(vtx_idx_translation_matrix);
        cudaFree(backward_vtx_idx_translation_matrix);
        cudaFree(coord_max);
        cudaFree(coord_min);
        cudaFree(zero_counters);
        cudaFree(bin_neighbours);
        cudaFree(help_arr);
        cudaFree(vtx_bin_assoc);
        cudaFree(farthest_neighbour);
        cudaFree(tmp_neigh_dist);
        cudaFree(tmp_neigh_idx);
    }
};


template struct SlicingKnnOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA
