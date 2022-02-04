//#define GOOGLE_CUDA 1

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#define HANDLE_ERROR( err ) ( tensorflow::functor::HandleError( err, __FILE__, __LINE__ ) )

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

static void HandleError( cudaError_t err, const char *file = __FILE__ , int line = __LINE__ )
{
    if (err != cudaSuccess)
    {
        printf( "Cuda Error: %s in %s at line %d\n", cudaGetErrorString( err ),
               file, line );
        exit( EXIT_FAILURE );
    }
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda Error: %s: %s.\n", msg, 
                cudaGetErrorString(err) );
        exit(EXIT_FAILURE);
    }                         
}


namespace cpu{

void out_of_range_error_message(const char *file, int line )
{
    printf("Cuda Error: Out of Range Error in %s at line %d\n", file, line);
}

void create_bin_neighbours_list(
        const int n_bins_x,
        const int n_bins_y, 
        int* bin_neighbours,
        size_t size_bin_neighbours)
{
    // size_t size_bin_neighbours = sizeof(bin_neighbours)/sizeof(int);

    int tmp_arr[3] = {0,-1,1};

    for (int j = 0; j<n_bins_y; j+=1){
        for (int i = 0; i<n_bins_x; i+=1){
            int index = (i + n_bins_x*j)*9;
            int counter = 0;
            for (size_t ii = 0; ii<3; ii++){
                int i_s = tmp_arr[ii];
                for (size_t jj = 0; jj<3; jj++){
                    int j_s = tmp_arr[jj];
                    int tmp_index = i+i_s + n_bins_x*(j+j_s);
                    if (index+counter>=size_bin_neighbours)
                        out_of_range_error_message(__FILE__, __LINE__);
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
void print_array(
        std::vector<T> in_arr,
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
        const size_t arr_size,
        const size_t start,
        const size_t end,
        bool convert_to_int = false
){
    // cudaMemcpy is synchronous
    std::vector<T> tmp_arr(arr_size);
    HANDLE_ERROR(cudaMemcpy(&tmp_arr.at(0),in_arr,arr_size*sizeof(T),cudaMemcpyDeviceToHost));

    for(size_t i = start ; i < end ; i += 1){
        float tmp_val = tmp_arr.at(i);
        if (convert_to_int == true)
            printf("i: %d;\t%d\n", i, (int)tmp_val);
        else
            printf("i: %d;\t%f\n", i, tmp_val);
    }
}

template <typename T>
void print_gpu_matrix(
        const T *in_arr,
        const size_t start,
        const size_t end,
        const size_t stride,
        bool convert_to_int = false
){
    // cudaMemcpy is synchronous
    int arr_size = end-start;
    std::vector<T> tmp_arr(arr_size);
    HANDLE_ERROR(cudaMemcpy(&tmp_arr.at(0),in_arr,arr_size*sizeof(T),cudaMemcpyDeviceToHost));

    for(size_t i = start ; i < end ; i += 1){
        float tmp_val = tmp_arr[i];
        if (i % stride == 0){
            printf("\n");
            printf("%d: ", (int)(i/stride));
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

__device__
void out_of_range_error_message(const char *file, int line )
{
    printf("Cuda Error: Out of Range Error in %s at line %d\n", file, line);
}

__global__
void calculate_n_vtx_per_bin_cumulative(
        const int n_bins,
        const int* d_n_vtx_per_bin,
        const size_t size_d_n_vtx_per_bin,
        int* d_n_vtx_per_bin_cumulative,
        const size_t size_d_n_vtx_per_bin_cumulative
        )
{

    int index = blockIdx.x * blockDim.x + threadIdx.x + 1; // start with the second element
    int stride = blockDim.x * gridDim.x;
    
    for(size_t i = index ; i < n_bins+1; i += stride){
        if ((i>=size_d_n_vtx_per_bin_cumulative) || ((i-1) >= size_d_n_vtx_per_bin)) // index safeguard
            out_of_range_error_message(__FILE__, __LINE__);
        d_n_vtx_per_bin_cumulative[i] = d_n_vtx_per_bin_cumulative[i-1] + d_n_vtx_per_bin[i-1];
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
        const size_t size_arr_to_copy, 
        T *arr_to_insert,
        const size_t size_arr_to_insert,
        const size_t start_pos,
        const size_t end_pos
){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(size_t i = index ; i < (end_pos-start_pos) ; i += stride){
        if (((start_pos+i)<0) || ((start_pos+i)>=size_arr_to_insert) || (i>=size_arr_to_copy)) // index safeguard
            out_of_range_error_message(__FILE__, __LINE__);
        arr_to_insert[start_pos+i] = arr_to_copy[i];
    }
}

template <typename T>
__global__
void insert_value_into_array(
        const T val_to_copy,
        T *arr_to_insert,
        const size_t size_arr_to_insert,
        const size_t pos
){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(size_t i = index ; i < 1 ; i += stride){
        if ((pos<0) || (pos>=size_arr_to_insert)) // index safeguard
            out_of_range_error_message(__FILE__, __LINE__);
        arr_to_insert[pos] = val_to_copy;
    }
}


template <typename T>
__device__
void print_device_array(
        const T *d_arr,
        const size_t start,
        const size_t end
){
    for(size_t i = start ; i < end ; i += 1){
        printf("%d\t%f\n", i, d_arr[i]);
    }
}

// calculate min and max of X and Y coordinates among all vertices
// make bins with widths: (x_max-x_min)/n_bins and (y_max-y_min)/n_bins
__global__
void constructPhaseSpaceBins(const float *d_coords, const float size_d_coords, const size_t n_coords,
                             const size_t start_vert, const size_t end_vert, const int* d_n_bins,
                             const size_t size_d_n_bins, const float* d_coords_min, const size_t size_d_coords_min,
                             const float* d_coords_max, const size_t size_d_coords_max, const int* d_features_to_bin_on,
                             const size_t size_d_features_to_bin_on, int *d_n_vtx_per_bin,
                             const size_t size_d_n_vtx_per_bin, int* d_bin_idx, const size_t size_d_bin_idx){

    // define which vertices belong to which bin
    int index = blockIdx.x * blockDim.x + threadIdx.x + start_vert;
    int stride = blockDim.x * gridDim.x;

    for(size_t i_v = index; i_v < end_vert; i_v += stride){
        
        size_t iDim = d_features_to_bin_on[0];
        if ((iDim<0) || (iDim >= size_d_n_bins) || (iDim >= size_d_coords_min) || (iDim >= size_d_coords_max))
            out_of_range_error_message(__FILE__, __LINE__);

        float coord = d_coords[I2D(i_v,iDim,n_coords)];
        size_t indx_1 = (size_t)((coord - d_coords_min[iDim])/((d_coords_max[iDim]-d_coords_min[iDim])/d_n_bins[iDim]));
        iDim = d_features_to_bin_on[1];
        coord = d_coords[I2D(i_v,iDim,n_coords)];
        size_t indx_2 = (size_t)((coord - d_coords_min[iDim])/((d_coords_max[iDim]-d_coords_min[iDim])/d_n_bins[iDim]));

        size_t bin_index = I2D(indx_2, indx_1, d_n_bins[0]);
        if (bin_index >= size_d_n_vtx_per_bin) // index safeguard
            out_of_range_error_message(__FILE__, __LINE__);
        atomicAdd(&d_n_vtx_per_bin[bin_index], 1); // to avoid race condition
        if (i_v >= size_d_bin_idx) // index safeguard
            out_of_range_error_message(__FILE__, __LINE__);
        d_bin_idx[i_v] = bin_index;
    }
}


__global__
void prepare_translation_matrix(const size_t start_vert, const size_t end_vert, const int* d_n_vtx_per_bin_cumulative,
                                const size_t size_d_n_vtx_per_bin_cumulative, const int* d_bin_idx,
                                const size_t size_d_bin_idx, int* d_zero_counters, const size_t size_d_zero_counters,
                                int* forward_translation_matrix, const size_t size_forward_translation_matrix){

    size_t index = blockIdx.x * blockDim.x + threadIdx.x + start_vert;
    size_t stride = blockDim.x * gridDim.x;
    for(size_t i_v = index; i_v < end_vert; i_v += stride){
        if (i_v >= size_d_bin_idx) // index safeguard
            out_of_range_error_message(__FILE__, __LINE__);
        int bin_index = d_bin_idx[i_v];
        //Jan: this needs some explanation how index_to_fill is unique and safe
        if ((bin_index >= size_d_n_vtx_per_bin_cumulative) || (bin_index >= size_d_zero_counters)) // index safeguard
            out_of_range_error_message(__FILE__, __LINE__);
        int index_to_fill = atomicAdd(&d_zero_counters[bin_index],1) + d_n_vtx_per_bin_cumulative[bin_index] + start_vert;
        if (index_to_fill >= size_forward_translation_matrix) // index safeguard
            out_of_range_error_message(__FILE__, __LINE__);
        forward_translation_matrix[index_to_fill] = i_v;
    }
}

__global__
void prepare_backward_translation_matrix(const size_t start_vert, const size_t end_vert,
                                         const int* forward_translation_matrix,
                                         const size_t size_forward_translation_matrix, int* backward_translation_matrix,
                                         const size_t size_backward_translation_matrix){
    
    size_t index = blockIdx.x * blockDim.x + threadIdx.x + start_vert;
    size_t stride = blockDim.x * gridDim.x;
    for(size_t i_v = index; i_v < end_vert; i_v += stride){
        if (i_v >= size_forward_translation_matrix) // index safeguard
            out_of_range_error_message(__FILE__, __LINE__);
        if ((forward_translation_matrix[i_v]<0) || (forward_translation_matrix[i_v] >= size_backward_translation_matrix)) // index safeguard
            out_of_range_error_message(__FILE__, __LINE__);
        backward_translation_matrix[forward_translation_matrix[i_v]] = i_v;
    }
}

template <typename T>
__global__
void translate_2d_matrix(const size_t start_vert, const size_t end_vert, const size_t matrix_width,
                         const int* translation_matrix, const size_t size_translation_matrix, const T *in_matrix,
                         const size_t size_in_matrix, T* out_matrix, const size_t size_out_matrix){

    size_t index = blockIdx.x * blockDim.x + threadIdx.x + start_vert;
    size_t stride = blockDim.x * gridDim.x;
    for(size_t i_counter = index; i_counter < end_vert; i_counter += stride){
        size_t real_index = translation_matrix[i_counter];
        for(size_t i_column = 0; i_column < matrix_width; i_column += 1){
            if (((matrix_width*real_index+i_column)<0) || ((matrix_width*real_index+i_column)>=size_in_matrix) || ((matrix_width*i_counter+i_column)>=size_out_matrix)) // index safeguard
                out_of_range_error_message(__FILE__, __LINE__);
            out_matrix[matrix_width*i_counter+i_column] = in_matrix[matrix_width*real_index+i_column];
        }
    }
}

template <typename T>
__global__
void translate_ind_matrix(const size_t start_vert, const size_t end_vert, const size_t matrix_width,
                          const int* translation_matrix, const size_t size_translation_matrix, T *in_matrix,
                          const size_t size_in_matrix){

    size_t index = blockIdx.x * blockDim.x + threadIdx.x + start_vert;
    size_t stride = blockDim.x * gridDim.x;
    for(size_t i_counter = index; i_counter < end_vert; i_counter += stride){
        for(size_t i_column = 0; i_column < matrix_width; i_column += 1){
            size_t final_index = matrix_width*i_counter+i_column;
            if (final_index>=size_in_matrix) // index safeguard
                out_of_range_error_message(__FILE__, __LINE__);
            const int tmp_val1 = in_matrix[final_index];
            if (tmp_val1==-1){
                in_matrix[final_index] = -1;
            }
            else{
                const size_t tmp_val2 = translation_matrix[tmp_val1];
                in_matrix[final_index] = tmp_val2;
            }
        }
    }
}

__global__
void translate_matrices_to_make_selfindex_first(const size_t matrix_height, const size_t matrix_width, int* indx_matrix,
                                                const size_t size_indx_matrix, float* dist_matrix,
                                                const size_t size_dist_matrix){

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for(size_t i_row = index; i_row < matrix_height; i_row += stride){
        if (indx_matrix[matrix_width*i_row] == i_row)
            continue;
        for(size_t i_column = 0; i_column < matrix_width; i_column += 1){
            size_t index = matrix_width*i_row+i_column;
            if (index>=size_indx_matrix || matrix_width*i_row>=size_indx_matrix) // index safeguard
                out_of_range_error_message(__FILE__, __LINE__);
            if (index>=size_dist_matrix || matrix_width*i_row>=size_dist_matrix) // index safeguard
                out_of_range_error_message(__FILE__, __LINE__);
            if (indx_matrix[index]==i_row){
                indx_matrix[index] = indx_matrix[matrix_width*i_row];
                indx_matrix[matrix_width*i_row] = i_row;
                const float tmp_val = dist_matrix[index];
                dist_matrix[index] = dist_matrix[matrix_width*i_row];
                dist_matrix[matrix_width*i_row] = tmp_val;
                break;
            }
        }
    }
}

__global__
void translate_content_of_2d_matrix(size_t n_element_in_matrix, const int* translation_matrix,
                                    const size_t size_translation_matrix, const int *in_matrix,
                                    const size_t size_in_matrix, int* out_matrix, const size_t size_out_matrix){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for(size_t i_counter = index; i_counter < n_element_in_matrix; i_counter += stride){
        int real_index = in_matrix[i_counter];
        int translated_index = -1;
        if (real_index>=size_translation_matrix) // index safeguard
            out_of_range_error_message(__FILE__, __LINE__);
        if (real_index>=0)
            translated_index = translation_matrix[real_index];
        out_matrix[i_counter] = translated_index;
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
        int *neigh_idx, const size_t size_neigh_idx, float *neigh_dist, const size_t size_neigh_dist, const int V,
        const int K){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (V*K>size_neigh_idx || V*K>size_neigh_dist)
        out_of_range_error_message(__FILE__, __LINE__);
    for(size_t i = index ; i < V*K ; i += stride){
        neigh_idx[i] = -1;
        neigh_dist[i] = 0;
    }
}

template <typename T>
__global__
void set_defaults(
        T *in_arr,
        const size_t size_in_arr,
        const size_t size_to_fill,
        const T def_val
){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (size_to_fill>size_in_arr)
        out_of_range_error_message(__FILE__, __LINE__);
    for(size_t i = index ; i < size_to_fill ; i += stride){
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
                    const size_t size_index_map_to_bins, // index_map_to_bins[i_v] -> bin number to which vertex belong
                    const int* d_n_vtx_per_bin,
                    const size_t size_d_n_vtx_per_bin,
                    const float *d_coords,
                    const size_t size_d_coords,
                    size_t start_vert,
                    size_t end_vert,
                    size_t n_coords, // number of dimentions
                    size_t K, // number of neighbours
                    float* neigh_dist, // distance matrix
                    const size_t size_neigh_dist, // distance matrix
                    int* neigh_idx, // indices matrix which corresponds to distance one
                    const size_t size_neigh_idx, // indices matrix which corresponds to distance one
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
        if((d_n_vtx_per_bin[indx_bin_to_use]+nfilled)<K){
            max_neighbours=(d_n_vtx_per_bin[indx_bin_to_use]+nfilled);
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
                int tmp_index = I2D(i_v,nfilled,K);
                if (tmp_index>=size_neigh_idx || tmp_index>=size_neigh_dist)
                    out_of_range_error_message(__FILE__, __LINE__); // index safeguard
                neigh_idx[tmp_index] = j_v;
                neigh_dist[tmp_index] = distsq;
                
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
                int tmp_index = I2D(i_v,maxidx_local,K);
                if (tmp_index>=size_neigh_idx || tmp_index>=size_neigh_dist)
                    out_of_range_error_message(__FILE__, __LINE__); // index safeguard
                neigh_idx[tmp_index] = j_v;
                neigh_dist[tmp_index] = distsq;

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
    const size_t size_d_coords_sorted,
    const int n_coords,
    const int K,
    const int bin_index_to_use,
    const int* d_bin_neighbours, // n_bins*9
    const size_t size_d_bin_neighbours,
    const int* d_n_vtx_per_bin_cumulative,
    const size_t size_d_n_vtx_per_bin_cumulative,
    const int* d_vtx_bin_assoc,
    const size_t size_d_vtx_bin_assoc,
    int* d_farthest_neighbour,
    const size_t size_d_farthest_neighbour,
    int* neigh_idx,
    const size_t size_neigh_idx,
    float* neigh_dist,
    const size_t size_neigh_dist
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

        if (nfilled==0){
            // the closest vertex is the vertex itself
            // prefill all relevant matrices
            int tmp_index = I2D(i_v,0,K);
            if (tmp_index>=size_neigh_idx || tmp_index>=size_neigh_dist)
                out_of_range_error_message(__FILE__, __LINE__); // index safeguard
            if (i_v>=size_d_farthest_neighbour)
                out_of_range_error_message(__FILE__, __LINE__); // index safeguard
            neigh_idx[tmp_index] = i_v;
            neigh_dist[tmp_index] = 0.0;
            d_farthest_neighbour[i_v] = 0;//Jan: why is this a vector? could just be a local register (faster and less prone to errors)
            nfilled += 1;
        }

        int vtx_bin = d_vtx_bin_assoc[i_v];
        int target_bin = d_bin_neighbours[bin_index_to_use + 9*vtx_bin];
        if (target_bin<0) // can happen when bin has less than 8 neighbour bins
            continue;
        int target_bin_start_vert = d_n_vtx_per_bin_cumulative[target_bin] + start_vert;
        int target_bin_end_vert  = d_n_vtx_per_bin_cumulative[target_bin+1] + start_vert;
        
        if((target_bin_end_vert -target_bin_start_vert +nfilled)<K){
            max_neighbours=(target_bin_end_vert -target_bin_start_vert +nfilled); // K or less
        }
        
        float maxdistsq = 0;
        
        if (nfilled>1){ // if nfilled==1 - neighbour is the vtx itself
            maxdistsq = neigh_dist[I2D(i_v,d_farthest_neighbour[i_v],K)];
        }
        
        for(size_t j_v=target_bin_start_vert;j_v<target_bin_end_vert;j_v++){
            if (i_v==j_v)
                continue;
            float distsq = calculateDistance(i_v,j_v,d_coords_sorted,n_coords);

            if(nfilled<max_neighbours){
                // filling in distances until we reach max_neighbours
                int tmp_index = I2D(i_v,nfilled,K);
                if (tmp_index>=size_neigh_idx || tmp_index>=size_neigh_dist)
                    out_of_range_error_message(__FILE__, __LINE__); // index safeguard
                neigh_idx[tmp_index] = j_v;
                neigh_dist[tmp_index] = distsq;

                if(distsq > maxdistsq){
                    if (i_v>=size_d_farthest_neighbour)
                        out_of_range_error_message(__FILE__, __LINE__); // index safeguard
                    maxdistsq = distsq;
                    d_farthest_neighbour[i_v] = nfilled;
                }
                nfilled++;
                continue;
            }

            // if we already filled max_neighbours distances, compare each new distance
            // with the current maximum. if distance is smaller - threw away current maximum,
            // fill in new distance and find new maximum
            if(distsq < maxdistsq){// automatically applies to max radius
                int tmp_index = I2D(i_v,d_farthest_neighbour[i_v],K);
                if (tmp_index>=size_neigh_idx || tmp_index>=size_neigh_dist)
                    out_of_range_error_message(__FILE__, __LINE__); // index safeguard
                //replace former max
                neigh_idx[tmp_index] = j_v;
                neigh_dist[tmp_index] = distsq;

                if (i_v>=size_d_farthest_neighbour)
                    out_of_range_error_message(__FILE__, __LINE__); // index safeguard
                //search new max
                d_farthest_neighbour[i_v] = searchLargestDistance(i_v,neigh_dist,K,maxdistsq);
            }
        }// loop through target bin vertices
    }// loop thorugh all vertices
}



} // gpu namespace

typedef Eigen::GpuDevice GPUDevice;


template <typename dummy>
struct SlicingKnnOpFunctor<GPUDevice, dummy> {
    void operator()(const GPUDevice& d,

            const float *d_coords, // accessible only on GPU!!!
            const int* d_row_splits, // accessible only on GPU!
            const int* d_n_bins, // accessible only on GPU!!!
            const float* d_coords_min, // accessible only on GPU!!!
            const float* d_coords_max, // accessible only on GPU!!!
            int *neigh_idx,
            float *neigh_dist,

            const int V, // # of vertices
            const int K, // # of neighbours to be found
            const int n_coords,
            const int n_rs,

            std::vector<int> features_to_bin_on
            ) {
        
        // ******************************************
        // STEP 1: memory allocation and set defaults
        // ******************************************

        // copy n_bins to CPU memory
        std::vector<int> n_bins(2);
        HANDLE_ERROR(cudaMemcpy(&n_bins.at(0),d_n_bins,2*sizeof(int),cudaMemcpyDeviceToHost));

        const size_t size_d_coords = V*n_coords;
        const size_t size_d_n_bins = 2;
        const size_t size_d_coords_min = 2;
        const size_t size_d_coords_max = 2;
        const size_t size_neigh_idx = V*K;
        const size_t size_neigh_dist = V*K;

        int blockSize = 256;
        const int n_bins_x = n_bins[0];
        const int n_bins_y = n_bins[1];
        int numBlocks_V = (V + blockSize - 1) / blockSize;
        int numBlocks_n_bins = (n_bins_x*n_bins_y + blockSize - 1) / blockSize;

        int* d_features_to_bin_on; 
        const size_t size_d_features_to_bin_on = 2;
        HANDLE_ERROR(cudaMalloc(&d_features_to_bin_on, size_d_features_to_bin_on*sizeof(int)));
        HANDLE_ERROR(cudaMemcpy(d_features_to_bin_on,&features_to_bin_on[0],2*sizeof(int),cudaMemcpyHostToDevice));

        float* d_coords_sorted;
        const size_t size_d_coords_sorted = V*n_coords;
        HANDLE_ERROR(cudaMalloc(&d_coords_sorted, size_d_coords_sorted*sizeof(float)));

        int* d_tmp_neigh_idx;
        const size_t size_d_tmp_neigh_idx = V*K;
        HANDLE_ERROR(cudaMalloc(&d_tmp_neigh_idx, size_d_tmp_neigh_idx*sizeof(int)));

        float* d_tmp_neigh_dist;
        const size_t size_d_tmp_neigh_dist = V*K;
        HANDLE_ERROR(cudaMalloc(&d_tmp_neigh_dist, size_d_tmp_neigh_dist*sizeof(float)));

        int *d_bin_idx;
        const size_t size_d_bin_idx = V;
        HANDLE_ERROR(cudaMalloc(&d_bin_idx, size_d_bin_idx*sizeof(int)));

        int *d_help_arr;
        const size_t size_d_help_arr = V;
        HANDLE_ERROR(cudaMalloc(&d_help_arr, size_d_help_arr*sizeof(int)));

        int *d_vtx_bin_assoc;
        const size_t size_d_vtx_bin_assoc = V;
        HANDLE_ERROR(cudaMalloc(&d_vtx_bin_assoc, size_d_vtx_bin_assoc*sizeof(int)));

        int *d_vtx_idx_translation_matrix;
        const size_t size_d_vtx_idx_translation_matrix = V;
        HANDLE_ERROR(cudaMalloc(&d_vtx_idx_translation_matrix, size_d_vtx_idx_translation_matrix*sizeof(int)));

        int *d_backward_vtx_idx_translation_matrix;
        const size_t size_d_backward_vtx_idx_translation_matrix = V;
        HANDLE_ERROR(cudaMalloc(&d_backward_vtx_idx_translation_matrix, size_d_backward_vtx_idx_translation_matrix*sizeof(int)));

        int* d_zero_counters;
        const size_t size_d_zero_counters = n_bins_x*n_bins_y;
        HANDLE_ERROR(cudaMalloc(&d_zero_counters, size_d_zero_counters*sizeof(int)));

        size_t size_bin_neighbours = 9*n_bins_x*n_bins_y;
        int* bin_neighbours = (int *)malloc(size_bin_neighbours*sizeof(int));
        int* d_bin_neighbours;
        const size_t size_d_bin_neighbours = 9*n_bins_x*n_bins_y;
        HANDLE_ERROR(cudaMalloc(&d_bin_neighbours, size_d_bin_neighbours*sizeof(int)));

        int* n_vtx_per_bin = (int *)malloc(n_bins_x*n_bins_y*sizeof(int));
        int* d_n_vtx_per_bin;
        const size_t size_d_n_vtx_per_bin = n_bins_x*n_bins_y;
        HANDLE_ERROR(cudaMalloc(&d_n_vtx_per_bin, size_d_n_vtx_per_bin*sizeof(int)));

        int* n_vtx_per_bin_cumulative = (int *)malloc((n_bins_x*n_bins_y+1)*sizeof(int));
        int* d_n_vtx_per_bin_cumulative;
        const size_t size_d_n_vtx_per_bin_cumulative = (n_bins_x*n_bins_y+1);
        HANDLE_ERROR(cudaMalloc(&d_n_vtx_per_bin_cumulative, size_d_n_vtx_per_bin_cumulative*sizeof(int)));

        int *d_farthest_neighbour;
        const size_t size_d_farthest_neighbour = V;
        HANDLE_ERROR(cudaMalloc(&d_farthest_neighbour, size_d_farthest_neighbour*sizeof(int)));

        // set default values global
        gpu::set_defaults<<<numBlocks_V,blockSize>>>(d_tmp_neigh_idx, size_d_tmp_neigh_idx, d_tmp_neigh_dist, size_d_tmp_neigh_dist,V, K);
        gpu::set_defaults<<<numBlocks_V,blockSize>>>(d_farthest_neighbour, size_d_farthest_neighbour, size_d_farthest_neighbour, -1);

        cpu::create_bin_neighbours_list(n_bins_x,n_bins_y,bin_neighbours,size_bin_neighbours);
        HANDLE_ERROR(cudaMemcpy(d_bin_neighbours,bin_neighbours,9*n_bins_x*n_bins_y*sizeof(int),cudaMemcpyHostToDevice));

        // needed since d_row_splits is only accessible in GPU memory
        std::vector<int> cpu_rowsplits(n_rs);
        HANDLE_ERROR(cudaMemcpy(&cpu_rowsplits.at(0),d_row_splits,n_rs*sizeof(int),cudaMemcpyDeviceToHost));

        for(size_t j_rs=0;j_rs<n_rs-1;j_rs++){ //n_rs-1 important!
            const int nvert_rs = cpu_rowsplits.at(j_rs+1) - cpu_rowsplits.at(j_rs);

            const int numBlocks_Vrs = (nvert_rs + blockSize - 1) / blockSize;
            const size_t start_vert = cpu_rowsplits.at(j_rs);
            const size_t end_vert = cpu_rowsplits.at(j_rs+1);

            // set default values per row split
            gpu::set_defaults<<<numBlocks_n_bins ,blockSize>>>(d_n_vtx_per_bin, size_d_n_vtx_per_bin, size_d_n_vtx_per_bin, 0);
            gpu::set_defaults<<<numBlocks_n_bins ,blockSize>>>(d_zero_counters, size_d_zero_counters, size_d_zero_counters,0);
            gpu::set_defaults<<<numBlocks_n_bins ,blockSize>>>(d_n_vtx_per_bin_cumulative, size_d_n_vtx_per_bin_cumulative, size_d_n_vtx_per_bin_cumulative, 0);
            HANDLE_ERROR(cudaDeviceSynchronize());

            // ********************************
            // STEP 2: fill in auxiliary arrays
            // ********************************

            gpu::constructPhaseSpaceBins<<<numBlocks_Vrs, blockSize>>>(
                    d_coords, size_d_coords, n_coords, start_vert, end_vert, d_n_bins, size_d_n_bins, d_coords_min, size_d_coords_min, d_coords_max, size_d_coords_max, d_features_to_bin_on, size_d_features_to_bin_on, d_n_vtx_per_bin, size_d_n_vtx_per_bin, d_bin_idx, size_d_bin_idx);
            HANDLE_ERROR(cudaDeviceSynchronize());

            gpu::calculate_n_vtx_per_bin_cumulative<<<1,1>>>(n_bins_x*n_bins_y, d_n_vtx_per_bin, size_d_n_vtx_per_bin, d_n_vtx_per_bin_cumulative, size_d_n_vtx_per_bin_cumulative);
            HANDLE_ERROR(cudaDeviceSynchronize());

            gpu::prepare_translation_matrix<<<numBlocks_Vrs, blockSize>>>(start_vert, end_vert, d_n_vtx_per_bin_cumulative, size_d_n_vtx_per_bin_cumulative, d_bin_idx, size_d_bin_idx, d_zero_counters, size_d_zero_counters, d_vtx_idx_translation_matrix, size_d_vtx_idx_translation_matrix);
            HANDLE_ERROR(cudaDeviceSynchronize());

            gpu::prepare_backward_translation_matrix<<<numBlocks_Vrs, blockSize>>>(start_vert, end_vert, d_vtx_idx_translation_matrix, size_d_vtx_idx_translation_matrix, d_backward_vtx_idx_translation_matrix, size_d_backward_vtx_idx_translation_matrix);
            HANDLE_ERROR(cudaDeviceSynchronize());

            gpu::translate_2d_matrix<<<numBlocks_Vrs,blockSize>>>(start_vert, end_vert, n_coords, d_vtx_idx_translation_matrix, size_d_vtx_idx_translation_matrix, d_coords, size_d_coords, d_coords_sorted, size_d_coords_sorted);
            HANDLE_ERROR(cudaDeviceSynchronize());

            HANDLE_ERROR(cudaMemcpy(&n_vtx_per_bin_cumulative[0],d_n_vtx_per_bin_cumulative,(n_bins_x*n_bins_y+1)*sizeof(int),cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(&n_vtx_per_bin[0],d_n_vtx_per_bin,n_bins_x*n_bins_y*sizeof(int),cudaMemcpyDeviceToHost));

            // write bin_vtx_assoc array
            for (int i=0; i<n_bins_x*n_bins_y; i++){
                int numBlocks_tmp = (n_vtx_per_bin[i] + blockSize - 1) / blockSize;
                gpu::set_defaults<<<numBlocks_tmp ,blockSize>>>(d_help_arr, size_d_help_arr, n_vtx_per_bin[i], i);
                HANDLE_ERROR(cudaDeviceSynchronize());
                int start_index = n_vtx_per_bin_cumulative[i]+start_vert;
                int end_index = n_vtx_per_bin_cumulative[i+1]+start_vert;
                numBlocks_tmp = (end_index-start_index + blockSize - 1) / blockSize;
                gpu::insert_into_array<<<numBlocks_V,blockSize>>>(d_help_arr, size_d_help_arr, d_vtx_bin_assoc, size_d_vtx_bin_assoc, start_index,end_index);
                HANDLE_ERROR(cudaDeviceSynchronize());
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
                    size_d_coords_sorted,
                    n_coords,
                    K,
                    counter,
                    d_bin_neighbours,
                    size_d_bin_neighbours,
                    d_n_vtx_per_bin_cumulative,
                    size_d_n_vtx_per_bin_cumulative,
                    d_vtx_bin_assoc,
                    size_d_vtx_bin_assoc,
                    d_farthest_neighbour,
                    size_d_farthest_neighbour,
                    d_tmp_neigh_idx,
                    size_d_tmp_neigh_idx,
                    d_tmp_neigh_dist,
                    size_d_tmp_neigh_dist
                    );
                HANDLE_ERROR(cudaDeviceSynchronize());
                counter++;
            }
        }

        // *******************************************************
        // STEP 4: backward-propagate cordinate neighbour matrices
        // *******************************************************

        gpu::translate_2d_matrix<<<numBlocks_V,blockSize>>>(0, V, K, d_backward_vtx_idx_translation_matrix, size_d_backward_vtx_idx_translation_matrix, d_tmp_neigh_idx, size_d_tmp_neigh_idx, neigh_idx, size_neigh_idx);
        HANDLE_ERROR(cudaDeviceSynchronize());
        gpu::translate_ind_matrix<<<numBlocks_V,blockSize>>>(0, V, K, d_vtx_idx_translation_matrix, size_d_vtx_idx_translation_matrix, neigh_idx, size_neigh_idx);
        gpu::translate_2d_matrix<<<numBlocks_V,blockSize>>>(0, V, K, d_backward_vtx_idx_translation_matrix, size_d_backward_vtx_idx_translation_matrix, d_tmp_neigh_dist, size_d_tmp_neigh_dist, neigh_dist, size_neigh_idx);
        HANDLE_ERROR(cudaDeviceSynchronize());
        gpu::translate_matrices_to_make_selfindex_first<<<numBlocks_V,blockSize>>>(V, K, neigh_idx, size_neigh_idx, neigh_dist, size_neigh_dist);
        HANDLE_ERROR(cudaDeviceSynchronize());

        // *****************************
        // STEP 5: free allocated memory
        // *****************************
        // Free memory
        HANDLE_ERROR(cudaFree(d_coords_sorted));
        HANDLE_ERROR(cudaFree(d_n_vtx_per_bin));
        free(n_vtx_per_bin);
        HANDLE_ERROR(cudaFree(d_n_vtx_per_bin_cumulative));
        free(n_vtx_per_bin_cumulative);
        HANDLE_ERROR(cudaFree(d_bin_idx));
        HANDLE_ERROR(cudaFree(d_vtx_idx_translation_matrix));
        HANDLE_ERROR(cudaFree(d_backward_vtx_idx_translation_matrix));
        HANDLE_ERROR(cudaFree(d_zero_counters));
        free(bin_neighbours);
        HANDLE_ERROR(cudaFree(d_bin_neighbours));
        HANDLE_ERROR(cudaFree(d_help_arr));
        HANDLE_ERROR(cudaFree(d_vtx_bin_assoc));
        HANDLE_ERROR(cudaFree(d_farthest_neighbour));
        HANDLE_ERROR(cudaFree(d_tmp_neigh_dist));
        HANDLE_ERROR(cudaFree(d_tmp_neigh_idx));
        HANDLE_ERROR(cudaFree(d_features_to_bin_on));
    }
};


template struct SlicingKnnOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA
