//#define GOOGLE_CUDA 1


#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "new5_knn_kernel.h"
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

// __device__
// int findTheBiggestNumberInArraySmalerThanGiven(const int* arr, const int l, const int r, const int x){
    /**
     * Find biggest number in the array that is smaller than given.
     *
     * @param int[] arr: input array to search through
     * @param int l: left index of theinput array to search from
     * @param int r: right index of the input array to search up to
     * @param int x: a given number
     */
//     if (r > l) {
//
//         int mid = l + (r - l) / 2;
//
//         if (mid==0)
//             return mid;
//
//         if (x > arr[mid-1] && x <= arr[mid])
//             return mid-1;
//
//         if (arr[mid] >= x)
//             return findTheBiggestNumberInArraySmalerThanGiven(arr, l, mid - 1, x);
//
//         return findTheBiggestNumberInArraySmalerThanGiven(arr, mid + 1, r, x);
//     }
//     if (r == l) {
//         if (r==0)
//             return 0;
//
//         if (x == arr[r])
//             return r-1;
//
//         if (x > arr[r] && x <= arr[r+1])
//             return r;
//
//         else if (x >= arr[r-1] && x < arr[r])
//             return r-1;
//     }
//     return -1;
// }

__global__
void perform_kNN_search(
    const int start_vert,
    const int end_vert,
    const float* d_coords_sorted,
    const int n_coords,
    const int K,
    const int bin_index_to_use,
    const int* bin_neighbours, // n_bins*9
    const int n_bins,
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
        // int vtx_bin = findTheBiggestNumberInArraySmalerThanGiven(n_vtx_per_bin_cumulative, 0, n_bins, i_v+1); // i_v+1 is needed due to function implementation
        int target_bin = bin_neighbours[bin_index_to_use + 9*vtx_bin];
        if (target_bin<0) // can happen when bin has less than 8 neighbour bins
            continue;
        int target_bin_start_vert = n_vtx_per_bin_cumulative[target_bin];
        int target_bin_end_vert  = n_vtx_per_bin_cumulative[target_bin+1];
        
        if((target_bin_end_vert -target_bin_start_vert +nfilled)<K){
            max_neighbours=(target_bin_end_vert -target_bin_start_vert +nfilled); // K or less
        }
        
        // printf("DEBUG: i_v: %d; coords: (%f,%f); vtx_bin: %d; target_bin: %d \n", i_v, d_coords_sorted[I2D(i_v,0,n_coords)], d_coords_sorted[I2D(i_v,1,n_coords)], vtx_bin, target_bin);
        
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
struct New5KnnOpFunctor<GPUDevice, dummy> {
    void operator()(const GPUDevice& d,

            const float *d_coords_sorted, // accessible only on GPU!!!
            int *neigh_idx,
            float *neigh_dist,

            const int V, // # of vertices
            const int K, // # of neighbours to be found
            const int n_coords,

            const int n_bins,
            const int* n_vtx_per_bin_cumulative, // size: n_bins_x*n_bins_y+1
            const int* bin_neighbours, // size: 9*n_bins_x*n_bins_y, bin itself + up to 8 neighbour bins
            const int* vtx_bin_assoc // size: V
            ) {
        
        // ******************************************
        // STEP 1: memory allocation and set defaults
        // ******************************************
        int blockSize = 256;
        int numBlocks_V = (V + blockSize - 1) / blockSize;

        const size_t start_vert = 0;
        const size_t end_vert = V;

        int *farthest_neighbour;
        cudaMallocManaged(&farthest_neighbour, V*sizeof(int));

        // set default values
        gpu::set_defaults<<<numBlocks_V,blockSize>>>(neigh_idx, neigh_dist, V, K);
        gpu::set_defaults<<<numBlocks_V,blockSize>>>(farthest_neighbour, V, -1);
        cudaDeviceSynchronize();

        // ***********************
        // STEP 2: find neighbours
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
                n_bins,
                n_vtx_per_bin_cumulative,
                vtx_bin_assoc,
                farthest_neighbour,
                neigh_idx,
                neigh_dist
                );
            cudaDeviceSynchronize();
            counter++;
        }

        // *****************************
        // STEP 3: free allocated memory
        // *****************************
        cudaFree(farthest_neighbour);
    }
};


template struct New5KnnOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA
