//#define GOOGLE_CUDA 1


#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "new3_knn_kernel.h"
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
        const bool tf_compat,
        const int V,
        const int K
){
    for(size_t i = 0 ; i < V*K ; i += 1){
        neigh_idx[i] = -1;
        neigh_dist[i] = 0;
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


__global__
void set_defaults(
        int *neigh_idx,
        float *neigh_dist,
        const bool tf_compat,
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
void prepare_indices(size_t start_vert, size_t end_vert, size_t bin_index, int* bin_idx, int* counters, int* tmp_indices){
    int index = blockIdx.x * blockDim.x + threadIdx.x + start_vert;
    int stride = blockDim.x * gridDim.x;
    for(size_t i_v = index; i_v < end_vert; i_v += stride){
        if (bin_idx[i_v] != bin_index)
            return;
        int index_to_fill = atomicAdd(&counters[0],1);
        tmp_indices[index_to_fill] = i_v;
    }
}

__global__
void clean_indices(size_t bin_index, int* tmp_indices, int *n_vtx_per_bin, int defValue){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(size_t i_v = index; i_v < n_vtx_per_bin[bin_index]; i_v += stride){
        tmp_indices[i_v] = defValue;
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
void calculate2dDistanceToTheBinEdges(
    const int start_vert,
    const int end_vert,
    const float *d_coords, // vertices coords
    const size_t n_coords, // # of dims
    const float* coords_2d_bins, // coords of bins
    const int n_bins_x, // number of bins X
    const int n_bins_y, // number of bins Y
    const int* bin_idx, // in which bin which vertex is
    const int* bin_bin_dist_idx, // bin-to-bin distance, indices
    const float* bin_bin_dist, // bin-to-bin distance
    int* bin_idxs_ordered_by_dist_to_the_vtx, // output
    float* bin_vtx_dist_ordered_by_dist_to_the_vtx // output
    ){

    int n_input_vertices = end_vert - start_vert;
    int n_bins = n_bins_x*n_bins_y;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(size_t i_v = index; i_v < n_input_vertices; i_v += stride){
        
        const int bin_index = bin_idx[i_v];
        // int bin_index_x = bin_index%n_bins_x;
        // int bin_index_y = (int)bin_index/n_bins_x;
        float x = d_coords[I2D(i_v,0,n_coords)];
        float y = d_coords[I2D(i_v,1,n_coords)];
                
        // copy matrices first
        for (int i=0; i<n_bins; i++){
            bin_idxs_ordered_by_dist_to_the_vtx[i_v*n_bins+i] = bin_bin_dist_idx[bin_index*n_bins+i];
            bin_vtx_dist_ordered_by_dist_to_the_vtx[i_v*n_bins+i] = bin_bin_dist[bin_index*n_bins+i];
            if (i==0){
                bin_vtx_dist_ordered_by_dist_to_the_vtx[i_v*n_bins+i] = 0.0;
            }
            // if (i==0)
            //     printf("DEVUG: i_v: %d; bin_index: %d; dist1: %f; dist2: %f\n",i_v, bin_index, bin_bin_dist[bin_index*n_bins+i], bin_vtx_dist_ordered_by_dist_to_the_vtx[i_v*n_bins+i]);
        }
                
        // count distance between vertex and neighbour bins with radius==1
        for (int j=1; j<9; j++){
            if (bin_vtx_dist_ordered_by_dist_to_the_vtx[i_v*n_bins+j]!=0.0)
                break;
            
            int target_bin = bin_idxs_ordered_by_dist_to_the_vtx[i_v*n_bins+j];
            
            float target_bin_coords[4];
            target_bin_coords[0] = coords_2d_bins[4*target_bin];
            target_bin_coords[1] = coords_2d_bins[4*target_bin+1];
            target_bin_coords[2] = coords_2d_bins[4*target_bin+2];
            target_bin_coords[3] = coords_2d_bins[4*target_bin+3];
            
            // find the closest distance to the target bin
            float distance_to_bin = 0.0;
            if ((x>target_bin_coords[0] && x<target_bin_coords[2]) || 
               (y>target_bin_coords[1] && y<target_bin_coords[3])){
               size_t iDim = 0;
               if (x>target_bin_coords[0] && x<target_bin_coords[2])
                   iDim = 1;

               float lowBinEdge = target_bin_coords[iDim];
               float highBinEdge = target_bin_coords[iDim+2];
               float d1 = pow((d_coords[I2D(i_v,iDim,n_coords)] - lowBinEdge),2);
               float d2 = pow((d_coords[I2D(i_v,iDim,n_coords)] - highBinEdge),2);
               distance_to_bin = (d1<d2) ? d1 : d2;
            }
            else{ // diagonal bin
               float bin_coord_x = target_bin_coords[0]; // left side
               float bin_coord_y = target_bin_coords[1]; // bottom side
               if (x>target_bin_coords[2])
                   bin_coord_x = target_bin_coords[2]; // right side
               if (y>target_bin_coords[3])
                   bin_coord_y = target_bin_coords[3]; // top side
               float pointCoord[2] = {bin_coord_x, bin_coord_y};
               distance_to_bin = calculate2dDistanceToThePoint(pointCoord, i_v, d_coords, n_coords);
            }
            bin_vtx_dist_ordered_by_dist_to_the_vtx[i_v*n_bins+j] = distance_to_bin;
        }
    }
}

__global__
void produce_all_possible_pairs(const size_t index_x, const size_t index_y, 
                                const size_t n_bins_x, const size_t n_bins_y,
                                int* out_x, int* out_y){
    // TODO improve it
    int max_radius = (n_bins_x > n_bins_y) ? n_bins_x : n_bins_y;

    int counter = 0;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int radius=1 + index; radius<max_radius; radius += stride){
        
        int new_index_x = index_x-radius;
        if (new_index_x>=0 && new_index_x<n_bins_x){
            int tmp_radius = -radius;
            while (tmp_radius<=radius){
                int new_index_y = index_y+tmp_radius;
                if (new_index_y>=0 && new_index_y<n_bins_y){
                    out_x[counter] = new_index_x;
                    out_y[counter] = new_index_y;
                    counter++;
                }
                tmp_radius++;
            }
        }
        new_index_x = index_x+radius;
        if (new_index_x>=0 && new_index_x<n_bins_x){
            int tmp_radius = -radius;
            while (tmp_radius<=radius){
                int new_index_y = index_y+tmp_radius;
                if (new_index_y>=0 && new_index_y<n_bins_y){
                    out_x[counter] = new_index_x;
                    out_y[counter] = new_index_y;
                    counter++;
                }
                tmp_radius++;
            }
        }

        int new_index_y = index_y-radius;
        if (new_index_y>=0 && new_index_y<n_bins_y){
            int tmp_radius = -radius+1;
            while (tmp_radius<=(radius-1)){
                int new_index_x = index_x+tmp_radius;
                if (new_index_x>=0 && new_index_x<n_bins_x){
                    out_x[counter] = new_index_x;
                    out_y[counter] = new_index_y;
                    counter++;
                }
                tmp_radius++;
            }
        }
        new_index_y = index_y+radius;
        if (new_index_y>=0 && new_index_y<n_bins_y){
            int tmp_radius = -radius+1;
            while (tmp_radius<=(radius-1)){
                int new_index_x = index_x+tmp_radius;
                if (new_index_x>=0 && new_index_x<n_bins_x){
                    out_x[counter] = new_index_x;
                    out_y[counter] = new_index_y;
                    counter++;
                }
                tmp_radius++;
            }
        }
    }
}


__global__
void calculate_bin_bin_dist_matrices(int *n_bins, const float* coords_2d_bins,
        float* bin_bin_dist, int* bin_bin_dist_idx){

    const int iBinX =  blockIdx.x * blockDim.x + threadIdx.x;
    if(iBinX >= n_bins[0])
        return;
    const int iBinY =  blockIdx.y * blockDim.y + threadIdx.y;
    if(iBinY >= n_bins[1])
        return;


    const int iBin[2] = {iBinX, iBinY};

    size_t bin_index = I2D(iBin[1], iBin[0], n_bins[0]);
    const float dx = coords_2d_bins[2] - coords_2d_bins[0];
    const float dy = coords_2d_bins[3] - coords_2d_bins[1];
    const bool dy_larger = (dy > dx);

    const size_t max_dNx = ((n_bins[0]-1 - iBin[0]) > iBin[0]) ? n_bins[0]-1 - iBin[0] : iBin[0];
    const size_t max_dNy = ((n_bins[1]-1 - iBin[1]) > iBin[1]) ? n_bins[1]-1 - iBin[1] : iBin[1];
    const size_t max_radius = (max_dNx>max_dNy) ? max_dNx : max_dNy;

    // first in the matrix has to be bin itself
    bin_bin_dist_idx[bin_index*n_bins[0]*n_bins[1]] = bin_index;
    bin_bin_dist[bin_index*n_bins[0]*n_bins[1]] = 0;
    
    int counter = 1;
    int new_index[2];
    int dN[2];

    for (int radius=1; radius<=max_radius; radius += 1){
        int counter2 = 1;
        int bin_shift[3] = {0,-1,1};
        size_t n_bins_to_shift = 3;
        while (counter2<=radius){
            if (counter2 > 1){
                bin_shift[0] = -counter2;
                bin_shift[1] = counter2;
                n_bins_to_shift = 2;
            }
            for (int symmetry_counter=0; symmetry_counter<2; symmetry_counter++){
                for (int tmp_counter=0; tmp_counter<n_bins_to_shift; tmp_counter++){
                    for (int i_sign=-1; i_sign<=1; i_sign += 2){ 
                        if (symmetry_counter==0){
                            dN[(int)(!dy_larger)] = i_sign*radius;
                            dN[(int)dy_larger] = bin_shift[tmp_counter];
                        }
                        else{
                            dN[(int)(!dy_larger)] = bin_shift[tmp_counter];
                            dN[(int)dy_larger] = i_sign*radius;
                            if (std::abs(dN[0]*dN[1])==pow(radius,2)) // avoid double-counting
                                continue;
                        }
                        new_index[0] = iBin[0]+dN[0];
                        new_index[1] = iBin[1]+dN[1];
                        if (new_index[0]<0 || new_index[1]<0 || new_index[0]>=n_bins[0] || new_index[1]>=n_bins[1])
                            continue;
                        int target_bin_index = I2D(new_index[1], new_index[0], n_bins[0]);
                        float dist = 0;

                        // printf("\nbin_index: %d",(int)bin_index);
                        // printf("\ntarget_bin_index: %d",(int)target_bin_index);
                        // printf("\ndN[0]: %d",(int)dN[0]);
                        // printf("\ndN[1]: %d",(int)dN[1]);

                        size_t abs_dNx = std::abs(dN[0]);
                        size_t abs_dNy = std::abs(dN[1]);
                        if (abs_dNx>=2)
                            dist += pow(dx*(abs_dNx-1),2);
                        if (abs_dNy>=2)
                            dist += pow(dx*(abs_dNy-1),2);
                        // dist = sqrt(dist);
                        // printf("\ndist: %f",dist);
                        int out_arr_idx = bin_index*n_bins[0]*n_bins[1] + counter;
                        bin_bin_dist_idx[out_arr_idx] = target_bin_index;
                        bin_bin_dist[out_arr_idx] = dist;
                        counter++;
                    }
                }
            }
            counter2++;
        }
    }

}

__global__
void prepare_vtx_to_process(
    const int start_vert,
    const int end_vert,
    const int* next_bin,
    int* n_vtx_to_process,
    int* vtx_to_process){

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for(size_t i_v = index + start_vert; i_v < end_vert; i_v += stride){
        if (next_bin[i_v]>=0){
            int n_filled = atomicAdd(&n_vtx_to_process[0],1);
            vtx_to_process[n_filled] = i_v;
            // vtx_to_process[atomicAdd(&n_vtx_to_process[0],1)] = i_v;
        }
    }
}

__global__
void perform_kNN_search(
    const float* d_coords_sorted,
    const int n_coords,
    const int K,
    const int n_bins_x,
    const int n_bins_y,
    const int* bin_idxs_ordered_by_dist_to_the_vtx,
    const float* bin_vtx_dist_ordered_by_dist_to_the_vtx,
    const int* bin_idx_sorted,
    const int* n_vtx_per_bin,
    const int* n_vtx_per_bin_cumulative,
    const int* n_vtx_to_process,
    const int* vtx_to_process,
    int* farthest_neighbour,
    int* next_bin,
    int* neigh_idx,
    float* neigh_dist,
    int* n_vtx_finished,
    float max_radius = -1.0 // max. radius to search for neighbours
    ){
    
    // DEBUG
    // int debug_vtx = 1;

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for(size_t i = index; i < n_vtx_to_process[0]; i += stride){
        // if (next_bin[i_v]<0) // all neighbours is already found for this vertex
        //     continue;
        int i_v = vtx_to_process[i];
        
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
                
        int target_bin = bin_idxs_ordered_by_dist_to_the_vtx[i_v*n_bins_x*n_bins_y+next_bin[i_v]];
        
        if((n_vtx_per_bin[target_bin]+nfilled)<K){
            max_neighbours=(n_vtx_per_bin[target_bin]+nfilled); // K or less
        }
        
        float maxdistsq = 0;
        float distance_to_bin = 0;
        
        if (nfilled>1){ // if nfilled==1 - neighbour is the vtx itself
            maxdistsq = neigh_dist[I2D(i_v,farthest_neighbour[i_v],K)];
            distance_to_bin = bin_vtx_dist_ordered_by_dist_to_the_vtx[i_v*n_bins_x*n_bins_y+next_bin[i_v]];
        }

        // if (i_v==debug_vtx )
            // printf("DEBUG: next_bin[%d]: %d; distance_to_bin: %f; maxdistsq: %f\n",next_bin[i_v], target_bin, distance_to_bin,maxdistsq);
        
        bool all_neig_are_fount_for_vtx = false;
        while (distance_to_bin>maxdistsq){
            if ((next_bin[i_v] + 1) < n_bins_x*n_bins_y){
                next_bin[i_v] = next_bin[i_v] + 1;
                target_bin = bin_idxs_ordered_by_dist_to_the_vtx[i_v*n_bins_x*n_bins_y+next_bin[i_v]];
                distance_to_bin = bin_vtx_dist_ordered_by_dist_to_the_vtx[i_v*n_bins_x*n_bins_y+next_bin[i_v]];
                // if (i_v==debug_vtx )
                    // printf("DEBUG: next_bin: %d; distance_to_bin: %f; maxdistsq: %f\n",next_bin[i_v], distance_to_bin,maxdistsq);
            }
            else{
                next_bin[i_v] = -1; // end of the search
                // n_vtx_finished += 1; // TODO: ATOMIC ADD
                atomicAdd(&n_vtx_finished[0], 1); // to avoid race condition
                all_neig_are_fount_for_vtx = true;
                break;
            }
        }
        if (all_neig_are_fount_for_vtx)
            continue;
        
        if (distance_to_bin<=maxdistsq){ // condition is "<=" and not "<" to take into account when maxdistsq and distance_to_bin are equal to 0 (their init values)
            int target_bin_start_vert = n_vtx_per_bin_cumulative[target_bin];
            int target_bin_end_vert = n_vtx_per_bin_cumulative[target_bin+1];

            // if (i_v==debug_vtx )
            //     printf("DEBUG: target_bin_start_vert: %d; target_bin_end_vert: %d\n",target_bin_start_vert,target_bin_end_vert);

            for(size_t j_v=target_bin_start_vert;j_v<target_bin_end_vert;j_v++){
                //fill up
                float distsq = calculateDistance(i_v,j_v,d_coords_sorted,n_coords);

                // if (i_v==debug_vtx ){
                //     printf("DEBUG: j_v: %d; distsq: %f\n",j_v,distsq);
                //     printf("j_v: %d; coords: %f %f %f %f\n", j_v, d_coords_sorted[j_v*4],  d_coords_sorted[j_v*4+1], d_coords_sorted[j_v*4+2], d_coords_sorted[j_v*4+3]);
                // }

                if(nfilled<max_neighbours && (max_radius<=0 || max_radius>=distsq)){
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
            }// loop through vertices
        }
        if ((next_bin[i_v] + 1) < n_bins_x*n_bins_y){
            next_bin[i_v] = next_bin[i_v] + 1;
        }
        else{
            next_bin[i_v] = -1;
            // n_vtx_finished += 1; // TODO: ATOMIC ADD
            atomicAdd(&n_vtx_finished[0], 1); // to avoid race condition
        }
        // if (i_v==debug_vtx )
        //         printf("DEBUG: next_bin: %d; farthest_neigh dist: %f\n",next_bin[i_v],neigh_dist[I2D(i_v,farthest_neighbour[i_v],K)]);
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
struct New3KnnOpFunctor<GPUDevice, dummy> {
    void operator()(const GPUDevice& d,

            const float *d_coords, // accessible only on GPU!!!
            const int* d_row_splits, // accessible only on GPU!!!
            int *neigh_idx,
            float *neigh_dist,

            const int V, // # of vertices
            const int K, // # of neighbours to be found
            const int n_coords,

            const int n_rs,
            const bool tf_compat,
            const float max_radius,
            const int n_bins_x,
            const int n_bins_y) {
        
        // *******************************************
        // STEP 1: memmory allocation and set defaults
        // *******************************************
        int blockSize = 256;
        int numBlocks_V = (V + blockSize - 1) / blockSize;
        int numBlocks_V_K = (V*K + blockSize - 1) / blockSize;
        int numBlocks_n_bins = (n_bins_x*n_bins_y + blockSize - 1) / blockSize;
        int numBlocks_n_bins_sq = (n_bins_x*n_bins_y*n_bins_x*n_bins_y + blockSize - 1) / blockSize;
        // TODO FIXME SOME FUNCTIONS DON'T SUPPORT SPLITS FOR TIME BEING...
        // const size_t start_vert = d_row_splits[0];
        // const size_t end_vert = d_row_splits[1];
        // const size_t V = end_vert-start_vert;
        const size_t start_vert = 0;
        const size_t end_vert = V;

        float* coords_2d_bins; // one need to do cuda memory allocation for arrays!
        cudaMallocManaged(&coords_2d_bins, 4*n_bins_x*n_bins_y*sizeof(float));

        int* n_vtx_per_bin;
        cudaMallocManaged(&n_vtx_per_bin, n_bins_x*n_bins_y*sizeof(int));

        int* n_vtx_per_bin_cumulative;
        cudaMallocManaged(&n_vtx_per_bin_cumulative, (n_bins_x*n_bins_y+1)*sizeof(int));

        int *bin_idx;
        cudaMallocManaged(&bin_idx, V*sizeof(int));

        int *bin_idx_sorted ;
        cudaMallocManaged(&bin_idx_sorted , V*sizeof(int));

        int *vtx_idx_translation_matrix;
        cudaMallocManaged(&vtx_idx_translation_matrix, V*sizeof(int));

        int *backward_vtx_idx_translation_matrix;
        cudaMallocManaged(&backward_vtx_idx_translation_matrix, V*sizeof(int));

        float *d_coords_sorted;
        cudaMallocManaged(&d_coords_sorted, n_coords*V*sizeof(float));

        int *bin_bin_dist_idx;
        cudaMallocManaged(&bin_bin_dist_idx, n_bins_x*n_bins_y*n_bins_x*n_bins_y*sizeof(int));

        float *bin_bin_dist;
        cudaMallocManaged(&bin_bin_dist, n_bins_x*n_bins_y*n_bins_x*n_bins_y*sizeof(float));

        float *bin_vtx_dist_ordered_by_dist_to_the_vtx;
        cudaMallocManaged(&bin_vtx_dist_ordered_by_dist_to_the_vtx, V*n_bins_x*n_bins_y*sizeof(float));

        int *bin_idxs_ordered_by_dist_to_the_vtx;
        cudaMallocManaged(&bin_idxs_ordered_by_dist_to_the_vtx, V*n_bins_x*n_bins_y*sizeof(int));

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

        int *neigh_idx_tmp;
        cudaMallocManaged(&neigh_idx_tmp, V*K*sizeof(int));

        float *neigh_dist_tmp;
        cudaMallocManaged(&neigh_dist_tmp, V*K*sizeof(float));

        int *next_bin;
        cudaMallocManaged(&next_bin, V*sizeof(int));

        int *farthest_neighbour;
        cudaMallocManaged(&farthest_neighbour, V*sizeof(int));

        int *vtx_to_process;
        cudaMallocManaged(&vtx_to_process, V*sizeof(int));

        int *n_vtx_finished;
        cudaMallocManaged(&n_vtx_finished, sizeof(int));

        int *n_vtx_to_process;
        cudaMallocManaged(&n_vtx_to_process, sizeof(int));

        // ********************************
        // STEP 2: fill in auxiliary arrays
        // ********************************
        // printf("\n\n*** STEP 2: fill in auxiliary arrays ***\n");

        // set default values
        gpu::set_defaults<<<1,1>>>(n_vtx_finished, 1, 0);
        gpu::set_defaults<<<numBlocks_V,blockSize>>>(neigh_idx, neigh_dist, tf_compat, V, K);
        gpu::set_defaults<<<numBlocks_n_bins ,blockSize>>>(n_vtx_per_bin, n_bins_x*n_bins_y, 0);
        gpu::set_defaults<<<numBlocks_n_bins ,blockSize>>>(zero_counters, n_bins_x*n_bins_y, 0);

        gpu::set_defaults<<<numBlocks_n_bins_sq,blockSize>>>(bin_bin_dist_idx, n_bins_x*n_bins_y*n_bins_x*n_bins_y, 0);
        gpu::set_defaults<<<numBlocks_n_bins_sq,blockSize>>>(bin_bin_dist, n_bins_x*n_bins_y*n_bins_x*n_bins_y, (float)0.0);
        gpu::set_defaults<<<numBlocks_V_K,blockSize>>>(neigh_idx_tmp, V*K, -1);
        gpu::set_defaults<<<numBlocks_V_K,blockSize>>>(neigh_dist_tmp, V*K, (float)0.0);
        gpu::set_defaults<<<numBlocks_V,blockSize>>>(next_bin, V, 0);
        gpu::set_defaults<<<numBlocks_V,blockSize>>>(farthest_neighbour, V, -1);
        n_vtx_to_process[0] = V;
        gpu::init_incremented_values<<<numBlocks_V,blockSize>>>(start_vert, end_vert, vtx_to_process);
        cudaDeviceSynchronize();

        // find min/max of 1st and 2nd coords
        gpu::find_min_max_2d_init<<<1,1>>>(d_coords,coord_min,coord_max);
        cudaDeviceSynchronize();
        gpu::find_min_max_2d<<<1,2>>>(d_coords,n_coords,start_vert,end_vert,coord_min,coord_max);
        cudaDeviceSynchronize();
        coord_min[0] -= (coord_max[0] - coord_min[0])*0.001;
        coord_max[0] += (coord_max[0] - coord_min[0])*0.001;
        coord_min[1] -= (coord_max[1] - coord_min[1])*0.001;
        coord_max[1] += (coord_max[1] - coord_min[1])*0.001;

        // printf("coord_min:");
        // cpu::print_2d_matrix(coord_min, 1, 0,2);
        // printf("coord_max:");
        // cpu::print_2d_matrix(coord_max, 1, 0,2);
        
        dim3 numblocks_2d(n_bins_x/32+1,n_bins_y/32+1);
        dim3 threadsperblock_2d(32,32);
        gpu::get_bin_coords<<<numblocks_2d,threadsperblock_2d,0,d.stream()>>>(coord_min, coord_max, coords_2d_bins, n_bins_x, n_bins_y);
        cudaDeviceSynchronize();

        // printf("\ncoords_2d_bins:\n");
        // cpu::print_2d_matrix(coords_2d_bins, 4, 0,4*n_bins_x*n_bins_y);
       
        gpu::constructPhaseSpaceBins<<<numBlocks_V, blockSize>>>(
                d_coords, n_coords, start_vert, end_vert, n_bins, coord_min, coord_max,
                n_vtx_per_bin, bin_idx);
        cudaDeviceSynchronize();

        // printf("\nbin_idx:\n");
        // cpu::print_array(bin_idx,start_vert, end_vert, true);

        // printf("\nn_vtx_per_bin:\n");
        // cpu::print_array(n_vtx_per_bin,0,n_bins_x*n_bins_y,true);
     
        // calculate n_vtx_per_bin_cumulative
        cpu::calculate_n_vtx_per_bin_cumulative(n_bins_x*n_bins_y, n_vtx_per_bin, n_vtx_per_bin_cumulative);
        cudaDeviceSynchronize();

        // printf("\nn_vtx_per_bin_cumulative:\n");
        // cpu::print_array(n_vtx_per_bin_cumulative,0,n_bins_x*n_bins_y+1,true);

        gpu::prepare_translation_matrix<<<numBlocks_V, blockSize>>>(start_vert, end_vert, n_vtx_per_bin_cumulative, bin_idx, zero_counters, vtx_idx_translation_matrix);
        cudaDeviceSynchronize();
        gpu::set_defaults<<<numBlocks_n_bins,blockSize>>>(zero_counters, n_bins_x*n_bins_y, 0);
        cudaDeviceSynchronize();
        gpu::prepare_backward_translation_matrix<<<numBlocks_V, blockSize>>>(start_vert, end_vert, vtx_idx_translation_matrix, backward_vtx_idx_translation_matrix);
        cudaDeviceSynchronize();

        // printf("\nvtx_idx_translation_matrix:\n");
        // cpu::print_array(vtx_idx_translation_matrix,start_vert, end_vert,true);

        // resort coordinates
        gpu::translate_2d_matrix<<<numBlocks_V,blockSize>>>(end_vert-start_vert, n_coords, vtx_idx_translation_matrix, d_coords, d_coords_sorted);
        gpu::translate_2d_matrix<<<numBlocks_V,blockSize>>>(end_vert-start_vert, 1, vtx_idx_translation_matrix, bin_idx, bin_idx_sorted);
        // synchronize threads
        cudaDeviceSynchronize();

        // printf("\nbin_idx_sorted:\n");
        // cpu::print_array(bin_idx_sorted,start_vert, end_vert, true);

        // printf("\nd_coords:\n");
        // gpu::print_2d_matrix<<<1,1>>>(d_coords, n_coords, n_coords*start_vert, n_coords*end_vert);
        cudaDeviceSynchronize();

        // printf("\nd_coords_sorted:\n");
        // cpu::print_2d_matrix(d_coords_sorted, n_coords, n_coords*start_vert, n_coords*end_vert);

        // ***********************
        // STEP 3: find neighbours
        // ***********************
        // printf("\n\n*** STEP 3: find neighbours ***\n");
    
        gpu::calculate_bin_bin_dist_matrices<<<numblocks_2d,threadsperblock_2d>>>(n_bins, coords_2d_bins, bin_bin_dist, bin_bin_dist_idx);
        cudaDeviceSynchronize();

        // printf("\nbin_bin_dist:\n");
        // cpu::print_2d_matrix(bin_bin_dist, n_bins_x*n_bins_y, 0,n_bins_x*n_bins_y*n_bins_x*n_bins_y);

        // printf("\nbin_bin_dist_idx:\n");
        // cpu::print_2d_matrix(bin_bin_dist_idx, n_bins_x*n_bins_y, 0,n_bins_x*n_bins_y*n_bins_x*n_bins_y, true);

        gpu::calculate2dDistanceToTheBinEdges<<<numblocks_2d,threadsperblock_2d>>>(
            start_vert,
            end_vert,
            d_coords_sorted, // vertices coords
            n_coords, // # of dims
            coords_2d_bins, // coords of bins
            n_bins_x, // number of bins
            n_bins_y, // number of bins
            bin_idx_sorted, // in which bin which vertex is
            bin_bin_dist_idx, // bin-to-bin distance, indices
            bin_bin_dist, // bin-to-bin distance
            bin_idxs_ordered_by_dist_to_the_vtx, // output
            bin_vtx_dist_ordered_by_dist_to_the_vtx // output
		);
        cudaDeviceSynchronize();

        // printf("bin_idxs_ordered_by_dist_to_the_vtx:\n");
        // cpu::print_2d_matrix(bin_idxs_ordered_by_dist_to_the_vtx, n_bins_x*n_bins_y, 0,n_bins_x*n_bins_y*V, true);
        
        // printf("\nbin_idx_sorted:");
        // cpu::print_2d_matrix(bin_idx_sorted, 1, 0,V, true);

        int counter = 0;
        int N_ITER = n_bins_x*n_bins_y;
        while (counter<N_ITER && n_vtx_finished[0] < V){
            // printf("\nITERATION #%d\n************\n", counter);
            // printf("n_vtx_to_process:");
            // cpu::print_2d_matrix(n_vtx_to_process, 1, 0,1, true);
            // printf("vtx_to_process:");
            // gpu::print_2d_matrix<<<1,1>>>(vtx_to_process, 1, 0,n_vtx_to_process[0], true);
            // cpu::print_2d_matrix(vtx_to_process, 1, 0,V, true);
            // printf("next_bin:");
            // cpu::print_2d_matrix(next_bin, 1, 0,V, true);

            int numBlocks_n_vtx_left = (n_vtx_to_process[0] + blockSize - 1) / blockSize;


            gpu::perform_kNN_search<<<numBlocks_n_vtx_left,blockSize>>>(
            // gpu::perform_kNN_search<<<1,1>>>(
                d_coords_sorted,
                n_coords,
                K,
                n_bins_x,
                n_bins_y,
                bin_idxs_ordered_by_dist_to_the_vtx,
                bin_vtx_dist_ordered_by_dist_to_the_vtx,
                bin_idx_sorted,
                n_vtx_per_bin,
                n_vtx_per_bin_cumulative,
                n_vtx_to_process,
                vtx_to_process,
                farthest_neighbour,
                next_bin,
                neigh_idx_tmp,
                neigh_dist_tmp,
                n_vtx_finished
                );
            cudaDeviceSynchronize();
            gpu::set_defaults<<<1,1>>>(n_vtx_to_process, 1, 0);
            gpu::prepare_vtx_to_process<<<numBlocks_V,blockSize>>>(
            // gpu::prepare_vtx_to_process<<<numBlocks_V,blockSize>>>(
                start_vert,
                end_vert,
                next_bin,
                n_vtx_to_process,
                vtx_to_process
                    );
            cudaDeviceSynchronize();
            counter++;

            // printf("neigh_idx_tmp:");
            // gpu::print_2d_matrix<<<1,1>>>(neigh_idx_tmp, K, 0,V*K, true);
            // cudaDeviceSynchronize();

            // DEBUG PRINTOUT
            // printf("n_vtx_finished: %d\n", n_vtx_finished[0]);
        }

        printf("***********\nN_ITER: %d\n***********\n", counter);


        gpu::translate_2d_matrix<<<numBlocks_V,blockSize>>>(end_vert-start_vert, K, backward_vtx_idx_translation_matrix, neigh_idx_tmp, neigh_idx);
        gpu::translate_2d_matrix<<<numBlocks_V,blockSize>>>(end_vert-start_vert, K, backward_vtx_idx_translation_matrix, neigh_dist_tmp, neigh_dist);
        cudaDeviceSynchronize();
        gpu::translate_content_of_2d_matrix<<<numBlocks_V_K,blockSize>>>(K*(end_vert-start_vert), vtx_idx_translation_matrix, neigh_idx, neigh_idx);
        cudaDeviceSynchronize();

        // Free memory
        cudaFree(coords_2d_bins);
        cudaFree(n_vtx_per_bin);
        cudaFree(n_vtx_per_bin_cumulative);
        cudaFree(bin_idx);
        cudaFree(bin_idx_sorted);
        cudaFree(vtx_idx_translation_matrix);
        cudaFree(backward_vtx_idx_translation_matrix);
        cudaFree(d_coords_sorted);
        cudaFree(bin_bin_dist_idx);
        cudaFree(bin_bin_dist);
        cudaFree(bin_vtx_dist_ordered_by_dist_to_the_vtx);
        cudaFree(bin_idxs_ordered_by_dist_to_the_vtx);
        cudaFree(n_bins);
        cudaFree(coord_max);
        cudaFree(coord_min);
        cudaFree(zero_counters);
        cudaFree(neigh_idx_tmp);
        cudaFree(neigh_dist_tmp);
        cudaFree(next_bin);
        cudaFree(farthest_neighbour);
        cudaFree(n_vtx_finished);
        cudaFree(vtx_to_process);
        cudaFree(n_vtx_to_process);
    }
};


template struct New3KnnOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA
