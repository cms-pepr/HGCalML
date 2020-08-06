//#define GOOGLE_CUDA 1


#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "new_knn_kernel.h"
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
        int *d_indices,
        float *d_dist,
        const bool tf_compat,
        const int n_vert,
        const int n_neigh
){
    for(size_t i = 0 ; i < n_vert*n_neigh ; i += 1){
        d_indices[i] = -1;
        d_dist[i] = 0;
    }
}

void print_array(
        const float *in_arr,
        const size_t start,
        const size_t end
){
    for(size_t i = start ; i < end ; i += 1){
        float tmp_val = in_arr[i];
        printf("i: %d;\t%f\n", i, tmp_val);
    }
}
void print_array(
        const int *in_arr,
        const size_t start,
        const size_t end
){
    for(size_t i = start ; i < end ; i += 1){
        float tmp_val = in_arr[i];
        printf("i: %d;\t%f\n", i, tmp_val);
    }
}

}// cpu namespace

namespace gpu{
__device__
float calculateDistance(size_t i_v, size_t j_v, const float * d_coord, size_t n_coords){
    float distsq=0;
    if(i_v == j_v)
        return 0;
    for(size_t i=0;i<n_coords;i++){
        float dist = d_coord[I2D(i_v,i,n_coords)] - d_coord[I2D(j_v,i,n_coords)];
        distsq += dist*dist;
    }
    return distsq;
}


__device__
int searchLargestDistance(int i_v, float* d_dist, int n_neigh, float& maxdist){
    maxdist=0;
    int maxidx=0;
    for(size_t n=0;n<n_neigh;n++){
        float distsq = d_dist[I2D(i_v,n,n_neigh)];
        if(distsq > maxdist){
            maxdist = distsq;
            maxidx = n;
        }
    }
    return maxidx;
}

__device__
float calculate2dDistanceToThePoint(float *pointCoord, size_t i_v, const float* d_coord, size_t n_coords){
    float distsq=0;
    for(size_t i=0;i<2;i++){
        float dist = d_coord[I2D(i_v,i,n_coords)] - pointCoord[i];
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
        int *d_indices,
        float *d_dist,
        const bool tf_compat,
        const int n_vert,
        const int n_neigh
){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(size_t i = index ; i < n_vert*n_neigh ; i += stride){
        // printf("before: d_indices[%d]: %f\n", i, d_indices[i]);
        d_indices[i] = -1;
        d_dist[i] = 0;
        // printf("after: d_indices[%d]: %f\n", i, d_indices[i]);
    }
    // for(size_t i_v =0 ; i_v < n_vert ; i_v++){
    //     for(size_t n = 0; n < n_neigh; n++){
    //         d_indices[I2D(i_v,n,n_neigh)] = -1;
    //         d_dist[I2D(i_v,n,n_neigh)] = 0;
    //     }
    // }
}

__global__
void print_neighbours(
        const size_t i_v,
        int *d_indices,
        float *d_dist,
        const int n_neigh
){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(size_t i = index ; i < n_neigh ; i += stride){
        printf(" %d (%f)", d_indices[I2D(i_v,i,n_neigh)], d_dist[I2D(i_v,i,n_neigh)]);
    }
}

__global__
void print_array(
        const float *in_arr,
        const size_t start,
        const size_t end
){
    int index = blockIdx.x * blockDim.x + threadIdx.x + start;
    int stride = blockDim.x * gridDim.x;
    for(size_t i = index ; i < end ; i += stride){
        float tmp_val = in_arr[i];
        printf("i: %d;\t%f\n", i, tmp_val);
    }
}
__global__
void print_array(
        const int *in_arr,
        const size_t start,
        const size_t end
){
    int index = blockIdx.x * blockDim.x + threadIdx.x + start;
    int stride = blockDim.x * gridDim.x;
    for(size_t i = index ; i < end ; i += stride){
        float tmp_val = in_arr[i];
        printf("i: %d;\t%f\n", i, tmp_val);
    }
}



__global__
void get_bin_coords(float* min, float* max, float *coords_2d_bins, const size_t n_bins_x, const size_t n_bins_y){

    const size_t iBinX =  blockIdx.x * blockDim.x + threadIdx.x;
    if(iBinX >= n_bins_x)
        return;
    const size_t iBinY =  blockIdx.y * blockDim.y + threadIdx.y;
    if(iBinY >= n_bins_y)
        return;

    // float binEdgesX[n_bins_x+1];
    // float binEdgesY[n_bins_y+1];
    // float *binEdges[2] = {binEdgesX, binEdgesY};
    
    // define phase-space bin edges
    size_t bin_index = I2D(iBinX, iBinY, n_bins_y);
    coords_2d_bins[4*bin_index] = min[0] + iBinX*(max[0] - min[0])/n_bins_x;
    coords_2d_bins[4*bin_index+1] = min[1] + iBinY*(max[1] - min[1])/n_bins_y;
    coords_2d_bins[4*bin_index+2] = min[0] + (iBinX+1)*(max[0] - min[0])/n_bins_x;
    coords_2d_bins[4*bin_index+3] = min[1] + (iBinY+1)*(max[1] - min[1])/n_bins_y;
    // binEdges[0][iBinX] = coords_2d_bins[4*bin_index];
    // binEdges[1][iBinY] = coords_2d_bins[4*bin_index+1];
    // binEdges[0][n_bins_x] = max[0];
    // binEdges[1][n_bins_y] = max[1];
}

// calculate min and max of X and Y coordinates among all vertices
// make bins with widths: (x_max-x_min)/n_bins and (y_max-y_min)/n_bins
__global__
void constructPhaseSpaceBins(const float *d_coord, size_t n_coords, size_t start_vert,
                             size_t end_vert, size_t* n_bins,
                             float* min, float* max, int *n_vertices_in_bin,
                             size_t *indices_of_vertices_in_bin){

    // define which vertices belong to which bin
    int index = blockIdx.x * blockDim.x + threadIdx.x + start_vert;
    int stride = blockDim.x * gridDim.x;
    for(size_t i_v = index; i_v < end_vert; i_v += stride){
        
        size_t iDim = 0;
        float coord = d_coord[I2D(i_v,iDim,n_coords)];
        size_t indx_1 = (size_t)((coord - min[iDim])/((max[iDim]-min[iDim])/n_bins[iDim]));
        iDim = 1;
        coord = d_coord[I2D(i_v,iDim,n_coords)];
        size_t indx_2 = (size_t)((coord - min[iDim])/((max[iDim]-min[iDim])/n_bins[iDim]));

        size_t bin_index = I2D(indx_1, indx_2, n_bins[1]);
        atomicAdd(&n_vertices_in_bin[bin_index], 1); // to avoid race condition
        indices_of_vertices_in_bin[i_v] = bin_index;
    }

}

__global__
void prepare_indices(size_t start_vert, size_t end_vert, size_t bin_index, size_t* indices_of_vertices_in_bin, int* counters, int* tmp_indices){
    int index = blockIdx.x * blockDim.x + threadIdx.x + start_vert;
    int stride = blockDim.x * gridDim.x;
    for(size_t i_v = index; i_v < end_vert; i_v += stride){
        if (indices_of_vertices_in_bin[i_v] != bin_index)
            return;
        int index_to_fill = atomicAdd(&counters[0],1);
        tmp_indices[index_to_fill] = i_v;
        // printf("i_v: %d; bin_index: %d; indices_of_vertices_in_bin[i_v]: %d; counters[0]: %d\n", i_v,bin_index, indices_of_vertices_in_bin[i_v], counters[0]);
    }
}

__global__
void clean_indices(size_t bin_index, int* tmp_indices, int *n_vertices_in_bin, int defValue){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(size_t i_v = index; i_v < n_vertices_in_bin[bin_index]; i_v += stride){
        tmp_indices[i_v] = defValue;
    }
}

// __global__
// void prepare_indices(size_t start_vert, size_t end_vert, int* indices_of_vertices_in_bin, int* counters){
//     int index = blockIdx.x * blockDim.x + threadIdx.x + start_vert;
//     int stride = blockDim.x * gridDim.x;
//     for(size_t i_v = index; i_v < end_vert; i_v += stride){
//         int bin_index = indices_of_vertices_in_bin[i_v];
//         tmp_indices[bin_index][atomicAdd(&counters[bin_index],1)] = i_v;
//     }
// }


__global__
void findNeighbours(const int* indices_of_vert_to_find_new_neigh, // vertices for which we want to find neighbours in the targe phase-space bin
                    const size_t n_vertices_to_loop, // size of the first input array
                    const size_t indx_bin_to_use, // index of the newly added bin
                    const size_t* index_map_to_bins, // index_map_to_bins[i_v] -> bin number to which vertex belong
                    const int* n_vertices_in_bin,
                    const float *d_coord,
                    size_t start_vert,
                    size_t end_vert,
                    size_t n_coords, // number of dimentions
                    size_t n_neigh, // number of neighbours
                    float* d_dist, // distance matrix
                    int* d_indices, // indices matrix which corresponds to distance one
                    float max_radius = -1.0 // max. radius to search for neighbours
                    ){
    
    // loop to assign indices and distances to other vertices
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(size_t i = index; i <  n_vertices_to_loop; i += stride){
    // for(size_t i = 0; i < n_vertices_to_loop; i++){
        size_t i_v = indices_of_vert_to_find_new_neigh[i];

        //protection against n_vert<n_neigh
        size_t max_neighbours = n_neigh;

        size_t nfilled=0;
        int running_index = max_neighbours - 1;

        while (running_index>=0){
            if (d_indices[I2D(i_v,running_index,n_neigh)] == -1) // default init value
                running_index -= 1;
            else{
                nfilled = running_index+1;
                break;
            }
        }
        
        //set default to self
        if((n_vertices_in_bin[indx_bin_to_use]+nfilled)<n_neigh){
            max_neighbours=(n_vertices_in_bin[indx_bin_to_use]+nfilled);
        }
        
        float maxdistsq = 0;
        size_t maxidx_local = 0;
        if (nfilled>0){
            maxidx_local = searchLargestDistance(i_v,d_dist,n_neigh,maxdistsq);
            // printf("i_v: %d; nfilled: %d; maxdistsq: %f; maxidx_local: %d\n", i_v, nfilled, maxdistsq, maxidx_local);
        }
        
        
        // assigning loop - searching neighbouth for i_v
        for(size_t j_v=start_vert;j_v<end_vert;j_v++){
            if(index_map_to_bins[j_v]!=indx_bin_to_use)
                continue;
            //fill up
            float distsq = calculateDistance(i_v,j_v,d_coord,n_coords);
          
            if(nfilled<max_neighbours && (max_radius<=0 || max_radius>=distsq)){
                // filling in distances until we reach max_neighbours
                d_indices[I2D(i_v,nfilled,n_neigh)] = j_v;
                d_dist[I2D(i_v,nfilled,n_neigh)] = distsq;
                
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
                d_indices[I2D(i_v,maxidx_local,n_neigh)] = j_v;
                d_dist[I2D(i_v,maxidx_local,n_neigh)] = distsq;

                //search new max
                maxidx_local = searchLargestDistance(i_v,d_dist,n_neigh,maxdistsq);
            }
        }// loop through vertices
    }// loop through vertices
}

__global__
void calculate2dDistanceToTheBinEdges(
    int* output_indices, // ???
    int* n_output_vertices, // ???
    const int* input_indices, // vertices for which we want to find neighbours in the targe phase-space bin
    const size_t n_input_vertices, // size of the first input array
    const float* target_bin_coords, // {x1, y1, x2, y2} coords of the target bin
    const float *d_coord,
    const size_t n_coords, // number of dimentions
    size_t n_neigh, // number of neighbours
    float* d_dist, // distance matrix
    int* d_indices, // indices matrix which corresponds to distance one
    float max_radius // max. radius to search for neighbours
    // int *tmp_indices
    ){


    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(size_t i = index; i < n_input_vertices; i += stride){
        size_t i_v = input_indices[i];
        
        // safety check
        float x = d_coord[I2D(i_v,0,n_coords)];
        float y = d_coord[I2D(i_v,1,n_coords)];
        if (x>target_bin_coords[0] && x<target_bin_coords[2]
            && y>target_bin_coords[1] && y<target_bin_coords[3]){
            continue; // i_v belongs to the target bin 
        }
        
        // check if i_v has required number of neighbours:
        size_t n_found_neighbours=0;
        int running_index = n_neigh - 1;
        while (running_index>=0){
            if (d_indices[I2D(i_v,running_index,n_neigh)] == -1) // default init value
                running_index -= 1;
            else{
                n_found_neighbours = running_index+1;
                break;
            }
        }
                
        // include i_v for output if it doesn't have enough neighbours
        if (n_found_neighbours<n_neigh){
            output_indices[atomicAdd(&n_output_vertices[0],1)] = i_v;
            continue;
        }
        
        // find the distance to the farthermost neighbour
        float maxdistsq = 0; // largest distance
        size_t maxidx_local = searchLargestDistance(i_v,d_dist,n_neigh,maxdistsq);
        
        // find the closest distance to the target bin
        float distance_to_bin = 0.0;
        if ((x>target_bin_coords[0] && x<target_bin_coords[2]) || 
            (y>target_bin_coords[1] && y<target_bin_coords[3])){
            size_t iDim = 0;
            if (x>target_bin_coords[0] && x<target_bin_coords[2])
                iDim = 1;
            
            float lowBinEdge = target_bin_coords[iDim];
            float highBinEdge = target_bin_coords[iDim+2];
            float d1 = pow((d_coord[I2D(i_v,iDim,n_coords)] - lowBinEdge),2);
            float d2 = pow((d_coord[I2D(i_v,iDim,n_coords)] - highBinEdge),2);
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
            distance_to_bin = calculate2dDistanceToThePoint(pointCoord, i_v, d_coord, n_coords);
        }
        
        // cout << "i_v: " << i_v << "; distance_to_bin: " << distance_to_bin << "; maxdistsq: " << maxdistsq << endl;
        
        if (distance_to_bin<maxdistsq){
            output_indices[n_output_vertices[0]] = i_v;
            n_output_vertices[0] += 1;
        }
    }
    
    // return array of proper size
    // for (int i=0; i<n_output_vertices[0]; i++){
    //     output_indices[i] = tmp_indices[i];
    // }
    
    return;
}


__global__
void new_knn_kernel(
        const float *d_coord,
        const int* d_row_splits,
        int *d_indices,
        float *d_dist,

        const int n_vert,
        const int n_neigh,
        const int n_coords,

        const int j_rs,
        const bool tf_compat,
        const float max_radius,
        const int n_bins_x,
        const int n_bins_y,
        float *coords_2d_bins, 
        int *n_vertices_in_bin,
        size_t *indices_of_vertices_in_bin
        ) {

    //really no buffering at all here

    const size_t start_vert = d_row_splits[j_rs];
    const size_t end_vert = d_row_splits[j_rs+1];

    const size_t i_v =  blockIdx.x * blockDim.x + threadIdx.x + start_vert;
    if(i_v >= end_vert || i_v>=n_vert)
        return;//this will be a problem with actual RS


    __syncthreads(); // block level synchronization barrier

    // for(size_t i=0; i<n_neigh*(end_vert-start_vert); i++){
    //     d_indices[i] = -1;
    // }
    // for (auto i : _grid_stride_range(0, n_neigh*(end_vert-start_vert)))
    // {
    //     d_indices[i] = -1;
    // }

	// float coords_2d_bins[4*n_bins_x*n_bins_y]; // {x1_bin_1, y1_bin_1, x2_bin_2, y2_bin_2, ...}
	// size_t n_vertices_in_bin[n_bins_x*n_bins_y];
    // for(int i=0; i<n_bins_x*n_bins_y; i++){
    //     n_vertices_in_bin[i] = 0;
    // }
	// size_t indices_of_vertices_in_bin[(end_vert-start_vert)];



    // __syncthreads(); // block level synchronization barrier


}
}

typedef Eigen::GpuDevice GPUDevice;


template <typename dummy>
struct NewKnnOpFunctor<GPUDevice, dummy> {
    void operator()(const GPUDevice& d,

            const float *d_coord,
            const int* d_row_splits,
            int *d_indices,
            // int *_d_indices,
            float *d_dist,

            const int n_vert,
            const int n_neigh,
            const int n_coords,

            const int n_rs,
            const bool tf_compat,
            const float max_radius,
            const int n_bins_x,
            const int n_bins_y) {
        
        // ********************
        // STEP 1: set defaults
        // ********************
        // int* d_indices; // one need to do cuda memory allocation for arrays!
        // cudaMallocManaged(&d_indices, n_vert*n_neigh*sizeof(int));
        // printf("\n\n*** STEP 1: set defaults ***\n");

        int blockSize = 256;
        int numBlocks = (n_vert + blockSize - 1) / blockSize;
        gpu::set_defaults<<<numBlocks,blockSize>>>(
                d_indices,
                d_dist,
                tf_compat,
                n_vert,
                n_neigh);
        cudaDeviceSynchronize();

        // printf("d_indices\n");
        // gpu::print_array<<<1,1>>>(d_indices,0,5);
        // cudaDeviceSynchronize();
        // printf("\n");
        
        // printf("d_coord\n");
        // gpu::print_array<<<1,1>>>(d_coord,0,10);
        // cudaDeviceSynchronize();



        // printf("n_vert: %d; n_neigh: %d\n", n_vert, n_neigh);
        // printf("d_coord[0]: %f\n",d_coord[0]);
        // printf("d_row_splits[0]: %d\n",d_row_splits[0]);
        // printf("d_dist[0]: %f\n",d_dist[0]);
        // printf("d_indices[0]: %d\n",d_indices[0]);
        
        // printf("d_indices: ");
        // for(int jj=0; jj<10; jj++){
        //     printf("%d ",d_indices[jj]);
        // }
        // printf("\n");
    //
    //     // printf("STEP 1 is finished\n");
    //
    //     // I am not sure if splits are working properly... TODO to test it
    //     // const size_t start_vert = d_row_splits[0];
    //     // const size_t end_vert = d_row_splits[1];
        const size_t start_vert = 0; // it's OK to create variables if one pass it as a copy
        const size_t end_vert = n_vert;
    //
    //     // ************************************
    //     // STEP 2: divide vertices between bins
    //     // ************************************
        // printf("\n\n*** STEP 2: divide vertices between bins ***\n");
        float* coords_2d_bins; // one need to do cuda memory allocation for arrays!
        cudaMallocManaged(&coords_2d_bins, 4*n_bins_x*n_bins_y*sizeof(float));
        int* n_vertices_in_bin;
        cudaMallocManaged(&n_vertices_in_bin, n_bins_x*n_bins_y*sizeof(size_t));
        // TODO it's OKish w/o parallelization, it's a small array
        for(int i=0; i<n_bins_x*n_bins_y; i++){
            n_vertices_in_bin[i] = 0;
        }
        size_t *indices_of_vertices_in_bin;
        cudaMallocManaged(&indices_of_vertices_in_bin, (end_vert-start_vert)*sizeof(size_t));
        size_t *n_bins;
        cudaMallocManaged(&n_bins, 2*sizeof(size_t));
        n_bins[0] = n_bins_x;
        n_bins[1] = n_bins_y;

        // TODO FIXME temporary do not search min and max, just use true value
        // to use some efficient existing solution
        float* min;
        float* max;
        cudaMallocManaged(&max, 2*sizeof(float));
        cudaMallocManaged(&min, 2*sizeof(float));
        min[0] = -0.0001;
        min[1] = -0.0001;
        max[0] = 1.0001;
        max[1] = 1.0001;

        dim3 pre_numblocks(n_bins_x/32+1,n_bins_y/32+1);
        dim3 pre_threadsperblock(32,32);
        gpu::get_bin_coords<<<pre_numblocks,pre_threadsperblock,0,d.stream()>>>(min, max, coords_2d_bins, n_bins_x, n_bins_y);
        cudaDeviceSynchronize();

        gpu::constructPhaseSpaceBins<<<numBlocks, blockSize>>>(
                d_coord, n_coords, start_vert, end_vert, n_bins, min, max,
                n_vertices_in_bin, indices_of_vertices_in_bin);
        cudaDeviceSynchronize();
        // printf("indices_of_vertices_in_bin: ");
        // for(int jj=0; jj<(end_vert-start_vert); jj++){
        //     printf("%d ",indices_of_vertices_in_bin[jj]);
        // }
        // printf("\n");

        // printf("n_vertices_in_bin\n");
        // gpu::print_array<<<1,1>>>(n_vertices_in_bin,0,n_bins_x*n_bins_y);
        // cudaDeviceSynchronize();
        // printf("\n");

        // ***********************
        // STEP 3: find neighbours
        // ***********************
        // printf("\n\n*** STEP 3: find neighbours ***\n");
        float* binCoords;
        cudaMallocManaged(&binCoords, 4*sizeof(float));
        int *counters;
        cudaMallocManaged(&counters, sizeof(int));
        int *tmp_indices;
        cudaMallocManaged(&tmp_indices, (end_vert-start_vert)*sizeof(int));
        int *tmp_indices2;
        cudaMallocManaged(&tmp_indices2, (end_vert-start_vert)*sizeof(int));

        for (size_t iBinX=0; iBinX<n_bins_x; iBinX++){
            for (size_t iBinY=0; iBinY<n_bins_y; iBinY++){
                size_t bin_index = I2D(iBinX, iBinY, n_bins_y);
                for (int i=0; i<4; i++){
                    binCoords[i] = coords_2d_bins[4*bin_index + i];
                }

                // prepare indices input array
                counters[0] = 0;

                gpu::prepare_indices<<<numBlocks, blockSize>>>(start_vert, end_vert, bin_index, indices_of_vertices_in_bin, counters, tmp_indices);
                cudaDeviceSynchronize();

                // printf("tmp_indices\n");
                // gpu::print_array<<<1,1>>>(tmp_indices,0,n_vertices_in_bin[bin_index]);
                // cudaDeviceSynchronize();
                // printf("\n");

                // printf("MAIN BIN INDX: %d\nn_vert_in_bin (total: %d): ",bin_index, n_vertices_in_bin[bin_index]);
                // for(int jj=0; jj<n_vertices_in_bin[bin_index]; jj++){
                //     printf("%d ",tmp_indices[jj]);
                // }
                // printf("\n");


                gpu::findNeighbours<<<numBlocks, blockSize>>>(tmp_indices, // vertices for which we want to find neighbours in the targe phase-space bin
                    n_vertices_in_bin[bin_index], // size of the first input array
                    bin_index, // index of the newly added bin
                    indices_of_vertices_in_bin, // index_map_to_bins[i_v] -> bin number to which vertex belong
                    n_vertices_in_bin,
                    d_coord,
                    start_vert,
                    end_vert,
                    n_coords, // number of dimentions
                    n_neigh, // number of neighbours
                    d_dist,
                    d_indices,
                    -1.0
                    );
                cudaDeviceSynchronize();

                // if (n_vertices_in_bin[bin_index]>0){
                //     printf("d_indices[0]: ");
                //     gpu::print_neighbours<<<1,1>>>(tmp_indices[0], d_indices, d_dist, n_neigh);
                //     cudaDeviceSynchronize();
                //     printf("\n");
                // }

                for (size_t iBinX2=0; iBinX2<n_bins_x; iBinX2++){
                    for (size_t iBinY2=0; iBinY2<n_bins_y; iBinY2++){
                        size_t bin_index2 = I2D(iBinX2, iBinY2, n_bins_y);
                        if (bin_index2 == bin_index)
                            continue;
                        for (int i=0; i<4; i++){
                            binCoords[i] = coords_2d_bins[4*bin_index2 + i];
                        }

                        counters[0] = 0;
                        gpu::calculate2dDistanceToTheBinEdges<<<numBlocks, blockSize>>>(
                            tmp_indices2, // out, not finished vertices 
                            counters, // out, # of not finished vertices
                            tmp_indices, // in, vertices which belong to bin #bin_index
                            n_vertices_in_bin[bin_index], // in, n_verts in bin #bin_index
                            binCoords, // {x1, y1, x2, y2} coords of the target bin
                            d_coord,
                            n_coords, 
                            n_neigh, 
                            d_dist, 
                            d_indices, 
                            -1.0
                            // tmp_ndices2
                            );
                        cudaDeviceSynchronize();

                        // printf("---> BIN_INDEX2: %d; # OF NOT FINISHED VERTS: %d\n",bin_index2, counters[0]);

                        if (counters[0]>0){


                            // find neighbours
                            gpu::findNeighbours<<<numBlocks, blockSize>>>(
                                tmp_indices2, // in, vertices to search neighbours for
                                counters[0], // in, n_verts
                                bin_index2, // in, index of the newly added bin
                                indices_of_vertices_in_bin, // index_map_to_bins[i_v] -> bin number to which vertex belong
                                n_vertices_in_bin,
                                d_coord,
                                start_vert,
                                end_vert,
                                n_coords, // number of dimentions
                                n_neigh, // number of neighbours
                                d_dist,
                                d_indices,
                                -1.0
                                );
                            cudaDeviceSynchronize();
                        } // if (counters[0]>0)
                    } // iBinY2
                } // iBinX2

            }// iBinY
        }//iBinX




        cudaFree(d_indices);
        cudaFree(coords_2d_bins);
        cudaFree(n_vertices_in_bin);
        cudaFree(indices_of_vertices_in_bin);
        cudaFree(min);
        cudaFree(max);
        cudaFree(n_bins);
        cudaFree(binCoords);
        cudaFree(counters);
        cudaFree(tmp_indices);
        cudaFree(tmp_indices2);
    

    }
};



template struct NewKnnOpFunctor<GPUDevice, int>;

}//functor
}//tensorflow


#endif  // GOOGLE_CUDA
