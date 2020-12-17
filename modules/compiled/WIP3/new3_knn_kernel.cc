
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "new3_knn_kernel.h"
#include "helpers.h"
#include <string> //size_t, just for helper function
#include <cmath>

#include <iostream> //remove later DEBUG FIXME

using namespace std;

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

void prepare_vtx_to_process(
    const int start_vert,
    const int end_vert,
    const int* next_bin,
    int* n_vtx_to_process,
    int* vtx_to_process){

    for(size_t i_v = start_vert; i_v < end_vert; i_v += 1){
        if (next_bin[i_v]>=0){
            vtx_to_process[n_vtx_to_process[0]] = i_v;
            n_vtx_to_process[0]++;
        }
    }
}

void calculate_bin_bin_dist_matrices(int *n_bins, const float* coords_2d_bins,
        float* bin_bin_dist, int* bin_bin_dist_idx){

    for (int iBinX=0; iBinX<n_bins[0]; iBinX++){
        for (int iBinY=0; iBinY<n_bins[1]; iBinY++){
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
    }
}

void print_2d_matrix(
        const float *in_arr,
        const size_t stride,
        const size_t start,
        const size_t end
){
    for(size_t i = start ; i < end ; i += 1){
        float tmp_val = in_arr[i];
        if (i % stride == 0){
            printf("\n");
            printf("i: %d: ", (int)(i/stride));
        }
        printf("\t%f", tmp_val);
    }
    printf("\n");
}

void print_2d_matrix(
        const int *in_arr,
        const size_t stride,
        const size_t start,
        const size_t end
){
    for(size_t i = start ; i < end ; i += 1){
        float tmp_val = in_arr[i];
        if (i % stride == 0){
            printf("\n");
            printf("i: %d: ", (int)(i/stride));
        }
        printf("\t%d", (int)tmp_val);
    }
    printf("\n");
}

void get_bin_coords(float* min, float* max, float *coords_2d_bins, const size_t n_bins_x, const size_t n_bins_y){
    for (int iBinX=0; iBinX<n_bins_x; iBinX++){
        for (int iBinY=0; iBinY<n_bins_y; iBinY++){
            // define phase-space bin edges
            size_t bin_index = I2D(iBinY, iBinX, n_bins_x);
            coords_2d_bins[4*bin_index] = min[0] + iBinX*(max[0] - min[0])/n_bins_x; // x1
            coords_2d_bins[4*bin_index+1] = min[1] + iBinY*(max[1] - min[1])/n_bins_y; // y1
            coords_2d_bins[4*bin_index+2] = min[0] + (iBinX+1)*(max[0] - min[0])/n_bins_x; // x2
            coords_2d_bins[4*bin_index+3] = min[1] + (iBinY+1)*(max[1] - min[1])/n_bins_y; // y2
        }
    }
}

int constructPhaseSpaceBins(const float *d_coord, size_t n_coords, size_t start_vert,
                             size_t end_vert, int* n_bins,
                             float* coords_min, float* coords_max, int *n_vtx_per_bin,
                             int* bin_idx){

    for(size_t i_v = start_vert; i_v < end_vert; i_v += 1){
        
        size_t iDim = 0;
        float coord = d_coord[I2D(i_v,iDim,n_coords)];
//         cout << "x: " << coord;
        size_t indx_1 = (size_t)((coord - coords_min[iDim])/((coords_max[iDim]-coords_min[iDim])/n_bins[iDim]));
        iDim = 1;
        coord = d_coord[I2D(i_v,iDim,n_coords)];
        size_t indx_2 = (size_t)((coord - coords_min[iDim])/((coords_max[iDim]-coords_min[iDim])/n_bins[iDim]));
//         cout << " y: " << coord;
        size_t bin_index = I2D(indx_2, indx_1, n_bins[0]);
//         cout << "\nbin_index: " << bin_index << "; "
        n_vtx_per_bin[bin_index] += 1; // to avoid race condition
        bin_idx[i_v] = bin_index;
    }
    
    return 0;

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

void prepare_translation_matrix(size_t start_vert, size_t end_vert, int* n_vtx_per_bin_cumulative, int* bin_idx, int* zero_counters, int* forward_translation_matrix, int* backward_translation_matrix){
    for(size_t i_v = start_vert; i_v < end_vert; i_v += 1){
        int bin_index = bin_idx[i_v];
        int index_to_fill = zero_counters[bin_index] + n_vtx_per_bin_cumulative[bin_index];
        zero_counters[bin_index]++;
        forward_translation_matrix[index_to_fill] = i_v;
    }
    for(size_t i_v = start_vert; i_v < end_vert; i_v += 1){
        backward_translation_matrix[forward_translation_matrix[i_v]] = i_v;
    }
}

template <typename T>
void translate_2d_matrix(size_t matrix_height, size_t matrix_width, const int* translation_matrix, const T *in_mattrix, T* out_matrix){
    for(size_t i_counter = 0; i_counter < matrix_height; i_counter += 1){
        size_t real_index = translation_matrix[i_counter];
        for(size_t i_column = 0; i_column < matrix_width; i_column += 1)
            out_matrix[matrix_width*i_counter+i_column] = in_mattrix[matrix_width*real_index+i_column];
    }
}

void translate_content_of_2d_matrix(size_t n_element_in_matrix, const int* translation_matrix, const int *in_mattrix, int* out_matrix){
    for(size_t i_counter = 0; i_counter < n_element_in_matrix; i_counter += 1){
        int real_index = in_mattrix[i_counter];
        int translated_index = -1;
        if (real_index>=0)
            translated_index = translation_matrix[real_index];
        out_matrix[i_counter] = translated_index;
    }
}

float calculate2dDistanceToThePoint(float *pointCoord, size_t i_v, const float* d_coord, size_t n_coords){
    float distsq=0;
    for(size_t i=0;i<2;i++){
        float dist = d_coord[I2D(i_v,i,n_coords)] - pointCoord[i];
        distsq += dist*dist;
    }
    return distsq;
}

void calculate2dDistanceToTheBinEdges(
    const int start_vert,
    const int end_vert,
    const float *d_coord, // vertices coords
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
    
    for(size_t i_v = 0; i_v < n_input_vertices; i_v += 1){
        
        int bin_index = bin_idx[i_v];
        int bin_index_x = bin_index%n_bins_x;
        int bin_index_y = (int)bin_index/n_bins_x;
        float x = d_coord[I2D(i_v,0,n_coords)];
        float y = d_coord[I2D(i_v,1,n_coords)];
                
//         printf("\nbin_index: %d; x: %d, y: %d\n",bin_index, bin_index_x, bin_index_y);
//         printf("vertex coords: x: %f; y: %f\n",x, y);
//         printf("vtx_bin_coords: %f, %f, %f, %f\n" ,coords_2d_bins[4*bin_index+0],coords_2d_bins[4*bin_index+1],coords_2d_bins[4*bin_index+2],coords_2d_bins[4*bin_index+3]);
        
        // copy matrices first
        for (int i=0; i<n_bins; i++){
            bin_idxs_ordered_by_dist_to_the_vtx[i_v*n_bins+i] = bin_bin_dist_idx[bin_index*n_bins+i];
            bin_vtx_dist_ordered_by_dist_to_the_vtx[i_v*n_bins+i] = bin_bin_dist[bin_index*n_bins+i];
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
            
//             printf("target_bin_coords: %f, %f, %f, %f\n" ,target_bin_coords[0],target_bin_coords[1],target_bin_coords[2],target_bin_coords[3]);
            
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
            bin_vtx_dist_ordered_by_dist_to_the_vtx[i_v*n_bins+j] = distance_to_bin;
        }
    }
    
    return;
}

float calculateDistance(size_t i_v, size_t j_v, const float * d_coord, size_t n_coords){
    if(i_v == j_v)
        return 0;
    float distsq=0;
    for(size_t i=0;i<n_coords;i++){
        float dist = d_coord[I2D(i_v,i,n_coords)] - d_coord[I2D(j_v,i,n_coords)];
        distsq += dist*dist;
    }
    return distsq;
}

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
    // int debug_vtx = 9;

    for(size_t i = 0; i <  n_vtx_to_process[0]; i += 1){
//         int i_v = vtx_to_process[i]; // get vtx index
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
        //     printf("DEBUG: next_bin[%d]: %d; distance_to_bin: %f; maxdistsq: %f\n",next_bin[i_v], target_bin, distance_to_bin,maxdistsq);
        
        bool all_neig_are_fount_for_vtx = false;
        while (distance_to_bin>maxdistsq){
            if ((next_bin[i_v] + 1) < n_bins_x*n_bins_y){
                next_bin[i_v] = next_bin[i_v] + 1;
                target_bin = bin_idxs_ordered_by_dist_to_the_vtx[i_v*n_bins_x*n_bins_y+next_bin[i_v]];
                distance_to_bin = bin_vtx_dist_ordered_by_dist_to_the_vtx[i_v*n_bins_x*n_bins_y+next_bin[i_v]];
                // if (i_v==debug_vtx )
                //     printf("DEBUG: next_bin: %d; distance_to_bin: %f; maxdistsq: %f\n",next_bin[i_v], distance_to_bin,maxdistsq);
            }
            else{
                next_bin[i_v] = -1; // end of the search
                n_vtx_finished[0] += 1;
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
            n_vtx_finished[0] += 1;
        }
        // if (i_v==debug_vtx )
        //         printf("DEBUG: next_bin: %d; farthest_neigh dist: %f\n",next_bin[i_v],neigh_dist[I2D(i_v,farthest_neighbour[i_v],K)]);
    }
}

void find_min_max_2d(
    const float *d_coords,
    const int n_coords,
    const int start_vert,
    const int end_vert,
    float* coord_min,
    float* coord_max
    )
{
    coord_min[0] = d_coords[0];
    coord_max[0] = d_coords[0];
    coord_min[1] = d_coords[1];
    coord_max[1] = d_coords[1];
    float coord_x, coord_y;
    for(size_t i=start_vert; i<end_vert; i++){
        coord_x = d_coords[i*n_coords];
        coord_y = d_coords[i*n_coords+1];
        if(coord_min[0] > coord_x)
            coord_min[0] = coord_x;
        if(coord_min[1] > coord_y)
            coord_min[1] = coord_y;
        if(coord_max[0] < coord_x)
            coord_max[0] = coord_x;
        if(coord_max[1] < coord_y)
            coord_max[1] = coord_y;
    }
    coord_min[0] -= (coord_max[0] - coord_min[0])*0.001;
    coord_max[0] += (coord_max[0] - coord_min[0])*0.001;
    coord_min[1] -= (coord_max[1] - coord_min[1])*0.001;
    coord_max[1] += (coord_max[1] - coord_min[1])*0.001;
}

void new3_knn_kernel(
        const float *d_coords,
        const int* d_row_splits,
        int *neigh_idx,
        float *neigh_dist,

        const int n_vert,
        const int n_neigh,
        const int n_coords,

        const int j_rs,
        const bool tf_compat,
        const float max_radius,
        const int n_bins_x,
        const int n_bins_y) {

    //really no buffering at all here

    const size_t start_vert = d_row_splits[j_rs];
    const size_t end_vert = d_row_splits[j_rs+1];

	int *n_bins = new int[2];
	n_bins[0] = n_bins_x;
	n_bins[1] = n_bins_y;
    
	float* coord_min = new float[2];
	float* coord_max = new float[2];
    find_min_max_2d(d_coords,n_coords,start_vert,end_vert,coord_min,coord_max);
    // printf("coord_min:");
    // print_2d_matrix(coord_min, 1, 0,2);
    // printf("coord_max:");
    // print_2d_matrix(coord_max, 1, 0,2);
	float *coords_2d_bins = new float[4*n_bins_x*n_bins_y];

	get_bin_coords(coord_min, coord_max, coords_2d_bins, n_bins_x, n_bins_y);

	float* bin_bin_dist = new float[n_bins_x*n_bins_y*(n_bins_x*n_bins_y)];
	int* bin_bin_dist_idx = new int[n_bins_x*n_bins_y*(n_bins_x*n_bins_y)];
	
	calculate_bin_bin_dist_matrices(n_bins, coords_2d_bins,
        bin_bin_dist, bin_bin_dist_idx);
        
    // printf("bin_bin_dist:\n");
    // print_2d_matrix(bin_bin_dist, n_bins_x*n_bins_y, 0,n_bins_x*n_bins_y*n_bins_x*n_bins_y);

    // printf("bin_bin_dist_idx:\n");
    // print_2d_matrix(bin_bin_dist_idx, n_bins_x*n_bins_y, 0,n_bins_x*n_bins_y*n_bins_x*n_bins_y);


	const int K = n_neigh;
	const int V = end_vert - start_vert;
	
	int *n_vtx_per_bin = new int[n_bins_x*n_bins_y];
	for (int i=0; i<n_bins_x*n_bins_y; i++){
		n_vtx_per_bin[i] = 0;
	}
	int *bin_idx = new int[end_vert-start_vert];

	constructPhaseSpaceBins(
        d_coords, n_coords, start_vert, end_vert, n_bins, coord_min, coord_max,
        n_vtx_per_bin, bin_idx);
        
    // printf("n_vtx_per_bin:\n");
    // print_2d_matrix(n_vtx_per_bin,1,0,n_bins_x*n_bins_y);
	
    // printf("coords_2d_bins:\n");
    // print_2d_matrix(coords_2d_bins,4,0,4*n_bins_x*n_bins_y);
	
    // printf("bin_idx:\n");
    // print_2d_matrix(bin_idx,1,0,V);
	
	int* zero_counters = new int[n_bins_x*n_bins_y];
	for (int i=0; i<n_bins_x*n_bins_y; i++){
		zero_counters[i] = 0;
	}
	
	int* n_vtx_per_bin_cumulative = new int[n_bins_x*n_bins_y+1];
	calculate_n_vtx_per_bin_cumulative(n_bins_x*n_bins_y, n_vtx_per_bin, n_vtx_per_bin_cumulative);
	
    // printf("n_vtx_per_bin_cumulative:\n");
    // print_2d_matrix(n_vtx_per_bin_cumulative,1,0,n_bins_x*n_bins_y+1);
	
	int* vtx_idx_translation_matrix = new int[(end_vert-start_vert)];
	int* backward_vtx_idx_translation_matrix = new int[(end_vert-start_vert)];
	prepare_translation_matrix(start_vert, end_vert, n_vtx_per_bin_cumulative, bin_idx, zero_counters, vtx_idx_translation_matrix, backward_vtx_idx_translation_matrix);


    // printf("vtx_idx_translation_matrix:\n");
    // print_2d_matrix(vtx_idx_translation_matrix,1,0,V);
	
    // printf("d_coords:\n");
    // print_2d_matrix(d_coords,4,0,4*V);
	
	float *d_coords_sorted = new float[n_coords*V];
	translate_2d_matrix(end_vert-start_vert, n_coords, vtx_idx_translation_matrix, d_coords, d_coords_sorted);

    // printf("d_coords_sorted:\n");
    // print_2d_matrix(d_coords_sorted,4,0,4*V);
	
	int *bin_idx_sorted = new int[end_vert-start_vert];
	translate_2d_matrix(end_vert-start_vert, 1, vtx_idx_translation_matrix, bin_idx, bin_idx_sorted);
	
    // printf("bin_idx_sorted:\n");
    // print_2d_matrix(bin_idx_sorted,1,0,end_vert-start_vert);
	
	int* bin_idxs_ordered_by_dist_to_the_vtx = new int[V * (n_bins_x*n_bins_y)];
	float* bin_vtx_dist_ordered_by_dist_to_the_vtx = new float[V * (n_bins_x*n_bins_y)];

    int* vtx_to_process = new int[V];
    for (int i=0; i<V; i++)
        vtx_to_process[i] = i;
    int* n_vtx_to_process = new int[1];
    n_vtx_to_process[0] = V;

	calculate2dDistanceToTheBinEdges(
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
		
    // printf("bin_idxs_ordered_by_dist_to_the_vtx:\n");
    // print_2d_matrix(bin_idxs_ordered_by_dist_to_the_vtx, n_bins_x*n_bins_y, 0,n_bins_x*n_bins_y*V);
	
    // printf("bin_vtx_dist_ordered_by_dist_to_the_vtx:\n");
    // print_2d_matrix(bin_vtx_dist_ordered_by_dist_to_the_vtx, n_bins_x*n_bins_y, 0,n_bins_x*n_bins_y*V);
	
	// INIT changable arrays/variables
	int *n_vtx_finished = new int[1];
    n_vtx_finished[0] = 0;
    int* neigh_idx_tmp = new int[V*K];
    float* neigh_dist_tmp = new float[V*K];
	for (int i=0; i<V * K; i++){
		neigh_idx_tmp[i] = -1;
		neigh_dist_tmp[i] = 0;
	}
	int* next_bin = new int[V];
	int* farthest_neighbour = new int[V];
	for (int i=0; i<V; i++){
		next_bin[i] = 0;
		farthest_neighbour[i] = -1;
	}

    // printf("bin_idx_sorted:\n");
    // print_2d_matrix(bin_idx_sorted, 1, 0,V);
	

	int counter = 0;
	int N_ITER = n_bins_x*n_bins_y;
	while (counter<N_ITER && n_vtx_finished[0] < V){
        // printf("\nITERATION #%d\n************\n", counter);
        // printf("n_vtx_to_process:");
        // print_2d_matrix(n_vtx_to_process, 1, 0,1);
        // printf("vtx_to_process:");
        // print_2d_matrix(vtx_to_process, 1, 0,n_vtx_to_process[0]);
        perform_kNN_search(
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

        n_vtx_to_process[0] = 0;
        prepare_vtx_to_process(
            start_vert,
            end_vert,
            next_bin,
            n_vtx_to_process,
            vtx_to_process
                );

		counter++;

        // DEBUG PRINTOUT
        // printf("n_vtx_finished: %d\n", n_vtx_finished[0]);
        // if (N_ITER % 10 == 0){
        //     printf("neigh_idx:");
        //     print_2d_matrix(neigh_idx,K,0,V*K);
        //     printf("neigh_dist:");
        //     print_2d_matrix(neigh_dist,K,0,V*K);
        // }

        // printf("neigh_idx_tmp:\n");
        // print_2d_matrix(neigh_idx_tmp, K, 0,V*K);
	}

    printf("***********\nN_ITER: %d\n***********\n", counter);

    // printf("neigh_idx_tmp:\n");
    // print_2d_matrix(neigh_idx_tmp,K,0,(end_vert-start_vert)*K);

    // printf("neigh_dist_tmp:\n");
    // print_2d_matrix(neigh_dist_tmp,K,0,(end_vert-start_vert)*K);

    /// DEBUG COMMENTING
    translate_2d_matrix(end_vert-start_vert, K, backward_vtx_idx_translation_matrix, neigh_idx_tmp, neigh_idx);
    translate_2d_matrix(end_vert-start_vert, K, backward_vtx_idx_translation_matrix, neigh_dist_tmp, neigh_dist);
    translate_content_of_2d_matrix(K*(end_vert-start_vert), vtx_idx_translation_matrix, neigh_idx, neigh_idx);

    // for (int i=0; i<(end_vert-start_vert)*K; i++){
    //     neigh_dist[i] = neigh_dist[i]*neigh_dist[i];
    // }

    delete[] n_bins;
    delete[] coord_min;
    delete[] coord_max;
    delete[] bin_bin_dist;
    delete[] bin_bin_dist_idx;
    delete[] n_vtx_per_bin;
    delete[] bin_idx;
    delete[] zero_counters;
    delete[] n_vtx_per_bin_cumulative;
    delete[] vtx_idx_translation_matrix;
    delete[] backward_vtx_idx_translation_matrix;
    delete[] d_coords_sorted;
    delete[] bin_idx_sorted;
    delete[] bin_idxs_ordered_by_dist_to_the_vtx;
    delete[] bin_vtx_dist_ordered_by_dist_to_the_vtx;
    delete[] neigh_idx_tmp;
    delete[] neigh_dist_tmp;
    delete[] next_bin;
    delete[] farthest_neighbour;
    delete[] n_vtx_to_process;
    delete[] vtx_to_process;
	
}
//
// CPU specialization
template<typename dummy>
struct New3KnnOpFunctor<CPUDevice, dummy> {
    void operator()(const CPUDevice &d,

            const float *d_coord,
            const int* d_row_splits,
            int *d_indices,
            float *d_dist,

            const int n_vert,
            const int n_neigh,
            const int n_coords,

            const int n_rs,
            const bool tf_compat,
            const float max_radius,
            const int n_bins_x,
            const int n_bins_y) {


        // set_defaults(d_indices,
        //         d_dist,
        //         tf_compat,
        //         n_vert,
        //         n_neigh);
        //really no buffering at all here

        for(size_t j_rs=0;j_rs<n_rs-1;j_rs++){
            new3_knn_kernel(d_coord,
                    d_row_splits,
                    d_indices,
                    d_dist,

                    n_vert,
                    n_neigh,
                    n_coords,

                    j_rs,
                    tf_compat,
                    max_radius,
                    n_bins_x,
                    n_bins_y);
        }
    }
};

template<typename Device>
class New3KnnOp : public OpKernel {
public:
    explicit New3KnnOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context,
                        context->GetAttr("n_neighbours", &K_));
        OP_REQUIRES_OK(context,
                        context->GetAttr("tf_compatible", &tf_compat_));
        OP_REQUIRES_OK(context,
                        context->GetAttr("max_radius", &max_radius_));
        OP_REQUIRES_OK(context,
                        context->GetAttr("n_bins_x", &n_bins_x_));
        OP_REQUIRES_OK(context,
                        context->GetAttr("n_bins_y", &n_bins_y_));

        if(max_radius_>0)
            max_radius_ *= max_radius_;//use squared

    }

    void Compute(OpKernelContext *context) override {

        const Tensor &d_coord_tensor = context->input(0);
        const Tensor &d_rs_tensor = context->input(1);


        int n_vert = d_coord_tensor.dim_size(0);
        int n_coords = d_coord_tensor.dim_size(1);
        int n_rs = d_rs_tensor.dim_size(0);

        TensorShape outputShape;
        outputShape.AddDim(n_vert);
        outputShape.AddDim(K_);

        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, outputShape, &output_tensor));

        Tensor *output_distances = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, outputShape, &output_distances));


        New3KnnOpFunctor<Device, int>()(
                context->eigen_device<Device>(),

                d_coord_tensor.flat<float>().data(),
                d_rs_tensor.flat<int>().data(),
                output_tensor->flat<int>().data(),
                output_distances->flat<float>().data(),

                n_vert,
                K_,
                n_coords,

                n_rs,
                tf_compat_,
                max_radius_,
                n_bins_x_,
                n_bins_y_
        );



    }

private:
    int K_;
    bool tf_compat_;
    float max_radius_;
    int n_bins_x_;
    int n_bins_y_;
};

REGISTER_KERNEL_BUILDER(Name("New3Knn").Device(DEVICE_CPU), New3KnnOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct New3KnnOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("New3Knn").Device(DEVICE_GPU), New3KnnOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}//functor
}//tensorflow
