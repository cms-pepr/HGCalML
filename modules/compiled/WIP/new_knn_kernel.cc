
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA


#include "tensorflow/core/framework/op_kernel.h"
#include "new_knn_kernel.h"
#include "helpers.h"
#include <string> //size_t, just for helper function
#include <cmath>

#include <iostream> //remove later DEBUG FIXME

using namespace std;

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

float calculateDistance(size_t i_v, size_t j_v, const float * d_coord, size_t n_coords){
    float distsq=0;
    if(i_v == j_v)
        return 0;
    for(size_t i=0;i<n_coords;i++){
        float dist = d_coord[I2D(i_v,i,n_coords)] - d_coord[I2D(j_v,i,n_coords)];
        distsq += dist*dist;
    }
    return sqrt(distsq);
}

int searchLargestDistance(int i_v, float* d_dist, int n_neigh, float& maxdist){

    maxdist=0;
    int maxidx=0;
    if(n_neigh < 2)
        return maxidx;
//     for(size_t n=1;n<n_neigh;n++){ //0 is self
    for(size_t n=0;n<n_neigh;n++){ //0 is self
        float distsq = d_dist[I2D(i_v,n,n_neigh)];
        if(distsq > maxdist){
            maxdist = distsq;
            maxidx = n;
        }
    }
    return maxidx;
}

void findNeighbours(const size_t* indices_of_vert_to_find_new_neigh, // vertices for which we want to find neighbours in the targe phase-space bin
                    const size_t n_vertices_to_loop, // size of the first input array
                    const size_t* n_verts_in_bins, // n_vertices in each phase-space bin
                    const size_t n_bins, // total number of phase-space bins
                    const size_t indx_bin_to_use, // index of the newly added bin
                    const float *d_coord, // coordinates structured in bins {coords_of_bin0, coords_of_bin1, ..., coords_of_binN}
                    size_t n_coords, // number of dimentions
                    size_t n_neigh, // number of neighbours
                    float* d_dist, // distance matrix
                    int* d_indices, // indices matrix which corresponds to distance one
                    float max_radius = -1.0 // max. radius to search for neighbours
                    ){
    
    // loop to assign indices and distances to other vertices
    size_t start_vert = 0;
    size_t end_vert = 0;
    for (size_t i = 0; i < n_bins; i++){
        if (i<indx_bin_to_use){
            start_vert += n_verts_in_bins[i];
        }
        if (i==indx_bin_to_use){
            end_vert = start_vert + n_verts_in_bins[i];
            break;
        }
    }
    
    size_t n_vert = end_vert - start_vert;
    
    cout << "DEBUG: start_vert: " << start_vert << "; end_vert: " << end_vert << endl;

    for(size_t i = 0; i < n_vertices_to_loop; i++){
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
        if((n_vert+nfilled)<n_neigh){
            max_neighbours=(n_vert+nfilled);
        }
        
        float maxdistsq = 0;
        size_t maxidx_local = 0;
        if (nfilled>0){
            maxidx_local = searchLargestDistance(i_v,d_dist,n_neigh,maxdistsq);
        }

        // assigning loop - searching neighbouth for i_v
        for(size_t j_v=start_vert;j_v<end_vert;j_v++){
            if(i_v == j_v)
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


void set_defaults(
        int *d_indices,
        float *d_dist,
        const bool tf_compat,
        const int n_vert,
        const int n_neigh
){
    for(size_t i_v =0 ; i_v < n_vert ; i_v++){
        for(size_t n = 0; n < n_neigh; n++){

            if(n){
                if(tf_compat)
                    d_indices[I2D(i_v,n,n_neigh)] = i_v;
                else
                    d_indices[I2D(i_v,n,n_neigh)] = -1;
            }
            else{
                d_indices[I2D(i_v,n,n_neigh)] = i_v;
            }
            d_dist[I2D(i_v,n,n_neigh)] = 0;

        }
    }
}

struct BinnedCoords{ 
    // Usage example:
    //
    // float* binEdgesX = binEdges[0];
    // float* binEdgesY = binEdges[1];
    // size_t nBinX = n_bins[0];
    // size_t nBinY = n_bins[1];
    // size_t bin_index = I2D(iBinX, iBinY, nBinY);
    //
    // // iDim = 0..(n_coords-1)
    // d_coord_in_bins = coords[bin_index][I2D(iVertex,iDim,n_coords)]
    //

    float **coords; // *d_coord_in_bins[n_bins_x*n_bins_y]
    size_t n_coords; // iDim = 0..(n_coords-1)
    size_t *n_bins; // {n_bins_x, n_bins_y}
    size_t *counter; // nVertices = counter[I2D(iBinX, iBinY, nBinY)] 
    float **binEdges; 
}; 
  
void freeMemory(BinnedCoords &binnedCoords){
    for(size_t i=0; i<binnedCoords.n_bins[1]*binnedCoords.n_bins[2]; i++){
        delete[] binnedCoords.coords[i];
    }
    delete[] binnedCoords.binEdges[0];
    delete[] binnedCoords.binEdges[1];
    delete[] binnedCoords.binEdges;
    delete[] binnedCoords.coords;
    delete[] binnedCoords.n_bins;
    delete[] binnedCoords.counter;
}

// calculate min and max of X and Y coordinates among all vertices
// make bins with widths: (x_max-x_min)/n_bins and (y_max-y_min)/n_bins
void constructPhaseSpaceBins(size_t start_vert, size_t end_vert, size_t n_bins_x, size_t n_bins_y, const float *d_coord, size_t n_coords, BinnedCoords &out){

    out.n_coords = n_coords;
    out.coords = new float*[n_bins_x*n_bins_y];
    for(size_t i=0; i<n_bins_x*n_bins_y; i++){
        out.coords[i] = new float[n_coords*(end_vert-start_vert)];
    }
    out.counter = new size_t[n_bins_x*n_bins_y];
    for(int i=0; i<n_bins_x*n_bins_y; i++)
        out.counter[i] = 0;

    out.binEdges = new float*[2];
    out.binEdges[0] = new float[n_bins_x+1];
    out.binEdges[1] = new float[n_bins_y+1];

    out.n_bins = new size_t[2];
    out.n_bins[0] = n_bins_x;
    out.n_bins[1] = n_bins_y;

    const size_t n_coords_to_use = 2; // use only {x,y} coordinates

    // initialize min/max. take the first vertex to fill in an initial min/max values
    int i_v = 0; 
    float max[n_coords_to_use] = {d_coord[I2D(i_v,0,n_coords)], d_coord[I2D(i_v,1,n_coords)]};
    float min[n_coords_to_use] = {d_coord[I2D(i_v,0,n_coords)], d_coord[I2D(i_v,1,n_coords)]};


    // calculate min and max values for {X,Y} vertices coordinates
    for(size_t i_v = start_vert; i_v < end_vert; i_v ++){
        for(size_t iDim=0;iDim<n_coords_to_use;iDim++){
            float coord = d_coord[I2D(i_v,iDim,n_coords)];
            if (coord>max[iDim])
                max[iDim] = coord;
            if (coord<min[iDim])
                min[iDim] = coord;
        }
    }

    // define phase-space bin edges
    for(size_t iDim=0;iDim<n_coords_to_use;iDim++){
        for(size_t iBin=0; iBin<out.n_bins[iDim]; iBin++){
            out.binEdges[iDim][iBin] = min[iDim] + iBin*(max[iDim] - min[iDim])/out.n_bins[iDim];
        }
        out.binEdges[iDim][out.n_bins[iDim]] = max[iDim];
    }

    // define which vertices belong to which bin
    for(size_t i_v = start_vert; i_v < end_vert; i_v++){
        size_t indx[2] = {0,0};
        for(size_t iDim=0;iDim<n_coords_to_use;iDim++){
            float coord = d_coord[I2D(i_v,iDim,n_coords)];
            while (coord>out.binEdges[iDim][indx[iDim]+1]){
                indx[iDim] += 1;
            }
        }
        // copy all coordinates to new array which correspond to proper bin
        size_t bin_index = I2D(indx[0], indx[1], out.n_bins[1]);
        size_t n_filled_vertices = out.counter[bin_index];
        for(size_t iDim=0;iDim<n_coords;iDim++){
            out.coords[bin_index][I2D(n_filled_vertices,iDim,n_coords)] = d_coord[I2D(i_v,iDim,n_coords)];
        }
        out.counter[bin_index] += 1;
    }
}

float calculate2dDistanceToThePoint(float *pointCoord, size_t i_v, const float* d_coord, size_t n_coords){
    float distsq=0;
    for(size_t i=0;i<2;i++){
        float dist = d_coord[I2D(i_v,i,n_coords)] - pointCoord[i];
        distsq += dist*dist;
    }
    return sqrt(distsq);
}

void calculate2dDistanceToTheBinEdges(
    size_t* output_indices, // ???
    size_t &n_output_vertices, // ???
    const size_t* input_indices, // vertices for which we want to find neighbours in the targe phase-space bin
    const size_t n_input_vertices, // size of the first input array
    const float* target_bin_coords, // {x1, y1, x2, y2} coords of the target bin
    const float *d_coord, // coordinates structured in bins {coords_of_bin0, coords_of_bin1, ..., coords_of_binN}
    size_t n_coords, // number of dimentions
    size_t n_neigh, // number of neighbours
    float* d_dist, // distance matrix
    int* d_indices, // indices matrix which corresponds to distance one
    float max_radius = -1.0 // max. radius to search for neighbours
    ){

    size_t tmp_indices[n_input_vertices]; // temporary array for output indices
    n_output_vertices = 0;
    
    for(size_t i = 0; i < n_input_vertices; i++){
        size_t i_v = input_indices[i];
        
        // safety check
        float x = d_coord[I2D(i_v,0,n_coords)];
        float y = d_coord[I2D(i_v,1,n_coords)];
        cout << "x: " << x << "; y: " << y << endl;
        if (target_bin_coords[0]>x && target_bin_coords[2]<x
            && target_bin_coords[1]>y && target_bin_coords[3]<y)
            continue; // i_v belongs to the target bin 
        
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
        
        cout << "n_found_neighbours: " << n_found_neighbours << "; n_neigh: " << n_neigh << endl;
        
        // include i_v for output if it doesn't have enough neighbours
        if (n_found_neighbours<n_neigh){
            tmp_indices[n_output_vertices] = i_v;
            n_output_vertices += 1;
            continue;
        }
        
        // find the distance to the farthermost neighbour
        float maxdistsq = 0; // largest distance
        size_t maxidx_local = searchLargestDistance(i_v,d_dist,n_neigh,maxdistsq);
        
        // find the distance to the target bin
        float distance_to_bin = 0.0;
        if ((target_bin_coords[0]>x && target_bin_coords[2]<x) || (target_bin_coords[1]>y && target_bin_coords[3]<y)){
            size_t iDim = 0;
            if (target_bin_coords[0]>y && target_bin_coords[2]<y)
                iDim = 1;
            
            float lowBinEdge = target_bin_coords[iDim];
            float highBinEdge = target_bin_coords[iDim+2];
            float d1 = abs(d_coord[I2D(i_v,iDim,n_coords)] - lowBinEdge);
            float d2 = abs(d_coord[I2D(i_v,iDim,n_coords)] - highBinEdge);
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
        
        cout << "distance_to_bin: " << distance_to_bin << "; maxdistsq: " << maxdistsq << endl;
        
        if (distance_to_bin<maxdistsq){
            tmp_indices[n_output_vertices] = i_v;
            n_output_vertices += 1;
        }
    }
    
    // return array of proper size
    //output_indices = new size_t[n_output_vertices];
    for (int i=0; i<n_output_vertices; i++){
        output_indices[i] = tmp_indices[i];
    }
    
    return;
}


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
        const int n_bins_y) {

    //really no buffering at all here

    const size_t start_vert = d_row_splits[j_rs];
    const size_t end_vert = d_row_splits[j_rs+1];

    BinnedCoords binnedCoords;
    constructPhaseSpaceBins(start_vert, end_vert, n_bins_x, n_bins_y, d_coord, n_coords, binnedCoords);


    float* binEdgesX = binnedCoords.binEdges[0];
    float* binEdgesY = binnedCoords.binEdges[1];

    // DEBUG PRINTOUT
    for (size_t iBinX=0; iBinX<n_bins_x; iBinX++){
        for (size_t iBinY=0; iBinY<n_bins_y; iBinY++){
            cout << endl;
            size_t bin_index = I2D(iBinX, iBinY, n_bins_y);
            float* d_coord_in_bin = binnedCoords.coords[bin_index];
            size_t n_vert_in_bin = binnedCoords.counter[I2D(iBinX, iBinY, n_bins_y)];
            cout << "Bin: " << bin_index << "(" << iBinX << "," << iBinY << "); nVert: " << n_vert_in_bin << endl;
            cout << "Bin Edges. X: " << binEdgesX[iBinX] << "-" << binEdgesX[iBinX+1] << "; Y: " << binEdgesY[iBinY] << "-" << binEdgesY[iBinY+1] << endl;
            for(size_t iVert=0; iVert<n_vert_in_bin; iVert++){
                cout << "Vertex: " << iVert << endl;
                for(size_t iDim=0; iDim<n_coords; iDim++){
                    float coord = d_coord_in_bin[I2D(iVert,iDim,n_coords)];
                    cout << coord << " ";
                }
                cout << endl;
            }
        }
    }// DEBUG PRINTOUT

    // const size_t outArraySize = 1000;
    // float d_dist[outArraySize];
    // int d_indices[outArraySize];
    // for (size_t i=0; i<outArraySize; i++){
    //     d_dist[i] = 0.0;
    //     d_indices[i] = -1;
    // }

    float d_coord_bins[(end_vert-start_vert)*2];

    size_t n_verts_in_bins[n_bins_x*n_bins_y];

    size_t counter = 0;
    for (size_t iBinX=0; iBinX<n_bins_x; iBinX++){
        for (size_t iBinY=0; iBinY<n_bins_y; iBinY++){
            size_t bin_index = I2D(iBinX, iBinY, n_bins_y);
            float* d_coord_in_bin = binnedCoords.coords[bin_index];
            size_t n_vert_in_bin = binnedCoords.counter[I2D(iBinX, iBinY, n_bins_y)];
            for (int i=0; i<n_vert_in_bin*2;i++){
                d_coord_bins[counter] = d_coord_in_bin[i];
                counter += 1;
            }
            n_verts_in_bins[bin_index] = n_vert_in_bin;       
        }
    }



	for (size_t iBinX=0; iBinX<n_bins_x; iBinX++){
		for (size_t iBinY=0; iBinY<n_bins_y; iBinY++){
		    size_t bin_index = I2D(iBinX, iBinY, n_bins_y);
		    const float binCoords[4] = {binEdgesX[iBinX], binEdgesY[iBinY], binEdgesX[iBinX+1], binEdgesY[iBinY+1]};

		    // prepare indices input array
		    size_t start_vert = 0;
		    size_t end_vert = 0;
		    for (size_t i = 0; i <= bin_index; i++){
		        if (i<bin_index){
		            start_vert += n_verts_in_bins[i];
		        }
		        if (i==bin_index){
		            end_vert = start_vert + n_verts_in_bins[i];
		            break;
		        }
		    }
		    size_t *tmp_indices = new size_t[n_verts_in_bins[bin_index]];
		    for (int i=0; i<n_verts_in_bins[bin_index]; i++){
		        tmp_indices[i] = start_vert+i;
		    }
		    
		    // find neighbours
		    findNeighbours(tmp_indices, //indices_of_vert_to_find_new_neigh,
		        n_verts_in_bins[bin_index], //n_vertices_to_loop, // size of the first input array
		        n_verts_in_bins, 
		        n_bins_x*n_bins_y, //n_bins,
		        bin_index, //indx_bin_to_use, // index of the newly added bin
		        d_coord_bins,
		        n_coords,
		        n_neigh, //n_neigh,
		        d_dist,
		        d_indices,
		        -1.0
		        );
		    
		    // clean up tmp array fpr output
		    size_t output_indices[n_verts_in_bins[bin_index]];
		    for (size_t i=0; i<n_verts_in_bins[bin_index]; i++){
		        output_indices[i] = -1;
		    }
		    size_t n_output_vertices;
		    
		    calculate2dDistanceToTheBinEdges(
		        output_indices, // ???
		        n_output_vertices, // ???
		        tmp_indices, //input_indices, // vertices for which we want to find neighbours in the targe phase-space bin
		        n_verts_in_bins[bin_index], //n_input_vertices, // size of the first input array
		        binCoords, //target_bin_coords, // {x1, y1, x2, y2} coords of the target bin
		        d_coord_bins, // coordinates structured in bins {coords_of_bin0, coords_of_bin1, ..., coords_of_binN}
		        n_coords, //n_coords, // number of dimentions
		        n_neigh, // number of neighbours
		        d_dist, // distance matrix
		        d_indices, // indices matrix which corresponds to distance one
		        -1.0//max_radius = -1.0 // max. radius to search for neighbours
		        );
		    
		    cout << "# OF NOT FINISHED VERTICES: " << n_output_vertices << endl;
		    for (int i=0; i<n_output_vertices; i++){
		        cout << output_indices[i] << " ";
		    }
		    
		    if (n_output_vertices>0){
		        
		        for (size_t iBinX2=0; iBinX2<n_bins_x; iBinX2++){
		            for (size_t iBinY2=0; iBinY2<n_bins_y; iBinY2++){
		                size_t bin_index2 = I2D(iBinX2, iBinY2, n_bins_y);
		                if (bin_index == bin_index2)
		                    continue;
		                const float binCoords[4] = {binEdgesX[iBinX2], binEdgesY[iBinY2], binEdgesX[iBinX2+1], binEdgesY[iBinY2+1]};
		                
		                // find neighbours
		                findNeighbours(output_indices, //indices_of_vert_to_find_new_neigh,
		                    n_output_vertices, //n_vertices_to_loop, // size of the first input array
		                    n_verts_in_bins, 
		                    n_bins_x*n_bins_y, //n_bins,
		                    bin_index2, //indx_bin_to_use, // index of the newly added bin
		                    d_coord_bins,
		                    n_coords,
		                    n_neigh, //n_neigh,
		                    d_dist,
		                    d_indices,
		                    -1.0
		                    );
		            }
		        }
		    } // if (n_output_vertices>0)
		    
		} // iBinY
	} // iBinX

	for (size_t iBinX=0; iBinX<n_bins_x; iBinX++){
		for (size_t iBinY=0; iBinY<n_bins_y; iBinY++){
		    size_t bin_index = I2D(iBinX, iBinY, n_bins_y);
		    size_t start_vert = 0;
		    size_t end_vert = 0;
		    for (size_t i = 0; i <= bin_index; i++){
		        if (i<bin_index){
		            start_vert += n_verts_in_bins[i];
		        }
		        if (i==bin_index){
		            end_vert = start_vert + n_verts_in_bins[i];
		            break;
		        }
		    }
		    
		    cout << endl << endl << "Bin: " << bin_index << "(" << iBinX << "," << iBinY << "); start_vert: " << start_vert << "; end_vert: " << end_vert << endl;

		    
		    for (int i=start_vert; i<end_vert; i++){
		        cout << endl << i << ": ";
		        for (int j=0; j<n_neigh; j++){
		            cout << d_indices[I2D(i,j,n_neigh)] << " ";
		        }
		    }
		    for (int i=start_vert; i<end_vert; i++){
		        cout << endl << i << ": ";
		        for (int j=0; j<n_neigh; j++){
		            cout << d_dist[I2D(i,j,n_neigh)] << " ";
		        }
		    }
		}
	}


    // free allocated memory
    freeMemory(binnedCoords);
    // delete[] distances;

}

// CPU specialization
template<typename dummy>
struct NewKnnOpFunctor<CPUDevice, dummy> {
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


        set_defaults(d_indices,
                d_dist,
                tf_compat,
                n_vert,
                n_neigh);
        //really no buffering at all here

        for(size_t j_rs=0;j_rs<n_rs-1;j_rs++){
            new_knn_kernel(d_coord,
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
class NewKnnOp : public OpKernel {
public:
    explicit NewKnnOp(OpKernelConstruction *context) : OpKernel(context) {
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


        NewKnnOpFunctor<Device, int>()(
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

REGISTER_KERNEL_BUILDER(Name("NewKnn").Device(DEVICE_CPU), NewKnnOp<CPUDevice>);

#ifdef GOOGLE_CUDA
extern template struct NewKnnOpFunctor<GPUDevice, int>;
REGISTER_KERNEL_BUILDER(Name("NewKnn").Device(DEVICE_GPU), NewKnnOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}//functor
}//tensorflow
