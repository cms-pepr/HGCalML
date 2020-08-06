
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
    return distsq;
}

int searchLargestDistance(int i_v, float* d_dist, int n_neigh, float& maxdist){

    maxdist=0;
    int maxidx=0;
    // uncomment the line below and comment 3 lines after it if we don't want to have in the neighbour indices matrix index of the vetex itself
    // FIXME TODO I prefil indices matrix with all "-1" while it is prefilled different for other kernels. I can use it if I remove requirement of counting "-1" in indices
    for(size_t n=0;n<n_neigh;n++){ //0 is self
    // if(n_neigh < 2)
    //    return maxidx;
    // for(size_t n=1;n<n_neigh;n++){ //0 is self

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
                    const size_t indx_bin_to_use, // index of the newly added bin
                    const size_t* index_map_to_bins, // index_map_to_bins[i_v] -> bin number to which vertex belong
                    const size_t* n_vertices_in_bin,
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
        if((n_vertices_in_bin[indx_bin_to_use]+nfilled)<n_neigh){
            max_neighbours=(n_vertices_in_bin[indx_bin_to_use]+nfilled);
        }
        
        float maxdistsq = 0;
        size_t maxidx_local = 0;
        if (nfilled>0){
            maxidx_local = searchLargestDistance(i_v,d_dist,n_neigh,maxdistsq);
        }
        
        // assigning loop - searching neighbouth for i_v
        for(size_t j_v=start_vert;j_v<end_vert;j_v++){
            if(index_map_to_bins[j_v]!=indx_bin_to_use)
                continue;
            // 2 lines below are NEEDED if we don't want to have in the neighbour indices matrix index of the vetex itself
//             if(i_v == j_v)
//                 continue;
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
 
// calculate min and max of X and Y coordinates among all vertices
// make bins with widths: (x_max-x_min)/n_bins and (y_max-y_min)/n_bins
void constructPhaseSpaceBins(const float *d_coord, size_t n_coords, size_t start_vert, 
                             size_t end_vert, size_t n_bins_x, size_t n_bins_y, 
                             float *coords_2d_bins, size_t *n_vertices_in_bin,
                             size_t *indices_of_vertices_in_bin){

    // use only {x,y} coordinates
    // initialize min/max. take the first vertex to fill in an initial min/max values
    int i_v = 0; 
    float max[2] = {d_coord[I2D(i_v,0,n_coords)], d_coord[I2D(i_v,1,n_coords)]};
    float min[2] = {d_coord[I2D(i_v,0,n_coords)], d_coord[I2D(i_v,1,n_coords)]};

    // calculate min and max values for {X,Y} vertices coordinates
    for(size_t i_v = start_vert; i_v < end_vert; i_v ++){
        for(size_t iDim=0;iDim<2;iDim++){
            float coord = d_coord[I2D(i_v,iDim,n_coords)];
            if (coord>max[iDim])
                max[iDim] = coord;
            if (coord<min[iDim])
                min[iDim] = coord;
        }
    }
    // to avoind any vertices on the outer bin boundaries
    for(size_t iDim=0;iDim<2;iDim++){
        max[iDim] += 0.0001*(max[iDim] - min[iDim]);
        min[iDim] -= 0.0001*(max[iDim] - min[iDim]);
    }

    float binEdgesX[n_bins_x+1];
    float binEdgesY[n_bins_y+1];
    float *binEdges[2] = {binEdgesX, binEdgesY};
    
    // define phase-space bin edges
    for (size_t iBinX=0; iBinX<n_bins_x; iBinX++){
        for (size_t iBinY=0; iBinY<n_bins_y; iBinY++){
            size_t bin_index = I2D(iBinX, iBinY, n_bins_y);
            coords_2d_bins[4*bin_index] = min[0] + iBinX*(max[0] - min[0])/n_bins_x;
            coords_2d_bins[4*bin_index+1] = min[1] + iBinY*(max[1] - min[1])/n_bins_y;
            coords_2d_bins[4*bin_index+2] = min[0] + (iBinX+1)*(max[0] - min[0])/n_bins_x;
            coords_2d_bins[4*bin_index+3] = min[1] + (iBinY+1)*(max[1] - min[1])/n_bins_y;
            binEdges[0][iBinX] = coords_2d_bins[4*bin_index];
            binEdges[1][iBinY] = coords_2d_bins[4*bin_index+1];
        }
    }
    binEdges[0][n_bins_x] = max[0];
    binEdges[1][n_bins_y] = max[1];

    // define which vertices belong to which bin
    for(size_t i_v = start_vert; i_v < end_vert; i_v++){
        size_t indx[2] = {0,0};
        for(size_t iDim=0;iDim<2;iDim++){
            float coord = d_coord[I2D(i_v,iDim,n_coords)];
            indx[iDim] = (size_t)((coord - binEdges[iDim][0])/(binEdges[iDim][1]-binEdges[iDim][0]));
        }

        size_t bin_index = I2D(indx[0], indx[1], n_bins_y);
        n_vertices_in_bin[bin_index] += 1;
        indices_of_vertices_in_bin[i_v] = bin_index;
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
    size_t* output_indices, // ???
    size_t &n_output_vertices, // ???
    const size_t* input_indices, // vertices for which we want to find neighbours in the targe phase-space bin
    const size_t n_input_vertices, // size of the first input array
    const float* target_bin_coords, // {x1, y1, x2, y2} coords of the target bin
    const float *d_coord,
    const size_t n_coords, // number of dimentions
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
        // cout << "x: " << x << "; y: " << y << endl;
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
            tmp_indices[n_output_vertices] = i_v;
            n_output_vertices += 1;
            continue;
        }
        
        // find the distance to the farthermost neighbour
        float maxdistsq = 0; // largest distance
        size_t maxidx_local = searchLargestDistance(i_v,d_dist,n_neigh,maxdistsq);
        
        // cout << "n_found_neighbours: " << n_found_neighbours << "; n_neigh: " << n_neigh << endl;
        // cout << "maxdistsq: " << maxdistsq << endl;
        
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

	// FIXME TODO implementation requires that input array of indices has all values equal to -1.
    for(int i=0; i<n_neigh*(end_vert-start_vert); i++){
        d_indices[i] = -1;
    }

	float coords_2d_bins[4*n_bins_x*n_bins_y]; // {x1_bin_1, y1_bin_1, x2_bin_2, y2_bin_2, ...}
	size_t n_vertices_in_bin[n_bins_x*n_bins_y];
    for(int i=0; i<n_bins_x*n_bins_y; i++){
        n_vertices_in_bin[i] = 0;
    }
	size_t indices_of_vertices_in_bin[(end_vert-start_vert)];

	constructPhaseSpaceBins(d_coord, n_coords, start_vert, end_vert, n_bins_x, n_bins_y,
		                    coords_2d_bins, n_vertices_in_bin, indices_of_vertices_in_bin);  

	for (size_t iBinX=0; iBinX<n_bins_x; iBinX++){
		for (size_t iBinY=0; iBinY<n_bins_y; iBinY++){
		    size_t bin_index = I2D(iBinX, iBinY, n_bins_y);
		    float *binCoords = new float[4];
		    for (int i=0; i<4; i++){
		        binCoords[i] = coords_2d_bins[4*bin_index + i];
		    }
		    
		    // prepare indices input array
		    size_t *tmp_indices = new size_t[n_vertices_in_bin[bin_index]];
		    size_t counter = 0;
		    for(size_t i_v = start_vert; i_v < end_vert; i_v ++){
		        if (indices_of_vertices_in_bin[i_v]==bin_index){
		            tmp_indices[counter] = i_v;
		            counter += 1;
		        }
		    }
		    
		    findNeighbours(tmp_indices, // vertices for which we want to find neighbours in the targe phase-space bin
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

		    // clean up tmp array for output
		    size_t output_indices[n_vertices_in_bin[bin_index]];
		    for (size_t i=0; i<n_vertices_in_bin[bin_index]; i++){
		        output_indices[i] = -1;
		    }
		    size_t n_output_vertices;

			for (size_t iBinX2=0; iBinX2<n_bins_x; iBinX2++){
		        for (size_t iBinY2=0; iBinY2<n_bins_y; iBinY2++){
		            size_t bin_index2 = I2D(iBinX2, iBinY2, n_bins_y);
		            if (bin_index2 == bin_index)
		                continue;
		            for (int i=0; i<4; i++){
		                binCoords[i] = coords_2d_bins[4*bin_index2 + i];
		            }

		            calculate2dDistanceToTheBinEdges(
		                output_indices, // ???
		                n_output_vertices, // ???
		                tmp_indices, //input_indices, // vertices for which we want to find neighbours in the targe phase-space bin
		                n_vertices_in_bin[bin_index], //n_input_vertices, // size of the first input array
		                binCoords, //target_bin_coords, // {x1, y1, x2, y2} coords of the target bin
		                d_coord,
		                n_coords, //n_coords, // number of dimentions
		                n_neigh, // number of neighbours
		                d_dist, // distance matrix
		                d_indices, // indices matrix which corresponds to distance one
		                -1.0//max_radius = -1.0 // max. radius to search for neighbours
		                );

		            if (n_output_vertices>0){

                        cout << "DEBUG: iBinX: " << iBinX << "; iBinY: " << iBinY << "; iBinX2: " << iBinX2 << "; iBinY2: " << iBinY2 << "; n_output_vertices: " << n_output_vertices << ";\tn_vert["<< iBinX << ","<<iBinY<<"]: " << n_vertices_in_bin[bin_index] << "; n_vert["<<iBinX2 << ","<<iBinY2<<"]: " << n_vertices_in_bin[bin_index2] << endl;
		                
		                // find neighbours
		                findNeighbours(output_indices,
		                    n_output_vertices,
		                    bin_index2, // index of the newly added bin
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
		            } // if (n_output_vertices>0)
		        } // iBinY2
		    } // iBinX2
		} // iBinY
	} // iBinX1

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
