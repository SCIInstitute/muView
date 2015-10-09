
#include <heart/cuda_common.h>
#include <Data/Mesh/TetMesh.h>
#include <SCI/Vex3.h>

#define BLOCK_SIZE 256

/*
union Point {
    struct {
        float x, y, z;
    };
    float data[3];
};


__device__ inline Point Difference(Point p0, Point p1){
    Point ret;
    ret.x = p0.x-p1.x;
    ret.y = p0.y-p1.y;
    ret.z = p0.z-p1.z;
    return ret;
}

__device__ inline float Distance(Point p0, Point p1){
    Point tmp = Difference( p0, p1 );
    return sqrt( tmp.x*tmp.x + tmp.y*tmp.y + tmp.z*tmp.z );
}


*/


__host__ __device__ inline SCI_Vex3 SCI_cross(const SCI_Vex3& v1, const SCI_Vex3& v2){
    SCI_Vex3 ret;
    ret.x = v1.y*v2.z - v1.z*v2.y;
    ret.y = v1.z*v2.x - v1.x*v2.z;
    ret.z = v1.x*v2.y - v1.y*v2.x;
    return ret;
}

__host__ __device__ inline SCI_Vex3 SCI_subtract(const SCI_Vex3& v1, const SCI_Vex3& v2){
    SCI_Vex3 ret;
    ret.x = v1.x - v2.x;
    ret.y = v1.y - v2.y;
    ret.z = v1.z - v2.z;
    return ret;
}

__host__ __device__ inline float SCI_dot(const SCI_Vex3& v1, const SCI_Vex3& v2){
        return ((v1.x*v2.x) + (v1.y*v2.y) + (v1.z*v2.z));
}



//__device__ bool isPointInsideTet( SCI_Vex3 t0, SCI_Vex3 t1, SCI_Vex3 t2, SCI_Vex3 t3, SCI_Vex3 p ){
__device__ bool isPointInsideTet( SCI_Vex3 * v, SCI_Vex3 p ){
    for(int i = 0; i < 4; i++){
        SCI_Vex3 a = v[i];
        SCI_Vex3 ba = SCI_subtract( v[(i+1)%4], a );
        SCI_Vex3 ca = SCI_subtract( v[(i+2)%4], a );
        SCI_Vex3 da = SCI_subtract( v[(i+3)%4], a );
        SCI_Vex3 n = SCI_cross( ba, ca );
        if( SCI_dot(n, da ) * SCI_dot(n, SCI_subtract(p,a) ) < 0 ) return false;
    }
    return true;
}

__global__ void voxel_associativity_kernel(Data_Mesh_Tetmesh * tets, int tetN, SCI_Vex3 * points, int pointN, SCI_Vex3 * vox_centers, int * vox_assoc, int vox_dim){

    SCI_Vex3 tmpPoints[4];

    int thrd = threadIdx.x;

    int vox_id = blockIdx.x * 256 + threadIdx.x;

    for(int i = 0; i < tetN; i++ ){

        // load (up to) the next 32 tets into shared memory
        tmpPoints[0] = points[ tets[ i ].data[0] ];
        tmpPoints[1] = points[ tets[ i ].data[1] ];
        tmpPoints[2] = points[ tets[ i ].data[2] ];
        tmpPoints[3] = points[ tets[ i ].data[3] ];

        if( isPointInsideTet( tmpPoints, vox_centers[vox_id] ) ){
            vox_assoc[vox_id] = i;
        }
    }

    /*
    for(int i = 0; i < tetN; i++ ){

        // load (up to) the next 32 tets into shared memory
        if( (i+thrd/4) < tetN ){
            tmpPoints[thrd] = points[ tets[ i+thrd/4 ].data[thrd%4] ];
        }
        __syncthreads( );

        for( int j = 0; j < 128/4 && i < tetN; j++, i++ ){
            //if( isPointInsideTet( tmpPoints[4*j+0], tmpPoints[4*j+1], tmpPoints[4*j+2], tmpPoints[4*j+3], vox_centers[vox_id] ) ){
            if( isPointInsideTet( tmpPoints, 4*j, vox_centers[vox_id] ) ){
                vox_assoc[vox_id] = i;
            }
        }
    }
    */

    //vox_assoc[vox_id] = 0;
}

__global__ void kernel(int *array)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  array[index] = 7;
}

extern "C"
void voxel_associativity( Data_Mesh_Tetmesh * h_tets, int tetN, SCI_Vex3 * h_points, int pointN, SCI_Vex3 * h_vox_centers, int * h_vox_assoc, int vox_dim ){

    Data_Mesh_Tetmesh * d_tets;
    SCI_Vex3          * d_points;
    int               * d_vox_assoc;
    SCI_Vex3          * d_vox_centers;

    int dev = findCudaDevice( 0, 0 );
    if( dev == -1 ) {
        return;
    }

    printf("%i",vox_dim);

    checkCudaErrors( cudaMalloc( (void**) &d_tets,        sizeof( Data_Mesh_Tetmesh ) * tetN                        ) );
    checkCudaErrors( cudaMalloc( (void**) &d_points,      sizeof( SCI_Vex3 )          * pointN                      ) );
    checkCudaErrors( cudaMalloc( (void**) &d_vox_assoc,   sizeof( int )               * vox_dim * vox_dim * vox_dim ) );
    checkCudaErrors( cudaMalloc( (void**) &d_vox_centers, sizeof( SCI_Vex3 )          * vox_dim * vox_dim * vox_dim ) );

    checkCudaErrors( cudaMemcpy( d_tets,        h_tets,        sizeof( Data_Mesh_Tetmesh ) * tetN,                        cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy( d_points,      h_points,      sizeof( SCI_Vex3 )          * pointN,                      cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy( d_vox_assoc,   h_vox_assoc,   sizeof( int )               * vox_dim * vox_dim * vox_dim, cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy( d_vox_centers, h_vox_centers, sizeof( SCI_Vex3 )          * vox_dim * vox_dim * vox_dim, cudaMemcpyHostToDevice ) );

    int block_size = 256;
    int grid_size  = vox_dim*vox_dim*vox_dim/256;
    printf("Launching kernel\n"); fflush(stdout);
    //voxel_associativity_kernel<<< grid_size, block_size >>>( d_tets, tetN, d_points, pointN, d_vox_centers, d_vox_assoc, vox_dim );
    for(int tet = 0; tet < 128; tet+=128){
        voxel_associativity_kernel<<< grid_size, block_size >>>( d_tets+tet, 128, d_points, pointN, d_vox_centers, d_vox_assoc, vox_dim );
    }

    checkCudaErrors( cudaMemcpy( h_vox_assoc, d_vox_assoc, sizeof( int ) * vox_dim * vox_dim * vox_dim,  cudaMemcpyDeviceToHost ) );

    checkCudaErrors( cudaFree( d_tets        ) );
    checkCudaErrors( cudaFree( d_points      ) );
    checkCudaErrors( cudaFree( d_vox_assoc   ) );
    checkCudaErrors( cudaFree( d_vox_centers ) );

    cudaDeviceReset();


    /*
    int num_elements = 256;

    int num_bytes = num_elements * sizeof(int);

    // pointers to host & device arrays
    int *device_array = 0;
    int *host_array = 0;

    // malloc a host array
    host_array = (int*)malloc(num_bytes);

    // cudaMalloc a device array
    cudaMalloc((void**)&device_array, num_bytes);

    int block_size = 128;
    int grid_size = num_elements / block_size;

    kernel<<<grid_size,block_size>>>(device_array);

    // download and inspect the result on the host:
    cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);

    // print out the result element by element
    for(int i=0; i < num_elements; ++i)
    {
      printf("%d ", host_array[i]);
    }

    // deallocate memory
    free(host_array);
    cudaFree(device_array);
    */

}
