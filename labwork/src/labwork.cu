#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#define ACTIVE_THREADS 4

int main(int argc, char **argv) {
    printf("USTH ICT Master 2018, Advanced Programming for HPC.\n");
    if (argc < 2) {
        printf("Usage: labwork <lwNum> <inputImage>\n");
        printf("   lwNum        labwork number\n");
        printf("   inputImage   the input file name, in JPEG format\n");
        return 0;
    }

    int lwNum = atoi(argv[1]);
    std::string inputFilename;

    // pre-initialize CUDA to avoid incorrect profiling
    printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if (lwNum != 2 ) {
        inputFilename = std::string(argv[2]);
        labwork.loadInputImage(inputFilename);
    }

    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    timer.start();
    float seqTime, ompTime;
    switch (lwNum) {
        case 1:
            timer.start();
            labwork.labwork1_CPU();
	    seqTime = timer.getElapsedTimeInMilliSec();
            printf("labwork 1 CPU Sequential  ellapsed %.1fms\n", lwNum, seqTime);
            labwork.saveOutputImage("labwork1-cpu-out.jpg");
            timer.start();
            labwork.labwork1_OpenMP();
	    ompTime = timer.getElapsedTimeInMilliSec();
            printf("labwork 1 CPU OpenMP ellapsed %.1fms\n", lwNum, ompTime);
            labwork.saveOutputImage("labwork1-openmp-out.jpg");

	    printf("OpenMP / Sequential ellapses: %.2f%% \n", ompTime/seqTime*100);
            break;
        case 2:
            labwork.labwork2_GPU();
            break;
        case 3:
            labwork.labwork3_GPU();
            labwork.saveOutputImage("labwork3-gpu-out.jpg");
            break;
        case 4:
            labwork.labwork4_GPU();
            labwork.saveOutputImage("labwork4-gpu-out.jpg");
            break;
        case 5:
            float cpuTime, gpuTime, gpuNonShared;
            timer.start();
            labwork.labwork5_CPU();
            cpuTime = timer.getElapsedTimeInMilliSec();

            labwork.saveOutputImage("labwork5-cpu-out.jpg");

            timer.start();
            labwork.labwork5_GPU_NonSharedMemory();
            gpuNonShared = timer.getElapsedTimeInMilliSec();

            timer.start();
            labwork.labwork5_GPU();
            gpuTime = timer.getElapsedTimeInMilliSec();

            labwork.saveOutputImage("labwork5-gpu-out.jpg");

            printf("Labwork 5 CPU ellapsed %.1fms\n", lwNum, cpuTime);
            printf("Labwork 5 GPU with shared memory ellapsed %.1fms\n", lwNum, gpuTime);
            printf("Labwork 5 GPU without shared memory ellapsed %.1fms\n", lwNum, gpuNonShared);
            printf("GPU with shared memory is faster than GPU without shared memory by: %.2f times\n", gpuNonShared/gpuTime);
            printf("GPU with shared memory is faster than CPU by: %.2f times\n", cpuTime/gpuTime);
            break;
        case 6:
            labwork.labwork6_GPU();
            labwork.saveOutputImage("labwork6-gpu-out.jpg");
            break;
        case 7:
            labwork.labwork7_GPU();
            labwork.saveOutputImage("labwork7-gpu-out.jpg");
            break;
        case 8:
            labwork.labwork8_GPU();
            labwork.saveOutputImage("labwork8-gpu-out.jpg");
            break;
        case 9:
            labwork.labwork9_GPU();
            labwork.saveOutputImage("labwork9-gpu-out.jpg");
            break;
        case 10:
            labwork.labwork10_GPU();
            labwork.saveOutputImage("labwork10-gpu-out.jpg");
            break;
    }
    printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
}

void Labwork::loadInputImage(std::string inputFileName) {
    inputImage = jpegLoader.load(inputFileName);
}

void Labwork::saveOutputImage(std::string outputFileName) {
    jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
}

void Labwork::labwork1_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++) {		// let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

void Labwork::labwork1_OpenMP() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
//#pragma omp parallel for schedule(dynamic) 
//#pragma omp parallel for schedule(static) 
//#pragma omp parallel for
#pragma omp target teams num_teams(4)
   {
    printf("Team number: %d\n", omp_get_team_num());
    for (int j = 0; j < 100; j++) {		// let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
    }
}

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (devProp.minor == 1) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

void Labwork::labwork2_GPU() {
   int numDevices = 0;
   cudaGetDeviceCount(&numDevices);
   printf("Number of devices: %d\n", numDevices);
   for(int i=0; i<numDevices; i++) {
       cudaDeviceProp prop;
       cudaGetDeviceProperties(&prop, i);
       printf("Device #%d\n", i);
       printf("-Name: %s\n", prop.name);
       printf("-Cores: %d\n", getSPcores(prop));
       printf("-Clockrate: %.2f Mhz\n", prop.clockRate*1.0/1000);
       printf("-Multiprocessors count: %d\n", prop.multiProcessorCount);
       printf("-Warp size: %d \n", prop.warpSize);
       printf("-Memory clockrate: %.2f Mhz\n", prop.memoryClockRate*1.0/1000);
       printf("-Memory Bus width: %d\n\n", prop.memoryBusWidth);
   }
}

// implement grayscale kernel
__global__ void grayscale(uchar3 *input, uchar3 *output) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3;
    output[tid].z = output[tid].y = output[tid].x;
}

void Labwork::labwork3_GPU() {
    int pixelCount = inputImage->width * inputImage->height; // number of pixel
    int blockSize = 1024;
    int numBlock = pixelCount / blockSize;
    uchar3 *devInput, *devGray; // declare device pointers

    // Allocate device memory
    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devGray, pixelCount * sizeof(uchar3));
    
    // Copy from host to device
    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
    
    // Call kernel
    grayscale<<<numBlock, blockSize>>>(devInput, devGray);

    // Allocate host memory
    outputImage = (char*) malloc(pixelCount * sizeof(char) * 3);

    // Copy from Device to host
    cudaMemcpy(outputImage, devGray, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(devInput);
    cudaFree(devGray);
}

// implement grayscale2D kernel
__global__ void grayscale2D(uchar3 *input, uchar3 *output, int imageWidth, int imageHeight) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx>=imageWidth) return; // check for out of bound index
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidy>=imageHeight) return; // check for out of bound index
    int tid = tidx + tidy * imageWidth;
    output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3;
    output[tid].z = output[tid].y = output[tid].x;
}
void Labwork::labwork4_GPU() {
    int pixelCount = inputImage->width * inputImage->height; // number of pixel
    // int blockSizex = 32;
    int blockSizex = 8;
    int blockSizey = blockSizex;
    dim3 gridSize = dim3((inputImage->width+blockSizex-1)/blockSizex, (inputImage->height+blockSizey-1)/blockSizey);
    dim3 blockSize = dim3(blockSizex, blockSizey);
    uchar3 *devInput, *devGray; // declare device pointers

    // Allocate device memory
    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devGray, pixelCount * sizeof(uchar3));
    
    // Copy from host to device
    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
    
    // Call kernel
    grayscale2D<<<gridSize, blockSize>>>(devInput, devGray, inputImage->width, inputImage->height);

    // Allocate host memory
    outputImage = (char*) malloc(pixelCount * sizeof(char) * 3);

    // Copy from Device to host
    cudaMemcpy(outputImage, devGray, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(devInput);
    cudaFree(devGray);
}

// CPU implementation of Gaussian Blur
void Labwork::labwork5_CPU() {
    int kernel[] = { 0, 0, 1, 2, 1, 0, 0,  
                     0, 3, 13, 22, 13, 3, 0,  
                     1, 13, 59, 97, 59, 13, 1,  
                     2, 22, 97, 159, 97, 22, 2,  
                     1, 13, 59, 97, 59, 13, 1,  
                     0, 3, 13, 22, 13, 3, 0,
                     0, 0, 1, 2, 1, 0, 0 };
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = (char*) malloc(pixelCount * sizeof(char) * 3);
    for (int row = 0; row < inputImage->height; row++) {
        for (int col = 0; col < inputImage->width; col++) {
            int sum = 0;
            int c = 0;
            for (int y = -3; y <= 3; y++) {
                for (int x = -3; x <= 3; x++) {
                    int i = col + x;
                    int j = row + y;
                    if (i < 0) continue;
                    if (i >= inputImage->width) continue;
                    if (j < 0) continue;
                    if (j >= inputImage->height) continue;
                    int tid = j * inputImage->width + i;
                    unsigned char gray = (inputImage->buffer[tid * 3] + inputImage->buffer[tid * 3 + 1] + inputImage->buffer[tid * 3 + 2])/3;
                    int coefficient = kernel[(y+3) * 7 + x + 3];
                    sum = sum + gray * coefficient;
                    c += coefficient;
                }
            }
            sum /= c;
            int posOut = row * inputImage->width + col;
            outputImage[posOut * 3] = outputImage[posOut * 3 + 1] = outputImage[posOut * 3 + 2] = sum;
        }
    }
}

__global__ void gaussianBlur(uchar3 *input, uchar3 *output, int *weights, int imageWidth, int imageHeight) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx>=imageWidth) return; // check for out of bound index
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidy>=imageHeight) return; // check for out of bound index
    int posOut = tidx + tidy * imageWidth;

    // Shared memory: each thread of the first 49 threads copy one element of weights
    __shared__ int sweights[49];
    int localTid = threadIdx.x + threadIdx.y * blockDim.x;
    if (localTid < 49) {
        sweights[localTid] = weights[localTid];
    }
    __syncthreads();

    int sum = 0;
    int c = 0;
    for (int y = -3; y <= 3; y++) {
        for (int x = -3; x <= 3; x++) {
            int i = tidx + x;
            int j = tidy + y;
            if (i < 0) continue;
            if (i >= imageWidth) continue;
            if (j < 0) continue;
            if (j >= imageHeight) continue;
            int tid = j * imageWidth + i;
            unsigned char gray = (input[tid].x + input[tid].y + input[tid].z) / 3;
            int coefficient = sweights[(y+3) * 7 + x + 3];
            sum = sum + gray * coefficient;
            c += coefficient;
        }
    }
    sum /= c;
    output[posOut].z = output[posOut].y = output[posOut].x = sum;
}
void Labwork::labwork5_GPU() {
    int weights[] = { 0, 0, 1, 2, 1, 0, 0,  
                 0, 3, 13, 22, 13, 3, 0,  
                 1, 13, 59, 97, 59, 13, 1,  
                 2, 22, 97, 159, 97, 22, 2,  
                 1, 13, 59, 97, 59, 13, 1,  
                 0, 3, 13, 22, 13, 3, 0,
                 0, 0, 1, 2, 1, 0, 0 };
    int pixelCount = inputImage->width * inputImage->height; // number of pixel
    int blockSizex = 16;
    int blockSizey = 8;
    dim3 gridSize = dim3((inputImage->width+blockSizex-1)/blockSizex, (inputImage->height+blockSizey-1)/blockSizey);
    dim3 blockSize = dim3(blockSizex, blockSizey);
    uchar3 *devInput, *devGray; // declare device pointers
    int *kernel;

    // Allocate device memory
    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devGray, pixelCount * sizeof(uchar3));
    cudaMalloc(&kernel, sizeof(weights));
    
    // Copy from host to device
    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel, weights, sizeof(weights), cudaMemcpyHostToDevice);
    
    // Call kernel
    gaussianBlur<<<gridSize, blockSize>>>(devInput, devGray, kernel, inputImage->width, inputImage->height);

    // Allocate host memory
    outputImage = (char*) malloc(pixelCount * sizeof(char) * 3);

    // Copy from Device to host
    cudaMemcpy(outputImage, devGray, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(devInput);
    cudaFree(devGray);
    cudaFree(kernel);
}

__global__ void gaussianBlurNonShared(uchar3 *input, uchar3 *output, int *weights, int imageWidth, int imageHeight) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx>=imageWidth) return; // check for out of bound index
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidy>=imageHeight) return; // check for out of bound index
    int posOut = tidx + tidy * imageWidth;

    int sum = 0;
    int c = 0;
    for (int y = -3; y <= 3; y++) {
        for (int x = -3; x <= 3; x++) {
            int i = tidx + x;
            int j = tidy + y;
            if (i < 0) continue;
            if (i >= imageWidth) continue;
            if (j < 0) continue;
            if (j >= imageHeight) continue;
            int tid = j * imageWidth + i;
            unsigned char gray = (input[tid].x + input[tid].y + input[tid].z) / 3;
            int coefficient = weights[(y+3) * 7 + x + 3];
            sum = sum + gray * coefficient;
            c += coefficient;
        }
    }
    sum /= c;
    output[posOut].z = output[posOut].y = output[posOut].x = sum;
}

void Labwork::labwork5_GPU_NonSharedMemory() {
    int weights[] = { 0, 0, 1, 2, 1, 0, 0,  
             0, 3, 13, 22, 13, 3, 0,  
             1, 13, 59, 97, 59, 13, 1,  
             2, 22, 97, 159, 97, 22, 2,  
             1, 13, 59, 97, 59, 13, 1,  
             0, 3, 13, 22, 13, 3, 0,
             0, 0, 1, 2, 1, 0, 0 };
    int pixelCount = inputImage->width * inputImage->height; // number of pixel
    int blockSizex = 16;
    int blockSizey = 8;
    dim3 gridSize = dim3((inputImage->width+blockSizex-1)/blockSizex, (inputImage->height+blockSizey-1)/blockSizey);
    dim3 blockSize = dim3(blockSizex, blockSizey);
    uchar3 *devInput, *devGray; // declare device pointers
    int *kernel;

    // Allocate device memory
    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devGray, pixelCount * sizeof(uchar3));
    cudaMalloc(&kernel, sizeof(weights));
    
    // Copy from host to device
    cudaMemcpy(devInput, inputImage->buffer, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel, weights, sizeof(weights), cudaMemcpyHostToDevice);
    
    // Call kernel
    gaussianBlurNonShared<<<gridSize, blockSize>>>(devInput, devGray, kernel, inputImage->width, inputImage->height);

    // Allocate host memory
    outputImage = (char*) malloc(pixelCount * sizeof(char) * 3);

    // Copy from Device to host
    cudaMemcpy(outputImage, devGray, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(devInput);
    cudaFree(devGray);
    cudaFree(kernel);
}

void Labwork::labwork6_GPU() {

}

void Labwork::labwork7_GPU() {

}

void Labwork::labwork8_GPU() {

}

void Labwork::labwork9_GPU() {

}

void Labwork::labwork10_GPU() {

}
