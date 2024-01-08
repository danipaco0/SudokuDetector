import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

'''
Generation of square matrix used for the blurring effect
Sigma is the a parameter of the Gaussian distribution
'''
def gaussian_kernel(size, sigma=1.0):
    kernel = np.zeros((size, size), dtype=np.float32)
    mean = size // 2
    for x in range(size):
        for y in range(size):
            kernel[x, y] = np.exp(-0.5 * ((x - mean) ** 2 + (y - mean) ** 2) / (sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel

'''
Kernel details:
x and y integers are the global position of a thread inside the grid.
A condition verifies if their position is not out of the size of the input image.
Each thread works on one pixel of the image.
An iteration allows to get each neighbor of the pixel and multiply it by the corresponding weight in the Gaussian kernel
The final value is then assigned to the output image.
Parallel execution on the GPU since each pixel has its own thread.
'''
def apply_gaussian_blur(image, kernel_size=9, sigma=1.5):
    gaussian_kernel_vals = gaussian_kernel(kernel_size, sigma)
    gaussian_kernel_vals = gaussian_kernel_vals.flatten()

    height, width = image.shape
    image = image.astype(np.uint8)

    kernel_code = '''
    __global__ void gaussianBlurKernel(unsigned char* input, unsigned char* output, int width, int height, const float* gaussianKernel, int kernelSize) {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        if (x >= width || y >= height) return;

        float blurValue = 0.0;
        int halfSize = kernelSize / 2;

        for (int ky = -halfSize; ky <= halfSize; ky++) {
            for (int kx = -halfSize; kx <= halfSize; kx++) {
                int pixelX = min(max(x + kx, 0), width - 1);
                int pixelY = min(max(y + ky, 0), height - 1);
                blurValue += input[pixelY * width + pixelX] * gaussianKernel[(ky + halfSize) * kernelSize + (kx + halfSize)];
            }
        }
        output[y * width + x] = static_cast<unsigned char>(blurValue);
    }
    '''

    mod = SourceModule(kernel_code) # Using kernel "kernel_code" written in C++
    gaussian_blur = mod.get_function("gaussianBlurKernel") # Using the function from the specified kernel

    # Memory allocation on the GPU for the input image, output image and kernel
    input_gpu = cuda.mem_alloc(image.nbytes)
    output_gpu = cuda.mem_alloc(image.nbytes) 
    kernel_gpu = cuda.mem_alloc(gaussian_kernel_vals.nbytes)
    # Data transfer from CPU to GPU
    cuda.memcpy_htod(input_gpu, image) 
    cuda.memcpy_htod(kernel_gpu, gaussian_kernel_vals)

    block = (16, 16, 1) # Block dimensions 16x16 in 2D == 256 threads per block 
    grid = (int(np.ceil(width / block[0])), int(np.ceil(height / block[1]))) # Grid dimensions based on the image size and the blocks size
    
    gaussian_blur(input_gpu, output_gpu, np.int32(width), np.int32(height), kernel_gpu, np.int32(kernel_size), block=block, grid=grid)

    result = np.zeros_like(image)
    cuda.memcpy_dtoh(result, output_gpu)

    return result
