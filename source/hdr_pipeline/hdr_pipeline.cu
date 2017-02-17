


#include <math/vector.h>

#include "color.cuh"


namespace
{
	constexpr unsigned int divup(unsigned int a, unsigned int b)
	{
		return (a + b - 1) / b;
	}

	template<typename T>
	void swap(T& a, T& b)
	{
		T c = b;
		b = a;
		a = c;
	}
}



void luminance(float* dest, const float* src, unsigned int width, unsigned int height)
{
	// TODO: compute luminance of src image
}



float downsample(float* buffer, float* src, unsigned int width, unsigned int height)
{
	// TODO: downsample and return average luminance

	return 1.0f;
}



__global__ void blur_kernel(float* dest, const float* src, unsigned int width, unsigned int height)
{
	constexpr float weights[] = {
		//  -16           -15         -14          -13          -12          -11          -10           -9           -8           -7           -6           -5           -4           -3           -2           -1
		0.00288204f, 0.00418319f, 0.00592754f, 0.00819980f, 0.01107369f, 0.01459965f, 0.01879116f, 0.02361161f, 0.02896398f, 0.03468581f, 0.04055144f, 0.04628301f, 0.05157007f, 0.05609637f, 0.05957069f, 0.06175773f,
		//    0
		0.06250444f,
		//    1             2           3            4            5            6            7            8            9           10           11           12           13           14           15           16
		0.06175773f, 0.05957069f, 0.05609637f, 0.05157007f, 0.04628301f, 0.04055144f, 0.03468581f, 0.02896398f, 0.02361161f, 0.01879116f, 0.01459965f, 0.01107369f, 0.00819980f, 0.00592754f, 0.00418319f, 0.00288204f
	};


}

void gaussian_blur(float* dest, const float* src, unsigned int width, unsigned int height)
{
	// TODO: gaussian blur of brightpass
}



void compose(float* output, const float* tonemapped, const float* blurred, unsigned int width, unsigned int height)
{
	// TODO: add blurred brightpass to tonemapped image
}



__global__ void tonemap_kernel(float* tonemapped, float* brightpass, const float* src, unsigned int width, unsigned int height, float exposure, float brightpass_threshold)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height)
	{
		// figure out input color
		math::float3 c = { src[3 * (y * width + x) + 0], src[3 * (y * width + x) + 1], src[3 * (y * width + x) + 2] };

		// compute tonemapped color
		math::float3 c_t = tonemap(c, exposure);

		// write out tonemapped color
		tonemapped[3 * (y * width + x) + 0] = c_t.x;
		tonemapped[3 * (y * width + x) + 1] = c_t.y;
		tonemapped[3 * (y * width + x) + 2] = c_t.z;

		// write out brightpass color
		math::float3 c_b = luminance(c_t) > brightpass_threshold ? c_t : math::float3 {0.0f, 0.0f, 0.0f};
		brightpass[3 * (y * width + x) + 0] = c_b.x;
		brightpass[3 * (y * width + x) + 1] = c_b.y;
		brightpass[3 * (y * width + x) + 2] = c_b.z;
	}
}

void tonemap(float* tonemapped, float* brightpass, const float* src, unsigned int width, unsigned int height, float exposure, float brightpass_threshold)
{
	const auto block_size = dim3 { 32U, 32U };

	auto num_blocks = dim3{ divup(width, block_size.x), divup(height, block_size.y) };

	tonemap_kernel<<<num_blocks, block_size>>>(tonemapped, brightpass, src, width, height, exposure, brightpass_threshold);
}
