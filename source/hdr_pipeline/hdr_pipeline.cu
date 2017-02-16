


#include <math/vector.h>

#include "color.cuh"


namespace
{
	constexpr unsigned int divup(unsigned int a, unsigned int b)
	{
		return (a + b - 1) / b;
	}

	const math::float2 gauss_kernel[] = {
		math::float2(-16.00f, 0.00288204f),
		math::float2(-15.00f, 0.00418319f),
		math::float2(-14.00f, 0.00592754f),
		math::float2(-13.00f, 0.00819980f),
		math::float2(-12.00f, 0.01107369f),
		math::float2(-11.00f, 0.01459965f),
		math::float2(-10.00f, 0.01879116f),
		math::float2( -9.00f, 0.02361161f),
		math::float2( -8.00f, 0.02896398f),
		math::float2( -7.00f, 0.03468581f),
		math::float2( -6.00f, 0.04055144f),
		math::float2( -5.00f, 0.04628301f),
		math::float2( -4.00f, 0.05157007f),
		math::float2( -3.00f, 0.05609637f),
		math::float2( -2.00f, 0.05957069f),
		math::float2( -1.00f, 0.06175773f),
		math::float2(  0.00f, 0.06250444f),
		math::float2(  1.00f, 0.06175773f),
		math::float2(  2.00f, 0.05957069f),
		math::float2(  3.00f, 0.05609637f),
		math::float2(  4.00f, 0.05157007f),
		math::float2(  5.00f, 0.04628301f),
		math::float2(  6.00f, 0.04055144f),
		math::float2(  7.00f, 0.03468581f),
		math::float2(  8.00f, 0.02896398f),
		math::float2(  9.00f, 0.02361161f),
		math::float2( 10.00f, 0.01879116f),
		math::float2( 11.00f, 0.01459965f),
		math::float2( 12.00f, 0.01107369f),
		math::float2( 13.00f, 0.00819980f),
		math::float2( 14.00f, 0.00592754f),
		math::float2( 15.00f, 0.00418319f),
		math::float2( 16.00f, 0.00288204f)
	};
}

template<typename T>
void swap(T& a,T& b)
{
	T c = b;
	b = a;
	a = c;
}

__global__ void luminance_kernel(float * dest, const float * input, unsigned int width, unsigned int height) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height) {
		const float * input_pixel = input + (width * y + x) * 3;
		float lum = (0.2126f * input_pixel[0] + 0.7152f * input_pixel[1] + 0.0722f * input_pixel[2]) ;
		dest[width * y + x] = lum;
	}
}

__host__ void luminance(float * dest, const float * input, unsigned int width, unsigned int height) {
	const dim3 block_size = { 32, 32};
	const dim3 num_blocks = { divup(width, block_size.x), divup(height, block_size.y)};
	luminance_kernel <<< num_blocks, block_size >>> (dest, input, width, height);
}

#define F 2
__global__ void downsample_kernel(float * output, float * luminance, unsigned int width, unsigned int height, unsigned int outputPitch, unsigned int  inputPitch) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	float sum = 0.0f;
	if (x >= width / F || y >= height / F) return;
	for (int j = 0; j < F; j++) {
		for (int i = 0; i < F; i++) {
			sum += luminance[(y*F + j) * inputPitch + (x * F + i)];
		}
	}
	output[y * outputPitch/F + x] = sum / (F * F);
}

__host__ void downsample(float * output, float * luminance, unsigned int width, unsigned int height) {
	printf("output %p luminance %p\n", output, luminance);
	const dim3 block_size = { 32, 32 };
	const dim3 num_blocks = { divup(width, block_size.x), divup(height, block_size.y) };
	unsigned int pitch = width;
	bool ping = false;
	while (width != 1 || height != 1) {
		//downsample_kernel << <num_blocks, block_size >> > ((ping) ? output : luminance, (ping) ? luminance : output, width, height, pitch);
		if (ping) {
			downsample_kernel << <num_blocks, block_size >> > (luminance, output, width, height, pitch, pitch/2);
		} else {
			downsample_kernel << <num_blocks, block_size >> > (output, luminance, width, height, pitch/2, pitch);
		}
		width = (width > 1) ? width / 2 : 1;
		height = (height > 1) ? height / 2 : 1;
		ping = !ping;	
		cudaDeviceSynchronize();
	} 
	printf("Done\n");
}
// TODO: implement gaussian blur for light bloom

__global__ void tonemap_kernel(uchar4* tonemapped, uchar4* brightpass, const float* src, unsigned int width, unsigned int height, float exposure, float brightpass_thdesthold)
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
		uchar4 out = { toSRGB8(c_t.x), toSRGB8(c_t.y), toSRGB8(c_t.z), 0xFFU };
		tonemapped[y * width + x] = out;
		brightpass[y * width + x] = luminance(c_t) > brightpass_thdesthold ? out : uchar4 { 0U, 0U, 0U, 0xFFU };
	}
}

void tonemap(uchar4* tonemapped, uchar4* brightpass, const float* src, unsigned int width, unsigned int height, float exposure, float brightpass_thdesthold)
{
	const auto block_size = dim3 { 32U, 32U };

	auto num_blocks = dim3{ divup(width, block_size.x), divup(height, block_size.y) };

	tonemap_kernel<<<num_blocks, block_size>>>(tonemapped, brightpass, src, width, height, exposure, brightpass_thdesthold);
}
