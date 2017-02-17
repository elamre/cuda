


#include <framework/CUDA/error.h>

#include "HDRPipeline.h"


namespace
{
	template <typename T>
	auto cudaMalloc(std::size_t size)
	{
		void* ptr;
		throw_error(::cudaMalloc(&ptr, size * sizeof(T)));
		return cuda_unique_ptr<T> { static_cast<T*>(ptr) };
	}

	template <typename T>
	auto cudaMallocZeroed(std::size_t size)
	{
		auto memory = cudaMalloc<T>(size);
		throw_error(cudaMemset(memory.get(), 0, size * sizeof(T)));
		return memory;
	}
}

HDRPipeline::HDRPipeline(unsigned int width, unsigned int height)
	: width(width),
	  height(height),
	  d_input_image(cudaMalloc<float>(width * height * 3)),
	  d_luminance_image(cudaMallocZeroed<float>(width * height)),
	  d_downsample_buffer(cudaMallocZeroed<float>(width * height)),
	  d_tonemapped_image(cudaMallocZeroed<float>(width * height * 3)),
	  d_brightpass_image(cudaMallocZeroed<float>(width * height * 3)),
	  d_blurred_image(cudaMallocZeroed<float>(width * height * 3)),
	  d_output_image(cudaMallocZeroed<float>(width * height * 3))
{
}

void HDRPipeline::consume(const float* input_image)
{
	// upload input data to GPU
	throw_error(cudaMemcpy(d_input_image.get(), input_image, width * height * 3 * 4U, cudaMemcpyHostToDevice));
}

void HDRPipeline::computeLuminance()
{
	void luminance(float* dest, const float* src, unsigned int width, unsigned int height);

	luminance(d_luminance_image.get(), d_input_image.get(), width, height);
}

float HDRPipeline::downsample()
{
	float downsample(float* buffer, float* src, unsigned int width, unsigned int height);

	return downsample(d_downsample_buffer.get(), d_luminance_image.get(), width, height);
}

void HDRPipeline::tonemap(float exposure, float brightpass_threshold)
{
	void tonemap(float* tonemapped, float* brightpass, const float* src, unsigned int width, unsigned int height, float exposure, float brightpass_threshold);

	tonemap(d_tonemapped_image.get(), d_brightpass_image.get(), d_input_image.get(), width, height, exposure, brightpass_threshold);
}

void HDRPipeline::blur()
{
	void gaussian_blur(float* dest, const float* src, unsigned int width, unsigned int height);

	gaussian_blur(d_blurred_image.get(), d_brightpass_image.get(), width, height);
}

void HDRPipeline::compose()
{
	void compose(float* output, const float* tonemapped, const float* blurred, unsigned int width, unsigned int height);

	compose(d_output_image.get(), d_tonemapped_image.get(), d_brightpass_image.get(), width, height);
}


image<float> HDRPipeline::readLuminance()
{
	image<float> luminance(width, height);
	// download output data from GPU
	throw_error(cudaMemcpy(data(luminance), d_luminance_image.get(), width * height * 4U, cudaMemcpyDeviceToHost));
	return luminance;
}

image<float> HDRPipeline::readDownsample()
{
	image<float> downsample(width, height);
	// download output data from GPU
	throw_error(cudaMemcpy(data(downsample), d_downsample_buffer.get(), width * height * 4U, cudaMemcpyDeviceToHost));
	return downsample;
}

image<RGB32F> HDRPipeline::readTonemapped()
{
	image<RGB32F> tonemapped(width, height);
	// download output data from GPU
	throw_error(cudaMemcpy(data(tonemapped), d_tonemapped_image.get(), width * height * 3 * 4U, cudaMemcpyDeviceToHost));
	return tonemapped;
}

image<RGB32F> HDRPipeline::readBrightpass()
{
	image<RGB32F> brightpass(width, height);
	// download output data from GPU
	throw_error(cudaMemcpy(data(brightpass), d_brightpass_image.get(), width * height * 3 * 4U, cudaMemcpyDeviceToHost));
	return brightpass;
}

image<RGB32F> HDRPipeline::readBlurred()
{
	image<RGB32F> blurred(width, height);
	// download output data from GPU
	throw_error(cudaMemcpy(data(blurred), d_blurred_image.get(), width * height * 3 * 4U, cudaMemcpyDeviceToHost));
	return blurred;
}

image<RGB32F> HDRPipeline::readOutput()
{
	image<RGB32F> output(width, height);
	// download output data from GPU
	throw_error(cudaMemcpy(data(output), d_output_image.get(), width * height * 3 * 4U, cudaMemcpyDeviceToHost));
	return output;
}
