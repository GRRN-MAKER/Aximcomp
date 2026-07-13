/*
 * AXIM Vulkan SPIR-V Runtime Executor — C ABI
 * ===========================================
 * The real cross-vendor GPU dispatch path (NVIDIA / AMD / Intel).
 * Implemented in axim_vulkan_executor.cpp, compiled only when the
 * Vulkan SDK is present (AXIM_BUILD_VULKAN). No CUDA anywhere.
 */
#ifndef AXIM_VULKAN_EXECUTOR_H
#define AXIM_VULKAN_EXECUTOR_H

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/* Create VkInstance/VkDevice, pick a compute queue. 0 on success. */
int axim_vk_init(void);

/* Name of the selected physical device (e.g. "NVIDIA GeForce RTX 4090"). */
const char* axim_vk_device_name(void);

/* Release all Vulkan resources. */
void axim_vk_shutdown(void);

/*
 * Load a compiled SPIR-V module from `spirv_path` and dispatch it with
 * `nbuf` storage buffers bound at binding 0..nbuf-1.
 *
 *   buffers[i]   host pointer to buffer i's data
 *   sizes[i]     size of buffer i in bytes
 *   is_output[i] 1 if buffer i is copied back after execution, else 0
 *   groups       number of workgroups dispatched in x
 *
 * Inputs are uploaded, the shader runs, outputs are copied back.
 * Returns 0 on success. Runs identically on NVIDIA, AMD, and Intel.
 */
int axim_vk_run_spirv(const char* spirv_path,
                      void** buffers, const size_t* sizes,
                      const int* is_output, int nbuf,
                      uint32_t groups);

/*
 * Same as axim_vk_run_spirv but also uploads `push_size` bytes of push
 * constant data (e.g. a uint element count) to the compute stage. Pass
 * push_data = nullptr, push_size = 0 for shaders with no push constants.
 */
int axim_vk_run_spirv_pc(const char* spirv_path,
                         void** buffers, const size_t* sizes,
                         const int* is_output, int nbuf,
                         uint32_t groups,
                         const void* push_data, uint32_t push_size);

#ifdef __cplusplus
}
#endif

#endif /* AXIM_VULKAN_EXECUTOR_H */
