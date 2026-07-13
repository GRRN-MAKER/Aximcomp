/*
 * AXIM Vulkan SPIR-V Runtime Executor
 * ===================================
 * The real cross-vendor GPU dispatcher: loads a compiled SPIR-V module,
 * creates device-local storage buffers, binds them to a compute pipeline,
 * and dispatches the shader on NVIDIA / AMD / Intel GPUs.
 *
 * Zero CUDA. Uses only the Khronos Vulkan API, which ships in the drivers
 * of all three vendors on Linux and Windows.
 *
 * Compiled only when AXIM_BUILD_VULKAN is defined (i.e. the Vulkan SDK is
 * present). On platforms without Vulkan (e.g. the macOS/Metal build) this
 * file is simply not compiled, so it never breaks other backends.
 */
#ifdef AXIM_BUILD_VULKAN

#include "axim_vulkan_executor.h"

#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>

namespace {

#define AXIM_VK_CHECK(expr)                                              \
    do {                                                                 \
        VkResult _r = (expr);                                            \
        if (_r != VK_SUCCESS) {                                          \
            std::fprintf(stderr, "[axim-vulkan] %s failed (%d)\n",       \
                         #expr, (int)_r);                                \
            return -1;                                                   \
        }                                                                \
    } while (0)

struct VulkanContext {
    VkInstance       instance   = VK_NULL_HANDLE;
    VkPhysicalDevice phys       = VK_NULL_HANDLE;
    VkDevice         device     = VK_NULL_HANDLE;
    VkQueue          queue      = VK_NULL_HANDLE;
    uint32_t         queueFamily = 0;
    VkCommandPool    cmdPool    = VK_NULL_HANDLE;
    char             deviceName[256] = {0};
    bool             ready      = false;
};

VulkanContext g_ctx;

uint32_t find_memory_type(uint32_t typeBits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(g_ctx.phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & want) == want) {
            return i;
        }
    }
    return UINT32_MAX;
}

} // namespace

extern "C" int axim_vk_init(void) {
    if (g_ctx.ready) return 0;

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "AXIM";
    /* Request 1.0 for the broadest ICD/loader compatibility (incl. Lavapipe
     * software Vulkan on CI). The compute path uses only core 1.0 features. */
    app.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    AXIM_VK_CHECK(vkCreateInstance(&ici, nullptr, &g_ctx.instance));

    uint32_t n = 0;
    AXIM_VK_CHECK(vkEnumeratePhysicalDevices(g_ctx.instance, &n, nullptr));
    if (n == 0) { std::fprintf(stderr, "[axim-vulkan] no devices\n"); return -1; }
    std::vector<VkPhysicalDevice> devs(n);
    AXIM_VK_CHECK(vkEnumeratePhysicalDevices(g_ctx.instance, &n, devs.data()));
    g_ctx.phys = devs[0]; /* prefer discrete in a fuller impl */

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(g_ctx.phys, &props);
    std::snprintf(g_ctx.deviceName, sizeof(g_ctx.deviceName), "%s", props.deviceName);

    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_ctx.phys, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(g_ctx.phys, &qn, qf.data());
    bool found = false;
    for (uint32_t i = 0; i < qn; ++i) {
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { g_ctx.queueFamily = i; found = true; break; }
    }
    if (!found) { std::fprintf(stderr, "[axim-vulkan] no compute queue\n"); return -1; }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = g_ctx.queueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    AXIM_VK_CHECK(vkCreateDevice(g_ctx.phys, &dci, nullptr, &g_ctx.device));
    vkGetDeviceQueue(g_ctx.device, g_ctx.queueFamily, 0, &g_ctx.queue);

    VkCommandPoolCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi.queueFamilyIndex = g_ctx.queueFamily;
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    AXIM_VK_CHECK(vkCreateCommandPool(g_ctx.device, &cpi, nullptr, &g_ctx.cmdPool));

    g_ctx.ready = true;
    return 0;
}

extern "C" const char* axim_vk_device_name(void) {
    return g_ctx.ready ? g_ctx.deviceName : "none";
}

extern "C" void axim_vk_shutdown(void) {
    if (!g_ctx.ready) return;
    vkDestroyCommandPool(g_ctx.device, g_ctx.cmdPool, nullptr);
    vkDestroyDevice(g_ctx.device, nullptr);
    vkDestroyInstance(g_ctx.instance, nullptr);
    g_ctx = VulkanContext{};
}

/* ── Host-visible storage buffer (upload/download friendly) ── */
namespace {

struct Buffer {
    VkBuffer       buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize   size = 0;
};

int create_buffer(VkDeviceSize size, Buffer& out) {
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    AXIM_VK_CHECK(vkCreateBuffer(g_ctx.device, &bi, nullptr, &out.buf));

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g_ctx.device, out.buf, &mr);
    uint32_t mt = find_memory_type(
        mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mt == UINT32_MAX) return -1;

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = mt;
    AXIM_VK_CHECK(vkAllocateMemory(g_ctx.device, &ai, nullptr, &out.mem));
    AXIM_VK_CHECK(vkBindBufferMemory(g_ctx.device, out.buf, out.mem, 0));
    out.size = size;
    return 0;
}

void destroy_buffer(Buffer& b) {
    if (b.buf) vkDestroyBuffer(g_ctx.device, b.buf, nullptr);
    if (b.mem) vkFreeMemory(g_ctx.device, b.mem, nullptr);
    b = Buffer{};
}

int upload(Buffer& b, const void* src, VkDeviceSize n) {
    void* p = nullptr;
    AXIM_VK_CHECK(vkMapMemory(g_ctx.device, b.mem, 0, n, 0, &p));
    std::memcpy(p, src, (size_t)n);
    vkUnmapMemory(g_ctx.device, b.mem);
    return 0;
}

int download(Buffer& b, void* dst, VkDeviceSize n) {
    void* p = nullptr;
    AXIM_VK_CHECK(vkMapMemory(g_ctx.device, b.mem, 0, n, 0, &p));
    std::memcpy(dst, p, (size_t)n);
    vkUnmapMemory(g_ctx.device, b.mem);
    return 0;
}

std::vector<uint32_t> load_spirv(const char* path) {
    std::vector<uint32_t> code;
    FILE* f = std::fopen(path, "rb");
    if (!f) return code;
    std::fseek(f, 0, SEEK_END);
    long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (bytes > 0 && (bytes % 4) == 0) {
        code.resize((size_t)bytes / 4);
        if (std::fread(code.data(), 1, (size_t)bytes, f) != (size_t)bytes) code.clear();
    }
    std::fclose(f);
    return code;
}

} // namespace

/*
 * Dispatch a compute shader with `nbuf` host-visible storage buffers bound
 * at binding 0..nbuf-1. buffers[i] points to host data of size sizes[i]
 * bytes; is_output[i] marks buffers copied back after execution. The
 * shader is dispatched with `groups` workgroups in x. Returns 0 on success.
 *
 * This is the vendor-neutral core: it runs on NVIDIA, AMD, and Intel with
 * the same SPIR-V module, no CUDA involved.
 */
extern "C" int axim_vk_run_spirv_pc(const char* spirv_path,
                                    void** buffers, const size_t* sizes,
                                    const int* is_output, int nbuf,
                                    uint32_t groups,
                                    const void* push_data, uint32_t push_size) {
    if (!g_ctx.ready) { if (axim_vk_init() != 0) return -1; }

    std::vector<uint32_t> code = load_spirv(spirv_path);
    if (code.empty()) { std::fprintf(stderr, "[axim-vulkan] bad SPIR-V: %s\n", spirv_path); return -1; }

    VkShaderModuleCreateInfo smi{};
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = code.size() * sizeof(uint32_t);
    smi.pCode = code.data();
    VkShaderModule shader = VK_NULL_HANDLE;
    AXIM_VK_CHECK(vkCreateShaderModule(g_ctx.device, &smi, nullptr, &shader));

    /* device buffers + upload inputs */
    std::vector<Buffer> devbufs((size_t)nbuf);
    for (int i = 0; i < nbuf; ++i) {
        if (create_buffer(sizes[i], devbufs[i]) != 0) return -1;
        if (!is_output[i]) upload(devbufs[i], buffers[i], sizes[i]);
    }

    /* descriptor set layout: nbuf storage buffers */
    std::vector<VkDescriptorSetLayoutBinding> binds((size_t)nbuf);
    for (int i = 0; i < nbuf; ++i) {
        binds[i] = {};
        binds[i].binding = (uint32_t)i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dli{};
    dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dli.bindingCount = (uint32_t)nbuf;
    dli.pBindings = binds.data();
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    AXIM_VK_CHECK(vkCreateDescriptorSetLayout(g_ctx.device, &dli, nullptr, &dsl));

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = push_size;

    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &dsl;
    if (push_data && push_size > 0) {
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcr;
    }
    VkPipelineLayout playout = VK_NULL_HANDLE;
    AXIM_VK_CHECK(vkCreatePipelineLayout(g_ctx.device, &pli, nullptr, &playout));

    VkComputePipelineCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = shader;
    cpi.stage.pName = "main";
    cpi.layout = playout;
    VkPipeline pipe = VK_NULL_HANDLE;
    AXIM_VK_CHECK(vkCreateComputePipelines(g_ctx.device, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe));

    /* descriptor pool + set */
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)nbuf};
    VkDescriptorPoolCreateInfo dpi{};
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = 1;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    AXIM_VK_CHECK(vkCreateDescriptorPool(g_ctx.device, &dpi, nullptr, &pool));

    VkDescriptorSetAllocateInfo dsa{};
    dsa.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsa.descriptorPool = pool;
    dsa.descriptorSetCount = 1;
    dsa.pSetLayouts = &dsl;
    VkDescriptorSet dset = VK_NULL_HANDLE;
    AXIM_VK_CHECK(vkAllocateDescriptorSets(g_ctx.device, &dsa, &dset));

    std::vector<VkDescriptorBufferInfo> dbi((size_t)nbuf);
    std::vector<VkWriteDescriptorSet>   wds((size_t)nbuf);
    for (int i = 0; i < nbuf; ++i) {
        dbi[i] = {devbufs[i].buf, 0, VK_WHOLE_SIZE};
        wds[i] = {};
        wds[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wds[i].dstSet = dset;
        wds[i].dstBinding = (uint32_t)i;
        wds[i].descriptorCount = 1;
        wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wds[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(g_ctx.device, (uint32_t)nbuf, wds.data(), 0, nullptr);

    /* record + submit */
    VkCommandBufferAllocateInfo cai{};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = g_ctx.cmdPool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    AXIM_VK_CHECK(vkAllocateCommandBuffers(g_ctx.device, &cai, &cmd));

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    AXIM_VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, playout, 0, 1, &dset, 0, nullptr);
    if (push_data && push_size > 0) {
        vkCmdPushConstants(cmd, playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size, push_data);
    }
    vkCmdDispatch(cmd, groups, 1, 1);
    AXIM_VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    AXIM_VK_CHECK(vkQueueSubmit(g_ctx.queue, 1, &si, VK_NULL_HANDLE));
    AXIM_VK_CHECK(vkQueueWaitIdle(g_ctx.queue));

    /* copy outputs back */
    for (int i = 0; i < nbuf; ++i) {
        if (is_output[i]) download(devbufs[i], buffers[i], sizes[i]);
    }

    /* cleanup */
    for (int i = 0; i < nbuf; ++i) destroy_buffer(devbufs[i]);
    vkDestroyDescriptorPool(g_ctx.device, pool, nullptr);
    vkDestroyPipeline(g_ctx.device, pipe, nullptr);
    vkDestroyPipelineLayout(g_ctx.device, playout, nullptr);
    vkDestroyDescriptorSetLayout(g_ctx.device, dsl, nullptr);
    vkDestroyShaderModule(g_ctx.device, shader, nullptr);
    return 0;
}

/* Convenience wrapper: dispatch with no push constants. */
extern "C" int axim_vk_run_spirv(const char* spirv_path,
                                 void** buffers, const size_t* sizes,
                                 const int* is_output, int nbuf,
                                 uint32_t groups) {
    return axim_vk_run_spirv_pc(spirv_path, buffers, sizes, is_output,
                                nbuf, groups, nullptr, 0);
}

#endif /* AXIM_BUILD_VULKAN */
