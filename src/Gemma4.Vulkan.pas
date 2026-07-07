{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information

 -------------------------------------------------------------------------------

  Gemma4.Vulkan - Vulkan device management and buffer allocation

  Pure Delphi Vulkan bindings and device management. Loads vulkan-1.dll
  at runtime, creates instance/device/queue, and provides GPU buffer
  allocation and command buffer management for compute shaders.

  No external Vulkan headers or 3rd-party bindings -- all function
  pointers and types are declared inline.

  Key types:
  - TVulkanDevice: Manages Vulkan lifecycle (instance, physical device,
    logical device, compute queue, command pool). Provides buffer
    allocation and command buffer recording/submission.
  - TVulkanBuffer: GPU buffer with mapped host pointer for data transfer.

  Dependencies: StdApp.Base, Gemma4.Types
===============================================================================}

unit Gemma4.Vulkan;

{$I StdApp.Defines.inc}

interface

uses
  WinApi.Windows,
  System.SysUtils,
  StdApp.Base,
  Gemma4.Types;

const
  CVK_ERR_LOAD_LIB = 'VK01';
  CVK_ERR_INSTANCE = 'VK02';
  CVK_ERR_DEVICE = 'VK03';
  CVK_ERR_QUEUE = 'VK04';
  CVK_ERR_BUFFER = 'VK05';
  CVK_ERR_MEMORY = 'VK06';
  CVK_ERR_COMMAND = 'VK07';
  CVK_ERR_FENCE = 'VK08';

  // Vulkan constants
  CVK_SUCCESS = 0;
  CVK_STRUCTURE_TYPE_APPLICATION_INFO = 0;
  CVK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1;
  CVK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2;
  CVK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3;
  CVK_STRUCTURE_TYPE_SUBMIT_INFO = 4;
  CVK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5;
  CVK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 8;
  CVK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO = 11;
  CVK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12;
  CVK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39;
  CVK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40;
  CVK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42;
  CVK_STRUCTURE_TYPE_MEMORY_BARRIER = 46;
  CVK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES = 1000280000;

  CVK_API_VERSION_1_0 = (1 shl 22) or (0 shl 12) or 0;
  CVK_API_VERSION_1_1 = (1 shl 22) or (1 shl 12) or 0;

  CVK_BUFFER_USAGE_STORAGE = $00000020;
  CVK_BUFFER_USAGE_TRANSFER_SRC = $00000001;
  CVK_BUFFER_USAGE_TRANSFER_DST = $00000002;

  CVK_SHARING_MODE_EXCLUSIVE = 0;

  CVK_MEMORY_PROPERTY_HOST_VISIBLE = $00000002;
  CVK_MEMORY_PROPERTY_HOST_COHERENT = $00000004;
  CVK_MEMORY_PROPERTY_DEVICE_LOCAL = $00000001;
  CVK_MEMORY_PROPERTY_HOST_CACHED = $00000008;

  CVK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXT_NAME = 'VK_KHR_shader_integer_dot_product';

  CVK_COMMAND_POOL_CREATE_RESET = $00000002;
  CVK_COMMAND_BUFFER_LEVEL_PRIMARY = 0;
  CVK_COMMAND_BUFFER_USAGE_ONE_TIME = $00000001;

  CVK_PIPELINE_BIND_POINT_COMPUTE = 1;
  CVK_PIPELINE_STAGE_COMPUTE_SHADER = $00000800;

  CVK_ACCESS_SHADER_READ = $00000020;
  CVK_ACCESS_SHADER_WRITE = $00000040;

  CVK_QUEUE_COMPUTE_BIT = $00000002;
  CVK_QUERY_TYPE_TIMESTAMP = 2;
  CVK_QUERY_RESULT_64 = $00000001;
  CVK_QUERY_RESULT_WAIT = $00000002;

  // Byte offset of float timestampPeriod inside VkPhysicalDeviceLimits
  // (computed from vulkan_core.h field order with win64 C alignment;
  // cross-checked: computed struct total = 504 = declared limits size)
  CVK_LIMITS_TIMESTAMP_PERIOD_OFS = 424;

  CVK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2;

  CVK_WHOLE_SIZE = UInt64(not UInt64(0));

  CVK_MAX_TIMEOUT = UInt64(not UInt64(0));

type
  // Vulkan opaque handles
  // Dispatchable handles are pointers in C; on 64-bit UInt64 matches size
  TVkInstance = UInt64;
  TVkPhysicalDevice = UInt64;
  TVkDevice = UInt64;
  TVkQueue = UInt64;
  TVkCommandBuffer = UInt64;

  // Non-dispatchable handles are uint64_t in C
  TVkBuffer = UInt64;
  TVkDeviceMemory = UInt64;
  TVkFence = UInt64;
  TVkQueryPool = UInt64;
  TVkPipeline = UInt64;
  TVkPipelineLayout = UInt64;
  TVkDescriptorSetLayout = UInt64;
  TVkDescriptorPool = UInt64;
  TVkDescriptorSet = UInt64;
  TVkShaderModule = UInt64;
  TVkCommandPool = UInt64;

  // Vulkan structures - NO packed, using natural alignment to match C
  // Sizes and offsets validated against vulkan_core.h from Vulkan SDK 1.4.350.0

  { TVkApplicationInfo }
  // C size: 48 bytes
  TVkApplicationInfo = record
    sType: UInt32;             // offset 0
    pNext: Pointer;            // offset 8
    pApplicationName: PAnsiChar; // offset 16
    applicationVersion: UInt32; // offset 24
    pEngineName: PAnsiChar;    // offset 32
    engineVersion: UInt32;     // offset 40
    apiVersion: UInt32;        // offset 44
  end;

  { TVkInstanceCreateInfo }
  // C size: 64 bytes
  TVkInstanceCreateInfo = record
    sType: UInt32;             // offset 0
    pNext: Pointer;            // offset 8
    flags: UInt32;             // offset 16
    pApplicationInfo: Pointer; // offset 24
    enabledLayerCount: UInt32; // offset 32
    ppEnabledLayerNames: Pointer; // offset 40
    enabledExtensionCount: UInt32; // offset 48
    ppEnabledExtensionNames: Pointer; // offset 56
  end;

  { TVkDeviceQueueCreateInfo }
  // C size: 40 bytes
  TVkDeviceQueueCreateInfo = record
    sType: UInt32;             // offset 0
    pNext: Pointer;            // offset 8
    flags: UInt32;             // offset 16
    queueFamilyIndex: UInt32;  // offset 20
    queueCount: UInt32;        // offset 24
    pQueuePriorities: Pointer; // offset 32
  end;

  { TVkDeviceCreateInfo }
  // C size: 72 bytes
  TVkDeviceCreateInfo = record
    sType: UInt32;             // offset 0
    pNext: Pointer;            // offset 8
    flags: UInt32;             // offset 16
    queueCreateInfoCount: UInt32; // offset 20
    pQueueCreateInfos: Pointer; // offset 24
    enabledLayerCount: UInt32; // offset 32
    ppEnabledLayerNames: Pointer; // offset 40
    enabledExtensionCount: UInt32; // offset 48
    ppEnabledExtensionNames: Pointer; // offset 56
    pEnabledFeatures: Pointer; // offset 64
  end;

  { TVkBufferCreateInfo }
  // C size: 56 bytes
  TVkBufferCreateInfo = record
    sType: UInt32;             // offset 0
    pNext: Pointer;            // offset 8
    flags: UInt32;             // offset 16
    size: UInt64;              // offset 24
    usage: UInt32;             // offset 32
    sharingMode: UInt32;       // offset 36
    queueFamilyIndexCount: UInt32; // offset 40
    pQueueFamilyIndices: Pointer; // offset 48
  end;

  { TVkMemoryAllocateInfo }
  // C size: 32 bytes
  TVkMemoryAllocateInfo = record
    sType: UInt32;             // offset 0
    pNext: Pointer;            // offset 8
    allocationSize: UInt64;    // offset 16
    memoryTypeIndex: UInt32;   // offset 24
  end;

  { TVkMemoryRequirements }
  // C size: 24 bytes
  TVkMemoryRequirements = record
    size: UInt64;              // offset 0
    alignment: UInt64;         // offset 8
    memoryTypeBits: UInt32;    // offset 16
  end;

  { TVkMemoryType }
  // C size: 8 bytes
  TVkMemoryType = record
    propertyFlags: UInt32;     // offset 0
    heapIndex: UInt32;         // offset 4
  end;

  { TVkMemoryHeap }
  // C size: 16 bytes (8 + 4 + 4 padding)
  TVkMemoryHeap = record
    size: UInt64;              // offset 0
    flags: UInt32;             // offset 8
  end;

  { TVkPhysicalDeviceMemoryProperties }
  // C size: 520 bytes
  TVkPhysicalDeviceMemoryProperties = record
    memoryTypeCount: UInt32;   // offset 0
    memoryTypes: array[0..31] of TVkMemoryType; // offset 4, 256 bytes
    memoryHeapCount: UInt32;   // offset 260
    memoryHeaps: array[0..15] of TVkMemoryHeap; // offset 264, 256 bytes
  end;

  { TVkExtent3D }
  // C size: 12 bytes
  TVkExtent3D = record
    width: UInt32;
    height: UInt32;
    depth: UInt32;
  end;

  { TVkQueueFamilyProperties }
  // C size: 24 bytes
  TVkQueueFamilyProperties = record
    queueFlags: UInt32;        // offset 0
    queueCount: UInt32;        // offset 4
    timestampValidBits: UInt32; // offset 8
    minImageTransferGranularity: TVkExtent3D; // offset 12
  end;

  { TVkPhysicalDeviceProperties }
  // C size: 824 bytes
  // limits at offset 296, sparseProperties at offset 800
  TVkPhysicalDeviceProperties = record
    apiVersion: UInt32;        // offset 0
    driverVersion: UInt32;     // offset 4
    vendorID: UInt32;          // offset 8
    deviceID: UInt32;          // offset 12
    deviceType: UInt32;        // offset 16
    deviceName: array[0..255] of AnsiChar; // offset 20, 256 bytes
    pipelineCacheUUID: array[0..15] of Byte; // offset 276, 16 bytes
    _limitspad: UInt32;        // offset 292, C alignment padding (VkPhysicalDeviceLimits requires 8-byte alignment)
    limits: array[0..503] of Byte; // offset 296, 504 bytes (VkPhysicalDeviceLimits)
    sparseProperties: array[0..19] of Byte; // offset 800, 20 bytes
    _endpad: UInt32;           // padding to match C total of 824 bytes
  end;

  { TVkQueryPoolCreateInfo }
  // C size: 32 bytes (sType 0, pNext 8, flags 16, queryType 20,
  // queryCount 24, pipelineStatistics 28) -- natural Delphi alignment matches
  TVkQueryPoolCreateInfo = record
    sType: UInt32;
    pNext: Pointer;
    flags: UInt32;
    queryType: UInt32;
    queryCount: UInt32;
    pipelineStatistics: UInt32;
  end;

  { TVkCommandPoolCreateInfo }
  // C size: 24 bytes
  TVkCommandPoolCreateInfo = record
    sType: UInt32;             // offset 0
    pNext: Pointer;            // offset 8
    flags: UInt32;             // offset 16
    queueFamilyIndex: UInt32;  // offset 20
  end;

  { TVkCommandBufferAllocateInfo }
  // C size: 32 bytes
  TVkCommandBufferAllocateInfo = record
    sType: UInt32;             // offset 0
    pNext: Pointer;            // offset 8
    commandPool: TVkCommandPool; // offset 16
    level: UInt32;             // offset 24
    commandBufferCount: UInt32; // offset 28
  end;

  { TVkCommandBufferBeginInfo }
  // C size: 32 bytes
  TVkCommandBufferBeginInfo = record
    sType: UInt32;             // offset 0
    pNext: Pointer;            // offset 8
    flags: UInt32;             // offset 16
    pInheritanceInfo: Pointer; // offset 24
  end;

  { TVkSubmitInfo }
  // C size: 72 bytes
  TVkSubmitInfo = record
    sType: UInt32;             // offset 0
    pNext: Pointer;            // offset 8
    waitSemaphoreCount: UInt32; // offset 16
    pWaitSemaphores: Pointer;  // offset 24
    pWaitDstStageMask: Pointer; // offset 32
    commandBufferCount: UInt32; // offset 40
    pCommandBuffers: Pointer;  // offset 48
    signalSemaphoreCount: UInt32; // offset 56
    pSignalSemaphores: Pointer; // offset 64
  end;

  { TVkFenceCreateInfo }
  // C size: 24 bytes
  TVkFenceCreateInfo = record
    sType: UInt32;             // offset 0
    pNext: Pointer;            // offset 8
    flags: UInt32;             // offset 16
  end;

  { TVkMemoryBarrier }
  // C size: 24 bytes -- validated against vulkan_core.h (VkMemoryBarrier)
  // NO packed - natural alignment matches C layout
  TVkMemoryBarrier = record
    sType: UInt32;             // offset 0
    pNext: Pointer;            // offset 8
    srcAccessMask: UInt32;     // offset 16 (VkAccessFlags)
    dstAccessMask: UInt32;     // offset 20 (VkAccessFlags)
  end;

  { TVkBufferCopy }
  // C size: 24 bytes -- validated against vulkan_core.h (VkBufferCopy)
  // Three VkDeviceSize fields, no alignment concerns
  TVkBufferCopy = record
    srcOffset: UInt64;         // offset 0
    dstOffset: UInt64;         // offset 8
    size: UInt64;              // offset 16
  end;

  { TVkExtensionProperties }
  // C size: 260 bytes -- char extensionName[256]; uint32_t specVersion
  // Natural Delphi alignment matches C here (byte array then uint32)
  TVkExtensionProperties = record
    extensionName: array[0..255] of AnsiChar; // offset 0
    specVersion: UInt32;                      // offset 256
  end;

  { TVkPhysicalDeviceShaderIntegerDotProductFeatures }
  // C size: 24 bytes -- validated against vulkan_core.h
  TVkPhysicalDeviceShaderIntegerDotProductFeatures = record
    sType: UInt32;                     // offset 0
    pNext: Pointer;                    // offset 8
    shaderIntegerDotProduct: UInt32;   // offset 16 (VkBool32)
  end;

  // Vulkan function pointer types
  // Validated against vulkan_core.h from Vulkan SDK 1.4.350.0
  // VkResult = int32 (VkResult is an enum)
  // Dispatchable handles passed as UInt64 (matches pointer size on 64-bit)

  { TvkCreateInstance }
  // VkResult vkCreateInstance(const VkInstanceCreateInfo*, const VkAllocationCallbacks*, VkInstance*)
  TvkCreateInstance = function(const pCreateInfo: Pointer; pAllocator: Pointer; out instance: TVkInstance): Int32; stdcall;

  { TvkDestroyInstance }
  // void vkDestroyInstance(VkInstance, const VkAllocationCallbacks*)
  TvkDestroyInstance = procedure(instance: TVkInstance; pAllocator: Pointer); stdcall;

  { TvkEnumeratePhysicalDevices }
  // VkResult vkEnumeratePhysicalDevices(VkInstance, uint32_t*, VkPhysicalDevice*)
  TvkEnumeratePhysicalDevices = function(instance: TVkInstance; var count: UInt32; devices: Pointer): Int32; stdcall;

  { TvkGetPhysicalDeviceProperties }
  // void vkGetPhysicalDeviceProperties(VkPhysicalDevice, VkPhysicalDeviceProperties*)
  TvkGetPhysicalDeviceProperties = procedure(device: TVkPhysicalDevice; props: Pointer); stdcall;

  { TvkGetPhysicalDeviceQueueFamilyProperties }
  // void vkGetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice, uint32_t*, VkQueueFamilyProperties*)
  TvkGetPhysicalDeviceQueueFamilyProperties = procedure(device: TVkPhysicalDevice; var count: UInt32; props: Pointer); stdcall;

  { TvkGetPhysicalDeviceMemoryProperties }
  // void vkGetPhysicalDeviceMemoryProperties(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties*)
  TvkGetPhysicalDeviceMemoryProperties = procedure(device: TVkPhysicalDevice; props: Pointer); stdcall;

  { TvkEnumerateDeviceExtensionProperties }
  // VkResult vkEnumerateDeviceExtensionProperties(VkPhysicalDevice, const char*, uint32_t*, VkExtensionProperties*)
  TvkEnumerateDeviceExtensionProperties = function(physicalDevice: TVkPhysicalDevice; pLayerName: PAnsiChar; var pPropertyCount: UInt32; pProperties: Pointer): Int32; stdcall;

  { TvkCreateDevice }
  // VkResult vkCreateDevice(VkPhysicalDevice, const VkDeviceCreateInfo*, const VkAllocationCallbacks*, VkDevice*)
  TvkCreateDevice = function(physicalDevice: TVkPhysicalDevice; const pCreateInfo: Pointer; pAllocator: Pointer; out device: TVkDevice): Int32; stdcall;

  { TvkDestroyDevice }
  // void vkDestroyDevice(VkDevice, const VkAllocationCallbacks*)
  TvkDestroyDevice = procedure(device: TVkDevice; pAllocator: Pointer); stdcall;

  { TvkGetDeviceQueue }
  // void vkGetDeviceQueue(VkDevice, uint32_t, uint32_t, VkQueue*)
  TvkGetDeviceQueue = procedure(device: TVkDevice; queueFamilyIndex: UInt32; queueIndex: UInt32; out queue: TVkQueue); stdcall;

  { TvkCreateBuffer }
  // VkResult vkCreateBuffer(VkDevice, const VkBufferCreateInfo*, const VkAllocationCallbacks*, VkBuffer*)
  TvkCreateBuffer = function(device: TVkDevice; const pCreateInfo: Pointer; pAllocator: Pointer; out buffer: TVkBuffer): Int32; stdcall;

  { TvkDestroyBuffer }
  // void vkDestroyBuffer(VkDevice, VkBuffer, const VkAllocationCallbacks*)
  TvkDestroyBuffer = procedure(device: TVkDevice; buffer: TVkBuffer; pAllocator: Pointer); stdcall;

  { TvkGetBufferMemoryRequirements }
  // void vkGetBufferMemoryRequirements(VkDevice, VkBuffer, VkMemoryRequirements*)
  TvkGetBufferMemoryRequirements = procedure(device: TVkDevice; buffer: TVkBuffer; out reqs: TVkMemoryRequirements); stdcall;

  { TvkAllocateMemory }
  // VkResult vkAllocateMemory(VkDevice, const VkMemoryAllocateInfo*, const VkAllocationCallbacks*, VkDeviceMemory*)
  TvkAllocateMemory = function(device: TVkDevice; const pAllocateInfo: Pointer; pAllocator: Pointer; out memory: TVkDeviceMemory): Int32; stdcall;

  { TvkFreeMemory }
  // void vkFreeMemory(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*)
  TvkFreeMemory = procedure(device: TVkDevice; memory: TVkDeviceMemory; pAllocator: Pointer); stdcall;

  { TvkBindBufferMemory }
  // VkResult vkBindBufferMemory(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize)
  TvkBindBufferMemory = function(device: TVkDevice; buffer: TVkBuffer; memory: TVkDeviceMemory; offset: UInt64): Int32; stdcall;

  { TvkMapMemory }
  // VkResult vkMapMemory(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkMemoryMapFlags, void**)
  TvkMapMemory = function(device: TVkDevice; memory: TVkDeviceMemory; offset: UInt64; size: UInt64; flags: UInt32; out ppData: Pointer): Int32; stdcall;

  { TvkUnmapMemory }
  // void vkUnmapMemory(VkDevice, VkDeviceMemory)
  TvkUnmapMemory = procedure(device: TVkDevice; memory: TVkDeviceMemory); stdcall;

  { TvkCreateCommandPool }
  // VkResult vkCreateCommandPool(VkDevice, const VkCommandPoolCreateInfo*, const VkAllocationCallbacks*, VkCommandPool*)
  TvkCreateCommandPool = function(device: TVkDevice; const pCreateInfo: Pointer; pAllocator: Pointer; out commandPool: TVkCommandPool): Int32; stdcall;

  { TvkDestroyCommandPool }
  // void vkDestroyCommandPool(VkDevice, VkCommandPool, const VkAllocationCallbacks*)
  TvkDestroyCommandPool = procedure(device: TVkDevice; commandPool: TVkCommandPool; pAllocator: Pointer); stdcall;

  { TvkAllocateCommandBuffers }
  // VkResult vkAllocateCommandBuffers(VkDevice, const VkCommandBufferAllocateInfo*, VkCommandBuffer*)
  TvkAllocateCommandBuffers = function(device: TVkDevice; const pAllocateInfo: Pointer; pCommandBuffers: Pointer): Int32; stdcall;

  { TvkBeginCommandBuffer }
  // VkResult vkBeginCommandBuffer(VkCommandBuffer, const VkCommandBufferBeginInfo*)
  TvkBeginCommandBuffer = function(commandBuffer: TVkCommandBuffer; const pBeginInfo: Pointer): Int32; stdcall;

  { TvkEndCommandBuffer }
  // VkResult vkEndCommandBuffer(VkCommandBuffer)
  TvkEndCommandBuffer = function(commandBuffer: TVkCommandBuffer): Int32; stdcall;

  { TvkQueueSubmit }
  // VkResult vkQueueSubmit(VkQueue, uint32_t, const VkSubmitInfo*, VkFence)
  TvkQueueSubmit = function(queue: TVkQueue; submitCount: UInt32; pSubmits: Pointer; fence: TVkFence): Int32; stdcall;

  { TvkQueueWaitIdle }
  // VkResult vkQueueWaitIdle(VkQueue)
  TvkQueueWaitIdle = function(queue: TVkQueue): Int32; stdcall;

  { TvkDeviceWaitIdle }
  // VkResult vkDeviceWaitIdle(VkDevice)
  TvkDeviceWaitIdle = function(device: TVkDevice): Int32; stdcall;

  { TvkCreateFence }
  // VkResult vkCreateFence(VkDevice, const VkFenceCreateInfo*, const VkAllocationCallbacks*, VkFence*)
  TvkCreateFence = function(device: TVkDevice; const pCreateInfo: Pointer; pAllocator: Pointer; out fence: TVkFence): Int32; stdcall;

  { TvkDestroyFence }
  // void vkDestroyFence(VkDevice, VkFence, const VkAllocationCallbacks*)
  TvkDestroyFence = procedure(device: TVkDevice; fence: TVkFence; pAllocator: Pointer); stdcall;

  { TvkWaitForFences }
  // VkResult vkWaitForFences(VkDevice, uint32_t, const VkFence*, VkBool32, uint64_t)
  TvkWaitForFences = function(device: TVkDevice; fenceCount: UInt32; pFences: Pointer; waitAll: UInt32; timeout: UInt64): Int32; stdcall;

  { TvkResetFences }
  // VkResult vkResetFences(VkDevice, uint32_t, const VkFence*)
  TvkResetFences = function(device: TVkDevice; fenceCount: UInt32; pFences: Pointer): Int32; stdcall;

  { TvkResetCommandBuffer }
  // VkResult vkResetCommandBuffer(VkCommandBuffer, VkCommandBufferResetFlags)
  TvkResetCommandBuffer = function(commandBuffer: TVkCommandBuffer; flags: UInt32): Int32; stdcall;

  { TvkCmdBindPipeline }
  // void vkCmdBindPipeline(VkCommandBuffer, VkPipelineBindPoint, VkPipeline)
  TvkCmdBindPipeline = procedure(commandBuffer: TVkCommandBuffer; pipelineBindPoint: UInt32; pipeline: TVkPipeline); stdcall;

  { TvkCmdBindDescriptorSets }
  // void vkCmdBindDescriptorSets(VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, uint32_t, uint32_t, const VkDescriptorSet*, uint32_t, const uint32_t*)
  TvkCmdBindDescriptorSets = procedure(commandBuffer: TVkCommandBuffer; pipelineBindPoint: UInt32; layout: TVkPipelineLayout; firstSet: UInt32; descriptorSetCount: UInt32; pDescriptorSets: Pointer; dynamicOffsetCount: UInt32; pDynamicOffsets: Pointer); stdcall;

  { TvkCmdDispatch }
  // void vkCmdDispatch(VkCommandBuffer, uint32_t, uint32_t, uint32_t)
  TvkCmdDispatch = procedure(commandBuffer: TVkCommandBuffer; groupCountX: UInt32; groupCountY: UInt32; groupCountZ: UInt32); stdcall;

  { TvkCmdPushConstants }
  // void vkCmdPushConstants(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, uint32_t, uint32_t, const void*)
  TvkCmdPushConstants = procedure(commandBuffer: TVkCommandBuffer; layout: TVkPipelineLayout; stageFlags: UInt32; offset: UInt32; size: UInt32; pValues: Pointer); stdcall;

  { TvkCreateShaderModule }
  // VkResult vkCreateShaderModule(VkDevice, const VkShaderModuleCreateInfo*, const VkAllocationCallbacks*, VkShaderModule*)
  TvkCreateShaderModule = function(device: TVkDevice; const pCreateInfo: Pointer; pAllocator: Pointer; out shaderModule: TVkShaderModule): Int32; stdcall;

  { TvkDestroyShaderModule }
  // void vkDestroyShaderModule(VkDevice, VkShaderModule, const VkAllocationCallbacks*)
  TvkDestroyShaderModule = procedure(device: TVkDevice; shaderModule: TVkShaderModule; pAllocator: Pointer); stdcall;

  { TvkCreateDescriptorSetLayout }
  // VkResult vkCreateDescriptorSetLayout(VkDevice, const VkDescriptorSetLayoutCreateInfo*, const VkAllocationCallbacks*, VkDescriptorSetLayout*)
  TvkCreateDescriptorSetLayout = function(device: TVkDevice; const pCreateInfo: Pointer; pAllocator: Pointer; out setLayout: TVkDescriptorSetLayout): Int32; stdcall;

  { TvkDestroyDescriptorSetLayout }
  // void vkDestroyDescriptorSetLayout(VkDevice, VkDescriptorSetLayout, const VkAllocationCallbacks*)
  TvkDestroyDescriptorSetLayout = procedure(device: TVkDevice; descriptorSetLayout: TVkDescriptorSetLayout; pAllocator: Pointer); stdcall;

  { TvkCreatePipelineLayout }
  // VkResult vkCreatePipelineLayout(VkDevice, const VkPipelineLayoutCreateInfo*, const VkAllocationCallbacks*, VkPipelineLayout*)
  TvkCreatePipelineLayout = function(device: TVkDevice; const pCreateInfo: Pointer; pAllocator: Pointer; out pipelineLayout: TVkPipelineLayout): Int32; stdcall;

  { TvkDestroyPipelineLayout }
  // void vkDestroyPipelineLayout(VkDevice, VkPipelineLayout, const VkAllocationCallbacks*)
  TvkDestroyPipelineLayout = procedure(device: TVkDevice; pipelineLayout: TVkPipelineLayout; pAllocator: Pointer); stdcall;

  { TvkCreateComputePipelines }
  // VkResult vkCreateComputePipelines(VkDevice, VkPipelineCache, uint32_t, const VkComputePipelineCreateInfo*, const VkAllocationCallbacks*, VkPipeline*)
  TvkCreateComputePipelines = function(device: TVkDevice; pipelineCache: UInt64; createInfoCount: UInt32; pCreateInfos: Pointer; pAllocator: Pointer; pPipelines: Pointer): Int32; stdcall;

  { TvkDestroyPipeline }
  // void vkDestroyPipeline(VkDevice, VkPipeline, const VkAllocationCallbacks*)
  TvkDestroyPipeline = procedure(device: TVkDevice; pipeline: TVkPipeline; pAllocator: Pointer); stdcall;

  { TvkCreateDescriptorPool }
  // VkResult vkCreateDescriptorPool(VkDevice, const VkDescriptorPoolCreateInfo*, const VkAllocationCallbacks*, VkDescriptorPool*)
  TvkCreateDescriptorPool = function(device: TVkDevice; const pCreateInfo: Pointer; pAllocator: Pointer; out descriptorPool: TVkDescriptorPool): Int32; stdcall;

  { TvkDestroyDescriptorPool }
  // void vkDestroyDescriptorPool(VkDevice, VkDescriptorPool, const VkAllocationCallbacks*)
  TvkDestroyDescriptorPool = procedure(device: TVkDevice; descriptorPool: TVkDescriptorPool; pAllocator: Pointer); stdcall;

  { TvkAllocateDescriptorSets }
  // VkResult vkAllocateDescriptorSets(VkDevice, const VkDescriptorSetAllocateInfo*, VkDescriptorSet*)
  TvkAllocateDescriptorSets = function(device: TVkDevice; const pAllocateInfo: Pointer; pDescriptorSets: Pointer): Int32; stdcall;

  { TvkUpdateDescriptorSets }
  // void vkUpdateDescriptorSets(VkDevice, uint32_t, const VkWriteDescriptorSet*, uint32_t, const VkCopyDescriptorSet*)
  TvkUpdateDescriptorSets = procedure(device: TVkDevice; descriptorWriteCount: UInt32; pDescriptorWrites: Pointer; descriptorCopyCount: UInt32; pDescriptorCopies: Pointer); stdcall;

  { TvkCmdCopyBuffer }
  // void vkCmdCopyBuffer(VkCommandBuffer, VkBuffer, VkBuffer, uint32_t, const VkBufferCopy*)
  TvkCmdCopyBuffer = procedure(commandBuffer: TVkCommandBuffer; srcBuffer: TVkBuffer; dstBuffer: TVkBuffer; regionCount: UInt32; pRegions: Pointer); stdcall;

  { TvkCmdPipelineBarrier }
  // void vkCmdPipelineBarrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags, uint32_t, const VkMemoryBarrier*, uint32_t, const VkBufferMemoryBarrier*, uint32_t, const VkImageMemoryBarrier*)
  TvkCmdPipelineBarrier = procedure(commandBuffer: TVkCommandBuffer; srcStageMask: UInt32; dstStageMask: UInt32; dependencyFlags: UInt32; memoryBarrierCount: UInt32; pMemoryBarriers: Pointer; bufferMemoryBarrierCount: UInt32; pBufferMemoryBarriers: Pointer; imageMemoryBarrierCount: UInt32; pImageMemoryBarriers: Pointer); stdcall;

  { TvkResetDescriptorPool }
  // VkResult vkResetDescriptorPool(VkDevice, VkDescriptorPool, VkDescriptorPoolResetFlags)
  TvkResetDescriptorPool = function(device: TVkDevice; descriptorPool: TVkDescriptorPool; flags: UInt32): Int32; stdcall;

  { TvkCreateQueryPool }
  // VkResult vkCreateQueryPool(VkDevice, const VkQueryPoolCreateInfo*, const VkAllocationCallbacks*, VkQueryPool*)
  TvkCreateQueryPool = function(device: TVkDevice; const pCreateInfo: Pointer; pAllocator: Pointer; out queryPool: TVkQueryPool): Int32; stdcall;

  { TvkDestroyQueryPool }
  // void vkDestroyQueryPool(VkDevice, VkQueryPool, const VkAllocationCallbacks*)
  TvkDestroyQueryPool = procedure(device: TVkDevice; queryPool: TVkQueryPool; pAllocator: Pointer); stdcall;

  { TvkCmdResetQueryPool }
  // void vkCmdResetQueryPool(VkCommandBuffer, VkQueryPool, uint32_t, uint32_t)
  TvkCmdResetQueryPool = procedure(commandBuffer: TVkCommandBuffer; queryPool: TVkQueryPool; firstQuery: UInt32; queryCount: UInt32); stdcall;

  { TvkCmdWriteTimestamp }
  // void vkCmdWriteTimestamp(VkCommandBuffer, VkPipelineStageFlagBits, VkQueryPool, uint32_t)
  TvkCmdWriteTimestamp = procedure(commandBuffer: TVkCommandBuffer; pipelineStage: UInt32; queryPool: TVkQueryPool; query: UInt32); stdcall;

  { TvkGetQueryPoolResults }
  // VkResult vkGetQueryPoolResults(VkDevice, VkQueryPool, uint32_t, uint32_t, size_t, void*, VkDeviceSize, VkQueryResultFlags)
  TvkGetQueryPoolResults = function(device: TVkDevice; queryPool: TVkQueryPool; firstQuery: UInt32; queryCount: UInt32; dataSize: NativeUInt; pData: Pointer; stride: UInt64; flags: UInt32): Int32; stdcall;


  { TVulkanBuffer }
  // GPU buffer with optional host-mapped pointer
  TVulkanBuffer = record
    Buffer: TVkBuffer;
    Memory: TVkDeviceMemory;
    MappedPtr: Pointer;
    ByteSize: UInt64;
    IsDeviceLocal: Boolean;
  end;

  { TVulkanDevice }
  // Manages Vulkan lifecycle and provides buffer/command operations
  TVulkanDevice = class(TBaseObject)
  private
    FLibHandle: THandle;
    FInstance: TVkInstance;
    FPhysicalDevice: TVkPhysicalDevice;
    FDevice: TVkDevice;
    FComputeQueue: TVkQueue;
    FCommandPool: TVkCommandPool;
    FCommandBuffer: TVkCommandBuffer;
    FFence: TVkFence;
    FComputeQueueFamily: UInt32;
    FMemoryProperties: TVkPhysicalDeviceMemoryProperties;
    FDeviceName: string;
    FIsInitialized: Boolean;
    FHasIntegerDotProduct: Boolean;

    // GPU timestamp profiling state
    FTimestampPool: TVkQueryPool;
    FTimestampCount: UInt32;
    FTimestampPeriod: Single;  // nanoseconds per timestamp tick

    // Function pointers
    FvkCreateInstance: TvkCreateInstance;
    FvkDestroyInstance: TvkDestroyInstance;
    FvkEnumeratePhysicalDevices: TvkEnumeratePhysicalDevices;
    FvkGetPhysicalDeviceProperties: TvkGetPhysicalDeviceProperties;
    FvkGetPhysicalDeviceQueueFamilyProperties: TvkGetPhysicalDeviceQueueFamilyProperties;
    FvkGetPhysicalDeviceMemoryProperties: TvkGetPhysicalDeviceMemoryProperties;
    FvkEnumerateDeviceExtensionProperties: TvkEnumerateDeviceExtensionProperties;
    FvkCreateDevice: TvkCreateDevice;
    FvkDestroyDevice: TvkDestroyDevice;
    FvkGetDeviceQueue: TvkGetDeviceQueue;
    FvkCreateBuffer: TvkCreateBuffer;
    FvkDestroyBuffer: TvkDestroyBuffer;
    FvkGetBufferMemoryRequirements: TvkGetBufferMemoryRequirements;
    FvkAllocateMemory: TvkAllocateMemory;
    FvkFreeMemory: TvkFreeMemory;
    FvkBindBufferMemory: TvkBindBufferMemory;
    FvkMapMemory: TvkMapMemory;
    FvkUnmapMemory: TvkUnmapMemory;
    FvkCreateCommandPool: TvkCreateCommandPool;
    FvkDestroyCommandPool: TvkDestroyCommandPool;
    FvkAllocateCommandBuffers: TvkAllocateCommandBuffers;
    FvkBeginCommandBuffer: TvkBeginCommandBuffer;
    FvkEndCommandBuffer: TvkEndCommandBuffer;
    FvkQueueSubmit: TvkQueueSubmit;
    FvkQueueWaitIdle: TvkQueueWaitIdle;
    FvkDeviceWaitIdle: TvkDeviceWaitIdle;
    FvkCreateFence: TvkCreateFence;
    FvkDestroyFence: TvkDestroyFence;
    FvkWaitForFences: TvkWaitForFences;
    FvkResetFences: TvkResetFences;
    FvkResetCommandBuffer: TvkResetCommandBuffer;
    FvkCmdBindPipeline: TvkCmdBindPipeline;
    FvkCmdBindDescriptorSets: TvkCmdBindDescriptorSets;
    FvkCmdDispatch: TvkCmdDispatch;
    FvkCmdPushConstants: TvkCmdPushConstants;
    FvkCreateShaderModule: TvkCreateShaderModule;
    FvkDestroyShaderModule: TvkDestroyShaderModule;
    FvkCreateDescriptorSetLayout: TvkCreateDescriptorSetLayout;
    FvkDestroyDescriptorSetLayout: TvkDestroyDescriptorSetLayout;
    FvkCreatePipelineLayout: TvkCreatePipelineLayout;
    FvkDestroyPipelineLayout: TvkDestroyPipelineLayout;
    FvkCreateComputePipelines: TvkCreateComputePipelines;
    FvkDestroyPipeline: TvkDestroyPipeline;
    FvkCreateDescriptorPool: TvkCreateDescriptorPool;
    FvkDestroyDescriptorPool: TvkDestroyDescriptorPool;
    FvkAllocateDescriptorSets: TvkAllocateDescriptorSets;
    FvkUpdateDescriptorSets: TvkUpdateDescriptorSets;
    FvkCmdCopyBuffer: TvkCmdCopyBuffer;
    FvkCmdPipelineBarrier: TvkCmdPipelineBarrier;
    FvkResetDescriptorPool: TvkResetDescriptorPool;
    FvkCreateQueryPool: TvkCreateQueryPool;
    FvkDestroyQueryPool: TvkDestroyQueryPool;
    FvkCmdResetQueryPool: TvkCmdResetQueryPool;
    FvkCmdWriteTimestamp: TvkCmdWriteTimestamp;
    FvkGetQueryPoolResults: TvkGetQueryPoolResults;

    function DoLoadLibrary(): Boolean;
    function DoLoadFunctions(): Boolean;
    function DoCreateInstance(): Boolean;
    function DoSelectPhysicalDevice(): Boolean;
    function DoFindComputeQueue(): Boolean;
    function DoCreateLogicalDevice(): Boolean;
    function DoCreateCommandPool(): Boolean;
    function DoCreateFence(): Boolean;
    function DoFindMemoryType(const ATypeBits: UInt32;
      const AProperties: UInt32): Int32;
  public
    constructor Create(); override;
    destructor Destroy(); override;

    // Initialize Vulkan (load lib, create instance/device/queue)
    function Init(): Boolean;

    // Shutdown and release all resources
    procedure Shutdown();

    function IsInitialized(): Boolean;

    // True when the device supports VK_KHR_shader_integer_dot_product
    // (DP4A hardware integer dot products, used by the q4q8 matvec path)
    function HasIntegerDotProduct(): Boolean;

    // Buffer operations
    function CreateBuffer(const ASize: UInt64;
      const AHostVisible: Boolean): TVulkanBuffer;
    procedure DestroyBuffer(var ABuffer: TVulkanBuffer);
    procedure UploadToBuffer(var ABuffer: TVulkanBuffer;
      const AData: Pointer; const ASize: UInt64);
    procedure DownloadFromBuffer(const ABuffer: TVulkanBuffer;
      const ADst: Pointer; const ASize: UInt64);

    // Upload to ANY buffer placement. Host-visible: direct mapped write.
    // Device-local: streams through a host-visible staging buffer using
    // vkCmdCopyBuffer in chunks. Use for the VRAM-resident weight blob.
    // ADstOffset places the data at a byte offset inside the destination.
    procedure UploadToDeviceBuffer(var ABuffer: TVulkanBuffer;
      const AData: Pointer; const ASize: UInt64;
      const ADstOffset: UInt64 = 0);

    // Download from ANY buffer placement (staging readback for device-local).
    // ASrcOffset reads from a byte offset inside the source buffer.
    procedure DownloadFromDeviceBuffer(const ABuffer: TVulkanBuffer;
      const ADst: Pointer; const ASize: UInt64;
      const ASrcOffset: UInt64 = 0);

    // Command buffer recording
    procedure BeginCommands();
    procedure EndCommands();
    procedure SubmitAndWait();

    // Record a compute->compute memory barrier on the current command buffer.
    // Makes prior shader writes visible to subsequent shader reads/writes.
    procedure RecordComputeBarrier();

    // -- GPU timestamp profiling --
    // All methods are safe no-ops if the pool was never created.
    function CreateTimestampPool(const ACount: UInt32): Boolean;
    procedure DestroyTimestampPool();
    procedure RecordResetTimestamps();  // record at command buffer start
    procedure RecordTimestamp(const AIndex: UInt32);
    function FetchTimestamps(out AValues: TArray<UInt64>): Boolean;

    // Expose for Shaders/Compute units
    property Device: TVkDevice read FDevice;
    property ComputeQueue: TVkQueue read FComputeQueue;
    property CommandBuffer: TVkCommandBuffer read FCommandBuffer;
    property DeviceName: string read FDeviceName;
    property TimestampPeriod: Single read FTimestampPeriod;

    // Expose function pointers for Shaders/Compute
    property vkCreateShaderModule: TvkCreateShaderModule read FvkCreateShaderModule;
    property vkDestroyShaderModule: TvkDestroyShaderModule read FvkDestroyShaderModule;
    property vkCreateDescriptorSetLayout: TvkCreateDescriptorSetLayout read FvkCreateDescriptorSetLayout;
    property vkDestroyDescriptorSetLayout: TvkDestroyDescriptorSetLayout read FvkDestroyDescriptorSetLayout;
    property vkCreatePipelineLayout: TvkCreatePipelineLayout read FvkCreatePipelineLayout;
    property vkDestroyPipelineLayout: TvkDestroyPipelineLayout read FvkDestroyPipelineLayout;
    property vkCreateComputePipelines: TvkCreateComputePipelines read FvkCreateComputePipelines;
    property vkDestroyPipeline: TvkDestroyPipeline read FvkDestroyPipeline;
    property vkCreateDescriptorPool: TvkCreateDescriptorPool read FvkCreateDescriptorPool;
    property vkDestroyDescriptorPool: TvkDestroyDescriptorPool read FvkDestroyDescriptorPool;
    property vkAllocateDescriptorSets: TvkAllocateDescriptorSets read FvkAllocateDescriptorSets;
    property vkUpdateDescriptorSets: TvkUpdateDescriptorSets read FvkUpdateDescriptorSets;
    property vkCmdBindPipeline: TvkCmdBindPipeline read FvkCmdBindPipeline;
    property vkCmdBindDescriptorSets: TvkCmdBindDescriptorSets read FvkCmdBindDescriptorSets;
    property vkCmdDispatch: TvkCmdDispatch read FvkCmdDispatch;
    property vkCmdPushConstants: TvkCmdPushConstants read FvkCmdPushConstants;
    property vkCmdCopyBuffer: TvkCmdCopyBuffer read FvkCmdCopyBuffer;
    property vkCmdPipelineBarrier: TvkCmdPipelineBarrier read FvkCmdPipelineBarrier;
    property vkResetDescriptorPool: TvkResetDescriptorPool read FvkResetDescriptorPool;
  end;

implementation

{ TVulkanDevice }

constructor TVulkanDevice.Create();
begin
  inherited Create();
  FLibHandle := 0;
  FInstance := 0;
  FPhysicalDevice := 0;
  FDevice := 0;
  FComputeQueue := 0;
  FCommandPool := 0;
  FCommandBuffer := 0;
  FFence := 0;
  FComputeQueueFamily := 0;
  FIsInitialized := False;
  FHasIntegerDotProduct := False;
  FDeviceName := '';
end;

destructor TVulkanDevice.Destroy();
begin
  Shutdown();
  inherited;
end;

function TVulkanDevice.DoLoadLibrary(): Boolean;
begin
  Result := False;
  FLibHandle := LoadLibrary('vulkan-1.dll');
  if FLibHandle = 0 then
  begin
    GetErrors().Add(esError, CVK_ERR_LOAD_LIB, 'Failed to load vulkan-1.dll');
    Exit;
  end;
  Result := True;
end;

function TVulkanDevice.DoLoadFunctions(): Boolean;

  function Load(const AName: AnsiString): Pointer;
  begin
    Result := GetProcAddress(FLibHandle, PAnsiChar(AName));
  end;

begin
  FvkCreateInstance := Load('vkCreateInstance');
  FvkDestroyInstance := Load('vkDestroyInstance');
  FvkEnumeratePhysicalDevices := Load('vkEnumeratePhysicalDevices');
  FvkGetPhysicalDeviceProperties := Load('vkGetPhysicalDeviceProperties');
  FvkGetPhysicalDeviceQueueFamilyProperties := Load('vkGetPhysicalDeviceQueueFamilyProperties');
  FvkGetPhysicalDeviceMemoryProperties := Load('vkGetPhysicalDeviceMemoryProperties');
  FvkEnumerateDeviceExtensionProperties := Load('vkEnumerateDeviceExtensionProperties');
  FvkCreateDevice := Load('vkCreateDevice');
  FvkDestroyDevice := Load('vkDestroyDevice');
  FvkGetDeviceQueue := Load('vkGetDeviceQueue');
  FvkCreateBuffer := Load('vkCreateBuffer');
  FvkDestroyBuffer := Load('vkDestroyBuffer');
  FvkGetBufferMemoryRequirements := Load('vkGetBufferMemoryRequirements');
  FvkAllocateMemory := Load('vkAllocateMemory');
  FvkFreeMemory := Load('vkFreeMemory');
  FvkBindBufferMemory := Load('vkBindBufferMemory');
  FvkMapMemory := Load('vkMapMemory');
  FvkUnmapMemory := Load('vkUnmapMemory');
  FvkCreateCommandPool := Load('vkCreateCommandPool');
  FvkDestroyCommandPool := Load('vkDestroyCommandPool');
  FvkAllocateCommandBuffers := Load('vkAllocateCommandBuffers');
  FvkBeginCommandBuffer := Load('vkBeginCommandBuffer');
  FvkEndCommandBuffer := Load('vkEndCommandBuffer');
  FvkQueueSubmit := Load('vkQueueSubmit');
  FvkQueueWaitIdle := Load('vkQueueWaitIdle');
  FvkDeviceWaitIdle := Load('vkDeviceWaitIdle');
  FvkCreateFence := Load('vkCreateFence');
  FvkDestroyFence := Load('vkDestroyFence');
  FvkWaitForFences := Load('vkWaitForFences');
  FvkResetFences := Load('vkResetFences');
  FvkResetCommandBuffer := Load('vkResetCommandBuffer');
  FvkCmdBindPipeline := Load('vkCmdBindPipeline');
  FvkCmdBindDescriptorSets := Load('vkCmdBindDescriptorSets');
  FvkCmdDispatch := Load('vkCmdDispatch');
  FvkCmdPushConstants := Load('vkCmdPushConstants');
  FvkCreateShaderModule := Load('vkCreateShaderModule');
  FvkDestroyShaderModule := Load('vkDestroyShaderModule');
  FvkCreateDescriptorSetLayout := Load('vkCreateDescriptorSetLayout');
  FvkDestroyDescriptorSetLayout := Load('vkDestroyDescriptorSetLayout');
  FvkCreatePipelineLayout := Load('vkCreatePipelineLayout');
  FvkDestroyPipelineLayout := Load('vkDestroyPipelineLayout');
  FvkCreateComputePipelines := Load('vkCreateComputePipelines');
  FvkDestroyPipeline := Load('vkDestroyPipeline');
  FvkCreateDescriptorPool := Load('vkCreateDescriptorPool');
  FvkDestroyDescriptorPool := Load('vkDestroyDescriptorPool');
  FvkAllocateDescriptorSets := Load('vkAllocateDescriptorSets');
  FvkUpdateDescriptorSets := Load('vkUpdateDescriptorSets');
  FvkCmdCopyBuffer := Load('vkCmdCopyBuffer');
  FvkCmdPipelineBarrier := Load('vkCmdPipelineBarrier');
  FvkResetDescriptorPool := Load('vkResetDescriptorPool');
  FvkCreateQueryPool := Load('vkCreateQueryPool');
  FvkDestroyQueryPool := Load('vkDestroyQueryPool');
  FvkCmdResetQueryPool := Load('vkCmdResetQueryPool');
  FvkCmdWriteTimestamp := Load('vkCmdWriteTimestamp');
  FvkGetQueryPoolResults := Load('vkGetQueryPoolResults');

  Result := Assigned(FvkCreateInstance) and
            Assigned(FvkDestroyInstance) and
            Assigned(FvkEnumeratePhysicalDevices) and
            Assigned(FvkGetPhysicalDeviceProperties) and
            Assigned(FvkGetPhysicalDeviceQueueFamilyProperties) and
            Assigned(FvkGetPhysicalDeviceMemoryProperties) and
            Assigned(FvkCreateDevice) and
            Assigned(FvkDestroyDevice) and
            Assigned(FvkGetDeviceQueue);
end;

function TVulkanDevice.DoCreateInstance(): Boolean;
var
  LAppInfo: TVkApplicationInfo;
  LCreateInfo: TVkInstanceCreateInfo;
  LResult: Int32;
begin
  Result := False;

  FillChar(LAppInfo, SizeOf(LAppInfo), 0);
  LAppInfo.sType := CVK_STRUCTURE_TYPE_APPLICATION_INFO;
  LAppInfo.pApplicationName := 'Gemma4.pas';
  LAppInfo.applicationVersion := 1;
  LAppInfo.pEngineName := 'Gemma4';
  LAppInfo.engineVersion := 1;
  LAppInfo.apiVersion := CVK_API_VERSION_1_1;

  FillChar(LCreateInfo, SizeOf(LCreateInfo), 0);
  LCreateInfo.sType := CVK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  LCreateInfo.pApplicationInfo := @LAppInfo;

  LResult := FvkCreateInstance(@LCreateInfo, nil, FInstance);
  if LResult <> CVK_SUCCESS then
  begin
    GetErrors().Add(esError, CVK_ERR_INSTANCE, 'vkCreateInstance failed: %d', [LResult]);
    Exit;
  end;

  Result := True;
end;

function TVulkanDevice.DoSelectPhysicalDevice(): Boolean;
var
  LCount: UInt32;
  LDevices: TArray<TVkPhysicalDevice>;
  LProps: TVkPhysicalDeviceProperties;
  LI: Integer;
  LBestIdx: Integer;
begin
  Result := False;
  LCount := 0;
  FvkEnumeratePhysicalDevices(FInstance, LCount, nil);
  if LCount = 0 then
  begin
    GetErrors().Add(esError, CVK_ERR_DEVICE, 'No Vulkan physical devices found');
    Exit;
  end;

  SetLength(LDevices, LCount);
  FvkEnumeratePhysicalDevices(FInstance, LCount, @LDevices[0]);

  // Prefer discrete GPU
  LBestIdx := 0;
  for LI := 0 to LCount - 1 do
  begin
    FvkGetPhysicalDeviceProperties(LDevices[LI], @LProps);
    if LProps.deviceType = CVK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU then
    begin
      LBestIdx := LI;
      Break;
    end;
  end;

  FPhysicalDevice := LDevices[LBestIdx];
  FvkGetPhysicalDeviceProperties(FPhysicalDevice, @LProps);
  FDeviceName := string(AnsiString(PAnsiChar(@LProps.deviceName[0])));
  FvkGetPhysicalDeviceMemoryProperties(FPhysicalDevice, @FMemoryProperties);

  // Nanoseconds per GPU timestamp tick (for the timestamp profiling API)
  FTimestampPeriod := PSingle(@LProps.limits[CVK_LIMITS_TIMESTAMP_PERIOD_OFS])^;

  // Subgroup ops (matvec reduction) are core in Vulkan 1.1
  if LProps.apiVersion < CVK_API_VERSION_1_1 then
  begin
    GetErrors().Add(esError, CVK_ERR_DEVICE,
      'Device does not support Vulkan 1.1 (subgroup operations required)');
    Exit;
  end;

  Status('Vulkan device: %s', [FDeviceName]);
  Result := True;
end;

function TVulkanDevice.DoFindComputeQueue(): Boolean;
var
  LCount: UInt32;
  LFamilies: TArray<TVkQueueFamilyProperties>;
  LI: Integer;
begin
  Result := False;
  LCount := 0;

  FvkGetPhysicalDeviceQueueFamilyProperties(FPhysicalDevice, LCount, nil);
  if LCount = 0 then
  begin
    GetErrors().Add(esError, CVK_ERR_QUEUE, 'No queue families found');
    Exit;
  end;
  SetLength(LFamilies, LCount);
  FvkGetPhysicalDeviceQueueFamilyProperties(FPhysicalDevice, LCount, @LFamilies[0]);

  for LI := 0 to LCount - 1 do
  begin
    if (LFamilies[LI].queueFlags and CVK_QUEUE_COMPUTE_BIT) <> 0 then
    begin
      FComputeQueueFamily := UInt32(LI);
      Result := True;
      Exit;
    end;
  end;

  GetErrors().Add(esError, CVK_ERR_QUEUE, 'No compute queue family found');
end;

function TVulkanDevice.DoCreateLogicalDevice(): Boolean;
var
  LQueueInfo: TVkDeviceQueueCreateInfo;
  LDeviceInfo: TVkDeviceCreateInfo;
  LDotFeatures: TVkPhysicalDeviceShaderIntegerDotProductFeatures;
  LExtCount: UInt32;
  LExtensions: TArray<TVkExtensionProperties>;
  LExtName: AnsiString;
  LExtNamePtr: PAnsiChar;
  LPriority: Single;
  LResult: Int32;
  LI: Integer;
begin
  Result := False;
  LPriority := 1.0;

  // Detect VK_KHR_shader_integer_dot_product (DP4A) on this device.
  // Any failure along the way leaves FHasIntegerDotProduct = False and the
  // device is created exactly as before (float matvec fallback path).
  FHasIntegerDotProduct := False;
  if Assigned(FvkEnumerateDeviceExtensionProperties) then
  begin
    LExtCount := 0;
    if (FvkEnumerateDeviceExtensionProperties(FPhysicalDevice, nil,
      LExtCount, nil) = CVK_SUCCESS) and (LExtCount > 0) then
    begin
      SetLength(LExtensions, LExtCount);
      if FvkEnumerateDeviceExtensionProperties(FPhysicalDevice, nil,
        LExtCount, @LExtensions[0]) = CVK_SUCCESS then
      begin
        for LI := 0 to Integer(LExtCount) - 1 do
        begin
          if string(AnsiString(PAnsiChar(@LExtensions[LI].extensionName[0]))) =
            CVK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXT_NAME then
          begin
            FHasIntegerDotProduct := True;
            Break;
          end;
        end;
      end;
    end;
  end;

  FillChar(LQueueInfo, SizeOf(LQueueInfo), 0);
  LQueueInfo.sType := CVK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  LQueueInfo.queueFamilyIndex := FComputeQueueFamily;
  LQueueInfo.queueCount := 1;
  LQueueInfo.pQueuePriorities := @LPriority;

  FillChar(LDeviceInfo, SizeOf(LDeviceInfo), 0);
  LDeviceInfo.sType := CVK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  LDeviceInfo.queueCreateInfoCount := 1;
  LDeviceInfo.pQueueCreateInfos := @LQueueInfo;

  if FHasIntegerDotProduct then
  begin
    // LExtName / LExtNamePtr / LDotFeatures stay alive across vkCreateDevice
    LExtName := CVK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXT_NAME;
    LExtNamePtr := PAnsiChar(LExtName);
    LDeviceInfo.enabledExtensionCount := 1;
    LDeviceInfo.ppEnabledExtensionNames := @LExtNamePtr;

    FillChar(LDotFeatures, SizeOf(LDotFeatures), 0);
    LDotFeatures.sType :=
      CVK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
    LDotFeatures.pNext := nil;
    LDotFeatures.shaderIntegerDotProduct := 1;
    LDeviceInfo.pNext := @LDotFeatures;

    Status('Integer dot product: supported', []);
  end
  else
    Status('Integer dot product: not supported (float path)', []);

  LResult := FvkCreateDevice(FPhysicalDevice, @LDeviceInfo, nil, FDevice);
  if LResult <> CVK_SUCCESS then
  begin
    GetErrors().Add(esError, CVK_ERR_DEVICE, 'vkCreateDevice failed: %d', [LResult]);
    Exit;
  end;

  FvkGetDeviceQueue(FDevice, FComputeQueueFamily, 0, FComputeQueue);
  Result := True;
end;

function TVulkanDevice.DoCreateCommandPool(): Boolean;
var
  LPoolInfo: TVkCommandPoolCreateInfo;
  LAllocInfo: TVkCommandBufferAllocateInfo;
  LResult: Int32;
begin
  Result := False;

  FillChar(LPoolInfo, SizeOf(LPoolInfo), 0);
  LPoolInfo.sType := CVK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  LPoolInfo.flags := CVK_COMMAND_POOL_CREATE_RESET;
  LPoolInfo.queueFamilyIndex := FComputeQueueFamily;

  LResult := FvkCreateCommandPool(FDevice, @LPoolInfo, nil, FCommandPool);
  if LResult <> CVK_SUCCESS then
  begin
    GetErrors().Add(esError, CVK_ERR_COMMAND, 'vkCreateCommandPool failed: %d', [LResult]);
    Exit;
  end;

  FillChar(LAllocInfo, SizeOf(LAllocInfo), 0);
  LAllocInfo.sType := CVK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  LAllocInfo.commandPool := FCommandPool;
  LAllocInfo.level := CVK_COMMAND_BUFFER_LEVEL_PRIMARY;
  LAllocInfo.commandBufferCount := 1;

  LResult := FvkAllocateCommandBuffers(FDevice, @LAllocInfo, @FCommandBuffer);
  if LResult <> CVK_SUCCESS then
  begin
    GetErrors().Add(esError, CVK_ERR_COMMAND, 'vkAllocateCommandBuffers failed: %d', [LResult]);
    Exit;
  end;

  Result := True;
end;

function TVulkanDevice.DoCreateFence(): Boolean;
var
  LFenceInfo: TVkFenceCreateInfo;
  LResult: Int32;
begin
  Result := False;

  FillChar(LFenceInfo, SizeOf(LFenceInfo), 0);
  LFenceInfo.sType := CVK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

  LResult := FvkCreateFence(FDevice, @LFenceInfo, nil, FFence);
  if LResult <> CVK_SUCCESS then
  begin
    GetErrors().Add(esError, CVK_ERR_FENCE, 'vkCreateFence failed: %d', [LResult]);
    Exit;
  end;

  Result := True;
end;

function TVulkanDevice.DoFindMemoryType(const ATypeBits: UInt32;
  const AProperties: UInt32): Int32;
var
  LI: Integer;
begin
  for LI := 0 to Integer(FMemoryProperties.memoryTypeCount) - 1 do
  begin
    if ((ATypeBits and (1 shl LI)) <> 0) and
       ((FMemoryProperties.memoryTypes[LI].propertyFlags and AProperties) = AProperties) then
      Exit(LI);
  end;
  Result := -1;
end;

function TVulkanDevice.Init(): Boolean;
begin
  Result := False;
  if FIsInitialized then Exit(True);

  if not DoLoadLibrary() then Exit;
  if not DoLoadFunctions() then Exit;
  if not DoCreateInstance() then Exit;
  if not DoSelectPhysicalDevice() then Exit;
  if not DoFindComputeQueue() then Exit;
  if not DoCreateLogicalDevice() then Exit;
  if not DoCreateCommandPool() then Exit;
  if not DoCreateFence() then Exit;

  FIsInitialized := True;
  Result := True;
end;

procedure TVulkanDevice.Shutdown();
begin
  if not FIsInitialized then Exit;

  if FDevice <> 0 then
    FvkDeviceWaitIdle(FDevice);

  // Safety net: pool is normally destroyed by its creator (TModel)
  DestroyTimestampPool();

  if FFence <> 0 then
    FvkDestroyFence(FDevice, FFence, nil);
  if FCommandPool <> 0 then
    FvkDestroyCommandPool(FDevice, FCommandPool, nil);
  if FDevice <> 0 then
    FvkDestroyDevice(FDevice, nil);
  if FInstance <> 0 then
    FvkDestroyInstance(FInstance, nil);
  if FLibHandle <> 0 then
    FreeLibrary(FLibHandle);

  FInstance := 0;
  FDevice := 0;
  FCommandPool := 0;
  FCommandBuffer := 0;
  FFence := 0;
  FLibHandle := 0;
  FIsInitialized := False;
end;

function TVulkanDevice.IsInitialized(): Boolean;
begin
  Result := FIsInitialized;
end;

function TVulkanDevice.HasIntegerDotProduct(): Boolean;
begin
  Result := FHasIntegerDotProduct;
end;

function TVulkanDevice.CreateBuffer(const ASize: UInt64;
  const AHostVisible: Boolean): TVulkanBuffer;
var
  LBufInfo: TVkBufferCreateInfo;
  LMemReqs: TVkMemoryRequirements;
  LAllocInfo: TVkMemoryAllocateInfo;
  LMemProps: UInt32;
  LMemTypeIdx: Int32;
  LResult: Int32;
begin
  Result := Default(TVulkanBuffer);
  Result.ByteSize := ASize;
  Result.IsDeviceLocal := not AHostVisible;

  FillChar(LBufInfo, SizeOf(LBufInfo), 0);
  LBufInfo.sType := CVK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  LBufInfo.size := ASize;
  LBufInfo.usage := CVK_BUFFER_USAGE_STORAGE or CVK_BUFFER_USAGE_TRANSFER_SRC or CVK_BUFFER_USAGE_TRANSFER_DST;
  LBufInfo.sharingMode := CVK_SHARING_MODE_EXCLUSIVE;

  LResult := FvkCreateBuffer(FDevice, @LBufInfo, nil, Result.Buffer);
  if LResult <> CVK_SUCCESS then
  begin
    GetErrors().Add(esError, CVK_ERR_BUFFER, 'vkCreateBuffer failed: %d', [LResult]);
    Exit;
  end;

  FvkGetBufferMemoryRequirements(FDevice, Result.Buffer, LMemReqs);

  if AHostVisible then
    // Prefer HOST_CACHED: CPU reads from write-combined (uncached) memory run
    // at ~240 MB/s; a cached type restores full-speed logits readback. The
    // fallback below covers devices with no cached host-visible type.
    LMemProps := CVK_MEMORY_PROPERTY_HOST_VISIBLE or CVK_MEMORY_PROPERTY_HOST_COHERENT or CVK_MEMORY_PROPERTY_HOST_CACHED
  else
    LMemProps := CVK_MEMORY_PROPERTY_DEVICE_LOCAL;

  LMemTypeIdx := DoFindMemoryType(LMemReqs.memoryTypeBits, LMemProps);
  if LMemTypeIdx < 0 then
  begin
    // Fallback to host visible if device local not available
    LMemProps := CVK_MEMORY_PROPERTY_HOST_VISIBLE or CVK_MEMORY_PROPERTY_HOST_COHERENT;
    LMemTypeIdx := DoFindMemoryType(LMemReqs.memoryTypeBits, LMemProps);
    Result.IsDeviceLocal := False;
  end;

  if LMemTypeIdx < 0 then
  begin
    GetErrors().Add(esError, CVK_ERR_MEMORY, 'No suitable memory type found');
    FvkDestroyBuffer(FDevice, Result.Buffer, nil);
    Result.Buffer := 0;
    Exit;
  end;

  FillChar(LAllocInfo, SizeOf(LAllocInfo), 0);
  LAllocInfo.sType := CVK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  LAllocInfo.allocationSize := LMemReqs.size;
  LAllocInfo.memoryTypeIndex := UInt32(LMemTypeIdx);

  LResult := FvkAllocateMemory(FDevice, @LAllocInfo, nil, Result.Memory);
  if LResult <> CVK_SUCCESS then
  begin
    GetErrors().Add(esError, CVK_ERR_MEMORY, 'vkAllocateMemory failed: %d', [LResult]);
    FvkDestroyBuffer(FDevice, Result.Buffer, nil);
    Result.Buffer := 0;
    Exit;
  end;

  FvkBindBufferMemory(FDevice, Result.Buffer, Result.Memory, 0);

  // Map host-visible buffers permanently
  if AHostVisible then
    FvkMapMemory(FDevice, Result.Memory, 0, CVK_WHOLE_SIZE, 0, Result.MappedPtr);
end;

procedure TVulkanDevice.DestroyBuffer(var ABuffer: TVulkanBuffer);
begin
  if ABuffer.MappedPtr <> nil then
  begin
    FvkUnmapMemory(FDevice, ABuffer.Memory);
    ABuffer.MappedPtr := nil;
  end;
  if ABuffer.Buffer <> 0 then
    FvkDestroyBuffer(FDevice, ABuffer.Buffer, nil);
  if ABuffer.Memory <> 0 then
    FvkFreeMemory(FDevice, ABuffer.Memory, nil);
  ABuffer := Default(TVulkanBuffer);
end;

procedure TVulkanDevice.UploadToBuffer(var ABuffer: TVulkanBuffer;
  const AData: Pointer; const ASize: UInt64);
begin
  if ABuffer.MappedPtr <> nil then
    Move(AData^, ABuffer.MappedPtr^, ASize);
end;

procedure TVulkanDevice.DownloadFromBuffer(const ABuffer: TVulkanBuffer;
  const ADst: Pointer; const ASize: UInt64);
begin
  if ABuffer.MappedPtr <> nil then
    Move(ABuffer.MappedPtr^, ADst^, ASize);
end;

procedure TVulkanDevice.UploadToDeviceBuffer(var ABuffer: TVulkanBuffer;
  const AData: Pointer; const ASize: UInt64;
  const ADstOffset: UInt64);
const
  // Staging chunk size: 256 MB keeps the transient allocation modest while
  // uploading the 4.3 GB weight blob in ~17 submits at load time.
  CStagingChunkSize = UInt64(256) * 1024 * 1024;
var
  LStaging: TVulkanBuffer;
  LChunk: UInt64;
  LOffset: UInt64;
  LRegion: TVkBufferCopy;
  LSrcPtr: Pointer;
begin
  // Host-visible destination: direct mapped write, no staging needed
  if ABuffer.MappedPtr <> nil then
  begin
    Move(AData^, Pointer(UIntPtr(ABuffer.MappedPtr) + UIntPtr(ADstOffset))^, ASize);
    Exit;
  end;

  // Device-local destination: stream chunks through a staging buffer
  LChunk := ASize;
  if LChunk > CStagingChunkSize then
    LChunk := CStagingChunkSize;

  LStaging := CreateBuffer(LChunk, True);
  if (LStaging.Buffer = 0) or (LStaging.MappedPtr = nil) then
  begin
    GetErrors().Add(esError, CVK_ERR_BUFFER,
      'Failed to create staging buffer for device-local upload');
    if LStaging.Buffer <> 0 then
      DestroyBuffer(LStaging);
    Exit;
  end;

  try
    LOffset := 0;
    while LOffset < ASize do
    begin
      LChunk := ASize - LOffset;
      if LChunk > CStagingChunkSize then
        LChunk := CStagingChunkSize;

      // Host write into staging, then GPU copy staging -> destination chunk
      LSrcPtr := Pointer(UIntPtr(AData) + UIntPtr(LOffset));
      Move(LSrcPtr^, LStaging.MappedPtr^, LChunk);

      LRegion.srcOffset := 0;
      LRegion.dstOffset := ADstOffset + LOffset;
      LRegion.size := LChunk;

      BeginCommands();
      FvkCmdCopyBuffer(FCommandBuffer, LStaging.Buffer, ABuffer.Buffer,
        1, @LRegion);
      EndCommands();
      SubmitAndWait();

      LOffset := LOffset + LChunk;
    end;
  finally
    DestroyBuffer(LStaging);
  end;
end;

procedure TVulkanDevice.DownloadFromDeviceBuffer(const ABuffer: TVulkanBuffer;
  const ADst: Pointer; const ASize: UInt64;
  const ASrcOffset: UInt64);
var
  LStaging: TVulkanBuffer;
  LRegion: TVkBufferCopy;
begin
  // Host-visible source: direct mapped read
  if ABuffer.MappedPtr <> nil then
  begin
    Move(Pointer(UIntPtr(ABuffer.MappedPtr) + UIntPtr(ASrcOffset))^, ADst^, ASize);
    Exit;
  end;

  // Device-local source: GPU copy into a staging buffer, then mapped read
  LStaging := CreateBuffer(ASize, True);
  if (LStaging.Buffer = 0) or (LStaging.MappedPtr = nil) then
  begin
    GetErrors().Add(esError, CVK_ERR_BUFFER,
      'Failed to create staging buffer for device-local download');
    if LStaging.Buffer <> 0 then
      DestroyBuffer(LStaging);
    Exit;
  end;

  try
    LRegion.srcOffset := ASrcOffset;
    LRegion.dstOffset := 0;
    LRegion.size := ASize;

    BeginCommands();
    FvkCmdCopyBuffer(FCommandBuffer, ABuffer.Buffer, LStaging.Buffer,
      1, @LRegion);
    EndCommands();
    SubmitAndWait();

    Move(LStaging.MappedPtr^, ADst^, ASize);
  finally
    DestroyBuffer(LStaging);
  end;
end;

procedure TVulkanDevice.BeginCommands();
var
  LBeginInfo: TVkCommandBufferBeginInfo;
begin
  FvkResetCommandBuffer(FCommandBuffer, 0);

  FillChar(LBeginInfo, SizeOf(LBeginInfo), 0);
  LBeginInfo.sType := CVK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  LBeginInfo.flags := CVK_COMMAND_BUFFER_USAGE_ONE_TIME;

  FvkBeginCommandBuffer(FCommandBuffer, @LBeginInfo);
end;

procedure TVulkanDevice.EndCommands();
begin
  FvkEndCommandBuffer(FCommandBuffer);
end;

procedure TVulkanDevice.SubmitAndWait();
var
  LSubmitInfo: TVkSubmitInfo;
begin
  FillChar(LSubmitInfo, SizeOf(LSubmitInfo), 0);
  LSubmitInfo.sType := CVK_STRUCTURE_TYPE_SUBMIT_INFO;
  LSubmitInfo.commandBufferCount := 1;
  LSubmitInfo.pCommandBuffers := @FCommandBuffer;

  FvkResetFences(FDevice, 1, @FFence);
  FvkQueueSubmit(FComputeQueue, 1, @LSubmitInfo, FFence);
  FvkWaitForFences(FDevice, 1, @FFence, 1, CVK_MAX_TIMEOUT);
end;

procedure TVulkanDevice.RecordComputeBarrier();
var
  LBarrier: TVkMemoryBarrier;
begin
  // Compute -> compute execution + memory dependency: prior shader writes
  // become available and visible to subsequent shader reads and writes.
  FillChar(LBarrier, SizeOf(LBarrier), 0);
  LBarrier.sType := CVK_STRUCTURE_TYPE_MEMORY_BARRIER;
  LBarrier.srcAccessMask := CVK_ACCESS_SHADER_WRITE;
  LBarrier.dstAccessMask := CVK_ACCESS_SHADER_READ or CVK_ACCESS_SHADER_WRITE;

  FvkCmdPipelineBarrier(
    FCommandBuffer,
    CVK_PIPELINE_STAGE_COMPUTE_SHADER,  // srcStageMask
    CVK_PIPELINE_STAGE_COMPUTE_SHADER,  // dstStageMask
    0,                                  // dependencyFlags
    1, @LBarrier,                       // memory barriers
    0, nil,                             // buffer memory barriers
    0, nil                              // image memory barriers
  );
end;

function TVulkanDevice.CreateTimestampPool(const ACount: UInt32): Boolean;
var
  LCreateInfo: TVkQueryPoolCreateInfo;
  LResult: Int32;
begin
  Result := False;
  DestroyTimestampPool();

  if not Assigned(FvkCreateQueryPool) then
    Exit;

  FillChar(LCreateInfo, SizeOf(LCreateInfo), 0);
  LCreateInfo.sType := CVK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  LCreateInfo.queryType := CVK_QUERY_TYPE_TIMESTAMP;
  LCreateInfo.queryCount := ACount;

  LResult := FvkCreateQueryPool(FDevice, @LCreateInfo, nil, FTimestampPool);
  if LResult <> CVK_SUCCESS then
  begin
    FTimestampPool := 0;
    Exit;
  end;

  FTimestampCount := ACount;
  Result := True;
end;

procedure TVulkanDevice.DestroyTimestampPool();
begin
  if (FTimestampPool <> 0) and Assigned(FvkDestroyQueryPool) then
    FvkDestroyQueryPool(FDevice, FTimestampPool, nil);
  FTimestampPool := 0;
  FTimestampCount := 0;
end;

procedure TVulkanDevice.RecordResetTimestamps();
begin
  // Must be recorded before any timestamp writes in the same command buffer
  if FTimestampPool <> 0 then
    FvkCmdResetQueryPool(FCommandBuffer, FTimestampPool, 0, FTimestampCount);
end;

procedure TVulkanDevice.RecordTimestamp(const AIndex: UInt32);
begin
  // COMPUTE_SHADER stage: the timestamp is written once all previously
  // recorded compute work has completed that stage
  if (FTimestampPool <> 0) and (AIndex < FTimestampCount) then
    FvkCmdWriteTimestamp(FCommandBuffer, CVK_PIPELINE_STAGE_COMPUTE_SHADER,
      FTimestampPool, AIndex);
end;

function TVulkanDevice.FetchTimestamps(out AValues: TArray<UInt64>): Boolean;
var
  LResult: Int32;
begin
  Result := False;
  AValues := nil;

  if FTimestampPool = 0 then
    Exit;

  SetLength(AValues, FTimestampCount);
  LResult := FvkGetQueryPoolResults(FDevice, FTimestampPool, 0,
    FTimestampCount, UInt64(FTimestampCount) * SizeOf(UInt64), @AValues[0],
    SizeOf(UInt64), CVK_QUERY_RESULT_64 or CVK_QUERY_RESULT_WAIT);

  Result := LResult = CVK_SUCCESS;
end;

end.
