{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information

 -------------------------------------------------------------------------------

  Gemma4.Shaders - SPIR-V shader loading and compute pipeline management

  Loads pre-compiled SPIR-V compute shaders, creates Vulkan compute
  pipelines with descriptor set layouts, and manages descriptor pools
  and sets for binding storage buffers to shaders.

  Each compute kernel (GEMM, RMSNorm, softmax, etc.) gets its own
  pipeline. Descriptor sets bind the input/output GPU buffers.

  Key types:
  - TComputePipeline: Holds a single compute pipeline with its
    layout, descriptor set layout, and shader module.
  - TShaderManager: Creates and caches compute pipelines, manages
    descriptor pool and set allocation.

  Dependencies: StdApp.Base, Gemma4.Types, Gemma4.Vulkan
===============================================================================}

unit Gemma4.Shaders;

{$I StdApp.Defines.inc}

interface

uses
  WinApi.Windows,
  System.SysUtils,
  System.Generics.Collections,
  StdApp.Base,
  Gemma4.Types,
  Gemma4.Vulkan;

const
  CSH_ERR_SHADER = 'SH01';
  CSH_ERR_PIPELINE = 'SH02';
  CSH_ERR_DESCRIPTOR = 'SH03';

  // Vulkan descriptor type constants
  CVK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7;

  // Vulkan shader stage flags
  CVK_SHADER_STAGE_COMPUTE = $00000020;

  // Vulkan structure types for shader/pipeline/descriptor
  CVK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16;
  CVK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18;
  CVK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29;
  CVK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30;
  CVK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32;
  CVK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33;
  CVK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34;
  CVK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35;

  // Max bindings per shader (storage buffers)
  CMaxBindings = 8;

  // Max descriptor sets from the pool
  CMaxDescriptorSets = 4096;

type
  // Vulkan structures for shader/pipeline creation
  // Validated against vulkan_core.h from Vulkan SDK 1.4.350.0
  // NO packed - using natural alignment to match C

  { TVkShaderModuleCreateInfo }
  // C size: 40 bytes
  TVkShaderModuleCreateInfo = record
    sType: UInt32;             // offset 0
    pNext: Pointer;            // offset 8
    flags: UInt32;             // offset 16
    codeSize: UInt64;          // offset 24 (size_t in C, 8 bytes on 64-bit)
    pCode: Pointer;            // offset 32
  end;

  { TVkDescriptorSetLayoutBinding }
  // C size: 24 bytes
  TVkDescriptorSetLayoutBinding = record
    binding: UInt32;           // offset 0
    descriptorType: UInt32;    // offset 4
    descriptorCount: UInt32;   // offset 8
    stageFlags: UInt32;        // offset 12
    pImmutableSamplers: Pointer; // offset 16
  end;

  { TVkDescriptorSetLayoutCreateInfo }
  // C size: 32 bytes
  TVkDescriptorSetLayoutCreateInfo = record
    sType: UInt32;             // offset 0
    pNext: Pointer;            // offset 8
    flags: UInt32;             // offset 16
    bindingCount: UInt32;      // offset 20
    pBindings: Pointer;        // offset 24
  end;

  { TVkPushConstantRange }
  // C size: 12 bytes (all UInt32, no pointers)
  TVkPushConstantRange = record
    stageFlags: UInt32;        // offset 0
    offset: UInt32;            // offset 4
    size: UInt32;              // offset 8
  end;

  { TVkPipelineLayoutCreateInfo }
  // C size: 48 bytes
  TVkPipelineLayoutCreateInfo = record
    sType: UInt32;             // offset 0
    pNext: Pointer;            // offset 8
    flags: UInt32;             // offset 16
    setLayoutCount: UInt32;    // offset 20
    pSetLayouts: Pointer;      // offset 24
    pushConstantRangeCount: UInt32; // offset 32
    pPushConstantRanges: Pointer; // offset 40
  end;

  { TVkPipelineShaderStageCreateInfo }
  // C size: 48 bytes
  TVkPipelineShaderStageCreateInfo = record
    sType: UInt32;             // offset 0
    pNext: Pointer;            // offset 8
    flags: UInt32;             // offset 16
    stage: UInt32;             // offset 20
    module: TVkShaderModule;   // offset 24 (uint64)
    pName: PAnsiChar;          // offset 32
    pSpecializationInfo: Pointer; // offset 40
  end;

  { TVkComputePipelineCreateInfo }
  // C size: 96 bytes
  TVkComputePipelineCreateInfo = record
    sType: UInt32;             // offset 0
    pNext: Pointer;            // offset 8
    flags: UInt32;             // offset 16
    stage: TVkPipelineShaderStageCreateInfo; // offset 24, 48 bytes
    layout: TVkPipelineLayout; // offset 72 (uint64)
    basePipelineHandle: TVkPipeline; // offset 80 (uint64)
    basePipelineIndex: Int32;  // offset 88
  end;

  { TVkDescriptorPoolSize }
  // C size: 8 bytes (all UInt32, no pointers)
  TVkDescriptorPoolSize = record
    descriptorType: UInt32;    // offset 0
    descriptorCount: UInt32;   // offset 4
  end;

  { TVkDescriptorPoolCreateInfo }
  // C size: 40 bytes
  TVkDescriptorPoolCreateInfo = record
    sType: UInt32;             // offset 0
    pNext: Pointer;            // offset 8
    flags: UInt32;             // offset 16
    maxSets: UInt32;           // offset 20
    poolSizeCount: UInt32;     // offset 24
    pPoolSizes: Pointer;       // offset 32
  end;

  { TVkDescriptorSetAllocateInfo }
  // C size: 40 bytes
  TVkDescriptorSetAllocateInfo = record
    sType: UInt32;             // offset 0
    pNext: Pointer;            // offset 8
    descriptorPool: TVkDescriptorPool; // offset 16 (uint64)
    descriptorSetCount: UInt32; // offset 24
    pSetLayouts: Pointer;      // offset 32
  end;

  { TVkDescriptorBufferInfo }
  // C size: 24 bytes (all uint64, no alignment issues)
  TVkDescriptorBufferInfo = record
    buffer: TVkBuffer;         // offset 0 (uint64)
    offset: UInt64;            // offset 8
    range: UInt64;             // offset 16
  end;

  { TVkWriteDescriptorSet }
  // C size: 64 bytes
  TVkWriteDescriptorSet = record
    sType: UInt32;             // offset 0
    pNext: Pointer;            // offset 8
    dstSet: TVkDescriptorSet;  // offset 16 (uint64)
    dstBinding: UInt32;        // offset 24
    dstArrayElement: UInt32;   // offset 28
    descriptorCount: UInt32;   // offset 32
    descriptorType: UInt32;    // offset 36
    pImageInfo: Pointer;       // offset 40
    pBufferInfo: Pointer;      // offset 48
    pTexelBufferView: Pointer; // offset 56
  end;


  { TComputePipeline }
  // Holds all Vulkan objects for a single compute shader
  TComputePipeline = record
    ShaderModule: TVkShaderModule;
    DescriptorSetLayout: TVkDescriptorSetLayout;
    PipelineLayout: TVkPipelineLayout;
    Pipeline: TVkPipeline;
    BindingCount: Integer;
    PushConstantSize: UInt32;
    IsValid: Boolean;
  end;

  { TShaderManager }
  // Creates and manages compute pipelines and descriptor sets
  TShaderManager = class(TBaseObject)
  private
    FDevice: TVulkanDevice;
    FDescriptorPool: TVkDescriptorPool;
    FPipelines: TDictionary<string, TComputePipeline>;
    FIsInitialized: Boolean;

    function DoCreateDescriptorPool(): Boolean;
  public
    constructor Create(); override;
    destructor Destroy(); override;

    // Initialize with a Vulkan device (does not take ownership)
    function Init(const ADevice: TVulkanDevice): Boolean;
    procedure Shutdown();
    function IsInitialized(): Boolean;

    // Create a compute pipeline from SPIR-V bytecode
    function CreatePipeline(
      const APipelineName: string;
      const ASpirvCode: Pointer;
      const ASpirvSize: UInt64;
      const ABindingCount: Integer;
      const APushConstantSize: UInt32 = 0
    ): Boolean;

    // Get a previously created pipeline
    function GetPipeline(const APipelineName: string;
      out APipeline: TComputePipeline): Boolean;

    // Allocate a descriptor set for a pipeline
    function AllocateDescriptorSet(
      const APipeline: TComputePipeline
    ): TVkDescriptorSet;

    // Bind a storage buffer to a descriptor set at a given binding
    procedure BindBuffer(
      const ADescriptorSet: TVkDescriptorSet;
      const ABinding: UInt32;
      const ABuffer: TVulkanBuffer
    );

    // Bind a sub-range of a storage buffer to a descriptor set
    procedure BindBufferRange(
      const ADescriptorSet: TVkDescriptorSet;
      const ABinding: UInt32;
      const ABuffer: TVulkanBuffer;
      const AOffset: UInt64;
      const ARange: UInt64
    );

    // Reset descriptor pool, returning all allocated sets for reuse
    procedure ResetDescriptorPool();

    // Record commands: bind pipeline, descriptor set, push constants, dispatch
    procedure RecordDispatch(
      const APipelineName: string;
      const ADescriptorSet: TVkDescriptorSet;
      const AGroupsX: UInt32;
      const AGroupsY: UInt32;
      const AGroupsZ: UInt32;
      const APushData: Pointer = nil;
      const APushSize: UInt32 = 0
    );

    // Destroy a specific pipeline
    procedure DestroyPipeline(const APipelineName: string);
  end;

implementation

{ TShaderManager }

constructor TShaderManager.Create();
begin
  inherited Create();
  FDevice := nil;
  FDescriptorPool := 0;
  FPipelines := TDictionary<string, TComputePipeline>.Create();
  FIsInitialized := False;
end;

destructor TShaderManager.Destroy();
begin
  Shutdown();
  FPipelines.Free();
  inherited;
end;

function TShaderManager.Init(const ADevice: TVulkanDevice): Boolean;
begin
  Result := False;
  if (ADevice = nil) or (not ADevice.IsInitialized()) then
  begin
    GetErrors().Add(esError, CSH_ERR_SHADER, 'Vulkan device not initialized');
    Exit;
  end;

  FDevice := ADevice;

  if not DoCreateDescriptorPool() then
    Exit;

  FIsInitialized := True;
  Result := True;
end;

procedure TShaderManager.Shutdown();
var
  LPair: TPair<string, TComputePipeline>;
  LPipeline: TComputePipeline;
begin
  if not FIsInitialized then Exit;

  for LPair in FPipelines do
  begin
    LPipeline := LPair.Value;
    if LPipeline.Pipeline <> 0 then
      FDevice.vkDestroyPipeline(FDevice.Device, LPipeline.Pipeline, nil);
    if LPipeline.PipelineLayout <> 0 then
      FDevice.vkDestroyPipelineLayout(FDevice.Device, LPipeline.PipelineLayout, nil);
    if LPipeline.DescriptorSetLayout <> 0 then
      FDevice.vkDestroyDescriptorSetLayout(FDevice.Device, LPipeline.DescriptorSetLayout, nil);
    if LPipeline.ShaderModule <> 0 then
      FDevice.vkDestroyShaderModule(FDevice.Device, LPipeline.ShaderModule, nil);
  end;
  FPipelines.Clear();

  if FDescriptorPool <> 0 then
    FDevice.vkDestroyDescriptorPool(FDevice.Device, FDescriptorPool, nil);

  FDescriptorPool := 0;
  FDevice := nil;
  FIsInitialized := False;
end;

function TShaderManager.IsInitialized(): Boolean;
begin
  Result := FIsInitialized;
end;

function TShaderManager.DoCreateDescriptorPool(): Boolean;
var
  LPoolSize: TVkDescriptorPoolSize;
  LPoolInfo: TVkDescriptorPoolCreateInfo;
  LResult: Int32;
begin
  Result := False;

  LPoolSize.descriptorType := CVK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  LPoolSize.descriptorCount := CMaxDescriptorSets * CMaxBindings;

  FillChar(LPoolInfo, SizeOf(LPoolInfo), 0);
  LPoolInfo.sType := CVK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  LPoolInfo.maxSets := CMaxDescriptorSets;
  LPoolInfo.poolSizeCount := 1;
  LPoolInfo.pPoolSizes := @LPoolSize;

  LResult := FDevice.vkCreateDescriptorPool(FDevice.Device, @LPoolInfo, nil, FDescriptorPool);
  if LResult <> CVK_SUCCESS then
  begin
    GetErrors().Add(esError, CSH_ERR_DESCRIPTOR, 'vkCreateDescriptorPool failed: %d', [LResult]);
    Exit;
  end;

  Result := True;
end;

function TShaderManager.CreatePipeline(
  const APipelineName: string;
  const ASpirvCode: Pointer;
  const ASpirvSize: UInt64;
  const ABindingCount: Integer;
  const APushConstantSize: UInt32): Boolean;
var
  LPipeline: TComputePipeline;
  LShaderInfo: TVkShaderModuleCreateInfo;
  LBindings: array[0..CMaxBindings - 1] of TVkDescriptorSetLayoutBinding;
  LLayoutInfo: TVkDescriptorSetLayoutCreateInfo;
  LPushRange: TVkPushConstantRange;
  LPipelineLayoutInfo: TVkPipelineLayoutCreateInfo;
  LStageInfo: TVkPipelineShaderStageCreateInfo;
  LComputeInfo: TVkComputePipelineCreateInfo;
  LI: Integer;
  LResult: Int32;
begin
  Result := False;
  LPipeline := Default(TComputePipeline);
  LPipeline.BindingCount := ABindingCount;
  LPipeline.PushConstantSize := APushConstantSize;

  // Create shader module
  FillChar(LShaderInfo, SizeOf(LShaderInfo), 0);
  LShaderInfo.sType := CVK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  LShaderInfo.codeSize := ASpirvSize;
  LShaderInfo.pCode := ASpirvCode;

  LResult := FDevice.vkCreateShaderModule(FDevice.Device, @LShaderInfo, nil, LPipeline.ShaderModule);
  if LResult <> CVK_SUCCESS then
  begin
    GetErrors().Add(esError, CSH_ERR_SHADER, 'vkCreateShaderModule failed for "%s": %d', [APipelineName, LResult]);
    Exit;
  end;

  // Create descriptor set layout with ABindingCount storage buffer bindings
  for LI := 0 to ABindingCount - 1 do
  begin
    FillChar(LBindings[LI], SizeOf(TVkDescriptorSetLayoutBinding), 0);
    LBindings[LI].binding := UInt32(LI);
    LBindings[LI].descriptorType := CVK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    LBindings[LI].descriptorCount := 1;
    LBindings[LI].stageFlags := CVK_SHADER_STAGE_COMPUTE;
  end;

  FillChar(LLayoutInfo, SizeOf(LLayoutInfo), 0);
  LLayoutInfo.sType := CVK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  LLayoutInfo.bindingCount := UInt32(ABindingCount);
  LLayoutInfo.pBindings := @LBindings[0];

  LResult := FDevice.vkCreateDescriptorSetLayout(FDevice.Device, @LLayoutInfo, nil, LPipeline.DescriptorSetLayout);
  if LResult <> CVK_SUCCESS then
  begin
    GetErrors().Add(esError, CSH_ERR_PIPELINE, 'vkCreateDescriptorSetLayout failed for "%s": %d', [APipelineName, LResult]);
    FDevice.vkDestroyShaderModule(FDevice.Device, LPipeline.ShaderModule, nil);
    Exit;
  end;

  // Create pipeline layout (with optional push constants)
  FillChar(LPipelineLayoutInfo, SizeOf(LPipelineLayoutInfo), 0);
  LPipelineLayoutInfo.sType := CVK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  LPipelineLayoutInfo.setLayoutCount := 1;
  LPipelineLayoutInfo.pSetLayouts := @LPipeline.DescriptorSetLayout;

  if APushConstantSize > 0 then
  begin
    FillChar(LPushRange, SizeOf(LPushRange), 0);
    LPushRange.stageFlags := CVK_SHADER_STAGE_COMPUTE;
    LPushRange.offset := 0;
    LPushRange.size := APushConstantSize;
    LPipelineLayoutInfo.pushConstantRangeCount := 1;
    LPipelineLayoutInfo.pPushConstantRanges := @LPushRange;
  end;

  LResult := FDevice.vkCreatePipelineLayout(FDevice.Device, @LPipelineLayoutInfo, nil, LPipeline.PipelineLayout);
  if LResult <> CVK_SUCCESS then
  begin
    GetErrors().Add(esError, CSH_ERR_PIPELINE, 'vkCreatePipelineLayout failed for "%s": %d', [APipelineName, LResult]);
    FDevice.vkDestroyDescriptorSetLayout(FDevice.Device, LPipeline.DescriptorSetLayout, nil);
    FDevice.vkDestroyShaderModule(FDevice.Device, LPipeline.ShaderModule, nil);
    Exit;
  end;

  // Create compute pipeline
  FillChar(LStageInfo, SizeOf(LStageInfo), 0);
  LStageInfo.sType := CVK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  LStageInfo.stage := CVK_SHADER_STAGE_COMPUTE;
  LStageInfo.module := LPipeline.ShaderModule;
  LStageInfo.pName := 'main';

  FillChar(LComputeInfo, SizeOf(LComputeInfo), 0);
  LComputeInfo.sType := CVK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  LComputeInfo.stage := LStageInfo;
  LComputeInfo.layout := LPipeline.PipelineLayout;

  LResult := FDevice.vkCreateComputePipelines(FDevice.Device, 0, 1, @LComputeInfo, nil, @LPipeline.Pipeline);
  if LResult <> CVK_SUCCESS then
  begin
    GetErrors().Add(esError, CSH_ERR_PIPELINE, 'vkCreateComputePipelines failed for "%s": %d', [APipelineName, LResult]);
    FDevice.vkDestroyPipelineLayout(FDevice.Device, LPipeline.PipelineLayout, nil);
    FDevice.vkDestroyDescriptorSetLayout(FDevice.Device, LPipeline.DescriptorSetLayout, nil);
    FDevice.vkDestroyShaderModule(FDevice.Device, LPipeline.ShaderModule, nil);
    Exit;
  end;

  LPipeline.IsValid := True;
  FPipelines.AddOrSetValue(APipelineName, LPipeline);
  Result := True;
end;

function TShaderManager.GetPipeline(const APipelineName: string;
  out APipeline: TComputePipeline): Boolean;
begin
  Result := FPipelines.TryGetValue(APipelineName, APipeline);
end;

function TShaderManager.AllocateDescriptorSet(
  const APipeline: TComputePipeline): TVkDescriptorSet;
var
  LAllocInfo: TVkDescriptorSetAllocateInfo;
  LResult: Int32;
begin
  Result := 0;

  FillChar(LAllocInfo, SizeOf(LAllocInfo), 0);
  LAllocInfo.sType := CVK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  LAllocInfo.descriptorPool := FDescriptorPool;
  LAllocInfo.descriptorSetCount := 1;
  LAllocInfo.pSetLayouts := @APipeline.DescriptorSetLayout;

  LResult := FDevice.vkAllocateDescriptorSets(FDevice.Device, @LAllocInfo, @Result);
  if LResult <> CVK_SUCCESS then
    GetErrors().Add(esError, CSH_ERR_DESCRIPTOR, 'vkAllocateDescriptorSets failed: %d', [LResult]);
end;

procedure TShaderManager.BindBuffer(
  const ADescriptorSet: TVkDescriptorSet;
  const ABinding: UInt32;
  const ABuffer: TVulkanBuffer);
var
  LBufferInfo: TVkDescriptorBufferInfo;
  LWrite: TVkWriteDescriptorSet;
begin
  LBufferInfo.buffer := ABuffer.Buffer;
  LBufferInfo.offset := 0;
  LBufferInfo.range := CVK_WHOLE_SIZE;

  FillChar(LWrite, SizeOf(LWrite), 0);
  LWrite.sType := CVK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  LWrite.dstSet := ADescriptorSet;
  LWrite.dstBinding := ABinding;
  LWrite.descriptorCount := 1;
  LWrite.descriptorType := CVK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  LWrite.pBufferInfo := @LBufferInfo;

  FDevice.vkUpdateDescriptorSets(FDevice.Device, 1, @LWrite, 0, nil);
end;

procedure TShaderManager.BindBufferRange(
  const ADescriptorSet: TVkDescriptorSet;
  const ABinding: UInt32;
  const ABuffer: TVulkanBuffer;
  const AOffset: UInt64;
  const ARange: UInt64);
var
  LBufferInfo: TVkDescriptorBufferInfo;
  LWrite: TVkWriteDescriptorSet;
begin
  LBufferInfo.buffer := ABuffer.Buffer;
  LBufferInfo.offset := AOffset;
  LBufferInfo.range := ARange;

  FillChar(LWrite, SizeOf(LWrite), 0);
  LWrite.sType := CVK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  LWrite.dstSet := ADescriptorSet;
  LWrite.dstBinding := ABinding;
  LWrite.descriptorCount := 1;
  LWrite.descriptorType := CVK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  LWrite.pBufferInfo := @LBufferInfo;

  FDevice.vkUpdateDescriptorSets(FDevice.Device, 1, @LWrite, 0, nil);
end;

procedure TShaderManager.ResetDescriptorPool();
begin
  if (FDescriptorPool <> 0) and Assigned(FDevice.vkResetDescriptorPool) then
    FDevice.vkResetDescriptorPool(FDevice.Device, FDescriptorPool, 0);
end;

procedure TShaderManager.RecordDispatch(
  const APipelineName: string;
  const ADescriptorSet: TVkDescriptorSet;
  const AGroupsX: UInt32;
  const AGroupsY: UInt32;
  const AGroupsZ: UInt32;
  const APushData: Pointer;
  const APushSize: UInt32);
var
  LPipeline: TComputePipeline;
begin
  if not FPipelines.TryGetValue(APipelineName, LPipeline) then
  begin
    GetErrors().Add(esError, CSH_ERR_PIPELINE, 'Pipeline not found: %s', [APipelineName]);
    Exit;
  end;

  FDevice.vkCmdBindPipeline(FDevice.CommandBuffer, CVK_PIPELINE_BIND_POINT_COMPUTE, LPipeline.Pipeline);
  FDevice.vkCmdBindDescriptorSets(FDevice.CommandBuffer, CVK_PIPELINE_BIND_POINT_COMPUTE,
    LPipeline.PipelineLayout, 0, 1, @ADescriptorSet, 0, nil);

  if (APushData <> nil) and (APushSize > 0) then
    FDevice.vkCmdPushConstants(FDevice.CommandBuffer, LPipeline.PipelineLayout,
      CVK_SHADER_STAGE_COMPUTE, 0, APushSize, APushData);

  FDevice.vkCmdDispatch(FDevice.CommandBuffer, AGroupsX, AGroupsY, AGroupsZ);
end;

procedure TShaderManager.DestroyPipeline(const APipelineName: string);
var
  LPipeline: TComputePipeline;
begin
  if not FPipelines.TryGetValue(APipelineName, LPipeline) then
    Exit;

  if LPipeline.Pipeline <> 0 then
    FDevice.vkDestroyPipeline(FDevice.Device, LPipeline.Pipeline, nil);
  if LPipeline.PipelineLayout <> 0 then
    FDevice.vkDestroyPipelineLayout(FDevice.Device, LPipeline.PipelineLayout, nil);
  if LPipeline.DescriptorSetLayout <> 0 then
    FDevice.vkDestroyDescriptorSetLayout(FDevice.Device, LPipeline.DescriptorSetLayout, nil);
  if LPipeline.ShaderModule <> 0 then
    FDevice.vkDestroyShaderModule(FDevice.Device, LPipeline.ShaderModule, nil);

  FPipelines.Remove(APipelineName);
end;

end.
