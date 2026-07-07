{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information

 -------------------------------------------------------------------------------

  Gemma4.Compute - Vulkan compute kernel dispatch

  Loads SPIR-V shaders from embedded Delphi resources (Gemma4.Shaders.res),
  creates compute pipelines via TShaderManager, and provides two API tiers
  for each GPU operation:

  1. RecordXxx methods: allocate a descriptor set, bind buffers, and record
     the dispatch into the CURRENT command buffer. NO submission, NO fence.
     Used by the GPU-resident forward pass, which records an entire token's
     work into ONE command buffer and submits once (see Gemma4.Model).
     RecordBarrier() records a compute->compute memory barrier between
     dependent dispatches.

  2. DispatchXxx methods: thin synchronous wrappers -- BeginCommands,
     RecordXxx, EndCommands, SubmitAndWait. Used by kernel validation
     tests (GPU-vs-CPU comparisons) and one-off operations.

  Shader resources are embedded via:
    $R Gemma4.Shaders.res
  and loaded at runtime using TResourceStream.

  Dependencies: StdApp.Base, Gemma4.Types, Gemma4.Vulkan, Gemma4.Shaders
===============================================================================}

unit Gemma4.Compute;

{$I StdApp.Defines.inc}

interface

uses
  WinApi.Windows,
  System.SysUtils,
  System.Classes,
  StdApp.Base,
  Gemma4.Types,
  Gemma4.Vulkan,
  Gemma4.Shaders;


const
  CCP_ERR_INIT = 'CP01';
  CCP_ERR_SHADER = 'CP02';
  CCP_ERR_DISPATCH = 'CP03';

  // Workgroup size of the element-wise shaders (mul/add/scale/gelu/copy)
  { CElementwiseWorkgroup }
  CElementwiseWorkgroup = 256;

  // Workgroup size of the per-invocation shaders (rope/attn_scores/attn_out)
  { CSmallWorkgroup }
  CSmallWorkgroup = 64;

type
  { TMatVecQ4PushConstants }
  // Push constants for the split-plane Q4 matvec shader
  TMatVecQ4PushConstants = packed record
    ncols: UInt32;
    nrows: UInt32;
    quantsVec4Ofs: UInt32; // bind residual of the quants plane / 16
    scalesWordOfs: UInt32; // bind residual of the scales plane / 4
  end;

  { TQuantizeQ8PushConstants }
  // Push constants for the Q8_1 activation quantize shader
  TQuantizeQ8PushConstants = packed record
    nblocks: UInt32;   // ncols / 32
  end;

  { TMatMatQ4Q8PushConstants }
  // Push constants for the Q4 x Q8 matrix-matrix (batched) shader
  TMatMatQ4Q8PushConstants = packed record
    ncols: UInt32;
    nrows: UInt32;
    quantsVec4Ofs: UInt32; // bind residual of the quants plane / 16
    scalesWordOfs: UInt32; // bind residual of the scales plane / 4
    batch: UInt32;         // number of activation rows (positions)
  end;

  { TRopeBatchPushConstants }
  // Push constants for the multi-position NEOX RoPE shader
  TRopeBatchPushConstants = packed record
    basePosition: UInt32;  // absolute position of row m = 0
    headDim: UInt32;
    rotAngles: UInt32;
    numHeads: UInt32;
    theta: Single;
    batch: UInt32;         // number of positions
  end;

  { TKvAppendPushConstants }
  // Push constants for the multi-entry KV ring append shader
  TKvAppendPushConstants = packed record
    kvStride: UInt32;      // floats per cache entry
    batch: UInt32;         // entries to append
    basePosition: UInt32;  // absolute position of entry m = 0
    ringEntries: UInt32;   // ring modulus of the destination cache
  end;

  { TAttnBatchPushConstants }
  // Push constants shared by attn_scores_batch and attn_out_batch.
  // Query m sits at absolute position basePosition+m and attends key
  // positions [pStart(m) .. basePosition+m] where pStart(m) =
  // max(0, pos-windowSize+1) when windowSize > 0, else 0 (full attention).
  TAttnBatchPushConstants = packed record
    headDim: UInt32;
    numHeads: UInt32;
    numKvHeads: UInt32;
    kvStride: UInt32;
    ringEntries: UInt32;   // ring modulus of the KV cache
    basePosition: UInt32;  // absolute position of query m = 0
    batch: UInt32;         // number of query positions
    windowSize: UInt32;    // sliding window size; 0 = full attention
    rowPitch: UInt32;      // allocated score-row length (padded, fixed)
  end;

  { TGeGluRowsPushConstants }
  // Push constants for the strided-source GeGLU shader:
  // dst[r*rowSize+i] = gelu(dst[..]) * src[srcOffset + r*srcStride + i]
  TGeGluRowsPushConstants = packed record
    rowSize: UInt32;
    numRows: UInt32;
    srcOffset: UInt32;
    srcStride: UInt32;
  end;

  { TMatMatF32PushConstants }
  // Push constants for the F32 matrix-matrix (batched) shader
  TMatMatF32PushConstants = packed record
    ncols: UInt32;
    nrows: UInt32;
    weightVec4Ofs: UInt32; // bind residual / 16
    batch: UInt32;
  end;

  { TAttnBidirPushConstants }
  // Push constants shared by the bidirectional scores/out shaders.
  // windowSize = 0 -> full bidirectional; > 0 -> abs(m - t) < windowSize.
  TAttnBidirPushConstants = packed record
    headDim: UInt32;
    numHeads: UInt32;
    numKvHeads: UInt32;
    kvStride: UInt32;
    seqLen: UInt32;
    windowSize: UInt32;
    scale: Single;      // query_pre_attn_scalar^-0.5
    rowPitch: UInt32;   // = seqLen
  end;

  { TRope2DPushConstants }
  // Push constants for the vision 2D RoPE shader (per-patch x/y positions)
  TRope2DPushConstants = packed record
    headDim: UInt32;
    numHeads: UInt32;
    batch: UInt32;
    theta: Single;
  end;

  { TSiluPushConstants }
  // Push constants for the element-wise SiLU shader (audio conformer)
  TSiluPushConstants = packed record
    count: UInt32;
  end;

  { TGluPushConstants }
  // Push constants for the GLU shader (audio lconv gate)
  TGluPushConstants = packed record
    count: UInt32;
    halfDim: UInt32;
  end;

  { TDwConv1dPushConstants }
  // Push constants for the causal depthwise conv1d shader (audio lconv).
  // wOffset is a FLOAT-element offset of the [C,1,K] weight in the blob.
  TDwConv1dPushConstants = packed record
    seqLen: UInt32;
    channels: UInt32;
    kernel: UInt32;
    wOffset: UInt32;
  end;

  { TAudioAttnPushConstants }
  // Push constants for the fused audio conformer attention shader
  // (chunked local attention with Transformer-XL relative position bias)
  TAudioAttnPushConstants = packed record
    seqLen: UInt32;
    numHeads: UInt32;
    headDim: UInt32;
    chunkSize: UInt32;   // 12
    ctxSize: UInt32;     // 24
    pastH: UInt32;       // 12
    layerIdx: UInt32;    // row into the per-layer qscale vec buffer
    kScale: Single;      // ln(1+e)/ln2
    softcap: Single;     // 50.0
    invalidVal: Single;  // -1e9
  end;

  { TClampPushConstants }
  // Push constants for the element-wise clamp-copy shader
  TClampPushConstants = packed record
    count: UInt32;
    minVal: Single;
    maxVal: Single;
  end;

  { TMatVecF32PushConstants }
  // Push constants for the vec4 F32 matvec shader
  TMatVecF32PushConstants = packed record
    ncols: UInt32;
    nrows: UInt32;
    weightVec4Ofs: UInt32; // bind residual / 16
    reserved: UInt32;
  end;

  { TRmsNormPushConstants }
  // Push constants for the whole-vector RMS norm shader
  TRmsNormPushConstants = packed record
    size: UInt32;         // vector length
    weightOffset: UInt32; // float-element offset into the weight buffer
    eps: Single;
  end;

  { TRmsNormRowsPushConstants }
  // Push constants for the row-wise RMS norm shader (per-head / PLE norms)
  TRmsNormRowsPushConstants = packed record
    rowSize: UInt32;      // elements per row
    numRows: UInt32;      // number of independent rows
    eps: Single;
    hasWeight: UInt32;    // 0 = weight-free norm (V norm), 1 = weighted
    weightOffset: UInt32; // float-element offset into the weight buffer
  end;

  { TMulPushConstants }
  // Push constants for element-wise multiply: dst[i] *= src[srcOffset+i]
  TMulPushConstants = packed record
    count: UInt32;
    srcOffset: UInt32;
  end;

  { TAddPushConstants }
  // Push constants for element-wise add: dst[i] += src[i]
  TAddPushConstants = packed record
    count: UInt32;
  end;

  { TScalePushConstants }
  // Push constants for scalar scale: dst[i] *= scale
  TScalePushConstants = packed record
    count: UInt32;
    scale: Single;
  end;

  { TGeluPushConstants }
  // Push constants for in-place GeLU (pytorch tanh variant)
  TGeluPushConstants = packed record
    count: UInt32;
  end;

  { TGeGluPushConstants }
  // Push constants for fused GeGLU: dst[i] = gelu(dst[i]) * src[srcOffset+i]
  TGeGluPushConstants = packed record
    count: UInt32;
    srcOffset: UInt32;
  end;

  { TCopyPushConstants }
  // Push constants for strided copy: dst[dstOffset+i] = src[srcOffset+i]
  TCopyPushConstants = packed record
    count: UInt32;
    srcOffset: UInt32;
    dstOffset: UInt32;
  end;

  { TRopePushConstants }
  // Push constants for the multi-head NEOX RoPE shader
  TRopePushConstants = packed record
    position: UInt32;   // token position
    headDim: UInt32;    // head dimension (256 sliding / 512 full)
    rotAngles: UInt32;  // number of rotating angle pairs (128 / 64)
    numHeads: UInt32;   // number of heads in the buffer (8 Q / 2 KV)
    theta: Single;      // RoPE base frequency
  end;

  { TAttnPushConstants }
  // Push constants shared by attn_scores and attn_out shaders
  TAttnPushConstants = packed record
    headDim: UInt32;    // per-head dimension
    numHeads: UInt32;   // query heads (8)
    numKvHeads: UInt32; // KV heads (2)
    kvStride: UInt32;   // floats per cache entry (numKvHeads * headDim)
    seqLen: UInt32;     // number of valid cache entries to attend over
    ringSize: UInt32;   // ring modulus of the cache (entries)
    cachePos0: UInt32;  // ring index of the first enumerated entry
  end;

  { TSoftmaxRowsPushConstants }
  // Push constants for the row-wise softmax shader
  TSoftmaxRowsPushConstants = packed record
    rowSize: UInt32;
    numRows: UInt32;
  end;

  { TComputeKernels }
  // Manages all GPU compute pipelines and provides record/dispatch methods
  TComputeKernels = class(TBaseObject)
  private
    FDevice: TVulkanDevice;
    FShaders: TShaderManager;
    FIsInitialized: Boolean;

    // Q8_1 activation scratch for the integer-dot matvec path
    // (created only when the device supports DP4A)
    FQ8Scratch: TVulkanBuffer;

    function DoLoadShaderResource(const AResourceName: string;
      out AData: TBytes): Boolean;
    function DoCreatePipelineFromResource(
      const APipelineName: string;
      const AResourceName: string;
      const ABindingCount: Integer;
      const APushConstantSize: UInt32 = 0
    ): Boolean;
    function DoInitAllPipelines(): Boolean;

    // Q8_1 quantize + integer-dot matvec (DP4A path). Same signature and
    // call contract as RecordMatVecQ4; dispatched from it internally.
    procedure DoRecordMatVecQ4Q8(
      const AWeightBuf: TVulkanBuffer;
      const AQuantsByteOfs: UInt64;
      const AScalesByteOfs: UInt64;
      const ATotalByteSize: UInt64;
      const AInputBuf: TVulkanBuffer;
      const AOutputBuf: TVulkanBuffer;
      const ARows: UInt32;
      const ACols: UInt32
    );
  public
    constructor Create(); override;
    destructor Destroy(); override;

    // Initialize all compute pipelines (call after TVulkanDevice.Init)
    function Init(const ADevice: TVulkanDevice): Boolean;
    procedure Shutdown();
    function IsInitialized(): Boolean;

    // -- Record methods --
    // Record a single dispatch into the CURRENT command buffer.
    // Caller is responsible for BeginCommands/EndCommands/SubmitAndWait
    // and for barriers between dependent dispatches.

    // Compute->compute memory barrier between dependent dispatches
    procedure RecordBarrier();

    // Split-plane Q4 matvec. AQuantsByteOfs MUST be 16-byte aligned and
    // AScalesByteOfs 4-byte aligned (guaranteed by the upload-time layout).
    procedure RecordMatVecQ4(
      const AWeightBuf: TVulkanBuffer;
      const AQuantsByteOfs: UInt64;
      const AScalesByteOfs: UInt64;
      const ATotalByteSize: UInt64;
      const AInputBuf: TVulkanBuffer;
      const AOutputBuf: TVulkanBuffer;
      const ARows: UInt32;
      const ACols: UInt32
    );

    // Vec4 F32 matvec. AWeightByteOfs MUST be 16-byte aligned (guaranteed
    // by the upload-time layout).
    procedure RecordMatVecF32(
      const AWeightBuf: TVulkanBuffer;
      const AWeightByteOfs: UInt64;
      const AWeightByteSize: UInt64;
      const AInputBuf: TVulkanBuffer;
      const AOutputBuf: TVulkanBuffer;
      const ARows: UInt32;
      const ACols: UInt32
    );

    // Q4 weights x Q8 activations MATRIX-matrix (batched positions) via
    // integer dot products. Quantizes ABatch rows of AInputBuf internally
    // (same fold-in pattern as RecordMatVecQ4's DP4A branch). Output is
    // position-major: [ABatch x ARows]. DP4A devices only -- caller gates.
    procedure RecordMatMatQ4Q8(
      const AWeightBuf: TVulkanBuffer;
      const AQuantsByteOfs: UInt64;
      const AScalesByteOfs: UInt64;
      const ATotalByteSize: UInt64;
      const AInputBuf: TVulkanBuffer;
      const AOutputBuf: TVulkanBuffer;
      const ARows: UInt32;
      const ACols: UInt32;
      const ABatch: UInt32
    );

    // Multi-position NEOX RoPE in-place on [ABatch x ANumHeads x AHeadDim]
    procedure RecordRoPEBatch(
      const ADataBuf: TVulkanBuffer;
      const ABasePosition: UInt32;
      const ABatch: UInt32;
      const AHeadDim: UInt32;
      const ARotAngles: UInt32;
      const ANumHeads: UInt32;
      const ATheta: Single
    );

    // Vision 2D RoPE in-place on [ABatch x ANumHeads x AHeadDim]; per-patch
    // (x, y) int positions in APosBuf. Channels [0, headDim/2) rotate by x,
    // [headDim/2, headDim) by y (NEOX pairs inside each half).
    procedure RecordRoPE2D(
      const ADataBuf: TVulkanBuffer;
      const APosBuf: TVulkanBuffer;
      const ABatch: UInt32;
      const AHeadDim: UInt32;
      const ANumHeads: UInt32;
      const ATheta: Single
    );

    // Element-wise clamp-copy: dst[i] := Clamp(src[i], min, max). Pass the
    // same buffer twice for in-place clamping (QAT clipped linears).
    procedure RecordClamp(
      const ADstBuf: TVulkanBuffer;
      const ASrcBuf: TVulkanBuffer;
      const ACount: UInt32;
      const AMinVal: Single;
      const AMaxVal: Single
    );

    // Append ABatch staged KV entries to ring slots (ABasePosition+m) mod
    // ARingEntries in one dispatch
    procedure RecordKvAppend(
      const ACacheBuf: TVulkanBuffer;
      const AStageBuf: TVulkanBuffer;
      const AKvStride: UInt32;
      const ABatch: UInt32;
      const ABasePosition: UInt32;
      const ARingEntries: UInt32
    );

    // Batched causal attention scores; rows padded to APush.rowPitch with
    // -1e30 (softmax_rows turns padding into exact 0)
    procedure RecordAttnScoresBatch(
      const AQueryBuf: TVulkanBuffer;
      const AKCacheBuf: TVulkanBuffer;
      const AScoresBuf: TVulkanBuffer;
      const APush: TAttnBatchPushConstants
    );

    // Batched weighted value sum; iterates only each query's valid range
    // (padded slots may alias uninitialized cache memory)
    procedure RecordAttnOutBatch(
      const AProbsBuf: TVulkanBuffer;
      const AVCacheBuf: TVulkanBuffer;
      const AOutBuf: TVulkanBuffer;
      const APush: TAttnBatchPushConstants
    );

    // Strided-source GeGLU:
    // dst[r*ARowSize+i] = gelu(dst[..]) * src[ASrcOffset + r*ASrcStride + i]
    procedure RecordGeGLURows(
      const ADstBuf: TVulkanBuffer;
      const ASrcBuf: TVulkanBuffer;
      const ARowSize: UInt32;
      const ANumRows: UInt32;
      const ASrcOffset: UInt32;
      const ASrcStride: UInt32
    );

    // F32 weights x F32 activations matrix-matrix (batched positions).
    // AWeightByteOfs must be 16-byte aligned (256-aligned by the packer).
    procedure RecordMatMatF32(
      const AWeightBuf: TVulkanBuffer;
      const AWeightByteOfs: UInt64;
      const AWeightByteSize: UInt64;
      const AInputBuf: TVulkanBuffer;
      const AOutputBuf: TVulkanBuffer;
      const ARows: UInt32;
      const ACols: UInt32;
      const ABatch: UInt32
    );

    // Bidirectional attention scores over a fully-staged [seq x kv] K buffer
    procedure RecordAttnScoresBidir(
      const AQueryBuf: TVulkanBuffer;
      const AKBuf: TVulkanBuffer;
      const AScoresBuf: TVulkanBuffer;
      const APush: TAttnBidirPushConstants
    );

    // Bidirectional weighted value sum (iterates the full sequence)
    procedure RecordAttnOutBidir(
      const AProbsBuf: TVulkanBuffer;
      const AVBuf: TVulkanBuffer;
      const AOutBuf: TVulkanBuffer;
      const APush: TAttnBidirPushConstants
    );

    // Element-wise SiLU in-place (audio conformer activation)
    procedure RecordSiLU(
      const ABuf: TVulkanBuffer;
      const ACount: UInt32
    );

    // GLU over rows of [2*halfDim] -> [halfDim] (audio lconv gate)
    procedure RecordGLU(
      const ADstBuf: TVulkanBuffer;
      const ASrcBuf: TVulkanBuffer;
      const ACount: UInt32;
      const AHalfDim: UInt32
    );

    // Causal depthwise conv1d over time-major [t, c] data (audio lconv)
    procedure RecordDwConv1d(
      const ADstBuf: TVulkanBuffer;
      const ASrcBuf: TVulkanBuffer;
      const AWeightBuf: TVulkanBuffer;
      const APush: TDwConv1dPushConstants
    );

    // Fused audio conformer attention (chunked local + rel-pos bias)
    procedure RecordAudioAttn(
      const AQBuf: TVulkanBuffer;
      const AKBuf: TVulkanBuffer;
      const AVBuf: TVulkanBuffer;
      const ARelKBuf: TVulkanBuffer;
      const AQScaleBuf: TVulkanBuffer;
      const AOutBuf: TVulkanBuffer;
      const APush: TAudioAttnPushConstants
    );

    // RMS norm over one vector: dst[i] = (src[i] / rms) * weight[weightOffset+i]
    procedure RecordRmsNorm(
      const ASrcBuf: TVulkanBuffer;
      const AWeightBuf: TVulkanBuffer;
      const ADstBuf: TVulkanBuffer;
      const ASize: UInt32;
      const AEps: Single;
      const AWeightOffset: UInt32 = 0
    ); overload;

    // RMS norm with the weight read from a byte offset inside a large buffer
    // (e.g. the 4.3 GB weight blob). The bind offset is aligned down to 256
    // bytes to satisfy minStorageBufferOffsetAlignment; the residual is
    // passed to the shader as a float-element push offset.
    procedure RecordRmsNorm(
      const ASrcBuf: TVulkanBuffer;
      const AWeightBuf: TVulkanBuffer;
      const AWeightByteOffset: UInt64;
      const AWeightByteSize: UInt64;
      const ADstBuf: TVulkanBuffer;
      const ASize: UInt32;
      const AEps: Single
    ); overload;

    // Row-wise RMS norm, in-place. Same weight vector applies to every row.
    // AHasWeight=False gives the weight-free variant (V norm).
    procedure RecordRmsNormRows(
      const ADataBuf: TVulkanBuffer;
      const AWeightBuf: TVulkanBuffer;
      const ARowSize: UInt32;
      const ANumRows: UInt32;
      const AEps: Single;
      const AHasWeight: Boolean;
      const AWeightOffset: UInt32 = 0
    ); overload;

    // Row-wise RMS norm with the weight read from a byte offset inside a
    // large buffer (same aligned-bind scheme as the RecordRmsNorm overload).
    procedure RecordRmsNormRows(
      const ADataBuf: TVulkanBuffer;
      const AWeightBuf: TVulkanBuffer;
      const AWeightByteOffset: UInt64;
      const AWeightByteSize: UInt64;
      const ARowSize: UInt32;
      const ANumRows: UInt32;
      const AEps: Single
    ); overload;

    // Element-wise multiply in-place: dst[i] *= src[srcOffset+i]
    procedure RecordMul(
      const ADstBuf: TVulkanBuffer;
      const ASrcBuf: TVulkanBuffer;
      const ACount: UInt32;
      const ASrcOffset: UInt32 = 0
    );

    // Element-wise add in-place: dst[i] += src[i]
    procedure RecordAdd(
      const ADstBuf: TVulkanBuffer;
      const ASrcBuf: TVulkanBuffer;
      const ACount: UInt32
    );

    // Scale in-place: dst[i] *= scale
    procedure RecordScale(
      const ASrcDstBuf: TVulkanBuffer;
      const ACount: UInt32;
      const AScale: Single
    );

    // GeLU (pytorch tanh variant) in-place
    procedure RecordGeLU(
      const ASrcDstBuf: TVulkanBuffer;
      const ACount: UInt32
    );

    // Fused GeGLU in-place: dst[i] = gelu_pytorch_tanh(dst[i]) * src[srcOffset+i]
    procedure RecordGeGLU(
      const ADstBuf: TVulkanBuffer;
      const ASrcBuf: TVulkanBuffer;
      const ACount: UInt32;
      const ASrcOffset: UInt32 = 0
    );

    // Strided copy: dst[dstOffset+i] = src[srcOffset+i]
    procedure RecordCopy(
      const ADstBuf: TVulkanBuffer;
      const ASrcBuf: TVulkanBuffer;
      const ACount: UInt32;
      const ASrcOffset: UInt32 = 0;
      const ADstOffset: UInt32 = 0
    );

    // NEOX RoPE in-place on a multi-head buffer [numHeads * headDim]
    procedure RecordRoPE(
      const ADataBuf: TVulkanBuffer;
      const APosition: UInt32;
      const AHeadDim: UInt32;
      const ARotAngles: UInt32;
      const ANumHeads: UInt32;
      const ATheta: Single
    );

    // Attention scores: scores[h*seqLen+t] = dot(q[h], kcache[entry(t), kv(h)])
    procedure RecordAttnScores(
      const AQueryBuf: TVulkanBuffer;
      const AKCacheBuf: TVulkanBuffer;
      const AScoresBuf: TVulkanBuffer;
      const APush: TAttnPushConstants
    );

    // Row-wise softmax in-place (numerically stable)
    procedure RecordSoftmaxRows(
      const ADataBuf: TVulkanBuffer;
      const ARowSize: UInt32;
      const ANumRows: UInt32
    );

    // Attention output: out[h*headDim+d] = sum_t probs[h*seqLen+t] * vcache[...]
    procedure RecordAttnOut(
      const AProbsBuf: TVulkanBuffer;
      const AVCacheBuf: TVulkanBuffer;
      const AOutBuf: TVulkanBuffer;
      const APush: TAttnPushConstants
    );

    // -- Dispatch wrappers --
    // Each method: BeginCommands, RecordXxx, EndCommands, SubmitAndWait.
    // Synchronous. Used by kernel tests and one-off operations.

    procedure DispatchRmsNorm(
      const ASrcBuf: TVulkanBuffer;
      const AWeightBuf: TVulkanBuffer;
      const ADstBuf: TVulkanBuffer;
      const ASize: UInt32;
      const AEps: Single
    );

    procedure DispatchRmsNormRows(
      const ADataBuf: TVulkanBuffer;
      const AWeightBuf: TVulkanBuffer;
      const ARowSize: UInt32;
      const ANumRows: UInt32;
      const AEps: Single;
      const AHasWeight: Boolean;
      const AWeightOffset: UInt32 = 0
    );

    procedure DispatchMul(
      const ADstBuf: TVulkanBuffer;
      const ASrcBuf: TVulkanBuffer;
      const ACount: UInt32;
      const ASrcOffset: UInt32 = 0
    );

    procedure DispatchAdd(
      const ADstBuf: TVulkanBuffer;
      const ASrcBuf: TVulkanBuffer;
      const ACount: UInt32
    );

    procedure DispatchScale(
      const ASrcDstBuf: TVulkanBuffer;
      const ACount: UInt32;
      const AScale: Single
    );

    procedure DispatchGeLU(
      const ASrcDstBuf: TVulkanBuffer;
      const ACount: UInt32
    );

    procedure DispatchCopy(
      const ADstBuf: TVulkanBuffer;
      const ASrcBuf: TVulkanBuffer;
      const ACount: UInt32;
      const ASrcOffset: UInt32 = 0;
      const ADstOffset: UInt32 = 0
    );

    procedure DispatchRoPE(
      const ADataBuf: TVulkanBuffer;
      const APosition: UInt32;
      const AHeadDim: UInt32;
      const ARotAngles: UInt32;
      const ANumHeads: UInt32;
      const ATheta: Single
    );

    procedure DispatchAttnScores(
      const AQueryBuf: TVulkanBuffer;
      const AKCacheBuf: TVulkanBuffer;
      const AScoresBuf: TVulkanBuffer;
      const APush: TAttnPushConstants
    );

    procedure DispatchSoftmaxRows(
      const ADataBuf: TVulkanBuffer;
      const ARowSize: UInt32;
      const ANumRows: UInt32
    );

    procedure DispatchAttnOut(
      const AProbsBuf: TVulkanBuffer;
      const AVCacheBuf: TVulkanBuffer;
      const AOutBuf: TVulkanBuffer;
      const APush: TAttnPushConstants
    );

    // Expose for custom dispatch
    property Device: TVulkanDevice read FDevice;
    property Shaders: TShaderManager read FShaders;
  end;

implementation

{$R Gemma4.Shaders.res}

const
  // -- Q8_1 activation scratch layout (integer-dot matvec/matmat paths) --

  // Largest matvec/matmat input row: intermediate_size = 10240 floats
  CQ8_MAX_COLS = 10240;

  // int8 quants, 32 B per 32-float block, CPrefillBatch rows
  CQ8_QS_PLANE_BYTES = CPrefillBatch * CQ8_MAX_COLS;

  // 2621440: 256-aligned (2621440/256 = 10240)
  CQ8_DS_PLANE_OFFSET = CQ8_QS_PLANE_BYTES;

  // packHalf2x16(d, s) per block, CPrefillBatch rows
  CQ8_DS_PLANE_BYTES = CPrefillBatch * (CQ8_MAX_COLS div 32) * 4;

{ TComputeKernels }

constructor TComputeKernels.Create();
begin
  inherited Create();
  FDevice := nil;
  FShaders := TShaderManager.Create();
  FIsInitialized := False;
  FQ8Scratch := Default(TVulkanBuffer);
end;

destructor TComputeKernels.Destroy();
begin
  Shutdown();
  FShaders.Free();
  inherited;
end;

function TComputeKernels.Init(const ADevice: TVulkanDevice): Boolean;
begin
  Result := False;

  if (ADevice = nil) or (not ADevice.IsInitialized()) then
  begin
    GetErrors().Add(esError, CCP_ERR_INIT, 'Vulkan device not initialized');
    Exit;
  end;

  FDevice := ADevice;
  FShaders.SetErrors(GetErrors());

  if not FShaders.Init(ADevice) then
    Exit;

  if not DoInitAllPipelines() then
    Exit;

  // Q8_1 activation scratch, device-local (DP4A path only)
  if FDevice.HasIntegerDotProduct() then
  begin
    FQ8Scratch := FDevice.CreateBuffer(
      CQ8_QS_PLANE_BYTES + CQ8_DS_PLANE_BYTES, False);
    if FQ8Scratch.Buffer = 0 then
      Exit;
  end;

  FIsInitialized := True;
  Result := True;
end;

procedure TComputeKernels.Shutdown();
begin
  if FIsInitialized then
    FShaders.Shutdown();
  if (FDevice <> nil) and FDevice.IsInitialized() and
    (FQ8Scratch.Buffer <> 0) then
    FDevice.DestroyBuffer(FQ8Scratch);
  FDevice := nil;
  FIsInitialized := False;
end;

function TComputeKernels.IsInitialized(): Boolean;
begin
  Result := FIsInitialized;
end;

function TComputeKernels.DoLoadShaderResource(const AResourceName: string;
  out AData: TBytes): Boolean;
var
  LStream: TResourceStream;
begin
  Result := False;
  try
    LStream := TResourceStream.Create(HInstance, AResourceName, RT_RCDATA);
    try
      SetLength(AData, LStream.Size);
      if LStream.Size > 0 then
        LStream.ReadBuffer(AData[0], LStream.Size);
      Result := True;
    finally
      LStream.Free();
    end;
  except
    on E: Exception do
      GetErrors().Add(esError, CCP_ERR_SHADER,
        'Failed to load shader resource "%s": %s', [AResourceName, E.Message]);
  end;
end;

function TComputeKernels.DoCreatePipelineFromResource(
  const APipelineName: string;
  const AResourceName: string;
  const ABindingCount: Integer;
  const APushConstantSize: UInt32): Boolean;
var
  LData: TBytes;
begin
  Result := False;

  if not DoLoadShaderResource(AResourceName, LData) then
    Exit;

  Result := FShaders.CreatePipeline(
    APipelineName,
    @LData[0],
    Length(LData),
    ABindingCount,
    APushConstantSize
  );

  if not Result then
    GetErrors().Add(esError, CCP_ERR_SHADER,
      'Failed to create pipeline "%s"', [APipelineName]);
end;

function TComputeKernels.DoInitAllPipelines(): Boolean;
begin
  Result := False;

  // MatVec Q4_0 split-plane: bindings = 4 (quants, scales, input, output)
  if not DoCreatePipelineFromResource('matvec_q4', 'GEMMA4_MATVEC_Q4',
    4, SizeOf(TMatVecQ4PushConstants)) then Exit;

  // MatVec F32 vec4: bindings = 3
  if not DoCreatePipelineFromResource('matvec_f32', 'GEMMA4_MATVEC_F32',
    3, SizeOf(TMatVecF32PushConstants)) then Exit;

  // RMS Norm: bindings = 3 (src, weight, dst)
  if not DoCreatePipelineFromResource('rms_norm', 'GEMMA4_RMS_NORM',
    3, SizeOf(TRmsNormPushConstants)) then Exit;

  // RMS Norm rows: bindings = 2 (data in-place, weight)
  if not DoCreatePipelineFromResource('rms_norm_rows', 'GEMMA4_RMS_NORM_ROWS',
    2, SizeOf(TRmsNormRowsPushConstants)) then Exit;

  // Mul: bindings = 2 (dst, src)
  if not DoCreatePipelineFromResource('mul', 'GEMMA4_MUL',
    2, SizeOf(TMulPushConstants)) then Exit;

  // Add: bindings = 2 (dst, src)
  if not DoCreatePipelineFromResource('add', 'GEMMA4_ADD',
    2, SizeOf(TAddPushConstants)) then Exit;

  // Scale: bindings = 1 (dst)
  if not DoCreatePipelineFromResource('scale', 'GEMMA4_SCALE',
    1, SizeOf(TScalePushConstants)) then Exit;

  // GeLU: bindings = 1 (dst)
  if not DoCreatePipelineFromResource('gelu', 'GEMMA4_GELU',
    1, SizeOf(TGeluPushConstants)) then Exit;

  // GeGLU: bindings = 2 (dst, src)
  if not DoCreatePipelineFromResource('geglu', 'GEMMA4_GEGLU',
    2, SizeOf(TGeGluPushConstants)) then Exit;

  // Copy: bindings = 2 (dst, src)
  if not DoCreatePipelineFromResource('copy', 'GEMMA4_COPY',
    2, SizeOf(TCopyPushConstants)) then Exit;

  // RoPE: bindings = 1 (data)
  if not DoCreatePipelineFromResource('rope', 'GEMMA4_ROPE',
    1, SizeOf(TRopePushConstants)) then Exit;

  // Attention scores: bindings = 3 (q, kcache, scores)
  if not DoCreatePipelineFromResource('attn_scores', 'GEMMA4_ATTN_SCORES',
    3, SizeOf(TAttnPushConstants)) then Exit;

  // Softmax rows: bindings = 1 (data)
  if not DoCreatePipelineFromResource('softmax_rows', 'GEMMA4_SOFTMAX_ROWS',
    1, SizeOf(TSoftmaxRowsPushConstants)) then Exit;

  // Attention output: bindings = 3 (probs, vcache, out)
  if not DoCreatePipelineFromResource('attn_out', 'GEMMA4_ATTN_OUT',
    3, SizeOf(TAttnPushConstants)) then Exit;

  // Embeddings encoder kernels (F32, no DP4A requirement)
  if not DoCreatePipelineFromResource('matmat_f32', 'GEMMA4_MATMAT_F32',
    3, SizeOf(TMatMatF32PushConstants)) then Exit;
  if not DoCreatePipelineFromResource('attn_scores_bidir', 'GEMMA4_ATTN_SCORES_BIDIR',
    3, SizeOf(TAttnBidirPushConstants)) then Exit;
  if not DoCreatePipelineFromResource('attn_out_bidir', 'GEMMA4_ATTN_OUT_BIDIR',
    3, SizeOf(TAttnBidirPushConstants)) then Exit;

  // Vision encoder kernels (F32, no DP4A requirement)
  // 2D RoPE: bindings = 2 (data in-place, positions)
  if not DoCreatePipelineFromResource('rope2d', 'GEMMA4_ROPE2D',
    2, SizeOf(TRope2DPushConstants)) then Exit;

  // Clamp-copy: bindings = 2 (dst, src)
  if not DoCreatePipelineFromResource('clamp', 'GEMMA4_CLAMP',
    2, SizeOf(TClampPushConstants)) then Exit;

  // Audio conformer kernels
  // SiLU in-place: bindings = 1
  if not DoCreatePipelineFromResource('silu', 'GEMMA4_SILU',
    1, SizeOf(TSiluPushConstants)) then Exit;
  // GLU: bindings = 2 (dst, src)
  if not DoCreatePipelineFromResource('glu', 'GEMMA4_GLU',
    2, SizeOf(TGluPushConstants)) then Exit;
  // Causal depthwise conv1d: bindings = 3 (dst, src, weights)
  if not DoCreatePipelineFromResource('dwconv1d', 'GEMMA4_DWCONV1D',
    3, SizeOf(TDwConv1dPushConstants)) then Exit;
  // Fused chunked local attention: bindings = 6 (q, k, v, relk, qs, out)
  if not DoCreatePipelineFromResource('audio_attn', 'GEMMA4_AUDIO_ATTN',
    6, SizeOf(TAudioAttnPushConstants)) then Exit;

  // Integer-dot matvec path (only when the device supports DP4A)
  if FDevice.HasIntegerDotProduct() then
  begin
    // Q8_1 activation quantize: bindings = 3 (input f32, qs out, ds out)
    if not DoCreatePipelineFromResource('quantize_q8', 'GEMMA4_QUANTIZE_Q8',
      3, SizeOf(TQuantizeQ8PushConstants)) then Exit;

    // Q4 x Q8 matvec: bindings = 5 (quants, scales, q8 qs, q8 ds, output)
    if not DoCreatePipelineFromResource('matvec_q4q8', 'GEMMA4_MATVEC_Q4Q8',
      5, SizeOf(TMatVecQ4PushConstants)) then Exit;

    // Batched-prefill kernels (only reachable via the DP4A-gated batch path)
    if not DoCreatePipelineFromResource('matmat_q4q8', 'GEMMA4_MATMAT_Q4Q8',
      5, SizeOf(TMatMatQ4Q8PushConstants)) then Exit;
    if not DoCreatePipelineFromResource('rope_batch', 'GEMMA4_ROPE_BATCH',
      1, SizeOf(TRopeBatchPushConstants)) then Exit;
    if not DoCreatePipelineFromResource('kv_append', 'GEMMA4_KV_APPEND',
      2, SizeOf(TKvAppendPushConstants)) then Exit;
    if not DoCreatePipelineFromResource('attn_scores_batch', 'GEMMA4_ATTN_SCORES_BATCH',
      3, SizeOf(TAttnBatchPushConstants)) then Exit;
    if not DoCreatePipelineFromResource('attn_out_batch', 'GEMMA4_ATTN_OUT_BATCH',
      3, SizeOf(TAttnBatchPushConstants)) then Exit;
    if not DoCreatePipelineFromResource('geglu_rows', 'GEMMA4_GEGLU_ROWS',
      2, SizeOf(TGeGluRowsPushConstants)) then Exit;
  end;

  Result := True;
end;

// ---------------------------------------------------------------------------
// Record methods
// ---------------------------------------------------------------------------

procedure TComputeKernels.RecordBarrier();
begin
  FDevice.RecordComputeBarrier();
end;

procedure TComputeKernels.RecordMatVecQ4(
  const AWeightBuf: TVulkanBuffer;
  const AQuantsByteOfs: UInt64;
  const AScalesByteOfs: UInt64;
  const ATotalByteSize: UInt64;
  const AInputBuf: TVulkanBuffer;
  const AOutputBuf: TVulkanBuffer;
  const ARows: UInt32;
  const ACols: UInt32);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
  LPush: TMatVecQ4PushConstants;
  LQuantsAligned: UInt64;
  LQuantsResidual: UInt64;
  LScalesAligned: UInt64;
  LScalesResidual: UInt64;
  LQuantsBytes: UInt64;
  LScalesBytes: UInt64;
begin
  // DP4A path: quantize activations to Q8_1 and use integer dot products.
  // Callers see no difference; the switch is internal to this method.
  if FDevice.HasIntegerDotProduct() then
  begin
    DoRecordMatVecQ4Q8(AWeightBuf, AQuantsByteOfs, AScalesByteOfs,
      ATotalByteSize, AInputBuf, AOutputBuf, ARows, ACols);
    Exit;
  end;

  if not FShaders.GetPipeline('matvec_q4', LPipeline) then Exit;

  // Plane sizes: 16 + 2 bytes per block of the 18-byte total
  LQuantsBytes := (ATotalByteSize div 18) * 16;
  LScalesBytes := (ATotalByteSize div 18) * 2;

  // Bind offsets aligned down to 256 bytes (minStorageBufferOffsetAlignment);
  // residuals become element offsets in the push constants. AQuantsByteOfs is
  // 16-byte aligned, so the residual is divisible by 16; AScalesByteOfs is
  // 16-byte aligned as well, so its residual is divisible by 4.
  LQuantsAligned := (AQuantsByteOfs div 256) * 256;
  LQuantsResidual := AQuantsByteOfs - LQuantsAligned;
  LScalesAligned := (AScalesByteOfs div 256) * 256;
  LScalesResidual := AScalesByteOfs - LScalesAligned;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBufferRange(LDescSet, 0, AWeightBuf, LQuantsAligned,
    LQuantsBytes + LQuantsResidual);
  FShaders.BindBufferRange(LDescSet, 1, AWeightBuf, LScalesAligned,
    LScalesBytes + LScalesResidual);
  FShaders.BindBuffer(LDescSet, 2, AInputBuf);
  FShaders.BindBuffer(LDescSet, 3, AOutputBuf);

  LPush.ncols := ACols;
  LPush.nrows := ARows;
  LPush.quantsVec4Ofs := UInt32(LQuantsResidual div 16);
  LPush.scalesWordOfs := UInt32(LScalesResidual div 4);

  FShaders.RecordDispatch('matvec_q4', LDescSet, (ARows + 7) div 8, 1, 1,
    @LPush, SizeOf(LPush));
end;

procedure TComputeKernels.DoRecordMatVecQ4Q8(
  const AWeightBuf: TVulkanBuffer;
  const AQuantsByteOfs: UInt64;
  const AScalesByteOfs: UInt64;
  const ATotalByteSize: UInt64;
  const AInputBuf: TVulkanBuffer;
  const AOutputBuf: TVulkanBuffer;
  const ARows: UInt32;
  const ACols: UInt32);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
  LQuantPush: TQuantizeQ8PushConstants;
  LPush: TMatVecQ4PushConstants;
  LQuantsAligned: UInt64;
  LQuantsResidual: UInt64;
  LScalesAligned: UInt64;
  LScalesResidual: UInt64;
  LQuantsBytes: UInt64;
  LScalesBytes: UInt64;
  LNBlocks: UInt32;
begin
  // Guard: the Q8 scratch covers CQ8_MAX_COLS floats; ncols is always a
  // multiple of the 32-element block size in this model
  if (ACols > CQ8_MAX_COLS) or ((ACols mod 32) <> 0) then
  begin
    GetErrors().Add(esError, CCP_ERR_DISPATCH,
      'MatVecQ4Q8: unsupported column count %d', [ACols]);
    Exit;
  end;

  LNBlocks := ACols div 32;

  // Pass 1: quantize the f32 activation vector to Q8_1 into the scratch
  // buffer. Dispatched per matvec (inputs differ every call); tiny relative
  // to the matvec it feeds. 8 threads per block, 64 threads per workgroup.
  if not FShaders.GetPipeline('quantize_q8', LPipeline) then Exit;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBuffer(LDescSet, 0, AInputBuf);
  FShaders.BindBufferRange(LDescSet, 1, FQ8Scratch, 0, CQ8_QS_PLANE_BYTES);
  FShaders.BindBufferRange(LDescSet, 2, FQ8Scratch, CQ8_DS_PLANE_OFFSET,
    CQ8_DS_PLANE_BYTES);

  LQuantPush.nblocks := LNBlocks;

  FShaders.RecordDispatch('quantize_q8', LDescSet,
    (LNBlocks * 8 + 63) div 64, 1, 1, @LQuantPush, SizeOf(LQuantPush));

  // Quantize writes must be visible to the matvec reads
  RecordBarrier();

  // Pass 2: Q4 weights x Q8 activations via hardware integer dot products.
  // Weight binding scheme identical to the float Q4 path.
  if not FShaders.GetPipeline('matvec_q4q8', LPipeline) then Exit;

  // Plane sizes: 16 + 2 bytes per block of the 18-byte total
  LQuantsBytes := (ATotalByteSize div 18) * 16;
  LScalesBytes := (ATotalByteSize div 18) * 2;

  // Bind offsets aligned down to 256 bytes (minStorageBufferOffsetAlignment);
  // residuals become element offsets in the push constants. AQuantsByteOfs is
  // 16-byte aligned, so the residual is divisible by 16; AScalesByteOfs is
  // 16-byte aligned as well, so its residual is divisible by 4.
  LQuantsAligned := (AQuantsByteOfs div 256) * 256;
  LQuantsResidual := AQuantsByteOfs - LQuantsAligned;
  LScalesAligned := (AScalesByteOfs div 256) * 256;
  LScalesResidual := AScalesByteOfs - LScalesAligned;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBufferRange(LDescSet, 0, AWeightBuf, LQuantsAligned,
    LQuantsBytes + LQuantsResidual);
  FShaders.BindBufferRange(LDescSet, 1, AWeightBuf, LScalesAligned,
    LScalesBytes + LScalesResidual);
  FShaders.BindBufferRange(LDescSet, 2, FQ8Scratch, 0, CQ8_QS_PLANE_BYTES);
  FShaders.BindBufferRange(LDescSet, 3, FQ8Scratch, CQ8_DS_PLANE_OFFSET,
    CQ8_DS_PLANE_BYTES);
  FShaders.BindBuffer(LDescSet, 4, AOutputBuf);

  LPush.ncols := ACols;
  LPush.nrows := ARows;
  LPush.quantsVec4Ofs := UInt32(LQuantsResidual div 16);
  LPush.scalesWordOfs := UInt32(LScalesResidual div 4);

  FShaders.RecordDispatch('matvec_q4q8', LDescSet, (ARows + 7) div 8, 1, 1,
    @LPush, SizeOf(LPush));
end;

procedure TComputeKernels.RecordMatMatQ4Q8(
  const AWeightBuf: TVulkanBuffer;
  const AQuantsByteOfs: UInt64;
  const AScalesByteOfs: UInt64;
  const ATotalByteSize: UInt64;
  const AInputBuf: TVulkanBuffer;
  const AOutputBuf: TVulkanBuffer;
  const ARows: UInt32;
  const ACols: UInt32;
  const ABatch: UInt32);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
  LQuantPush: TQuantizeQ8PushConstants;
  LPush: TMatMatQ4Q8PushConstants;
  LQuantsAligned: UInt64;
  LQuantsResidual: UInt64;
  LScalesAligned: UInt64;
  LScalesResidual: UInt64;
  LQuantsBytes: UInt64;
  LScalesBytes: UInt64;
  LNBlocks: UInt32;
begin
  // Guards: scratch covers CPrefillBatch rows of CQ8_MAX_COLS floats
  if (ACols > CQ8_MAX_COLS) or ((ACols mod 32) <> 0) or
    (ABatch = 0) or (ABatch > CPrefillBatch) then
  begin
    GetErrors().Add(esError, CCP_ERR_DISPATCH,
      'MatMatQ4Q8: unsupported cols %d / batch %d', [ACols, ABatch]);
    Exit;
  end;

  // Pass 1: quantize ABatch contiguous activation rows to Q8_1. The shader
  // is row-agnostic: blocks never cross row boundaries (cols mod 32 = 0).
  LNBlocks := ABatch * (ACols div 32);

  if not FShaders.GetPipeline('quantize_q8', LPipeline) then Exit;
  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBuffer(LDescSet, 0, AInputBuf);
  FShaders.BindBufferRange(LDescSet, 1, FQ8Scratch, 0, CQ8_QS_PLANE_BYTES);
  FShaders.BindBufferRange(LDescSet, 2, FQ8Scratch, CQ8_DS_PLANE_OFFSET,
    CQ8_DS_PLANE_BYTES);
  LQuantPush.nblocks := LNBlocks;
  FShaders.RecordDispatch('quantize_q8', LDescSet,
    (LNBlocks * 8 + 63) div 64, 1, 1, @LQuantPush, SizeOf(LQuantPush));

  RecordBarrier();

  // Pass 2: matrix-matrix integer-dot kernel. Weight binding scheme is
  // identical to the matvec paths.
  if not FShaders.GetPipeline('matmat_q4q8', LPipeline) then Exit;

  LQuantsBytes := (ATotalByteSize div 18) * 16;
  LScalesBytes := (ATotalByteSize div 18) * 2;
  LQuantsAligned := (AQuantsByteOfs div 256) * 256;
  LQuantsResidual := AQuantsByteOfs - LQuantsAligned;
  LScalesAligned := (AScalesByteOfs div 256) * 256;
  LScalesResidual := AScalesByteOfs - LScalesAligned;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBufferRange(LDescSet, 0, AWeightBuf, LQuantsAligned,
    LQuantsBytes + LQuantsResidual);
  FShaders.BindBufferRange(LDescSet, 1, AWeightBuf, LScalesAligned,
    LScalesBytes + LScalesResidual);
  FShaders.BindBufferRange(LDescSet, 2, FQ8Scratch, 0, CQ8_QS_PLANE_BYTES);
  FShaders.BindBufferRange(LDescSet, 3, FQ8Scratch, CQ8_DS_PLANE_OFFSET,
    CQ8_DS_PLANE_BYTES);
  FShaders.BindBuffer(LDescSet, 4, AOutputBuf);

  LPush.ncols := ACols;
  LPush.nrows := ARows;
  LPush.quantsVec4Ofs := UInt32(LQuantsResidual div 16);
  LPush.scalesWordOfs := UInt32(LScalesResidual div 4);
  LPush.batch := ABatch;

  // 8 output rows per workgroup in x, 8 positions (M_TILE) per workgroup in y
  FShaders.RecordDispatch('matmat_q4q8', LDescSet,
    (ARows + 7) div 8, (ABatch + 7) div 8, 1, @LPush, SizeOf(LPush));
end;

procedure TComputeKernels.RecordRoPEBatch(
  const ADataBuf: TVulkanBuffer;
  const ABasePosition: UInt32;
  const ABatch: UInt32;
  const AHeadDim: UInt32;
  const ARotAngles: UInt32;
  const ANumHeads: UInt32;
  const ATheta: Single);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
  LPush: TRopeBatchPushConstants;
  LTotal: UInt32;
begin
  if not FShaders.GetPipeline('rope_batch', LPipeline) then Exit;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBuffer(LDescSet, 0, ADataBuf);

  LPush.basePosition := ABasePosition;
  LPush.headDim := AHeadDim;
  LPush.rotAngles := ARotAngles;
  LPush.numHeads := ANumHeads;
  LPush.theta := ATheta;
  LPush.batch := ABatch;

  // One invocation per (position, head, angle pair)
  LTotal := ABatch * ANumHeads * (AHeadDim div 2);
  FShaders.RecordDispatch('rope_batch', LDescSet, (LTotal + 63) div 64, 1, 1,
    @LPush, SizeOf(LPush));
end;

procedure TComputeKernels.RecordRoPE2D(
  const ADataBuf: TVulkanBuffer;
  const APosBuf: TVulkanBuffer;
  const ABatch: UInt32;
  const AHeadDim: UInt32;
  const ANumHeads: UInt32;
  const ATheta: Single);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
  LPush: TRope2DPushConstants;
  LTotal: UInt32;
begin
  if not FShaders.GetPipeline('rope2d', LPipeline) then Exit;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBuffer(LDescSet, 0, ADataBuf);
  FShaders.BindBuffer(LDescSet, 1, APosBuf);

  LPush.headDim := AHeadDim;
  LPush.numHeads := ANumHeads;
  LPush.batch := ABatch;
  LPush.theta := ATheta;

  // One invocation per (patch, head, pair); pairs per head = headDim/2
  LTotal := ABatch * ANumHeads * (AHeadDim div 2);
  FShaders.RecordDispatch('rope2d', LDescSet, (LTotal + 63) div 64, 1, 1,
    @LPush, SizeOf(LPush));
end;

procedure TComputeKernels.RecordClamp(
  const ADstBuf: TVulkanBuffer;
  const ASrcBuf: TVulkanBuffer;
  const ACount: UInt32;
  const AMinVal: Single;
  const AMaxVal: Single);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
  LPush: TClampPushConstants;
begin
  if not FShaders.GetPipeline('clamp', LPipeline) then Exit;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBuffer(LDescSet, 0, ADstBuf);
  FShaders.BindBuffer(LDescSet, 1, ASrcBuf);

  LPush.count := ACount;
  LPush.minVal := AMinVal;
  LPush.maxVal := AMaxVal;

  FShaders.RecordDispatch('clamp', LDescSet, (ACount + 255) div 256, 1, 1,
    @LPush, SizeOf(LPush));
end;

procedure TComputeKernels.RecordSiLU(
  const ABuf: TVulkanBuffer;
  const ACount: UInt32);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
  LPush: TSiluPushConstants;
begin
  if not FShaders.GetPipeline('silu', LPipeline) then Exit;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBuffer(LDescSet, 0, ABuf);

  LPush.count := ACount;

  FShaders.RecordDispatch('silu', LDescSet, (ACount + 255) div 256, 1, 1,
    @LPush, SizeOf(LPush));
end;

procedure TComputeKernels.RecordGLU(
  const ADstBuf: TVulkanBuffer;
  const ASrcBuf: TVulkanBuffer;
  const ACount: UInt32;
  const AHalfDim: UInt32);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
  LPush: TGluPushConstants;
begin
  if not FShaders.GetPipeline('glu', LPipeline) then Exit;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBuffer(LDescSet, 0, ADstBuf);
  FShaders.BindBuffer(LDescSet, 1, ASrcBuf);

  LPush.count := ACount;
  LPush.halfDim := AHalfDim;

  FShaders.RecordDispatch('glu', LDescSet, (ACount + 255) div 256, 1, 1,
    @LPush, SizeOf(LPush));
end;

procedure TComputeKernels.RecordDwConv1d(
  const ADstBuf: TVulkanBuffer;
  const ASrcBuf: TVulkanBuffer;
  const AWeightBuf: TVulkanBuffer;
  const APush: TDwConv1dPushConstants);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
  LTotal: UInt32;
begin
  if not FShaders.GetPipeline('dwconv1d', LPipeline) then Exit;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBuffer(LDescSet, 0, ADstBuf);
  FShaders.BindBuffer(LDescSet, 1, ASrcBuf);
  FShaders.BindBuffer(LDescSet, 2, AWeightBuf);

  LTotal := APush.seqLen * APush.channels;
  FShaders.RecordDispatch('dwconv1d', LDescSet, (LTotal + 255) div 256, 1, 1,
    @APush, SizeOf(APush));
end;

procedure TComputeKernels.RecordAudioAttn(
  const AQBuf: TVulkanBuffer;
  const AKBuf: TVulkanBuffer;
  const AVBuf: TVulkanBuffer;
  const ARelKBuf: TVulkanBuffer;
  const AQScaleBuf: TVulkanBuffer;
  const AOutBuf: TVulkanBuffer;
  const APush: TAudioAttnPushConstants);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
begin
  if not FShaders.GetPipeline('audio_attn', LPipeline) then Exit;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBuffer(LDescSet, 0, AQBuf);
  FShaders.BindBuffer(LDescSet, 1, AKBuf);
  FShaders.BindBuffer(LDescSet, 2, AVBuf);
  FShaders.BindBuffer(LDescSet, 3, ARelKBuf);
  FShaders.BindBuffer(LDescSet, 4, AQScaleBuf);
  FShaders.BindBuffer(LDescSet, 5, AOutBuf);

  // One workgroup per (query token, head)
  FShaders.RecordDispatch('audio_attn', LDescSet, APush.seqLen,
    APush.numHeads, 1, @APush, SizeOf(APush));
end;

procedure TComputeKernels.RecordKvAppend(
  const ACacheBuf: TVulkanBuffer;
  const AStageBuf: TVulkanBuffer;
  const AKvStride: UInt32;
  const ABatch: UInt32;
  const ABasePosition: UInt32;
  const ARingEntries: UInt32);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
  LPush: TKvAppendPushConstants;
  LTotal: UInt32;
begin
  if not FShaders.GetPipeline('kv_append', LPipeline) then Exit;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBuffer(LDescSet, 0, ACacheBuf);
  FShaders.BindBuffer(LDescSet, 1, AStageBuf);

  LPush.kvStride := AKvStride;
  LPush.batch := ABatch;
  LPush.basePosition := ABasePosition;
  LPush.ringEntries := ARingEntries;

  // One invocation per staged element
  LTotal := ABatch * AKvStride;
  FShaders.RecordDispatch('kv_append', LDescSet, (LTotal + 63) div 64, 1, 1,
    @LPush, SizeOf(LPush));
end;

procedure TComputeKernels.RecordAttnScoresBatch(
  const AQueryBuf: TVulkanBuffer;
  const AKCacheBuf: TVulkanBuffer;
  const AScoresBuf: TVulkanBuffer;
  const APush: TAttnBatchPushConstants);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
  LTotal: UInt32;
begin
  if not FShaders.GetPipeline('attn_scores_batch', LPipeline) then Exit;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBuffer(LDescSet, 0, AQueryBuf);
  FShaders.BindBuffer(LDescSet, 1, AKCacheBuf);
  FShaders.BindBuffer(LDescSet, 2, AScoresBuf);

  // One 32-lane warp per (position, head, padded row slot); 8 warps per group
  LTotal := APush.batch * APush.numHeads * APush.rowPitch;
  FShaders.RecordDispatch('attn_scores_batch', LDescSet,
    (LTotal + 7) div 8, 1, 1, @APush, SizeOf(APush));
end;

procedure TComputeKernels.RecordAttnOutBatch(
  const AProbsBuf: TVulkanBuffer;
  const AVCacheBuf: TVulkanBuffer;
  const AOutBuf: TVulkanBuffer;
  const APush: TAttnBatchPushConstants);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
  LTotal: UInt32;
begin
  if not FShaders.GetPipeline('attn_out_batch', LPipeline) then Exit;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBuffer(LDescSet, 0, AProbsBuf);
  FShaders.BindBuffer(LDescSet, 1, AVCacheBuf);
  FShaders.BindBuffer(LDescSet, 2, AOutBuf);

  // One workgroup per (position, head, 32-dim block); headDim is /32-exact
  LTotal := APush.batch * APush.numHeads * (APush.headDim div 32);
  FShaders.RecordDispatch('attn_out_batch', LDescSet,
    LTotal, 1, 1, @APush, SizeOf(APush));
end;

procedure TComputeKernels.RecordGeGLURows(
  const ADstBuf: TVulkanBuffer;
  const ASrcBuf: TVulkanBuffer;
  const ARowSize: UInt32;
  const ANumRows: UInt32;
  const ASrcOffset: UInt32;
  const ASrcStride: UInt32);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
  LPush: TGeGluRowsPushConstants;
  LTotal: UInt32;
begin
  if not FShaders.GetPipeline('geglu_rows', LPipeline) then Exit;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBuffer(LDescSet, 0, ADstBuf);
  FShaders.BindBuffer(LDescSet, 1, ASrcBuf);

  LPush.rowSize := ARowSize;
  LPush.numRows := ANumRows;
  LPush.srcOffset := ASrcOffset;
  LPush.srcStride := ASrcStride;

  LTotal := ANumRows * ARowSize;
  FShaders.RecordDispatch('geglu_rows', LDescSet, (LTotal + 255) div 256, 1, 1,
    @LPush, SizeOf(LPush));
end;

procedure TComputeKernels.RecordMatMatF32(
  const AWeightBuf: TVulkanBuffer;
  const AWeightByteOfs: UInt64;
  const AWeightByteSize: UInt64;
  const AInputBuf: TVulkanBuffer;
  const AOutputBuf: TVulkanBuffer;
  const ARows: UInt32;
  const ACols: UInt32;
  const ABatch: UInt32);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
  LPush: TMatMatF32PushConstants;
  LAlignedOffset: UInt64;
  LResidual: UInt64;
begin
  if not FShaders.GetPipeline('matmat_f32', LPipeline) then Exit;

  // Aligned-bind scheme (mirror of RecordMatVecF32): bind down to 256 B,
  // residual becomes a vec4-element push offset
  LAlignedOffset := (AWeightByteOfs div 256) * 256;
  LResidual := AWeightByteOfs - LAlignedOffset;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBufferRange(LDescSet, 0, AWeightBuf, LAlignedOffset,
    AWeightByteSize + LResidual);
  FShaders.BindBuffer(LDescSet, 1, AInputBuf);
  FShaders.BindBuffer(LDescSet, 2, AOutputBuf);

  LPush.ncols := ACols;
  LPush.nrows := ARows;
  LPush.weightVec4Ofs := UInt32(LResidual div 16);
  LPush.batch := ABatch;

  FShaders.RecordDispatch('matmat_f32', LDescSet,
    (ARows + 7) div 8, (ABatch + 7) div 8, 1, @LPush, SizeOf(LPush));
end;

procedure TComputeKernels.RecordAttnScoresBidir(
  const AQueryBuf: TVulkanBuffer;
  const AKBuf: TVulkanBuffer;
  const AScoresBuf: TVulkanBuffer;
  const APush: TAttnBidirPushConstants);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
  LTotal: UInt32;
begin
  if not FShaders.GetPipeline('attn_scores_bidir', LPipeline) then Exit;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBuffer(LDescSet, 0, AQueryBuf);
  FShaders.BindBuffer(LDescSet, 1, AKBuf);
  FShaders.BindBuffer(LDescSet, 2, AScoresBuf);

  // One 32-lane warp per (position, head, key slot); 8 warps per group
  LTotal := APush.seqLen * APush.numHeads * APush.rowPitch;
  FShaders.RecordDispatch('attn_scores_bidir', LDescSet,
    (LTotal + 7) div 8, 1, 1, @APush, SizeOf(APush));
end;

procedure TComputeKernels.RecordAttnOutBidir(
  const AProbsBuf: TVulkanBuffer;
  const AVBuf: TVulkanBuffer;
  const AOutBuf: TVulkanBuffer;
  const APush: TAttnBidirPushConstants);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
  LTotal: UInt32;
begin
  if not FShaders.GetPipeline('attn_out_bidir', LPipeline) then Exit;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBuffer(LDescSet, 0, AProbsBuf);
  FShaders.BindBuffer(LDescSet, 1, AVBuf);
  FShaders.BindBuffer(LDescSet, 2, AOutBuf);

  // One workgroup per (position, head, 32-dim block); headDim is /32-exact
  LTotal := APush.seqLen * APush.numHeads * (APush.headDim div 32);
  FShaders.RecordDispatch('attn_out_bidir', LDescSet,
    LTotal, 1, 1, @APush, SizeOf(APush));
end;

procedure TComputeKernels.RecordMatVecF32(
  const AWeightBuf: TVulkanBuffer;
  const AWeightByteOfs: UInt64;
  const AWeightByteSize: UInt64;
  const AInputBuf: TVulkanBuffer;
  const AOutputBuf: TVulkanBuffer;
  const ARows: UInt32;
  const ACols: UInt32);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
  LPush: TMatVecF32PushConstants;
  LAlignedOffset: UInt64;
  LResidual: UInt64;
begin
  if not FShaders.GetPipeline('matvec_f32', LPipeline) then Exit;

  // Aligned-bind scheme: bind offset down to 256 bytes, residual as a
  // vec4-element push offset (AWeightByteOfs is 16-byte aligned)
  LAlignedOffset := (AWeightByteOfs div 256) * 256;
  LResidual := AWeightByteOfs - LAlignedOffset;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBufferRange(LDescSet, 0, AWeightBuf, LAlignedOffset,
    AWeightByteSize + LResidual);
  FShaders.BindBuffer(LDescSet, 1, AInputBuf);
  FShaders.BindBuffer(LDescSet, 2, AOutputBuf);

  LPush.ncols := ACols;
  LPush.nrows := ARows;
  LPush.weightVec4Ofs := UInt32(LResidual div 16);
  LPush.reserved := 0;

  FShaders.RecordDispatch('matvec_f32', LDescSet, (ARows + 7) div 8, 1, 1,
    @LPush, SizeOf(LPush));
end;

procedure TComputeKernels.RecordRmsNorm(
  const ASrcBuf: TVulkanBuffer;
  const AWeightBuf: TVulkanBuffer;
  const ADstBuf: TVulkanBuffer;
  const ASize: UInt32;
  const AEps: Single;
  const AWeightOffset: UInt32);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
  LPush: TRmsNormPushConstants;
begin
  if not FShaders.GetPipeline('rms_norm', LPipeline) then Exit;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBuffer(LDescSet, 0, ASrcBuf);
  FShaders.BindBuffer(LDescSet, 1, AWeightBuf);
  FShaders.BindBuffer(LDescSet, 2, ADstBuf);

  LPush.size := ASize;
  LPush.weightOffset := AWeightOffset;
  LPush.eps := AEps;

  FShaders.RecordDispatch('rms_norm', LDescSet, 1, 1, 1,
    @LPush, SizeOf(LPush));
end;

procedure TComputeKernels.RecordRmsNorm(
  const ASrcBuf: TVulkanBuffer;
  const AWeightBuf: TVulkanBuffer;
  const AWeightByteOffset: UInt64;
  const AWeightByteSize: UInt64;
  const ADstBuf: TVulkanBuffer;
  const ASize: UInt32;
  const AEps: Single);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
  LPush: TRmsNormPushConstants;
  LAlignedOffset: UInt64;
  LResidual: UInt64;
begin
  if not FShaders.GetPipeline('rms_norm', LPipeline) then Exit;

  // Align the bind offset down to 256 bytes (covers every implementation's
  // minStorageBufferOffsetAlignment); pass the residual as an element offset.
  LAlignedOffset := (AWeightByteOffset div 256) * 256;
  LResidual := AWeightByteOffset - LAlignedOffset;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBuffer(LDescSet, 0, ASrcBuf);
  FShaders.BindBufferRange(LDescSet, 1, AWeightBuf, LAlignedOffset,
    AWeightByteSize + LResidual);
  FShaders.BindBuffer(LDescSet, 2, ADstBuf);

  LPush.size := ASize;
  LPush.weightOffset := UInt32(LResidual div SizeOf(Single));
  LPush.eps := AEps;

  FShaders.RecordDispatch('rms_norm', LDescSet, 1, 1, 1,
    @LPush, SizeOf(LPush));
end;

procedure TComputeKernels.RecordRmsNormRows(
  const ADataBuf: TVulkanBuffer;
  const AWeightBuf: TVulkanBuffer;
  const ARowSize: UInt32;
  const ANumRows: UInt32;
  const AEps: Single;
  const AHasWeight: Boolean;
  const AWeightOffset: UInt32);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
  LPush: TRmsNormRowsPushConstants;
begin
  if not FShaders.GetPipeline('rms_norm_rows', LPipeline) then Exit;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBuffer(LDescSet, 0, ADataBuf);
  FShaders.BindBuffer(LDescSet, 1, AWeightBuf);

  LPush.rowSize := ARowSize;
  LPush.numRows := ANumRows;
  LPush.eps := AEps;
  if AHasWeight then
    LPush.hasWeight := 1
  else
    LPush.hasWeight := 0;
  LPush.weightOffset := AWeightOffset;

  FShaders.RecordDispatch('rms_norm_rows', LDescSet, ANumRows, 1, 1,
    @LPush, SizeOf(LPush));
end;

procedure TComputeKernels.RecordRmsNormRows(
  const ADataBuf: TVulkanBuffer;
  const AWeightBuf: TVulkanBuffer;
  const AWeightByteOffset: UInt64;
  const AWeightByteSize: UInt64;
  const ARowSize: UInt32;
  const ANumRows: UInt32;
  const AEps: Single);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
  LPush: TRmsNormRowsPushConstants;
  LAlignedOffset: UInt64;
  LResidual: UInt64;
begin
  if not FShaders.GetPipeline('rms_norm_rows', LPipeline) then Exit;

  // Same aligned-bind scheme as the RecordRmsNorm range overload
  LAlignedOffset := (AWeightByteOffset div 256) * 256;
  LResidual := AWeightByteOffset - LAlignedOffset;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBuffer(LDescSet, 0, ADataBuf);
  FShaders.BindBufferRange(LDescSet, 1, AWeightBuf, LAlignedOffset,
    AWeightByteSize + LResidual);

  LPush.rowSize := ARowSize;
  LPush.numRows := ANumRows;
  LPush.eps := AEps;
  LPush.hasWeight := 1;
  LPush.weightOffset := UInt32(LResidual div SizeOf(Single));

  FShaders.RecordDispatch('rms_norm_rows', LDescSet, ANumRows, 1, 1,
    @LPush, SizeOf(LPush));
end;

procedure TComputeKernels.RecordMul(
  const ADstBuf: TVulkanBuffer;
  const ASrcBuf: TVulkanBuffer;
  const ACount: UInt32;
  const ASrcOffset: UInt32);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
  LPush: TMulPushConstants;
  LGroups: UInt32;
begin
  if not FShaders.GetPipeline('mul', LPipeline) then Exit;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBuffer(LDescSet, 0, ADstBuf);
  FShaders.BindBuffer(LDescSet, 1, ASrcBuf);

  LPush.count := ACount;
  LPush.srcOffset := ASrcOffset;
  LGroups := (ACount + CElementwiseWorkgroup - 1) div CElementwiseWorkgroup;

  FShaders.RecordDispatch('mul', LDescSet, LGroups, 1, 1,
    @LPush, SizeOf(LPush));
end;

procedure TComputeKernels.RecordAdd(
  const ADstBuf: TVulkanBuffer;
  const ASrcBuf: TVulkanBuffer;
  const ACount: UInt32);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
  LPush: TAddPushConstants;
  LGroups: UInt32;
begin
  if not FShaders.GetPipeline('add', LPipeline) then Exit;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBuffer(LDescSet, 0, ADstBuf);
  FShaders.BindBuffer(LDescSet, 1, ASrcBuf);

  LPush.count := ACount;
  LGroups := (ACount + CElementwiseWorkgroup - 1) div CElementwiseWorkgroup;

  FShaders.RecordDispatch('add', LDescSet, LGroups, 1, 1,
    @LPush, SizeOf(LPush));
end;

procedure TComputeKernels.RecordScale(
  const ASrcDstBuf: TVulkanBuffer;
  const ACount: UInt32;
  const AScale: Single);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
  LPush: TScalePushConstants;
  LGroups: UInt32;
begin
  if not FShaders.GetPipeline('scale', LPipeline) then Exit;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBuffer(LDescSet, 0, ASrcDstBuf);

  LPush.count := ACount;
  LPush.scale := AScale;
  LGroups := (ACount + CElementwiseWorkgroup - 1) div CElementwiseWorkgroup;

  FShaders.RecordDispatch('scale', LDescSet, LGroups, 1, 1,
    @LPush, SizeOf(LPush));
end;

procedure TComputeKernels.RecordGeLU(
  const ASrcDstBuf: TVulkanBuffer;
  const ACount: UInt32);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
  LPush: TGeluPushConstants;
  LGroups: UInt32;
begin
  if not FShaders.GetPipeline('gelu', LPipeline) then Exit;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBuffer(LDescSet, 0, ASrcDstBuf);

  LPush.count := ACount;
  LGroups := (ACount + CElementwiseWorkgroup - 1) div CElementwiseWorkgroup;

  FShaders.RecordDispatch('gelu', LDescSet, LGroups, 1, 1,
    @LPush, SizeOf(LPush));
end;

procedure TComputeKernels.RecordGeGLU(
  const ADstBuf: TVulkanBuffer;
  const ASrcBuf: TVulkanBuffer;
  const ACount: UInt32;
  const ASrcOffset: UInt32);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
  LPush: TGeGluPushConstants;
  LGroups: UInt32;
begin
  if not FShaders.GetPipeline('geglu', LPipeline) then Exit;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBuffer(LDescSet, 0, ADstBuf);
  FShaders.BindBuffer(LDescSet, 1, ASrcBuf);

  LPush.count := ACount;
  LPush.srcOffset := ASrcOffset;
  LGroups := (ACount + CElementwiseWorkgroup - 1) div CElementwiseWorkgroup;

  FShaders.RecordDispatch('geglu', LDescSet, LGroups, 1, 1,
    @LPush, SizeOf(LPush));
end;

procedure TComputeKernels.RecordCopy(
  const ADstBuf: TVulkanBuffer;
  const ASrcBuf: TVulkanBuffer;
  const ACount: UInt32;
  const ASrcOffset: UInt32;
  const ADstOffset: UInt32);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
  LPush: TCopyPushConstants;
  LGroups: UInt32;
begin
  if not FShaders.GetPipeline('copy', LPipeline) then Exit;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBuffer(LDescSet, 0, ADstBuf);
  FShaders.BindBuffer(LDescSet, 1, ASrcBuf);

  LPush.count := ACount;
  LPush.srcOffset := ASrcOffset;
  LPush.dstOffset := ADstOffset;
  LGroups := (ACount + CElementwiseWorkgroup - 1) div CElementwiseWorkgroup;

  FShaders.RecordDispatch('copy', LDescSet, LGroups, 1, 1,
    @LPush, SizeOf(LPush));
end;

procedure TComputeKernels.RecordRoPE(
  const ADataBuf: TVulkanBuffer;
  const APosition: UInt32;
  const AHeadDim: UInt32;
  const ARotAngles: UInt32;
  const ANumHeads: UInt32;
  const ATheta: Single);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
  LPush: TRopePushConstants;
  LGroups: UInt32;
  LTotal: UInt32;
begin
  if not FShaders.GetPipeline('rope', LPipeline) then Exit;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBuffer(LDescSet, 0, ADataBuf);

  LPush.position := APosition;
  LPush.headDim := AHeadDim;
  LPush.rotAngles := ARotAngles;
  LPush.numHeads := ANumHeads;
  LPush.theta := ATheta;

  // One invocation per (head, angle pair)
  LTotal := ANumHeads * (AHeadDim div 2);
  LGroups := (LTotal + CSmallWorkgroup - 1) div CSmallWorkgroup;

  FShaders.RecordDispatch('rope', LDescSet, LGroups, 1, 1,
    @LPush, SizeOf(LPush));
end;

procedure TComputeKernels.RecordAttnScores(
  const AQueryBuf: TVulkanBuffer;
  const AKCacheBuf: TVulkanBuffer;
  const AScoresBuf: TVulkanBuffer;
  const APush: TAttnPushConstants);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
  LGroups: UInt32;
  LTotal: UInt32;
begin
  if not FShaders.GetPipeline('attn_scores', LPipeline) then Exit;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBuffer(LDescSet, 0, AQueryBuf);
  FShaders.BindBuffer(LDescSet, 1, AKCacheBuf);
  FShaders.BindBuffer(LDescSet, 2, AScoresBuf);

  // One invocation per (head, cache entry)
  LTotal := APush.numHeads * APush.seqLen;
  LGroups := (LTotal + CSmallWorkgroup - 1) div CSmallWorkgroup;

  FShaders.RecordDispatch('attn_scores', LDescSet, LGroups, 1, 1,
    @APush, SizeOf(APush));
end;

procedure TComputeKernels.RecordSoftmaxRows(
  const ADataBuf: TVulkanBuffer;
  const ARowSize: UInt32;
  const ANumRows: UInt32);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
  LPush: TSoftmaxRowsPushConstants;
begin
  if not FShaders.GetPipeline('softmax_rows', LPipeline) then Exit;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBuffer(LDescSet, 0, ADataBuf);

  LPush.rowSize := ARowSize;
  LPush.numRows := ANumRows;

  FShaders.RecordDispatch('softmax_rows', LDescSet, ANumRows, 1, 1,
    @LPush, SizeOf(LPush));
end;

procedure TComputeKernels.RecordAttnOut(
  const AProbsBuf: TVulkanBuffer;
  const AVCacheBuf: TVulkanBuffer;
  const AOutBuf: TVulkanBuffer;
  const APush: TAttnPushConstants);
var
  LPipeline: TComputePipeline;
  LDescSet: TVkDescriptorSet;
  LGroups: UInt32;
  LTotal: UInt32;
begin
  if not FShaders.GetPipeline('attn_out', LPipeline) then Exit;

  LDescSet := FShaders.AllocateDescriptorSet(LPipeline);
  FShaders.BindBuffer(LDescSet, 0, AProbsBuf);
  FShaders.BindBuffer(LDescSet, 1, AVCacheBuf);
  FShaders.BindBuffer(LDescSet, 2, AOutBuf);

  // One invocation per (head, dim)
  LTotal := APush.numHeads * APush.headDim;
  LGroups := (LTotal + CSmallWorkgroup - 1) div CSmallWorkgroup;

  FShaders.RecordDispatch('attn_out', LDescSet, LGroups, 1, 1,
    @APush, SizeOf(APush));
end;

// ---------------------------------------------------------------------------
// Dispatch wrappers (synchronous)
// ---------------------------------------------------------------------------

procedure TComputeKernels.DispatchRmsNorm(
  const ASrcBuf: TVulkanBuffer;
  const AWeightBuf: TVulkanBuffer;
  const ADstBuf: TVulkanBuffer;
  const ASize: UInt32;
  const AEps: Single);
begin
  FDevice.BeginCommands();
  RecordRmsNorm(ASrcBuf, AWeightBuf, ADstBuf, ASize, AEps, 0);
  FDevice.EndCommands();
  FDevice.SubmitAndWait();
end;

procedure TComputeKernels.DispatchRmsNormRows(
  const ADataBuf: TVulkanBuffer;
  const AWeightBuf: TVulkanBuffer;
  const ARowSize: UInt32;
  const ANumRows: UInt32;
  const AEps: Single;
  const AHasWeight: Boolean;
  const AWeightOffset: UInt32);
begin
  FDevice.BeginCommands();
  RecordRmsNormRows(ADataBuf, AWeightBuf, ARowSize, ANumRows, AEps,
    AHasWeight, AWeightOffset);
  FDevice.EndCommands();
  FDevice.SubmitAndWait();
end;

procedure TComputeKernels.DispatchMul(
  const ADstBuf: TVulkanBuffer;
  const ASrcBuf: TVulkanBuffer;
  const ACount: UInt32;
  const ASrcOffset: UInt32);
begin
  FDevice.BeginCommands();
  RecordMul(ADstBuf, ASrcBuf, ACount, ASrcOffset);
  FDevice.EndCommands();
  FDevice.SubmitAndWait();
end;

procedure TComputeKernels.DispatchAdd(
  const ADstBuf: TVulkanBuffer;
  const ASrcBuf: TVulkanBuffer;
  const ACount: UInt32);
begin
  FDevice.BeginCommands();
  RecordAdd(ADstBuf, ASrcBuf, ACount);
  FDevice.EndCommands();
  FDevice.SubmitAndWait();
end;

procedure TComputeKernels.DispatchScale(
  const ASrcDstBuf: TVulkanBuffer;
  const ACount: UInt32;
  const AScale: Single);
begin
  FDevice.BeginCommands();
  RecordScale(ASrcDstBuf, ACount, AScale);
  FDevice.EndCommands();
  FDevice.SubmitAndWait();
end;

procedure TComputeKernels.DispatchGeLU(
  const ASrcDstBuf: TVulkanBuffer;
  const ACount: UInt32);
begin
  FDevice.BeginCommands();
  RecordGeLU(ASrcDstBuf, ACount);
  FDevice.EndCommands();
  FDevice.SubmitAndWait();
end;

procedure TComputeKernels.DispatchCopy(
  const ADstBuf: TVulkanBuffer;
  const ASrcBuf: TVulkanBuffer;
  const ACount: UInt32;
  const ASrcOffset: UInt32;
  const ADstOffset: UInt32);
begin
  FDevice.BeginCommands();
  RecordCopy(ADstBuf, ASrcBuf, ACount, ASrcOffset, ADstOffset);
  FDevice.EndCommands();
  FDevice.SubmitAndWait();
end;

procedure TComputeKernels.DispatchRoPE(
  const ADataBuf: TVulkanBuffer;
  const APosition: UInt32;
  const AHeadDim: UInt32;
  const ARotAngles: UInt32;
  const ANumHeads: UInt32;
  const ATheta: Single);
begin
  FDevice.BeginCommands();
  RecordRoPE(ADataBuf, APosition, AHeadDim, ARotAngles, ANumHeads, ATheta);
  FDevice.EndCommands();
  FDevice.SubmitAndWait();
end;

procedure TComputeKernels.DispatchAttnScores(
  const AQueryBuf: TVulkanBuffer;
  const AKCacheBuf: TVulkanBuffer;
  const AScoresBuf: TVulkanBuffer;
  const APush: TAttnPushConstants);
begin
  FDevice.BeginCommands();
  RecordAttnScores(AQueryBuf, AKCacheBuf, AScoresBuf, APush);
  FDevice.EndCommands();
  FDevice.SubmitAndWait();
end;

procedure TComputeKernels.DispatchSoftmaxRows(
  const ADataBuf: TVulkanBuffer;
  const ARowSize: UInt32;
  const ANumRows: UInt32);
begin
  FDevice.BeginCommands();
  RecordSoftmaxRows(ADataBuf, ARowSize, ANumRows);
  FDevice.EndCommands();
  FDevice.SubmitAndWait();
end;

procedure TComputeKernels.DispatchAttnOut(
  const AProbsBuf: TVulkanBuffer;
  const AVCacheBuf: TVulkanBuffer;
  const AOutBuf: TVulkanBuffer;
  const APush: TAttnPushConstants);
begin
  FDevice.BeginCommands();
  RecordAttnOut(AProbsBuf, AVCacheBuf, AOutBuf, APush);
  FDevice.EndCommands();
  FDevice.SubmitAndWait();
end;

end.
