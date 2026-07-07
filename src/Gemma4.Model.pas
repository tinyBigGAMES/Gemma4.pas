{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information

 -------------------------------------------------------------------------------

  Gemma4.Model - Forward pass and generation loop

  Orchestrates the full Gemma 4 E4B text decoder forward pass:
    1. Token embedding lookup (scaled by sqrt(hidden_size))
    2. PLE pre-computation (per-layer embedding: tok embed + projected hidden)
    3. 42-layer decoder loop per layer:
       - Pre-attn RMSNorm -> Attention (Q/K/V proj, Q/K/V norms, RoPE, GQA) -> Post-attn RMSNorm -> Residual
       - Pre-FFN RMSNorm -> GeGLU MLP (gate/up/down) -> Post-FFN RMSNorm -> Residual
       - PLE (gate -> GeLU -> mul precomputed -> project -> norm -> residual)
       - Layer scalar output scaling
    4. Final RMSNorm -> lm_head projection -> logit soft-capping
    5. Sampling (top-k, top-p, temperature)

  GPU acceleration via TComputeKernels (when available):
    - MatVec projections (Q/K/V/O, MLP gate/up/down, lm_head) on GPU
    - RMSNorm on GPU
    - Single contiguous GPU weight buffer with descriptor offset binding
    - CPU fallback for all operations when GPU unavailable

  Dependencies: StdApp.Base, Gemma4.Types, Gemma4.Config,
    Gemma4.Tensors, Gemma4.Layers, Gemma4.Attention, Gemma4.Quant,
    Gemma4.Vulkan, Gemma4.Compute
===============================================================================}

unit Gemma4.Model;

{$I StdApp.Defines.inc}

interface

uses
  System.SysUtils,
  System.Math,
  System.Diagnostics,
  System.Generics.Collections,
  StdApp.Base,
  Gemma4.Types,
  Gemma4.Config,
  Gemma4.Tensors,
  Gemma4.Layers,
  Gemma4.Attention,
  Gemma4.Quant,
  Gemma4.Vulkan,
  Gemma4.Compute;

const
  CMD_ERR_WEIGHTS = 'MD01';
  CMD_ERR_FORWARD = 'MD02';
  CMD_ERR_GPU = 'MD03';

  // Maximum sequence length for the GPU-resident KV cache. Full-attention
  // cache slots are sized entries * 2 * 512 floats (32 MB each at 8192).
  // Sliding slots are 512-entry rings regardless of this value.
  CGpuMaxSeq = 8192;

type
  { TSamplingParams }
  TSamplingParams = record
    Temperature: Single;
    TopK: Integer;
    TopP: Single;
    RepetitionPenalty: Single;
    procedure SetDefaults();
  end;

  { TGenerateCallback }
  // Called for each generated token. Return False to stop generation.
  TGenerateCallback = reference to function(
    const ATokenId: Integer;
    const ATokenText: string;
    const AUserData: Pointer
  ): Boolean;

  { TGpuTensorLoc }
  // Where a tensor lives inside FGpuWeights after the upload-time repack.
  // Q4 tensors are split into a 16-byte-aligned quants plane (16 bytes per
  // block, readable as one uvec4 per block) followed by an fp16 scales
  // plane (2 bytes per block). F32 tensors are copied verbatim at a
  // 16-byte-aligned offset (vec4 loads).
  TGpuTensorLoc = record
    BaseOfs: UInt64;    // 16-byte-aligned byte offset (quants plane / f32 data)
    ScalesOfs: UInt64;  // Q4 only: byte offset of the scales plane
    ByteSize: UInt64;   // total bytes occupied starting at BaseOfs
    IsQ4: Boolean;
  end;

  { TSoftTokenBlock }
  // A run of consecutive prompt positions whose residual-stream rows are
  // replaced by precomputed multimodal soft tokens (raw language-model-space
  // rows -- NO sqrt(hidden) embedding scale). Rows is flat
  // [RowCount x HiddenSize]. PLE at these positions: identity lookup from
  // the PAD token, context projection from the soft row itself (HF projects
  // the merged embeddings -- modeling_gemma4.py line 1684).
  TSoftTokenBlock = record
    Position: Integer;        // absolute position of the first substituted row
    RowCount: Integer;        // number of consecutive substituted rows
    Rows: TArray<Single>;     // RowCount * HiddenSize floats
  end;

  { TModel }
  // Gemma 4 E4B text decoder with optional GPU acceleration
  TModel = class(TBaseObject)
  private
    FConfig: TModelConfig;
    FWeights: TWeightStore;
    FKvCache: TKvCache;
    FOwnsWeights: Boolean;
    FIsLoaded: Boolean;
    FPosition: Integer;

    // CPU scratch buffers (reused across forward calls)
    FHidden: TArray<Single>;       // [hidden_size]
    FNormBuf: TArray<Single>;      // [hidden_size]
    FAttnOut: TArray<Single>;      // [num_heads * head_dim]
    FMlpOut: TArray<Single>;       // [hidden_size]
    FQuery: TArray<Single>;        // [num_heads * head_dim]
    FKeyBuf: TArray<Single>;       // [num_kv_heads * head_dim]
    FValueBuf: TArray<Single>;     // [num_kv_heads * head_dim]
    FLogits: TArray<Single>;       // [vocab_size]
    FMlpGate: TArray<Single>;      // [intermediate_size] GPU MLP scratch
    FMlpUp: TArray<Single>;        // [intermediate_size] GPU MLP scratch

    // PLE (per-layer embedding) scratch buffers
    FPLEInput: TArray<Single>;     // [n_embd_per_layer * n_layer] precomputed per forward
    FPLEGate: TArray<Single>;      // [n_embd_per_layer] gate output scratch

    // GPU state
    FComputeKernels: TComputeKernels; // reference, not owned
    FUseGPU: Boolean;
    FGpuWeights: TVulkanBuffer;    // single contiguous weight buffer
    FGpuBufA: TVulkanBuffer;       // working input buffer
    FGpuBufB: TVulkanBuffer;       // working output buffer
    FGpuNormWeight: TVulkanBuffer;  // reusable norm gamma buffer

    // GPU-resident state (single-submit forward pass, PLAN-gpu-residency)
    FGpuHidden: TVulkanBuffer;     // [2560] residual stream
    FGpuNorm: TVulkanBuffer;       // [2560] normed activations
    FGpuQ: TVulkanBuffer;          // [8*512] Q (sized for max head dim)
    FGpuKStage: TVulkanBuffer;     // [2*512] K staging before ring append
    FGpuVStage: TVulkanBuffer;     // [2*512] V staging
    FGpuScores: TVulkanBuffer;     // [8*CGpuMaxSeq] attention scores/probs
    FGpuAttnOut: TVulkanBuffer;    // [8*512] attention output
    FGpuMlpGate: TVulkanBuffer;    // [10240] gate activations
    FGpuMlpUp: TVulkanBuffer;      // [10240] up activations
    FGpuTmp: TVulkanBuffer;        // [2560] sub-block output scratch
    FGpuPLEInput: TVulkanBuffer;   // [10752] per-token PLE input (host-visible upload target)
    FGpuPLEProj: TVulkanBuffer;    // [10752] PLE context projection scratch
    FGpuPLEGate: TVulkanBuffer;    // [256] PLE gate scratch
    FGpuLogits: TVulkanBuffer;     // [262144] logits (host-visible for download)
    FGpuEmbed: TVulkanBuffer;      // [2560] embedding upload target (host-visible)
    FGpuKvK: TArray<TVulkanBuffer>; // per unique slot: K cache (ring or full)
    FGpuKvV: TArray<TVulkanBuffer>; // per unique slot: V cache
    // Batched-prefill state (allocated only when FUseBatchPrefill)
    FUseBatchPrefill: Boolean;       // FUseGPU and device DP4A support
    FGpuBatchHidden: TVulkanBuffer;  // [B x 2560] residual stream
    FGpuBatchNorm: TVulkanBuffer;    // [B x 2560] normed activations
    FGpuBatchQ: TVulkanBuffer;       // [B x 8 x 512] (max head dim)
    FGpuBatchKStage: TVulkanBuffer;  // [B x 2 x 512]
    FGpuBatchVStage: TVulkanBuffer;  // [B x 2 x 512]
    FGpuBatchScores: TVulkanBuffer;  // [B x 8 x CGpuMaxSeq]
    FGpuBatchAttnOut: TVulkanBuffer; // [B x 8 x 512]
    FGpuBatchMlpGate: TVulkanBuffer; // [B x 10240]
    FGpuBatchMlpUp: TVulkanBuffer;   // [B x 10240]
    FGpuBatchTmp: TVulkanBuffer;     // [B x 2560] sub-block output scratch
    FGpuBatchPLEProj: TVulkanBuffer; // [B x 10752]
    FGpuBatchPLEGate: TVulkanBuffer; // [B x 256]
    FGpuBatchEmbed: TVulkanBuffer;   // [B x 2560] host-visible upload target
    FGpuBatchPLEInput: TVulkanBuffer;// [B x 10752] host-visible upload target
    FBatchEmbedStage: TArray<Single>;   // CPU staging [B x 2560]
    FBatchPLEStage: TArray<Single>;     // CPU staging [B x 10752]
    FLayerScalars: TArray<Single>; // per-layer output scalar, loaded once
    FGpuOfs: TDictionary<string, TGpuTensorLoc>; // upload-time GPU layout table

    // Soft-token substitution state for the current Generate call
    FSoftBlocks: TArray<TSoftTokenBlock>;
    FGpuSoftStage: TVulkanBuffer;      // host-visible [CPrefillBatch x hidden]
    FBatchSoftStage: TArray<Single>;   // CPU packing for the batch path

    // Per-token phase profiling (accumulated over one Generate call, prefill excluded)
    FProfUploadMs: Double;    // descriptor pool reset + host-visible input writes
    FProfRecordMs: Double;    // command buffer recording (CPU)
    FProfFenceMs: Double;     // SubmitAndWait -- actual GPU execution time
    FProfDownloadMs: Double;  // logits readback (1 MB)
    FProfSampleMs: Double;    // CPU top-K select + softcap + sampling
    FProfTokens: Integer;     // tokens accumulated since last reset
    // GPU-internal split from timestamp queries (ms, accumulated):
    // 0=embed+ple-pre, 1..3=full layer attn/mlp/ple, 4..6=sliding layer
    // attn/mlp/ple, 7=all layers total, 8=final norm + lm_head
    FProfGpuMs: array[0..8] of Double;
    FProfElapsedSec: Double;  // wall-clock generation time (prefill excluded)
    FProfPrefillTokens: Integer;  // prompt tokens processed before generation
    FProfPrefillSec: Double;      // wall-clock prompt processing time
    FProfPrefillGpuMs: array[0..8] of Double; // same slot meaning as FProfGpuMs
    FProfPrefillChunks: Integer;              // batched chunks measured
    FLoadProgress: TCallback<TLoadProgressCallback>;

    procedure DoAllocateBuffers();
    function DoUploadWeightsToGPU(): Boolean;
    procedure DoFreeGpuBuffers();

    // GPU-resident forward pass (single command buffer per token)
    function DoAllocateResidentGpuBuffers(): Boolean;
    procedure DoLoadLayerScalars();
    procedure DoUploadTokenInputsGpu(const ATokenId: Integer);
    procedure DoRecordNormGpu(const ATensorName: string;
      const ASrcBuf: TVulkanBuffer; const ADstBuf: TVulkanBuffer;
      const ASize: Integer);
    procedure DoRecordNormRowsGpu(const ATensorName: string;
      const ADataBuf: TVulkanBuffer; const ARowSize: Integer;
      const ANumRows: Integer);
    procedure DoRecordProjGpu(const ATensorName: string;
      const AInputBuf: TVulkanBuffer; const AOutputBuf: TVulkanBuffer;
      const ARows: Integer; const ACols: Integer);
    procedure DoRecordPlePrecomputeGpu();
    procedure DoRecordAttentionGpu(const ALayerIdx: Integer);
    procedure DoRecordMlpGpu(const ALayerIdx: Integer);
    procedure DoRecordPleGpu(const ALayerIdx: Integer);
    procedure DoRecordDecoderLayerGpu(const ALayerIdx: Integer);
    procedure DoRecordFinalGpu();
    procedure DoForwardGpu(const ATokenId: Integer);

    // Batched prefill (chunk of up to CPrefillBatch prompt positions)
    procedure DoUploadBatchInputsGpu(const ATokens: TArray<Integer>;
      const AFirst: Integer; const ACount: Integer);
    procedure DoRecordNormGpuBatch(const ATensorName: string;
      const ASrcBuf: TVulkanBuffer; const ADstBuf: TVulkanBuffer;
      const ASize: Integer; const ABatch: Integer);
    procedure DoRecordProjGpuBatch(const ATensorName: string;
      const AInputBuf: TVulkanBuffer; const AOutputBuf: TVulkanBuffer;
      const ARows: Integer; const ACols: Integer; const ABatch: Integer);
    procedure DoRecordPlePrecomputeGpuBatch(const ABatch: Integer);
    procedure DoRecordAttentionGpuBatch(const ALayerIdx: Integer;
      const ABatch: Integer);
    procedure DoRecordMlpGpuBatch(const ALayerIdx: Integer;
      const ABatch: Integer);
    procedure DoRecordPleGpuBatch(const ALayerIdx: Integer;
      const ABatch: Integer);
    procedure DoRecordDecoderLayerGpuBatch(const ALayerIdx: Integer;
      const ABatch: Integer);
    procedure DoForwardGpuBatch(const ATokens: TArray<Integer>;
      const AFirst: Integer; const ACount: Integer; const AIsLast: Boolean);

    // GPU dispatch helpers
    procedure DoGpuMatVec(const AInfo: TTensorInfo;
      const AInput: PSingle; const AInputCount: Integer;
      const AOutput: PSingle; const ARows: Integer; const ACols: Integer);
    procedure DoGpuRmsNorm(const ANormInfo: TTensorInfo;
      const ASrc: PSingle; const ADst: PSingle;
      const ASize: Integer; const AEps: Single);

    // Diagnostic: log min/max/mean of a float buffer (first forward only)
    procedure DoTraceBuffer(const ALabel: string;
      const AData: PSingle; const ACount: Integer);

    procedure DoEmbeddingLookup(const ATokenId: Integer);
    // Soft-token substitution: pointer to the row for absolute position
    // APos, or nil when the position is not substituted
    function DoSoftRowAt(const APos: Integer): PSingle;
    procedure DoPrecomputePLE(const ATokenId: Integer);
    procedure DoApplyPLE(const ALayerIdx: Integer);
    procedure DoDecoderLayer(const ALayerIdx: Integer);
    procedure DoAttention(const ALayerIdx: Integer);
    procedure DoMLP(const ALayerIdx: Integer);
    procedure DoFinalNormAndProject();
    function DoSample(const AParams: TSamplingParams): Integer;
    function DoSelectTopKIndices(const AK: Integer): TArray<Integer>;
    function DoSampleTopP(var ALogits: TArray<Single>;
      const ATopP: Single; const ATemperature: Single): Integer;

    function LayerPrefix(const ALayerIdx: Integer): string;
  public
    constructor Create(); override;
    destructor Destroy(); override;

    function Load(const AWeights: TWeightStore;
      const AConfig: TModelConfig;
      const AOwn: Boolean = False;
      const AComputeKernels: TComputeKernels = nil): Boolean;
    procedure SetLoadProgressCallback(const ACallback: TLoadProgressCallback;
      const AUserData: Pointer = nil);

    function Forward(const ATokenId: Integer): TArray<Single>;

    function Generate(
      const AInputTokens: TArray<Integer>;
      const AMaxTokens: Integer;
      const AParams: TSamplingParams;
      const ACallback: TGenerateCallback;
      const AUserData: Pointer = nil;
      const ASoftBlocks: TArray<TSoftTokenBlock> = nil
    ): TArray<Integer>;

    procedure Reset();

    // Snapshot the profiling accumulators from the last Generate call
    // into per-token averages (see TInferenceStats in Gemma4.Types)
    procedure GetStats(out AStats: TInferenceStats);

    property Config: TModelConfig read FConfig;
    property Position: Integer read FPosition;
    property IsLoaded: Boolean read FIsLoaded;
    property UseGPU: Boolean read FUseGPU;
  end;

implementation

{ TSamplingParams }

procedure TSamplingParams.SetDefaults();
begin
  Temperature := 1.0;
  TopK := 64;
  TopP := 0.95;
  RepetitionPenalty := 1.0;
end;

{ TModel }

constructor TModel.Create();
begin
  inherited Create();
  FWeights := nil;
  FKvCache := TKvCache.Create();
  FOwnsWeights := False;
  FIsLoaded := False;
  FPosition := 0;
  FComputeKernels := nil;
  FUseGPU := False;
  FUseBatchPrefill := False;
  FGpuWeights := Default(TVulkanBuffer);
  FGpuBufA := Default(TVulkanBuffer);
  FGpuBufB := Default(TVulkanBuffer);
  FGpuNormWeight := Default(TVulkanBuffer);
  FGpuOfs := TDictionary<string, TGpuTensorLoc>.Create();
  FSoftBlocks := nil;
  FGpuSoftStage := Default(TVulkanBuffer);
end;

destructor TModel.Destroy();
begin
  DoFreeGpuBuffers();
  FGpuOfs.Free();
  FKvCache.Free();
  if FOwnsWeights and (FWeights <> nil) then
    FWeights.Free();
  inherited;
end;

const
  // GPU timestamp query indices for the per-token GPU-side profile.
  // One representative full-attention layer and one sliding-window layer
  // are instrumented (pattern: layers 0-4 sliding, 5 full, repeating).

  { CPadTokenId }
  // text_config.pad_token_id -- embed + PLE source for substituted positions
  CPadTokenId = 0;

  { CTsStart }
  CTsStart = 0;        // command buffer start (before embed copy)
  { CTsPlePre }
  CTsPlePre = 1;       // after embed copy + PLE precompute
  { CTsLFullStart }
  CTsLFullStart = 2;   // full-attention profile layer: start
  { CTsLFullAttn }
  CTsLFullAttn = 3;    // full-attention profile layer: after attention block
  { CTsLFullMlp }
  CTsLFullMlp = 4;     // full-attention profile layer: after MLP block
  { CTsLFullEnd }
  CTsLFullEnd = 5;     // full-attention profile layer: after PLE + scale
  { CTsLSlidStart }
  CTsLSlidStart = 6;   // sliding-window profile layer: start
  { CTsLSlidAttn }
  CTsLSlidAttn = 7;    // sliding-window profile layer: after attention block
  { CTsLSlidMlp }
  CTsLSlidMlp = 8;     // sliding-window profile layer: after MLP block
  { CTsLSlidEnd }
  CTsLSlidEnd = 9;     // sliding-window profile layer: after PLE + scale
  { CTsLayersEnd }
  CTsLayersEnd = 10;   // after all decoder layers
  { CTsEnd }
  CTsEnd = 11;         // after final norm + lm_head (logits ready)
  { CTsCount }
  CTsCount = 12;
  { CProfLayerFull }
  CProfLayerFull = 5;    // instrumented full-attention layer index
  { CProfLayerSliding }
  CProfLayerSliding = 6; // instrumented sliding-window layer index

function TModel.LayerPrefix(const ALayerIdx: Integer): string;
begin
  Result := Format('model.language_model.layers.%d.', [ALayerIdx]);
end;

procedure TModel.DoAllocateBuffers();
var
  LMaxHeadDim: Integer;
  LMaxInputSize: UInt64;
  LMaxOutputSize: UInt64;
begin
  LMaxHeadDim := Max(FConfig.HeadDim, FConfig.GlobalHeadDim);

  // CPU scratch buffers
  SetLength(FHidden, FConfig.HiddenSize);
  SetLength(FNormBuf, FConfig.HiddenSize);
  SetLength(FAttnOut, FConfig.NumAttentionHeads * LMaxHeadDim);
  SetLength(FMlpOut, FConfig.HiddenSize);
  SetLength(FQuery, FConfig.NumAttentionHeads * LMaxHeadDim);
  SetLength(FKeyBuf, FConfig.NumKeyValueHeads * LMaxHeadDim);
  SetLength(FValueBuf, FConfig.NumKeyValueHeads * LMaxHeadDim);
  SetLength(FLogits, FConfig.VocabSize);

  // MLP intermediate scratch (needed for GPU decomposition of SwiGLU)
  SetLength(FMlpGate, FConfig.IntermediateSize);
  SetLength(FMlpUp, FConfig.IntermediateSize);

  // PLE scratch buffers
  SetLength(FPLEInput, FConfig.HiddenSizePerLayerInput * FConfig.NumHiddenLayers);
  SetLength(FPLEGate, FConfig.HiddenSizePerLayerInput);

  // GPU working buffers
  if FUseGPU then
  begin
    // BufA: input side -- max of hidden_size, intermediate_size, num_heads*head_dim
    LMaxInputSize := UInt64(Max(FConfig.IntermediateSize,
      Max(FConfig.HiddenSize, FConfig.NumAttentionHeads * LMaxHeadDim))) * SizeOf(Single);

    // BufB: output side -- max is vocab_size (for lm_head projection)
    LMaxOutputSize := UInt64(Max(FConfig.VocabSize,
      Max(FConfig.IntermediateSize, FConfig.NumAttentionHeads * LMaxHeadDim))) * SizeOf(Single);

    FGpuBufA := FComputeKernels.Device.CreateBuffer(LMaxInputSize, True);
    FGpuBufB := FComputeKernels.Device.CreateBuffer(LMaxOutputSize, True);
    FGpuNormWeight := FComputeKernels.Device.CreateBuffer(
      UInt64(FConfig.HiddenSize) * SizeOf(Single), True);

    if (FGpuBufA.Buffer = 0) or (FGpuBufB.Buffer = 0) or
       (FGpuNormWeight.Buffer = 0) then
    begin
      GetErrors().Add(esError, CMD_ERR_GPU, 'Failed to allocate GPU working buffers');
      DoFreeGpuBuffers();
      FUseGPU := False;
    end;

    // GPU-resident buffers for the single-submit forward pass
    if FUseGPU then
    begin
      if not DoAllocateResidentGpuBuffers() then
      begin
        GetErrors().Add(esError, CMD_ERR_GPU, 'Failed to allocate GPU-resident buffers');
        DoFreeGpuBuffers();
        FUseGPU := False;
      end;
    end;
  end;
end;

function TModel.DoUploadWeightsToGPU(): Boolean;
var
  LI: Integer;
  LB: UInt64;
  LInfo: TTensorInfo;
  LLoc: TGpuTensorLoc;
  LCursor: UInt64;
  LBlocks: UInt64;
  LStage: TBytes;
  LSrc: PByte;
  LCpuCheck: UInt32;
  LGpuCheck: UInt32;
  LTmpBlock: array[0..15] of Byte;

  // Reorder one Q4_0 block's 16 nibble bytes from the interleaved source
  // layout (byte j = v[2j] low nibble, v[2j+1] high nibble) to the planar
  // GPU layout (byte j = v[j] low nibble, v[j+16] high nibble). Planar
  // order lets the matvec shader read the input vector as two contiguous
  // 16-element runs per block (vec4 loads) instead of stride-2 gathers.
  procedure DoReorderQ4Block(const ASrc: PByte; const ADst: PByte);
  var
    LJ: Integer;
    LByteLo: Byte;
    LByteHi: Byte;
  begin
    for LJ := 0 to 15 do
    begin
      // v[LJ] lives in source byte (LJ div 2), v[LJ+16] in byte (LJ div 2)+8;
      // both in the low nibble when LJ is even, high nibble when odd
      LByteLo := ASrc[LJ shr 1];
      LByteHi := ASrc[(LJ shr 1) + 8];
      if (LJ and 1) = 0 then
        ADst[LJ] := Byte((LByteLo and $0F) or ((LByteHi and $0F) shl 4))
      else
        ADst[LJ] := Byte((LByteLo shr 4) or (LByteHi and $F0));
    end;
  end;

begin
  Result := False;

  // Pass 1: assign the GPU-side layout. Every tensor starts 16-byte
  // aligned. Q4 tensors: quants plane (16 bytes/block) then scales plane
  // (2 bytes/block) -- same total bytes as the packed 18-byte blocks.
  FGpuOfs.Clear();
  LCursor := 0;
  for LI := 0 to FWeights.TensorCount() - 1 do
  begin
    LInfo := FWeights.GetTensorInfo(LI);
    LCursor := (LCursor + 15) and not UInt64(15);

    LLoc.BaseOfs := LCursor;
    LLoc.ByteSize := LInfo.DataSize;
    LLoc.IsQ4 := LInfo.OutputDtype = dkQ4_0;
    if LLoc.IsQ4 then
    begin
      LBlocks := LInfo.DataSize div 18;
      LLoc.ScalesOfs := LCursor + LBlocks * 16;
    end
    else
      LLoc.ScalesOfs := 0;

    FGpuOfs.AddOrSetValue(LInfo.TensorName, LLoc);
    LCursor := LCursor + LInfo.DataSize;
  end;

  if LCursor = 0 then
  begin
    GetErrors().Add(esError, CMD_ERR_GPU, 'Weight data size is zero');
    Exit;
  end;

  // Create the weight buffer in device-local VRAM (see PLAN-gpu-residency:
  // host-visible placement caps throughput at ~5 tok/s over PCIe)
  FGpuWeights := FComputeKernels.Device.CreateBuffer(LCursor, False);
  if FGpuWeights.Buffer = 0 then
  begin
    GetErrors().Add(esError, CMD_ERR_GPU, 'Failed to allocate GPU weight buffer');
    Exit;
  end;

  if FGpuWeights.IsDeviceLocal then
    Status('GPU weight buffer: %.1f MB in device-local VRAM',
      [LCursor / (1024.0 * 1024.0)])
  else
    Status('WARNING: GPU weight buffer fell back to HOST memory (%.1f MB) - expect degraded speed',
      [LCursor / (1024.0 * 1024.0)]);

  // Pass 2: per-tensor repack into a transient CPU staging block, then
  // staged upload at the tensor's new offset. Largest tensor (embed_tokens,
  // ~377 MB Q4) bounds the transient allocation.
  for LI := 0 to FWeights.TensorCount() - 1 do
  begin
    LInfo := FWeights.GetTensorInfo(LI);
    if not FGpuOfs.TryGetValue(LInfo.TensorName, LLoc) then Continue;

    LSrc := PByte(FWeights.GetRawPointer(LInfo));

    if LLoc.IsQ4 then
    begin
      // Split 18-byte blocks [scale u16 | 16 nibble bytes] into planes
      LBlocks := LInfo.DataSize div 18;
      SetLength(LStage, LInfo.DataSize);
      for LB := 0 to LBlocks - 1 do
      begin
        // Quants: bytes 2..17 of the block -> 16 bytes at LB*16,
        // nibble-reordered to the planar GPU layout (see DoReorderQ4Block)
        DoReorderQ4Block(@LSrc[LB * 18 + 2], @LStage[LB * 16]);
        // Scale: bytes 0..1 of the block -> 2 bytes in the scales plane
        Move(LSrc[LB * 18], LStage[LBlocks * 16 + LB * 2], 2);
      end;
      FComputeKernels.Device.UploadToDeviceBuffer(FGpuWeights, @LStage[0],
        LInfo.DataSize, LLoc.BaseOfs);
    end
    else
    begin
      // F32 (norm weights, scalars): verbatim copy at the aligned offset
      FComputeKernels.Device.UploadToDeviceBuffer(FGpuWeights, LSrc,
        LInfo.DataSize, LLoc.BaseOfs);
    end;

    if FLoadProgress.IsAssigned() then
      FLoadProgress.Callback(psInProgress, LI + 1, FWeights.TensorCount(),
        FLoadProgress.UserData);
  end;
  SetLength(LStage, 0);

  // Verify: first 4 bytes of tensor 0 at its GPU offset vs the expected
  // repacked source bytes (Q4: nibble-reordered quants plane, block 0)
  LInfo := FWeights.GetTensorInfo(0);
  FGpuOfs.TryGetValue(LInfo.TensorName, LLoc);
  LSrc := PByte(FWeights.GetRawPointer(LInfo));
  LCpuCheck := 0;
  if LLoc.IsQ4 then
  begin
    DoReorderQ4Block(@LSrc[2], @LTmpBlock[0]);
    Move(LTmpBlock[0], LCpuCheck, SizeOf(UInt32));
  end
  else
    Move(LSrc[0], LCpuCheck, SizeOf(UInt32));
  LGpuCheck := 0;
  FComputeKernels.Device.DownloadFromDeviceBuffer(FGpuWeights, @LGpuCheck,
    SizeOf(UInt32), LLoc.BaseOfs);
  Status('GPU weight verify: CPU=0x%s GPU=0x%s match=%s',
    [IntToHex(LCpuCheck, 8), IntToHex(LGpuCheck, 8),
     BoolToStr(LCpuCheck = LGpuCheck, True)]);

  if LCpuCheck <> LGpuCheck then
  begin
    GetErrors().Add(esError, CMD_ERR_GPU, 'GPU weight upload verification FAILED');
    Exit;
  end;

  Status('GPU weights uploaded (split-plane repack): %.1f MB',
    [LCursor / (1024.0 * 1024.0)]);
  Result := True;
end;

procedure TModel.DoFreeGpuBuffers();
var
  LI: Integer;
begin
  if FComputeKernels = nil then Exit;
  // TComputeKernels.Shutdown() nils its device reference, so Device can be
  // nil here even when the kernels object itself is still alive. Calling
  // IsInitialized() on a nil device reads a field off nil -> AV.
  if FComputeKernels.Device = nil then Exit;
  if not FComputeKernels.Device.IsInitialized() then Exit;

  FComputeKernels.Device.DestroyTimestampPool();

  if FGpuWeights.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuWeights);
  if FGpuBufA.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuBufA);
  if FGpuBufB.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuBufB);
  if FGpuNormWeight.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuNormWeight);

  // GPU-resident buffers
  if FGpuHidden.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuHidden);
  if FGpuNorm.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuNorm);
  if FGpuQ.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuQ);
  if FGpuKStage.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuKStage);
  if FGpuVStage.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuVStage);
  if FGpuScores.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuScores);
  if FGpuAttnOut.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuAttnOut);
  if FGpuMlpGate.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuMlpGate);
  if FGpuMlpUp.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuMlpUp);
  if FGpuTmp.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuTmp);
  if FGpuPLEInput.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuPLEInput);
  if FGpuPLEProj.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuPLEProj);
  if FGpuPLEGate.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuPLEGate);
  if FGpuLogits.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuLogits);
  if FGpuEmbed.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuEmbed);
  if FGpuSoftStage.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuSoftStage);

  for LI := 0 to High(FGpuKvK) do
    if FGpuKvK[LI].Buffer <> 0 then
      FComputeKernels.Device.DestroyBuffer(FGpuKvK[LI]);
  for LI := 0 to High(FGpuKvV) do
    if FGpuKvV[LI].Buffer <> 0 then
      FComputeKernels.Device.DestroyBuffer(FGpuKvV[LI]);
  SetLength(FGpuKvK, 0);
  SetLength(FGpuKvV, 0);

  // Batched-prefill buffers
  if FGpuBatchHidden.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuBatchHidden);
  if FGpuBatchNorm.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuBatchNorm);
  if FGpuBatchQ.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuBatchQ);
  if FGpuBatchKStage.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuBatchKStage);
  if FGpuBatchVStage.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuBatchVStage);
  if FGpuBatchScores.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuBatchScores);
  if FGpuBatchAttnOut.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuBatchAttnOut);
  if FGpuBatchMlpGate.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuBatchMlpGate);
  if FGpuBatchMlpUp.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuBatchMlpUp);
  if FGpuBatchTmp.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuBatchTmp);
  if FGpuBatchPLEProj.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuBatchPLEProj);
  if FGpuBatchPLEGate.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuBatchPLEGate);
  if FGpuBatchEmbed.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuBatchEmbed);
  if FGpuBatchPLEInput.Buffer <> 0 then
    FComputeKernels.Device.DestroyBuffer(FGpuBatchPLEInput);
  FUseBatchPrefill := False;
end;

function TModel.DoAllocateResidentGpuBuffers(): Boolean;
var
  LI: Integer;
  LMaxHeadDim: Integer;
  LEntries: Integer;
  LStride: Integer;
  LOk: Boolean;
begin
  Result := False;
  LMaxHeadDim := Max(FConfig.HeadDim, FConfig.GlobalHeadDim);

  // Device-local working state (only ever touched by shaders)
  FGpuHidden := FComputeKernels.Device.CreateBuffer(
    UInt64(FConfig.HiddenSize) * SizeOf(Single), False);
  FGpuNorm := FComputeKernels.Device.CreateBuffer(
    UInt64(FConfig.HiddenSize) * SizeOf(Single), False);
  FGpuQ := FComputeKernels.Device.CreateBuffer(
    UInt64(FConfig.NumAttentionHeads * LMaxHeadDim) * SizeOf(Single), False);
  FGpuKStage := FComputeKernels.Device.CreateBuffer(
    UInt64(FConfig.NumKeyValueHeads * LMaxHeadDim) * SizeOf(Single), False);
  FGpuVStage := FComputeKernels.Device.CreateBuffer(
    UInt64(FConfig.NumKeyValueHeads * LMaxHeadDim) * SizeOf(Single), False);
  FGpuScores := FComputeKernels.Device.CreateBuffer(
    UInt64(FConfig.NumAttentionHeads) * UInt64(CGpuMaxSeq) * SizeOf(Single), False);
  FGpuAttnOut := FComputeKernels.Device.CreateBuffer(
    UInt64(FConfig.NumAttentionHeads * LMaxHeadDim) * SizeOf(Single), False);
  FGpuMlpGate := FComputeKernels.Device.CreateBuffer(
    UInt64(FConfig.IntermediateSize) * SizeOf(Single), False);
  FGpuMlpUp := FComputeKernels.Device.CreateBuffer(
    UInt64(FConfig.IntermediateSize) * SizeOf(Single), False);
  FGpuTmp := FComputeKernels.Device.CreateBuffer(
    UInt64(FConfig.HiddenSize) * SizeOf(Single), False);
  FGpuPLEProj := FComputeKernels.Device.CreateBuffer(
    UInt64(FConfig.HiddenSizePerLayerInput * FConfig.NumHiddenLayers) * SizeOf(Single), False);
  FGpuPLEGate := FComputeKernels.Device.CreateBuffer(
    UInt64(FConfig.HiddenSizePerLayerInput) * SizeOf(Single), False);

  // Host-visible buffers: per-token upload targets and the logits download
  FGpuPLEInput := FComputeKernels.Device.CreateBuffer(
    UInt64(FConfig.HiddenSizePerLayerInput * FConfig.NumHiddenLayers) * SizeOf(Single), True);
  FGpuEmbed := FComputeKernels.Device.CreateBuffer(
    UInt64(FConfig.HiddenSize) * SizeOf(Single), True);
  FGpuLogits := FComputeKernels.Device.CreateBuffer(
    UInt64(FConfig.VocabSize) * SizeOf(Single), True);
  // Soft-token substitution staging (single-token AND batch paths)
  FGpuSoftStage := FComputeKernels.Device.CreateBuffer(
    UInt64(CPrefillBatch) * UInt64(FConfig.HiddenSize) * SizeOf(Single), True);
  SetLength(FBatchSoftStage, CPrefillBatch * FConfig.HiddenSize);

  LOk := (FGpuHidden.Buffer <> 0) and (FGpuNorm.Buffer <> 0) and
    (FGpuQ.Buffer <> 0) and (FGpuKStage.Buffer <> 0) and
    (FGpuVStage.Buffer <> 0) and (FGpuScores.Buffer <> 0) and
    (FGpuAttnOut.Buffer <> 0) and (FGpuMlpGate.Buffer <> 0) and
    (FGpuMlpUp.Buffer <> 0) and (FGpuTmp.Buffer <> 0) and
    (FGpuPLEProj.Buffer <> 0) and (FGpuPLEGate.Buffer <> 0) and
    (FGpuPLEInput.Buffer <> 0) and (FGpuEmbed.Buffer <> 0) and
    (FGpuLogits.Buffer <> 0) and (FGpuSoftStage.Buffer <> 0);
  if not LOk then Exit;

  // GPU KV cache: one K + one V buffer per unique slot. Slot i is owned by
  // layer i (layers 0..23 have their own caches; 24..41 share via KvCacheMap).
  // Geometry follows the owner layer's attention type:
  //   sliding: 512-entry ring x (2 * 256) floats  = 1 MB
  //   full:    CGpuMaxSeq entries x (2 * 512) floats
  SetLength(FGpuKvK, FConfig.NumUniqueKvCaches);
  SetLength(FGpuKvV, FConfig.NumUniqueKvCaches);
  for LI := 0 to FConfig.NumUniqueKvCaches - 1 do
  begin
    if FConfig.LayerTypes[LI] = akFull then
    begin
      LEntries := CGpuMaxSeq;
      LStride := FConfig.NumKeyValueHeads * FConfig.GlobalHeadDim;
    end
    else
    begin
      // Window + one full prefill chunk: batched appends never overwrite
      // keys still inside any in-chunk query's window (see PLAN-batched-prefill)
      LEntries := FConfig.SlidingWindow + CPrefillBatch;
      LStride := FConfig.NumKeyValueHeads * FConfig.HeadDim;
    end;

    FGpuKvK[LI] := FComputeKernels.Device.CreateBuffer(
      UInt64(LEntries) * UInt64(LStride) * SizeOf(Single), False);
    FGpuKvV[LI] := FComputeKernels.Device.CreateBuffer(
      UInt64(LEntries) * UInt64(LStride) * SizeOf(Single), False);

    if (FGpuKvK[LI].Buffer = 0) or (FGpuKvV[LI].Buffer = 0) then Exit;
  end;

  // Timestamp query pool for GPU-side per-token profiling (non-fatal on failure;
  // all timestamp calls are safe no-ops without a pool)
  FComputeKernels.Device.CreateTimestampPool(CTsCount);

  // Batched-prefill buffers (DP4A devices only; ~120 MB device-local)
  FUseBatchPrefill := FComputeKernels.Device.HasIntegerDotProduct();
  if FUseBatchPrefill then
  begin
    FGpuBatchHidden := FComputeKernels.Device.CreateBuffer(
      UInt64(CPrefillBatch) * UInt64(FConfig.HiddenSize) * SizeOf(Single), False);
    FGpuBatchNorm := FComputeKernels.Device.CreateBuffer(
      UInt64(CPrefillBatch) * UInt64(FConfig.HiddenSize) * SizeOf(Single), False);
    FGpuBatchQ := FComputeKernels.Device.CreateBuffer(
      UInt64(CPrefillBatch) * UInt64(FConfig.NumAttentionHeads * LMaxHeadDim) * SizeOf(Single), False);
    FGpuBatchKStage := FComputeKernels.Device.CreateBuffer(
      UInt64(CPrefillBatch) * UInt64(FConfig.NumKeyValueHeads * LMaxHeadDim) * SizeOf(Single), False);
    FGpuBatchVStage := FComputeKernels.Device.CreateBuffer(
      UInt64(CPrefillBatch) * UInt64(FConfig.NumKeyValueHeads * LMaxHeadDim) * SizeOf(Single), False);
    FGpuBatchScores := FComputeKernels.Device.CreateBuffer(
      UInt64(CPrefillBatch) * UInt64(FConfig.NumAttentionHeads) * UInt64(CGpuMaxSeq) * SizeOf(Single), False);
    FGpuBatchAttnOut := FComputeKernels.Device.CreateBuffer(
      UInt64(CPrefillBatch) * UInt64(FConfig.NumAttentionHeads * LMaxHeadDim) * SizeOf(Single), False);
    FGpuBatchMlpGate := FComputeKernels.Device.CreateBuffer(
      UInt64(CPrefillBatch) * UInt64(FConfig.IntermediateSize) * SizeOf(Single), False);
    FGpuBatchMlpUp := FComputeKernels.Device.CreateBuffer(
      UInt64(CPrefillBatch) * UInt64(FConfig.IntermediateSize) * SizeOf(Single), False);
    FGpuBatchTmp := FComputeKernels.Device.CreateBuffer(
      UInt64(CPrefillBatch) * UInt64(FConfig.HiddenSize) * SizeOf(Single), False);
    FGpuBatchPLEProj := FComputeKernels.Device.CreateBuffer(
      UInt64(CPrefillBatch) * UInt64(FConfig.HiddenSizePerLayerInput * FConfig.NumHiddenLayers) * SizeOf(Single), False);
    FGpuBatchPLEGate := FComputeKernels.Device.CreateBuffer(
      UInt64(CPrefillBatch) * UInt64(FConfig.HiddenSizePerLayerInput) * SizeOf(Single), False);
    FGpuBatchEmbed := FComputeKernels.Device.CreateBuffer(
      UInt64(CPrefillBatch) * UInt64(FConfig.HiddenSize) * SizeOf(Single), True);
    FGpuBatchPLEInput := FComputeKernels.Device.CreateBuffer(
      UInt64(CPrefillBatch) * UInt64(FConfig.HiddenSizePerLayerInput * FConfig.NumHiddenLayers) * SizeOf(Single), True);

    if (FGpuBatchHidden.Buffer = 0) or (FGpuBatchNorm.Buffer = 0) or
       (FGpuBatchQ.Buffer = 0) or (FGpuBatchKStage.Buffer = 0) or
       (FGpuBatchVStage.Buffer = 0) or (FGpuBatchScores.Buffer = 0) or
       (FGpuBatchAttnOut.Buffer = 0) or (FGpuBatchMlpGate.Buffer = 0) or
       (FGpuBatchMlpUp.Buffer = 0) or (FGpuBatchTmp.Buffer = 0) or
       (FGpuBatchPLEProj.Buffer = 0) or (FGpuBatchPLEGate.Buffer = 0) or
       (FGpuBatchEmbed.Buffer = 0) or (FGpuBatchPLEInput.Buffer = 0) then
    begin
      // Batched prefill is an optimization: fall back to token-by-token
      Status('Batched prefill: buffer allocation failed, using per-token prefill', []);
      FUseBatchPrefill := False;
    end
    else
    begin
      SetLength(FBatchEmbedStage, CPrefillBatch * FConfig.HiddenSize);
      SetLength(FBatchPLEStage,
        CPrefillBatch * FConfig.HiddenSizePerLayerInput * FConfig.NumHiddenLayers);
      Status('Batched prefill: enabled (chunk %d)', [CPrefillBatch]);
    end;
  end
  else
    Status('Batched prefill: unavailable (no integer dot product)', []);

  Result := True;
end;

procedure TModel.DoLoadLayerScalars();
var
  LI: Integer;
  LInfo: TTensorInfo;
begin
  SetLength(FLayerScalars, FConfig.NumHiddenLayers);
  for LI := 0 to FConfig.NumHiddenLayers - 1 do
  begin
    if FWeights.FindTensor(LayerPrefix(LI) + 'layer_scalar', LInfo) then
      FLayerScalars[LI] := PSingle(FWeights.GetRawPointer(LInfo))^
    else
      FLayerScalars[LI] := 1.0;
  end;
end;

procedure TModel.DoUploadTokenInputsGpu(const ATokenId: Integer);
var
  LInfo: TTensorInfo;
  LPtr: Pointer;
  LTotalDim: Integer;
  LPLEDim: Integer;
  LSoftRow: PSingle;
  LEffToken: Integer;
begin
  // Soft-token substitution: embed + PLE come from the PAD token (HF
  // behavior); the soft row itself is staged and copied over FGpuHidden
  // AFTER the PLE precompute in DoForwardGpu
  LSoftRow := DoSoftRowAt(FPosition);
  LEffToken := ATokenId;
  if LSoftRow <> nil then
    LEffToken := CPadTokenId;

  // Embedding row: dequant + scale sqrt(hidden_size) into FHidden (CPU),
  // then upload to the host-visible staging buffer. The GPU copies it into
  // FGpuHidden as the first recorded dispatch.
  DoEmbeddingLookup(LEffToken);
  FComputeKernels.Device.UploadToBuffer(FGpuEmbed, @FHidden[0],
    UInt64(FConfig.HiddenSize) * SizeOf(Single));

  // PLE identity row: dequant + scale sqrt(256) = 16 into FPLEInput (CPU),
  // then upload. The GPU adds the context projection and applies the final
  // 1/sqrt(2) scale (see DoRecordPlePrecomputeGpu).
  LPLEDim := FConfig.HiddenSizePerLayerInput;
  LTotalDim := LPLEDim * FConfig.NumHiddenLayers;

  if not FWeights.FindTensor('model.language_model.embed_tokens_per_layer.weight', LInfo) then
    Exit;

  LPtr := FWeights.GetRawPointer(LInfo);
  if LInfo.OutputDtype = dkQ4_0 then
    Q4ToF32Buffer(
      Pointer(UIntPtr(LPtr) + UIntPtr(UInt64(LEffToken) * Q4ByteSize(LTotalDim))),
      @FPLEInput[0], Q4BlockCount(LTotalDim))
  else
    Move(
      Pointer(UIntPtr(LPtr) + UIntPtr(UInt64(LEffToken) * UInt64(LTotalDim) * SizeOf(Single)))^,
      FPLEInput[0], LTotalDim * SizeOf(Single));

  CpuScaleF32(@FPLEInput[0], Sqrt(Single(LPLEDim)), LTotalDim);

  FComputeKernels.Device.UploadToBuffer(FGpuPLEInput, @FPLEInput[0],
    UInt64(LTotalDim) * SizeOf(Single));

  // Stage the soft row for the post-PLE overwrite of FGpuHidden
  if LSoftRow <> nil then
    FComputeKernels.Device.UploadToBuffer(FGpuSoftStage, LSoftRow,
      UInt64(FConfig.HiddenSize) * SizeOf(Single));
end;

procedure TModel.DoRecordNormGpu(const ATensorName: string;
  const ASrcBuf: TVulkanBuffer; const ADstBuf: TVulkanBuffer;
  const ASize: Integer);
var
  LInfo: TTensorInfo;
  LLoc: TGpuTensorLoc;
begin
  if not FWeights.FindTensor(ATensorName, LInfo) then Exit;
  if not FGpuOfs.TryGetValue(ATensorName, LLoc) then Exit;

  FComputeKernels.RecordRmsNorm(ASrcBuf, FGpuWeights, LLoc.BaseOfs,
    LLoc.ByteSize, ADstBuf, UInt32(ASize), FConfig.RmsNormEps);
end;

procedure TModel.DoRecordNormRowsGpu(const ATensorName: string;
  const ADataBuf: TVulkanBuffer; const ARowSize: Integer;
  const ANumRows: Integer);
var
  LInfo: TTensorInfo;
  LLoc: TGpuTensorLoc;
begin
  if not FWeights.FindTensor(ATensorName, LInfo) then Exit;
  if not FGpuOfs.TryGetValue(ATensorName, LLoc) then Exit;

  FComputeKernels.RecordRmsNormRows(ADataBuf, FGpuWeights, LLoc.BaseOfs,
    LLoc.ByteSize, UInt32(ARowSize), UInt32(ANumRows), FConfig.RmsNormEps);
end;

procedure TModel.DoRecordProjGpu(const ATensorName: string;
  const AInputBuf: TVulkanBuffer; const AOutputBuf: TVulkanBuffer;
  const ARows: Integer; const ACols: Integer);
var
  LInfo: TTensorInfo;
  LLoc: TGpuTensorLoc;
begin
  if not FWeights.FindTensor(ATensorName, LInfo) then Exit;
  if not FGpuOfs.TryGetValue(ATensorName, LLoc) then Exit;

  if LLoc.IsQ4 then
    FComputeKernels.RecordMatVecQ4(FGpuWeights, LLoc.BaseOfs, LLoc.ScalesOfs,
      LLoc.ByteSize, AInputBuf, AOutputBuf, UInt32(ARows), UInt32(ACols))
  else
    FComputeKernels.RecordMatVecF32(FGpuWeights, LLoc.BaseOfs, LLoc.ByteSize,
      AInputBuf, AOutputBuf, UInt32(ARows), UInt32(ACols));
end;

procedure TModel.DoRecordPlePrecomputeGpu();
begin
  // Context component (bible PLE section):
  //   ple_ctx = per_layer_model_projection @ inputs_embeds   (FGpuHidden = embed here)
  //   ple_ctx *= hidden_size^-0.5
  //   ple_ctx = RMSNorm per 256-row (per_layer_projection_norm)
  //   FGpuPLEInput = (FGpuPLEInput + ple_ctx) * 2^-0.5
  DoRecordProjGpu('model.language_model.per_layer_model_projection.weight',
    FGpuHidden, FGpuPLEProj,
    FConfig.HiddenSizePerLayerInput * FConfig.NumHiddenLayers, FConfig.HiddenSize);
  FComputeKernels.RecordBarrier();

  FComputeKernels.RecordScale(FGpuPLEProj,
    UInt32(FConfig.HiddenSizePerLayerInput * FConfig.NumHiddenLayers),
    1.0 / Sqrt(Single(FConfig.HiddenSize)));
  FComputeKernels.RecordBarrier();

  DoRecordNormRowsGpu('model.language_model.per_layer_projection_norm.weight',
    FGpuPLEProj, FConfig.HiddenSizePerLayerInput, FConfig.NumHiddenLayers);
  FComputeKernels.RecordBarrier();

  FComputeKernels.RecordAdd(FGpuPLEInput, FGpuPLEProj,
    UInt32(FConfig.HiddenSizePerLayerInput * FConfig.NumHiddenLayers));
  FComputeKernels.RecordBarrier();

  FComputeKernels.RecordScale(FGpuPLEInput,
    UInt32(FConfig.HiddenSizePerLayerInput * FConfig.NumHiddenLayers),
    1.0 / Sqrt(2.0));
  FComputeKernels.RecordBarrier();
end;

procedure TModel.DoRecordAttentionGpu(const ALayerIdx: Integer);
var
  LPrefix: string;
  LAttnKind: TAttentionKind;
  LHeadDim: Integer;
  LKvSlotIdx: Integer;
  LRopeTheta: Single;
  LRopePartial: Single;
  LRotaryDim: Integer;
  LRotAngles: Integer;
  LHasKv: Boolean;
  LKvStride: Integer;
  LRingEntries: Integer;
  LRingIdx: Integer;
  LSeqLen: Integer;
  LPush: TAttnPushConstants;
begin
  LPrefix := LayerPrefix(ALayerIdx);
  LAttnKind := FConfig.LayerTypes[ALayerIdx];
  LKvSlotIdx := FConfig.KvCacheMap[ALayerIdx];
  LHasKv := ALayerIdx < (FConfig.NumHiddenLayers - FConfig.NumKvSharedLayers);

  if LAttnKind = akFull then
  begin
    LHeadDim := FConfig.GlobalHeadDim;
    LRopeTheta := FConfig.RopeFull.Theta;
    LRopePartial := FConfig.RopeFull.PartialRotaryFactor;
    LRingEntries := CGpuMaxSeq;
  end
  else
  begin
    LHeadDim := FConfig.HeadDim;
    LRopeTheta := FConfig.RopeSliding.Theta;
    LRopePartial := 1.0;
    LRingEntries := FConfig.SlidingWindow + CPrefillBatch;
  end;

  // Rotating angle pairs -- must match CpuApplyRoPE exactly
  LRotaryDim := Round(LHeadDim * LRopePartial);
  if (LRotaryDim mod 2) <> 0 then
    Dec(LRotaryDim);
  LRotAngles := LRotaryDim div 2;

  LKvStride := FConfig.NumKeyValueHeads * LHeadDim;

  // Append slot for THIS position; enumeration below is position-window
  // based: seqLen entries ending at FPosition, starting at cachePos0.
  // Sliding: window entries max. Full: ring = CGpuMaxSeq, so cachePos0 = 0
  // and behavior is byte-identical to before.
  LRingIdx := FPosition mod LRingEntries;
  if LAttnKind = akFull then
    LSeqLen := Min(FPosition + 1, LRingEntries)
  else
    LSeqLen := Min(FPosition + 1, FConfig.SlidingWindow);

  // Stage 1: q/k/v projections -- all read FGpuNorm, write disjoint buffers
  DoRecordProjGpu(LPrefix + 'self_attn.q_proj.weight', FGpuNorm, FGpuQ,
    FConfig.NumAttentionHeads * LHeadDim, FConfig.HiddenSize);
  if LHasKv then
  begin
    DoRecordProjGpu(LPrefix + 'self_attn.k_proj.weight', FGpuNorm, FGpuKStage,
      FConfig.NumKeyValueHeads * LHeadDim, FConfig.HiddenSize);
    DoRecordProjGpu(LPrefix + 'self_attn.v_proj.weight', FGpuNorm, FGpuVStage,
      FConfig.NumKeyValueHeads * LHeadDim, FConfig.HiddenSize);
  end;
  FComputeKernels.RecordBarrier();

  // Stage 2: per-head norms -- each in-place on its own buffer
  DoRecordNormRowsGpu(LPrefix + 'self_attn.q_norm.weight', FGpuQ,
    LHeadDim, FConfig.NumAttentionHeads);
  if LHasKv then
  begin
    DoRecordNormRowsGpu(LPrefix + 'self_attn.k_norm.weight', FGpuKStage,
      LHeadDim, FConfig.NumKeyValueHeads);
    // Weight-free per-head V norm (no gamma; dummy weight binding unused)
    FComputeKernels.RecordRmsNormRows(FGpuVStage, FGpuVStage,
      UInt32(LHeadDim), UInt32(FConfig.NumKeyValueHeads), FConfig.RmsNormEps,
      False, 0);
  end;
  FComputeKernels.RecordBarrier();

  // Stage 3: RoPE Q and K -- independent buffers
  FComputeKernels.RecordRoPE(FGpuQ, UInt32(FPosition), UInt32(LHeadDim),
    UInt32(LRotAngles), UInt32(FConfig.NumAttentionHeads), LRopeTheta);
  if LHasKv then
    FComputeKernels.RecordRoPE(FGpuKStage, UInt32(FPosition), UInt32(LHeadDim),
      UInt32(LRotAngles), UInt32(FConfig.NumKeyValueHeads), LRopeTheta);
  FComputeKernels.RecordBarrier();

  // Stage 4: ring append -- copies into disjoint cache buffers
  if LHasKv then
  begin
    FComputeKernels.RecordCopy(FGpuKvK[LKvSlotIdx], FGpuKStage,
      UInt32(LKvStride), 0, UInt32(LRingIdx * LKvStride));
    FComputeKernels.RecordCopy(FGpuKvV[LKvSlotIdx], FGpuVStage,
      UInt32(LKvStride), 0, UInt32(LRingIdx * LKvStride));
    FComputeKernels.RecordBarrier();
  end;

  // Stage 5-7: scores -> softmax -> weighted value sum (strict chain)
  LPush.headDim := UInt32(LHeadDim);
  LPush.numHeads := UInt32(FConfig.NumAttentionHeads);
  LPush.numKvHeads := UInt32(FConfig.NumKeyValueHeads);
  LPush.kvStride := UInt32(LKvStride);
  LPush.seqLen := UInt32(LSeqLen);
  LPush.ringSize := UInt32(LRingEntries);
  LPush.cachePos0 := UInt32((FPosition - LSeqLen + 1) mod LRingEntries);

  FComputeKernels.RecordAttnScores(FGpuQ, FGpuKvK[LKvSlotIdx], FGpuScores, LPush);
  FComputeKernels.RecordBarrier();
  FComputeKernels.RecordSoftmaxRows(FGpuScores, UInt32(LSeqLen),
    UInt32(FConfig.NumAttentionHeads));
  FComputeKernels.RecordBarrier();
  FComputeKernels.RecordAttnOut(FGpuScores, FGpuKvV[LKvSlotIdx], FGpuAttnOut, LPush);
  FComputeKernels.RecordBarrier();

  // O projection -> FGpuTmp (barrier added by the caller)
  DoRecordProjGpu(LPrefix + 'self_attn.o_proj.weight', FGpuAttnOut, FGpuTmp,
    FConfig.HiddenSize, FConfig.NumAttentionHeads * LHeadDim);
end;

procedure TModel.DoRecordMlpGpu(const ALayerIdx: Integer);
var
  LPrefix: string;
begin
  LPrefix := LayerPrefix(ALayerIdx);

  // GeGLU: down( gelu(gate(x)) * up(x) ) -> FGpuTmp
  // gate and up both read FGpuNorm and write disjoint buffers -- no barrier
  // between them
  DoRecordProjGpu(LPrefix + 'mlp.gate_proj.weight', FGpuNorm, FGpuMlpGate,
    FConfig.IntermediateSize, FConfig.HiddenSize);
  DoRecordProjGpu(LPrefix + 'mlp.up_proj.weight', FGpuNorm, FGpuMlpUp,
    FConfig.IntermediateSize, FConfig.HiddenSize);
  FComputeKernels.RecordBarrier();

  FComputeKernels.RecordGeGLU(FGpuMlpGate, FGpuMlpUp,
    UInt32(FConfig.IntermediateSize), 0);
  FComputeKernels.RecordBarrier();

  DoRecordProjGpu(LPrefix + 'mlp.down_proj.weight', FGpuMlpGate, FGpuTmp,
    FConfig.HiddenSize, FConfig.IntermediateSize);
end;

procedure TModel.DoRecordPleGpu(const ALayerIdx: Integer);
var
  LPrefix: string;
  LPLEDim: Integer;
begin
  LPrefix := LayerPrefix(ALayerIdx);
  LPLEDim := FConfig.HiddenSizePerLayerInput;

  // PLE block (bible): gate -> gelu -> mul(precomputed slice) -> project
  //                    -> norm -> residual
  DoRecordProjGpu(LPrefix + 'per_layer_input_gate.weight', FGpuHidden,
    FGpuPLEGate, LPLEDim, FConfig.HiddenSize);
  FComputeKernels.RecordBarrier();

  // Fused GeGLU with this layer's precomputed PLE input slice
  FComputeKernels.RecordGeGLU(FGpuPLEGate, FGpuPLEInput, UInt32(LPLEDim),
    UInt32(ALayerIdx * LPLEDim));
  FComputeKernels.RecordBarrier();

  DoRecordProjGpu(LPrefix + 'per_layer_projection.weight', FGpuPLEGate,
    FGpuTmp, FConfig.HiddenSize, LPLEDim);
  FComputeKernels.RecordBarrier();

  // Post-PLE norm in-place on FGpuTmp
  DoRecordNormGpu(LPrefix + 'post_per_layer_input_norm.weight', FGpuTmp,
    FGpuTmp, FConfig.HiddenSize);
  FComputeKernels.RecordBarrier();

  // PLE residual
  FComputeKernels.RecordAdd(FGpuHidden, FGpuTmp, UInt32(FConfig.HiddenSize));
  FComputeKernels.RecordBarrier();
end;

procedure TModel.DoRecordDecoderLayerGpu(const ALayerIdx: Integer);
var
  LPrefix: string;
  LTsBase: Integer;
begin
  LPrefix := LayerPrefix(ALayerIdx);

  // Timestamp base for the two instrumented profile layers (-1 = no stamps)
  LTsBase := -1;
  if ALayerIdx = CProfLayerFull then
    LTsBase := CTsLFullStart
  else if ALayerIdx = CProfLayerSliding then
    LTsBase := CTsLSlidStart;

  if LTsBase >= 0 then
    FComputeKernels.Device.RecordTimestamp(UInt32(LTsBase));

  // --- Attention block (sandwich norm) ---
  DoRecordNormGpu(LPrefix + 'input_layernorm.weight', FGpuHidden, FGpuNorm,
    FConfig.HiddenSize);
  FComputeKernels.RecordBarrier();
  DoRecordAttentionGpu(ALayerIdx);
  FComputeKernels.RecordBarrier();
  DoRecordNormGpu(LPrefix + 'post_attention_layernorm.weight', FGpuTmp,
    FGpuTmp, FConfig.HiddenSize);
  FComputeKernels.RecordBarrier();
  FComputeKernels.RecordAdd(FGpuHidden, FGpuTmp, UInt32(FConfig.HiddenSize));
  FComputeKernels.RecordBarrier();

  if LTsBase >= 0 then
    FComputeKernels.Device.RecordTimestamp(UInt32(LTsBase) + 1);

  // --- MLP block (sandwich norm) ---
  DoRecordNormGpu(LPrefix + 'pre_feedforward_layernorm.weight', FGpuHidden,
    FGpuNorm, FConfig.HiddenSize);
  FComputeKernels.RecordBarrier();
  DoRecordMlpGpu(ALayerIdx);
  FComputeKernels.RecordBarrier();
  DoRecordNormGpu(LPrefix + 'post_feedforward_layernorm.weight', FGpuTmp,
    FGpuTmp, FConfig.HiddenSize);
  FComputeKernels.RecordBarrier();
  FComputeKernels.RecordAdd(FGpuHidden, FGpuTmp, UInt32(FConfig.HiddenSize));
  FComputeKernels.RecordBarrier();

  if LTsBase >= 0 then
    FComputeKernels.Device.RecordTimestamp(UInt32(LTsBase) + 2);

  // --- PLE block (after FFN residual; ends with its own barrier) ---
  DoRecordPleGpu(ALayerIdx);

  // --- Layer output scaling (applies to the whole hidden state) ---
  FComputeKernels.RecordScale(FGpuHidden, UInt32(FConfig.HiddenSize),
    FLayerScalars[ALayerIdx]);
  FComputeKernels.RecordBarrier();

  if LTsBase >= 0 then
    FComputeKernels.Device.RecordTimestamp(UInt32(LTsBase) + 3);
end;

procedure TModel.DoRecordFinalGpu();
var
  LInfo: TTensorInfo;
begin
  // Final norm -> tied lm_head projection into the host-visible logits buffer
  DoRecordNormGpu('model.language_model.norm.weight', FGpuHidden, FGpuNorm,
    FConfig.HiddenSize);
  FComputeKernels.RecordBarrier();

  if FWeights.FindTensor('model.language_model.lm_head.weight', LInfo) then
    DoRecordProjGpu('model.language_model.lm_head.weight', FGpuNorm,
      FGpuLogits, FConfig.VocabSize, FConfig.HiddenSize)
  else if FConfig.TieWordEmbeddings then
    DoRecordProjGpu('model.language_model.embed_tokens.weight', FGpuNorm,
      FGpuLogits, FConfig.VocabSize, FConfig.HiddenSize);
end;

procedure TModel.DoForwardGpu(const ATokenId: Integer);
var
  LI: Integer;
  LSw: TStopwatch;
  LTs: TArray<UInt64>;
  LPeriodMs: Double;
begin
  if FPosition >= CGpuMaxSeq then
  begin
    GetErrors().Add(esError, CMD_ERR_FORWARD,
      'Position %d exceeds GPU KV cache capacity %d', [FPosition, CGpuMaxSeq]);
    Exit;
  end;

  // --- Phase: upload (pool reset + host-visible input writes) ---
  LSw := TStopwatch.StartNew();

  // Reclaim all descriptor sets from the previous token
  FComputeKernels.Shaders.ResetDescriptorPool();

  // Host writes BEFORE recording -- vkQueueSubmit makes them visible
  DoUploadTokenInputsGpu(ATokenId);

  FProfUploadMs := FProfUploadMs + LSw.Elapsed.TotalMilliseconds;

  // --- Phase: record (CPU command buffer build) ---
  LSw := TStopwatch.StartNew();

  // Record the ENTIRE token forward pass into one command buffer
  FComputeKernels.Device.BeginCommands();

  // GPU timestamps: reset the query pool, stamp the start
  FComputeKernels.Device.RecordResetTimestamps();
  FComputeKernels.Device.RecordTimestamp(CTsStart);

  // Embedding -> hidden state
  FComputeKernels.RecordCopy(FGpuHidden, FGpuEmbed, UInt32(FConfig.HiddenSize));
  FComputeKernels.RecordBarrier();

  // Soft-token substitution BEFORE PLE: HF projects the merged embeddings
  // (modeling_gemma4.py line 1684), so the PLE context projection must see
  // the raw soft row; only the PLE identity lookup comes from the PAD token
  if DoSoftRowAt(FPosition) <> nil then
  begin
    FComputeKernels.RecordCopy(FGpuHidden, FGpuSoftStage,
      UInt32(FConfig.HiddenSize));
    FComputeKernels.RecordBarrier();
  end;

  // PLE precompute (context projection; reads FGpuHidden)
  DoRecordPlePrecomputeGpu();

  FComputeKernels.Device.RecordTimestamp(CTsPlePre);

  // 42 decoder layers
  for LI := 0 to FConfig.NumHiddenLayers - 1 do
    DoRecordDecoderLayerGpu(LI);

  FComputeKernels.Device.RecordTimestamp(CTsLayersEnd);

  // Final norm + lm_head
  DoRecordFinalGpu();

  FComputeKernels.Device.RecordTimestamp(CTsEnd);

  FComputeKernels.Device.EndCommands();

  FProfRecordMs := FProfRecordMs + LSw.Elapsed.TotalMilliseconds;

  // --- Phase: fence (GPU execution) ---
  LSw := TStopwatch.StartNew();

  // THE ONLY fence wait of the token
  FComputeKernels.Device.SubmitAndWait();

  FProfFenceMs := FProfFenceMs + LSw.Elapsed.TotalMilliseconds;

  // --- Phase: download (logits readback) ---
  LSw := TStopwatch.StartNew();

  // Download logits (1 MB). RAW logits -- softcap now happens in DoSample,
  // applied only to the top-K survivors (monotonicity preserves the set).
  FComputeKernels.Device.DownloadFromBuffer(FGpuLogits, @FLogits[0],
    UInt64(FConfig.VocabSize) * SizeOf(Single));

  FProfDownloadMs := FProfDownloadMs + LSw.Elapsed.TotalMilliseconds;

  // Read back GPU timestamps and accumulate the internal split
  if FComputeKernels.Device.FetchTimestamps(LTs) then
  begin
    LPeriodMs := FComputeKernels.Device.TimestampPeriod / 1.0E6; // ns/tick -> ms/tick
    FProfGpuMs[0] := FProfGpuMs[0] + (LTs[CTsPlePre] - LTs[CTsStart]) * LPeriodMs;
    FProfGpuMs[1] := FProfGpuMs[1] + (LTs[CTsLFullAttn] - LTs[CTsLFullStart]) * LPeriodMs;
    FProfGpuMs[2] := FProfGpuMs[2] + (LTs[CTsLFullMlp] - LTs[CTsLFullAttn]) * LPeriodMs;
    FProfGpuMs[3] := FProfGpuMs[3] + (LTs[CTsLFullEnd] - LTs[CTsLFullMlp]) * LPeriodMs;
    FProfGpuMs[4] := FProfGpuMs[4] + (LTs[CTsLSlidAttn] - LTs[CTsLSlidStart]) * LPeriodMs;
    FProfGpuMs[5] := FProfGpuMs[5] + (LTs[CTsLSlidMlp] - LTs[CTsLSlidAttn]) * LPeriodMs;
    FProfGpuMs[6] := FProfGpuMs[6] + (LTs[CTsLSlidEnd] - LTs[CTsLSlidMlp]) * LPeriodMs;
    FProfGpuMs[7] := FProfGpuMs[7] + (LTs[CTsLayersEnd] - LTs[CTsPlePre]) * LPeriodMs;
    FProfGpuMs[8] := FProfGpuMs[8] + (LTs[CTsEnd] - LTs[CTsLayersEnd]) * LPeriodMs;
  end;

  Inc(FPosition);
  Inc(FProfTokens);
end;

procedure TModel.DoUploadBatchInputsGpu(const ATokens: TArray<Integer>;
  const AFirst: Integer; const ACount: Integer);
var
  LM: Integer;
  LInfo: TTensorInfo;
  LPtr: Pointer;
  LTotalDim: Integer;
  LPLEDim: Integer;
  LTokenId: Integer;
  LSoftRow: PSingle;
  LHasSoft: Boolean;
begin
  LPLEDim := FConfig.HiddenSizePerLayerInput;
  LTotalDim := LPLEDim * FConfig.NumHiddenLayers;

  if not FWeights.FindTensor('model.language_model.embed_tokens_per_layer.weight', LInfo) then
    Exit;
  LPtr := FWeights.GetRawPointer(LInfo);

  LHasSoft := False;
  for LM := 0 to ACount - 1 do
  begin
    LTokenId := ATokens[AFirst + LM];

    // Soft-token substitution: embed + PLE from PAD; pack the soft row for
    // the post-PLE overwrite of FGpuBatchHidden (DoForwardGpuBatch)
    LSoftRow := DoSoftRowAt(FPosition + LM);
    if LSoftRow <> nil then
    begin
      LTokenId := CPadTokenId;
      Move(LSoftRow^, FBatchSoftStage[LM * FConfig.HiddenSize],
        FConfig.HiddenSize * SizeOf(Single));
      LHasSoft := True;
    end;

    // Embedding row (dequant + sqrt(hidden) scale) via the existing helper,
    // then copy into this position's staging row
    DoEmbeddingLookup(LTokenId);
    Move(FHidden[0], FBatchEmbedStage[LM * FConfig.HiddenSize],
      FConfig.HiddenSize * SizeOf(Single));

    // PLE identity row (same dequant + sqrt(256) scale as
    // DoUploadTokenInputsGpu) straight into the staging row
    if LInfo.OutputDtype = dkQ4_0 then
      Q4ToF32Buffer(
        Pointer(UIntPtr(LPtr) + UIntPtr(UInt64(LTokenId) * Q4ByteSize(LTotalDim))),
        @FBatchPLEStage[LM * LTotalDim], Q4BlockCount(LTotalDim))
    else
      Move(
        Pointer(UIntPtr(LPtr) + UIntPtr(UInt64(LTokenId) * UInt64(LTotalDim) * SizeOf(Single)))^,
        FBatchPLEStage[LM * LTotalDim], LTotalDim * SizeOf(Single));

    CpuScaleF32(@FBatchPLEStage[LM * LTotalDim], Sqrt(Single(LPLEDim)), LTotalDim);
  end;

  FComputeKernels.Device.UploadToBuffer(FGpuBatchEmbed, @FBatchEmbedStage[0],
    UInt64(ACount) * UInt64(FConfig.HiddenSize) * SizeOf(Single));
  FComputeKernels.Device.UploadToBuffer(FGpuBatchPLEInput, @FBatchPLEStage[0],
    UInt64(ACount) * UInt64(LTotalDim) * SizeOf(Single));
  if LHasSoft then
    FComputeKernels.Device.UploadToBuffer(FGpuSoftStage, @FBatchSoftStage[0],
      UInt64(ACount) * UInt64(FConfig.HiddenSize) * SizeOf(Single));
end;

procedure TModel.DoRecordNormGpuBatch(const ATensorName: string;
  const ASrcBuf: TVulkanBuffer; const ADstBuf: TVulkanBuffer;
  const ASize: Integer; const ABatch: Integer);
begin
  // rms_norm_rows is in-place: copy src rows into dst first when they are
  // different buffers, then norm dst rows against the shared gamma
  if ASrcBuf.Buffer <> ADstBuf.Buffer then
  begin
    FComputeKernels.RecordCopy(ADstBuf, ASrcBuf, UInt32(ABatch * ASize));
    FComputeKernels.RecordBarrier();
  end;
  DoRecordNormRowsGpu(ATensorName, ADstBuf, ASize, ABatch);
end;

procedure TModel.DoRecordProjGpuBatch(const ATensorName: string;
  const AInputBuf: TVulkanBuffer; const AOutputBuf: TVulkanBuffer;
  const ARows: Integer; const ACols: Integer; const ABatch: Integer);
var
  LInfo: TTensorInfo;
  LLoc: TGpuTensorLoc;
begin
  if not FWeights.FindTensor(ATensorName, LInfo) then Exit;
  if not FGpuOfs.TryGetValue(ATensorName, LLoc) then Exit;

  if LLoc.IsQ4 then
    FComputeKernels.RecordMatMatQ4Q8(FGpuWeights, LLoc.BaseOfs, LLoc.ScalesOfs,
      LLoc.ByteSize, AInputBuf, AOutputBuf, UInt32(ARows), UInt32(ACols),
      UInt32(ABatch))
  else
    // Not expected for this model (all projections are Q4); fail loud so a
    // silent slow path never masks a packer regression
    GetErrors().Add(esError, CMD_ERR_GPU,
      'Batched projection requires Q4 tensor: %s', [ATensorName]);
end;

procedure TModel.DoRecordPlePrecomputeGpuBatch(const ABatch: Integer);
var
  LTotalDim: Integer;
begin
  LTotalDim := FConfig.HiddenSizePerLayerInput * FConfig.NumHiddenLayers;

  // ple_ctx = per_layer_model_projection @ embeds  -> [batch x 10752]
  DoRecordProjGpuBatch('model.language_model.per_layer_model_projection.weight',
    FGpuBatchHidden, FGpuBatchPLEProj, LTotalDim, FConfig.HiddenSize, ABatch);
  FComputeKernels.RecordBarrier();

  FComputeKernels.RecordScale(FGpuBatchPLEProj, UInt32(ABatch * LTotalDim),
    1.0 / Sqrt(Single(FConfig.HiddenSize)));
  FComputeKernels.RecordBarrier();

  // Per-256 row norm: [batch x 42] rows of 256, contiguous
  DoRecordNormRowsGpu('model.language_model.per_layer_projection_norm.weight',
    FGpuBatchPLEProj, FConfig.HiddenSizePerLayerInput,
    ABatch * FConfig.NumHiddenLayers);
  FComputeKernels.RecordBarrier();

  FComputeKernels.RecordAdd(FGpuBatchPLEInput, FGpuBatchPLEProj,
    UInt32(ABatch * LTotalDim));
  FComputeKernels.RecordBarrier();

  FComputeKernels.RecordScale(FGpuBatchPLEInput, UInt32(ABatch * LTotalDim),
    1.0 / Sqrt(2.0));
  FComputeKernels.RecordBarrier();
end;

procedure TModel.DoRecordAttentionGpuBatch(const ALayerIdx: Integer;
  const ABatch: Integer);
var
  LPrefix: string;
  LAttnKind: TAttentionKind;
  LHeadDim: Integer;
  LKvSlotIdx: Integer;
  LRopeTheta: Single;
  LRopePartial: Single;
  LRotaryDim: Integer;
  LRotAngles: Integer;
  LHasKv: Boolean;
  LKvStride: Integer;
  LRingEntries: Integer;
  LWindowSize: Integer;
  LRowPitch: Integer;
  LPush: TAttnBatchPushConstants;
begin
  LPrefix := LayerPrefix(ALayerIdx);
  LAttnKind := FConfig.LayerTypes[ALayerIdx];
  LKvSlotIdx := FConfig.KvCacheMap[ALayerIdx];
  LHasKv := ALayerIdx < (FConfig.NumHiddenLayers - FConfig.NumKvSharedLayers);

  if LAttnKind = akFull then
  begin
    LHeadDim := FConfig.GlobalHeadDim;
    LRopeTheta := FConfig.RopeFull.Theta;
    LRopePartial := FConfig.RopeFull.PartialRotaryFactor;
    LRingEntries := CGpuMaxSeq;
    LWindowSize := 0;                       // 0 = full causal
    LRowPitch := FPosition + ABatch;        // longest query row
  end
  else
  begin
    LHeadDim := FConfig.HeadDim;
    LRopeTheta := FConfig.RopeSliding.Theta;
    LRopePartial := 1.0;
    LRingEntries := FConfig.SlidingWindow + CPrefillBatch;
    LWindowSize := FConfig.SlidingWindow;
    LRowPitch := Min(FPosition + ABatch, FConfig.SlidingWindow);
  end;

  LRotaryDim := Round(LHeadDim * LRopePartial);
  if (LRotaryDim mod 2) <> 0 then
    Dec(LRotaryDim);
  LRotAngles := LRotaryDim div 2;

  LKvStride := FConfig.NumKeyValueHeads * LHeadDim;

  // Stage 1: batched q/k/v projections from FGpuBatchNorm
  DoRecordProjGpuBatch(LPrefix + 'self_attn.q_proj.weight', FGpuBatchNorm,
    FGpuBatchQ, FConfig.NumAttentionHeads * LHeadDim, FConfig.HiddenSize, ABatch);
  if LHasKv then
  begin
    DoRecordProjGpuBatch(LPrefix + 'self_attn.k_proj.weight', FGpuBatchNorm,
      FGpuBatchKStage, FConfig.NumKeyValueHeads * LHeadDim, FConfig.HiddenSize, ABatch);
    DoRecordProjGpuBatch(LPrefix + 'self_attn.v_proj.weight', FGpuBatchNorm,
      FGpuBatchVStage, FConfig.NumKeyValueHeads * LHeadDim, FConfig.HiddenSize, ABatch);
  end;
  FComputeKernels.RecordBarrier();

  // Stage 2: per-head norms -- rows scale by batch
  DoRecordNormRowsGpu(LPrefix + 'self_attn.q_norm.weight', FGpuBatchQ,
    LHeadDim, ABatch * FConfig.NumAttentionHeads);
  if LHasKv then
  begin
    DoRecordNormRowsGpu(LPrefix + 'self_attn.k_norm.weight', FGpuBatchKStage,
      LHeadDim, ABatch * FConfig.NumKeyValueHeads);
    FComputeKernels.RecordRmsNormRows(FGpuBatchVStage, FGpuBatchVStage,
      UInt32(LHeadDim), UInt32(ABatch * FConfig.NumKeyValueHeads),
      FConfig.RmsNormEps, False, 0);
  end;
  FComputeKernels.RecordBarrier();

  // Stage 3: batched RoPE (per-position angles)
  FComputeKernels.RecordRoPEBatch(FGpuBatchQ, UInt32(FPosition), UInt32(ABatch),
    UInt32(LHeadDim), UInt32(LRotAngles), UInt32(FConfig.NumAttentionHeads),
    LRopeTheta);
  if LHasKv then
    FComputeKernels.RecordRoPEBatch(FGpuBatchKStage, UInt32(FPosition),
      UInt32(ABatch), UInt32(LHeadDim), UInt32(LRotAngles),
      UInt32(FConfig.NumKeyValueHeads), LRopeTheta);
  FComputeKernels.RecordBarrier();

  // Stage 4: batched ring append
  if LHasKv then
  begin
    FComputeKernels.RecordKvAppend(FGpuKvK[LKvSlotIdx], FGpuBatchKStage,
      UInt32(LKvStride), UInt32(ABatch), UInt32(FPosition), UInt32(LRingEntries));
    FComputeKernels.RecordKvAppend(FGpuKvV[LKvSlotIdx], FGpuBatchVStage,
      UInt32(LKvStride), UInt32(ABatch), UInt32(FPosition), UInt32(LRingEntries));
    FComputeKernels.RecordBarrier();
  end;

  // Stage 5-7: causal scores -> softmax -> weighted value sum
  LPush.headDim := UInt32(LHeadDim);
  LPush.numHeads := UInt32(FConfig.NumAttentionHeads);
  LPush.numKvHeads := UInt32(FConfig.NumKeyValueHeads);
  LPush.kvStride := UInt32(LKvStride);
  LPush.ringEntries := UInt32(LRingEntries);
  LPush.basePosition := UInt32(FPosition);
  LPush.batch := UInt32(ABatch);
  LPush.windowSize := UInt32(LWindowSize);
  LPush.rowPitch := UInt32(LRowPitch);

  FComputeKernels.RecordAttnScoresBatch(FGpuBatchQ, FGpuKvK[LKvSlotIdx],
    FGpuBatchScores, LPush);
  FComputeKernels.RecordBarrier();
  FComputeKernels.RecordSoftmaxRows(FGpuBatchScores, UInt32(LRowPitch),
    UInt32(ABatch * FConfig.NumAttentionHeads));
  FComputeKernels.RecordBarrier();
  FComputeKernels.RecordAttnOutBatch(FGpuBatchScores, FGpuKvV[LKvSlotIdx],
    FGpuBatchAttnOut, LPush);
  FComputeKernels.RecordBarrier();

  // Batched O projection -> FGpuBatchTmp (barrier added by the caller)
  DoRecordProjGpuBatch(LPrefix + 'self_attn.o_proj.weight', FGpuBatchAttnOut,
    FGpuBatchTmp, FConfig.HiddenSize, FConfig.NumAttentionHeads * LHeadDim, ABatch);
end;

procedure TModel.DoRecordMlpGpuBatch(const ALayerIdx: Integer;
  const ABatch: Integer);
var
  LPrefix: string;
begin
  LPrefix := LayerPrefix(ALayerIdx);

  DoRecordProjGpuBatch(LPrefix + 'mlp.gate_proj.weight', FGpuBatchNorm,
    FGpuBatchMlpGate, FConfig.IntermediateSize, FConfig.HiddenSize, ABatch);
  DoRecordProjGpuBatch(LPrefix + 'mlp.up_proj.weight', FGpuBatchNorm,
    FGpuBatchMlpUp, FConfig.IntermediateSize, FConfig.HiddenSize, ABatch);
  FComputeKernels.RecordBarrier();

  // Dense batched GeGLU: gate and up rows are both contiguous [B x 10240]
  FComputeKernels.RecordGeGLU(FGpuBatchMlpGate, FGpuBatchMlpUp,
    UInt32(ABatch * FConfig.IntermediateSize), 0);
  FComputeKernels.RecordBarrier();

  DoRecordProjGpuBatch(LPrefix + 'mlp.down_proj.weight', FGpuBatchMlpGate,
    FGpuBatchTmp, FConfig.HiddenSize, FConfig.IntermediateSize, ABatch);
end;

procedure TModel.DoRecordPleGpuBatch(const ALayerIdx: Integer;
  const ABatch: Integer);
var
  LPrefix: string;
  LPLEDim: Integer;
  LTotalDim: Integer;
begin
  LPrefix := LayerPrefix(ALayerIdx);
  LPLEDim := FConfig.HiddenSizePerLayerInput;
  LTotalDim := LPLEDim * FConfig.NumHiddenLayers;

  DoRecordProjGpuBatch(LPrefix + 'per_layer_input_gate.weight', FGpuBatchHidden,
    FGpuBatchPLEGate, LPLEDim, FConfig.HiddenSize, ABatch);
  FComputeKernels.RecordBarrier();

  // Strided GeGLU: gate rows [B x 256] contiguous; the PLE input slice for
  // position m sits at m*10752 + layer*256 (position-major layout)
  FComputeKernels.RecordGeGLURows(FGpuBatchPLEGate, FGpuBatchPLEInput,
    UInt32(LPLEDim), UInt32(ABatch), UInt32(ALayerIdx * LPLEDim),
    UInt32(LTotalDim));
  FComputeKernels.RecordBarrier();

  DoRecordProjGpuBatch(LPrefix + 'per_layer_projection.weight', FGpuBatchPLEGate,
    FGpuBatchTmp, FConfig.HiddenSize, LPLEDim, ABatch);
  FComputeKernels.RecordBarrier();

  // Batch path norms [B x 2560] rows in place -- numerically identical to
  // the single-token DoRecordNormGpu (same weighted RMS per row, same eps)
  DoRecordNormRowsGpu(LPrefix + 'post_per_layer_input_norm.weight', FGpuBatchTmp,
    FConfig.HiddenSize, ABatch);
  FComputeKernels.RecordBarrier();

  FComputeKernels.RecordAdd(FGpuBatchHidden, FGpuBatchTmp,
    UInt32(ABatch * FConfig.HiddenSize));
  FComputeKernels.RecordBarrier();
end;

procedure TModel.DoRecordDecoderLayerGpuBatch(const ALayerIdx: Integer;
  const ABatch: Integer);
var
  LPrefix: string;
  LTsBase: Integer;
begin
  LPrefix := LayerPrefix(ALayerIdx);

  // Timestamp base for the two instrumented profile layers (-1 = no stamps)
  LTsBase := -1;
  if ALayerIdx = CProfLayerFull then
    LTsBase := CTsLFullStart
  else if ALayerIdx = CProfLayerSliding then
    LTsBase := CTsLSlidStart;

  if LTsBase >= 0 then
    FComputeKernels.Device.RecordTimestamp(UInt32(LTsBase));

  // --- Attention block (sandwich norm) ---
  DoRecordNormGpuBatch(LPrefix + 'input_layernorm.weight', FGpuBatchHidden,
    FGpuBatchNorm, FConfig.HiddenSize, ABatch);
  FComputeKernels.RecordBarrier();
  DoRecordAttentionGpuBatch(ALayerIdx, ABatch);
  FComputeKernels.RecordBarrier();
  DoRecordNormRowsGpu(LPrefix + 'post_attention_layernorm.weight', FGpuBatchTmp,
    FConfig.HiddenSize, ABatch);
  FComputeKernels.RecordBarrier();
  FComputeKernels.RecordAdd(FGpuBatchHidden, FGpuBatchTmp,
    UInt32(ABatch * FConfig.HiddenSize));
  FComputeKernels.RecordBarrier();

  if LTsBase >= 0 then
    FComputeKernels.Device.RecordTimestamp(UInt32(LTsBase) + 1);

  // --- MLP block (sandwich norm) ---
  DoRecordNormGpuBatch(LPrefix + 'pre_feedforward_layernorm.weight',
    FGpuBatchHidden, FGpuBatchNorm, FConfig.HiddenSize, ABatch);
  FComputeKernels.RecordBarrier();
  DoRecordMlpGpuBatch(ALayerIdx, ABatch);
  FComputeKernels.RecordBarrier();
  DoRecordNormRowsGpu(LPrefix + 'post_feedforward_layernorm.weight', FGpuBatchTmp,
    FConfig.HiddenSize, ABatch);
  FComputeKernels.RecordBarrier();
  FComputeKernels.RecordAdd(FGpuBatchHidden, FGpuBatchTmp,
    UInt32(ABatch * FConfig.HiddenSize));
  FComputeKernels.RecordBarrier();

  if LTsBase >= 0 then
    FComputeKernels.Device.RecordTimestamp(UInt32(LTsBase) + 2);

  // --- PLE block (after FFN residual; ends with its own barrier) ---
  DoRecordPleGpuBatch(ALayerIdx, ABatch);

  // --- Layer output scaling over all positions ---
  FComputeKernels.RecordScale(FGpuBatchHidden,
    UInt32(ABatch * FConfig.HiddenSize), FLayerScalars[ALayerIdx]);
  FComputeKernels.RecordBarrier();

  if LTsBase >= 0 then
    FComputeKernels.Device.RecordTimestamp(UInt32(LTsBase) + 3);
end;

procedure TModel.DoForwardGpuBatch(const ATokens: TArray<Integer>;
  const AFirst: Integer; const ACount: Integer; const AIsLast: Boolean);
var
  LI: Integer;
  LTs: TArray<UInt64>;
  LPeriodMs: Double;
  LM: Integer;
  LRunStart: Integer;
  LRunLen: Integer;
  LAnySoft: Boolean;
begin
  if (ACount <= 0) or (ACount > CPrefillBatch) then
  begin
    GetErrors().Add(esError, CMD_ERR_FORWARD,
      'Batch size %d out of range 1..%d', [ACount, CPrefillBatch]);
    Exit;
  end;
  if FPosition + ACount > CGpuMaxSeq then
  begin
    GetErrors().Add(esError, CMD_ERR_FORWARD,
      'Batch [%d..%d) exceeds GPU KV cache capacity %d',
      [FPosition, FPosition + ACount, CGpuMaxSeq]);
    Exit;
  end;

  // Reclaim descriptor sets from the previous submission, stage inputs
  FComputeKernels.Shaders.ResetDescriptorPool();
  DoUploadBatchInputsGpu(ATokens, AFirst, ACount);

  FComputeKernels.Device.BeginCommands();

  FComputeKernels.Device.RecordResetTimestamps();
  FComputeKernels.Device.RecordTimestamp(CTsStart);

  // Staged embeddings -> batched residual stream
  FComputeKernels.RecordCopy(FGpuBatchHidden, FGpuBatchEmbed,
    UInt32(ACount * FConfig.HiddenSize));
  FComputeKernels.RecordBarrier();

  // Soft-token substitution BEFORE PLE: overwrite substituted rows of the
  // batched residual stream with the staged multimodal rows. Image blocks
  // (280 rows) always straddle 256-token chunks, so contiguous runs are
  // copied per chunk intersection. Runs before the PLE precompute so the
  // context projection sees the raw soft rows (HF merged-embeds semantics,
  // modeling_gemma4.py line 1684); PLE identity lookups come from PAD.
  LAnySoft := False;
  LM := 0;
  while LM < ACount do
  begin
    if DoSoftRowAt(FPosition + LM) <> nil then
    begin
      LRunStart := LM;
      LRunLen := 0;
      while (LM < ACount) and (DoSoftRowAt(FPosition + LM) <> nil) do
      begin
        Inc(LRunLen);
        Inc(LM);
      end;
      FComputeKernels.RecordCopy(FGpuBatchHidden, FGpuSoftStage,
        UInt32(LRunLen * FConfig.HiddenSize),
        UInt32(LRunStart * FConfig.HiddenSize),
        UInt32(LRunStart * FConfig.HiddenSize));
      LAnySoft := True;
    end
    else
      Inc(LM);
  end;
  if LAnySoft then
    FComputeKernels.RecordBarrier();

  DoRecordPlePrecomputeGpuBatch(ACount);

  FComputeKernels.Device.RecordTimestamp(CTsPlePre);

  for LI := 0 to FConfig.NumHiddenLayers - 1 do
    DoRecordDecoderLayerGpuBatch(LI, ACount);

  FComputeKernels.Device.RecordTimestamp(CTsLayersEnd);

  // Only the LAST prompt position of the LAST chunk needs logits: copy its
  // hidden row into the single-token buffer and reuse the existing final
  // norm + lm_head recording
  if AIsLast then
  begin
    FComputeKernels.RecordCopy(FGpuHidden, FGpuBatchHidden,
      UInt32(FConfig.HiddenSize), UInt32((ACount - 1) * FConfig.HiddenSize), 0);
    FComputeKernels.RecordBarrier();
    DoRecordFinalGpu();
  end;

  FComputeKernels.Device.RecordTimestamp(CTsEnd);

  FComputeKernels.Device.EndCommands();
  FComputeKernels.Device.SubmitAndWait();

  // Accumulate the per-chunk GPU split (same slot meaning as FProfGpuMs)
  if FComputeKernels.Device.FetchTimestamps(LTs) then
  begin
    LPeriodMs := FComputeKernels.Device.TimestampPeriod / 1.0E6;
    FProfPrefillGpuMs[0] := FProfPrefillGpuMs[0] + (LTs[CTsPlePre] - LTs[CTsStart]) * LPeriodMs;
    FProfPrefillGpuMs[1] := FProfPrefillGpuMs[1] + (LTs[CTsLFullAttn] - LTs[CTsLFullStart]) * LPeriodMs;
    FProfPrefillGpuMs[2] := FProfPrefillGpuMs[2] + (LTs[CTsLFullMlp] - LTs[CTsLFullAttn]) * LPeriodMs;
    FProfPrefillGpuMs[3] := FProfPrefillGpuMs[3] + (LTs[CTsLFullEnd] - LTs[CTsLFullMlp]) * LPeriodMs;
    FProfPrefillGpuMs[4] := FProfPrefillGpuMs[4] + (LTs[CTsLSlidAttn] - LTs[CTsLSlidStart]) * LPeriodMs;
    FProfPrefillGpuMs[5] := FProfPrefillGpuMs[5] + (LTs[CTsLSlidMlp] - LTs[CTsLSlidAttn]) * LPeriodMs;
    FProfPrefillGpuMs[6] := FProfPrefillGpuMs[6] + (LTs[CTsLSlidEnd] - LTs[CTsLSlidMlp]) * LPeriodMs;
    FProfPrefillGpuMs[7] := FProfPrefillGpuMs[7] + (LTs[CTsLayersEnd] - LTs[CTsPlePre]) * LPeriodMs;
    FProfPrefillGpuMs[8] := FProfPrefillGpuMs[8] + (LTs[CTsEnd] - LTs[CTsLayersEnd]) * LPeriodMs;
    Inc(FProfPrefillChunks);
  end;

  if AIsLast then
    FComputeKernels.Device.DownloadFromBuffer(FGpuLogits, @FLogits[0],
      UInt64(FConfig.VocabSize) * SizeOf(Single));

  Inc(FPosition, ACount);
end;

procedure TModel.DoGpuMatVec(const AInfo: TTensorInfo;
  const AInput: PSingle; const AInputCount: Integer;
  const AOutput: PSingle; const ARows: Integer; const ACols: Integer);
var
  LLoc: TGpuTensorLoc;
begin
  // Upload input vector to GPU
  FComputeKernels.Device.UploadToBuffer(FGpuBufA, AInput,
    UInt64(AInputCount) * SizeOf(Single));

  // Legacy per-op path: translate through the upload-time layout table and
  // submit synchronously (the Dispatch matvec wrappers were removed)
  if FGpuOfs.TryGetValue(AInfo.TensorName, LLoc) then
  begin
    FComputeKernels.Device.BeginCommands();
    if LLoc.IsQ4 then
      FComputeKernels.RecordMatVecQ4(FGpuWeights, LLoc.BaseOfs, LLoc.ScalesOfs,
        LLoc.ByteSize, FGpuBufA, FGpuBufB, UInt32(ARows), UInt32(ACols))
    else
      FComputeKernels.RecordMatVecF32(FGpuWeights, LLoc.BaseOfs, LLoc.ByteSize,
        FGpuBufA, FGpuBufB, UInt32(ARows), UInt32(ACols));
    FComputeKernels.Device.EndCommands();
    FComputeKernels.Device.SubmitAndWait();
  end;

  // Download result
  FComputeKernels.Device.DownloadFromBuffer(FGpuBufB, AOutput,
    UInt64(ARows) * SizeOf(Single));
end;

procedure TModel.DoGpuRmsNorm(const ANormInfo: TTensorInfo;
  const ASrc: PSingle; const ADst: PSingle;
  const ASize: Integer; const AEps: Single);
var
  LByteSize: UInt64;
begin
  LByteSize := UInt64(ASize) * SizeOf(Single);

  // Upload source to BufA
  FComputeKernels.Device.UploadToBuffer(FGpuBufA, ASrc, LByteSize);

  // Upload norm weight to dedicated small buffer (from memory-mapped weights)
  FComputeKernels.Device.UploadToBuffer(FGpuNormWeight,
    FWeights.GetRawPointer(ANormInfo), LByteSize);

  // Dispatch RmsNorm: src=BufA, weight=NormWeight, dst=BufB
  FComputeKernels.DispatchRmsNorm(FGpuBufA, FGpuNormWeight, FGpuBufB,
    UInt32(ASize), AEps);

  // Download result
  FComputeKernels.Device.DownloadFromBuffer(FGpuBufB, ADst, LByteSize);
end;

procedure TModel.DoTraceBuffer(const ALabel: string;
  const AData: PSingle; const ACount: Integer);
var
  LI: Integer;
  LMin: Single;
  LMax: Single;
  LSum: Double;
  LVal: Single;
  LNanCount: Integer;
  LInfCount: Integer;
begin
  if FPosition > 0 then Exit; // only trace first token

  LMin := 1e30;
  LMax := -1e30;
  LSum := 0.0;
  LNanCount := 0;
  LInfCount := 0;

  for LI := 0 to ACount - 1 do
  begin
    LVal := PSingle(UIntPtr(AData) + UIntPtr(LI * SizeOf(Single)))^;
    if IsNan(LVal) then
      Inc(LNanCount)
    else if IsInfinite(LVal) then
      Inc(LInfCount)
    else
    begin
      if LVal < LMin then LMin := LVal;
      if LVal > LMax then LMax := LVal;
      LSum := LSum + LVal;
    end;
  end;

  Status('TRACE %s: min=%.6g max=%.6g mean=%.6g nan=%d inf=%d [%d]',
    [ALabel, LMin, LMax, LSum / Max(ACount - LNanCount - LInfCount, 1),
     LNanCount, LInfCount, ACount]);
end;

function TModel.Load(const AWeights: TWeightStore;
  const AConfig: TModelConfig;
  const AOwn: Boolean;
  const AComputeKernels: TComputeKernels): Boolean;
begin
  Result := False;

  if (AWeights = nil) or (not AWeights.IsOpen()) then
  begin
    GetErrors().Add(esError, CMD_ERR_WEIGHTS, 'Weight store is not open');
    Exit;
  end;

  FWeights := AWeights;
  FConfig := AConfig;
  FOwnsWeights := AOwn;

  // Setup GPU if compute kernels provided and initialized
  FComputeKernels := AComputeKernels;
  FUseGPU := (AComputeKernels <> nil) and AComputeKernels.IsInitialized();

  // Load per-layer output scalars once (used by both CPU and GPU paths;
  // the GPU path reads them from this array instead of the weight map)
  DoLoadLayerScalars();

  DoAllocateBuffers();

  // Upload weights to GPU
  if FUseGPU then
  begin
    if not DoUploadWeightsToGPU() then
    begin
      Status('GPU weight upload failed, falling back to CPU');
      DoFreeGpuBuffers();
      FUseGPU := False;
    end;
  end;

  // Initialize KV cache
  FKvCache.Init(
    FConfig.NumUniqueKvCaches,
    FConfig.NumKeyValueHeads,
    Max(FConfig.HeadDim, FConfig.GlobalHeadDim),
    FConfig.MaxPositionEmbeddings
  );

  FPosition := 0;
  FIsLoaded := True;
  Result := True;
end;

procedure TModel.SetLoadProgressCallback(
  const ACallback: TLoadProgressCallback;
  const AUserData: Pointer);
begin
  FLoadProgress.Callback := ACallback;
  FLoadProgress.UserData := AUserData;
end;

procedure TModel.DoEmbeddingLookup(const ATokenId: Integer);
var
  LInfo: TTensorInfo;
  LPtr: Pointer;
  LEmbSize: Integer;
begin
  if not FWeights.FindTensor('model.language_model.embed_tokens.weight', LInfo) then
  begin
    GetErrors().Add(esError, CMD_ERR_WEIGHTS, 'Embedding tensor not found');
    Exit;
  end;

  LEmbSize := FConfig.HiddenSize;
  LPtr := FWeights.GetRawPointer(LInfo);

  if LInfo.OutputDtype = dkQ4_0 then
  begin
    Q4ToF32Buffer(
      Pointer(UIntPtr(LPtr) + UIntPtr(UInt64(ATokenId) * Q4ByteSize(LEmbSize))),
      @FHidden[0],
      Q4BlockCount(LEmbSize)
    );
  end
  else
  begin
    Move(
      Pointer(UIntPtr(LPtr) + UIntPtr(UInt64(ATokenId) * UInt64(LEmbSize) * SizeOf(Single)))^,
      FHidden[0],
      LEmbSize * SizeOf(Single)
    );
  end;

  CpuScaleF32(@FHidden[0], Sqrt(Single(FConfig.HiddenSize)), FConfig.HiddenSize);
end;

function TModel.DoSoftRowAt(const APos: Integer): PSingle;
var
  LI: Integer;
  LOfs: Integer;
begin
  Result := nil;
  for LI := 0 to High(FSoftBlocks) do
  begin
    LOfs := APos - FSoftBlocks[LI].Position;
    if (LOfs >= 0) and (LOfs < FSoftBlocks[LI].RowCount) then
      Exit(@FSoftBlocks[LI].Rows[LOfs * FConfig.HiddenSize]);
  end;
end;

procedure TModel.DoPrecomputePLE(const ATokenId: Integer);
var
  LInfo: TTensorInfo;
  LNormInfo: TTensorInfo;
  LPtr: Pointer;
  LTotalDim: Integer;
  LPLEDim: Integer;
  LProj: TArray<Single>;
  LI: Integer;
begin
  // ---------------------------------------------------------------------------
  // PLE pre-computation (runs once per forward pass, before the layer loop).
  // Combines token-level per-layer embeddings with a projection of the
  // initial hidden state, matching llama.cpp's project_per_layer_inputs().
  //
  // Result: FPLEInput [n_embd_per_layer * n_layer] ready for per-layer use.
  // ---------------------------------------------------------------------------

  LPLEDim := FConfig.HiddenSizePerLayerInput;  // 256
  LTotalDim := LPLEDim * FConfig.NumHiddenLayers;  // 256 * 42 = 10752

  // Step 1: Look up token in embed_tokens_per_layer.weight -> [10752]
  if not FWeights.FindTensor('model.language_model.embed_tokens_per_layer.weight', LInfo) then
    Exit;

  LPtr := FWeights.GetRawPointer(LInfo);
  if LInfo.OutputDtype = dkQ4_0 then
    Q4ToF32Buffer(
      Pointer(UIntPtr(LPtr) + UIntPtr(UInt64(ATokenId) * Q4ByteSize(LTotalDim))),
      @FPLEInput[0], Q4BlockCount(LTotalDim))
  else
    Move(
      Pointer(UIntPtr(LPtr) + UIntPtr(UInt64(ATokenId) * UInt64(LTotalDim) * SizeOf(Single)))^,
      FPLEInput[0], LTotalDim * SizeOf(Single));

  // Step 2: Scale tok embed by sqrt(n_embd_per_layer)
  CpuScaleF32(@FPLEInput[0], Sqrt(Single(LPLEDim)), LTotalDim);

  // Step 3: Project hidden state through per_layer_model_projection
  // [10752, 2560] @ [2560] -> [10752]
  if not FWeights.FindTensor('model.language_model.per_layer_model_projection.weight', LInfo) then
    Exit;

  SetLength(LProj, LTotalDim);
  if LInfo.OutputDtype = dkQ4_0 then
    CpuMatVecQ4(FWeights.GetRawPointer(LInfo), @FHidden[0], @LProj[0],
      LTotalDim, FConfig.HiddenSize)
  else
    CpuMatVecF32(FWeights.GetRawPointer(LInfo), @FHidden[0], @LProj[0],
      LTotalDim, FConfig.HiddenSize);

  // Step 4: Scale projection by 1/sqrt(n_embd)
  CpuScaleF32(@LProj[0], 1.0 / Sqrt(Single(FConfig.HiddenSize)), LTotalDim);

  // Step 5: Per-layer-slice RMS norm with per_layer_projection_norm.weight [256]
  // Same weight applied independently to each [n_embd_per_layer] slice
  if FWeights.FindTensor('model.language_model.per_layer_projection_norm.weight', LNormInfo) then
  begin
    for LI := 0 to FConfig.NumHiddenLayers - 1 do
      CpuRmsNorm(@LProj[LI * LPLEDim],
        FWeights.GetRawPointer(LNormInfo),
        @LProj[LI * LPLEDim],
        LPLEDim, FConfig.RmsNormEps);
  end;

  // Step 6: Add projection to tok embed
  CpuAddF32(@LProj[0], @FPLEInput[0], LTotalDim);

  // Step 7: Scale combined result by 1/sqrt(2)
  CpuScaleF32(@FPLEInput[0], 1.0 / Sqrt(2.0), LTotalDim);
end;

procedure TModel.DoApplyPLE(const ALayerIdx: Integer);
var
  LPrefix: string;
  LInfo: TTensorInfo;
  LNormInfo: TTensorInfo;
  LPLEDim: Integer;
  LOffset: Integer;
begin
  // ---------------------------------------------------------------------------
  // Per-layer PLE application (after FFN residual, before layer_scalar).
  // Matches llama.cpp's per-layer PLE block:
  //   gate -> GeLU -> multiply with precomputed input -> project -> norm -> residual
  // ---------------------------------------------------------------------------

  LPLEDim := FConfig.HiddenSizePerLayerInput;  // 256
  LPrefix := LayerPrefix(ALayerIdx);
  LOffset := ALayerIdx * LPLEDim;

  // Gate: FHidden [2560] -> FPLEGate [256]
  // per_layer_input_gate.weight [256, 2560]
  if not FWeights.FindTensor(LPrefix + 'per_layer_input_gate.weight', LInfo) then
    Exit;

  if LInfo.OutputDtype = dkQ4_0 then
    CpuMatVecQ4(FWeights.GetRawPointer(LInfo), @FHidden[0], @FPLEGate[0],
      LPLEDim, FConfig.HiddenSize)
  else
    CpuMatVecF32(FWeights.GetRawPointer(LInfo), @FHidden[0], @FPLEGate[0],
      LPLEDim, FConfig.HiddenSize);

  // GeLU activation on gate output
  CpuGeluPytorchTanh(@FPLEGate[0], LPLEDim);

  // Element-wise multiply with precomputed PLE input for this layer
  // FPLEGate[i] *= FPLEInput[offset + i]
  CpuMulF32(@FPLEInput[LOffset], @FPLEGate[0], LPLEDim);

  // Project back: FPLEGate [256] -> FMlpOut [2560] (reuse FMlpOut as scratch)
  // per_layer_projection.weight [2560, 256]
  if not FWeights.FindTensor(LPrefix + 'per_layer_projection.weight', LInfo) then
    Exit;

  if LInfo.OutputDtype = dkQ4_0 then
    CpuMatVecQ4(FWeights.GetRawPointer(LInfo), @FPLEGate[0], @FMlpOut[0],
      FConfig.HiddenSize, LPLEDim)
  else
    CpuMatVecF32(FWeights.GetRawPointer(LInfo), @FPLEGate[0], @FMlpOut[0],
      FConfig.HiddenSize, LPLEDim);

  // Post-PLE RMS norm
  // post_per_layer_input_norm.weight [2560]
  if FWeights.FindTensor(LPrefix + 'post_per_layer_input_norm.weight', LNormInfo) then
    CpuRmsNorm(@FMlpOut[0], FWeights.GetRawPointer(LNormInfo), @FMlpOut[0],
      FConfig.HiddenSize, FConfig.RmsNormEps);

  // PLE residual: FHidden += projected PLE output
  CpuResidualAdd(@FMlpOut[0], @FHidden[0], FConfig.HiddenSize);
end;

procedure TModel.DoAttention(const ALayerIdx: Integer);
var
  LPrefix: string;
  LInfo: TTensorInfo;
  LNormInfo: TTensorInfo;
  LAttnKind: TAttentionKind;
  LHeadDim: Integer;
  LKvSlotIdx: Integer;
  LSlidingWindow: Integer;
  LRopeTheta: Single;
  LRopePartial: Single;
  LH: Integer;
  LNormPtr: PSingle;
  LHasKv: Boolean;
begin
  LPrefix := LayerPrefix(ALayerIdx);
  LAttnKind := FConfig.LayerTypes[ALayerIdx];
  LKvSlotIdx := FConfig.KvCacheMap[ALayerIdx];

  // Layers beyond (NumHiddenLayers - NumKvSharedLayers) share KV caches
  // with earlier layers. They have no K/V projection weights -- only Q.
  LHasKv := ALayerIdx < (FConfig.NumHiddenLayers - FConfig.NumKvSharedLayers);

  if LAttnKind = akFull then
  begin
    LHeadDim := FConfig.GlobalHeadDim;
    LSlidingWindow := 0;
    LRopeTheta := FConfig.RopeFull.Theta;
    LRopePartial := FConfig.RopeFull.PartialRotaryFactor;
  end
  else
  begin
    LHeadDim := FConfig.HeadDim;
    LSlidingWindow := FConfig.SlidingWindow;
    LRopeTheta := FConfig.RopeSliding.Theta;
    LRopePartial := 1.0;
  end;

  // Q projection: [num_heads * head_dim, hidden_size]
  if FWeights.FindTensor(LPrefix + 'self_attn.q_proj.weight', LInfo) then
  begin
    if FUseGPU then
      DoGpuMatVec(LInfo, @FNormBuf[0], FConfig.HiddenSize,
        @FQuery[0], FConfig.NumAttentionHeads * LHeadDim, FConfig.HiddenSize)
    else if LInfo.OutputDtype = dkQ4_0 then
      CpuMatVecQ4(FWeights.GetRawPointer(LInfo), @FNormBuf[0], @FQuery[0],
        FConfig.NumAttentionHeads * LHeadDim, FConfig.HiddenSize)
    else
      CpuMatVecF32(FWeights.GetRawPointer(LInfo), @FNormBuf[0], @FQuery[0],
        FConfig.NumAttentionHeads * LHeadDim, FConfig.HiddenSize);
  end;

  // Per-head Q norm with learned weight (self_attn.q_norm.weight, shape [head_dim])
  // Q norm is always applied, even for shared-KV layers
  if FWeights.FindTensor(LPrefix + 'self_attn.q_norm.weight', LNormInfo) then
  begin
    LNormPtr := FWeights.GetRawPointer(LNormInfo);
    for LH := 0 to FConfig.NumAttentionHeads - 1 do
      CpuRmsNorm(@FQuery[LH * LHeadDim], LNormPtr, @FQuery[LH * LHeadDim],
        LHeadDim, FConfig.RmsNormEps);
  end;

  // K/V projection, norms, and cache append -- only for layers with own KV
  if LHasKv then
  begin
    // K projection: [num_kv_heads * head_dim, hidden_size]
    if FWeights.FindTensor(LPrefix + 'self_attn.k_proj.weight', LInfo) then
    begin
      if FUseGPU then
        DoGpuMatVec(LInfo, @FNormBuf[0], FConfig.HiddenSize,
          @FKeyBuf[0], FConfig.NumKeyValueHeads * LHeadDim, FConfig.HiddenSize)
      else if LInfo.OutputDtype = dkQ4_0 then
        CpuMatVecQ4(FWeights.GetRawPointer(LInfo), @FNormBuf[0], @FKeyBuf[0],
          FConfig.NumKeyValueHeads * LHeadDim, FConfig.HiddenSize)
      else
        CpuMatVecF32(FWeights.GetRawPointer(LInfo), @FNormBuf[0], @FKeyBuf[0],
          FConfig.NumKeyValueHeads * LHeadDim, FConfig.HiddenSize);
    end;

    // V projection: [num_kv_heads * head_dim, hidden_size]
    if FWeights.FindTensor(LPrefix + 'self_attn.v_proj.weight', LInfo) then
    begin
      if FUseGPU then
        DoGpuMatVec(LInfo, @FNormBuf[0], FConfig.HiddenSize,
          @FValueBuf[0], FConfig.NumKeyValueHeads * LHeadDim, FConfig.HiddenSize)
      else if LInfo.OutputDtype = dkQ4_0 then
        CpuMatVecQ4(FWeights.GetRawPointer(LInfo), @FNormBuf[0], @FValueBuf[0],
          FConfig.NumKeyValueHeads * LHeadDim, FConfig.HiddenSize)
      else
        CpuMatVecF32(FWeights.GetRawPointer(LInfo), @FNormBuf[0], @FValueBuf[0],
          FConfig.NumKeyValueHeads * LHeadDim, FConfig.HiddenSize);
    end;

    // Per-head K norm with learned weight (self_attn.k_norm.weight, shape [head_dim])
    if FWeights.FindTensor(LPrefix + 'self_attn.k_norm.weight', LNormInfo) then
    begin
      LNormPtr := FWeights.GetRawPointer(LNormInfo);
      for LH := 0 to FConfig.NumKeyValueHeads - 1 do
        CpuRmsNorm(@FKeyBuf[LH * LHeadDim], LNormPtr, @FKeyBuf[LH * LHeadDim],
          LHeadDim, FConfig.RmsNormEps);
    end;

    // Per-head V norm -- weight-free RMS norm (no learned gamma)
    for LH := 0 to FConfig.NumKeyValueHeads - 1 do
      CpuRmsNormNoWeight(@FValueBuf[LH * LHeadDim], @FValueBuf[LH * LHeadDim],
        LHeadDim, FConfig.RmsNormEps);

    // RoPE on K
    for LH := 0 to FConfig.NumKeyValueHeads - 1 do
      CpuApplyRoPE(@FKeyBuf[LH * LHeadDim], FPosition, LHeadDim, LRopeTheta, LRopePartial);

    // KV cache append
    for LH := 0 to FConfig.NumKeyValueHeads - 1 do
      FKvCache.Append(LKvSlotIdx, LH, @FKeyBuf[LH * LHeadDim],
        @FValueBuf[LH * LHeadDim], LHeadDim);
  end;
  // Shared-KV layers skip all of the above and use the existing KV cache
  // at KvCacheMap[ALayerIdx], which points to an earlier layer's cache.

  // RoPE on Q -- always applied, even for shared-KV layers
  for LH := 0 to FConfig.NumAttentionHeads - 1 do
    CpuApplyRoPE(@FQuery[LH * LHeadDim], FPosition, LHeadDim, LRopeTheta, LRopePartial);

  // Multi-head attention with GQA (CPU -- requires KV cache access)
  FillChar(FAttnOut[0], FConfig.NumAttentionHeads * LHeadDim * SizeOf(Single), 0);
  CpuMultiHeadAttention(
    @FQuery[0], FKvCache, LKvSlotIdx, @FAttnOut[0],
    FConfig.NumAttentionHeads, FConfig.NumKeyValueHeads,
    LHeadDim, FPosition, LSlidingWindow, 0.0  // Gemma 4: no attention softcapping
  );

  // O projection: [hidden_size, num_heads * head_dim]
  if FWeights.FindTensor(LPrefix + 'self_attn.o_proj.weight', LInfo) then
  begin
    if FUseGPU then
      DoGpuMatVec(LInfo, @FAttnOut[0], FConfig.NumAttentionHeads * LHeadDim,
        @FMlpOut[0], FConfig.HiddenSize, FConfig.NumAttentionHeads * LHeadDim)
    else if LInfo.OutputDtype = dkQ4_0 then
      CpuMatVecQ4(FWeights.GetRawPointer(LInfo), @FAttnOut[0], @FMlpOut[0],
        FConfig.HiddenSize, FConfig.NumAttentionHeads * LHeadDim)
    else
      CpuMatVecF32(FWeights.GetRawPointer(LInfo), @FAttnOut[0], @FMlpOut[0],
        FConfig.HiddenSize, FConfig.NumAttentionHeads * LHeadDim);
  end;

  // Note: residual add is now managed by DoDecoderLayer, not here
end;

procedure TModel.DoMLP(const ALayerIdx: Integer);
var
  LPrefix: string;
  LGateInfo: TTensorInfo;
  LUpInfo: TTensorInfo;
  LDownInfo: TTensorInfo;
  LIsQ4: Boolean;
begin
  LPrefix := LayerPrefix(ALayerIdx);

  if not FWeights.FindTensor(LPrefix + 'mlp.gate_proj.weight', LGateInfo) then Exit;
  if not FWeights.FindTensor(LPrefix + 'mlp.up_proj.weight', LUpInfo) then Exit;
  if not FWeights.FindTensor(LPrefix + 'mlp.down_proj.weight', LDownInfo) then Exit;

  if FUseGPU then
  begin
    // GPU path: decompose SwiGLU into individual dispatches
    // Gate projection: [intermediate_size, hidden_size] * normBuf
    DoGpuMatVec(LGateInfo, @FNormBuf[0], FConfig.HiddenSize,
      @FMlpGate[0], FConfig.IntermediateSize, FConfig.HiddenSize);

    // Up projection: [intermediate_size, hidden_size] * normBuf
    DoGpuMatVec(LUpInfo, @FNormBuf[0], FConfig.HiddenSize,
      @FMlpUp[0], FConfig.IntermediateSize, FConfig.HiddenSize);

    // CPU: activation and element-wise multiply (GeGLU -- Gemma 4 uses GeLU gating)
    CpuGeluPytorchTanh(@FMlpGate[0], FConfig.IntermediateSize);
    CpuMulF32(@FMlpUp[0], @FMlpGate[0], FConfig.IntermediateSize);

    // Down projection: [hidden_size, intermediate_size] * (gate * up)
    DoGpuMatVec(LDownInfo, @FMlpGate[0], FConfig.IntermediateSize,
      @FMlpOut[0], FConfig.HiddenSize, FConfig.IntermediateSize);
  end
  else
  begin
    // CPU path: monolithic SwiGLU
    LIsQ4 := LGateInfo.OutputDtype = dkQ4_0;
    CpuMlpSwiGLU(
      @FNormBuf[0],
      FWeights.GetRawPointer(LGateInfo),
      FWeights.GetRawPointer(LUpInfo),
      FWeights.GetRawPointer(LDownInfo),
      @FMlpOut[0],
      FConfig.HiddenSize,
      FConfig.IntermediateSize,
      LIsQ4
    );
  end;

  // Note: residual add is now managed by DoDecoderLayer, not here
end;

procedure TModel.DoDecoderLayer(const ALayerIdx: Integer);
var
  LPrefix: string;
  LInfo: TTensorInfo;
  LScalar: Single;
  LTrace: Boolean;
begin
  LPrefix := LayerPrefix(ALayerIdx);
  LTrace := (ALayerIdx = 0) and (FPosition = 0);

  // Step 1: Pre-attention RMSNorm -> FNormBuf
  if FWeights.FindTensor(LPrefix + 'input_layernorm.weight', LInfo) then
  begin
    if FUseGPU then
      DoGpuRmsNorm(LInfo, @FHidden[0], @FNormBuf[0],
        FConfig.HiddenSize, FConfig.RmsNormEps)
    else
      CpuRmsNorm(@FHidden[0], FWeights.GetRawPointer(LInfo), @FNormBuf[0],
        FConfig.HiddenSize, FConfig.RmsNormEps);
  end;
  if LTrace then DoTraceBuffer('L0.1-pre_attn_norm', @FNormBuf[0], FConfig.HiddenSize);

  // Step 2: Attention (reads FNormBuf, writes O-proj result to FMlpOut)
  DoAttention(ALayerIdx);
  if LTrace then DoTraceBuffer('L0.2-attn_out', @FMlpOut[0], FConfig.HiddenSize);

  // Step 3: Post-attention RMSNorm (in-place on FMlpOut)
  if FWeights.FindTensor(LPrefix + 'post_attention_layernorm.weight', LInfo) then
  begin
    if FUseGPU then
      DoGpuRmsNorm(LInfo, @FMlpOut[0], @FMlpOut[0],
        FConfig.HiddenSize, FConfig.RmsNormEps)
    else
      CpuRmsNorm(@FMlpOut[0], FWeights.GetRawPointer(LInfo), @FMlpOut[0],
        FConfig.HiddenSize, FConfig.RmsNormEps);
  end;
  if LTrace then DoTraceBuffer('L0.3-post_attn_norm', @FMlpOut[0], FConfig.HiddenSize);

  // Step 4: Attention residual -> FHidden becomes attn_out
  CpuResidualAdd(@FMlpOut[0], @FHidden[0], FConfig.HiddenSize);
  if LTrace then DoTraceBuffer('L0.4-attn_residual', @FHidden[0], FConfig.HiddenSize);

  // Step 5: Pre-FFN RMSNorm -> FNormBuf
  if FWeights.FindTensor(LPrefix + 'pre_feedforward_layernorm.weight', LInfo) then
  begin
    if FUseGPU then
      DoGpuRmsNorm(LInfo, @FHidden[0], @FNormBuf[0],
        FConfig.HiddenSize, FConfig.RmsNormEps)
    else
      CpuRmsNorm(@FHidden[0], FWeights.GetRawPointer(LInfo), @FNormBuf[0],
        FConfig.HiddenSize, FConfig.RmsNormEps);
  end;
  if LTrace then DoTraceBuffer('L0.5-pre_ffn_norm', @FNormBuf[0], FConfig.HiddenSize);

  // Step 6: MLP (reads FNormBuf, writes result to FMlpOut)
  DoMLP(ALayerIdx);
  if LTrace then DoTraceBuffer('L0.6-mlp_out', @FMlpOut[0], FConfig.HiddenSize);

  // Step 7: Post-FFN RMSNorm (in-place on FMlpOut)
  if FWeights.FindTensor(LPrefix + 'post_feedforward_layernorm.weight', LInfo) then
  begin
    if FUseGPU then
      DoGpuRmsNorm(LInfo, @FMlpOut[0], @FMlpOut[0],
        FConfig.HiddenSize, FConfig.RmsNormEps)
    else
      CpuRmsNorm(@FMlpOut[0], FWeights.GetRawPointer(LInfo), @FMlpOut[0],
        FConfig.HiddenSize, FConfig.RmsNormEps);
  end;
  if LTrace then DoTraceBuffer('L0.7-post_ffn_norm', @FMlpOut[0], FConfig.HiddenSize);

  // Step 8: FFN residual -> FHidden becomes layer output
  CpuResidualAdd(@FMlpOut[0], @FHidden[0], FConfig.HiddenSize);
  if LTrace then DoTraceBuffer('L0.8-ffn_residual', @FHidden[0], FConfig.HiddenSize);

  // Step 9: Per-layer embedding (PLE)
  DoApplyPLE(ALayerIdx);
  if LTrace then DoTraceBuffer('L0.9-ple', @FHidden[0], FConfig.HiddenSize);

  // Step 10: Layer output scaling
  if FWeights.FindTensor(LPrefix + 'layer_scalar', LInfo) then
  begin
    LScalar := PSingle(FWeights.GetRawPointer(LInfo))^;
    if LTrace then Status('L0.10-layer_scalar = %.6f', [LScalar]);
    CpuScaleF32(@FHidden[0], LScalar, FConfig.HiddenSize);
  end;
  if LTrace then DoTraceBuffer('L0.10-scaled', @FHidden[0], FConfig.HiddenSize);
end;

procedure TModel.DoFinalNormAndProject();
var
  LInfo: TTensorInfo;
begin
  // Final RMSNorm
  if FWeights.FindTensor('model.language_model.norm.weight', LInfo) then
  begin
    if FUseGPU then
      DoGpuRmsNorm(LInfo, @FHidden[0], @FNormBuf[0],
        FConfig.HiddenSize, FConfig.RmsNormEps)
    else
      CpuRmsNorm(@FHidden[0], FWeights.GetRawPointer(LInfo), @FNormBuf[0],
        FConfig.HiddenSize, FConfig.RmsNormEps);
  end;

  // lm_head projection: [vocab_size, hidden_size]
  if FWeights.FindTensor('model.language_model.lm_head.weight', LInfo) then
  begin
    if FUseGPU then
      DoGpuMatVec(LInfo, @FNormBuf[0], FConfig.HiddenSize,
        @FLogits[0], FConfig.VocabSize, FConfig.HiddenSize)
    else if LInfo.OutputDtype = dkQ4_0 then
      CpuMatVecQ4(FWeights.GetRawPointer(LInfo), @FNormBuf[0], @FLogits[0],
        FConfig.VocabSize, FConfig.HiddenSize)
    else
      CpuMatVecF32(FWeights.GetRawPointer(LInfo), @FNormBuf[0], @FLogits[0],
        FConfig.VocabSize, FConfig.HiddenSize);
  end
  else if FConfig.TieWordEmbeddings then
  begin
    if FWeights.FindTensor('model.language_model.embed_tokens.weight', LInfo) then
    begin
      if FUseGPU then
        DoGpuMatVec(LInfo, @FNormBuf[0], FConfig.HiddenSize,
          @FLogits[0], FConfig.VocabSize, FConfig.HiddenSize)
      else if LInfo.OutputDtype = dkQ4_0 then
        CpuMatVecQ4(FWeights.GetRawPointer(LInfo), @FNormBuf[0], @FLogits[0],
          FConfig.VocabSize, FConfig.HiddenSize)
      else
        CpuMatVecF32(FWeights.GetRawPointer(LInfo), @FNormBuf[0], @FLogits[0],
          FConfig.VocabSize, FConfig.HiddenSize);
    end;
  end;
end;

function TModel.Forward(const ATokenId: Integer): TArray<Single>;
var
  LI: Integer;
  LSoftRow: PSingle;
  LEffToken: Integer;
begin
  // GPU-resident path: entire token recorded into one command buffer,
  // one submit, one fence wait, logits downloaded (PLAN-gpu-residency).
  if FUseGPU then
  begin
    DoForwardGpu(ATokenId);
    Exit(Copy(FLogits));
  end;

  // Reset descriptor pool before each forward pass to reclaim GPU descriptor sets
  if FUseGPU then
    FComputeKernels.Shaders.ResetDescriptorPool();

  // Soft-token substitution: embed + PLE identity come from the PAD token;
  // the soft row replaces the residual stream BEFORE the PLE precompute so
  // the context projection sees it (HF merged-embeds semantics)
  LSoftRow := DoSoftRowAt(FPosition);
  LEffToken := ATokenId;
  if LSoftRow <> nil then
    LEffToken := CPadTokenId;

  // 1. Embedding lookup (CPU)
  DoEmbeddingLookup(LEffToken);
  DoTraceBuffer('embed', @FHidden[0], FConfig.HiddenSize);

  // Soft-token substitution (before PLE: the projection must see the row)
  if LSoftRow <> nil then
    Move(LSoftRow^, FHidden[0], FConfig.HiddenSize * SizeOf(Single));

  // 2. Pre-compute PLE input for all layers (once per forward pass)
  DoPrecomputePLE(LEffToken);

  // 3. Decoder layers (PLE applied inside each layer after FFN residual)
  for LI := 0 to FConfig.NumHiddenLayers - 1 do
  begin
    DoDecoderLayer(LI);
    if (LI = 0) or (LI = 1) then
      DoTraceBuffer(Format('layer%d', [LI]), @FHidden[0], FConfig.HiddenSize);
  end;

  // 4. Final norm + projection
  DoFinalNormAndProject();
  DoTraceBuffer('logits', @FLogits[0], Min(1000, FConfig.VocabSize));

  // 5. Advance position
  Inc(FPosition);

  Result := Copy(FLogits);
end;

function TModel.DoSelectTopKIndices(const AK: Integer): TArray<Integer>;
var
  LI: Integer;
  LJ: Integer;
  LLen: Integer;
  LHeapVal: TArray<Single>;
  LHeapIdx: TArray<Integer>;
  LVal: Single;
  LPos: Integer;
  LChild: Integer;
  LTmp: Integer;
begin
  LLen := Length(FLogits);

  // Min-heap of the current top-K (value, index) pairs; the root is the
  // SMALLEST of the top-K. Single pass: any value larger than the root
  // replaces it and sifts down. O(V log K).
  SetLength(LHeapVal, AK);
  SetLength(LHeapIdx, AK);
  for LI := 0 to AK - 1 do
  begin
    LHeapVal[LI] := -1e30;
    LHeapIdx[LI] := -1;
  end;

  for LI := 0 to LLen - 1 do
  begin
    LVal := FLogits[LI];
    if LVal <= LHeapVal[0] then
      Continue;

    // Replace the root with (LVal, LI) and restore the heap property
    LPos := 0;
    while True do
    begin
      LChild := LPos * 2 + 1;
      if LChild >= AK then
        Break;
      // Pick the smaller child
      if (LChild + 1 < AK) and (LHeapVal[LChild + 1] < LHeapVal[LChild]) then
        Inc(LChild);
      if LHeapVal[LChild] >= LVal then
        Break;
      LHeapVal[LPos] := LHeapVal[LChild];
      LHeapIdx[LPos] := LHeapIdx[LChild];
      LPos := LChild;
    end;
    LHeapVal[LPos] := LVal;
    LHeapIdx[LPos] := LI;
  end;

  // Collect valid indices and sort ascending (insertion sort; K is small).
  // Ascending order preserves the legacy full-vocab cumsum semantics.
  SetLength(Result, AK);
  LLen := 0;
  for LI := 0 to AK - 1 do
    if LHeapIdx[LI] >= 0 then
    begin
      Result[LLen] := LHeapIdx[LI];
      Inc(LLen);
    end;
  SetLength(Result, LLen);

  for LI := 1 to LLen - 1 do
  begin
    LTmp := Result[LI];
    LJ := LI - 1;
    while (LJ >= 0) and (Result[LJ] > LTmp) do
    begin
      Result[LJ + 1] := Result[LJ];
      Dec(LJ);
    end;
    Result[LJ + 1] := LTmp;
  end;
end;

function TModel.DoSampleTopP(var ALogits: TArray<Single>;
  const ATopP: Single; const ATemperature: Single): Integer;
var
  LI: Integer;
  LLen: Integer;
  LMax: Single;
  LCumSum: Single;
  LRand: Single;
  LVal: Single;
begin
  LLen := Length(ALogits);

  if (ATemperature > 0.0) and (ATemperature <> 1.0) then
  begin
    LVal := 1.0 / ATemperature;
    for LI := 0 to LLen - 1 do
      ALogits[LI] := ALogits[LI] * LVal;
  end;

  CpuSoftmaxF32(@ALogits[0], LLen);

  Result := 0;
  LMax := ALogits[0];
  for LI := 1 to LLen - 1 do
  begin
    if ALogits[LI] > LMax then
    begin
      LMax := ALogits[LI];
      Result := LI;
    end;
  end;

  if ATemperature > 0.0 then
  begin
    LCumSum := 0.0;
    LRand := Random * ATopP;
    for LI := 0 to LLen - 1 do
    begin
      LCumSum := LCumSum + ALogits[LI];
      if LCumSum >= LRand then
      begin
        Result := LI;
        Break;
      end;
    end;
  end;
end;

function TModel.DoSample(const AParams: TSamplingParams): Integer;
var
  LIndices: TArray<Integer>;
  LVals: TArray<Single>;
  LLogits: TArray<Single>;
  LI: Integer;
  LCount: Integer;
  LCap: Single;
  LInvTemp: Single;
  LMax: Single;
  LSum: Single;
  LCum: Single;
  LRand: Single;
  LBest: Integer;
begin
  // Fallback: no top-k requested -- softcap the full vocab and use the
  // legacy full-vocab path
  if (AParams.TopK <= 0) or (AParams.TopK >= FConfig.VocabSize) then
  begin
    LLogits := Copy(FLogits);
    if FConfig.FinalLogitSoftcapping > 0.0 then
      CpuLogitSoftcap(@LLogits[0], Length(LLogits),
        FConfig.FinalLogitSoftcapping);
    Exit(DoSampleTopP(LLogits, AParams.TopP, AParams.Temperature));
  end;

  // Softcap is monotonic: top-K on RAW logits selects the identical
  // candidate set. Select first, softcap only the K survivors.
  LIndices := DoSelectTopKIndices(AParams.TopK);
  LCount := Length(LIndices);
  SetLength(LVals, LCount);
  LCap := FConfig.FinalLogitSoftcapping;

  for LI := 0 to LCount - 1 do
  begin
    LVals[LI] := FLogits[LIndices[LI]];
    if LCap > 0.0 then
      LVals[LI] := LCap * Tanh(LVals[LI] / LCap);
  end;

  // Temperature
  if (AParams.Temperature > 0.0) and (AParams.Temperature <> 1.0) then
  begin
    LInvTemp := 1.0 / AParams.Temperature;
    for LI := 0 to LCount - 1 do
      LVals[LI] := LVals[LI] * LInvTemp;
  end;

  // Numerically stable softmax over the K survivors -- identical to the
  // legacy full-vocab softmax where non-survivors were masked to -1e30
  LMax := LVals[0];
  for LI := 1 to LCount - 1 do
    if LVals[LI] > LMax then
      LMax := LVals[LI];

  LSum := 0.0;
  for LI := 0 to LCount - 1 do
  begin
    LVals[LI] := Exp(LVals[LI] - LMax);
    LSum := LSum + LVals[LI];
  end;
  for LI := 0 to LCount - 1 do
    LVals[LI] := LVals[LI] / LSum;

  // Greedy argmax (temperature = 0 result and top-p fallback)
  LBest := 0;
  for LI := 1 to LCount - 1 do
    if LVals[LI] > LVals[LBest] then
      LBest := LI;

  // Legacy top-p: cumsum in ascending vocab-index order against
  // uniform(0, TopP) -- LIndices is ascending, so semantics are preserved
  if AParams.Temperature > 0.0 then
  begin
    LRand := Random * AParams.TopP;
    LCum := 0.0;
    for LI := 0 to LCount - 1 do
    begin
      LCum := LCum + LVals[LI];
      if LCum >= LRand then
      begin
        LBest := LI;
        Break;
      end;
    end;
  end;

  Result := LIndices[LBest];
end;

function TModel.Generate(
  const AInputTokens: TArray<Integer>;
  const AMaxTokens: Integer;
  const AParams: TSamplingParams;
  const ACallback: TGenerateCallback;
  const AUserData: Pointer;
  const ASoftBlocks: TArray<TSoftTokenBlock>): TArray<Integer>;
var
  LResult: TList<Integer>;
  LI: Integer;
  LTokenId: Integer;
  LContinue: Boolean;
  LStopwatch: TStopwatch;
  LSampleSw: TStopwatch;
  LChunkStart: Integer;
  LChunkCount: Integer;
begin
  // Soft-token substitution list for this call (validated up front)
  FSoftBlocks := ASoftBlocks;
  for LI := 0 to High(FSoftBlocks) do
    if Length(FSoftBlocks[LI].Rows) <>
      FSoftBlocks[LI].RowCount * FConfig.HiddenSize then
    begin
      GetErrors().Add(esError, CMD_ERR_FORWARD,
        'Soft block %d: %d floats, expected %d x %d',
        [LI, Length(FSoftBlocks[LI].Rows), FSoftBlocks[LI].RowCount,
        FConfig.HiddenSize]);
      FSoftBlocks := nil;
      Exit(nil);
    end;

  LResult := TList<Integer>.Create();
  try
    // Prompt processing (prefill) measurement
    FProfPrefillTokens := Length(AInputTokens);
    FillChar(FProfPrefillGpuMs, SizeOf(FProfPrefillGpuMs), 0);
    FProfPrefillChunks := 0;
    LStopwatch := TStopwatch.StartNew();

    if FUseGPU and FUseBatchPrefill and (Length(AInputTokens) > 0) then
    begin
      // Batched prefill: chunks of up to CPrefillBatch positions per
      // submission; only the last chunk records final norm + lm_head
      LChunkStart := 0;
      while LChunkStart < Length(AInputTokens) do
      begin
        LChunkCount := Min(CPrefillBatch, Length(AInputTokens) - LChunkStart);
        DoForwardGpuBatch(AInputTokens, LChunkStart, LChunkCount,
          LChunkStart + LChunkCount >= Length(AInputTokens));
        Inc(LChunkStart, LChunkCount);
      end;
    end
    else
    begin
      for LI := 0 to High(AInputTokens) do
        Forward(AInputTokens[LI]);
    end;

    FProfPrefillSec := LStopwatch.ElapsedMilliseconds / 1000.0;

    LTokenId := DoSample(AParams);
    LI := 0;

    // Generation throughput measurement (prefill excluded)
    LStopwatch := TStopwatch.StartNew();

    // Reset per-token phase accumulators (prefill excluded, matching stopwatch)
    FProfUploadMs := 0.0;
    FProfRecordMs := 0.0;
    FProfFenceMs := 0.0;
    FProfDownloadMs := 0.0;
    FProfSampleMs := 0.0;
    FProfTokens := 0;
    FillChar(FProfGpuMs, SizeOf(FProfGpuMs), 0);
    FProfElapsedSec := 0.0;

    while LI < AMaxTokens do
    begin
      LResult.Add(LTokenId);

      if IsEosToken(LTokenId) then
        Break;

      if Assigned(ACallback) then
      begin
        LContinue := ACallback(LTokenId, '', AUserData);
        if not LContinue then
          Break;
      end;

      Forward(LTokenId);

      // --- Phase: sample (CPU top-K + softcap + top-p) ---
      LSampleSw := TStopwatch.StartNew();
      LTokenId := DoSample(AParams);
      FProfSampleMs := FProfSampleMs + LSampleSw.Elapsed.TotalMilliseconds;

      Inc(LI);
    end;

    // Capture wall-clock generation time; all reporting now happens via
    // GetStats -> TInferenceStats (no Status spew into token output)
    FProfElapsedSec := LStopwatch.ElapsedMilliseconds / 1000.0;

    Result := LResult.ToArray();
  finally
    LResult.Free();
    FSoftBlocks := nil;
  end;
end;

procedure TModel.Reset();
begin
  FKvCache.Clear();
  FPosition := 0;
end;

procedure TModel.GetStats(out AStats: TInferenceStats);
var
  LInv: Double;
begin
  AStats := Default(TInferenceStats);

  AStats.TokenCount := FProfTokens;
  AStats.ElapsedSec := FProfElapsedSec;
  if FProfElapsedSec > 0.0 then
    AStats.TokensPerSec := FProfTokens / FProfElapsedSec;

  AStats.PrefillTokenCount := FProfPrefillTokens;
  AStats.PrefillSec := FProfPrefillSec;
  if FProfPrefillSec > 0.0 then
    AStats.PrefillTokensPerSec := FProfPrefillTokens / FProfPrefillSec;

  // Batched-prefill GPU split (per-chunk averages)
  AStats.HasPrefillGpuStats := FUseBatchPrefill and (FProfPrefillChunks > 0)
    and (FProfPrefillGpuMs[7] > 0.0);
  if AStats.HasPrefillGpuStats then
  begin
    AStats.PrefillChunkCount := FProfPrefillChunks;
    AStats.PrefillGpu.EmbedPlePreMs := FProfPrefillGpuMs[0] / FProfPrefillChunks;
    AStats.PrefillGpu.FullAttnMs := FProfPrefillGpuMs[1] / FProfPrefillChunks;
    AStats.PrefillGpu.FullMlpMs := FProfPrefillGpuMs[2] / FProfPrefillChunks;
    AStats.PrefillGpu.FullPleMs := FProfPrefillGpuMs[3] / FProfPrefillChunks;
    AStats.PrefillGpu.SlidAttnMs := FProfPrefillGpuMs[4] / FProfPrefillChunks;
    AStats.PrefillGpu.SlidMlpMs := FProfPrefillGpuMs[5] / FProfPrefillChunks;
    AStats.PrefillGpu.SlidPleMs := FProfPrefillGpuMs[6] / FProfPrefillChunks;
    AStats.PrefillGpu.LayersTotalMs := FProfPrefillGpuMs[7] / FProfPrefillChunks;
    AStats.PrefillGpu.FinalLmHeadMs := FProfPrefillGpuMs[8] / FProfPrefillChunks;
    AStats.PrefillGpu.ExtrapolatedLayersMs :=
      (7.0 * (FProfPrefillGpuMs[1] + FProfPrefillGpuMs[2] + FProfPrefillGpuMs[3]) +
       35.0 * (FProfPrefillGpuMs[4] + FProfPrefillGpuMs[5] + FProfPrefillGpuMs[6]))
      / FProfPrefillChunks;
  end;

  if FProfTokens <= 0 then
    Exit;

  // Convert accumulated totals to per-token averages
  LInv := 1.0 / FProfTokens;

  AStats.Phases.UploadMs := FProfUploadMs * LInv;
  AStats.Phases.CmdRecordMs := FProfRecordMs * LInv;
  AStats.Phases.FenceMs := FProfFenceMs * LInv;
  AStats.Phases.DownloadMs := FProfDownloadMs * LInv;
  AStats.Phases.SampleMs := FProfSampleMs * LInv;
  AStats.Phases.TotalMs := (FProfUploadMs + FProfRecordMs + FProfFenceMs +
    FProfDownloadMs + FProfSampleMs) * LInv;

  // GPU-internal split is only available on the GPU path with a working
  // timestamp pool (accumulators stay zero otherwise)
  AStats.HasGpuStats := FUseGPU and (FProfGpuMs[7] > 0.0);
  if not AStats.HasGpuStats then
    Exit;

  AStats.Gpu.EmbedPlePreMs := FProfGpuMs[0] * LInv;
  AStats.Gpu.FullAttnMs := FProfGpuMs[1] * LInv;
  AStats.Gpu.FullMlpMs := FProfGpuMs[2] * LInv;
  AStats.Gpu.FullPleMs := FProfGpuMs[3] * LInv;
  AStats.Gpu.SlidAttnMs := FProfGpuMs[4] * LInv;
  AStats.Gpu.SlidMlpMs := FProfGpuMs[5] * LInv;
  AStats.Gpu.SlidPleMs := FProfGpuMs[6] * LInv;
  AStats.Gpu.LayersTotalMs := FProfGpuMs[7] * LInv;
  AStats.Gpu.FinalLmHeadMs := FProfGpuMs[8] * LInv;

  // Project the two instrumented layers across the 7 full + 35 sliding
  // layers as a consistency check against LayersTotalMs
  AStats.Gpu.ExtrapolatedLayersMs :=
    (7.0 * (FProfGpuMs[1] + FProfGpuMs[2] + FProfGpuMs[3]) +
     35.0 * (FProfGpuMs[4] + FProfGpuMs[5] + FProfGpuMs[6])) * LInv;
end;

end.
