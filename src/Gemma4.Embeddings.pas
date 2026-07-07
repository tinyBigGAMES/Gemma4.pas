{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information

 -------------------------------------------------------------------------------

  Gemma4.Embeddings - EmbeddingGemma-300m text embedding engine

  Single-pass bidirectional GPU encoder (Gemma 3 architecture) + CPU head
  (mean pool -> dense 768->3072 -> dense 3072->768 -> L2 normalize).
  F32 throughout -- no quantization. TModel/TInference are untouched.

  Dependencies: StdApp.Base, StdApp.JSON, StdApp.VirtualMemory, StdApp.VFS,
    Gemma4.Types, Gemma4.Tensors, Gemma4.Tokenizer, Gemma4.Vulkan,
    Gemma4.Compute
===============================================================================}

unit Gemma4.Embeddings;

{$I StdApp.Defines.inc}

interface

uses
  System.SysUtils,
  System.Classes,
  System.Math,
  StdApp.Base,
  StdApp.JSON,
  StdApp.VirtualMemory,
  StdApp.VFS,
  Gemma4.Types,
  Gemma4.Tensors,
  Gemma4.Tokenizer,
  Gemma4.Vulkan,
  Gemma4.Compute;

const
  CEMB_ERR_OPEN = 'EM01';
  CEMB_ERR_FORWARD = 'EM02';

  CEmbedManifestPath = 'embeddings/manifest.json';
  CEmbedWeightsPath = 'embeddings/weights.bin';
  CEmbedConfigPath = 'embeddings/config.json';
  CEmbedTokenizerPath = 'embeddings/tokenizer.json';
  CEmbedQueryPrefix = 'task: search result | query: '; // Exact sentence-transformers prompt strings (config_sentence_transformers.json)
  CEmbedDocumentPrefix = 'title: none | text: ';
  CEmbedMaxTokens = 2048;
  CEmbedBosId = 2;   // add_bos_token: true
  CEmbedEosId = 1;   // add_eos_token: true

type
  { TEmbedConfig }
  // Parsed embeddings/config.json (Gemma3TextModel)
  TEmbedConfig = record
    HiddenSize: Integer;
    NumLayers: Integer;
    NumHeads: Integer;
    NumKvHeads: Integer;
    HeadDim: Integer;
    IntermediateSize: Integer;
    SlidingWindow: Integer;
    MaxPositions: Integer;
    RmsNormEps: Single;
    RopeThetaSliding: Single;
    RopeThetaFull: Single;
    AttnScale: Single;                 // query_pre_attn_scalar^-0.5
    LayerTypes: TArray<TAttentionKind>;
    function LoadFromJson(const AJsonStr: string): Boolean;
  end;

  { TEmbeddings }
  // Text embedding engine. Open() once, Embed*() many, Close() when done.
  // GPU-only: Vulkan init failure is a hard Open() failure (no CPU path).
  TEmbeddings = class(TBaseObject)
  private
    FWeightStore: TWeightStore;
    FTokenizer: TTokenizer;
    FConfig: TEmbedConfig;
    FDevice: TVulkanDevice;
    FKernels: TComputeKernels;
    FIsLoaded: Boolean;

    // GPU state
    FGpuWeights: TVulkanBuffer;        // whole F32 blob, device-local
    FGpuHidden: TVulkanBuffer;         // [N x 768] residual stream
    FGpuNorm: TVulkanBuffer;           // [N x 768]
    FGpuTmp: TVulkanBuffer;            // [N x 768]
    FGpuQ: TVulkanBuffer;              // [N x 768]; reused as attn-out
    FGpuK: TVulkanBuffer;              // [N x 256] full-sequence keys
    FGpuV: TVulkanBuffer;              // [N x 256] full-sequence values
    FGpuScores: TVulkanBuffer;         // [N x 3 x N] (50 MB at N=2048)
    FGpuMlpGate: TVulkanBuffer;        // [N x 1152]
    FGpuMlpUp: TVulkanBuffer;          // [N x 1152]
    FGpuEmbedIn: TVulkanBuffer;        // [N x 768] host-visible upload
    FGpuHiddenOut: TVulkanBuffer;      // [N x 768] host-visible download
    FLoadProgress: TCallback<TLoadProgressCallback>;

    function DoReadVpkFileAsString(const AVpkPath: string;
      const AInternalPath: string): string;
    function DoInitGpu(): Boolean;
    procedure DoShutdownGpu();
    function DoUploadWeights(): Boolean;
    function DoAllocateBuffers(): Boolean;
    procedure DoFreeBuffers();
    function DoTensorLoc(const AName: string;
      out AOfs: UInt64; out ASize: UInt64): Boolean;
    procedure DoRecordNormRows(const ATensorName: string;
      const ABuf: TVulkanBuffer; const ARowSize: Integer;
      const ANumRows: Integer);
    procedure DoRecordProj(const ATensorName: string;
      const AIn: TVulkanBuffer; const AOut: TVulkanBuffer;
      const ARows: Integer; const ACols: Integer; const ABatch: Integer);
    procedure DoRecordLayer(const ALayerIdx: Integer; const ASeqLen: Integer);
    function DoForward(const ATokens: TArray<Integer>;
      out AHidden: TArray<Single>): Boolean;
    procedure DoCpuHead(const AHidden: TArray<Single>;
      const ASeqLen: Integer; out AResult: TArray<Single>);
  public
    constructor Create(); override;
    destructor Destroy(); override;

    procedure SetStatusCallback(const ACallback: TStatusCallback;
      const AUserData: Pointer = nil); override;
    procedure SetLoadProgressCallback(const ACallback: TLoadProgressCallback;
      const AUserData: Pointer = nil);

    function Open(const AVpkPath: string): Boolean;
    procedure Close();
    function IsLoaded(): Boolean;

    // Raw text -> 768-dim L2-normalized embedding (empty array on failure)
    function Embed(const AText: string): TArray<Single>;
    // Retrieval helpers with the model's exact prompt prefixes
    function EmbedQuery(const AText: string): TArray<Single>;
    function EmbedDocument(const AText: string): TArray<Single>;

    // Cosine similarity of two normalized embeddings
    class function Similarity(const AA: TArray<Single>;
      const AB: TArray<Single>): Single; static;
  end;

implementation

{ TEmbedConfig }

function TEmbedConfig.LoadFromJson(const AJsonStr: string): Boolean;
var
  LJson: TJSON;
  LItems: TArray<TJSON>;
  LI: Integer;
  LKind: string;
begin
  Result := False;
  LJson := TJSON.FromString(AJsonStr);
  try
    if LJson.IsNull() then
      Exit;

    HiddenSize := LJson.Get('hidden_size').AsInt32(0);
    NumLayers := LJson.Get('num_hidden_layers').AsInt32(0);
    NumHeads := LJson.Get('num_attention_heads').AsInt32(0);
    NumKvHeads := LJson.Get('num_key_value_heads').AsInt32(0);
    HeadDim := LJson.Get('head_dim').AsInt32(0);
    IntermediateSize := LJson.Get('intermediate_size').AsInt32(0);
    SlidingWindow := LJson.Get('sliding_window').AsInt32(0);
    MaxPositions := LJson.Get('max_position_embeddings').AsInt32(0);
    RmsNormEps := LJson.Get('rms_norm_eps').AsSingle(1e-6);
    RopeThetaSliding := LJson.Get('rope_local_base_freq').AsSingle(10000.0);
    RopeThetaFull := LJson.Get('rope_theta').AsSingle(1000000.0);
    AttnScale := 1.0 / Sqrt(LJson.Get('query_pre_attn_scalar').AsSingle(256.0));

    // layer_types array: 'sliding_attention' / 'full_attention'
    LItems := LJson.Get('layer_types').Items();
    SetLength(LayerTypes, Length(LItems));
    for LI := 0 to High(LItems) do
    begin
      LKind := LItems[LI].AsString('');
      if LKind = 'full_attention' then
        LayerTypes[LI] := akFull
      else
        LayerTypes[LI] := akSliding;
    end;

    Result := (HiddenSize > 0) and (NumLayers > 0) and
      (Length(LayerTypes) = NumLayers);
  finally
    LJson.Free();
  end;
end;

{ TEmbeddings }

constructor TEmbeddings.Create();
begin
  inherited Create();
  FWeightStore := TWeightStore.Create();
  FTokenizer := TTokenizer.Create();
  FDevice := TVulkanDevice.Create();
  FKernels := TComputeKernels.Create();
  FIsLoaded := False;
  FGpuWeights := Default(TVulkanBuffer);
  FGpuHidden := Default(TVulkanBuffer);
  FGpuNorm := Default(TVulkanBuffer);
  FGpuTmp := Default(TVulkanBuffer);
  FGpuQ := Default(TVulkanBuffer);
  FGpuK := Default(TVulkanBuffer);
  FGpuV := Default(TVulkanBuffer);
  FGpuScores := Default(TVulkanBuffer);
  FGpuMlpGate := Default(TVulkanBuffer);
  FGpuMlpUp := Default(TVulkanBuffer);
  FGpuEmbedIn := Default(TVulkanBuffer);
  FGpuHiddenOut := Default(TVulkanBuffer);
end;

destructor TEmbeddings.Destroy();
begin
  Close();
  // Reverse dependency order: kernels depend on the device
  FKernels.Free();
  FDevice.Free();
  FTokenizer.Free();
  FWeightStore.Free();
  inherited;
end;

procedure TEmbeddings.SetStatusCallback(const ACallback: TStatusCallback;
  const AUserData: Pointer);
begin
  inherited SetStatusCallback(ACallback, AUserData);
  FWeightStore.SetStatusCallback(ACallback, AUserData);
  FTokenizer.SetStatusCallback(ACallback, AUserData);
  FDevice.SetStatusCallback(ACallback, AUserData);
  FKernels.SetStatusCallback(ACallback, AUserData);
end;

procedure TEmbeddings.SetLoadProgressCallback(
  const ACallback: TLoadProgressCallback;
  const AUserData: Pointer);
begin
  FLoadProgress.Callback := ACallback;
  FLoadProgress.UserData := AUserData;
end;

function TEmbeddings.DoReadVpkFileAsString(const AVpkPath: string;
  const AInternalPath: string): string;
var
  LVFS: TVFS;
  LView: TVirtualMemoryView<Byte>;
  LBytes: TBytes;
begin
  Result := '';
  LVFS := TVFS.Create();
  try
    LVFS.SetErrors(GetErrors());
    if not LVFS.Open(AVpkPath) then
      Exit;

    if not LVFS.FileExists(AInternalPath) then
    begin
      LVFS.Close();
      Exit;
    end;

    LView := LVFS.OpenFile(AInternalPath);
    try
      SetLength(LBytes, LView.Size);
      if LView.Size > 0 then
        LView.Read(LBytes[0], LView.Size);
      Result := TEncoding.UTF8.GetString(LBytes);
    finally
      LView.Free();
    end;

    LVFS.Close();
  finally
    LVFS.Free();
  end;
end;

function TEmbeddings.DoInitGpu(): Boolean;
begin
  Result := False;

  FDevice.SetErrors(GetErrors());
  if not FDevice.Init() then
  begin
    GetErrors().Add(esError, CEMB_ERR_OPEN, 'Vulkan init failed');
    Exit;
  end;

  Status('Vulkan initialized: %s', [FDevice.DeviceName]);

  FKernels.SetErrors(GetErrors());
  if not FKernels.Init(FDevice) then
  begin
    GetErrors().Add(esError, CEMB_ERR_OPEN, 'Compute kernel init failed');
    FDevice.Shutdown();
    Exit;
  end;

  Result := True;
end;

procedure TEmbeddings.DoShutdownGpu();
begin
  if FKernels.IsInitialized() then
    FKernels.Shutdown();

  if FDevice.IsInitialized() then
    FDevice.Shutdown();
end;

function TEmbeddings.DoTensorLoc(const AName: string;
  out AOfs: UInt64; out ASize: UInt64): Boolean;
var
  LInfo: TTensorInfo;
begin
  Result := FWeightStore.FindTensor(AName, LInfo);
  if Result then
  begin
    AOfs := LInfo.Offset;
    ASize := LInfo.DataSize;
  end;
end;

function TEmbeddings.DoUploadWeights(): Boolean;
var
  LInfo: TTensorInfo;
  LI: Integer;
  LTotal: UInt64;
  LBase: Pointer;
begin
  Result := False;

  // Blob size = max(offset + size); blob base = pointer of the offset-0
  // tensor (contiguous by construction of PackEmbeddings)
  LTotal := 0;
  LBase := nil;
  for LI := 0 to FWeightStore.TensorCount() - 1 do
  begin
    LInfo := FWeightStore.GetTensorInfo(LI);
    if LInfo.Offset + LInfo.DataSize > LTotal then
      LTotal := LInfo.Offset + LInfo.DataSize;
    if LInfo.Offset = 0 then
      LBase := FWeightStore.GetRawPointer(LInfo);
  end;
  if (LBase = nil) or (LTotal = 0) then
  begin
    GetErrors().Add(esError, CEMB_ERR_OPEN, 'Empty embeddings manifest');
    Exit;
  end;

  FGpuWeights := FDevice.CreateBuffer(LTotal, False);
  if FGpuWeights.Buffer = 0 then
  begin
    GetErrors().Add(esError, CEMB_ERR_OPEN,
      'Failed to create embeddings weight buffer (%.1f MB)',
      [LTotal / (1024 * 1024)]);
    Exit;
  end;

  // Placement matters: the encoder streams these weights per layer
  if FGpuWeights.IsDeviceLocal then
    Status('Embeddings weights buffer: DEVICE-LOCAL (VRAM)')
  else
    Status('WARNING: embeddings weights buffer fell back to HOST memory');

  FDevice.UploadToDeviceBuffer(FGpuWeights, LBase, LTotal);

  Status('Embeddings weights uploaded: %.1f MB F32', [LTotal / (1024 * 1024)]);
  Result := True;
end;

function TEmbeddings.DoAllocateBuffers(): Boolean;
var
  LN: UInt64;
  LKv: UInt64;
begin
  LN := CEmbedMaxTokens;
  LKv := UInt64(FConfig.NumKvHeads * FConfig.HeadDim);

  FGpuHidden := FDevice.CreateBuffer(LN * UInt64(FConfig.HiddenSize) * SizeOf(Single), False);
  FGpuNorm := FDevice.CreateBuffer(LN * UInt64(FConfig.HiddenSize) * SizeOf(Single), False);
  FGpuTmp := FDevice.CreateBuffer(LN * UInt64(FConfig.HiddenSize) * SizeOf(Single), False);
  FGpuQ := FDevice.CreateBuffer(LN * UInt64(FConfig.NumHeads * FConfig.HeadDim) * SizeOf(Single), False);
  FGpuK := FDevice.CreateBuffer(LN * LKv * SizeOf(Single), False);
  FGpuV := FDevice.CreateBuffer(LN * LKv * SizeOf(Single), False);
  FGpuScores := FDevice.CreateBuffer(LN * UInt64(FConfig.NumHeads) * LN * SizeOf(Single), False);
  FGpuMlpGate := FDevice.CreateBuffer(LN * UInt64(FConfig.IntermediateSize) * SizeOf(Single), False);
  FGpuMlpUp := FDevice.CreateBuffer(LN * UInt64(FConfig.IntermediateSize) * SizeOf(Single), False);
  FGpuEmbedIn := FDevice.CreateBuffer(LN * UInt64(FConfig.HiddenSize) * SizeOf(Single), True);
  FGpuHiddenOut := FDevice.CreateBuffer(LN * UInt64(FConfig.HiddenSize) * SizeOf(Single), True);

  Result := (FGpuHidden.Buffer <> 0) and (FGpuNorm.Buffer <> 0) and
    (FGpuTmp.Buffer <> 0) and (FGpuQ.Buffer <> 0) and (FGpuK.Buffer <> 0) and
    (FGpuV.Buffer <> 0) and (FGpuScores.Buffer <> 0) and
    (FGpuMlpGate.Buffer <> 0) and (FGpuMlpUp.Buffer <> 0) and
    (FGpuEmbedIn.Buffer <> 0) and (FGpuHiddenOut.Buffer <> 0);
  if not Result then
    GetErrors().Add(esError, CEMB_ERR_OPEN, 'Embeddings GPU buffers failed');
end;

procedure TEmbeddings.DoFreeBuffers();
begin
  if not FDevice.IsInitialized() then
    Exit;
  if FGpuWeights.Buffer <> 0 then FDevice.DestroyBuffer(FGpuWeights);
  if FGpuHidden.Buffer <> 0 then FDevice.DestroyBuffer(FGpuHidden);
  if FGpuNorm.Buffer <> 0 then FDevice.DestroyBuffer(FGpuNorm);
  if FGpuTmp.Buffer <> 0 then FDevice.DestroyBuffer(FGpuTmp);
  if FGpuQ.Buffer <> 0 then FDevice.DestroyBuffer(FGpuQ);
  if FGpuK.Buffer <> 0 then FDevice.DestroyBuffer(FGpuK);
  if FGpuV.Buffer <> 0 then FDevice.DestroyBuffer(FGpuV);
  if FGpuScores.Buffer <> 0 then FDevice.DestroyBuffer(FGpuScores);
  if FGpuMlpGate.Buffer <> 0 then FDevice.DestroyBuffer(FGpuMlpGate);
  if FGpuMlpUp.Buffer <> 0 then FDevice.DestroyBuffer(FGpuMlpUp);
  if FGpuEmbedIn.Buffer <> 0 then FDevice.DestroyBuffer(FGpuEmbedIn);
  if FGpuHiddenOut.Buffer <> 0 then FDevice.DestroyBuffer(FGpuHiddenOut);
end;

function TEmbeddings.Open(const AVpkPath: string): Boolean;
var
  LConfigStr: string;
  LTokenizerStr: string;
begin
  Result := False;

  if FIsLoaded then
    Close();

  if FLoadProgress.IsAssigned() then
    FLoadProgress.Callback(psStart, 0, 0, FLoadProgress.UserData);

  // Config
  LConfigStr := DoReadVpkFileAsString(AVpkPath, CEmbedConfigPath);
  if LConfigStr = '' then
  begin
    GetErrors().Add(esError, CEMB_ERR_OPEN, 'Missing %s', [CEmbedConfigPath]);
    Exit;
  end;
  if not FConfig.LoadFromJson(LConfigStr) then
  begin
    GetErrors().Add(esError, CEMB_ERR_OPEN, 'Bad embeddings config');
    Exit;
  end;

  // Tokenizer
  LTokenizerStr := DoReadVpkFileAsString(AVpkPath, CEmbedTokenizerPath);
  if (LTokenizerStr = '') or (not FTokenizer.LoadFromString(LTokenizerStr)) then
  begin
    GetErrors().Add(esError, CEMB_ERR_OPEN, 'Tokenizer load failed');
    Exit;
  end;

  // Weights (folder-aware overload)
  if not FWeightStore.Open(AVpkPath, CEmbedManifestPath, CEmbedWeightsPath) then
  begin
    GetErrors().Add(esError, CEMB_ERR_OPEN, 'Weight store open failed');
    Exit;
  end;

  // GPU
  if not DoInitGpu() then Exit;
  if not DoUploadWeights() then Exit;

  if FLoadProgress.IsAssigned() then
    FLoadProgress.Callback(psInProgress, 1, 1, FLoadProgress.UserData);

  if not DoAllocateBuffers() then Exit;

  FIsLoaded := True;
  Status('Embeddings ready: %d layers, hidden %d, max %d tokens',
    [FConfig.NumLayers, FConfig.HiddenSize, CEmbedMaxTokens]);

  if FLoadProgress.IsAssigned() then
    FLoadProgress.Callback(psEnd, 0, 0, FLoadProgress.UserData);

  Result := True;
end;

procedure TEmbeddings.Close();
begin
  DoFreeBuffers();
  DoShutdownGpu();
  FWeightStore.Close();
  FIsLoaded := False;
end;

function TEmbeddings.IsLoaded(): Boolean;
begin
  Result := FIsLoaded;
end;

procedure TEmbeddings.DoRecordNormRows(const ATensorName: string;
  const ABuf: TVulkanBuffer; const ARowSize: Integer;
  const ANumRows: Integer);
var
  LOfs: UInt64;
  LSize: UInt64;
begin
  if not DoTensorLoc(ATensorName, LOfs, LSize) then Exit;
  // Weighted per-row RMS norm against gamma at a byte offset in the blob
  FKernels.RecordRmsNormRows(ABuf, FGpuWeights, LOfs, LSize,
    UInt32(ARowSize), UInt32(ANumRows), FConfig.RmsNormEps);
end;

procedure TEmbeddings.DoRecordProj(const ATensorName: string;
  const AIn: TVulkanBuffer; const AOut: TVulkanBuffer;
  const ARows: Integer; const ACols: Integer; const ABatch: Integer);
var
  LOfs: UInt64;
  LSize: UInt64;
begin
  if not DoTensorLoc(ATensorName, LOfs, LSize) then Exit;
  FKernels.RecordMatMatF32(FGpuWeights, LOfs, LSize, AIn, AOut,
    UInt32(ARows), UInt32(ACols), UInt32(ABatch));
end;

procedure TEmbeddings.DoRecordLayer(const ALayerIdx: Integer;
  const ASeqLen: Integer);
var
  LPrefix: string;
  LTheta: Single;
  LWindow: Integer;
  LPush: TAttnBidirPushConstants;
begin
  LPrefix := Format('layers.%d.', [ALayerIdx]);

  if FConfig.LayerTypes[ALayerIdx] = akFull then
  begin
    LTheta := FConfig.RopeThetaFull;
    LWindow := 0;                       // 0 = fully bidirectional
  end
  else
  begin
    LTheta := FConfig.RopeThetaSliding;
    LWindow := FConfig.SlidingWindow;   // abs(m - t) < window
  end;

  // Attention block (sandwich norm)
  FKernels.RecordCopy(FGpuNorm, FGpuHidden,
    UInt32(ASeqLen * FConfig.HiddenSize));
  FKernels.RecordBarrier();
  DoRecordNormRows(LPrefix + 'input_layernorm.weight', FGpuNorm,
    FConfig.HiddenSize, ASeqLen);
  FKernels.RecordBarrier();

  DoRecordProj(LPrefix + 'self_attn.q_proj.weight', FGpuNorm, FGpuQ,
    FConfig.NumHeads * FConfig.HeadDim, FConfig.HiddenSize, ASeqLen);
  DoRecordProj(LPrefix + 'self_attn.k_proj.weight', FGpuNorm, FGpuK,
    FConfig.NumKvHeads * FConfig.HeadDim, FConfig.HiddenSize, ASeqLen);
  DoRecordProj(LPrefix + 'self_attn.v_proj.weight', FGpuNorm, FGpuV,
    FConfig.NumKvHeads * FConfig.HeadDim, FConfig.HiddenSize, ASeqLen);
  FKernels.RecordBarrier();

  DoRecordNormRows(LPrefix + 'self_attn.q_norm.weight', FGpuQ,
    FConfig.HeadDim, ASeqLen * FConfig.NumHeads);
  DoRecordNormRows(LPrefix + 'self_attn.k_norm.weight', FGpuK,
    FConfig.HeadDim, ASeqLen * FConfig.NumKvHeads);
  FKernels.RecordBarrier();

  // Full rotary: rotAngles = headDim / 2; positions 0..seq-1
  FKernels.RecordRoPEBatch(FGpuQ, 0, UInt32(ASeqLen),
    UInt32(FConfig.HeadDim), UInt32(FConfig.HeadDim div 2),
    UInt32(FConfig.NumHeads), LTheta);
  FKernels.RecordRoPEBatch(FGpuK, 0, UInt32(ASeqLen),
    UInt32(FConfig.HeadDim), UInt32(FConfig.HeadDim div 2),
    UInt32(FConfig.NumKvHeads), LTheta);
  FKernels.RecordBarrier();

  LPush.headDim := UInt32(FConfig.HeadDim);
  LPush.numHeads := UInt32(FConfig.NumHeads);
  LPush.numKvHeads := UInt32(FConfig.NumKvHeads);
  LPush.kvStride := UInt32(FConfig.NumKvHeads * FConfig.HeadDim);
  LPush.seqLen := UInt32(ASeqLen);
  LPush.windowSize := UInt32(LWindow);
  LPush.scale := FConfig.AttnScale;
  LPush.rowPitch := UInt32(ASeqLen);

  FKernels.RecordAttnScoresBidir(FGpuQ, FGpuK, FGpuScores, LPush);
  FKernels.RecordBarrier();
  FKernels.RecordSoftmaxRows(FGpuScores, UInt32(ASeqLen),
    UInt32(ASeqLen * FConfig.NumHeads));
  FKernels.RecordBarrier();
  // FGpuQ is free after scores: reuse it as the attention output buffer
  FKernels.RecordAttnOutBidir(FGpuScores, FGpuV, FGpuQ, LPush);
  FKernels.RecordBarrier();

  DoRecordProj(LPrefix + 'self_attn.o_proj.weight', FGpuQ, FGpuTmp,
    FConfig.HiddenSize, FConfig.NumHeads * FConfig.HeadDim, ASeqLen);
  FKernels.RecordBarrier();
  DoRecordNormRows(LPrefix + 'post_attention_layernorm.weight', FGpuTmp,
    FConfig.HiddenSize, ASeqLen);
  FKernels.RecordBarrier();
  FKernels.RecordAdd(FGpuHidden, FGpuTmp,
    UInt32(ASeqLen * FConfig.HiddenSize));
  FKernels.RecordBarrier();

  // MLP block (sandwich norm)
  FKernels.RecordCopy(FGpuNorm, FGpuHidden,
    UInt32(ASeqLen * FConfig.HiddenSize));
  FKernels.RecordBarrier();
  DoRecordNormRows(LPrefix + 'pre_feedforward_layernorm.weight', FGpuNorm,
    FConfig.HiddenSize, ASeqLen);
  FKernels.RecordBarrier();
  DoRecordProj(LPrefix + 'mlp.gate_proj.weight', FGpuNorm, FGpuMlpGate,
    FConfig.IntermediateSize, FConfig.HiddenSize, ASeqLen);
  DoRecordProj(LPrefix + 'mlp.up_proj.weight', FGpuNorm, FGpuMlpUp,
    FConfig.IntermediateSize, FConfig.HiddenSize, ASeqLen);
  FKernels.RecordBarrier();
  FKernels.RecordGeGLU(FGpuMlpGate, FGpuMlpUp,
    UInt32(ASeqLen * FConfig.IntermediateSize), 0);
  FKernels.RecordBarrier();
  DoRecordProj(LPrefix + 'mlp.down_proj.weight', FGpuMlpGate, FGpuTmp,
    FConfig.HiddenSize, FConfig.IntermediateSize, ASeqLen);
  FKernels.RecordBarrier();
  DoRecordNormRows(LPrefix + 'post_feedforward_layernorm.weight', FGpuTmp,
    FConfig.HiddenSize, ASeqLen);
  FKernels.RecordBarrier();
  FKernels.RecordAdd(FGpuHidden, FGpuTmp,
    UInt32(ASeqLen * FConfig.HiddenSize));
  FKernels.RecordBarrier();
end;

function TEmbeddings.DoForward(const ATokens: TArray<Integer>;
  out AHidden: TArray<Single>): Boolean;
var
  LN: Integer;
  LI: Integer;
  LInfo: TTensorInfo;
  LPtr: PSingle;
  LStage: TArray<Single>;
  LScale: Single;
  LJ: Integer;
begin
  Result := False;
  LN := Length(ATokens);
  if (LN <= 0) or (LN > CEmbedMaxTokens) then
  begin
    GetErrors().Add(esError, CEMB_ERR_FORWARD,
      'Token count %d out of range 1..%d', [LN, CEmbedMaxTokens]);
    Exit;
  end;

  // CPU: embedding rows * sqrt(hidden) (Gemma3TextScaledWordEmbedding)
  if not FWeightStore.FindTensor('embed_tokens.weight', LInfo) then
  begin
    GetErrors().Add(esError, CEMB_ERR_FORWARD, 'embed_tokens.weight missing');
    Exit;
  end;
  LPtr := PSingle(FWeightStore.GetRawPointer(LInfo));
  LScale := Sqrt(Single(FConfig.HiddenSize));
  SetLength(LStage, LN * FConfig.HiddenSize);
  for LI := 0 to LN - 1 do
    for LJ := 0 to FConfig.HiddenSize - 1 do
      LStage[LI * FConfig.HiddenSize + LJ] :=
        PSingle(UIntPtr(LPtr) + UIntPtr(
          (UInt64(ATokens[LI]) * UInt64(FConfig.HiddenSize) + UInt64(LJ)) *
          SizeOf(Single)))^ * LScale;

  FKernels.Shaders.ResetDescriptorPool();
  FDevice.UploadToBuffer(FGpuEmbedIn, @LStage[0],
    UInt64(LN) * UInt64(FConfig.HiddenSize) * SizeOf(Single));

  FDevice.BeginCommands();
  FKernels.RecordCopy(FGpuHidden, FGpuEmbedIn,
    UInt32(LN * FConfig.HiddenSize));
  FKernels.RecordBarrier();

  for LI := 0 to FConfig.NumLayers - 1 do
    DoRecordLayer(LI, LN);

  // Final norm, then copy to the host-visible download buffer
  DoRecordNormRows('norm.weight', FGpuHidden, FConfig.HiddenSize, LN);
  FKernels.RecordBarrier();
  FKernels.RecordCopy(FGpuHiddenOut, FGpuHidden,
    UInt32(LN * FConfig.HiddenSize));

  FDevice.EndCommands();
  FDevice.SubmitAndWait();

  SetLength(AHidden, LN * FConfig.HiddenSize);
  FDevice.DownloadFromBuffer(FGpuHiddenOut, @AHidden[0],
    UInt64(LN) * UInt64(FConfig.HiddenSize) * SizeOf(Single));
  Result := True;
end;

procedure TEmbeddings.DoCpuHead(const AHidden: TArray<Single>;
  const ASeqLen: Integer; out AResult: TArray<Single>);
var
  LMean: TArray<Single>;
  LMid: TArray<Single>;
  LInfo2: TTensorInfo;
  LInfo3: TTensorInfo;
  LW2: PSingle;
  LW3: PSingle;
  LI: Integer;
  LJ: Integer;
  LSum: Double;
  LNorm: Double;
begin
  SetLength(AResult, 0);

  // Mean pool over all positions (include_prompt = true)
  SetLength(LMean, FConfig.HiddenSize);
  for LJ := 0 to FConfig.HiddenSize - 1 do
  begin
    LSum := 0.0;
    for LI := 0 to ASeqLen - 1 do
      LSum := LSum + AHidden[LI * FConfig.HiddenSize + LJ];
    LMean[LJ] := LSum / ASeqLen;
  end;

  if not FWeightStore.FindTensor('dense_2.linear.weight', LInfo2) then Exit;
  if not FWeightStore.FindTensor('dense_3.linear.weight', LInfo3) then Exit;
  LW2 := PSingle(FWeightStore.GetRawPointer(LInfo2));
  LW3 := PSingle(FWeightStore.GetRawPointer(LInfo3));

  // dense_2: [3072 x 768], no bias
  SetLength(LMid, 3072);
  for LI := 0 to 3071 do
  begin
    LSum := 0.0;
    for LJ := 0 to FConfig.HiddenSize - 1 do
      LSum := LSum + PSingle(UIntPtr(LW2) + UIntPtr(
        (UInt64(LI) * UInt64(FConfig.HiddenSize) + UInt64(LJ)) *
        SizeOf(Single)))^ * LMean[LJ];
    LMid[LI] := LSum;
  end;

  // dense_3: [768 x 3072], no bias
  SetLength(AResult, FConfig.HiddenSize);
  for LI := 0 to FConfig.HiddenSize - 1 do
  begin
    LSum := 0.0;
    for LJ := 0 to 3071 do
      LSum := LSum + PSingle(UIntPtr(LW3) + UIntPtr(
        (UInt64(LI) * 3072 + UInt64(LJ)) * SizeOf(Single)))^ * LMid[LJ];
    AResult[LI] := LSum;
  end;

  // L2 normalize
  LNorm := 0.0;
  for LI := 0 to FConfig.HiddenSize - 1 do
    LNorm := LNorm + Double(AResult[LI]) * Double(AResult[LI]);
  LNorm := Sqrt(LNorm);
  if LNorm > 0.0 then
    for LI := 0 to FConfig.HiddenSize - 1 do
      AResult[LI] := AResult[LI] / LNorm;
end;

function TEmbeddings.Embed(const AText: string): TArray<Single>;
var
  LIds: TArray<Integer>;
  LTokens: TArray<Integer>;
  LHidden: TArray<Single>;
  LI: Integer;
begin
  SetLength(Result, 0);
  if not FIsLoaded then Exit;

  // Tokenize; specials added manually (add_bos_token + add_eos_token true)
  LIds := FTokenizer.Encode(AText);
  SetLength(LTokens, Length(LIds) + 2);
  LTokens[0] := CEmbedBosId;
  for LI := 0 to High(LIds) do
    LTokens[LI + 1] := LIds[LI];
  LTokens[High(LTokens)] := CEmbedEosId;

  // Truncate to the model's max context (keep BOS..., force final EOS)
  if Length(LTokens) > CEmbedMaxTokens then
  begin
    SetLength(LTokens, CEmbedMaxTokens);
    LTokens[CEmbedMaxTokens - 1] := CEmbedEosId;
  end;

  if not DoForward(LTokens, LHidden) then Exit;
  DoCpuHead(LHidden, Length(LTokens), Result);
end;

function TEmbeddings.EmbedQuery(const AText: string): TArray<Single>;
begin
  Result := Embed(CEmbedQueryPrefix + AText);
end;

function TEmbeddings.EmbedDocument(const AText: string): TArray<Single>;
begin
  Result := Embed(CEmbedDocumentPrefix + AText);
end;

class function TEmbeddings.Similarity(const AA: TArray<Single>;
  const AB: TArray<Single>): Single;
var
  LI: Integer;
  LSum: Double;
begin
  Result := 0.0;
  if (Length(AA) = 0) or (Length(AA) <> Length(AB)) then Exit;
  LSum := 0.0;
  for LI := 0 to High(AA) do
    LSum := LSum + Double(AA[LI]) * Double(AB[LI]);
  Result := LSum;
end;

end.
