{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information

 -------------------------------------------------------------------------------

  Gemma4.Vision - Gemma 4 vision encoder (image -> soft tokens)

  GPU F32 implementation of the Gemma 4 vision tower on E4B/encoders.bin:
  patch embedding (input_proj + learned 2D position table), 16 bidirectional
  encoder layers (sandwich norms, q/k/v norms, 2D RoPE theta=100, GeGLU,
  QAT clip clamps on every linear), then 3x3 spatial pooling * sqrt(768)
  (CPU) and the multimodal embedder (weight-free norm + [2560x768]
  projection, GPU) into language-model space.

  EncodeImage input: patchified pixels in [0,1] ([N x 768], 3*16*16 per
  patch) + per-patch (x, y) grid positions. Output: [N/9 x 2560] soft-token
  rows for TModel.Generate's ASoftBlocks substitution list (280 rows for a
  standard image, 70 per video frame later).

  Pass a shared TComputeKernels (TInference's) to Open() to reuse the live
  Vulkan device; with nil, TVision owns its own device (standalone tests).
  In shared mode, free/close TVision BEFORE the owning kernels/device.

  Dependencies: StdApp.Base, StdApp.JSON, StdApp.VFS, StdApp.VirtualMemory,
    Gemma4.Types, Gemma4.Tensors, Gemma4.Vulkan, Gemma4.Compute
===============================================================================}

unit Gemma4.Vision;

{$I StdApp.Defines.inc}

interface

uses
  System.SysUtils,
  System.Math,
  StdApp.Base,
  StdApp.JSON,
  StdApp.VFS,
  StdApp.VirtualMemory,
  Gemma4.Types,
  Gemma4.Tensors,
  Gemma4.Vulkan,
  Gemma4.Compute;

const
  CVIS_ERR_OPEN = 'VS01';
  CVIS_ERR_ENCODE = 'VS02';

  CVisManifestPath = 'E4B/encoders_manifest.json';
  CVisWeightsPath = 'E4B/encoders.bin';
  CVisConfigPath = 'E4B/config.json';
  CVisPrefix = 'model.vision_tower.';
  CVisEmbedPrefix = 'model.embed_vision.';
  CVisEmbedProjName = 'model.embed_vision.embedding_projection.weight';

  // 280 soft tokens * 3^2 pooling = 2520 patches for a standard image
  CVisMaxPatches = 2520;

type
  { TVisionConfig }
  // Parsed vision_config from E4B/config.json plus the text hidden size
  TVisionConfig = record
    HiddenSize: Integer;           // 768
    NumLayers: Integer;            // 16
    NumHeads: Integer;             // 12
    NumKvHeads: Integer;           // 12
    HeadDim: Integer;              // 64
    IntermediateSize: Integer;     // 3072
    PatchSize: Integer;            // 16
    PoolK: Integer;                // 3
    PosTableSize: Integer;         // 10240
    RmsNormEps: Single;            // 1e-6
    RopeTheta: Single;             // 100.0
    TextHiddenSize: Integer;       // 2560 (soft-token width)
    function LoadFromJson(const AJsonStr: string): Boolean;
  end;

  { TClipBounds }
  // QAT clip scalars of one clipped linear (read once from the blob)
  TClipBounds = record
    InMin: Single;
    InMax: Single;
    OutMin: Single;
    OutMax: Single;
  end;

  { TVisionLayerClips }
  // Clip bounds for the seven clipped linears of one encoder layer
  TVisionLayerClips = record
    QProj: TClipBounds;
    KProj: TClipBounds;
    VProj: TClipBounds;
    OProj: TClipBounds;
    GateProj: TClipBounds;
    UpProj: TClipBounds;
    DownProj: TClipBounds;
  end;

  { TVision }
  // Vision encoder engine. Open() once, EncodeImage() per image, Close().
  // GPU-only: Vulkan init failure is a hard Open() failure.
  TVision = class(TBaseObject)
  private
    FWeightStore: TWeightStore;
    FConfig: TVisionConfig;
    FDevice: TVulkanDevice;        // owned only when FOwnsGpu
    FKernels: TComputeKernels;     // owned only when FOwnsGpu
    FOwnsGpu: Boolean;
    FIsLoaded: Boolean;
    FClips: TArray<TVisionLayerClips>;
    FWinBase: UInt64;              // vision byte window base in the blob

    // GPU state
    FGpuWeights: TVulkanBuffer;    // vision window of encoders.bin, F32
    FGpuHidden: TVulkanBuffer;     // [N x 768] residual stream
    FGpuNorm: TVulkanBuffer;       // [N x 768] normed input to q/k/v, mlp
    FGpuTmp: TVulkanBuffer;        // [N x 768] scratch / block output
    FGpuQ: TVulkanBuffer;          // [N x 768]; reused as attention output
    FGpuK: TVulkanBuffer;          // [N x 768]
    FGpuV: TVulkanBuffer;          // [N x 768]
    FGpuScores: TVulkanBuffer;     // [N x 12 x N] (305 MB at N = 2520)
    FGpuMlpGate: TVulkanBuffer;    // [N x 3072]
    FGpuMlpUp: TVulkanBuffer;      // [N x 3072]
    FGpuProj: TVulkanBuffer;       // [280 x 2560] soft-token projection out
    FGpuPos: TVulkanBuffer;        // [N x 2] int32 positions, host-visible
    FGpuStageIn: TVulkanBuffer;    // [N x 768] host-visible upload
    FGpuStagePE: TVulkanBuffer;    // [N x 768] host-visible posemb upload
    FGpuStageOut: TVulkanBuffer;   // [N x 768] host-visible download

    function DoReadVpkFileAsString(const AVpkPath: string;
      const AInternalPath: string): string;
    function DoInitGpu(): Boolean;
    procedure DoShutdownGpu();
    function DoUploadWeights(): Boolean;
    function DoAllocateBuffers(): Boolean;
    procedure DoFreeBuffers();
    function DoLoadClips(): Boolean;
    function DoReadClipScalar(const AName: string;
      out AValue: Single): Boolean;
    function DoTensorLoc(const AName: string;
      out AOfs: UInt64; out ASize: UInt64): Boolean;
    procedure DoRecordNormRows(const ATensorName: string;
      const ABuf: TVulkanBuffer; const ARowSize: Integer;
      const ANumRows: Integer);
    procedure DoRecordClippedProj(const ATensorName: string;
      const AClip: TClipBounds; const AIn: TVulkanBuffer;
      const AOut: TVulkanBuffer; const ARows: Integer; const ACols: Integer;
      const ABatch: Integer; const AInElems: Integer);
    procedure DoRecordLayer(const ALayerIdx: Integer;
      const ANumPatches: Integer);
    function DoPatchPrep(const APatches: TArray<Single>;
      const APositions: TArray<Integer>; const ANumPatches: Integer): Boolean;
    function DoEncoderForward(const ANumPatches: Integer;
      out AHidden: TArray<Single>): Boolean;
    function DoPoolCpu(const AHidden: TArray<Single>;
      const APositions: TArray<Integer>; const ANumPatches: Integer;
      out APooled: TArray<Single>; out AOutLen: Integer): Boolean;
    function DoProjectGpu(const APooled: TArray<Single>;
      const AOutLen: Integer; out ASoftTokens: TArray<Single>): Boolean;
  public
    constructor Create(); override;
    destructor Destroy(); override;

    procedure SetStatusCallback(const ACallback: TStatusCallback;
      const AUserData: Pointer = nil); override;

    // Open the VPK's vision weights. AKernels = a live shared TComputeKernels
    // (TInference's) to reuse its device; nil = create an own device.
    function Open(const AVpkPath: string;
      const AKernels: TComputeKernels = nil): Boolean;
    procedure Close();
    function IsLoaded(): Boolean;

    // Patchified pixels in [0,1] ([ANumPatches x 3*16*16]) + per-patch (x, y)
    // grid positions ([ANumPatches x 2]) -> soft tokens
    // [(ANumPatches div 9) x 2560]. False + empty array on failure.
    function EncodeImage(const APatches: TArray<Single>;
      const APositions: TArray<Integer>;
      out ASoftTokens: TArray<Single>): Boolean;

    property Config: TVisionConfig read FConfig;
  end;

implementation

{ TVisionConfig }

function TVisionConfig.LoadFromJson(const AJsonStr: string): Boolean;
var
  LJson: TJSON;
  LVis: TJSON;
begin
  Result := False;
  LJson := TJSON.FromString(AJsonStr);
  try
    if LJson.IsNull() then
      Exit;

    LVis := LJson.Get('vision_config');
    if LVis.IsNull() then
      Exit;

    HiddenSize := LVis.Get('hidden_size').AsInt32(0);
    NumLayers := LVis.Get('num_hidden_layers').AsInt32(0);
    NumHeads := LVis.Get('num_attention_heads').AsInt32(0);
    NumKvHeads := LVis.Get('num_key_value_heads').AsInt32(0);
    HeadDim := LVis.Get('head_dim').AsInt32(0);
    IntermediateSize := LVis.Get('intermediate_size').AsInt32(0);
    PatchSize := LVis.Get('patch_size').AsInt32(0);
    PoolK := LVis.Get('pooling_kernel_size').AsInt32(3);
    PosTableSize := LVis.Get('position_embedding_size').AsInt32(0);
    RmsNormEps := LVis.Get('rms_norm_eps').AsSingle(1e-6);
    RopeTheta := LVis.Get('rope_parameters').Get('rope_theta').AsSingle(100.0);
    TextHiddenSize := LJson.Get('text_config').Get('hidden_size').AsInt32(0);

    Result := (HiddenSize = 768) and (NumLayers > 0) and (HeadDim > 0) and
      (PoolK > 0) and (PosTableSize > 0) and (TextHiddenSize > 0);
  finally
    LJson.Free();
  end;
end;

{ TVision }

constructor TVision.Create();
begin
  inherited Create();
  FWeightStore := TWeightStore.Create();
  FDevice := nil;
  FKernels := nil;
  FOwnsGpu := False;
  FIsLoaded := False;
  FWinBase := 0;
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
  FGpuProj := Default(TVulkanBuffer);
  FGpuPos := Default(TVulkanBuffer);
  FGpuStageIn := Default(TVulkanBuffer);
  FGpuStagePE := Default(TVulkanBuffer);
  FGpuStageOut := Default(TVulkanBuffer);
end;

destructor TVision.Destroy();
begin
  Close();
  // Owned GPU objects only; shared kernels/device belong to the caller and
  // MUST outlive this object (free TVision before TInference)
  if FOwnsGpu then
  begin
    FKernels.Free();
    FDevice.Free();
  end;
  FWeightStore.Free();
  inherited;
end;

procedure TVision.SetStatusCallback(const ACallback: TStatusCallback;
  const AUserData: Pointer);
begin
  inherited SetStatusCallback(ACallback, AUserData);
  FWeightStore.SetStatusCallback(ACallback, AUserData);
  if FOwnsGpu then
  begin
    if FDevice <> nil then
      FDevice.SetStatusCallback(ACallback, AUserData);
    if FKernels <> nil then
      FKernels.SetStatusCallback(ACallback, AUserData);
  end;
end;

function TVision.DoReadVpkFileAsString(const AVpkPath: string;
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

function TVision.DoInitGpu(): Boolean;
begin
  Result := False;

  if not FOwnsGpu then
  begin
    // Shared mode: the caller's device/kernels are already live
    if (FKernels = nil) or (not FKernels.IsInitialized()) then
    begin
      GetErrors().Add(esError, CVIS_ERR_OPEN, 'Shared kernels not initialized');
      Exit;
    end;
    FDevice := FKernels.Device;
    Exit(True);
  end;

  FDevice := TVulkanDevice.Create();
  FKernels := TComputeKernels.Create();
  FDevice.SetErrors(GetErrors());
  FKernels.SetErrors(GetErrors());

  if not FDevice.Init() then
  begin
    GetErrors().Add(esError, CVIS_ERR_OPEN, 'Vulkan init failed');
    Exit;
  end;
  Status('Vulkan initialized: %s', [FDevice.DeviceName]);

  if not FKernels.Init(FDevice) then
  begin
    GetErrors().Add(esError, CVIS_ERR_OPEN, 'Compute kernel init failed');
    FDevice.Shutdown();
    Exit;
  end;

  Result := True;
end;

procedure TVision.DoShutdownGpu();
begin
  if not FOwnsGpu then
  begin
    FDevice := nil;    // non-owned reference; the caller shuts it down
    Exit;
  end;
  if (FKernels <> nil) and FKernels.IsInitialized() then
    FKernels.Shutdown();
  if (FDevice <> nil) and FDevice.IsInitialized() then
    FDevice.Shutdown();
end;

function TVision.DoTensorLoc(const AName: string;
  out AOfs: UInt64; out ASize: UInt64): Boolean;
var
  LInfo: TTensorInfo;
begin
  Result := FWeightStore.FindTensor(AName, LInfo);
  if Result then
  begin
    AOfs := LInfo.Offset - FWinBase;   // rebased into the uploaded window
    ASize := LInfo.DataSize;
  end;
end;

function TVision.DoUploadWeights(): Boolean;
var
  LInfo: TTensorInfo;
  LI: Integer;
  LMinOfs: UInt64;
  LMaxEnd: UInt64;
  LTotal: UInt64;
  LBase: Pointer;
  LIsVision: Boolean;
begin
  Result := False;

  // The vision tensors (model.vision_tower.* + model.embed_vision.*) occupy
  // a contiguous byte window inside encoders.bin (~677 MB of 1.9 GB). Upload
  // ONLY that window; DoTensorLoc rebases offsets by FWinBase.
  LMinOfs := High(UInt64);
  LMaxEnd := 0;
  for LI := 0 to FWeightStore.TensorCount() - 1 do
  begin
    LInfo := FWeightStore.GetTensorInfo(LI);
    LIsVision := LInfo.TensorName.StartsWith(CVisPrefix) or
      LInfo.TensorName.StartsWith(CVisEmbedPrefix);
    if not LIsVision then
      Continue;
    if LInfo.Offset < LMinOfs then
      LMinOfs := LInfo.Offset;
    if LInfo.Offset + LInfo.DataSize > LMaxEnd then
      LMaxEnd := LInfo.Offset + LInfo.DataSize;
  end;
  if LMaxEnd <= LMinOfs then
  begin
    GetErrors().Add(esError, CVIS_ERR_OPEN, 'No vision tensors in manifest');
    Exit;
  end;

  FWinBase := LMinOfs;
  LTotal := LMaxEnd - LMinOfs;

  // Window base pointer = raw pointer of the tensor sitting AT LMinOfs
  LBase := nil;
  for LI := 0 to FWeightStore.TensorCount() - 1 do
  begin
    LInfo := FWeightStore.GetTensorInfo(LI);
    if LInfo.Offset = LMinOfs then
    begin
      LBase := FWeightStore.GetRawPointer(LInfo);
      Break;
    end;
  end;
  if LBase = nil then
  begin
    GetErrors().Add(esError, CVIS_ERR_OPEN, 'Vision window base not found');
    Exit;
  end;

  FGpuWeights := FDevice.CreateBuffer(LTotal, False);
  if FGpuWeights.Buffer = 0 then
  begin
    GetErrors().Add(esError, CVIS_ERR_OPEN,
      'Vision weight buffer failed (%.1f MB)', [LTotal / (1024 * 1024)]);
    Exit;
  end;

  // Placement matters: the encoder streams these weights per layer
  if FGpuWeights.IsDeviceLocal then
    Status('Vision weights buffer: DEVICE-LOCAL (VRAM)')
  else
    Status('WARNING: vision weights buffer fell back to HOST memory');

  FDevice.UploadToDeviceBuffer(FGpuWeights, LBase, LTotal);
  Status('Vision weights uploaded: %.1f MB F32', [LTotal / (1024 * 1024)]);
  Result := True;
end;

function TVision.DoAllocateBuffers(): Boolean;
var
  LN: UInt64;
  LH: UInt64;
  LMaxSoft: UInt64;
begin
  LN := CVisMaxPatches;
  LH := UInt64(FConfig.HiddenSize);
  LMaxSoft := LN div UInt64(FConfig.PoolK * FConfig.PoolK);

  FGpuHidden := FDevice.CreateBuffer(LN * LH * SizeOf(Single), False);
  FGpuNorm := FDevice.CreateBuffer(LN * LH * SizeOf(Single), False);
  FGpuTmp := FDevice.CreateBuffer(LN * LH * SizeOf(Single), False);
  FGpuQ := FDevice.CreateBuffer(
    LN * UInt64(FConfig.NumHeads * FConfig.HeadDim) * SizeOf(Single), False);
  FGpuK := FDevice.CreateBuffer(
    LN * UInt64(FConfig.NumKvHeads * FConfig.HeadDim) * SizeOf(Single), False);
  FGpuV := FDevice.CreateBuffer(
    LN * UInt64(FConfig.NumKvHeads * FConfig.HeadDim) * SizeOf(Single), False);
  FGpuScores := FDevice.CreateBuffer(
    LN * UInt64(FConfig.NumHeads) * LN * SizeOf(Single), False);
  FGpuMlpGate := FDevice.CreateBuffer(
    LN * UInt64(FConfig.IntermediateSize) * SizeOf(Single), False);
  FGpuMlpUp := FDevice.CreateBuffer(
    LN * UInt64(FConfig.IntermediateSize) * SizeOf(Single), False);
  FGpuProj := FDevice.CreateBuffer(
    LMaxSoft * UInt64(FConfig.TextHiddenSize) * SizeOf(Single), False);
  FGpuPos := FDevice.CreateBuffer(LN * 2 * SizeOf(Int32), True);
  FGpuStageIn := FDevice.CreateBuffer(LN * LH * SizeOf(Single), True);
  FGpuStagePE := FDevice.CreateBuffer(LN * LH * SizeOf(Single), True);
  FGpuStageOut := FDevice.CreateBuffer(LN * LH * SizeOf(Single), True);

  Result := (FGpuHidden.Buffer <> 0) and (FGpuNorm.Buffer <> 0) and
    (FGpuTmp.Buffer <> 0) and (FGpuQ.Buffer <> 0) and (FGpuK.Buffer <> 0) and
    (FGpuV.Buffer <> 0) and (FGpuScores.Buffer <> 0) and
    (FGpuMlpGate.Buffer <> 0) and (FGpuMlpUp.Buffer <> 0) and
    (FGpuProj.Buffer <> 0) and (FGpuPos.Buffer <> 0) and
    (FGpuStageIn.Buffer <> 0) and (FGpuStagePE.Buffer <> 0) and
    (FGpuStageOut.Buffer <> 0);
  if not Result then
    GetErrors().Add(esError, CVIS_ERR_OPEN, 'Vision GPU buffers failed');
end;

procedure TVision.DoFreeBuffers();
begin
  if (FDevice = nil) or (not FDevice.IsInitialized()) then
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
  if FGpuProj.Buffer <> 0 then FDevice.DestroyBuffer(FGpuProj);
  if FGpuPos.Buffer <> 0 then FDevice.DestroyBuffer(FGpuPos);
  if FGpuStageIn.Buffer <> 0 then FDevice.DestroyBuffer(FGpuStageIn);
  if FGpuStagePE.Buffer <> 0 then FDevice.DestroyBuffer(FGpuStagePE);
  if FGpuStageOut.Buffer <> 0 then FDevice.DestroyBuffer(FGpuStageOut);
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
  FGpuProj := Default(TVulkanBuffer);
  FGpuPos := Default(TVulkanBuffer);
  FGpuStageIn := Default(TVulkanBuffer);
  FGpuStagePE := Default(TVulkanBuffer);
  FGpuStageOut := Default(TVulkanBuffer);
end;

function TVision.DoReadClipScalar(const AName: string;
  out AValue: Single): Boolean;
var
  LInfo: TTensorInfo;
begin
  AValue := 0.0;
  Result := FWeightStore.FindTensor(AName, LInfo);
  if Result then
    AValue := PSingle(FWeightStore.GetRawPointer(LInfo))^;
end;

function TVision.DoLoadClips(): Boolean;
var
  LI: Integer;
  LPrefix: string;

  function LoadOne(const ABase: string; out AClip: TClipBounds): Boolean;
  begin
    Result :=
      DoReadClipScalar(ABase + '.input_min', AClip.InMin) and
      DoReadClipScalar(ABase + '.input_max', AClip.InMax) and
      DoReadClipScalar(ABase + '.output_min', AClip.OutMin) and
      DoReadClipScalar(ABase + '.output_max', AClip.OutMax);
  end;

begin
  Result := False;
  SetLength(FClips, FConfig.NumLayers);
  for LI := 0 to FConfig.NumLayers - 1 do
  begin
    LPrefix := Format('%sencoder.layers.%d.', [CVisPrefix, LI]);
    if not (LoadOne(LPrefix + 'self_attn.q_proj', FClips[LI].QProj) and
      LoadOne(LPrefix + 'self_attn.k_proj', FClips[LI].KProj) and
      LoadOne(LPrefix + 'self_attn.v_proj', FClips[LI].VProj) and
      LoadOne(LPrefix + 'self_attn.o_proj', FClips[LI].OProj) and
      LoadOne(LPrefix + 'mlp.gate_proj', FClips[LI].GateProj) and
      LoadOne(LPrefix + 'mlp.up_proj', FClips[LI].UpProj) and
      LoadOne(LPrefix + 'mlp.down_proj', FClips[LI].DownProj)) then
    begin
      GetErrors().Add(esError, CVIS_ERR_OPEN,
        'Missing clip scalars, layer %d', [LI]);
      Exit;
    end;
  end;
  Result := True;
end;

procedure TVision.DoRecordNormRows(const ATensorName: string;
  const ABuf: TVulkanBuffer; const ARowSize: Integer;
  const ANumRows: Integer);
var
  LOfs: UInt64;
  LSize: UInt64;
begin
  if not DoTensorLoc(ATensorName, LOfs, LSize) then Exit;
  // Weighted per-row RMS norm against gamma at a byte offset in the window
  FKernels.RecordRmsNormRows(ABuf, FGpuWeights, LOfs, LSize,
    UInt32(ARowSize), UInt32(ANumRows), FConfig.RmsNormEps);
end;

procedure TVision.DoRecordClippedProj(const ATensorName: string;
  const AClip: TClipBounds; const AIn: TVulkanBuffer;
  const AOut: TVulkanBuffer; const ARows: Integer; const ACols: Integer;
  const ABatch: Integer; const AInElems: Integer);
var
  LOfs: UInt64;
  LSize: UInt64;
begin
  // QAT clipped linear: clamp(input) -> matmul -> clamp(output).
  // AIn is clamped IN-PLACE -- the caller must pass a consumable buffer
  // (copy shared inputs like FGpuNorm into FGpuTmp first).
  if not DoTensorLoc(ATensorName, LOfs, LSize) then Exit;

  FKernels.RecordClamp(AIn, AIn, UInt32(AInElems), AClip.InMin, AClip.InMax);
  FKernels.RecordBarrier();
  FKernels.RecordMatMatF32(FGpuWeights, LOfs, LSize, AIn, AOut,
    UInt32(ARows), UInt32(ACols), UInt32(ABatch));
  FKernels.RecordBarrier();
  FKernels.RecordClamp(AOut, AOut, UInt32(ABatch * ARows),
    AClip.OutMin, AClip.OutMax);
  FKernels.RecordBarrier();
end;

procedure TVision.DoRecordLayer(const ALayerIdx: Integer;
  const ANumPatches: Integer);
var
  LPrefix: string;
  LN: Integer;
  LH: Integer;
  LPush: TAttnBidirPushConstants;
begin
  LPrefix := Format('%sencoder.layers.%d.', [CVisPrefix, ALayerIdx]);
  LN := ANumPatches;
  LH := FConfig.HiddenSize;

  // ---- Attention block (sandwich norm) ----
  FKernels.RecordCopy(FGpuNorm, FGpuHidden, UInt32(LN * LH));
  FKernels.RecordBarrier();
  DoRecordNormRows(LPrefix + 'input_layernorm.weight', FGpuNorm, LH, LN);
  FKernels.RecordBarrier();

  // Q/K/V projections: FGpuNorm is shared, so each projection clamps a
  // fresh copy in FGpuTmp (input clamps differ per projection)
  FKernels.RecordCopy(FGpuTmp, FGpuNorm, UInt32(LN * LH));
  FKernels.RecordBarrier();
  DoRecordClippedProj(LPrefix + 'self_attn.q_proj.linear.weight',
    FClips[ALayerIdx].QProj, FGpuTmp, FGpuQ,
    FConfig.NumHeads * FConfig.HeadDim, LH, LN, LN * LH);

  FKernels.RecordCopy(FGpuTmp, FGpuNorm, UInt32(LN * LH));
  FKernels.RecordBarrier();
  DoRecordClippedProj(LPrefix + 'self_attn.k_proj.linear.weight',
    FClips[ALayerIdx].KProj, FGpuTmp, FGpuK,
    FConfig.NumKvHeads * FConfig.HeadDim, LH, LN, LN * LH);

  FKernels.RecordCopy(FGpuTmp, FGpuNorm, UInt32(LN * LH));
  FKernels.RecordBarrier();
  DoRecordClippedProj(LPrefix + 'self_attn.v_proj.linear.weight',
    FClips[ALayerIdx].VProj, FGpuTmp, FGpuV,
    FConfig.NumKvHeads * FConfig.HeadDim, LH, LN, LN * LH);

  // Q/K weighted per-head norms; V weight-free norm; then 2D RoPE on Q/K
  DoRecordNormRows(LPrefix + 'self_attn.q_norm.weight', FGpuQ,
    FConfig.HeadDim, LN * FConfig.NumHeads);
  DoRecordNormRows(LPrefix + 'self_attn.k_norm.weight', FGpuK,
    FConfig.HeadDim, LN * FConfig.NumKvHeads);
  FKernels.RecordRmsNormRows(FGpuV, FGpuWeights, UInt32(FConfig.HeadDim),
    UInt32(LN * FConfig.NumKvHeads), FConfig.RmsNormEps, False);
  FKernels.RecordBarrier();

  FKernels.RecordRoPE2D(FGpuQ, FGpuPos, UInt32(LN), UInt32(FConfig.HeadDim),
    UInt32(FConfig.NumHeads), FConfig.RopeTheta);
  FKernels.RecordRoPE2D(FGpuK, FGpuPos, UInt32(LN), UInt32(FConfig.HeadDim),
    UInt32(FConfig.NumKvHeads), FConfig.RopeTheta);
  FKernels.RecordBarrier();

  // Fully bidirectional attention, scale = 1.0 (Gemma 4 vision)
  LPush.headDim := UInt32(FConfig.HeadDim);
  LPush.numHeads := UInt32(FConfig.NumHeads);
  LPush.numKvHeads := UInt32(FConfig.NumKvHeads);
  LPush.kvStride := UInt32(FConfig.NumKvHeads * FConfig.HeadDim);
  LPush.seqLen := UInt32(LN);
  LPush.windowSize := 0;
  LPush.scale := 1.0;
  LPush.rowPitch := UInt32(LN);

  FKernels.RecordAttnScoresBidir(FGpuQ, FGpuK, FGpuScores, LPush);
  FKernels.RecordBarrier();
  FKernels.RecordSoftmaxRows(FGpuScores, UInt32(LN),
    UInt32(LN * FConfig.NumHeads));
  FKernels.RecordBarrier();
  // FGpuQ is free after scores: reuse it as the attention output buffer
  FKernels.RecordAttnOutBidir(FGpuScores, FGpuV, FGpuQ, LPush);
  FKernels.RecordBarrier();

  // o_proj consumes FGpuQ: clamp direct, output to FGpuTmp
  DoRecordClippedProj(LPrefix + 'self_attn.o_proj.linear.weight',
    FClips[ALayerIdx].OProj, FGpuQ, FGpuTmp,
    LH, FConfig.NumHeads * FConfig.HeadDim, LN, LN * LH);

  DoRecordNormRows(LPrefix + 'post_attention_layernorm.weight', FGpuTmp,
    LH, LN);
  FKernels.RecordBarrier();
  FKernels.RecordAdd(FGpuHidden, FGpuTmp, UInt32(LN * LH));
  FKernels.RecordBarrier();

  // ---- MLP block (sandwich norm) ----
  FKernels.RecordCopy(FGpuNorm, FGpuHidden, UInt32(LN * LH));
  FKernels.RecordBarrier();
  DoRecordNormRows(LPrefix + 'pre_feedforward_layernorm.weight', FGpuNorm,
    LH, LN);
  FKernels.RecordBarrier();

  FKernels.RecordCopy(FGpuTmp, FGpuNorm, UInt32(LN * LH));
  FKernels.RecordBarrier();
  DoRecordClippedProj(LPrefix + 'mlp.gate_proj.linear.weight',
    FClips[ALayerIdx].GateProj, FGpuTmp, FGpuMlpGate,
    FConfig.IntermediateSize, LH, LN, LN * LH);

  FKernels.RecordCopy(FGpuTmp, FGpuNorm, UInt32(LN * LH));
  FKernels.RecordBarrier();
  DoRecordClippedProj(LPrefix + 'mlp.up_proj.linear.weight',
    FClips[ALayerIdx].UpProj, FGpuTmp, FGpuMlpUp,
    FConfig.IntermediateSize, LH, LN, LN * LH);

  FKernels.RecordGeGLU(FGpuMlpGate, FGpuMlpUp,
    UInt32(LN * FConfig.IntermediateSize), 0);
  FKernels.RecordBarrier();

  // down_proj consumes FGpuMlpGate (3072-wide): clamp direct
  DoRecordClippedProj(LPrefix + 'mlp.down_proj.linear.weight',
    FClips[ALayerIdx].DownProj, FGpuMlpGate, FGpuTmp,
    LH, FConfig.IntermediateSize, LN, LN * FConfig.IntermediateSize);

  DoRecordNormRows(LPrefix + 'post_feedforward_layernorm.weight', FGpuTmp,
    LH, LN);
  FKernels.RecordBarrier();
  FKernels.RecordAdd(FGpuHidden, FGpuTmp, UInt32(LN * LH));
  FKernels.RecordBarrier();
end;

function TVision.DoPatchPrep(const APatches: TArray<Single>;
  const APositions: TArray<Integer>; const ANumPatches: Integer): Boolean;
var
  LStage: TArray<Single>;
  LPE: TArray<Single>;
  LInfo: TTensorInfo;
  LTable: PSingle;
  LRowX: PSingle;
  LRowY: PSingle;
  LP: Integer;
  LJ: Integer;
  LX: Integer;
  LY: Integer;
  LH: Integer;
begin
  Result := False;
  LH := FConfig.HiddenSize;

  // Gemma 4 applies no normalization; the encoder itself maps [0,1] -> [-1,1]
  SetLength(LStage, ANumPatches * LH);
  for LJ := 0 to ANumPatches * LH - 1 do
    LStage[LJ] := 2.0 * (APatches[LJ] - 0.5);

  // Position embeddings from the learned 2D table [2, 10240, 768]:
  // row = table[0][x] + table[1][y] (added AFTER input_proj on the GPU).
  // The table is read straight off the memory-mapped blob.
  if not FWeightStore.FindTensor(
    CVisPrefix + 'patch_embedder.position_embedding_table', LInfo) then
  begin
    GetErrors().Add(esError, CVIS_ERR_ENCODE, 'Position table missing');
    Exit;
  end;
  LTable := PSingle(FWeightStore.GetRawPointer(LInfo));

  SetLength(LPE, ANumPatches * LH);
  for LP := 0 to ANumPatches - 1 do
  begin
    LX := APositions[LP * 2];
    LY := APositions[LP * 2 + 1];
    if (LX < 0) or (LX >= FConfig.PosTableSize) or
      (LY < 0) or (LY >= FConfig.PosTableSize) then
    begin
      GetErrors().Add(esError, CVIS_ERR_ENCODE,
        'Patch %d position (%d,%d) out of table range', [LP, LX, LY]);
      Exit;
    end;
    LRowX := PSingle(UIntPtr(LTable) +
      UIntPtr(UInt64(LX) * UInt64(LH) * SizeOf(Single)));
    LRowY := PSingle(UIntPtr(LTable) + UIntPtr(
      (UInt64(FConfig.PosTableSize) + UInt64(LY)) * UInt64(LH) * SizeOf(Single)));
    for LJ := 0 to LH - 1 do
      LPE[LP * LH + LJ] :=
        PSingle(UIntPtr(LRowX) + UIntPtr(LJ * SizeOf(Single)))^ +
        PSingle(UIntPtr(LRowY) + UIntPtr(LJ * SizeOf(Single)))^;
  end;

  FDevice.UploadToBuffer(FGpuStageIn, @LStage[0],
    UInt64(ANumPatches) * UInt64(LH) * SizeOf(Single));
  FDevice.UploadToBuffer(FGpuStagePE, @LPE[0],
    UInt64(ANumPatches) * UInt64(LH) * SizeOf(Single));
  FDevice.UploadToBuffer(FGpuPos, @APositions[0],
    UInt64(ANumPatches) * 2 * SizeOf(Int32));
  Result := True;
end;

function TVision.DoEncoderForward(const ANumPatches: Integer;
  out AHidden: TArray<Single>): Boolean;
var
  LOfs: UInt64;
  LSize: UInt64;
  LI: Integer;
  LH: Integer;
begin
  Result := False;
  LH := FConfig.HiddenSize;

  if not DoTensorLoc(CVisPrefix + 'patch_embedder.input_proj.weight',
    LOfs, LSize) then
  begin
    GetErrors().Add(esError, CVIS_ERR_ENCODE, 'input_proj missing');
    Exit;
  end;

  FKernels.Shaders.ResetDescriptorPool();
  FDevice.BeginCommands();

  // Patch embedding: input_proj (NOT clipped), then add position embeddings
  FKernels.RecordMatMatF32(FGpuWeights, LOfs, LSize, FGpuStageIn, FGpuHidden,
    UInt32(LH), UInt32(LH), UInt32(ANumPatches));
  FKernels.RecordBarrier();
  FKernels.RecordAdd(FGpuHidden, FGpuStagePE, UInt32(ANumPatches * LH));
  FKernels.RecordBarrier();

  // 16 bidirectional encoder layers; NO final norm after the last layer
  for LI := 0 to FConfig.NumLayers - 1 do
    DoRecordLayer(LI, ANumPatches);

  FKernels.RecordCopy(FGpuStageOut, FGpuHidden, UInt32(ANumPatches * LH));

  FDevice.EndCommands();
  FDevice.SubmitAndWait();

  SetLength(AHidden, ANumPatches * LH);
  FDevice.DownloadFromBuffer(FGpuStageOut, @AHidden[0],
    UInt64(ANumPatches) * UInt64(LH) * SizeOf(Single));
  Result := True;
end;

function TVision.DoPoolCpu(const AHidden: TArray<Single>;
  const APositions: TArray<Integer>; const ANumPatches: Integer;
  out APooled: TArray<Single>; out AOutLen: Integer): Boolean;
var
  LK: Integer;
  LKSq: Integer;
  LMaxX: Integer;
  LKernelsPerRow: Integer;
  LCounts: TArray<Integer>;
  LP: Integer;
  LJ: Integer;
  LIdx: Integer;
  LH: Integer;
  LScale: Single;
begin
  Result := False;
  LH := FConfig.HiddenSize;
  LK := FConfig.PoolK;
  LKSq := LK * LK;
  AOutLen := ANumPatches div LKSq;

  // Grid width from the positions (max x + 1); pooling groups patches by
  // (x div k, y div k) exactly like the HF pooler's kernel_idxs
  LMaxX := 0;
  for LP := 0 to ANumPatches - 1 do
    if APositions[LP * 2] > LMaxX then
      LMaxX := APositions[LP * 2];
  Inc(LMaxX);
  LKernelsPerRow := LMaxX div LK;

  SetLength(APooled, AOutLen * LH);
  FillChar(APooled[0], AOutLen * LH * SizeOf(Single), 0);
  SetLength(LCounts, AOutLen);
  FillChar(LCounts[0], AOutLen * SizeOf(Integer), 0);

  for LP := 0 to ANumPatches - 1 do
  begin
    LIdx := (APositions[LP * 2] div LK) +
      LKernelsPerRow * (APositions[LP * 2 + 1] div LK);
    if (LIdx < 0) or (LIdx >= AOutLen) then
    begin
      GetErrors().Add(esError, CVIS_ERR_ENCODE,
        'Pool kernel index %d out of range 0..%d (grid mismatch)',
        [LIdx, AOutLen - 1]);
      Exit;
    end;
    Inc(LCounts[LIdx]);
    for LJ := 0 to LH - 1 do
      APooled[LIdx * LH + LJ] := APooled[LIdx * LH + LJ] +
        AHidden[LP * LH + LJ];
  end;

  // Every kernel must average exactly k^2 patches (no padding in our path)
  for LIdx := 0 to AOutLen - 1 do
    if LCounts[LIdx] <> LKSq then
    begin
      GetErrors().Add(esError, CVIS_ERR_ENCODE,
        'Pool kernel %d has %d patches, expected %d',
        [LIdx, LCounts[LIdx], LKSq]);
      Exit;
    end;

  // Average, then the sqrt(hidden_size) pooler scaling
  LScale := Sqrt(Single(LH)) / LKSq;
  for LJ := 0 to AOutLen * LH - 1 do
    APooled[LJ] := APooled[LJ] * LScale;

  Result := True;
end;

function TVision.DoProjectGpu(const APooled: TArray<Single>;
  const AOutLen: Integer; out ASoftTokens: TArray<Single>): Boolean;
var
  LOfs: UInt64;
  LSize: UInt64;
  LH: Integer;
  LT: Integer;
begin
  Result := False;
  LH := FConfig.HiddenSize;
  LT := FConfig.TextHiddenSize;

  if not DoTensorLoc(CVisEmbedProjName, LOfs, LSize) then
  begin
    GetErrors().Add(esError, CVIS_ERR_ENCODE, 'embedding_projection missing');
    Exit;
  end;

  FDevice.UploadToBuffer(FGpuStageIn, @APooled[0],
    UInt64(AOutLen) * UInt64(LH) * SizeOf(Single));

  FKernels.Shaders.ResetDescriptorPool();
  FDevice.BeginCommands();

  // MultimodalEmbedder: weight-free RMS norm, then [2560 x 768] projection
  FKernels.RecordCopy(FGpuTmp, FGpuStageIn, UInt32(AOutLen * LH));
  FKernels.RecordBarrier();
  FKernels.RecordRmsNormRows(FGpuTmp, FGpuWeights, UInt32(LH),
    UInt32(AOutLen), FConfig.RmsNormEps, False);
  FKernels.RecordBarrier();
  FKernels.RecordMatMatF32(FGpuWeights, LOfs, LSize, FGpuTmp, FGpuProj,
    UInt32(LT), UInt32(LH), UInt32(AOutLen));
  FKernels.RecordBarrier();
  FKernels.RecordCopy(FGpuStageOut, FGpuProj, UInt32(AOutLen * LT));

  FDevice.EndCommands();
  FDevice.SubmitAndWait();

  SetLength(ASoftTokens, AOutLen * LT);
  FDevice.DownloadFromBuffer(FGpuStageOut, @ASoftTokens[0],
    UInt64(AOutLen) * UInt64(LT) * SizeOf(Single));
  Result := True;
end;

function TVision.Open(const AVpkPath: string;
  const AKernels: TComputeKernels): Boolean;
var
  LConfigStr: string;
begin
  Result := False;

  if FIsLoaded then
    Close();

  FOwnsGpu := AKernels = nil;
  if not FOwnsGpu then
    FKernels := AKernels;

  LConfigStr := DoReadVpkFileAsString(AVpkPath, CVisConfigPath);
  if LConfigStr = '' then
  begin
    GetErrors().Add(esError, CVIS_ERR_OPEN, 'Missing %s', [CVisConfigPath]);
    Exit;
  end;
  if not FConfig.LoadFromJson(LConfigStr) then
  begin
    GetErrors().Add(esError, CVIS_ERR_OPEN, 'Bad vision config');
    Exit;
  end;

  if not FWeightStore.Open(AVpkPath, CVisManifestPath, CVisWeightsPath) then
  begin
    GetErrors().Add(esError, CVIS_ERR_OPEN, 'Encoder weight store open failed');
    Exit;
  end;

  if not DoLoadClips() then Exit;
  if not DoInitGpu() then Exit;
  if not DoUploadWeights() then Exit;
  if not DoAllocateBuffers() then Exit;

  FIsLoaded := True;
  Status('Vision encoder ready: %d layers, hidden %d, max %d patches',
    [FConfig.NumLayers, FConfig.HiddenSize, CVisMaxPatches]);
  Result := True;
end;

procedure TVision.Close();
begin
  DoFreeBuffers();
  DoShutdownGpu();
  FWeightStore.Close();
  SetLength(FClips, 0);
  FIsLoaded := False;
end;

function TVision.IsLoaded(): Boolean;
begin
  Result := FIsLoaded;
end;

function TVision.EncodeImage(const APatches: TArray<Single>;
  const APositions: TArray<Integer>;
  out ASoftTokens: TArray<Single>): Boolean;
var
  LN: Integer;
  LHidden: TArray<Single>;
  LPooled: TArray<Single>;
  LOutLen: Integer;
begin
  Result := False;
  SetLength(ASoftTokens, 0);
  if not FIsLoaded then
  begin
    GetErrors().Add(esError, CVIS_ERR_ENCODE, 'Vision encoder not loaded');
    Exit;
  end;

  LN := Length(APositions) div 2;
  if (LN <= 0) or (LN > CVisMaxPatches) or
    (LN mod (FConfig.PoolK * FConfig.PoolK) <> 0) or
    (Length(APatches) <> LN * FConfig.HiddenSize) or
    (Length(APositions) <> LN * 2) then
  begin
    GetErrors().Add(esError, CVIS_ERR_ENCODE,
      'Bad patch input: %d patches, %d pixels, %d positions',
      [LN, Length(APatches), Length(APositions)]);
    Exit;
  end;

  if not DoPatchPrep(APatches, APositions, LN) then Exit;
  if not DoEncoderForward(LN, LHidden) then Exit;
  if not DoPoolCpu(LHidden, APositions, LN, LPooled, LOutLen) then Exit;
  if not DoProjectGpu(LPooled, LOutLen, ASoftTokens) then Exit;

  Status('Image encoded: %d patches -> %d soft tokens', [LN, LOutLen]);
  Result := True;
end;

end.
