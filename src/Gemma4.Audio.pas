{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information

 -------------------------------------------------------------------------------

  Gemma4.Audio - Gemma 4 audio encoder (waveform -> soft tokens)

  Full port of the HF Gemma 4 audio stack (Universal Speech Model):

  CPU frontend (Gemma4AudioFeatureExtractor, all-default config):
    16 kHz mono F32 -> semicausal left pad (160) -> 321-sample frames at
    hop 160 (last sample dropped, preemphasis = 0) -> periodic Hann(320)
    -> rFFT 512 -> magnitude (257 bins) -> HTK mel filterbank (128 mels,
    0..8 kHz, norm=None) -> log(mel + 1e-3)  => [T x 128] log-mel frames.

  CPU subsampler (Gemma4AudioSubSampleConvProjection):
    two Conv2d(3x3, stride 2, pad 1, no bias) + channel LayerNorm
    (scale-only) + ReLU: [T x 128] -> [T/4 x (32 freq x 32 ch) = 1024].

  GPU conformer (Gemma4AudioModel, 12 layers, hidden 1024, on
  E4B/encoders.bin): input projection, then per layer FF(x0.5 residual,
  SiLU) -> chunked local attention (chunk 12, ctx 24, Transformer-XL
  rel-pos bias, tanh softcap 50, QAT clip clamps on every linear) ->
  lconv (GLU + causal depthwise conv k5) -> FF -> RMS norm out; then
  output_proj (1024 -> 1536, WITH bias) and the multimodal embedder
  (weight-free norm + [2560 x 1536]) into language-model space.

  EncodeAudio: samples -> [T/4 x 2560] soft rows (40 ms per token,
  audio soft-token id 258881, max 30 s = 750 tokens). The engine never
  feeds padded frames, so every produced token is valid.

  Share TInference's TComputeKernels via Open() exactly like TVision;
  free/close TAudio BEFORE the owning kernels/device.

  Dependencies: StdApp.Base, StdApp.JSON, StdApp.VFS, StdApp.VirtualMemory,
    Gemma4.Types, Gemma4.Tensors, Gemma4.Vulkan, Gemma4.Compute
===============================================================================}

unit Gemma4.Audio;

{$I StdApp.Defines.inc}

interface

uses
  System.SysUtils,
  System.Classes,
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
  CAUD_ERR_OPEN = 'AU01';
  CAUD_ERR_ENCODE = 'AU02';

  CAudManifestPath = 'E4B/encoders_manifest.json';
  CAudWeightsPath = 'E4B/encoders.bin';
  CAudConfigPath = 'E4B/config.json';
  CAudPrefix = 'model.audio_tower.';
  CAudEmbedPrefix = 'model.embed_audio.';
  CAudEmbedProjName = 'model.embed_audio.embedding_projection.weight';
  CAudSampleRate = 16000;
  CAudFrameLength = 320;         // 20 ms
  CAudHopLength = 160;           // 10 ms
  CAudFftLength = 512;
  CAudFreqBins = CAudFftLength div 2 + 1;   // 257
  CAudMelBins = 128;
  CAudMelFloor = 1e-3;
  CAudMaxSamples = 480000;       // 30 s -> 750 soft tokens
  CAudMaxTokens = 750;

type
  { TAudioConfig }
  // Parsed audio_config from E4B/config.json plus the text hidden size
  TAudioConfig = record
    HiddenSize: Integer;          // 1024
    NumLayers: Integer;           // 12
    NumHeads: Integer;            // 8
    HeadDim: Integer;             // 128 (hidden / heads)
    ConvKernel: Integer;          // 5
    ChunkSize: Integer;           // 12
    CtxLeft: Integer;             // 13 (past horizon = 13 - 1)
    CtxRight: Integer;            // 0
    CtxSize: Integer;             // chunk + (left-1) + right = 24
    LogitCap: Single;             // 50.0
    InvalidLogit: Single;         // -1e9
    ResidualWeight: Single;       // 0.5
    GradClip: Single;             // 1e10
    RmsNormEps: Single;           // 1e-6
    SubCh0: Integer;              // 128
    SubCh1: Integer;              // 32
    OutputProjDims: Integer;      // 1536
    TextHiddenSize: Integer;      // 2560
    function LoadFromJson(const AJsonStr: string): Boolean;
  end;

  { TAudClipBounds }
  // QAT clip scalars of one clipped linear (read once from the blob)
  TAudClipBounds = record
    InMin: Single;
    InMax: Single;
    OutMin: Single;
    OutMax: Single;
  end;

  { TAudioLayerClips }
  // Clip bounds for the ten clipped linears of one conformer layer
  TAudioLayerClips = record
    Ff1A: TAudClipBounds;      // feed_forward1.ffw_layer_1
    Ff1B: TAudClipBounds;      // feed_forward1.ffw_layer_2
    Ff2A: TAudClipBounds;      // feed_forward2.ffw_layer_1
    Ff2B: TAudClipBounds;      // feed_forward2.ffw_layer_2
    AttnQ: TAudClipBounds;
    AttnK: TAudClipBounds;
    AttnV: TAudClipBounds;
    AttnPost: TAudClipBounds;
    LcStart: TAudClipBounds;   // lconv1d.linear_start
    LcEnd: TAudClipBounds;     // lconv1d.linear_end
  end;

  { TAudio }
  // Audio encoder engine. Open() once, EncodeAudio() per clip, Close().
  // GPU-only for the conformer; mel + subsampler run on the CPU.
  TAudio = class(TBaseObject)
  private
    FWeightStore: TWeightStore;
    FConfig: TAudioConfig;
    FDevice: TVulkanDevice;        // owned only when FOwnsGpu
    FKernels: TComputeKernels;     // owned only when FOwnsGpu
    FOwnsGpu: Boolean;
    FIsLoaded: Boolean;
    FClips: TArray<TAudioLayerClips>;
    FWinBase: UInt64;              // audio byte window base in the blob

    // GPU state
    FGpuWeights: TVulkanBuffer;    // audio window of encoders.bin, F32
    FGpuHidden: TVulkanBuffer;     // [N x 1024] residual stream
    FGpuNorm: TVulkanBuffer;       // [N x 1024] normed block input
    FGpuTmp: TVulkanBuffer;        // [N x 1024] scratch
    FGpuTmp2: TVulkanBuffer;       // [N x 1024] scratch (dwconv out)
    FGpuQ: TVulkanBuffer;          // [N x 1024]
    FGpuK: TVulkanBuffer;          // [N x 1024]
    FGpuV: TVulkanBuffer;          // [N x 1024]
    FGpuAttnO: TVulkanBuffer;      // [N x 1024] attention output
    FGpuMlp: TVulkanBuffer;        // [N x 4096] FF intermediate
    FGpuLc: TVulkanBuffer;         // [N x 2048] lconv GLU input
    FGpuRelK: TVulkanBuffer;       // [12 x 13 x 1024] per-layer rel keys
    FGpuQScale: TVulkanBuffer;     // [12 x 128] per-layer q scale vectors
    FGpuOut1536: TVulkanBuffer;    // [N x 1536] output_proj result
    FGpuProj: TVulkanBuffer;       // [N x 2560] soft-token projection
    FGpuStageIn: TVulkanBuffer;    // host-visible upload [N x 1536] max
    FGpuStageOut: TVulkanBuffer;   // host-visible download [N x 2560]

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
    function DoTensorPtr(const AName: string): PSingle;
    function DoPrepareRelPos(): Boolean;
    function DoPrepareQScales(): Boolean;
    function DoSubsampleCpu(const AFeatures: TArray<Single>;
      const ANumFrames: Integer; out AHidden: TArray<Single>;
      out ANumTokens: Integer): Boolean;
    procedure DoRecordNormRows(const ATensorName: string;
      const ABuf: TVulkanBuffer; const ARowSize: Integer;
      const ANumRows: Integer);
    procedure DoRecordClippedProj(const ATensorName: string;
      const AClip: TAudClipBounds; const AIn: TVulkanBuffer;
      const AOut: TVulkanBuffer; const ARows: Integer; const ACols: Integer;
      const ABatch: Integer; const AInElems: Integer);
    procedure DoRecordFeedForward(const APrefix: string;
      const AClipA: TAudClipBounds; const AClipB: TAudClipBounds;
      const AN: Integer);
    procedure DoRecordLayer(const ALayerIdx: Integer; const AN: Integer);
    function DoConformerForward(const AHidden: TArray<Single>;
      const AN: Integer; out ASoftTokens: TArray<Single>): Boolean;
  public
    constructor Create(); override;
    destructor Destroy(); override;

    procedure SetStatusCallback(const ACallback: TStatusCallback;
      const AUserData: Pointer = nil); override;

    // Open the VPK's audio weights. AKernels = a live shared TComputeKernels
    // (TInference's) to reuse its device; nil = create an own device.
    function Open(const AVpkPath: string;
      const AKernels: TComputeKernels = nil): Boolean;
    procedure Close();
    function IsLoaded(): Boolean;

    // 16 kHz mono F32 samples -> log-mel frames [ANumFrames x 128].
    // Pure CPU, static: usable without Open() (parity tests).
    class function ComputeMel(const ASamples: TArray<Single>;
      out AFeatures: TArray<Single>; out ANumFrames: Integer): Boolean;

    // Load a RIFF WAV file (PCM16 or IEEE F32, any channel count, any
    // rate) to 16 kHz mono F32 samples: channels averaged, rate linearly
    // resampled. Pure CPU, static. False on missing/unsupported file.
    class function LoadWaveFile(const AFileName: string;
      out ASamples: TArray<Single>): Boolean;

    // Log-mel frames -> soft tokens [N x 2560] (subsample + conformer).
    function EncodeFeatures(const AFeatures: TArray<Single>;
      const ANumFrames: Integer; out ASoftTokens: TArray<Single>;
      out ANumTokens: Integer): Boolean;

    // 16 kHz mono F32 samples -> soft tokens (ComputeMel + EncodeFeatures)
    function EncodeAudio(const ASamples: TArray<Single>;
      out ASoftTokens: TArray<Single>; out ANumTokens: Integer): Boolean;

    property Config: TAudioConfig read FConfig;
  end;

implementation

{ TAudioConfig }

function TAudioConfig.LoadFromJson(const AJsonStr: string): Boolean;
var
  LJson: TJSON;
  LAud: TJSON;
  LSubArr: TArray<TJSON>;
begin
  Result := False;
  LJson := TJSON.FromString(AJsonStr);
  try
    if LJson.IsNull() then
      Exit;

    LAud := LJson.Get('audio_config');
    if LAud.IsNull() then
      Exit;

    HiddenSize := LAud.Get('hidden_size').AsInt32(0);
    NumLayers := LAud.Get('num_hidden_layers').AsInt32(0);
    NumHeads := LAud.Get('num_attention_heads').AsInt32(0);
    if NumHeads > 0 then
      HeadDim := HiddenSize div NumHeads
    else
      HeadDim := 0;
    ConvKernel := LAud.Get('conv_kernel_size').AsInt32(5);
    ChunkSize := LAud.Get('attention_chunk_size').AsInt32(12);
    CtxLeft := LAud.Get('attention_context_left').AsInt32(13);
    CtxRight := LAud.Get('attention_context_right').AsInt32(0);
    CtxSize := ChunkSize + (CtxLeft - 1) + CtxRight;
    LogitCap := LAud.Get('attention_logit_cap').AsSingle(50.0);
    InvalidLogit := LAud.Get('attention_invalid_logits_value').AsSingle(-1e9);
    ResidualWeight := LAud.Get('residual_weight').AsSingle(0.5);
    GradClip := LAud.Get('gradient_clipping').AsSingle(1e10);
    RmsNormEps := LAud.Get('rms_norm_eps').AsSingle(1e-6);
    LSubArr := LAud.Get('subsampling_conv_channels').Items();
    if Length(LSubArr) >= 2 then
    begin
      SubCh0 := LSubArr[0].AsInt32(128);
      SubCh1 := LSubArr[1].AsInt32(32);
    end
    else
    begin
      SubCh0 := 128;
      SubCh1 := 32;
    end;
    OutputProjDims := LAud.Get('output_proj_dims').AsInt32(0);
    TextHiddenSize := LJson.Get('text_config').Get('hidden_size').AsInt32(0);

    Result := (HiddenSize = 1024) and (NumLayers > 0) and (HeadDim > 0) and
      (OutputProjDims > 0) and (TextHiddenSize > 0);
  finally
    LJson.Free();
  end;
end;

{ AudFft512 }
// Iterative radix-2 complex FFT, N = 512, forward (numpy convention:
// X[k] = sum x[n] * exp(-2*pi*i*k*n/N)). Input: ARe/AIm length 512.
procedure AudFft512(var ARe: TArray<Double>; var AIm: TArray<Double>);
const
  { CN }
  CN = 512;
var
  LI: Integer;
  LJ: Integer;
  LK: Integer;
  LBit: Integer;
  LLen: Integer;
  LHalf: Integer;
  LAng: Double;
  LWRe: Double;
  LWIm: Double;
  LCurRe: Double;
  LCurIm: Double;
  LTmpRe: Double;
  LTmpIm: Double;
  LURe: Double;
  LUIm: Double;
begin
  // Bit-reversal permutation
  LJ := 0;
  for LI := 1 to CN - 1 do
  begin
    LBit := CN shr 1;
    while (LJ and LBit) <> 0 do
    begin
      LJ := LJ xor LBit;
      LBit := LBit shr 1;
    end;
    LJ := LJ or LBit;
    if LI < LJ then
    begin
      LTmpRe := ARe[LI]; ARe[LI] := ARe[LJ]; ARe[LJ] := LTmpRe;
      LTmpIm := AIm[LI]; AIm[LI] := AIm[LJ]; AIm[LJ] := LTmpIm;
    end;
  end;

  // Butterflies
  LLen := 2;
  while LLen <= CN do
  begin
    LHalf := LLen shr 1;
    LAng := -2.0 * Pi / LLen;
    LWRe := Cos(LAng);
    LWIm := Sin(LAng);
    LI := 0;
    while LI < CN do
    begin
      LURe := 1.0;
      LUIm := 0.0;
      for LK := 0 to LHalf - 1 do
      begin
        LCurRe := ARe[LI + LK + LHalf] * LURe - AIm[LI + LK + LHalf] * LUIm;
        LCurIm := ARe[LI + LK + LHalf] * LUIm + AIm[LI + LK + LHalf] * LURe;
        ARe[LI + LK + LHalf] := ARe[LI + LK] - LCurRe;
        AIm[LI + LK + LHalf] := AIm[LI + LK] - LCurIm;
        ARe[LI + LK] := ARe[LI + LK] + LCurRe;
        AIm[LI + LK] := AIm[LI + LK] + LCurIm;
        LTmpRe := LURe * LWRe - LUIm * LWIm;
        LUIm := LURe * LWIm + LUIm * LWRe;
        LURe := LTmpRe;
      end;
      LI := LI + LLen;
    end;
    LLen := LLen shl 1;
  end;
end;

{ AudBuildMelFilters }
// HF mel_filter_bank port: 128 triangular HTK mel filters over 257 rFFT
// bins, 0..8000 Hz, norm=None. Output layout [bin * 128 + mel].
procedure AudBuildMelFilters(var AFilters: TArray<Double>);
var
  LMelMax: Double;
  LHz: array[0..CAudMelBins + 1] of Double;
  LFreq: Double;
  LI: Integer;
  LJ: Integer;
  LDown: Double;
  LUp: Double;
  LV: Double;
begin
  SetLength(AFilters, CAudFreqBins * CAudMelBins);

  // HTK mel scale: mel = 2595 * log10(1 + f/700); 130 equally spaced mel
  // points between mel(0)=0 and mel(8000), converted back to Hz
  LMelMax := 2595.0 * Log10(1.0 + 8000.0 / 700.0);
  for LJ := 0 to CAudMelBins + 1 do
    LHz[LJ] := 700.0 * (Power(10.0, (LMelMax * LJ / (CAudMelBins + 1)) / 2595.0) - 1.0);

  for LI := 0 to CAudFreqBins - 1 do
  begin
    // fft_freqs = linspace(0, 8000, 257)
    LFreq := LI * (8000.0 / (CAudFreqBins - 1));
    for LJ := 0 to CAudMelBins - 1 do
    begin
      LDown := (LFreq - LHz[LJ]) / (LHz[LJ + 1] - LHz[LJ]);
      LUp := (LHz[LJ + 2] - LFreq) / (LHz[LJ + 2] - LHz[LJ + 1]);
      LV := Min(LDown, LUp);
      if LV < 0.0 then
        LV := 0.0;
      AFilters[LI * CAudMelBins + LJ] := LV;
    end;
  end;
end;

{ TAudio }

class function TAudio.LoadWaveFile(const AFileName: string;
  out ASamples: TArray<Single>): Boolean;
const
  // Encoder input rate (feature extractor's sampling_rate)
  CTargetRate = 16000;
var
  LStream: TFileStream;
  LChunkId: array[0..3] of AnsiChar;
  LChunkSize: Cardinal;
  LRiffType: array[0..3] of AnsiChar;
  LFormatTag: Word;
  LChannels: Word;
  LSampleRate: Cardinal;
  LBitsPerSample: Word;
  LHaveFmt: Boolean;
  LDataBytes: TArray<Byte>;
  LFrameCount: Integer;
  LMono: TArray<Single>;
  LI: Integer;
  LCh: Integer;
  LSum: Double;
  LOutCount: Integer;
  LSrcPos: Double;
  LSrcIdx: Integer;
  LFrac: Double;
  LNext: Single;
begin
  Result := False;
  ASamples := nil;
  LHaveFmt := False;
  LFormatTag := 0;
  LChannels := 0;
  LSampleRate := 0;
  LBitsPerSample := 0;
  LDataBytes := nil;

  if not FileExists(AFileName) then
    Exit;

  try
    LStream := TFileStream.Create(AFileName, fmOpenRead or fmShareDenyWrite);
    try
      // RIFF header: 'RIFF' <size> 'WAVE'
      if LStream.Read(LChunkId, 4) <> 4 then
        Exit;
      if LChunkId <> 'RIFF' then
        Exit;
      LStream.ReadBuffer(LChunkSize, 4);
      if LStream.Read(LRiffType, 4) <> 4 then
        Exit;
      if LRiffType <> 'WAVE' then
        Exit;

      // Walk chunks for 'fmt ' and 'data' (word-aligned sizes)
      while LStream.Position + 8 <= LStream.Size do
      begin
        LStream.ReadBuffer(LChunkId, 4);
        LStream.ReadBuffer(LChunkSize, 4);
        if LChunkId = 'fmt ' then
        begin
          LStream.ReadBuffer(LFormatTag, 2);
          LStream.ReadBuffer(LChannels, 2);
          LStream.ReadBuffer(LSampleRate, 4);
          LStream.Seek(6, soCurrent);   // byte rate (4) + block align (2)
          LStream.ReadBuffer(LBitsPerSample, 2);
          // Skip any fmt extension bytes
          if LChunkSize > 16 then
            LStream.Seek(LChunkSize - 16, soCurrent);
          LHaveFmt := True;
        end
        else if LChunkId = 'data' then
        begin
          SetLength(LDataBytes, LChunkSize);
          if LChunkSize > 0 then
            LStream.ReadBuffer(LDataBytes[0], LChunkSize);
        end
        else
          LStream.Seek(LChunkSize, soCurrent);
        // Chunks are word-aligned: skip the pad byte after odd sizes
        if (LChunkSize and 1) = 1 then
          LStream.Seek(1, soCurrent);
      end;
    finally
      LStream.Free();
    end;
  except
    Exit;
  end;

  if (not LHaveFmt) or (LDataBytes = nil) or (LChannels = 0) or
     (LSampleRate = 0) then
    Exit;

  // Decode + downmix to mono. Supported: PCM16 (tag 1, 16-bit) and
  // IEEE float (tag 3, 32-bit).
  if (LFormatTag = 1) and (LBitsPerSample = 16) then
  begin
    LFrameCount := Length(LDataBytes) div (2 * LChannels);
    SetLength(LMono, LFrameCount);
    for LI := 0 to LFrameCount - 1 do
    begin
      LSum := 0.0;
      for LCh := 0 to LChannels - 1 do
        LSum := LSum +
          PSmallInt(@LDataBytes[(LI * LChannels + LCh) * 2])^;
      LMono[LI] := (LSum / LChannels) / 32768.0;
    end;
  end
  else if (LFormatTag = 3) and (LBitsPerSample = 32) then
  begin
    LFrameCount := Length(LDataBytes) div (4 * LChannels);
    SetLength(LMono, LFrameCount);
    for LI := 0 to LFrameCount - 1 do
    begin
      LSum := 0.0;
      for LCh := 0 to LChannels - 1 do
        LSum := LSum +
          PSingle(@LDataBytes[(LI * LChannels + LCh) * 4])^;
      LMono[LI] := LSum / LChannels;
    end;
  end
  else
    Exit;   // unsupported encoding (compressed, 8/24-bit, ...)

  if LFrameCount = 0 then
    Exit;

  // Resample to the encoder rate by linear interpolation
  if LSampleRate = CTargetRate then
    ASamples := LMono
  else
  begin
    LOutCount := Round(Int64(LFrameCount) * CTargetRate / LSampleRate);
    if LOutCount < 1 then
      Exit;
    SetLength(ASamples, LOutCount);
    for LI := 0 to LOutCount - 1 do
    begin
      LSrcPos := LI * (LSampleRate / CTargetRate);
      LSrcIdx := Trunc(LSrcPos);
      LFrac := LSrcPos - LSrcIdx;
      if LSrcIdx >= LFrameCount - 1 then
      begin
        LSrcIdx := LFrameCount - 1;
        LNext := LMono[LSrcIdx];
        LFrac := 0.0;
      end
      else
        LNext := LMono[LSrcIdx + 1];
      ASamples[LI] := LMono[LSrcIdx] + LFrac * (LNext - LMono[LSrcIdx]);
    end;
  end;

  Result := True;
end;

class function TAudio.ComputeMel(const ASamples: TArray<Single>;
  out AFeatures: TArray<Single>; out ANumFrames: Integer): Boolean;
var
  LWindow: TArray<Double>;
  LFilters: TArray<Double>;
  LRe: TArray<Double>;
  LIm: TArray<Double>;
  LMag: array[0..CAudFreqBins - 1] of Double;
  LNumSamples: Integer;
  LPadded: Integer;
  LF: Integer;
  LS: Integer;
  LJ: Integer;
  LIdx: Integer;
  LAcc: Double;
begin
  Result := False;
  AFeatures := nil;
  ANumFrames := 0;

  LNumSamples := Length(ASamples);
  if (LNumSamples <= 0) or (LNumSamples > CAudMaxSamples) then
    Exit;

  // Semicausal left pad of frame_length/2 zeros, frames of 321 at hop 160;
  // the 321st sample is dropped (preemphasis = 0 in the Gemma 4 config)
  LPadded := CAudFrameLength div 2 + LNumSamples;
  ANumFrames := (LPadded - (CAudFrameLength + 1)) div CAudHopLength + 1;
  if ANumFrames <= 0 then
  begin
    ANumFrames := 0;
    Exit;
  end;

  // Periodic Hann: w[n] = 0.5 - 0.5*cos(2*pi*n / frame_length)
  SetLength(LWindow, CAudFrameLength);
  for LS := 0 to CAudFrameLength - 1 do
    LWindow[LS] := 0.5 - 0.5 * Cos(2.0 * Pi * LS / CAudFrameLength);

  AudBuildMelFilters(LFilters);

  SetLength(AFeatures, ANumFrames * CAudMelBins);
  SetLength(LRe, CAudFftLength);
  SetLength(LIm, CAudFftLength);

  for LF := 0 to ANumFrames - 1 do
  begin
    // Windowed frame (implicit zeros in the left-pad region), then
    // zero-pad 320 -> 512 for the rFFT
    for LS := 0 to CAudFrameLength - 1 do
    begin
      LIdx := LF * CAudHopLength + LS - (CAudFrameLength div 2);
      if (LIdx >= 0) and (LIdx < LNumSamples) then
        LRe[LS] := ASamples[LIdx] * LWindow[LS]
      else
        LRe[LS] := 0.0;
      LIm[LS] := 0.0;
    end;
    for LS := CAudFrameLength to CAudFftLength - 1 do
    begin
      LRe[LS] := 0.0;
      LIm[LS] := 0.0;
    end;

    AudFft512(LRe, LIm);
    for LS := 0 to CAudFreqBins - 1 do
      LMag[LS] := Sqrt(LRe[LS] * LRe[LS] + LIm[LS] * LIm[LS]);

    // Mel projection + log floor: log(mag @ filters + 1e-3)
    for LJ := 0 to CAudMelBins - 1 do
    begin
      LAcc := 0.0;
      for LS := 0 to CAudFreqBins - 1 do
        LAcc := LAcc + LMag[LS] * LFilters[LS * CAudMelBins + LJ];
      AFeatures[LF * CAudMelBins + LJ] := Ln(LAcc + CAudMelFloor);
    end;
  end;

  Result := True;
end;

constructor TAudio.Create();
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
  FGpuTmp2 := Default(TVulkanBuffer);
  FGpuQ := Default(TVulkanBuffer);
  FGpuK := Default(TVulkanBuffer);
  FGpuV := Default(TVulkanBuffer);
  FGpuAttnO := Default(TVulkanBuffer);
  FGpuMlp := Default(TVulkanBuffer);
  FGpuLc := Default(TVulkanBuffer);
  FGpuRelK := Default(TVulkanBuffer);
  FGpuQScale := Default(TVulkanBuffer);
  FGpuOut1536 := Default(TVulkanBuffer);
  FGpuProj := Default(TVulkanBuffer);
  FGpuStageIn := Default(TVulkanBuffer);
  FGpuStageOut := Default(TVulkanBuffer);
end;

destructor TAudio.Destroy();
begin
  Close();
  // Owned GPU objects only; shared kernels/device belong to the caller and
  // MUST outlive this object (free TAudio before TInference)
  if FOwnsGpu then
  begin
    FKernels.Free();
    FDevice.Free();
  end;
  FWeightStore.Free();
  inherited;
end;

procedure TAudio.SetStatusCallback(const ACallback: TStatusCallback;
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

function TAudio.DoReadVpkFileAsString(const AVpkPath: string;
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

function TAudio.DoInitGpu(): Boolean;
begin
  Result := False;

  if not FOwnsGpu then
  begin
    // Shared mode: the caller's device/kernels are already live
    if (FKernels = nil) or (not FKernels.IsInitialized()) then
    begin
      GetErrors().Add(esError, CAUD_ERR_OPEN, 'Shared kernels not initialized');
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
    GetErrors().Add(esError, CAUD_ERR_OPEN, 'Vulkan init failed');
    Exit;
  end;
  Status('Vulkan initialized: %s', [FDevice.DeviceName]);

  if not FKernels.Init(FDevice) then
  begin
    GetErrors().Add(esError, CAUD_ERR_OPEN, 'Compute kernel init failed');
    FDevice.Shutdown();
    Exit;
  end;

  Result := True;
end;

procedure TAudio.DoShutdownGpu();
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

function TAudio.DoTensorLoc(const AName: string;
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

function TAudio.DoTensorPtr(const AName: string): PSingle;
var
  LInfo: TTensorInfo;
begin
  Result := nil;
  if FWeightStore.FindTensor(AName, LInfo) then
    Result := PSingle(FWeightStore.GetRawPointer(LInfo));
end;

function TAudio.DoUploadWeights(): Boolean;
var
  LInfo: TTensorInfo;
  LI: Integer;
  LMinOfs: UInt64;
  LMaxEnd: UInt64;
  LTotal: UInt64;
  LBase: Pointer;
  LIsAudio: Boolean;
begin
  Result := False;

  // The audio tensors (model.audio_tower.* + model.embed_audio.*) occupy
  // a contiguous byte window inside encoders.bin. Upload ONLY that window;
  // DoTensorLoc rebases offsets by FWinBase.
  LMinOfs := High(UInt64);
  LMaxEnd := 0;
  for LI := 0 to FWeightStore.TensorCount() - 1 do
  begin
    LInfo := FWeightStore.GetTensorInfo(LI);
    LIsAudio := LInfo.TensorName.StartsWith(CAudPrefix) or
      LInfo.TensorName.StartsWith(CAudEmbedPrefix);
    if not LIsAudio then
      Continue;
    if LInfo.Offset < LMinOfs then
      LMinOfs := LInfo.Offset;
    if LInfo.Offset + LInfo.DataSize > LMaxEnd then
      LMaxEnd := LInfo.Offset + LInfo.DataSize;
  end;
  if LMaxEnd <= LMinOfs then
  begin
    GetErrors().Add(esError, CAUD_ERR_OPEN, 'No audio tensors in manifest');
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
    GetErrors().Add(esError, CAUD_ERR_OPEN, 'Audio window base not found');
    Exit;
  end;

  FGpuWeights := FDevice.CreateBuffer(LTotal, False);
  if FGpuWeights.Buffer = 0 then
  begin
    GetErrors().Add(esError, CAUD_ERR_OPEN,
      'Audio weight buffer failed (%.1f MB)', [LTotal / (1024 * 1024)]);
    Exit;
  end;

  // Placement matters: the conformer streams these weights per layer
  if FGpuWeights.IsDeviceLocal then
    Status('Audio weights buffer: DEVICE-LOCAL (VRAM)')
  else
    Status('WARNING: audio weights buffer fell back to HOST memory');

  FDevice.UploadToDeviceBuffer(FGpuWeights, LBase, LTotal);
  Status('Audio weights uploaded: %.1f MB F32', [LTotal / (1024 * 1024)]);
  Result := True;
end;

function TAudio.DoAllocateBuffers(): Boolean;
var
  LN: UInt64;
  LH: UInt64;
begin
  LN := CAudMaxTokens;
  LH := UInt64(FConfig.HiddenSize);

  FGpuHidden := FDevice.CreateBuffer(LN * LH * SizeOf(Single), False);
  FGpuNorm := FDevice.CreateBuffer(LN * LH * SizeOf(Single), False);
  FGpuTmp := FDevice.CreateBuffer(LN * LH * SizeOf(Single), False);
  FGpuTmp2 := FDevice.CreateBuffer(LN * LH * SizeOf(Single), False);
  FGpuQ := FDevice.CreateBuffer(LN * LH * SizeOf(Single), False);
  FGpuK := FDevice.CreateBuffer(LN * LH * SizeOf(Single), False);
  FGpuV := FDevice.CreateBuffer(LN * LH * SizeOf(Single), False);
  FGpuAttnO := FDevice.CreateBuffer(LN * LH * SizeOf(Single), False);
  FGpuMlp := FDevice.CreateBuffer(LN * LH * 4 * SizeOf(Single), False);
  FGpuLc := FDevice.CreateBuffer(LN * LH * 2 * SizeOf(Single), False);
  FGpuRelK := FDevice.CreateBuffer(
    UInt64(FConfig.NumLayers) * 13 * LH * SizeOf(Single), False);
  FGpuQScale := FDevice.CreateBuffer(
    UInt64(FConfig.NumLayers) * UInt64(FConfig.HeadDim) * SizeOf(Single), False);
  FGpuOut1536 := FDevice.CreateBuffer(
    LN * UInt64(FConfig.OutputProjDims) * SizeOf(Single), False);
  FGpuProj := FDevice.CreateBuffer(
    LN * UInt64(FConfig.TextHiddenSize) * SizeOf(Single), False);
  FGpuStageIn := FDevice.CreateBuffer(
    LN * UInt64(FConfig.OutputProjDims) * SizeOf(Single), True);
  FGpuStageOut := FDevice.CreateBuffer(
    LN * UInt64(FConfig.TextHiddenSize) * SizeOf(Single), True);

  Result := (FGpuHidden.Buffer <> 0) and (FGpuNorm.Buffer <> 0) and
    (FGpuTmp.Buffer <> 0) and (FGpuTmp2.Buffer <> 0) and
    (FGpuQ.Buffer <> 0) and (FGpuK.Buffer <> 0) and (FGpuV.Buffer <> 0) and
    (FGpuAttnO.Buffer <> 0) and (FGpuMlp.Buffer <> 0) and
    (FGpuLc.Buffer <> 0) and (FGpuRelK.Buffer <> 0) and
    (FGpuQScale.Buffer <> 0) and (FGpuOut1536.Buffer <> 0) and
    (FGpuProj.Buffer <> 0) and (FGpuStageIn.Buffer <> 0) and
    (FGpuStageOut.Buffer <> 0);
  if not Result then
    GetErrors().Add(esError, CAUD_ERR_OPEN, 'Audio GPU buffers failed');
end;

procedure TAudio.DoFreeBuffers();
begin
  if (FDevice = nil) or (not FDevice.IsInitialized()) then
    Exit;
  if FGpuWeights.Buffer <> 0 then FDevice.DestroyBuffer(FGpuWeights);
  if FGpuHidden.Buffer <> 0 then FDevice.DestroyBuffer(FGpuHidden);
  if FGpuNorm.Buffer <> 0 then FDevice.DestroyBuffer(FGpuNorm);
  if FGpuTmp.Buffer <> 0 then FDevice.DestroyBuffer(FGpuTmp);
  if FGpuTmp2.Buffer <> 0 then FDevice.DestroyBuffer(FGpuTmp2);
  if FGpuQ.Buffer <> 0 then FDevice.DestroyBuffer(FGpuQ);
  if FGpuK.Buffer <> 0 then FDevice.DestroyBuffer(FGpuK);
  if FGpuV.Buffer <> 0 then FDevice.DestroyBuffer(FGpuV);
  if FGpuAttnO.Buffer <> 0 then FDevice.DestroyBuffer(FGpuAttnO);
  if FGpuMlp.Buffer <> 0 then FDevice.DestroyBuffer(FGpuMlp);
  if FGpuLc.Buffer <> 0 then FDevice.DestroyBuffer(FGpuLc);
  if FGpuRelK.Buffer <> 0 then FDevice.DestroyBuffer(FGpuRelK);
  if FGpuQScale.Buffer <> 0 then FDevice.DestroyBuffer(FGpuQScale);
  if FGpuOut1536.Buffer <> 0 then FDevice.DestroyBuffer(FGpuOut1536);
  if FGpuProj.Buffer <> 0 then FDevice.DestroyBuffer(FGpuProj);
  if FGpuStageIn.Buffer <> 0 then FDevice.DestroyBuffer(FGpuStageIn);
  if FGpuStageOut.Buffer <> 0 then FDevice.DestroyBuffer(FGpuStageOut);
  FGpuWeights := Default(TVulkanBuffer);
  FGpuHidden := Default(TVulkanBuffer);
  FGpuNorm := Default(TVulkanBuffer);
  FGpuTmp := Default(TVulkanBuffer);
  FGpuTmp2 := Default(TVulkanBuffer);
  FGpuQ := Default(TVulkanBuffer);
  FGpuK := Default(TVulkanBuffer);
  FGpuV := Default(TVulkanBuffer);
  FGpuAttnO := Default(TVulkanBuffer);
  FGpuMlp := Default(TVulkanBuffer);
  FGpuLc := Default(TVulkanBuffer);
  FGpuRelK := Default(TVulkanBuffer);
  FGpuQScale := Default(TVulkanBuffer);
  FGpuOut1536 := Default(TVulkanBuffer);
  FGpuProj := Default(TVulkanBuffer);
  FGpuStageIn := Default(TVulkanBuffer);
  FGpuStageOut := Default(TVulkanBuffer);
end;

function TAudio.DoReadClipScalar(const AName: string;
  out AValue: Single): Boolean;
var
  LInfo: TTensorInfo;
begin
  AValue := 0.0;
  Result := FWeightStore.FindTensor(AName, LInfo);
  if Result then
    AValue := PSingle(FWeightStore.GetRawPointer(LInfo))^;
end;

function TAudio.DoLoadClips(): Boolean;
var
  LI: Integer;
  LPrefix: string;

  function LoadOne(const ABase: string; out AClip: TAudClipBounds): Boolean;
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
    LPrefix := Format('%slayers.%d.', [CAudPrefix, LI]);
    if not (LoadOne(LPrefix + 'feed_forward1.ffw_layer_1', FClips[LI].Ff1A) and
      LoadOne(LPrefix + 'feed_forward1.ffw_layer_2', FClips[LI].Ff1B) and
      LoadOne(LPrefix + 'feed_forward2.ffw_layer_1', FClips[LI].Ff2A) and
      LoadOne(LPrefix + 'feed_forward2.ffw_layer_2', FClips[LI].Ff2B) and
      LoadOne(LPrefix + 'self_attn.q_proj', FClips[LI].AttnQ) and
      LoadOne(LPrefix + 'self_attn.k_proj', FClips[LI].AttnK) and
      LoadOne(LPrefix + 'self_attn.v_proj', FClips[LI].AttnV) and
      LoadOne(LPrefix + 'self_attn.post', FClips[LI].AttnPost) and
      LoadOne(LPrefix + 'lconv1d.linear_start', FClips[LI].LcStart) and
      LoadOne(LPrefix + 'lconv1d.linear_end', FClips[LI].LcEnd)) then
    begin
      GetErrors().Add(esError, CAUD_ERR_OPEN,
        'Missing audio clip scalars, layer %d', [LI]);
      Exit;
    end;
  end;
  Result := True;
end;

function TAudio.DoPrepareRelPos(): Boolean;
var
  LPos: TArray<Single>;
  LRelK: TArray<Single>;
  LW: PSingle;
  LH: Integer;
  LTs: Integer;
  LLogInc: Double;
  LInv: Double;
  LP: Integer;
  LT: Integer;
  LLayer: Integer;
  LR: Integer;
  LC: Integer;
  LAcc: Double;
  LName: string;
begin
  // Sinusoidal rel-pos table: 13 positions (12 down to 0), [sin | cos]
  // halves over 512 timescales (min 1, max 10000). Then each layer's
  // relative_k_proj (plain linear, no clipping) is applied on the CPU
  // once at Open; the result is uploaded as [layers, 13, hidden].
  Result := False;
  LH := FConfig.HiddenSize;
  LTs := LH div 2;
  LLogInc := Ln(10000.0) / (LTs - 1);

  SetLength(LPos, 13 * LH);
  for LP := 0 to 12 do
  begin
    for LT := 0 to LTs - 1 do
    begin
      LInv := Exp(-LT * LLogInc);
      LPos[LP * LH + LT] := Sin((12 - LP) * LInv);
      LPos[LP * LH + LTs + LT] := Cos((12 - LP) * LInv);
    end;
  end;

  SetLength(LRelK, FConfig.NumLayers * 13 * LH);
  for LLayer := 0 to FConfig.NumLayers - 1 do
  begin
    LName := Format('%slayers.%d.self_attn.relative_k_proj.weight',
      [CAudPrefix, LLayer]);
    LW := DoTensorPtr(LName);
    if LW = nil then
    begin
      GetErrors().Add(esError, CAUD_ERR_OPEN, 'Missing %s', [LName]);
      Exit;
    end;
    // out[p, r] = sum_c w[r, c] * pos[p, c]   (nn.Linear, bias=False)
    for LP := 0 to 12 do
    begin
      for LR := 0 to LH - 1 do
      begin
        LAcc := 0.0;
        for LC := 0 to LH - 1 do
          LAcc := LAcc + PSingle(UIntPtr(LW) +
            UIntPtr((UInt64(LR) * UInt64(LH) + UInt64(LC)) * SizeOf(Single)))^ *
            LPos[LP * LH + LC];
        LRelK[(LLayer * 13 + LP) * LH + LR] := LAcc;
      end;
    end;
  end;

  FDevice.UploadToDeviceBuffer(FGpuRelK, @LRelK[0],
    UInt64(Length(LRelK)) * SizeOf(Single));
  Result := True;
end;

function TAudio.DoPrepareQScales(): Boolean;
var
  LQs: TArray<Single>;
  LPds: PSingle;
  LLayer: Integer;
  LD: Integer;
  LScale: Double;
  LName: string;
  LV: Double;
begin
  // q' = q * softplus(per_dim_scale) * head_dim^-0.5 / ln(2) -- folded
  // into one per-layer per-dim vector consumed by the attention shader
  Result := False;
  LScale := Power(FConfig.HeadDim, -0.5) / Ln(2.0);
  SetLength(LQs, FConfig.NumLayers * FConfig.HeadDim);
  for LLayer := 0 to FConfig.NumLayers - 1 do
  begin
    LName := Format('%slayers.%d.self_attn.per_dim_scale',
      [CAudPrefix, LLayer]);
    LPds := DoTensorPtr(LName);
    if LPds = nil then
    begin
      GetErrors().Add(esError, CAUD_ERR_OPEN, 'Missing %s', [LName]);
      Exit;
    end;
    for LD := 0 to FConfig.HeadDim - 1 do
    begin
      LV := PSingle(UIntPtr(LPds) + UIntPtr(LD * SizeOf(Single)))^;
      // softplus(x) = ln(1 + e^x)
      LQs[LLayer * FConfig.HeadDim + LD] := Ln(1.0 + Exp(LV)) * LScale;
    end;
  end;

  FDevice.UploadToDeviceBuffer(FGpuQScale, @LQs[0],
    UInt64(Length(LQs)) * SizeOf(Single));
  Result := True;
end;

function TAudio.DoSubsampleCpu(const AFeatures: TArray<Single>;
  const ANumFrames: Integer; out AHidden: TArray<Single>;
  out ANumTokens: Integer): Boolean;
var
  LW0: PSingle;
  LG0: PSingle;
  LW1: PSingle;
  LG1: PSingle;
  LBuf0: TArray<Single>;
  LBuf1: TArray<Single>;
  LT0: Integer;
  LF0: Integer;
  LT1: Integer;
  LF1: Integer;
  LOc: Integer;
  LIc: Integer;
  LKh: Integer;
  LKw: Integer;
  LIh: Integer;
  LIw: Integer;
  LOt: Integer;
  LOf: Integer;
  LAcc: Double;
  LMean: Double;
  LVar: Double;
  LV: Double;
  LC0: Integer;
  LC1: Integer;

  function W0At(const AOc: Integer; const AKh: Integer;
    const AKw: Integer): Single;
  begin
    // layer0.conv.weight [C0, 1, 3, 3]
    Result := PSingle(UIntPtr(LW0) +
      UIntPtr(((AOc * 3 + AKh) * 3 + AKw) * SizeOf(Single)))^;
  end;

  function W1At(const AOc: Integer; const AIc: Integer; const AKh: Integer;
    const AKw: Integer): Single;
  begin
    // layer1.conv.weight [C1, C0, 3, 3]
    Result := PSingle(UIntPtr(LW1) +
      UIntPtr((((AOc * LC0 + AIc) * 3 + AKh) * 3 + AKw) * SizeOf(Single)))^;
  end;

begin
  // Gemma4AudioSubSampleConvProjection on the CPU (cost < 0.3 s):
  // two Conv2d(3x3, s2, p1, no bias) + LayerNorm over CHANNELS (scale
  // only, bias-free) + ReLU, then [T4, F4 * C1] with freq-major layout
  // (HF permute(0,2,3,1).reshape). Both intermediate buffers are stored
  // channel-minor [t, f, c] so the channel LayerNorm is contiguous.
  Result := False;
  AHidden := nil;
  ANumTokens := 0;
  LC0 := FConfig.SubCh0;   // 128
  LC1 := FConfig.SubCh1;   // 32

  LW0 := DoTensorPtr(CAudPrefix + 'subsample_conv_projection.layer0.conv.weight');
  LG0 := DoTensorPtr(CAudPrefix + 'subsample_conv_projection.layer0.norm.weight');
  LW1 := DoTensorPtr(CAudPrefix + 'subsample_conv_projection.layer1.conv.weight');
  LG1 := DoTensorPtr(CAudPrefix + 'subsample_conv_projection.layer1.norm.weight');
  if (LW0 = nil) or (LG0 = nil) or (LW1 = nil) or (LG1 = nil) then
  begin
    GetErrors().Add(esError, CAUD_ERR_ENCODE, 'Subsample tensors missing');
    Exit;
  end;

  // Conv 0: [T, 128] 1ch -> [T0, F0] with C0 channels (stride 2, pad 1)
  LT0 := (ANumFrames - 1) div 2 + 1;
  LF0 := (CAudMelBins - 1) div 2 + 1;   // 64
  SetLength(LBuf0, LT0 * LF0 * LC0);
  for LOt := 0 to LT0 - 1 do
  begin
    for LOf := 0 to LF0 - 1 do
    begin
      for LOc := 0 to LC0 - 1 do
      begin
        LAcc := 0.0;
        for LKh := 0 to 2 do
        begin
          LIh := LOt * 2 - 1 + LKh;
          if (LIh < 0) or (LIh >= ANumFrames) then
            Continue;
          for LKw := 0 to 2 do
          begin
            LIw := LOf * 2 - 1 + LKw;
            if (LIw < 0) or (LIw >= CAudMelBins) then
              Continue;
            LAcc := LAcc + AFeatures[LIh * CAudMelBins + LIw] *
              W0At(LOc, LKh, LKw);
          end;
        end;
        LBuf0[(LOt * LF0 + LOf) * LC0 + LOc] := LAcc;
      end;
      // LayerNorm over the C0 channels at (LOt, LOf), scale-only, + ReLU
      LMean := 0.0;
      for LOc := 0 to LC0 - 1 do
        LMean := LMean + LBuf0[(LOt * LF0 + LOf) * LC0 + LOc];
      LMean := LMean / LC0;
      LVar := 0.0;
      for LOc := 0 to LC0 - 1 do
      begin
        LV := LBuf0[(LOt * LF0 + LOf) * LC0 + LOc] - LMean;
        LVar := LVar + LV * LV;
      end;
      LVar := LVar / LC0;
      for LOc := 0 to LC0 - 1 do
      begin
        LV := (LBuf0[(LOt * LF0 + LOf) * LC0 + LOc] - LMean) /
          Sqrt(LVar + FConfig.RmsNormEps) *
          PSingle(UIntPtr(LG0) + UIntPtr(LOc * SizeOf(Single)))^;
        if LV < 0.0 then
          LV := 0.0;
        LBuf0[(LOt * LF0 + LOf) * LC0 + LOc] := LV;
      end;
    end;
  end;

  // Conv 1: [T0, F0] C0ch -> [T1, F1] C1ch (stride 2, pad 1)
  LT1 := (LT0 - 1) div 2 + 1;
  LF1 := (LF0 - 1) div 2 + 1;   // 32
  SetLength(LBuf1, LT1 * LF1 * LC1);
  for LOt := 0 to LT1 - 1 do
  begin
    for LOf := 0 to LF1 - 1 do
    begin
      for LOc := 0 to LC1 - 1 do
      begin
        LAcc := 0.0;
        for LKh := 0 to 2 do
        begin
          LIh := LOt * 2 - 1 + LKh;
          if (LIh < 0) or (LIh >= LT0) then
            Continue;
          for LKw := 0 to 2 do
          begin
            LIw := LOf * 2 - 1 + LKw;
            if (LIw < 0) or (LIw >= LF0) then
              Continue;
            for LIc := 0 to LC0 - 1 do
              LAcc := LAcc + LBuf0[(LIh * LF0 + LIw) * LC0 + LIc] *
                W1At(LOc, LIc, LKh, LKw);
          end;
        end;
        LBuf1[(LOt * LF1 + LOf) * LC1 + LOc] := LAcc;
      end;
      // LayerNorm over the C1 channels + ReLU
      LMean := 0.0;
      for LOc := 0 to LC1 - 1 do
        LMean := LMean + LBuf1[(LOt * LF1 + LOf) * LC1 + LOc];
      LMean := LMean / LC1;
      LVar := 0.0;
      for LOc := 0 to LC1 - 1 do
      begin
        LV := LBuf1[(LOt * LF1 + LOf) * LC1 + LOc] - LMean;
        LVar := LVar + LV * LV;
      end;
      LVar := LVar / LC1;
      for LOc := 0 to LC1 - 1 do
      begin
        LV := (LBuf1[(LOt * LF1 + LOf) * LC1 + LOc] - LMean) /
          Sqrt(LVar + FConfig.RmsNormEps) *
          PSingle(UIntPtr(LG1) + UIntPtr(LOc * SizeOf(Single)))^;
        if LV < 0.0 then
          LV := 0.0;
        LBuf1[(LOt * LF1 + LOf) * LC1 + LOc] := LV;
      end;
    end;
  end;

  // [T1, F1, C1] channel-minor IS the HF permute+reshape row layout
  ANumTokens := LT1;
  AHidden := LBuf1;
  Result := True;
end;

procedure TAudio.DoRecordNormRows(const ATensorName: string;
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

procedure TAudio.DoRecordClippedProj(const ATensorName: string;
  const AClip: TAudClipBounds; const AIn: TVulkanBuffer;
  const AOut: TVulkanBuffer; const ARows: Integer; const ACols: Integer;
  const ABatch: Integer; const AInElems: Integer);
var
  LOfs: UInt64;
  LSize: UInt64;
begin
  // QAT clipped linear: clamp(input) -> matmul -> clamp(output).
  // AIn is clamped IN-PLACE -- the caller must pass a consumable buffer.
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

procedure TAudio.DoRecordFeedForward(const APrefix: string;
  const AClipA: TAudClipBounds; const AClipB: TAudClipBounds;
  const AN: Integer);
var
  LH: Integer;
  LI4: Integer;
begin
  // Gemma4AudioFeedForward: residual = h; h = clamp(h); h = preNorm(h);
  // h = ffw1(h); h = silu(h); h = ffw2(h); h = clamp(h); h = postNorm(h);
  // h *= 0.5; h += residual   (residual stays in FGpuHidden)
  LH := FConfig.HiddenSize;
  LI4 := LH * 4;

  FKernels.RecordCopy(FGpuNorm, FGpuHidden, UInt32(AN * LH));
  FKernels.RecordBarrier();
  FKernels.RecordClamp(FGpuNorm, FGpuNorm, UInt32(AN * LH),
    -FConfig.GradClip, FConfig.GradClip);
  FKernels.RecordBarrier();
  DoRecordNormRows(APrefix + 'pre_layer_norm.weight', FGpuNorm, LH, AN);
  FKernels.RecordBarrier();

  DoRecordClippedProj(APrefix + 'ffw_layer_1.linear.weight', AClipA,
    FGpuNorm, FGpuMlp, LI4, LH, AN, AN * LH);
  FKernels.RecordSiLU(FGpuMlp, UInt32(AN * LI4));
  FKernels.RecordBarrier();
  DoRecordClippedProj(APrefix + 'ffw_layer_2.linear.weight', AClipB,
    FGpuMlp, FGpuTmp, LH, LI4, AN, AN * LI4);

  FKernels.RecordClamp(FGpuTmp, FGpuTmp, UInt32(AN * LH),
    -FConfig.GradClip, FConfig.GradClip);
  FKernels.RecordBarrier();
  DoRecordNormRows(APrefix + 'post_layer_norm.weight', FGpuTmp, LH, AN);
  FKernels.RecordBarrier();
  FKernels.RecordScale(FGpuTmp, UInt32(AN * LH), FConfig.ResidualWeight);
  FKernels.RecordBarrier();
  FKernels.RecordAdd(FGpuHidden, FGpuTmp, UInt32(AN * LH));
  FKernels.RecordBarrier();
end;

procedure TAudio.DoRecordLayer(const ALayerIdx: Integer; const AN: Integer);
var
  LPrefix: string;
  LH: Integer;
  LPush: TAudioAttnPushConstants;
  LConvPush: TDwConv1dPushConstants;
  LOfs: UInt64;
  LSize: UInt64;
begin
  LPrefix := Format('%slayers.%d.', [CAudPrefix, ALayerIdx]);
  LH := FConfig.HiddenSize;

  // ---- Feed-forward 1 (half residual) ----
  DoRecordFeedForward(LPrefix + 'feed_forward1.',
    FClips[ALayerIdx].Ff1A, FClips[ALayerIdx].Ff1B, AN);

  // ---- Attention block ----
  // residual = h; x = norm_pre_attn(clamp(h)); attn; clamp; norm_post_attn;
  // h += x
  FKernels.RecordCopy(FGpuNorm, FGpuHidden, UInt32(AN * LH));
  FKernels.RecordBarrier();
  FKernels.RecordClamp(FGpuNorm, FGpuNorm, UInt32(AN * LH),
    -FConfig.GradClip, FConfig.GradClip);
  FKernels.RecordBarrier();
  DoRecordNormRows(LPrefix + 'norm_pre_attn.weight', FGpuNorm, LH, AN);
  FKernels.RecordBarrier();

  // Q/K/V projections: FGpuNorm is shared, so each projection clamps a
  // fresh copy in FGpuTmp (input clamps differ per projection)
  FKernels.RecordCopy(FGpuTmp, FGpuNorm, UInt32(AN * LH));
  FKernels.RecordBarrier();
  DoRecordClippedProj(LPrefix + 'self_attn.q_proj.linear.weight',
    FClips[ALayerIdx].AttnQ, FGpuTmp, FGpuQ, LH, LH, AN, AN * LH);

  FKernels.RecordCopy(FGpuTmp, FGpuNorm, UInt32(AN * LH));
  FKernels.RecordBarrier();
  DoRecordClippedProj(LPrefix + 'self_attn.k_proj.linear.weight',
    FClips[ALayerIdx].AttnK, FGpuTmp, FGpuK, LH, LH, AN, AN * LH);

  FKernels.RecordCopy(FGpuTmp, FGpuNorm, UInt32(AN * LH));
  FKernels.RecordBarrier();
  DoRecordClippedProj(LPrefix + 'self_attn.v_proj.linear.weight',
    FClips[ALayerIdx].AttnV, FGpuTmp, FGpuV, LH, LH, AN, AN * LH);

  // Fused chunked local attention with rel-pos bias, softcap, softmax
  LPush.seqLen := UInt32(AN);
  LPush.numHeads := UInt32(FConfig.NumHeads);
  LPush.headDim := UInt32(FConfig.HeadDim);
  LPush.chunkSize := UInt32(FConfig.ChunkSize);
  LPush.ctxSize := UInt32(FConfig.CtxSize);
  LPush.pastH := UInt32(FConfig.CtxLeft - 1);
  LPush.layerIdx := UInt32(ALayerIdx);
  LPush.kScale := Ln(1.0 + Exp(1.0)) / Ln(2.0);
  LPush.softcap := FConfig.LogitCap;
  LPush.invalidVal := FConfig.InvalidLogit;
  FKernels.RecordAudioAttn(FGpuQ, FGpuK, FGpuV, FGpuRelK, FGpuQScale,
    FGpuAttnO, LPush);
  FKernels.RecordBarrier();

  DoRecordClippedProj(LPrefix + 'self_attn.post.linear.weight',
    FClips[ALayerIdx].AttnPost, FGpuAttnO, FGpuTmp, LH, LH, AN, AN * LH);

  FKernels.RecordClamp(FGpuTmp, FGpuTmp, UInt32(AN * LH),
    -FConfig.GradClip, FConfig.GradClip);
  FKernels.RecordBarrier();
  DoRecordNormRows(LPrefix + 'norm_post_attn.weight', FGpuTmp, LH, AN);
  FKernels.RecordBarrier();
  FKernels.RecordAdd(FGpuHidden, FGpuTmp, UInt32(AN * LH));
  FKernels.RecordBarrier();

  // ---- lconv block ----
  // residual = h; x = pre_layer_norm(h) [no clamp]; linear_start; GLU;
  // causal depthwise conv; clamp; conv_norm; silu; linear_end; h += x
  FKernels.RecordCopy(FGpuNorm, FGpuHidden, UInt32(AN * LH));
  FKernels.RecordBarrier();
  DoRecordNormRows(LPrefix + 'lconv1d.pre_layer_norm.weight', FGpuNorm,
    LH, AN);
  FKernels.RecordBarrier();

  DoRecordClippedProj(LPrefix + 'lconv1d.linear_start.linear.weight',
    FClips[ALayerIdx].LcStart, FGpuNorm, FGpuLc, LH * 2, LH, AN, AN * LH);

  FKernels.RecordGLU(FGpuTmp, FGpuLc, UInt32(AN * LH), UInt32(LH));
  FKernels.RecordBarrier();

  if not DoTensorLoc(LPrefix + 'lconv1d.depthwise_conv1d.weight',
    LOfs, LSize) then Exit;
  LConvPush.seqLen := UInt32(AN);
  LConvPush.channels := UInt32(LH);
  LConvPush.kernel := UInt32(FConfig.ConvKernel);
  LConvPush.wOffset := UInt32(LOfs div SizeOf(Single));
  FKernels.RecordDwConv1d(FGpuTmp2, FGpuTmp, FGpuWeights, LConvPush);
  FKernels.RecordBarrier();

  FKernels.RecordClamp(FGpuTmp2, FGpuTmp2, UInt32(AN * LH),
    -FConfig.GradClip, FConfig.GradClip);
  FKernels.RecordBarrier();
  DoRecordNormRows(LPrefix + 'lconv1d.conv_norm.weight', FGpuTmp2, LH, AN);
  FKernels.RecordBarrier();
  FKernels.RecordSiLU(FGpuTmp2, UInt32(AN * LH));
  FKernels.RecordBarrier();

  DoRecordClippedProj(LPrefix + 'lconv1d.linear_end.linear.weight',
    FClips[ALayerIdx].LcEnd, FGpuTmp2, FGpuTmp, LH, LH, AN, AN * LH);

  FKernels.RecordAdd(FGpuHidden, FGpuTmp, UInt32(AN * LH));
  FKernels.RecordBarrier();

  // ---- Feed-forward 2 (half residual) ----
  DoRecordFeedForward(LPrefix + 'feed_forward2.',
    FClips[ALayerIdx].Ff2A, FClips[ALayerIdx].Ff2B, AN);

  // ---- Layer tail: h = norm_out(clamp(h)) ----
  FKernels.RecordClamp(FGpuHidden, FGpuHidden, UInt32(AN * LH),
    -FConfig.GradClip, FConfig.GradClip);
  FKernels.RecordBarrier();
  DoRecordNormRows(LPrefix + 'norm_out.weight', FGpuHidden, LH, AN);
  FKernels.RecordBarrier();
end;

function TAudio.DoConformerForward(const AHidden: TArray<Single>;
  const AN: Integer; out ASoftTokens: TArray<Single>): Boolean;
var
  LOfs: UInt64;
  LSize: UInt64;
  LI: Integer;
  LJ: Integer;
  LH: Integer;
  LP: Integer;
  LT: Integer;
  LBias: PSingle;
  LBiasRows: TArray<Single>;
begin
  Result := False;
  LH := FConfig.HiddenSize;
  LP := FConfig.OutputProjDims;
  LT := FConfig.TextHiddenSize;

  if not DoTensorLoc(CAudPrefix + 'subsample_conv_projection.' +
    'input_proj_linear.weight', LOfs, LSize) then
  begin
    GetErrors().Add(esError, CAUD_ERR_ENCODE, 'input_proj_linear missing');
    Exit;
  end;

  // ---- Submit A: input projection + 12 conformer layers ----
  FDevice.UploadToBuffer(FGpuStageIn, @AHidden[0],
    UInt64(AN) * UInt64(LH) * SizeOf(Single));

  FKernels.Shaders.ResetDescriptorPool();
  FDevice.BeginCommands();

  FKernels.RecordMatMatF32(FGpuWeights, LOfs, LSize, FGpuStageIn, FGpuHidden,
    UInt32(LH), UInt32(LH), UInt32(AN));
  FKernels.RecordBarrier();

  for LI := 0 to FConfig.NumLayers - 1 do
    DoRecordLayer(LI, AN);

  FDevice.EndCommands();
  FDevice.SubmitAndWait();

  // ---- Submit B: output_proj (+bias), embedder norm + projection ----
  LBias := DoTensorPtr(CAudPrefix + 'output_proj.bias');
  if LBias = nil then
  begin
    GetErrors().Add(esError, CAUD_ERR_ENCODE, 'output_proj.bias missing');
    Exit;
  end;
  if not DoTensorLoc(CAudPrefix + 'output_proj.weight', LOfs, LSize) then
  begin
    GetErrors().Add(esError, CAUD_ERR_ENCODE, 'output_proj.weight missing');
    Exit;
  end;

  // Replicate the [1536] bias to [N x 1536] for a plain element-wise add
  SetLength(LBiasRows, AN * LP);
  for LI := 0 to AN - 1 do
    for LJ := 0 to LP - 1 do
      LBiasRows[LI * LP + LJ] :=
        PSingle(UIntPtr(LBias) + UIntPtr(LJ * SizeOf(Single)))^;
  FDevice.UploadToBuffer(FGpuStageIn, @LBiasRows[0],
    UInt64(AN) * UInt64(LP) * SizeOf(Single));

  FKernels.Shaders.ResetDescriptorPool();
  FDevice.BeginCommands();

  FKernels.RecordMatMatF32(FGpuWeights, LOfs, LSize, FGpuHidden, FGpuOut1536,
    UInt32(LP), UInt32(LH), UInt32(AN));
  FKernels.RecordBarrier();
  FKernels.RecordAdd(FGpuOut1536, FGpuStageIn, UInt32(AN * LP));
  FKernels.RecordBarrier();

  // MultimodalEmbedder: weight-free RMS norm, then [2560 x 1536] projection
  FKernels.RecordRmsNormRows(FGpuOut1536, FGpuWeights, UInt32(LP),
    UInt32(AN), FConfig.RmsNormEps, False);
  FKernels.RecordBarrier();

  if not DoTensorLoc(CAudEmbedProjName, LOfs, LSize) then
  begin
    GetErrors().Add(esError, CAUD_ERR_ENCODE, 'embedding_projection missing');
    Exit;
  end;
  FKernels.RecordMatMatF32(FGpuWeights, LOfs, LSize, FGpuOut1536, FGpuProj,
    UInt32(LT), UInt32(LP), UInt32(AN));
  FKernels.RecordBarrier();
  FKernels.RecordCopy(FGpuStageOut, FGpuProj, UInt32(AN * LT));

  FDevice.EndCommands();
  FDevice.SubmitAndWait();

  SetLength(ASoftTokens, AN * LT);
  FDevice.DownloadFromBuffer(FGpuStageOut, @ASoftTokens[0],
    UInt64(AN) * UInt64(LT) * SizeOf(Single));
  Result := True;
end;

function TAudio.Open(const AVpkPath: string;
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

  LConfigStr := DoReadVpkFileAsString(AVpkPath, CAudConfigPath);
  if LConfigStr = '' then
  begin
    GetErrors().Add(esError, CAUD_ERR_OPEN, 'Missing %s', [CAudConfigPath]);
    Exit;
  end;
  if not FConfig.LoadFromJson(LConfigStr) then
  begin
    GetErrors().Add(esError, CAUD_ERR_OPEN, 'Bad audio config');
    Exit;
  end;

  if not FWeightStore.Open(AVpkPath, CAudManifestPath, CAudWeightsPath) then
  begin
    GetErrors().Add(esError, CAUD_ERR_OPEN, 'Encoder weight store open failed');
    Exit;
  end;

  if not DoLoadClips() then Exit;
  if not DoInitGpu() then Exit;
  if not DoUploadWeights() then Exit;
  if not DoAllocateBuffers() then Exit;
  if not DoPrepareRelPos() then Exit;
  if not DoPrepareQScales() then Exit;

  FIsLoaded := True;
  Status('Audio encoder ready: %d layers, hidden %d, max %d tokens',
    [FConfig.NumLayers, FConfig.HiddenSize, CAudMaxTokens]);
  Result := True;
end;

procedure TAudio.Close();
begin
  DoFreeBuffers();
  DoShutdownGpu();
  FWeightStore.Close();
  SetLength(FClips, 0);
  FIsLoaded := False;
end;

function TAudio.IsLoaded(): Boolean;
begin
  Result := FIsLoaded;
end;

function TAudio.EncodeFeatures(const AFeatures: TArray<Single>;
  const ANumFrames: Integer; out ASoftTokens: TArray<Single>;
  out ANumTokens: Integer): Boolean;
var
  LHidden: TArray<Single>;
begin
  Result := False;
  SetLength(ASoftTokens, 0);
  ANumTokens := 0;
  if not FIsLoaded then
  begin
    GetErrors().Add(esError, CAUD_ERR_ENCODE, 'Audio encoder not loaded');
    Exit;
  end;
  if (ANumFrames <= 0) or
    (Length(AFeatures) <> ANumFrames * CAudMelBins) then
  begin
    GetErrors().Add(esError, CAUD_ERR_ENCODE,
      'Bad feature input: %d frames, %d values',
      [ANumFrames, Length(AFeatures)]);
    Exit;
  end;

  if not DoSubsampleCpu(AFeatures, ANumFrames, LHidden, ANumTokens) then
    Exit;
  if ANumTokens > CAudMaxTokens then
  begin
    GetErrors().Add(esError, CAUD_ERR_ENCODE,
      'Audio too long: %d tokens (max %d)', [ANumTokens, CAudMaxTokens]);
    Exit;
  end;

  if not DoConformerForward(LHidden, ANumTokens, ASoftTokens) then
    Exit;

  Status('Audio encoded: %d frames -> %d soft tokens',
    [ANumFrames, ANumTokens]);
  Result := True;
end;

function TAudio.EncodeAudio(const ASamples: TArray<Single>;
  out ASoftTokens: TArray<Single>; out ANumTokens: Integer): Boolean;
var
  LFeatures: TArray<Single>;
  LNumFrames: Integer;
begin
  Result := False;
  SetLength(ASoftTokens, 0);
  ANumTokens := 0;
  if not ComputeMel(ASamples, LFeatures, LNumFrames) then
  begin
    GetErrors().Add(esError, CAUD_ERR_ENCODE,
      'Mel extraction failed (%d samples, max %d)',
      [Length(ASamples), CAudMaxSamples]);
    Exit;
  end;
  Result := EncodeFeatures(LFeatures, LNumFrames, ASoftTokens, ANumTokens);
end;

end.
