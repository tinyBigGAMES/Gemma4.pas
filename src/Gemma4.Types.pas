{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information

 -------------------------------------------------------------------------------

  Gemma4.Types - Shared types, records, constants, and enums

  Foundation types used across all Gemma4 units. Defines data type
  enumerations, tensor metadata records, Q4_0 block layout, layer
  classification, attention kinds, and model-wide constants derived
  from the Gemma 4 E4B config.json.

  Dependencies: none
===============================================================================}

unit Gemma4.Types;

{$I StdApp.Defines.inc}

interface

uses
  System.SysUtils;

const
  // Model identity
  CModelName = 'Gemma 4 E4B';
  CModelVersion = '1.0.0';

  // Text decoder architecture (from config.json text_config)
  CHiddenSize = 2560;
  CNumHiddenLayers = 42;
  CNumAttentionHeads = 8;
  CNumKeyValueHeads = 2;
  CHeadDim = 256;
  CGlobalHeadDim = 512;
  CIntermediateSize = 10240;
  CVocabSize = 262144;
  CHiddenSizePerLayerInput = 256;
  CSlidingWindow = 512;
  CNumKvSharedLayers = 18;
  CNumUniqueKvCaches = CNumHiddenLayers - CNumKvSharedLayers; // 24
  CRmsNormEps: Single = 1e-6;
  CLogitSoftcap: Single = 30.0;

  // Max prompt positions processed per batched-prefill submission. Sizes the
  // Q8 activation scratch (Compute) and all batch buffers + sliding-ring
  // growth (Model). Sliding KV rings are SlidingWindow + CPrefillBatch
  // entries so a full chunk of appends never overwrites keys still needed
  // by earlier queries in the same chunk.
  CPrefillBatch = 256;

  // RoPE parameters
  CRopeThetaSliding: Single = 10000.0;
  CRopeThetaFull: Single = 1000000.0;
  CRopePartialRotaryFactor: Single = 0.25;

  // Token IDs
  CBosTokenId = 2;
  CEosTokenId1 = 1;
  CEosTokenId2 = 106;
  CEosTokenId3 = 50;
  CPadTokenId = 0;

  // Q4_0 quantization format
  CQ4BlockSize = 32; // number of weights per Q4_0 block
  CQ4BytesPerBlock = 18; // 2 bytes scale (fp16) + 16 bytes nibbles

  // Layer pattern: 5 sliding + 1 full, repeated 7 times = 42 layers
  CSlidingPatternRepeat = 5;
  CFullPatternInterval = 6; // every 6th layer is full attention

  // VPK paths
  CVpkManifestPath = 'E4B/manifest.json';
  CVpkConfigPath = 'E4B/config.json';
  CVpkTokenizerPath = 'E4B/tokenizer.json';
  CVpkTokenizerConfigPath = 'E4B/tokenizer_config.json';
  CVpkGenerationConfigPath = 'E4B/generation_config.json';
  CVpkChatTemplatePath = 'E4B/chat_template.jinja';
  CVpkWeightsPath = 'E4B/weights.bin';

type
  // -----------------------------------------------------------------------
  // Data type enumeration
  // -----------------------------------------------------------------------

  { TDtypeKind }
  // Identifies the storage format of a tensor's elements
  TDtypeKind = (
    dkUnknown,
    dkBF16,    // bfloat16 -- source format in safetensors
    dkF16,     // float16 -- IEEE half precision
    dkF32,     // float32 -- norms, scalars, biases
    dkQ4_0     // 4-bit quantized -- weight matrices
  );

  // -----------------------------------------------------------------------
  // Attention layer classification
  // -----------------------------------------------------------------------

  { TAttentionKind }
  // Distinguishes the two attention modes in Gemma 4's hybrid pattern
  TAttentionKind = (
    akSliding,  // local sliding window attention, head_dim=256
    akFull      // global full attention, head_dim=512
  );

  // -----------------------------------------------------------------------
  // Tensor metadata (manifest entry)
  // -----------------------------------------------------------------------

  { TTensorInfo }
  // Describes a single tensor's location and format within weights.bin
  TTensorInfo = record
    TensorName: string;
    SourceDtype: TDtypeKind;
    OutputDtype: TDtypeKind;
    Offset: UInt64;
    DataSize: UInt64;
    Shape: TArray<Integer>;
    function DimCount(): Integer;
    function ElementCount(): UInt64;
    function ToString(): string;
  end;

  // -----------------------------------------------------------------------
  // Q4_0 block layout
  // -----------------------------------------------------------------------

  { TQ4Block }
  // On-disk/in-memory layout of a single Q4_0 quantization block.
  // Contains one fp16 scale factor followed by 16 bytes of packed
  // 4-bit nibbles representing 32 quantized weights.
  TQ4Block = packed record
    Scale: UInt16;       // fp16 scale factor
    Nibbles: array[0..15] of Byte; // 32 x 4-bit values, packed two per byte
  end;

  // -----------------------------------------------------------------------
  // Layer type map
  // -----------------------------------------------------------------------

  { TLayerTypeArray }
  TLayerTypeArray = array[0..CNumHiddenLayers - 1] of TAttentionKind;

  // -----------------------------------------------------------------------
  // BFloat16 helper
  // -----------------------------------------------------------------------

  { TBFloat16 }
  // Minimal record for converting bfloat16 <-> float32
  TBFloat16 = packed record
    Bits: UInt16;
    class function ToSingle(const AValue: UInt16): Single; static; inline;
    class function FromSingle(const AValue: Single): UInt16; static; inline;
  end;

  // -----------------------------------------------------------------------
  // Float16 (IEEE 754 half) helper
  // -----------------------------------------------------------------------

  { TFloat16 }
  // Minimal record for converting float16 <-> float32
  TFloat16 = packed record
    Bits: UInt16;
    class function ToSingle(const AValue: UInt16): Single; static;
    class function FromSingle(const AValue: Single): UInt16; static;
  end;

  // -----------------------------------------------------------------------
  // Inference statistics
  // -----------------------------------------------------------------------

  { TStatsDetailKind }
  // Detail level for TInferenceStats.FormatText output
  TStatsDetailKind = (
    sdkBasic,     // headline only: prompt/generated token counts and speeds
    sdkDetailed   // headline + CPU phase split + GPU-internal split
  );

  { TInferencePhaseStats }
  // CPU-side per-token phase averages (milliseconds) for the GPU forward
  // path: host input uploads, command buffer recording, fence wait
  // (= actual GPU execution), logits readback, and CPU sampling.
  TInferencePhaseStats = record
    UploadMs: Double;
    CmdRecordMs: Double;
    FenceMs: Double;
    DownloadMs: Double;
    SampleMs: Double;
    TotalMs: Double;
  end;

  { TInferenceGpuStats }
  // GPU-internal per-token averages (milliseconds) from timestamp queries.
  // FullXxx = the instrumented full-attention layer, SlidXxx = the
  // instrumented sliding-window layer. ExtrapolatedLayersMs projects the
  // two instrumented layers across all layers (7 full + 35 sliding) as a
  // consistency check against LayersTotalMs.
  TInferenceGpuStats = record
    EmbedPlePreMs: Double;
    LayersTotalMs: Double;
    FinalLmHeadMs: Double;
    FullAttnMs: Double;
    FullMlpMs: Double;
    FullPleMs: Double;
    SlidAttnMs: Double;
    SlidMlpMs: Double;
    SlidPleMs: Double;
    ExtrapolatedLayersMs: Double;
  end;

  { TInferenceStats }
  // Complete post-generation statistics snapshot. Per-token phase and GPU
  // averages cover generation only; prompt processing is reported via the
  // Prefill fields. Populated via TModel.GetStats / TEngine.GetStats.
  // All fields are public so callers can render the data any way they
  // want; FormatText renders the standard report at the requested detail
  // level, with ANSI TTY color codes by default or plain text.
  TInferenceStats = record
    TokenCount: Integer;
    ElapsedSec: Double;
    TokensPerSec: Double;
    PrefillTokenCount: Integer;   // prompt tokens processed before generation
    PrefillSec: Double;           // wall-clock prompt processing time
    PrefillTokensPerSec: Double;  // prompt processing speed
    Position: Integer;            // absolute KV position after the last Generate (prompt + generated)
    HasGpuStats: Boolean;   // False on the CPU path or if timestamps failed
    Phases: TInferencePhaseStats;
    Gpu: TInferenceGpuStats;
    // Batched-prefill GPU split (per-chunk averages; zero when prefill ran
    // token-by-token or timestamps were unavailable)
    HasPrefillGpuStats: Boolean;
    PrefillChunkCount: Integer;
    PrefillGpu: TInferenceGpuStats;
    function FormatText(const ADetail: TStatsDetailKind = sdkBasic;
      const AColored: Boolean = True): string;
  end;

  { TProgressState }
  TProgressState = (
    psPrep,
    psStart,
    psInProgress,
    psEnd
  );

  { TLoadProgressCallback }
  TLoadProgressCallback = reference to procedure(
    const AState: TProgressState;
    const AStep: Integer;
    const ATotal: Integer;
    const AUserData: Pointer
  );

// -----------------------------------------------------------------------
// Utility functions
// -----------------------------------------------------------------------

function DtypeKindFromString(const AValue: string): TDtypeKind;
function DtypeKindToString(const AValue: TDtypeKind): string;
function AttentionKindFromString(const AValue: string): TAttentionKind;
function BuildLayerTypeMap(): TLayerTypeArray;
function IsEosToken(const ATokenId: Integer): Boolean;

implementation

// -----------------------------------------------------------------------
// ANSI TTY color escapes for TInferenceStats.FormatText. These mirror
// StdApp.Console's COLOR_* values but are declared locally so this
// foundation unit keeps its zero-dependency character.
// -----------------------------------------------------------------------
const
  { CANSI_RESET }
  CANSI_RESET = #27'[0m';

  { CANSI_CYAN }
  CANSI_CYAN = #27'[36m';

  { CANSI_WHITE }
  CANSI_WHITE = #27'[37m';

  { CANSI_YELLOW }
  CANSI_YELLOW = #27'[33m';

{ TInferenceStats }

function TInferenceStats.FormatText(const ADetail: TStatsDetailKind;
  const AColored: Boolean): string;
var
  LHead: string;
  LText: string;
  LNum: string;
  LReset: string;

  // Append one line to Result, wrapped in the body text color
  procedure DoAddLine(const ALine: string);
  begin
    Result := Result + sLineBreak + LText + ALine + LReset;
  end;

begin
  if AColored then
  begin
    LHead := CANSI_CYAN;
    LText := CANSI_WHITE;
    LNum := CANSI_YELLOW;
    LReset := CANSI_RESET;
  end
  else
  begin
    LHead := '';
    LText := '';
    LNum := '';
    LReset := '';
  end;

  Result := LHead + 'Inference Statistics' + LReset;

  if PrefillTokenCount > 0 then
    DoAddLine(Format('  Prompt: %s%d%s tokens in %s%.1f%s s (%s%.1f%s tok/s prefill)',
      [LNum, PrefillTokenCount, LText, LNum, PrefillSec, LText,
       LNum, PrefillTokensPerSec, LText]));

  DoAddLine(Format('  Generated %s%d%s tokens in %s%.1f%s s (%s%.1f%s tok/s)',
    [LNum, TokenCount, LText, LNum, ElapsedSec, LText,
     LNum, TokensPerSec, LText]));

  // Basic detail stops at the headline numbers
  if ADetail = sdkBasic then
    Exit;

  DoAddLine(Format('  Per-token avg: upload %.2f | record %.2f | ' +
    'fence %.2f | download %.2f | sample %.2f | sum %.2f ms',
    [Phases.UploadMs, Phases.CmdRecordMs, Phases.FenceMs,
     Phases.DownloadMs, Phases.SampleMs, Phases.TotalMs]));

  if HasGpuStats then
  begin
    DoAddLine(Format('  GPU split: embed+ple-pre %.2f | %d layers %.2f | ' +
      'final+lm_head %.2f ms',
      [Gpu.EmbedPlePreMs, CNumHiddenLayers, Gpu.LayersTotalMs,
       Gpu.FinalLmHeadMs]));
    DoAddLine(Format('  Full layer: attn %.3f mlp %.3f ple %.3f ms | ' +
      'Sliding layer: attn %.3f mlp %.3f ple %.3f ms',
      [Gpu.FullAttnMs, Gpu.FullMlpMs, Gpu.FullPleMs,
       Gpu.SlidAttnMs, Gpu.SlidMlpMs, Gpu.SlidPleMs]));
    DoAddLine(Format('  Extrapolated layers (7 full + 35 sliding): %.2f ms',
      [Gpu.ExtrapolatedLayersMs]));
  end;

  if HasPrefillGpuStats then
  begin
    DoAddLine(Format('  Prefill GPU split (per chunk, %d chunks): ' +
      'embed+ple-pre %.2f | %d layers %.2f | final+lm_head %.2f ms',
      [PrefillChunkCount, PrefillGpu.EmbedPlePreMs, CNumHiddenLayers,
       PrefillGpu.LayersTotalMs, PrefillGpu.FinalLmHeadMs]));
    DoAddLine(Format('  Prefill full layer: attn %.3f mlp %.3f ple %.3f ms | ' +
      'Sliding layer: attn %.3f mlp %.3f ple %.3f ms',
      [PrefillGpu.FullAttnMs, PrefillGpu.FullMlpMs, PrefillGpu.FullPleMs,
       PrefillGpu.SlidAttnMs, PrefillGpu.SlidMlpMs, PrefillGpu.SlidPleMs]));
    DoAddLine(Format('  Prefill extrapolated layers (7 full + 35 sliding): %.2f ms',
      [PrefillGpu.ExtrapolatedLayersMs]));
  end;
end;

{ TTensorInfo }

function TTensorInfo.DimCount(): Integer;
begin
  Result := Length(Shape);
end;

function TTensorInfo.ElementCount(): UInt64;
var
  LI: Integer;
begin
  if Length(Shape) = 0 then
    Exit(1); // scalar

  Result := 1;
  for LI := 0 to High(Shape) do
    Result := Result * UInt64(Shape[LI]);
end;

function TTensorInfo.ToString(): string;
var
  LShapeStr: string;
  LI: Integer;
begin
  LShapeStr := '[';
  for LI := 0 to High(Shape) do
  begin
    if LI > 0 then
      LShapeStr := LShapeStr + ', ';
    LShapeStr := LShapeStr + IntToStr(Shape[LI]);
  end;
  LShapeStr := LShapeStr + ']';

  Result := Format('%s %s -> %s offset=%d size=%d shape=%s', [
    TensorName,
    DtypeKindToString(SourceDtype),
    DtypeKindToString(OutputDtype),
    Offset,
    DataSize,
    LShapeStr
  ]);
end;

{ TBFloat16 }

class function TBFloat16.ToSingle(const AValue: UInt16): Single;
var
  LBits: UInt32;
begin
  // bf16 is the upper 16 bits of a float32, so shift left by 16
  LBits := UInt32(AValue) shl 16;
  Move(LBits, Result, SizeOf(Single));
end;

class function TBFloat16.FromSingle(const AValue: Single): UInt16;
var
  LBits: UInt32;
begin
  Move(AValue, LBits, SizeOf(UInt32));
  // Truncate lower 16 bits (round-to-nearest-even would be better
  // but truncation matches the common fast path)
  Result := UInt16(LBits shr 16);
end;

{ TFloat16 }

class function TFloat16.ToSingle(const AValue: UInt16): Single;
var
  LSign: UInt32;
  LExponent: UInt32;
  LMantissa: UInt32;
  LBits: UInt32;
begin
  LSign := (UInt32(AValue) shr 15) and 1;
  LExponent := (UInt32(AValue) shr 10) and $1F;
  LMantissa := UInt32(AValue) and $3FF;

  if LExponent = 0 then
  begin
    if LMantissa = 0 then
    begin
      // +/- zero
      LBits := LSign shl 31;
    end
    else
    begin
      // Denormalized: convert to normalized float32
      while (LMantissa and $400) = 0 do
      begin
        LMantissa := LMantissa shl 1;
        Inc(LExponent);
      end;
      Inc(LExponent);
      LMantissa := LMantissa and $3FF;
      LBits := (LSign shl 31) or ((LExponent + (127 - 15)) shl 23) or (LMantissa shl 13);
    end;
  end
  else if LExponent = $1F then
  begin
    // Inf or NaN
    LBits := (LSign shl 31) or ($FF shl 23) or (LMantissa shl 13);
  end
  else
  begin
    // Normalized
    LBits := (LSign shl 31) or ((LExponent + (127 - 15)) shl 23) or (LMantissa shl 13);
  end;

  Move(LBits, Result, SizeOf(Single));
end;

class function TFloat16.FromSingle(const AValue: Single): UInt16;
var
  LBits: UInt32;
  LSign: UInt32;
  LExponent: Int32;
  LMantissa: UInt32;
begin
  Move(AValue, LBits, SizeOf(UInt32));
  LSign := (LBits shr 31) and 1;
  LExponent := Int32((LBits shr 23) and $FF) - 127 + 15;
  LMantissa := LBits and $7FFFFF;

  if LExponent <= 0 then
    Result := UInt16(LSign shl 15) // flush to zero
  else if LExponent >= $1F then
    Result := UInt16((LSign shl 15) or ($1F shl 10)) // infinity
  else
    Result := UInt16((LSign shl 15) or (UInt32(LExponent) shl 10) or (LMantissa shr 13));
end;

{ Utility functions }

function DtypeKindFromString(const AValue: string): TDtypeKind;
begin
  if AValue = 'BF16' then
    Result := dkBF16
  else if AValue = 'F16' then
    Result := dkF16
  else if AValue = 'F32' then
    Result := dkF32
  else if AValue = 'Q4_0' then
    Result := dkQ4_0
  else
    Result := dkUnknown;
end;

function DtypeKindToString(const AValue: TDtypeKind): string;
begin
  case AValue of
    dkBF16:    Result := 'BF16';
    dkF16:     Result := 'F16';
    dkF32:     Result := 'F32';
    dkQ4_0:    Result := 'Q4_0';
  else
    Result := 'UNKNOWN';
  end;
end;

function AttentionKindFromString(const AValue: string): TAttentionKind;
begin
  if AValue = 'full_attention' then
    Result := akFull
  else
    Result := akSliding; // default to sliding
end;

function BuildLayerTypeMap(): TLayerTypeArray;
var
  LI: Integer;
begin
  // Pattern: 5 sliding + 1 full, repeated 7 times = 42 layers
  // Full attention at indices 5, 11, 17, 23, 29, 35, 41
  for LI := 0 to CNumHiddenLayers - 1 do
  begin
    if ((LI + 1) mod CFullPatternInterval) = 0 then
      Result[LI] := akFull
    else
      Result[LI] := akSliding;
  end;
end;

function IsEosToken(const ATokenId: Integer): Boolean;
begin
  Result := (ATokenId = CEosTokenId1) or
            (ATokenId = CEosTokenId2) or
            (ATokenId = CEosTokenId3);
end;

end.
