{===============================================================================
  Gemma4.pas� - Local LLM inference in Pascal

  Copyright � 2026-present tinyBigGAMES� LLC
  All Rights Reserved.

  See LICENSE for license information

 -------------------------------------------------------------------------------

  Gemma4.Attention - Attention mechanism (CPU reference)

  Implements the hybrid attention used by Gemma 4 E4B:
  - Rotary Position Embeddings (RoPE) with two configurations:
    - Sliding: theta=10000, default, full rotation
    - Full: theta=1000000, proportional, partial_rotary_factor=0.25
  - Grouped Query Attention (GQA) with 8 Q heads and 2 KV heads
  - Sliding window attention (window=512) for local layers
  - Full attention for global layers
  - KV cache with shared cache routing (18 shared layers)
  - Logit soft-capping at 30.0

  Key types:
  - TKvCache: Stores key/value tensors for all unique cache slots
  - TAttention: Computes single-head or multi-head attention

  Dependencies: Gemma4.Types, Gemma4.Tensors, Gemma4.Layers
===============================================================================}

unit Gemma4.Attention;

{$I StdApp.Defines.inc}

interface

uses
  System.SysUtils,
  System.Math,
  StdApp.Base,
  Gemma4.Types,
  Gemma4.Tensors,
  Gemma4.Layers;

type
  { TKvCacheSlot }
  // Single KV cache slot for one unique cache index.
  // Stores key and value vectors for all positions up to max sequence length.
  TKvCacheSlot = record
    Keys: TArray<TArray<Single>>;    // [seq_pos][head_dim]
    Values: TArray<TArray<Single>>;  // [seq_pos][head_dim]
    Length: Integer;                  // number of positions stored
    MaxLength: Integer;              // allocated capacity
  end;

  { TKvCache }
  // Manages all KV cache slots for the model.
  // Gemma 4 E4B has 24 unique slots (42 layers - 18 shared = 24).
  TKvCache = class(TBaseObject)
  private
    FSlots: TArray<TKvCacheSlot>;
    FNumSlots: Integer;
    FNumKvHeads: Integer;
    FMaxSeqLen: Integer;
  public
    constructor Create(); override;
    destructor Destroy(); override;

    procedure Init(
      const ANumSlots: Integer;
      const ANumKvHeads: Integer;
      const AHeadDim: Integer;
      const AMaxSeqLen: Integer
    );

    // Append a key/value pair at the current position for a given slot and head
    procedure Append(
      const ASlotIdx: Integer;
      const AHeadIdx: Integer;
      const AKey: PSingle;
      const AValue: PSingle;
      const AHeadDim: Integer
    );

    // Get current sequence length for a slot
    function GetLength(const ASlotIdx: Integer): Integer;

    // Get key/value pointer for a specific slot, position, and head
    function GetKeyPtr(
      const ASlotIdx: Integer;
      const APosition: Integer
    ): PSingle;

    function GetValuePtr(
      const ASlotIdx: Integer;
      const APosition: Integer
    ): PSingle;

    // Reset all caches
    procedure Clear();
  end;

// -----------------------------------------------------------------------
// RoPE (Rotary Position Embeddings)
// -----------------------------------------------------------------------

// Apply RoPE to a vector in-place.
// AData: [head_dim] vector to rotate
// APosition: token position index
// AHeadDim: dimension of the head
// ATheta: base frequency (10000 for sliding, 1000000 for full)
// APartialFactor: fraction of dims to rotate (1.0 for sliding, 0.25 for full)
procedure CpuApplyRoPE(
  const AData: PSingle;
  const APosition: Integer;
  const AHeadDim: Integer;
  const ATheta: Single;
  const APartialFactor: Single
);

// -----------------------------------------------------------------------
// Attention computation
// -----------------------------------------------------------------------

// Compute multi-head attention for a single token position.
//
// AQuery: [num_heads * head_dim] -- query vector for current position
// AKvCache: KV cache object
// AKvSlotIdx: which KV cache slot to use
// AOutput: [num_heads * head_dim] -- output vector (pre-allocated)
// ANumHeads: number of query heads (8)
// ANumKvHeads: number of KV heads (2) -- GQA
// AHeadDim: dimension per head (256 or 512)
// ACurrentPos: current token position (for causal mask)
// ASlidingWindow: window size (512 for sliding, 0 for full = no window limit)
// ASoftcap: logit soft-capping value (30.0, or 0.0 to disable)
procedure CpuMultiHeadAttention(
  const AQuery: PSingle;
  const AKvCache: TKvCache;
  const AKvSlotIdx: Integer;
  const AOutput: PSingle;
  const ANumHeads: Integer;
  const ANumKvHeads: Integer;
  const AHeadDim: Integer;
  const ACurrentPos: Integer;
  const ASlidingWindow: Integer;
  const ASoftcap: Single
);

implementation

{ TKvCache }

constructor TKvCache.Create();
begin
  inherited Create();
  FNumSlots := 0;
  FNumKvHeads := 0;
  FMaxSeqLen := 0;
end;

destructor TKvCache.Destroy();
begin
  inherited;
end;

procedure TKvCache.Init(
  const ANumSlots: Integer;
  const ANumKvHeads: Integer;
  const AHeadDim: Integer;
  const AMaxSeqLen: Integer);
var
  LI: Integer;
begin
  FNumSlots := ANumSlots;
  FNumKvHeads := ANumKvHeads;
  FMaxSeqLen := AMaxSeqLen;

  SetLength(FSlots, ANumSlots);
  for LI := 0 to ANumSlots - 1 do
  begin
    FSlots[LI].Length := 0;
    FSlots[LI].MaxLength := AMaxSeqLen;
    SetLength(FSlots[LI].Keys, AMaxSeqLen);
    SetLength(FSlots[LI].Values, AMaxSeqLen);
  end;
end;

procedure TKvCache.Append(
  const ASlotIdx: Integer;
  const AHeadIdx: Integer;
  const AKey: PSingle;
  const AValue: PSingle;
  const AHeadDim: Integer);
var
  LPos: Integer;
  LOffset: Integer;
  LTotalDim: Integer;
begin
  LPos := FSlots[ASlotIdx].Length;
  LTotalDim := FNumKvHeads * AHeadDim;
  LOffset := AHeadIdx * AHeadDim;

  // Allocate the position arrays if not yet done
  if Length(FSlots[ASlotIdx].Keys[LPos]) = 0 then
  begin
    SetLength(FSlots[ASlotIdx].Keys[LPos], LTotalDim);
    SetLength(FSlots[ASlotIdx].Values[LPos], LTotalDim);
  end;

  // Copy key and value for this head into the position
  Move(AKey^, FSlots[ASlotIdx].Keys[LPos][LOffset], AHeadDim * SizeOf(Single));
  Move(AValue^, FSlots[ASlotIdx].Values[LPos][LOffset], AHeadDim * SizeOf(Single));

  // Only increment length after all heads for this position are written.
  // The caller is responsible for calling Append for each KV head, then
  // incrementing via a separate mechanism. For simplicity, we increment
  // after the last head writes.
  if AHeadIdx = FNumKvHeads - 1 then
    Inc(FSlots[ASlotIdx].Length);
end;

function TKvCache.GetLength(const ASlotIdx: Integer): Integer;
begin
  Result := FSlots[ASlotIdx].Length;
end;

function TKvCache.GetKeyPtr(
  const ASlotIdx: Integer;
  const APosition: Integer): PSingle;
begin
  Result := @FSlots[ASlotIdx].Keys[APosition][0];
end;

function TKvCache.GetValuePtr(
  const ASlotIdx: Integer;
  const APosition: Integer): PSingle;
begin
  Result := @FSlots[ASlotIdx].Values[APosition][0];
end;

procedure TKvCache.Clear();
var
  LI: Integer;
begin
  for LI := 0 to FNumSlots - 1 do
    FSlots[LI].Length := 0;
end;

{ RoPE }

procedure CpuApplyRoPE(
  const AData: PSingle;
  const APosition: Integer;
  const AHeadDim: Integer;
  const ATheta: Single;
  const APartialFactor: Single);
var
  LI: Integer;
  LRotaryDim: Integer;
  LRotAngles: Integer;
  LHalfDim: Integer;
  LFreq: Single;
  LAngle: Single;
  LCosA: Single;
  LSinA: Single;
  LX0: Single;
  LX1: Single;
begin
  // NEOX pairing offset is ALWAYS head_dim/2. HF rotate_half splits the FULL
  // head vector in half regardless of how many angle pairs actually rotate.
  LHalfDim := AHeadDim div 2;

  // Number of angle pairs that rotate. Proportional RoPE (full attention,
  // partial_rotary_factor=0.25, head_dim=512): inv_freq has nonzero entries
  // only for the first int(0.25*512)/2 = 64 pairs, computed over the FULL
  // head_dim; the remaining pairs are zero-frequency (cos=1, sin=0) so they
  // pass through unchanged. Sliding (factor=1.0, head_dim=256): all 128 pairs.
  // Reference: transformers modeling_rope_utils._compute_proportional_rope_parameters
  LRotaryDim := Round(AHeadDim * APartialFactor);
  // Must be even
  if (LRotaryDim mod 2) <> 0 then
    Dec(LRotaryDim);
  LRotAngles := LRotaryDim div 2;

  // NEOX-style half-split rotation (matches HuggingFace Gemma 4):
  // Pairs are (i, i + half) NOT adjacent (i, i+1).
  //
  // HF rotate_half: [-x2, x1] where x1 = first half, x2 = second half
  // output = x * cos + rotate_half(x) * sin
  //
  // For element i (first half):
  //   out[i] = x[i] * cos - x[i + half] * sin
  // For element i + half (second half):
  //   out[i + half] = x[i + half] * cos + x[i] * sin
  //
  // Frequency for pair i: 1 / theta^(2*i / dim)
  for LI := 0 to LRotAngles - 1 do
  begin
    LFreq := 1.0 / Power(ATheta, Single(LI * 2) / Single(AHeadDim));
    LAngle := Single(APosition) * LFreq;
    LCosA := Cos(LAngle);
    LSinA := Sin(LAngle);

    LX0 := PSingle(UIntPtr(AData) + UIntPtr(LI * SizeOf(Single)))^;
    LX1 := PSingle(UIntPtr(AData) + UIntPtr((LI + LHalfDim) * SizeOf(Single)))^;

    // Rotation: first half and second half cross-paired
    PSingle(UIntPtr(AData) + UIntPtr(LI * SizeOf(Single)))^ :=
      LX0 * LCosA - LX1 * LSinA;
    PSingle(UIntPtr(AData) + UIntPtr((LI + LHalfDim) * SizeOf(Single)))^ :=
      LX1 * LCosA + LX0 * LSinA;
  end;
  // Angle pairs beyond LRotAngles are zero-frequency: identity pass-through.
  // Pairs (LI, LI + LHalfDim) for LI in [LRotAngles..LHalfDim-1] are untouched.
end;

{ Multi-head attention }

procedure CpuMultiHeadAttention(
  const AQuery: PSingle;
  const AKvCache: TKvCache;
  const AKvSlotIdx: Integer;
  const AOutput: PSingle;
  const ANumHeads: Integer;
  const ANumKvHeads: Integer;
  const AHeadDim: Integer;
  const ACurrentPos: Integer;
  const ASlidingWindow: Integer;
  const ASoftcap: Single);
var
  LHead: Integer;
  LKvHead: Integer;
  LPos: Integer;
  LDim: Integer;
  LSeqLen: Integer;
  LStartPos: Integer;
  LScore: Single;
  LQPtr: PSingle;
  LKPtr: PSingle;
  LVPtr: PSingle;
  LScores: TArray<Single>;
  LHeadOutput: TArray<Single>;
  LKvHeadOffset: Integer;
  LNumPositions: Integer;
  LScoreIdx: Integer;
begin
  LSeqLen := AKvCache.GetLength(AKvSlotIdx);

  // Determine the start position for sliding window
  if (ASlidingWindow > 0) and (LSeqLen > ASlidingWindow) then
    LStartPos := LSeqLen - ASlidingWindow
  else
    LStartPos := 0;

  LNumPositions := LSeqLen - LStartPos;
  if LNumPositions <= 0 then
    Exit;

  SetLength(LScores, LNumPositions);
  SetLength(LHeadOutput, AHeadDim);

  // Process each query head
  for LHead := 0 to ANumHeads - 1 do
  begin
    // GQA: map query head to KV head
    // With 8 Q heads and 2 KV heads, heads 0-3 share KV head 0,
    // heads 4-7 share KV head 1
    LKvHead := LHead div (ANumHeads div ANumKvHeads);
    LKvHeadOffset := LKvHead * AHeadDim;

    LQPtr := PSingle(UIntPtr(AQuery) + UIntPtr(LHead * AHeadDim * SizeOf(Single)));

    // Compute attention scores: Q * K^T / sqrt(head_dim)
    LScoreIdx := 0;
    for LPos := LStartPos to LSeqLen - 1 do
    begin
      LKPtr := AKvCache.GetKeyPtr(AKvSlotIdx, LPos);
      // Offset to the correct KV head within the concatenated KV vector
      LKPtr := PSingle(UIntPtr(LKPtr) + UIntPtr(LKvHeadOffset * SizeOf(Single)));

      // Dot product Q . K
      LScore := 0.0;
      for LDim := 0 to AHeadDim - 1 do
        LScore := LScore +
          PSingle(UIntPtr(LQPtr) + UIntPtr(LDim * SizeOf(Single)))^ *
          PSingle(UIntPtr(LKPtr) + UIntPtr(LDim * SizeOf(Single)))^;

      // Gemma 4 uses attention scaling = 1.0 (no division by sqrt(head_dim))
      // The Q/K per-head norms replace the need for this scaling.

      // Logit soft-capping
      if ASoftcap > 0.0 then
        LScore := Tanh(LScore / ASoftcap) * ASoftcap;

      // Causal mask: positions after current are -inf
      if LPos > ACurrentPos then
        LScore := -1e9;

      LScores[LScoreIdx] := LScore;
      Inc(LScoreIdx);
    end;

    // Softmax over scores
    CpuSoftmaxF32(@LScores[0], LNumPositions);

    // Weighted sum of values
    FillChar(LHeadOutput[0], AHeadDim * SizeOf(Single), 0);
    LScoreIdx := 0;
    for LPos := LStartPos to LSeqLen - 1 do
    begin
      LVPtr := AKvCache.GetValuePtr(AKvSlotIdx, LPos);
      LVPtr := PSingle(UIntPtr(LVPtr) + UIntPtr(LKvHeadOffset * SizeOf(Single)));

      for LDim := 0 to AHeadDim - 1 do
        LHeadOutput[LDim] := LHeadOutput[LDim] +
          LScores[LScoreIdx] *
          PSingle(UIntPtr(LVPtr) + UIntPtr(LDim * SizeOf(Single)))^;

      Inc(LScoreIdx);
    end;

    // Copy head output to the correct position in the output vector
    Move(LHeadOutput[0],
      PSingle(UIntPtr(AOutput) + UIntPtr(LHead * AHeadDim * SizeOf(Single)))^,
      AHeadDim * SizeOf(Single));
  end;
end;

end.
