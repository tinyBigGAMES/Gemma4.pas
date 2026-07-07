{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information

 -------------------------------------------------------------------------------

  Gemma4.Quant - Q4_0 quantization and dequantization

  Converts tensor data between formats used in the packing and
  inference pipelines:

  Packing (offline):
    bf16 buffer -> f32 buffer  (BF16ToF32Buffer)
    f32 buffer  -> Q4_0 blocks (F32ToQ4Buffer)

  Inference (runtime):
    Q4_0 block  -> f32 values  (DequantQ4Block)
    Q4_0 buffer -> f32 buffer  (Q4ToF32Buffer)

  Q4_0 format: groups of 32 weights, each group stored as one
  TQ4Block (18 bytes = 2-byte fp16 scale + 16 bytes of packed
  4-bit nibbles). Each nibble stores a signed value in [0..15]
  representing the quantized weight offset by 8 (i.e. actual
  value = nibble - 8, range [-8..+7]).

  Dependencies: Gemma4.Types
===============================================================================}

unit Gemma4.Quant;

{$I StdApp.Defines.inc}

interface

uses
  System.SysUtils,
  System.Math,
  StdApp.Base,
  Gemma4.Types;

// -----------------------------------------------------------------------
// Buffer conversion (packing pipeline)
// -----------------------------------------------------------------------

// Convert a buffer of bf16 values to f32.
// ASrc points to packed UInt16 bf16 values.
// ADst points to pre-allocated Single buffer.
// ACount is the number of elements.
procedure BF16ToF32Buffer(
  const ASrc: Pointer;
  const ADst: Pointer;
  const ACount: UInt64
);

// Quantize a buffer of f32 values to Q4_0 blocks.
// ASrc points to Single values (count must be a multiple of 32).
// ADst points to pre-allocated TQ4Block buffer.
// ACount is the number of f32 elements (must be multiple of CQ4BlockSize).
// Returns the number of Q4 blocks written.
function F32ToQ4Buffer(
  const ASrc: Pointer;
  const ADst: Pointer;
  const ACount: UInt64
): UInt64;

// Calculate the number of Q4_0 blocks needed for AElementCount f32 values.
// Rounds up to the nearest multiple of CQ4BlockSize.
function Q4BlockCount(const AElementCount: UInt64): UInt64;

// Calculate the byte size of Q4_0 storage for AElementCount f32 values.
function Q4ByteSize(const AElementCount: UInt64): UInt64;

// -----------------------------------------------------------------------
// Single-block operations (inference pipeline)
// -----------------------------------------------------------------------

// Dequantize a single Q4_0 block into 32 f32 values.
// ADst must point to a buffer of at least 32 Singles.
procedure DequantQ4Block(
  const ABlock: TQ4Block;
  const ADst: PSingle
);

// Dequantize a buffer of Q4_0 blocks into f32.
// ASrc points to packed TQ4Block data.
// ADst points to pre-allocated Single buffer.
// ABlockCount is the number of Q4 blocks.
procedure Q4ToF32Buffer(
  const ASrc: Pointer;
  const ADst: Pointer;
  const ABlockCount: UInt64
);

implementation

procedure BF16ToF32Buffer(
  const ASrc: Pointer;
  const ADst: Pointer;
  const ACount: UInt64
);
var
  LSrcPtr: PUInt16;
  LDstPtr: PSingle;
  LI: UInt64;
begin
  LSrcPtr := PUInt16(ASrc);
  LDstPtr := PSingle(ADst);

  for LI := 0 to ACount - 1 do
  begin
    LDstPtr^ := TBFloat16.ToSingle(LSrcPtr^);
    Inc(LSrcPtr);
    Inc(LDstPtr);
  end;
end;

function F32ToQ4Buffer(
  const ASrc: Pointer;
  const ADst: Pointer;
  const ACount: UInt64
): UInt64;
var
  LSrcPtr: PSingle;
  LDstPtr: ^TQ4Block;
  LBlockIdx: UInt64;
  LNumBlocks: UInt64;
  LI: Integer;
  LMax: Single;
  LVal: Single;
  LScale: Single;
  LInvScale: Single;
  LQuantized: Integer;
  LNibble: Byte;
  LFloats: PSingle;
begin
  LNumBlocks := ACount div CQ4BlockSize;
  LSrcPtr := PSingle(ASrc);
  LDstPtr := ADst;

  for LBlockIdx := 0 to LNumBlocks - 1 do
  begin
    LFloats := LSrcPtr;

    // Find the absolute maximum value in this block of 32 floats
    LMax := 0.0;
    for LI := 0 to CQ4BlockSize - 1 do
    begin
      LVal := Abs(PSingle(UIntPtr(LFloats) + UIntPtr(LI * SizeOf(Single)))^);
      if LVal > LMax then
        LMax := LVal;
    end;

    // Compute scale: maps [-max..+max] to [-8..+7]
    // Q4_0 uses asymmetric range: nibble 0..15 maps to -8..+7
    LScale := LMax / 7.0;

    // Store scale as fp16
    LDstPtr^.Scale := TFloat16.FromSingle(LScale);

    // Compute inverse scale for quantization
    if LScale > 0.0 then
      LInvScale := 1.0 / LScale
    else
      LInvScale := 0.0;

    // Quantize 32 floats into 16 bytes of packed nibbles (interleaved layout)
    // Byte j: low nibble stores float[j*2], high nibble stores float[j*2+1]
    for LI := 0 to 15 do
    begin
      // Low nibble (even index)
      LVal := PSingle(UIntPtr(LFloats) + UIntPtr((LI * 2) * SizeOf(Single)))^;
      LQuantized := Round(LVal * LInvScale);
      if LQuantized < -8 then LQuantized := -8;
      if LQuantized > 7 then LQuantized := 7;
      LNibble := Byte(LQuantized + 8);

      // High nibble (odd index)
      LVal := PSingle(UIntPtr(LFloats) + UIntPtr((LI * 2 + 1) * SizeOf(Single)))^;
      LQuantized := Round(LVal * LInvScale);
      if LQuantized < -8 then LQuantized := -8;
      if LQuantized > 7 then LQuantized := 7;
      LNibble := LNibble or (Byte(LQuantized + 8) shl 4);

      LDstPtr^.Nibbles[LI] := LNibble;
    end;

    // Advance pointers
    Inc(LSrcPtr, CQ4BlockSize);
    Inc(LDstPtr);
  end;

  Result := LNumBlocks;
end;

function Q4BlockCount(const AElementCount: UInt64): UInt64;
begin
  Result := (AElementCount + CQ4BlockSize - 1) div CQ4BlockSize;
end;

function Q4ByteSize(const AElementCount: UInt64): UInt64;
begin
  Result := Q4BlockCount(AElementCount) * UInt64(SizeOf(TQ4Block));
end;

procedure DequantQ4Block(
  const ABlock: TQ4Block;
  const ADst: PSingle
);
var
  LScale: Single;
  LI: Integer;
  LByte: Byte;
  LLow: Integer;
  LHigh: Integer;
begin
  // Decode fp16 scale back to f32
  LScale := TFloat16.ToSingle(ABlock.Scale);

  // Unpack 16 bytes into 32 floats (interleaved layout)
  // Byte j: low nibble = float[j*2], high nibble = float[j*2+1]
  for LI := 0 to 15 do
  begin
    LByte := ABlock.Nibbles[LI];

    // Low nibble -> even index
    LLow := Integer(LByte and $0F) - 8;
    PSingle(UIntPtr(ADst) + UIntPtr((LI * 2) * SizeOf(Single)))^ :=
      Single(LLow) * LScale;

    // High nibble -> odd index
    LHigh := Integer(LByte shr 4) - 8;
    PSingle(UIntPtr(ADst) + UIntPtr((LI * 2 + 1) * SizeOf(Single)))^ :=
      Single(LHigh) * LScale;
  end;
end;

procedure Q4ToF32Buffer(
  const ASrc: Pointer;
  const ADst: Pointer;
  const ABlockCount: UInt64
);
var
  LSrcPtr: ^TQ4Block;
  LDstPtr: PSingle;
  LI: UInt64;
begin
  LSrcPtr := ASrc;
  LDstPtr := PSingle(ADst);

  for LI := 0 to ABlockCount - 1 do
  begin
    DequantQ4Block(LSrcPtr^, LDstPtr);
    Inc(LSrcPtr);
    Inc(LDstPtr, CQ4BlockSize);
  end;
end;

end.
