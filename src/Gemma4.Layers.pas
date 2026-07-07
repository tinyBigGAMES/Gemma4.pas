{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information

 -------------------------------------------------------------------------------

  Gemma4.Layers - Neural network layer operations (CPU reference)

  Implements the building-block operations used in each Gemma 4
  decoder layer. These are CPU-only reference implementations for
  correctness testing. The Vulkan compute path (Phase 4) will
  replace these with GPU kernels.

  Operations:
  - RMSNorm: Root mean square layer normalization
  - GeLU_pytorch_tanh: Gaussian error linear unit (tanh approximation)
  - SwiGLU MLP: Gated linear unit with SiLU activation (gate + up + down)
  - Residual add: x = x + sublayer(x)
  - Logit soft-capping: tanh(x / cap) * cap

  Dependencies: Gemma4.Types, Gemma4.Tensors
===============================================================================}

unit Gemma4.Layers;

{$I StdApp.Defines.inc}

interface

uses
  System.SysUtils,
  System.Math,
  Gemma4.Types,
  Gemma4.Tensors;

// -----------------------------------------------------------------------
// RMSNorm
// -----------------------------------------------------------------------

// RMSNorm: out[i] = (x[i] / rms) * weight[i]
// where rms = sqrt(mean(x^2) + eps)
// AInput and AOutput can be the same buffer (in-place).
// AWeight is the learned scale vector.
procedure CpuRmsNorm(
  const AInput: PSingle;
  const AWeight: PSingle;
  const AOutput: PSingle;
  const ASize: Integer;
  const AEps: Single
);

// -----------------------------------------------------------------------
// Activation functions
// -----------------------------------------------------------------------

// GeLU with tanh approximation (pytorch variant):
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// In-place: AData is modified.
procedure CpuGeluPytorchTanh(
  const AData: PSingle;
  const ACount: Integer
);

// RMSNorm without learned weight (weight-free):
// out[i] = x[i] / rms(x)
// where rms = sqrt(mean(x^2) + eps)
// Used for V norm in Gemma 4 attention (no gamma multiply).
// AInput and AOutput can be the same buffer (in-place).
procedure CpuRmsNormNoWeight(
  const AInput: PSingle;
  const AOutput: PSingle;
  const ASize: Integer;
  const AEps: Single
);

// SiLU (Sigmoid Linear Unit) aka Swish:
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// In-place: AData is modified.
procedure CpuSiLU(
  const AData: PSingle;
  const ACount: Integer
);

// -----------------------------------------------------------------------
// MLP (SwiGLU variant used by Gemma 4)
// -----------------------------------------------------------------------

// Gemma 4 MLP structure:
//   gate = silu(x @ gate_proj)
//   up   = x @ up_proj
//   out  = (gate * up) @ down_proj
//
// AGateProj, AUpProj: [intermediate_size x hidden_size] weight matrices
// ADownProj: [hidden_size x intermediate_size] weight matrix
// AInput: [hidden_size] input vector
// AOutput: [hidden_size] output vector (pre-allocated)
// AHiddenSize, AIntermediateSize: dimensions
procedure CpuMlpSwiGLU(
  const AInput: PSingle;
  const AGateProj: Pointer;  // Q4_0 or f32 weight matrix
  const AUpProj: Pointer;    // Q4_0 or f32 weight matrix
  const ADownProj: Pointer;  // Q4_0 or f32 weight matrix
  const AOutput: PSingle;
  const AHiddenSize: Integer;
  const AIntermediateSize: Integer;
  const AIsQ4: Boolean
);

// -----------------------------------------------------------------------
// Residual connection
// -----------------------------------------------------------------------

// ADst[i] += ASrc[i]
procedure CpuResidualAdd(
  const ASrc: PSingle;
  const ADst: PSingle;
  const ACount: Integer
);

// -----------------------------------------------------------------------
// Logit soft-capping
// -----------------------------------------------------------------------

// out[i] = tanh(x[i] / cap) * cap
// Used on attention logits before softmax.
procedure CpuLogitSoftcap(
  const AData: PSingle;
  const ACount: Integer;
  const ACap: Single
);

implementation

const
  CSqrt2OverPi: Single = 0.7978845608; // sqrt(2 / pi)
  CGeluCoeff: Single = 0.044715;

procedure CpuRmsNorm(
  const AInput: PSingle;
  const AWeight: PSingle;
  const AOutput: PSingle;
  const ASize: Integer;
  const AEps: Single);
var
  LI: Integer;
  LSumSq: Single;
  LRms: Single;
  LScale: Single;
  LVal: Single;
begin
  // Compute sum of squares
  LSumSq := 0.0;
  for LI := 0 to ASize - 1 do
  begin
    LVal := PSingle(UIntPtr(AInput) + UIntPtr(LI * SizeOf(Single)))^;
    LSumSq := LSumSq + LVal * LVal;
  end;

  // RMS = sqrt(mean(x^2) + eps)
  LRms := Sqrt(LSumSq / ASize + AEps);
  LScale := 1.0 / LRms;

  // Normalize and scale by learned weight
  for LI := 0 to ASize - 1 do
  begin
    LVal := PSingle(UIntPtr(AInput) + UIntPtr(LI * SizeOf(Single)))^;
    PSingle(UIntPtr(AOutput) + UIntPtr(LI * SizeOf(Single)))^ :=
      LVal * LScale *
      PSingle(UIntPtr(AWeight) + UIntPtr(LI * SizeOf(Single)))^;
  end;
end;

procedure CpuRmsNormNoWeight(
  const AInput: PSingle;
  const AOutput: PSingle;
  const ASize: Integer;
  const AEps: Single);
var
  LI: Integer;
  LSumSq: Single;
  LRms: Single;
  LScale: Single;
  LVal: Single;
begin
  // Compute sum of squares
  LSumSq := 0.0;
  for LI := 0 to ASize - 1 do
  begin
    LVal := PSingle(UIntPtr(AInput) + UIntPtr(LI * SizeOf(Single)))^;
    LSumSq := LSumSq + LVal * LVal;
  end;

  // RMS = sqrt(mean(x^2) + eps)
  LRms := Sqrt(LSumSq / ASize + AEps);
  LScale := 1.0 / LRms;

  // Normalize without weight multiply
  for LI := 0 to ASize - 1 do
  begin
    LVal := PSingle(UIntPtr(AInput) + UIntPtr(LI * SizeOf(Single)))^;
    PSingle(UIntPtr(AOutput) + UIntPtr(LI * SizeOf(Single)))^ := LVal * LScale;
  end;
end;

procedure CpuGeluPytorchTanh(
  const AData: PSingle;
  const ACount: Integer);
var
  LI: Integer;
  LX: Single;
  LInner: Single;
begin
  for LI := 0 to ACount - 1 do
  begin
    LX := PSingle(UIntPtr(AData) + UIntPtr(LI * SizeOf(Single)))^;
    // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    LInner := CSqrt2OverPi * (LX + CGeluCoeff * LX * LX * LX);
    PSingle(UIntPtr(AData) + UIntPtr(LI * SizeOf(Single)))^ :=
      0.5 * LX * (1.0 + Tanh(LInner));
  end;
end;

procedure CpuSiLU(
  const AData: PSingle;
  const ACount: Integer);
var
  LI: Integer;
  LX: Single;
begin
  for LI := 0 to ACount - 1 do
  begin
    LX := PSingle(UIntPtr(AData) + UIntPtr(LI * SizeOf(Single)))^;
    // silu(x) = x / (1 + exp(-x))
    PSingle(UIntPtr(AData) + UIntPtr(LI * SizeOf(Single)))^ :=
      LX / (1.0 + Exp(-LX));
  end;
end;

procedure CpuMlpSwiGLU(
  const AInput: PSingle;
  const AGateProj: Pointer;
  const AUpProj: Pointer;
  const ADownProj: Pointer;
  const AOutput: PSingle;
  const AHiddenSize: Integer;
  const AIntermediateSize: Integer;
  const AIsQ4: Boolean);
var
  LGate: TArray<Single>;
  LUp: TArray<Single>;
  LI: Integer;
begin
  SetLength(LGate, AIntermediateSize);
  SetLength(LUp, AIntermediateSize);

  // gate = input @ gate_proj^T  -> [intermediate_size]
  // up   = input @ up_proj^T    -> [intermediate_size]
  if AIsQ4 then
  begin
    CpuMatVecQ4(AGateProj, AInput, @LGate[0], AIntermediateSize, AHiddenSize);
    CpuMatVecQ4(AUpProj, AInput, @LUp[0], AIntermediateSize, AHiddenSize);
  end
  else
  begin
    CpuMatVecF32(AGateProj, AInput, @LGate[0], AIntermediateSize, AHiddenSize);
    CpuMatVecF32(AUpProj, AInput, @LUp[0], AIntermediateSize, AHiddenSize);
  end;

  // gate = gelu(gate) -- Gemma 4 uses GeGLU, not SwiGLU
  CpuGeluPytorchTanh(@LGate[0], AIntermediateSize);

  // hidden = gate * up (element-wise)
  for LI := 0 to AIntermediateSize - 1 do
    LGate[LI] := LGate[LI] * LUp[LI];

  // output = hidden @ down_proj^T  -> [hidden_size]
  if AIsQ4 then
    CpuMatVecQ4(ADownProj, @LGate[0], AOutput, AHiddenSize, AIntermediateSize)
  else
    CpuMatVecF32(ADownProj, @LGate[0], AOutput, AHiddenSize, AIntermediateSize);
end;

procedure CpuResidualAdd(
  const ASrc: PSingle;
  const ADst: PSingle;
  const ACount: Integer);
var
  LI: Integer;
begin
  for LI := 0 to ACount - 1 do
    PSingle(UIntPtr(ADst) + UIntPtr(LI * SizeOf(Single)))^ :=
      PSingle(UIntPtr(ADst) + UIntPtr(LI * SizeOf(Single)))^ +
      PSingle(UIntPtr(ASrc) + UIntPtr(LI * SizeOf(Single)))^;
end;

procedure CpuLogitSoftcap(
  const AData: PSingle;
  const ACount: Integer;
  const ACap: Single);
var
  LI: Integer;
  LVal: Single;
  LInvCap: Single;
begin
  LInvCap := 1.0 / ACap;
  for LI := 0 to ACount - 1 do
  begin
    LVal := PSingle(UIntPtr(AData) + UIntPtr(LI * SizeOf(Single)))^;
    PSingle(UIntPtr(AData) + UIntPtr(LI * SizeOf(Single)))^ :=
      Tanh(LVal * LInvCap) * ACap;
  end;
end;

end.
