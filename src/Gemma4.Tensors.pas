{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information

 -------------------------------------------------------------------------------

  Gemma4.Tensors - Tensor storage and basic CPU operations

  Provides TTensor for holding f32 data with shape metadata, and
  TWeightStore for loading the manifest + weights from a VPK archive
  and looking up individual weight tensors by name.

  CPU math operations are provided as standalone procedures for the
  correctness reference path (Phase 3). These will be replaced by
  Vulkan compute shaders in Phase 4.

  Key types:
  - TTensor: Owns an f32 buffer with shape/stride metadata.
    Supports create, resize, fill, copy, and element access.
  - TWeightStore: Opens a VPK, parses manifest.json, holds
    memory-mapped weights, provides GetTensor for lookups.

  Dependencies: StdApp.Base, StdApp.JSON, StdApp.VFS,
    StdApp.VirtualMemory, Gemma4.Types, Gemma4.Quant
===============================================================================}

unit Gemma4.Tensors;

{$I StdApp.Defines.inc}

interface

uses
  System.SysUtils,
  System.Math,
  System.Generics.Collections,
  System.Classes,
  StdApp.Base,
  StdApp.JSON,
  StdApp.VFS,
  StdApp.VirtualMemory,
  Gemma4.Types,
  Gemma4.Quant;

const
  CTS_ERR_SHAPE = 'TS01';
  CTS_ERR_VPK = 'TS02';
  CTS_ERR_MANIFEST = 'TS03';
  CTS_ERR_TENSOR = 'TS04';

type
  { TTensor }
  // Owns a contiguous f32 buffer with shape metadata.
  // All inference math operates on TTensor instances.
  TTensor = class(TBaseObject)
  private
    FData: TArray<Single>;
    FShape: TArray<Integer>;
    FSize: Integer; // total element count

    function GetItem(AIndex: Integer): Single;
    procedure SetItem(AIndex: Integer; const AValue: Single);
  public
    constructor Create(); override;
    destructor Destroy(); override;

    // Allocate with shape
    procedure Reshape(const AShape: TArray<Integer>);

    // Create from existing data (copies)
    procedure LoadFromPointer(const ASrc: Pointer;
      const ACount: Integer; const AShape: TArray<Integer>);

    // Create from Q4_0 data (dequantizes to f32)
    procedure LoadFromQ4(const ASrc: Pointer;
      const ABlockCount: Integer; const AShape: TArray<Integer>);

    // Basic operations
    procedure Zero();
    procedure Fill(const AValue: Single);
    procedure CopyFrom(const AOther: TTensor);
    procedure AddInPlace(const AOther: TTensor);
    procedure ScaleInPlace(const AFactor: Single);

    // Access
    function DataPtr(): PSingle;
    function DimCount(): Integer;
    function Dim(const AIndex: Integer): Integer;

    property Item[AIndex: Integer]: Single read GetItem write SetItem; default;
    property Shape: TArray<Integer> read FShape;
    property Size: Integer read FSize;
    property Data: TArray<Single> read FData;
  end;

  { TWeightStore }
  // Opens a VPK archive, parses the manifest, and provides
  // tensor lookup by name. Weight data stays memory-mapped.
  TWeightStore = class(TBaseObject)
  private
    FVFS: TVFS;
    FWeightsView: TVirtualMemoryView<Byte>;
    FManifest: TList<TTensorInfo>;
    FLookup: TDictionary<string, Integer>;
    FIsOpen: Boolean;

    function DoParseManifest(const AJsonStr: string): Boolean;
  public
    constructor Create(); override;
    destructor Destroy(); override;

    // Open VPK and load manifest (default E4B model paths)
    function Open(const AVpkPath: string): Boolean; overload;
    // Open VPK with explicit per-model manifest/weights paths inside the
    // archive (e.g. 'embeddings/manifest.json', 'embeddings/weights.bin')
    function Open(const AVpkPath: string; const AManifestPath: string;
      const AWeightsPath: string): Boolean; overload;
    procedure Close();
    function IsOpen(): Boolean;

    // Look up a tensor by name
    function FindTensor(const AName: string;
      out AInfo: TTensorInfo): Boolean;

    // Load a tensor from the weight store into an f32 TTensor.
    // Handles dequantization for Q4_0 tensors automatically.
    function LoadTensor(const AName: string): TTensor;

    // Get raw pointer into weight data for a tensor
    function GetRawPointer(const AInfo: TTensorInfo): Pointer;

    // Enumeration
    function TensorCount(): Integer;
    function GetTensorInfo(const AIndex: Integer): TTensorInfo;
  end;

// -----------------------------------------------------------------------
// CPU math operations (correctness reference)
// -----------------------------------------------------------------------

// Matrix-vector multiply: ADst = AMatrix * AVec
// AMatrix is [ARows x ACols], AVec is [ACols], ADst is [ARows]
procedure CpuMatVecF32(
  const AMatrix: PSingle;
  const AVec: PSingle;
  const ADst: PSingle;
  const ARows: Integer;
  const ACols: Integer
);

// Matrix-vector multiply with Q4_0 matrix (dequantizes on the fly)
// AMatrix is Q4_0 blocks for [ARows x ACols], AVec is f32 [ACols], ADst is f32 [ARows]
procedure CpuMatVecQ4(
  const AMatrix: Pointer;
  const AVec: PSingle;
  const ADst: PSingle;
  const ARows: Integer;
  const ACols: Integer
);

// Element-wise add: ADst[i] += ASrc[i]
procedure CpuAddF32(
  const ASrc: PSingle;
  const ADst: PSingle;
  const ACount: Integer
);

// Element-wise multiply: ADst[i] *= ASrc[i]
procedure CpuMulF32(
  const ASrc: PSingle;
  const ADst: PSingle;
  const ACount: Integer
);

// Scale: ADst[i] *= AScale
procedure CpuScaleF32(
  const ADst: PSingle;
  const AScale: Single;
  const ACount: Integer
);

// Softmax in-place over ACount elements
procedure CpuSoftmaxF32(
  const AData: PSingle;
  const ACount: Integer
);

implementation

uses
  System.IOUtils;

{ TTensor }

constructor TTensor.Create();
begin
  inherited Create();
  FSize := 0;
end;

destructor TTensor.Destroy();
begin
  inherited;
end;

function TTensor.GetItem(AIndex: Integer): Single;
begin
  Result := FData[AIndex];
end;

procedure TTensor.SetItem(AIndex: Integer; const AValue: Single);
begin
  FData[AIndex] := AValue;
end;

procedure TTensor.Reshape(const AShape: TArray<Integer>);
var
  LI: Integer;
  LTotal: Integer;
begin
  LTotal := 1;
  for LI := 0 to High(AShape) do
    LTotal := LTotal * AShape[LI];

  FShape := Copy(AShape);
  FSize := LTotal;
  SetLength(FData, LTotal);
end;

procedure TTensor.LoadFromPointer(const ASrc: Pointer;
  const ACount: Integer; const AShape: TArray<Integer>);
begin
  Reshape(AShape);
  if ACount > FSize then
    raise ERangeError.Create('Source count exceeds tensor size');
  Move(ASrc^, FData[0], ACount * SizeOf(Single));
end;

procedure TTensor.LoadFromQ4(const ASrc: Pointer;
  const ABlockCount: Integer; const AShape: TArray<Integer>);
begin
  Reshape(AShape);
  Q4ToF32Buffer(ASrc, @FData[0], ABlockCount);
end;

procedure TTensor.Zero();
begin
  if FSize > 0 then
    FillChar(FData[0], FSize * SizeOf(Single), 0);
end;

procedure TTensor.Fill(const AValue: Single);
var
  LI: Integer;
begin
  for LI := 0 to FSize - 1 do
    FData[LI] := AValue;
end;

procedure TTensor.CopyFrom(const AOther: TTensor);
begin
  FShape := Copy(AOther.FShape);
  FSize := AOther.FSize;
  FData := Copy(AOther.FData);
end;

procedure TTensor.AddInPlace(const AOther: TTensor);
begin
  CpuAddF32(@AOther.FData[0], @FData[0], Min(FSize, AOther.FSize));
end;

procedure TTensor.ScaleInPlace(const AFactor: Single);
begin
  CpuScaleF32(@FData[0], AFactor, FSize);
end;

function TTensor.DataPtr(): PSingle;
begin
  if FSize > 0 then
    Result := @FData[0]
  else
    Result := nil;
end;

function TTensor.DimCount(): Integer;
begin
  Result := Length(FShape);
end;

function TTensor.Dim(const AIndex: Integer): Integer;
begin
  Result := FShape[AIndex];
end;

{ TWeightStore }

constructor TWeightStore.Create();
begin
  inherited Create();
  FVFS := TVFS.Create();
  FManifest := TList<TTensorInfo>.Create();
  FLookup := TDictionary<string, Integer>.Create();
  FWeightsView := nil;
  FIsOpen := False;
end;

destructor TWeightStore.Destroy();
begin
  Close();
  FLookup.Free();
  FManifest.Free();
  FWeightsView.Free();
  FVFS.Free();
  inherited;
end;

function TWeightStore.Open(const AVpkPath: string): Boolean;
begin
  Result := Open(AVpkPath, CVpkManifestPath, CVpkWeightsPath);
end;

function TWeightStore.Open(const AVpkPath: string;
  const AManifestPath: string; const AWeightsPath: string): Boolean;
var
  LManifestView: TVirtualMemoryView<Byte>;
  LManifestBytes: TBytes;
  LManifestStr: string;
begin
  Result := False;
  Close();

  FVFS.SetErrors(GetErrors());
  if not FVFS.Open(AVpkPath) then
  begin
    GetErrors().Add(esError, CTS_ERR_VPK,
      'Failed to open VPK: %s', [AVpkPath]);
    Exit;
  end;

  // Read manifest.json
  if not FVFS.FileExists(AManifestPath) then
  begin
    GetErrors().Add(esError, CTS_ERR_MANIFEST,
      'Manifest not found in VPK: %s', [AManifestPath]);
    FVFS.Close();
    Exit;
  end;

  LManifestView := FVFS.OpenFile(AManifestPath);
  try
    SetLength(LManifestBytes, LManifestView.Size);
    LManifestView.Read(LManifestBytes[0], LManifestView.Size);
    LManifestStr := TEncoding.UTF8.GetString(LManifestBytes);
  finally
    LManifestView.Free();
  end;

  if not DoParseManifest(LManifestStr) then
  begin
    FVFS.Close();
    Exit;
  end;

  // Open weights view
  if not FVFS.FileExists(AWeightsPath) then
  begin
    GetErrors().Add(esError, CTS_ERR_VPK,
      'Weights file not found in VPK: %s', [AWeightsPath]);
    FVFS.Close();
    Exit;
  end;

  FWeightsView := FVFS.OpenFile(AWeightsPath);
  FIsOpen := True;
  Result := True;
end;

procedure TWeightStore.Close();
begin
  FreeAndNil(FWeightsView);
  FLookup.Clear();
  FManifest.Clear();
  FVFS.Close();
  FIsOpen := False;
end;

function TWeightStore.IsOpen(): Boolean;
begin
  Result := FIsOpen;
end;

function TWeightStore.DoParseManifest(const AJsonStr: string): Boolean;
var
  LJSON: TJSON;
  LTensorsArr: TJSON;
  LItems: TArray<TJSON>;
  LI: Integer;
  LItem: TJSON;
  LInfo: TTensorInfo;
  LShapeArr: TJSON;
  LShapeItems: TArray<TJSON>;
  LJ: Integer;
begin
  Result := False;

  LJSON := TJSON.FromString(AJsonStr);
  try
    if LJSON.IsNull() then
    begin
      GetErrors().Add(esError, CTS_ERR_MANIFEST,
        'Failed to parse manifest JSON');
      Exit;
    end;

    LTensorsArr := LJSON.Get('tensors');
    if LTensorsArr.IsNull() then
    begin
      GetErrors().Add(esError, CTS_ERR_MANIFEST,
        'Missing tensors array in manifest');
      Exit;
    end;

    LItems := LTensorsArr.Items();
    for LI := 0 to High(LItems) do
    begin
      LItem := LItems[LI];
      LInfo := Default(TTensorInfo);
      LInfo.TensorName := LItem.Get('name').AsString('');
      LInfo.SourceDtype := DtypeKindFromString(LItem.Get('source_dtype').AsString(''));
      LInfo.OutputDtype := DtypeKindFromString(LItem.Get('output_dtype').AsString(''));
      LInfo.Offset := LItem.Get('offset').AsUInt64(0);
      LInfo.DataSize := LItem.Get('size').AsUInt64(0);

      LShapeArr := LItem.Get('shape');
      if (not LShapeArr.IsNull()) and (LShapeArr.Count() > 0) then
      begin
        LShapeItems := LShapeArr.Items();
        SetLength(LInfo.Shape, Length(LShapeItems));
        for LJ := 0 to High(LShapeItems) do
          LInfo.Shape[LJ] := LShapeItems[LJ].AsInt32(0);
      end;

      FLookup.AddOrSetValue(LInfo.TensorName, FManifest.Count);
      FManifest.Add(LInfo);
    end;

    Result := FManifest.Count > 0;
  finally
    LJSON.Free();
  end;
end;

function TWeightStore.FindTensor(const AName: string;
  out AInfo: TTensorInfo): Boolean;
var
  LIndex: Integer;
begin
  Result := FLookup.TryGetValue(AName, LIndex);
  if Result then
    AInfo := FManifest[LIndex]
  else
    AInfo := Default(TTensorInfo);
end;

function TWeightStore.GetRawPointer(const AInfo: TTensorInfo): Pointer;
begin
  if (FWeightsView = nil) or (not FIsOpen) then
    raise EInvalidOperation.Create('Weight store is not open');

  Result := Pointer(UIntPtr(FWeightsView.Memory) + UIntPtr(AInfo.Offset));
end;

function TWeightStore.LoadTensor(const AName: string): TTensor;
var
  LInfo: TTensorInfo;
  LPtr: Pointer;
  LBlockCount: UInt64;
begin
  Result := nil;

  if not FindTensor(AName, LInfo) then
  begin
    GetErrors().Add(esError, CTS_ERR_TENSOR,
      'Tensor not found: %s', [AName]);
    Exit;
  end;

  LPtr := GetRawPointer(LInfo);
  Result := TTensor.Create();

  if LInfo.OutputDtype = dkQ4_0 then
  begin
    // Dequantize Q4_0 to f32
    LBlockCount := LInfo.DataSize div UInt64(SizeOf(TQ4Block));
    Result.LoadFromQ4(LPtr, LBlockCount, LInfo.Shape);
  end
  else
  begin
    // F32 -- direct copy
    Result.LoadFromPointer(LPtr, LInfo.ElementCount(), LInfo.Shape);
  end;
end;

function TWeightStore.TensorCount(): Integer;
begin
  Result := FManifest.Count;
end;

function TWeightStore.GetTensorInfo(const AIndex: Integer): TTensorInfo;
begin
  Result := FManifest[AIndex];
end;

{ CPU math operations }

procedure CpuMatVecF32(
  const AMatrix: PSingle;
  const AVec: PSingle;
  const ADst: PSingle;
  const ARows: Integer;
  const ACols: Integer);
var
  LRow: Integer;
  LCol: Integer;
  LSum: Single;
  LRowPtr: PSingle;
begin
  for LRow := 0 to ARows - 1 do
  begin
    LSum := 0.0;
    LRowPtr := PSingle(UIntPtr(AMatrix) + UIntPtr(LRow * ACols * SizeOf(Single)));
    for LCol := 0 to ACols - 1 do
    begin
      LSum := LSum +
        PSingle(UIntPtr(LRowPtr) + UIntPtr(LCol * SizeOf(Single)))^ *
        PSingle(UIntPtr(AVec) + UIntPtr(LCol * SizeOf(Single)))^;
    end;
    PSingle(UIntPtr(ADst) + UIntPtr(LRow * SizeOf(Single)))^ := LSum;
  end;
end;

procedure CpuMatVecQ4(
  const AMatrix: Pointer;
  const AVec: PSingle;
  const ADst: PSingle;
  const ARows: Integer;
  const ACols: Integer);
var
  LRow: Integer;
  LBlocksPerRow: Integer;
  LBlockIdx: Integer;
  LRowStart: Pointer;
  LBlock: ^TQ4Block;
  LScale: Single;
  LByte: Byte;
  LLow: Integer;
  LHigh: Integer;
  LBI: Integer;
  LVecIdx: Integer;
  LSum: Single;
begin
  LBlocksPerRow := ACols div CQ4BlockSize;

  for LRow := 0 to ARows - 1 do
  begin
    LSum := 0.0;
    LRowStart := Pointer(UIntPtr(AMatrix) +
      UIntPtr(LRow * LBlocksPerRow * SizeOf(TQ4Block)));

    for LBlockIdx := 0 to LBlocksPerRow - 1 do
    begin
      LBlock := Pointer(UIntPtr(LRowStart) +
        UIntPtr(LBlockIdx * SizeOf(TQ4Block)));
      LScale := TFloat16.ToSingle(LBlock^.Scale);
      LVecIdx := LBlockIdx * CQ4BlockSize;

      for LBI := 0 to 15 do
      begin
        LByte := LBlock^.Nibbles[LBI];

        // Interleaved layout: low nibble = vec[j*2], high nibble = vec[j*2+1]
        LLow := Integer(LByte and $0F) - 8;
        LSum := LSum + (Single(LLow) * LScale) *
          PSingle(UIntPtr(AVec) + UIntPtr((LVecIdx + LBI * 2) * SizeOf(Single)))^;

        LHigh := Integer(LByte shr 4) - 8;
        LSum := LSum + (Single(LHigh) * LScale) *
          PSingle(UIntPtr(AVec) + UIntPtr((LVecIdx + LBI * 2 + 1) * SizeOf(Single)))^;
      end;
    end;

    PSingle(UIntPtr(ADst) + UIntPtr(LRow * SizeOf(Single)))^ := LSum;
  end;
end;

procedure CpuAddF32(
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

procedure CpuMulF32(
  const ASrc: PSingle;
  const ADst: PSingle;
  const ACount: Integer);
var
  LI: Integer;
begin
  for LI := 0 to ACount - 1 do
    PSingle(UIntPtr(ADst) + UIntPtr(LI * SizeOf(Single)))^ :=
      PSingle(UIntPtr(ADst) + UIntPtr(LI * SizeOf(Single)))^ *
      PSingle(UIntPtr(ASrc) + UIntPtr(LI * SizeOf(Single)))^;
end;

procedure CpuScaleF32(
  const ADst: PSingle;
  const AScale: Single;
  const ACount: Integer);
var
  LI: Integer;
begin
  for LI := 0 to ACount - 1 do
    PSingle(UIntPtr(ADst) + UIntPtr(LI * SizeOf(Single)))^ :=
      PSingle(UIntPtr(ADst) + UIntPtr(LI * SizeOf(Single)))^ * AScale;
end;

procedure CpuSoftmaxF32(
  const AData: PSingle;
  const ACount: Integer);
var
  LI: Integer;
  LMax: Single;
  LSum: Single;
  LVal: Single;
begin
  // Find max for numerical stability
  LMax := PSingle(AData)^;
  for LI := 1 to ACount - 1 do
  begin
    LVal := PSingle(UIntPtr(AData) + UIntPtr(LI * SizeOf(Single)))^;
    if LVal > LMax then
      LMax := LVal;
  end;

  // Exponentiate and sum
  LSum := 0.0;
  for LI := 0 to ACount - 1 do
  begin
    LVal := Exp(PSingle(UIntPtr(AData) + UIntPtr(LI * SizeOf(Single)))^ - LMax);
    PSingle(UIntPtr(AData) + UIntPtr(LI * SizeOf(Single)))^ := LVal;
    LSum := LSum + LVal;
  end;

  // Normalize
  if LSum > 0.0 then
  begin
    LVal := 1.0 / LSum;
    for LI := 0 to ACount - 1 do
      PSingle(UIntPtr(AData) + UIntPtr(LI * SizeOf(Single)))^ :=
        PSingle(UIntPtr(AData) + UIntPtr(LI * SizeOf(Single)))^ * LVal;
  end;
end;

end.
