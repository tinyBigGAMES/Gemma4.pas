{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information

 -------------------------------------------------------------------------------

  Gemma4.Safetensors - Safetensors file parser

  Parses the safetensors binary format: 8-byte LE header length,
  followed by a JSON header mapping tensor names to their dtype,
  shape, and byte offsets within the data section. The data section
  starts at byte offset (8 + header_length).

  Uses TVirtualMemory<Byte> to memory-map the file (14.8GB) and
  TJSON to parse the header. Produces a list of TTensorInfo records
  describing every tensor in the file.

  Key types:
  - TSafetensors: Memory-maps a .safetensors file, parses the header,
    and provides access to tensor metadata and raw data pointers.

  Dependencies: StdApp.Base, StdApp.JSON, StdApp.VirtualMemory,
    Gemma4.Types
===============================================================================}

unit Gemma4.Safetensors;

{$I StdApp.Defines.inc}

interface

uses
  System.SysUtils,
  System.Generics.Collections,
  System.Classes,
  StdApp.Base,
  StdApp.JSON,
  StdApp.VirtualMemory,
  Gemma4.Types;

const
  // Error codes
  CST_ERR_OPEN = 'ST01';
  CST_ERR_TOO_SMALL = 'ST02';
  CST_ERR_HEADER_SIZE = 'ST03';
  CST_ERR_HEADER_PARSE = 'ST04';
  CST_ERR_NOT_OPEN = 'ST05';

type
  { TSafetensors }
  // Parses a .safetensors file and provides tensor metadata + data access.
  // Memory-maps the file for zero-copy access to the raw tensor data.
  TSafetensors = class(TBaseObject)
  private
    FFile: TVirtualMemory<Byte>;
    FHeaderLength: UInt64;
    FDataOffset: UInt64;
    FTensors: TList<TTensorInfo>;
    FLookup: TDictionary<string, Integer>;
    FIsOpen: Boolean;

    function DoParseHeader(): Boolean;
    function DoParseDataOffsets(
      const AName: string;
      const ATensorObj: TJSON
    ): TTensorInfo;
    function SafetensorsDtypeToKind(const AValue: string): TDtypeKind;
  public
    constructor Create(); override;
    destructor Destroy(); override;

    // Open and parse a .safetensors file
    function Open(const AFilename: string): Boolean;
    procedure Close();
    function IsOpen(): Boolean;

    // Tensor enumeration
    function TensorCount(): Integer;
    function GetTensor(const AIndex: Integer): TTensorInfo;
    function FindTensor(const AName: string; out AInfo: TTensorInfo): Boolean;
    function GetTensors(): TList<TTensorInfo>;

    // Raw data access -- returns pointer into memory-mapped file
    function GetDataPointer(const AInfo: TTensorInfo): Pointer;

    // File-level info
    function FileSize(): UInt64;
    function DataOffset(): UInt64;
  end;

implementation

{ TSafetensors }

constructor TSafetensors.Create();
begin
  inherited Create();
  FFile := TVirtualMemory<Byte>.Create();
  FTensors := TList<TTensorInfo>.Create();
  FLookup := TDictionary<string, Integer>.Create();
  FIsOpen := False;
  FHeaderLength := 0;
  FDataOffset := 0;
end;

destructor TSafetensors.Destroy();
begin
  Close();
  FLookup.Free();
  FTensors.Free();
  FFile.Free();
  inherited;
end;

function TSafetensors.Open(const AFilename: string): Boolean;
begin
  Result := False;
  Close();

  // Memory-map the file read-only
  if not FFile.Open(AFilename, TVirtualMemoryMode.ReadOnly) then
  begin
    GetErrors().Add(esError, CST_ERR_OPEN,
      'Failed to open safetensors file: %s', [AFilename]);
    Exit;
  end;

  // Minimum size: 8 bytes for header length
  if FFile.Size < 8 then
  begin
    GetErrors().Add(esError, CST_ERR_TOO_SMALL,
      'Safetensors file too small: %d bytes', [FFile.Size]);
    FFile.Close();
    Exit;
  end;

  // Read 8-byte LE header length
  FHeaderLength := PUInt64(FFile.Memory)^;

  // Validate header length fits within file
  if (8 + FHeaderLength) > FFile.Size then
  begin
    GetErrors().Add(esError, CST_ERR_HEADER_SIZE,
      'Header length %d exceeds file size %d',
      [FHeaderLength, FFile.Size]);
    FFile.Close();
    Exit;
  end;

  // Data starts immediately after the header
  FDataOffset := 8 + FHeaderLength;

  // Parse the JSON header
  if not DoParseHeader() then
  begin
    FFile.Close();
    Exit;
  end;

  FIsOpen := True;
  Result := True;
end;

procedure TSafetensors.Close();
begin
  FLookup.Clear();
  FTensors.Clear();
  FFile.Close();
  FIsOpen := False;
  FHeaderLength := 0;
  FDataOffset := 0;
end;

function TSafetensors.IsOpen(): Boolean;
begin
  Result := FIsOpen;
end;

function TSafetensors.DoParseHeader(): Boolean;
var
  LHeaderStr: string;
  LHeaderBytes: TBytes;
  LJSON: TJSON;
  LPairs: TArray<TJSONPair>;
  LI: Integer;
  LInfo: TTensorInfo;
begin
  Result := False;

  // Extract header JSON string from mapped memory
  SetLength(LHeaderBytes, FHeaderLength);
  Move(
    Pointer(UIntPtr(FFile.Memory) + 8)^,
    LHeaderBytes[0],
    FHeaderLength
  );
  LHeaderStr := TEncoding.UTF8.GetString(LHeaderBytes);

  // Parse JSON
  LJSON := TJSON.FromString(LHeaderStr);
  try
    if LJSON.IsNull() then
    begin
      GetErrors().Add(esError, CST_ERR_HEADER_PARSE,
        'Failed to parse safetensors header JSON');
      Exit;
    end;

    // Use Pairs() to iterate -- tensor names contain dots which would
    // be misinterpreted as path separators by Get()
    LPairs := LJSON.Pairs();
    for LI := 0 to High(LPairs) do
    begin
      // Skip metadata entry
      if LPairs[LI].NodeName = '__metadata__' then
        Continue;

      if LPairs[LI].Value.IsNull() then
        Continue;

      LInfo := DoParseDataOffsets(LPairs[LI].NodeName, LPairs[LI].Value);
      FLookup.AddOrSetValue(LInfo.TensorName, FTensors.Count);
      FTensors.Add(LInfo);
    end;

    Result := True;
  finally
    LJSON.Free();
  end;
end;

function TSafetensors.DoParseDataOffsets(
  const AName: string;
  const ATensorObj: TJSON
): TTensorInfo;
var
  LDtypeStr: string;
  LOffsetsArr: TJSON;
  LShapeArr: TJSON;
  LBeginOffset: UInt64;
  LEndOffset: UInt64;
  LShapeItems: TArray<TJSON>;
  LI: Integer;
begin
  Result := Default(TTensorInfo);
  Result.TensorName := AName;

  // Parse dtype string (safetensors uses uppercase like "BF16", "F32", "F16")
  LDtypeStr := ATensorObj.Get('dtype').AsString('');
  Result.SourceDtype := SafetensorsDtypeToKind(LDtypeStr);
  // Output dtype will be determined by the packer (Q4_0 for weights, F32 for norms)
  Result.OutputDtype := Result.SourceDtype;

  // Parse data_offsets: [begin, end] -- byte offsets relative to data section start
  LOffsetsArr := ATensorObj.Get('data_offsets');
  if (not LOffsetsArr.IsNull()) and (LOffsetsArr.Count() >= 2) then
  begin
    LBeginOffset := LOffsetsArr.Get('[0]').AsUInt64(0);
    LEndOffset := LOffsetsArr.Get('[1]').AsUInt64(0);
    // Store absolute offset from file start
    Result.Offset := FDataOffset + LBeginOffset;
    Result.DataSize := LEndOffset - LBeginOffset;
  end;

  // Parse shape: array of integers
  LShapeArr := ATensorObj.Get('shape');
  if (not LShapeArr.IsNull()) and (LShapeArr.Count() > 0) then
  begin
    LShapeItems := LShapeArr.Items();
    SetLength(Result.Shape, Length(LShapeItems));
    for LI := 0 to High(LShapeItems) do
      Result.Shape[LI] := LShapeItems[LI].AsInt32(0);
  end
  else
  begin
    SetLength(Result.Shape, 0); // scalar
  end;
end;

function TSafetensors.SafetensorsDtypeToKind(const AValue: string): TDtypeKind;
begin
  // Safetensors uses uppercase dtype strings
  if AValue = 'BF16' then
    Result := dkBF16
  else if AValue = 'F16' then
    Result := dkF16
  else if AValue = 'F32' then
    Result := dkF32
  else if AValue = 'I32' then
    Result := dkF32 // treat int32 as f32 for our purposes
  else if AValue = 'I64' then
    Result := dkF32
  else
    Result := dkUnknown;
end;

function TSafetensors.TensorCount(): Integer;
begin
  Result := FTensors.Count;
end;

function TSafetensors.GetTensor(const AIndex: Integer): TTensorInfo;
begin
  Result := FTensors[AIndex];
end;

function TSafetensors.FindTensor(const AName: string;
  out AInfo: TTensorInfo): Boolean;
var
  LIndex: Integer;
begin
  Result := FLookup.TryGetValue(AName, LIndex);
  if Result then
    AInfo := FTensors[LIndex]
  else
    AInfo := Default(TTensorInfo);
end;

function TSafetensors.GetTensors(): TList<TTensorInfo>;
begin
  Result := FTensors;
end;

function TSafetensors.GetDataPointer(const AInfo: TTensorInfo): Pointer;
begin
  if not FIsOpen then
    raise EInvalidOperation.Create('Safetensors file is not open');

  if (AInfo.Offset + AInfo.DataSize) > FFile.Size then
    raise ERangeError.CreateFmt(
      'Tensor "%s" data range [%d..%d] exceeds file size %d',
      [AInfo.TensorName, AInfo.Offset,
       AInfo.Offset + AInfo.DataSize, FFile.Size]);

  Result := Pointer(UIntPtr(FFile.Memory) + UIntPtr(AInfo.Offset));
end;

function TSafetensors.FileSize(): UInt64;
begin
  Result := FFile.Size;
end;

function TSafetensors.DataOffset(): UInt64;
begin
  Result := FDataOffset;
end;

end.
