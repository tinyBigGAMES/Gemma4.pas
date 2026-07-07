{===============================================================================
  Gemma4.pas� - Local LLM inference in Pascal

  Copyright � 2026-present tinyBigGAMES� LLC
  All Rights Reserved.

  See LICENSE for license information

 -------------------------------------------------------------------------------

  Gemma4.Packer - Offline packing tool

  Reads a .safetensors file (bf16 weights), quantizes weight tensors
  to Q4_0, keeps norms/scalars as F32, and writes:
    - weights/weights.bin  (flat binary blob of all tensor data)
    - manifest.json        (tensor directory with offsets into weights.bin)
    - meta/                (copies of config.json, tokenizer.json, etc.)

  Then packs the output directory into a .vpk archive via TVFS.

  Dependencies: StdApp.Base, StdApp.JSON, StdApp.Console,
    StdApp.VirtualMemory, StdApp.VFS, Gemma4.Types,
    Gemma4.Safetensors, Gemma4.Quant
===============================================================================}

unit Gemma4.Packer;

{$I StdApp.Defines.inc}

interface

uses
  System.SysUtils,
  System.IOUtils,
  System.Classes,
  System.Generics.Collections,
  StdApp.Base,
  StdApp.JSON,
  StdApp.Console,
  StdApp.Utils,
  StdApp.VirtualMemory,
  StdApp.VFS,
  Gemma4.Types,
  Gemma4.Safetensors,
  Gemma4.Quant;

const
  CPK_ERR_SAFETENSORS = 'PK01';
  CPK_ERR_OUTPUT_DIR = 'PK02';
  CPK_ERR_WEIGHTS = 'PK03';
  CPK_ERR_MANIFEST = 'PK04';
  CPK_ERR_META_COPY = 'PK05';
  CPK_ERR_VPK_PACK = 'PK06';

type
  { TPackerProgressKind }
  TPackerProgressKind = (
    ppStarting,
    ppTensorBegin,
    ppTensorEnd,
    ppWritingManifest,
    ppCopyingMeta,
    ppPackingVPK,
    ppVpkFileBegin,
    ppVpkFileEnd,
    ppVpkWriting,
    ppCompleted
  );

  { TPackerProgressInfo }
  TPackerProgressInfo = record
    Kind: TPackerProgressKind;
    TensorName: string;
    TensorIndex: Integer;
    TensorTotal: Integer;
    BytesWritten: UInt64;
    TotalBytes: UInt64;
  end;

  { TPackerProgressCallback }
  TPackerProgressCallback = reference to procedure(
    const AInfo: TPackerProgressInfo;
    const AUserData: Pointer
  );

  { TPacker }
  // Orchestrates the offline conversion from safetensors to VPK.
  TPacker = class(TBaseObject)
  private
    FSafetensorsPath: string;
    FOutputDir: string;
    FVpkPath: string;
    FProgressCallback: TCallback<TPackerProgressCallback>;

    procedure DoProgress(
      const AKind: TPackerProgressKind;
      const ATensorName: string;
      const ATensorIndex: Integer;
      const ATensorTotal: Integer;
      const ABytesWritten: UInt64;
      const ATotalBytes: UInt64
    );
    function DoCreateOutputDirs(): Boolean;
    function DoConvertTensors(
      const ASafetensors: TSafetensors
    ): Boolean;
    function DoCopyMetaFiles(): Boolean;
    function DoPackVPK(): Boolean;
    function DoShouldQuantize(const AInfo: TTensorInfo): Boolean;
    function DoIsEncoderTensor(const AName: string): Boolean;
  public
    constructor Create(); override;
    destructor Destroy(); override;

    // Configure paths before calling Pack
    procedure SetSafetensorsPath(const APath: string);
    procedure SetOutputDir(const APath: string);
    procedure SetVpkPath(const APath: string);
    procedure SetProgressCallback(
      const ACallback: TPackerProgressCallback;
      const AUserData: Pointer = nil
    );

    // Execute the full pipeline
    function Pack(): Boolean;

    // Pack EmbeddingGemma-300m: F32 pass-through (no quantization), main
    // model + the two sentence-transformers dense heads, flat model folder.
    // Every tensor offset is 256-aligned so GPU aligned binds get vec4-exact
    // residuals. Re-packs the multi-model root (parent of AOutputDir).
    function PackEmbeddings(
      const ASafetensorsDir: string;
      const AOutputDir: string;
      const AVpkPath: string
    ): Boolean;
  end;

implementation

{ TPacker }

constructor TPacker.Create();
begin
  inherited Create();
  FSafetensorsPath := '';
  FOutputDir := '';
  FVpkPath := '';
end;

destructor TPacker.Destroy();
begin
  inherited;
end;

procedure TPacker.SetSafetensorsPath(const APath: string);
begin
  FSafetensorsPath := APath;
end;

procedure TPacker.SetOutputDir(const APath: string);
begin
  FOutputDir := APath;
end;

procedure TPacker.SetVpkPath(const APath: string);
begin
  FVpkPath := APath;
end;

procedure TPacker.SetProgressCallback(
  const ACallback: TPackerProgressCallback;
  const AUserData: Pointer);
begin
  FProgressCallback.Callback := ACallback;
  FProgressCallback.UserData := AUserData;
end;

procedure TPacker.DoProgress(
  const AKind: TPackerProgressKind;
  const ATensorName: string;
  const ATensorIndex: Integer;
  const ATensorTotal: Integer;
  const ABytesWritten: UInt64;
  const ATotalBytes: UInt64);
var
  LInfo: TPackerProgressInfo;
begin
  if not FProgressCallback.IsAssigned() then
    Exit;

  LInfo.Kind := AKind;
  LInfo.TensorName := ATensorName;
  LInfo.TensorIndex := ATensorIndex;
  LInfo.TensorTotal := ATensorTotal;
  LInfo.BytesWritten := ABytesWritten;
  LInfo.TotalBytes := ATotalBytes;

  FProgressCallback.Callback(LInfo, FProgressCallback.UserData);
end;

function TPacker.DoShouldQuantize(const AInfo: TTensorInfo): Boolean;
begin
  // Quantize to Q4_0 only if:
  //   1. Source dtype is BF16 (weight matrices)
  //   2. It is a 2D weight tensor (rows x cols)
  //   3. Element count is a multiple of 32 (Q4 block size)
  // Everything else (norms, biases, scalars, embeddings with odd shapes)
  // stays as F32.
  Result := (AInfo.SourceDtype = dkBF16) and
            (AInfo.DimCount() = 2) and
            ((AInfo.ElementCount() mod CQ4BlockSize) = 0);
end;

function TPacker.DoIsEncoderTensor(const AName: string): Boolean;
begin
  // Vision/audio towers + multimodal embedders are packed into
  // encoders.bin as F32 (never quantized -- precision-sensitive)
  Result := AName.StartsWith('model.vision_tower.') or
            AName.StartsWith('model.audio_tower.') or
            AName.StartsWith('model.embed_vision.') or
            AName.StartsWith('model.embed_audio.');
end;

function TPacker.DoCreateOutputDirs(): Boolean;
begin
  // Flat per-model layout: FOutputDir IS the model folder (e.g. ...\Gemma4\E4B);
  // all files (weights.bin, manifest.json, meta files) live directly inside it
  Result := False;
  try
    if not TDirectory.Exists(FOutputDir) then
      TUtils.CreateDirInPath(FOutputDir);

    Result := True;
  except
    on E: Exception do
      GetErrors().Add(esError, CPK_ERR_OUTPUT_DIR,
        'Failed to create output directories: %s', [E.Message]);
  end;
end;

function TPacker.DoConvertTensors(
  const ASafetensors: TSafetensors): Boolean;
var
  LWeightsPath: string;
  LManifestPath: string;
  LEncodersPath: string;
  LEncManifestPath: string;
  LWeightsFile: TFileStream;
  LEncodersFile: TFileStream;
  LManifest: TJSON;
  LEncManifest: TJSON;
  LTargetManifest: TJSON;
  LInfo: TTensorInfo;
  LI: Integer;
  LJ: Integer;
  LTotalTensors: Integer;
  LSrcPtr: Pointer;
  LF32Buf: TArray<Single>;
  LQ4Buf: TArray<TQ4Block>;
  LElementCount: UInt64;
  LBlockCount: UInt64;
  LWriteSize: UInt64;
  LCurrentOffset: UInt64;
  LEncOffset: UInt64;
  LEntryOffset: UInt64;
  LPad: UInt64;
  LZero: array[0..255] of Byte;
  LOutputDtype: TDtypeKind;
begin
  LWeightsPath := TPath.Combine(FOutputDir, 'weights.bin');
  LManifestPath := TPath.Combine(FOutputDir, 'manifest.json');
  LEncodersPath := TPath.Combine(FOutputDir, 'encoders.bin');
  LEncManifestPath := TPath.Combine(FOutputDir, 'encoders_manifest.json');
  LCurrentOffset := 0;
  LEncOffset := 0;
  LTotalTensors := ASafetensors.TensorCount();

  DoProgress(ppStarting, '', 0, LTotalTensors, 0, 0);

  LManifest := TJSON.Create();
  LEncManifest := TJSON.Create();
  try
    LManifest.BeginArray('tensors');
    LEncManifest.BeginArray('tensors');

    LWeightsFile := TFileStream.Create(LWeightsPath, fmCreate);
    try
      LEncodersFile := TFileStream.Create(LEncodersPath, fmCreate);
      try
        for LI := 0 to LTotalTensors - 1 do
        begin
          LInfo := ASafetensors.GetTensor(LI);
          LSrcPtr := ASafetensors.GetDataPointer(LInfo);
          LElementCount := LInfo.ElementCount();

          DoProgress(ppTensorBegin, LInfo.TensorName, LI + 1,
            LTotalTensors, LCurrentOffset, 0);

          if DoIsEncoderTensor(LInfo.TensorName) then
          begin
            // --- Encoder stream: F32 pass-through, 256-aligned offsets ---
            // (same alignment contract as PackEmbeddings.DoAppendF32:
            // GPU aligned binds need vec4-exact residuals)
            LPad := (256 - (LEncOffset mod 256)) mod 256;
            if LPad > 0 then
            begin
              FillChar(LZero, SizeOf(LZero), 0);
              LEncodersFile.WriteBuffer(LZero, LPad);
              LEncOffset := LEncOffset + LPad;
            end;

            LOutputDtype := dkF32;
            SetLength(LF32Buf, LElementCount);
            if LInfo.SourceDtype = dkBF16 then
              BF16ToF32Buffer(LSrcPtr, @LF32Buf[0], LElementCount)
            else
              Move(LSrcPtr^, LF32Buf[0], LElementCount * SizeOf(Single));

            LWriteSize := LElementCount * SizeOf(Single);
            LEncodersFile.WriteBuffer(LF32Buf[0], LWriteSize);

            LEntryOffset := LEncOffset;
            LTargetManifest := LEncManifest;
            LEncOffset := LEncOffset + LWriteSize;
          end
          else if DoShouldQuantize(LInfo) then
          begin
            // --- Decoder stream: BF16 -> F32 -> Q4_0 (unchanged policy) ---
            LOutputDtype := dkQ4_0;
            SetLength(LF32Buf, LElementCount);
            BF16ToF32Buffer(LSrcPtr, @LF32Buf[0], LElementCount);

            LBlockCount := Q4BlockCount(LElementCount);
            SetLength(LQ4Buf, LBlockCount);
            F32ToQ4Buffer(@LF32Buf[0], @LQ4Buf[0], LElementCount);

            LWriteSize := LBlockCount * UInt64(SizeOf(TQ4Block));
            LWeightsFile.WriteBuffer(LQ4Buf[0], LWriteSize);

            LEntryOffset := LCurrentOffset;
            LTargetManifest := LManifest;
            LCurrentOffset := LCurrentOffset + LWriteSize;
          end
          else
          begin
            // --- Decoder stream: F32 (norms, scalars, non-2D) unchanged ---
            LOutputDtype := dkF32;
            SetLength(LF32Buf, LElementCount);

            if LInfo.SourceDtype = dkBF16 then
              BF16ToF32Buffer(LSrcPtr, @LF32Buf[0], LElementCount)
            else
              Move(LSrcPtr^, LF32Buf[0], LElementCount * SizeOf(Single));

            LWriteSize := LElementCount * SizeOf(Single);
            LWeightsFile.WriteBuffer(LF32Buf[0], LWriteSize);

            LEntryOffset := LCurrentOffset;
            LTargetManifest := LManifest;
            LCurrentOffset := LCurrentOffset + LWriteSize;
          end;

          // Manifest entry into whichever manifest owns this tensor
          LTargetManifest
            .BeginObject()
              .Add('name', LInfo.TensorName)
              .Add('source_dtype', DtypeKindToString(LInfo.SourceDtype))
              .Add('output_dtype', DtypeKindToString(LOutputDtype))
              .Add('offset', Int64(LEntryOffset))
              .Add('size', Int64(LWriteSize));

          LTargetManifest.BeginArray('shape');
          for LJ := 0 to High(LInfo.Shape) do
            LTargetManifest.Add(Int64(LInfo.Shape[LJ]));
          LTargetManifest.EndArray();

          LTargetManifest.EndObject();

          DoProgress(ppTensorEnd, LInfo.TensorName, LI + 1,
            LTotalTensors, LCurrentOffset, 0);
        end;
      finally
        LEncodersFile.Free();
      end;
    finally
      LWeightsFile.Free();
    end;

    LManifest.EndArray();
    LEncManifest.EndArray();

    // Write both manifests
    DoProgress(ppWritingManifest, '', 0, 0, LCurrentOffset, 0);
    LManifest.SaveToFile(LManifestPath);
    LEncManifest.SaveToFile(LEncManifestPath);

    Result := True;
  finally
    LEncManifest.Free();
    LManifest.Free();
  end;
end;

function TPacker.DoCopyMetaFiles(): Boolean;
var
  LSafetensorsDir: string;
  LMetaDir: string;
  LFiles: TArray<string>;
  LI: Integer;
  LSrcFile: string;
  LDstFile: string;
  LFilename: string;
begin
  Result := False;
  DoProgress(ppCopyingMeta, '', 0, 0, 0, 0);

  LSafetensorsDir := TPath.GetDirectoryName(FSafetensorsPath);
  // Flat per-model layout: meta files live beside weights.bin
  LMetaDir := FOutputDir;

  // List of meta files to copy from the safetensors directory
  LFiles := TArray<string>.Create(
    'config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'generation_config.json',
    'processor_config.json',
    'chat_template.jinja'
  );

  try
    for LI := 0 to High(LFiles) do
    begin
      LFilename := LFiles[LI];
      LSrcFile := TPath.Combine(LSafetensorsDir, LFilename);
      LDstFile := TPath.Combine(LMetaDir, LFilename);

      if TFile.Exists(LSrcFile) then
        TFile.Copy(LSrcFile, LDstFile, True);
      // Non-fatal if a meta file doesn't exist (some are optional)
    end;

    Result := True;
  except
    on E: Exception do
      GetErrors().Add(esError, CPK_ERR_META_COPY,
        'Failed to copy meta files: %s', [E.Message]);
  end;
end;

function TPacker.DoPackVPK(): Boolean;
var
  LVFS: TVFS;
begin
  DoProgress(ppPackingVPK, '', 0, 0, 0, 0);

  LVFS := TVFS.Create();
  try
    LVFS.SetErrors(GetErrors());
    // Pack the multi-model root (parent of the model folder): the VPK
    // contains E4B\ and embeddings\ as top-level folders. Per-file and
    // archive-write progress is forwarded so multi-GB packs are never
    // silent (TensorName/Index/Total carry file path/index/count here).
    Result := LVFS.PackDirectory(
      TPath.GetDirectoryName(TPath.GetFullPath(FOutputDir)), FVpkPath,
      procedure(const AInfo: TVFSPackInfo; var ACancel: Boolean;
        const AUserData: Pointer)
      begin
        ACancel := False;
        case AInfo.Status of
          cpsFileBegin:
            DoProgress(ppVpkFileBegin, AInfo.EntryPath, AInfo.FileIndex,
              AInfo.FileCount, AInfo.BytesWritten, AInfo.TotalBytes);
          cpsFileEnd:
            DoProgress(ppVpkFileEnd, AInfo.EntryPath, AInfo.FileIndex,
              AInfo.FileCount, AInfo.BytesWritten, AInfo.TotalBytes);
          cpsWritingArchive:
            DoProgress(ppVpkWriting, AInfo.Filename, AInfo.FileIndex,
              AInfo.FileCount, AInfo.BytesWritten, AInfo.TotalBytes);
        end;
      end);
    if not Result then
      GetErrors().Add(esError, CPK_ERR_VPK_PACK,
        'Failed to pack VPK archive');
  finally
    LVFS.Free();
  end;
end;

function TPacker.Pack(): Boolean;
var
  LSafetensors: TSafetensors;
begin
  Result := False;

  // Validate inputs
  if not TFile.Exists(FSafetensorsPath) then
  begin
    GetErrors().Add(esError, CPK_ERR_SAFETENSORS,
      'Safetensors file not found: %s', [FSafetensorsPath]);
    Exit;
  end;

  if FOutputDir = '' then
  begin
    GetErrors().Add(esError, CPK_ERR_OUTPUT_DIR,
      'Output directory not specified');
    Exit;
  end;

  if FVpkPath = '' then
  begin
    GetErrors().Add(esError, CPK_ERR_VPK_PACK,
      'VPK output path not specified');
    Exit;
  end;

  // Create output directory structure
  if not DoCreateOutputDirs() then
    Exit;

  // Open and parse safetensors
  LSafetensors := TSafetensors.Create();
  try
    LSafetensors.SetErrors(GetErrors());

    if not LSafetensors.Open(FSafetensorsPath) then
    begin
      GetErrors().Add(esError, CPK_ERR_SAFETENSORS,
        'Failed to open safetensors file');
      Exit;
    end;

    Status('Opened safetensors: %d tensors, %.2f GB', [
      LSafetensors.TensorCount(),
      LSafetensors.FileSize() / (1024.0 * 1024.0 * 1024.0)
    ]);

    // Convert tensors and write weights.bin + manifest.json
    if not DoConvertTensors(LSafetensors) then
      Exit;

    LSafetensors.Close();
  finally
    LSafetensors.Free();
  end;

  // Copy meta files
  if not DoCopyMetaFiles() then
    Exit;

  // Pack into VPK
  if not DoPackVPK() then
    Exit;

  DoProgress(ppCompleted, '', 0, 0, 0, 0);
  Result := True;
end;

function TPacker.PackEmbeddings(
  const ASafetensorsDir: string;
  const AOutputDir: string;
  const AVpkPath: string): Boolean;
var
  LMain: TSafetensors;
  LDense2: TSafetensors;
  LDense3: TSafetensors;
  LManifest: TJSON;
  LWeightsFile: TFileStream;
  LCurrentOffset: UInt64;
  LI: Integer;
  LInfo: TTensorInfo;

  // Append one F32 tensor (raw copy) to the blob, 256-aligning its offset,
  // and add its manifest entry under AStoreName. Norm gammas get +1.0
  // baked in: HF Gemma3RMSNorm computes x_norm * (1 + weight), while the
  // runtime shader multiplies by the raw weight (Gemma 4 convention).
  procedure DoAppendF32(const ASrc: TSafetensors; const AInfo: TTensorInfo;
    const AStoreName: string);
  var
    LPad: UInt64;
    LZero: array[0..255] of Byte;
    LWriteSize: UInt64;
    LJ: Integer;
    LNormBuf: TArray<Single>;
    LSrcPtr: PSingle;
    LCount: Integer;
    LK: Integer;
  begin
    // 256-align for GPU aligned binds (vec4-exact residuals)
    LPad := (256 - (LCurrentOffset mod 256)) mod 256;
    if LPad > 0 then
    begin
      FillChar(LZero, SizeOf(LZero), 0);
      LWeightsFile.WriteBuffer(LZero, LPad);
      LCurrentOffset := LCurrentOffset + LPad;
    end;

    LWriteSize := AInfo.ElementCount() * SizeOf(Single);

    if AStoreName.Contains('norm') then
    begin
      // Bake the Gemma3 (1 + weight) convention into the stored gamma
      LCount := Integer(AInfo.ElementCount());
      LSrcPtr := PSingle(ASrc.GetDataPointer(AInfo));
      SetLength(LNormBuf, LCount);
      for LK := 0 to LCount - 1 do
        LNormBuf[LK] :=
          PSingle(UIntPtr(LSrcPtr) + UIntPtr(LK * SizeOf(Single)))^ + 1.0;
      LWeightsFile.WriteBuffer(LNormBuf[0], LWriteSize);
    end
    else
      LWeightsFile.WriteBuffer(ASrc.GetDataPointer(AInfo)^, LWriteSize);

    LManifest
      .BeginObject()
        .Add('name', AStoreName)
        .Add('source_dtype', DtypeKindToString(dkF32))
        .Add('output_dtype', DtypeKindToString(dkF32))
        .Add('offset', Int64(LCurrentOffset))
        .Add('size', Int64(LWriteSize));
    LManifest.BeginArray('shape');
    for LJ := 0 to High(AInfo.Shape) do
      LManifest.Add(Int64(AInfo.Shape[LJ]));
    LManifest.EndArray();
    LManifest.EndObject();

    LCurrentOffset := LCurrentOffset + LWriteSize;
  end;

begin
  Result := False;

  try
    if not TDirectory.Exists(AOutputDir) then
      TUtils.CreateDirInPath(AOutputDir);
  except
    on E: Exception do
    begin
      GetErrors().Add(esError, CPK_ERR_OUTPUT_DIR,
        'Failed to create embeddings output dir: %s', [E.Message]);
      Exit;
    end;
  end;

  LMain := TSafetensors.Create();
  LDense2 := TSafetensors.Create();
  LDense3 := TSafetensors.Create();
  LManifest := TJSON.Create();
  try
    LMain.SetErrors(GetErrors());
    LDense2.SetErrors(GetErrors());
    LDense3.SetErrors(GetErrors());

    if not LMain.Open(TPath.Combine(ASafetensorsDir, 'model.safetensors')) then
    begin
      GetErrors().Add(esError, CPK_ERR_SAFETENSORS,
        'Failed to open embeddings model.safetensors');
      Exit;
    end;
    if not LDense2.Open(TPath.Combine(ASafetensorsDir,
      '2_Dense' + TPath.DirectorySeparatorChar + 'model.safetensors')) then
    begin
      GetErrors().Add(esError, CPK_ERR_SAFETENSORS,
        'Failed to open embeddings 2_Dense model.safetensors');
      Exit;
    end;
    if not LDense3.Open(TPath.Combine(ASafetensorsDir,
      '3_Dense' + TPath.DirectorySeparatorChar + 'model.safetensors')) then
    begin
      GetErrors().Add(esError, CPK_ERR_SAFETENSORS,
        'Failed to open embeddings 3_Dense model.safetensors');
      Exit;
    end;

    LCurrentOffset := 0;
    LManifest.BeginArray('tensors');

    LWeightsFile := TFileStream.Create(
      TPath.Combine(AOutputDir, 'weights.bin'), fmCreate);
    try
      // Main model: all tensors F32 pass-through under their own names
      for LI := 0 to LMain.TensorCount() - 1 do
      begin
        LInfo := LMain.GetTensor(LI);
        DoProgress(ppTensorBegin, LInfo.TensorName, LI + 1,
          LMain.TensorCount() + 2, LCurrentOffset, 0);
        DoAppendF32(LMain, LInfo, LInfo.TensorName);
        DoProgress(ppTensorEnd, LInfo.TensorName, LI + 1,
          LMain.TensorCount() + 2, LCurrentOffset, 0);
      end;

      // Sentence-transformers dense heads under invented stable names
      LInfo := LDense2.GetTensor(0);
      DoAppendF32(LDense2, LInfo, 'dense_2.linear.weight');
      LInfo := LDense3.GetTensor(0);
      DoAppendF32(LDense3, LInfo, 'dense_3.linear.weight');
    finally
      LWeightsFile.Free();
    end;

    LManifest.EndArray();

    DoProgress(ppWritingManifest, '', 0, 0, LCurrentOffset, 0);
    LManifest.SaveToFile(TPath.Combine(AOutputDir, 'manifest.json'));

    // Meta files, flat beside weights.bin
    try
      TFile.Copy(TPath.Combine(ASafetensorsDir, 'config.json'),
        TPath.Combine(AOutputDir, 'config.json'), True);
      TFile.Copy(TPath.Combine(ASafetensorsDir, 'tokenizer.json'),
        TPath.Combine(AOutputDir, 'tokenizer.json'), True);
      TFile.Copy(TPath.Combine(ASafetensorsDir, 'tokenizer_config.json'),
        TPath.Combine(AOutputDir, 'tokenizer_config.json'), True);
    except
      on E: Exception do
      begin
        GetErrors().Add(esError, CPK_ERR_META_COPY,
          'Failed to copy embeddings meta files: %s', [E.Message]);
        Exit;
      end;
    end;

    // Re-pack the multi-model root (parent of AOutputDir). DoPackVPK reads
    // FOutputDir/FVpkPath, so point them at the embeddings model folder.
    FOutputDir := AOutputDir;
    FVpkPath := AVpkPath;
    if not DoPackVPK() then
      Exit;

    DoProgress(ppCompleted, '', 0, 0, 0, 0);
    Result := True;
  finally
    LManifest.Free();
    LDense3.Free();
    LDense2.Free();
    LMain.Free();
  end;
end;

end.
