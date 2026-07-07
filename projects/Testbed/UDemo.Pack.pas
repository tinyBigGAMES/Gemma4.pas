{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information
===============================================================================}

unit UDemo.Pack;

interface

uses
  System.SysUtils,
  System.IOUtils,
  StdApp.Console,
  Gemma4.Packer,
  UTestbed.Common;

type
  { TDemoPack }
  // Packs both models in one operation: Gemma 4 E4B (bf16 -> Q4_0 decoder
  // + F32 encoders) then EmbeddingGemma-300m (F32 pass-through), producing
  // a single Gemma4.vpk containing both model folders.
  TDemoPack = class(TBaseDemoCase)
  private
    procedure DoHandleProgress(
      const AInfo: TPackerProgressInfo;
      const AUserData: Pointer
    );
    function DoPackE4B(): Boolean;
    function DoPackEmbeddings(): Boolean;
  public
    constructor Create(); override;
    procedure OnRender(); override;
  end;

implementation

{ TDemoPack }

constructor TDemoPack.Create();
begin
  inherited;
  Title := 'Pack: E4B + Embeddings -> Gemma4.vpk';
end;

procedure TDemoPack.DoHandleProgress(
  const AInfo: TPackerProgressInfo;
  const AUserData: Pointer);
var
  LTotalGB: Double;
  LPct: Double;
begin
  case AInfo.Kind of
    ppStarting:
      PrintLn(COLOR_GREEN + 'Starting conversion: %d tensors',
        [AInfo.TensorTotal]);
    ppTensorBegin:
      Print(COLOR_YELLOW + '  [%d/%d] %s ... ',
        [AInfo.TensorIndex, AInfo.TensorTotal, AInfo.TensorName]);
    ppTensorEnd:
      PrintLn(COLOR_GREEN + 'done', []);
    ppWritingManifest:
      PrintLn(COLOR_WHITE + 'Writing manifest.json...', []);
    ppCopyingMeta:
      PrintLn(COLOR_WHITE + 'Copying meta files...', []);
    ppPackingVPK:
      PrintLn(COLOR_WHITE + 'Packing VPK archive...', []);
    ppVpkFileBegin:
      Print(COLOR_YELLOW + '  [%d/%d] %s ... ',
        [AInfo.TensorIndex, AInfo.TensorTotal, AInfo.TensorName]);
    ppVpkFileEnd:
    begin
      if AInfo.TotalBytes > 0 then
        LPct := 100.0 * AInfo.BytesWritten / AInfo.TotalBytes
      else
        LPct := 100.0;
      PrintLn(COLOR_GREEN + 'done (%.1f%%)', [LPct]);
    end;
    ppVpkWriting:
    begin
      LTotalGB := AInfo.TotalBytes / (1024.0 * 1024.0 * 1024.0);
      PrintLn(COLOR_WHITE + 'Writing archive to disk: %s (%.2f GB)...',
        [AInfo.TensorName, LTotalGB]);
    end;
    ppCompleted:
    begin
      PrintLn('', []);
      PrintLn(COLOR_GREEN + 'Pack complete!', []);
    end;
  end;
end;

function TDemoPack.DoPackE4B(): Boolean;
var
  LPacker: TPacker;
begin
  Result := False;
  PrintLn(COLOR_CYAN + '=== Phase 1: Gemma 4 E4B ===', []);
  PrintLn(COLOR_WHITE + 'Source: %s', [CSafetensorsFile]);
  PrintLn(COLOR_WHITE + 'Build : %s', [CVpkBuildDir]);
  PrintLn('', []);

  if not TFile.Exists(CSafetensorsFile) then
  begin
    PrintLn(COLOR_RED + 'Source not found: %s', [CSafetensorsFile]);
    Exit;
  end;

  LPacker := TPacker.Create();
  try
    LPacker.SetSafetensorsPath(CSafetensorsFile);
    LPacker.SetOutputDir(CVpkBuildDir);
    LPacker.SetVpkPath(CVpkOutputFile);
    LPacker.SetProgressCallback(DoHandleProgress);

    Result := LPacker.Pack();
    if not Result then
      PrintLn(COLOR_RED + 'E4B pack FAILED', []);

    FlushErrors(LPacker.GetErrors());
  finally
    LPacker.Free();
  end;
end;

function TDemoPack.DoPackEmbeddings(): Boolean;
var
  LPacker: TPacker;
begin
  Result := False;
  PrintLn('', []);
  PrintLn(COLOR_CYAN + '=== Phase 2: EmbeddingGemma-300m ===', []);
  PrintLn(COLOR_WHITE + 'Source: %s', [CEmbSafetensorsDir]);
  PrintLn(COLOR_WHITE + 'Build : %s', [CEmbBuildDir]);
  PrintLn('', []);

  if not TDirectory.Exists(CEmbSafetensorsDir) then
  begin
    PrintLn(COLOR_RED + 'Source not found: %s', [CEmbSafetensorsDir]);
    Exit;
  end;

  LPacker := TPacker.Create();
  try
    LPacker.SetProgressCallback(DoHandleProgress);

    Result := LPacker.PackEmbeddings(CEmbSafetensorsDir, CEmbBuildDir,
      CVpkOutputFile);
    if not Result then
      PrintLn(COLOR_RED + 'Embeddings pack FAILED', []);

    FlushErrors(LPacker.GetErrors());
  finally
    LPacker.Free();
  end;
end;

procedure TDemoPack.OnRender();
begin
  PrintLn(COLOR_WHITE + 'Output: %s', [CVpkOutputFile]);
  PrintLn('', []);

  if not DoPackE4B() then
    Exit;

  if not DoPackEmbeddings() then
    Exit;

  PrintLn('', []);
  PrintLn(COLOR_GREEN + 'VPK created: %s', [CVpkOutputFile]);
end;

end.
