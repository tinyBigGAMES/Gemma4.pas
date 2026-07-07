{==============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information

 ------------------------------------------------------------------------------

  Gemma4.Image - Gemma 4 image preprocessing (image file -> vision patches)

  Byte-exact port of HuggingFace Gemma4ImageProcessorPil
  (.claude\research\huggingface\image_processing_pil_gemma4.py):

    1. Decode PNG/JPEG/BMP/GIF via VCL Graphics -> interleaved RGB8
    2. Aspect-ratio-preserving resize (dims floored to multiples of 48),
       PIL BICUBIC in the uint8 domain - exact port of Pillow's Resample.c
       fixed-point separable resampler (PRECISION_BITS = 22)
    3. Rescale * (1/255) computed in Double, stored Single (matches numpy)
    4. Patchify 16x16 -> [N x 768] F32 + [N x 2] int32 (x, y) positions

  Output feeds TVision.EncodeImage directly. Padding rows are NOT produced
  (TVision consumes real patches only, same as the vis_test reference dump).

  Known documented deviation: for transparent images loaded through the
  generic path (GIF, palette PNG), pixels are composited onto white, while
  PIL convert("RGB") drops the alpha channel. RGB/RGBA 8-bit PNGs take the
  direct scanline path, which drops alpha exactly like PIL.
==============================================================================}

unit Gemma4.Image;

{$I StdApp.Defines.inc}

interface

uses
  Winapi.Windows,
  System.SysUtils,
  System.Classes,
  System.Math,
  System.Types,
  Vcl.Graphics,
  Vcl.Imaging.pngimage,
  Vcl.Imaging.jpeg,
  Vcl.Imaging.GIFImg,
  StdApp.Base;

const
  CIMG_ERR_LOAD = 'IM01';
  CIMG_ERR_SIZE = 'IM02';
  CIMG_ERR_PROCESS = 'IM03';

  CImgPatchSize = 16;
  CImgMaxSoftTokens = 280;
  CImgPoolKernel = 3;

  CImgMaxPatches = CImgMaxSoftTokens * CImgPoolKernel * CImgPoolKernel;

  // resized height/width must be divisible by pooling_kernel * patch = 48
  CImgSideMult = CImgPoolKernel * CImgPatchSize;

  // patch_size * patch_size * 3 channels = 768
  CImgPatchDim = CImgPatchSize * CImgPatchSize * 3;

  // Pillow Resample.c: PRECISION_BITS = 32 - 8 - 2
  CImgPrecisionBits = 22;

  // numpy: uint8 * (1/255) in float64, then cast to float32
  CImgRescale: Double = 1.0 / 255.0;

type
  { TRGBImage }
  // Interleaved RGB8, row-major, 3 bytes per pixel
  TRGBImage = record
    Width: Integer;
    Height: Integer;
    Pixels: TArray<Byte>;
  end;

  { TImagePipeline }
  // Image file -> patches/positions for TVision.EncodeImage.
  // Create() once, ProcessFile() per image, Free().
  TImagePipeline = class(TBaseObject)
  private
    function DoLoadPng(const AFileName: string; out AImage: TRGBImage): Boolean;
    function DoLoadGeneric(const AFileName: string; out AImage: TRGBImage): Boolean;
    function DoBitmapToRGB(const ABitmap: TBitmap; out AImage: TRGBImage): Boolean;
    procedure DoPrecomputeCoeffs(const AInSize: Integer; const AOutSize: Integer;
      out ABounds: TArray<Integer>; out ACoeffs: TArray<Integer>;
      out AKSize: Integer);
    procedure DoResampleHorizontal(const AIn: TRGBImage; const AOutWidth: Integer;
      out AOut: TRGBImage);
    procedure DoResampleVertical(const AIn: TRGBImage; const AOutHeight: Integer;
      out AOut: TRGBImage);
    procedure DoPatchify(const AImage: TRGBImage; out APatches: TArray<Single>;
      out APositions: TArray<Integer>);
  public
    // Port of get_aspect_ratio_preserving_size. False = image cannot fit
    // the patch budget (both dims floored to zero, or budget exceeded).
    // AMaxSoftTokens: 280 for images (default), 70 per video frame.
    class function ComputeTargetSize(const AHeight: Integer;
      const AWidth: Integer; out ATargetHeight: Integer;
      out ATargetWidth: Integer;
      const AMaxSoftTokens: Integer = CImgMaxSoftTokens): Boolean;
    // Decode an image file to interleaved RGB8 (PNG/JPEG/BMP/GIF)
    function LoadImageFile(const AFileName: string;
      out AImage: TRGBImage): Boolean;
    // RGB8 -> resize -> rescale -> patchify. ANumSoftTokens = patches div 9.
    function ProcessRGB(const AImage: TRGBImage; out APatches: TArray<Single>;
      out APositions: TArray<Integer>; out ANumSoftTokens: Integer;
      const AMaxSoftTokens: Integer = CImgMaxSoftTokens): Boolean;
    // LoadImageFile + ProcessRGB
    function ProcessFile(const AFileName: string; out APatches: TArray<Single>;
      out APositions: TArray<Integer>; out ANumSoftTokens: Integer;
      const AMaxSoftTokens: Integer = CImgMaxSoftTokens): Boolean;
  end;

implementation

type
  { TRGBTripleRow }
  // Scanline view for pf24bit bitmaps and TPngImage color data (both BGR)
  TRGBTripleRow = array[0..MaxInt div SizeOf(TRGBTriple) - 1] of TRGBTriple;
  PRGBTripleRow = ^TRGBTripleRow;

{ BicubicFilter }
// Pillow bicubic_filter, a = -0.5 (Keys)
function BicubicFilter(const AX: Double): Double;
const
  { CA }
  CA = -0.5;
var
  LX: Double;
begin
  LX := Abs(AX);
  if LX < 1.0 then
    Result := ((CA + 2.0) * LX - (CA + 3.0)) * LX * LX + 1.0
  else if LX < 2.0 then
    Result := (((LX - 5.0) * LX + 8.0) * LX - 4.0) * CA
  else
    Result := 0.0;
end;

{ Clip8 }
// Pillow clip8: clamp BEFORE the fraction shift
function Clip8(const AValue: Integer): Byte;
begin
  if AValue >= (1 shl (CImgPrecisionBits + 8)) then
    Result := 255
  else if AValue <= 0 then
    Result := 0
  else
    Result := Byte(AValue shr CImgPrecisionBits);
end;

{ TImagePipeline }

class function TImagePipeline.ComputeTargetSize(const AHeight: Integer;
  const AWidth: Integer; out ATargetHeight: Integer;
  out ATargetWidth: Integer; const AMaxSoftTokens: Integer): Boolean;
var
  LFactor: Double;
  LIdealH: Double;
  LIdealW: Double;
  LMaxSide: Integer;
  LMaxPatches: Integer;
  LTargetPx: Int64;
begin
  Result := False;
  ATargetHeight := 0;
  ATargetWidth := 0;
  if (AHeight <= 0) or (AWidth <= 0) or (AMaxSoftTokens <= 0) then
    Exit;

  // max_patches = max_soft_tokens * pooling_kernel^2; budget in pixels
  LMaxPatches := AMaxSoftTokens * CImgPoolKernel * CImgPoolKernel;
  LTargetPx := Int64(LMaxPatches) * CImgPatchSize * CImgPatchSize;

  // factor = sqrt(target_px / total_px); ideal dims scaled by it
  LFactor := Sqrt(LTargetPx / (Int64(AHeight) * Int64(AWidth)));
  LIdealH := LFactor * AHeight;
  LIdealW := LFactor * AWidth;

  // Round DOWN to the nearest multiple of 48
  ATargetHeight := Floor(LIdealH / CImgSideMult) * CImgSideMult;
  ATargetWidth := Floor(LIdealW / CImgSideMult) * CImgSideMult;

  // Both zero: aspect ratio too extreme for the budget at all
  if (ATargetHeight = 0) and (ATargetWidth = 0) then
    Exit;

  LMaxSide := (LMaxPatches div (CImgPoolKernel * CImgPoolKernel)) *
    CImgSideMult;
  if ATargetHeight = 0 then
  begin
    // Degenerate flat-wide image: minimum height, width from aspect ratio
    ATargetHeight := CImgSideMult;
    ATargetWidth := Min(Floor(AWidth / AHeight) * CImgSideMult, LMaxSide);
  end
  else if ATargetWidth = 0 then
  begin
    // Degenerate tall-narrow image: minimum width, height from aspect ratio
    ATargetWidth := CImgSideMult;
    ATargetHeight := Min(Floor(AHeight / AWidth) * CImgSideMult, LMaxSide);
  end;

  // Reference guard: resized area must not exceed the patch budget
  if Int64(ATargetHeight) * Int64(ATargetWidth) > LTargetPx then
  begin
    ATargetHeight := 0;
    ATargetWidth := 0;
    Exit;
  end;

  Result := True;
end;

procedure TImagePipeline.DoPrecomputeCoeffs(const AInSize: Integer;
  const AOutSize: Integer; out ABounds: TArray<Integer>;
  out ACoeffs: TArray<Integer>; out AKSize: Integer);
var
  LScale: Double;
  LFilterScale: Double;
  LSupport: Double;
  LCenter: Double;
  LWW: Double;
  LSS: Double;
  LW: Double;
  LV: Double;
  LXX: Integer;
  LX: Integer;
  LXMin: Integer;
  LXMax: Integer;
  LKK: TArray<Double>;
begin
  // Port of Pillow Resample.c precompute_coeffs + normalize_coeffs_8bpc.
  // in0 = 0, in1 = AInSize (no box cropping).
  LScale := AInSize / AOutSize;
  LFilterScale := LScale;
  if LFilterScale < 1.0 then
    LFilterScale := 1.0;

  // Bicubic filter support = 2.0, widened for downscale
  LSupport := 2.0 * LFilterScale;
  AKSize := Ceil(LSupport) * 2 + 1;

  SetLength(LKK, AOutSize * AKSize);
  SetLength(ACoeffs, AOutSize * AKSize);
  SetLength(ABounds, AOutSize * 2);

  for LXX := 0 to AOutSize - 1 do
  begin
    LCenter := (LXX + 0.5) * LScale;
    LWW := 0.0;
    LSS := 1.0 / LFilterScale;

    // Pillow rounds via C truncation (int)(x + 0.5) -- Trunc matches
    LXMin := Trunc(LCenter - LSupport + 0.5);
    if LXMin < 0 then
      LXMin := 0;
    LXMax := Trunc(LCenter + LSupport + 0.5);
    if LXMax > AInSize then
      LXMax := AInSize;
    LXMax := LXMax - LXMin;

    for LX := 0 to LXMax - 1 do
    begin
      LW := BicubicFilter((LX + LXMin - LCenter + 0.5) * LSS);
      LKK[LXX * AKSize + LX] := LW;
      LWW := LWW + LW;
    end;
    // Normalize so the taps sum to 1
    for LX := 0 to LXMax - 1 do
    begin
      if LWW <> 0.0 then
        LKK[LXX * AKSize + LX] := LKK[LXX * AKSize + LX] / LWW;
    end;
    // Remaining slots stay zero
    for LX := LXMax to AKSize - 1 do
      LKK[LXX * AKSize + LX] := 0.0;

    ABounds[LXX * 2 + 0] := LXMin;
    ABounds[LXX * 2 + 1] := LXMax;
  end;

  // Fixed-point quantization: 22 fraction bits, round-half-away-from-zero
  for LX := 0 to (AOutSize * AKSize) - 1 do
  begin
    LV := LKK[LX] * (1 shl CImgPrecisionBits);
    if LKK[LX] < 0.0 then
      ACoeffs[LX] := Trunc(-0.5 + LV)
    else
      ACoeffs[LX] := Trunc(0.5 + LV);
  end;
end;

procedure TImagePipeline.DoResampleHorizontal(const AIn: TRGBImage;
  const AOutWidth: Integer; out AOut: TRGBImage);
var
  LBounds: TArray<Integer>;
  LCoeffs: TArray<Integer>;
  LKSize: Integer;
  LY: Integer;
  LXX: Integer;
  LX: Integer;
  LXMin: Integer;
  LXMax: Integer;
  LSS0: Integer;
  LSS1: Integer;
  LSS2: Integer;
  LK: Integer;
  LSrc: Integer;
  LDst: Integer;
begin
  DoPrecomputeCoeffs(AIn.Width, AOutWidth, LBounds, LCoeffs, LKSize);
  AOut.Width := AOutWidth;
  AOut.Height := AIn.Height;
  SetLength(AOut.Pixels, AOutWidth * AIn.Height * 3);

  for LY := 0 to AIn.Height - 1 do
  begin
    for LXX := 0 to AOutWidth - 1 do
    begin
      LXMin := LBounds[LXX * 2 + 0];
      LXMax := LBounds[LXX * 2 + 1];
      // 32-bit accumulators seeded with the half-ulp rounding bias (PIL exact)
      LSS0 := 1 shl (CImgPrecisionBits - 1);
      LSS1 := LSS0;
      LSS2 := LSS0;
      for LX := 0 to LXMax - 1 do
      begin
        LK := LCoeffs[LXX * LKSize + LX];
        LSrc := (LY * AIn.Width + (LXMin + LX)) * 3;
        LSS0 := LSS0 + AIn.Pixels[LSrc + 0] * LK;
        LSS1 := LSS1 + AIn.Pixels[LSrc + 1] * LK;
        LSS2 := LSS2 + AIn.Pixels[LSrc + 2] * LK;
      end;
      LDst := (LY * AOutWidth + LXX) * 3;
      AOut.Pixels[LDst + 0] := Clip8(LSS0);
      AOut.Pixels[LDst + 1] := Clip8(LSS1);
      AOut.Pixels[LDst + 2] := Clip8(LSS2);
    end;
  end;
end;

procedure TImagePipeline.DoResampleVertical(const AIn: TRGBImage;
  const AOutHeight: Integer; out AOut: TRGBImage);
var
  LBounds: TArray<Integer>;
  LCoeffs: TArray<Integer>;
  LKSize: Integer;
  LYY: Integer;
  LY: Integer;
  LX: Integer;
  LYMin: Integer;
  LYMax: Integer;
  LSS0: Integer;
  LSS1: Integer;
  LSS2: Integer;
  LK: Integer;
  LSrc: Integer;
  LDst: Integer;
begin
  DoPrecomputeCoeffs(AIn.Height, AOutHeight, LBounds, LCoeffs, LKSize);
  AOut.Width := AIn.Width;
  AOut.Height := AOutHeight;
  SetLength(AOut.Pixels, AIn.Width * AOutHeight * 3);

  for LYY := 0 to AOutHeight - 1 do
  begin
    LYMin := LBounds[LYY * 2 + 0];
    LYMax := LBounds[LYY * 2 + 1];
    for LX := 0 to AIn.Width - 1 do
    begin
      LSS0 := 1 shl (CImgPrecisionBits - 1);
      LSS1 := LSS0;
      LSS2 := LSS0;
      for LY := 0 to LYMax - 1 do
      begin
        LK := LCoeffs[LYY * LKSize + LY];
        LSrc := ((LYMin + LY) * AIn.Width + LX) * 3;
        LSS0 := LSS0 + AIn.Pixels[LSrc + 0] * LK;
        LSS1 := LSS1 + AIn.Pixels[LSrc + 1] * LK;
        LSS2 := LSS2 + AIn.Pixels[LSrc + 2] * LK;
      end;
      LDst := (LYY * AIn.Width + LX) * 3;
      AOut.Pixels[LDst + 0] := Clip8(LSS0);
      AOut.Pixels[LDst + 1] := Clip8(LSS1);
      AOut.Pixels[LDst + 2] := Clip8(LSS2);
    end;
  end;
end;

procedure TImagePipeline.DoPatchify(const AImage: TRGBImage;
  out APatches: TArray<Single>; out APositions: TArray<Integer>);
var
  LGridW: Integer;
  LGridH: Integer;
  LPY: Integer;
  LPX: Integer;
  LIY: Integer;
  LIX: Integer;
  LC: Integer;
  LPatch: Integer;
  LSrc: Integer;
  LDst: Integer;
begin
  LGridW := AImage.Width div CImgPatchSize;
  LGridH := AImage.Height div CImgPatchSize;
  SetLength(APatches, LGridW * LGridH * CImgPatchDim);
  SetLength(APositions, LGridW * LGridH * 2);

  // Patches row-major over (patch_y, patch_x); within a patch (iy, ix, c)
  // with channel last -- matches convert_image_to_patches + meshgrid 'xy'
  for LPY := 0 to LGridH - 1 do
  begin
    for LPX := 0 to LGridW - 1 do
    begin
      LPatch := LPY * LGridW + LPX;
      APositions[LPatch * 2 + 0] := LPX;  // x
      APositions[LPatch * 2 + 1] := LPY;  // y
      for LIY := 0 to CImgPatchSize - 1 do
      begin
        for LIX := 0 to CImgPatchSize - 1 do
        begin
          LSrc := ((LPY * CImgPatchSize + LIY) * AImage.Width +
            (LPX * CImgPatchSize + LIX)) * 3;
          LDst := LPatch * CImgPatchDim + (LIY * CImgPatchSize + LIX) * 3;
          for LC := 0 to 2 do
          begin
            // Byte * Double -> Double, assigned to Single (numpy-exact)
            APatches[LDst + LC] := AImage.Pixels[LSrc + LC] * CImgRescale;
          end;
        end;
      end;
    end;
  end;
end;

function TImagePipeline.DoLoadPng(const AFileName: string;
  out AImage: TRGBImage): Boolean;
var
  LPng: TPngImage;
  LLine: PRGBTripleRow;
  LY: Integer;
  LX: Integer;
  LDst: Integer;
begin
  // TPngImage keeps color triples (BGR) in Scanline with alpha stored
  // separately -- reading Scanline drops alpha exactly like PIL convert("RGB")
  LPng := TPngImage.Create();
  try
    LPng.LoadFromFile(AFileName);
    if ((LPng.Header.ColorType = COLOR_RGB) or
      (LPng.Header.ColorType = COLOR_RGBALPHA)) and
      (LPng.Header.BitDepth = 8) then
    begin
      AImage.Width := LPng.Width;
      AImage.Height := LPng.Height;
      SetLength(AImage.Pixels, LPng.Width * LPng.Height * 3);
      for LY := 0 to LPng.Height - 1 do
      begin
        LLine := PRGBTripleRow(LPng.Scanline[LY]);
        LDst := LY * LPng.Width * 3;
        for LX := 0 to LPng.Width - 1 do
        begin
          AImage.Pixels[LDst + LX * 3 + 0] := LLine^[LX].rgbtRed;
          AImage.Pixels[LDst + LX * 3 + 1] := LLine^[LX].rgbtGreen;
          AImage.Pixels[LDst + LX * 3 + 2] := LLine^[LX].rgbtBlue;
        end;
      end;
      Result := True;
    end
    else
    begin
      // Grayscale / palette PNG: render through the generic 24-bit path
      Result := DoLoadGeneric(AFileName, AImage);
    end;
  finally
    LPng.Free();
  end;
end;

function TImagePipeline.DoLoadGeneric(const AFileName: string;
  out AImage: TRGBImage): Boolean;
var
  LPicture: TPicture;
  LBitmap: TBitmap;
begin
  Result := False;
  LPicture := TPicture.Create();
  LBitmap := TBitmap.Create();
  try
    LPicture.LoadFromFile(AFileName);
    if (LPicture.Width <= 0) or (LPicture.Height <= 0) then
    begin
      GetErrors().Add(esError, CIMG_ERR_LOAD, 'Empty image: ' + AFileName);
      Exit;
    end;
    LBitmap.PixelFormat := pf24bit;
    LBitmap.SetSize(LPicture.Width, LPicture.Height);
    // White backdrop for transparent formats (documented deviation from
    // PIL's drop-alpha; only affects images with transparency)
    LBitmap.Canvas.Brush.Color := clWhite;
    LBitmap.Canvas.FillRect(Rect(0, 0, LPicture.Width, LPicture.Height));
    LBitmap.Canvas.Draw(0, 0, LPicture.Graphic);
    Result := DoBitmapToRGB(LBitmap, AImage);
  finally
    LBitmap.Free();
    LPicture.Free();
  end;
end;

function TImagePipeline.DoBitmapToRGB(const ABitmap: TBitmap;
  out AImage: TRGBImage): Boolean;
var
  LLine: PRGBTripleRow;
  LY: Integer;
  LX: Integer;
  LDst: Integer;
begin
  // pf24bit scanlines are BGR triples, rows accessed top-down via ScanLine[]
  AImage.Width := ABitmap.Width;
  AImage.Height := ABitmap.Height;
  SetLength(AImage.Pixels, ABitmap.Width * ABitmap.Height * 3);
  for LY := 0 to ABitmap.Height - 1 do
  begin
    LLine := PRGBTripleRow(ABitmap.ScanLine[LY]);
    LDst := LY * ABitmap.Width * 3;
    for LX := 0 to ABitmap.Width - 1 do
    begin
      AImage.Pixels[LDst + LX * 3 + 0] := LLine^[LX].rgbtRed;
      AImage.Pixels[LDst + LX * 3 + 1] := LLine^[LX].rgbtGreen;
      AImage.Pixels[LDst + LX * 3 + 2] := LLine^[LX].rgbtBlue;
    end;
  end;
  Result := True;
end;

function TImagePipeline.LoadImageFile(const AFileName: string;
  out AImage: TRGBImage): Boolean;
begin
  Result := False;
  AImage.Width := 0;
  AImage.Height := 0;
  AImage.Pixels := nil;

  if not FileExists(AFileName) then
  begin
    GetErrors().Add(esError, CIMG_ERR_LOAD, 'File not found: ' + AFileName);
    Exit;
  end;

  try
    if SameText(ExtractFileExt(AFileName), '.png') then
      Result := DoLoadPng(AFileName, AImage)
    else
      Result := DoLoadGeneric(AFileName, AImage);
  except
    on E: Exception do
    begin
      GetErrors().Add(esError, CIMG_ERR_LOAD,
        'Decode failed: ' + AFileName + ' -- ' + E.Message);
      Result := False;
    end;
  end;
end;

function TImagePipeline.ProcessRGB(const AImage: TRGBImage;
  out APatches: TArray<Single>; out APositions: TArray<Integer>;
  out ANumSoftTokens: Integer; const AMaxSoftTokens: Integer): Boolean;
var
  LTargetH: Integer;
  LTargetW: Integer;
  LCur: TRGBImage;
  LTmp: TRGBImage;
begin
  Result := False;
  APatches := nil;
  APositions := nil;
  ANumSoftTokens := 0;

  if (AImage.Width <= 0) or (AImage.Height <= 0) or
    (Length(AImage.Pixels) <> AImage.Width * AImage.Height * 3) then
  begin
    GetErrors().Add(esError, CIMG_ERR_PROCESS, 'Invalid RGB input image');
    Exit;
  end;

  if not ComputeTargetSize(AImage.Height, AImage.Width, LTargetH, LTargetW,
    AMaxSoftTokens) then
  begin
    GetErrors().Add(esError, CIMG_ERR_SIZE, Format(
      'Cannot fit %dx%d into the patch budget',
      [AImage.Width, AImage.Height]));
    Exit;
  end;

  LCur := AImage;
  // Pillow two-pass order: horizontal first, then vertical; no-op passes
  // are skipped (identical to ImagingResampleInner)
  if LTargetW <> LCur.Width then
  begin
    DoResampleHorizontal(LCur, LTargetW, LTmp);
    LCur := LTmp;
  end;
  if LTargetH <> LCur.Height then
  begin
    DoResampleVertical(LCur, LTargetH, LTmp);
    LCur := LTmp;
  end;

  DoPatchify(LCur, APatches, APositions);
  ANumSoftTokens := (Length(APositions) div 2) div
    (CImgPoolKernel * CImgPoolKernel);
  Result := True;
end;

function TImagePipeline.ProcessFile(const AFileName: string;
  out APatches: TArray<Single>; out APositions: TArray<Integer>;
  out ANumSoftTokens: Integer; const AMaxSoftTokens: Integer): Boolean;
var
  LImage: TRGBImage;
begin
  Result := False;
  APatches := nil;
  APositions := nil;
  ANumSoftTokens := 0;
  if not LoadImageFile(AFileName, LImage) then
    Exit;
  Result := ProcessRGB(LImage, APatches, APositions, ANumSoftTokens,
    AMaxSoftTokens);
end;

end.
