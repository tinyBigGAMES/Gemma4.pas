{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information

 -------------------------------------------------------------------------------

  Gemma4.Video - Gemma 4 video frame extraction (Windows Media Foundation)

  Extracts uniformly sampled RGB frames from any WMF-decodable video file
  (mp4/h264, mkv, wmv, ...) via IMFSourceReader with the built-in video
  processor converting to RGB32. OS-shipped COM only (mfplat/mfreadwrite),
  same zero-third-party standing as the Vulkan bindings: all required MF
  interfaces and GUIDs are declared here.

  HF Gemma4VideoProcessor mapping (video_processing_gemma4.py):
    32 uniformly sampled frames; each frame is processed like an image but
    with max_soft_tokens = 70 (max_patches = 630); do_normalize is the
    identity (mean 0, std 1), so the pixel path is rescale-only, exactly
    TImagePipeline.ProcessRGB(..., 70). Canonical prompt expansion per
    frame (processing_gemma4.py): "MM:SS <boi>" + <|video|>(258884) x n
    + "<eoi>", frames joined with single spaces.

  Known documented deviation: HF resizes video frames with torchvision's
  antialiased bicubic; we reuse the PIL-exact uint8 resampler. The two
  differ by sub-LSB rounding noise, absorbed by the encoder tolerance.

  Dependencies: Winapi.Windows, Winapi.ActiveX, StdApp.Base, Gemma4.Image
===============================================================================}

unit Gemma4.Video;

{$I StdApp.Defines.inc}

interface

uses
  Winapi.Windows,
  Winapi.ActiveX,
  System.SysUtils,
  System.Math,
  StdApp.Base,
  Gemma4.Image;

const
  CVID_ERR_OPEN = 'VD01';
  CVID_ERR_READ = 'VD02';

  // HF Gemma4VideoProcessor default frame count
  CVidNumFrames = 32;

  // Per-frame soft-token budget (max_patches = 70 * 9 = 630)
  CVidMaxSoftTokens = 70;

type
  { TVideoFrame }
  // One sampled frame: RGB pixels + its timestamp in seconds
  TVideoFrame = record
    Image: TRGBImage;
    TimestampSec: Double;
  end;

  { TVideo }
  // WMF frame extractor. Stateless per call; safe to reuse.
  TVideo = class(TBaseObject)
  public
    // Extract ANumFrames uniformly sampled frames. False + errors on
    // failure (missing file, undecodable format, no video stream).
    function ExtractFrames(const AFileName: string;
      const ANumFrames: Integer; out AFrames: TArray<TVideoFrame>): Boolean;
  end;

implementation

// ---------------------------------------------------------------------------
// Minimal Media Foundation declarations (validated against mfapi.h,
// mfreadwrite.h, mfidl.h from the Windows SDK). Interfaces declare their
// COMPLETE vtables in order -- never reorder or omit methods.
// ---------------------------------------------------------------------------

const
  { CMF_VERSION }
  // (MF_SDK_VERSION shl 16) or MF_API_VERSION
  CMF_VERSION = $00020070;

  CMFSTARTUP_FULL = 0;

  CMF_SOURCE_READER_FIRST_VIDEO_STREAM = $FFFFFFFC;
  CMF_SOURCE_READER_MEDIASOURCE = $FFFFFFFF;
  CMF_SOURCE_READERF_ENDOFSTREAM = $00000002;

  CMF_MT_MAJOR_TYPE: TGUID = '{48EBA18E-F8C9-4687-BF11-0A74C9F96A8F}';
  CMF_MT_SUBTYPE: TGUID = '{F7E34C9A-42E8-4714-B74B-CB29D72C35E5}';
  CMFMediaType_Video: TGUID = '{73646976-0000-0010-8000-00AA00389B71}';

  CMFVideoFormat_RGB32: TGUID = '{00000016-0000-0010-8000-00AA00389B71}';

  CMF_MT_FRAME_SIZE: TGUID = '{1652C33D-D6B2-4012-B834-72030849A37D}';
  CMF_MT_DEFAULT_STRIDE: TGUID = '{644B4E48-1E02-4516-B0EB-C01CA9D49AC6}';

  CMF_SOURCE_READER_ENABLE_VIDEO_PROCESSING: TGUID =
    '{FB394F3D-CCF1-42EE-BBB3-F9B845D5681D}';

  CMF_PD_DURATION: TGUID = '{6C990D33-BB8E-477A-8598-0D5D96FCD88A}';

  { CGUID_NULL }
  CGUID_NULL: TGUID = '{00000000-0000-0000-0000-000000000000}';

type
  { IMFAttributes }
  // mfobjects.h -- complete vtable, do not reorder
  IMFAttributes = interface(IUnknown)
    ['{2CD2D921-C447-44A7-A13C-4ADABFC247E3}']
    function GetItem(const AKey: TGUID; AValue: Pointer): HRESULT; stdcall;
    function GetItemType(const AKey: TGUID; out AType: UInt32): HRESULT; stdcall;
    function CompareItem(const AKey: TGUID; const AValue: TPropVariant;
      out AResult: BOOL): HRESULT; stdcall;
    function Compare(const ATheirs: IMFAttributes; const AMatchType: UInt32;
      out AResult: BOOL): HRESULT; stdcall;
    function GetUINT32(const AKey: TGUID; out AValue: UInt32): HRESULT; stdcall;
    function GetUINT64(const AKey: TGUID; out AValue: UInt64): HRESULT; stdcall;
    function GetDouble(const AKey: TGUID; out AValue: Double): HRESULT; stdcall;
    function GetGUID(const AKey: TGUID; out AValue: TGUID): HRESULT; stdcall;
    function GetStringLength(const AKey: TGUID;
      out ALength: UInt32): HRESULT; stdcall;
    function GetString(const AKey: TGUID; AValue: PWideChar;
      const ASize: UInt32; ALength: PCardinal): HRESULT; stdcall;
    function GetAllocatedString(const AKey: TGUID; out AValue: PWideChar;
      out ALength: UInt32): HRESULT; stdcall;
    function GetBlobSize(const AKey: TGUID;
      out ASize: UInt32): HRESULT; stdcall;
    function GetBlob(const AKey: TGUID; ABuf: Pointer;
      const ASize: UInt32; ARead: PCardinal): HRESULT; stdcall;
    function GetAllocatedBlob(const AKey: TGUID; out ABuf: Pointer;
      out ASize: UInt32): HRESULT; stdcall;
    function GetUnknown(const AKey: TGUID; const AIid: TGUID;
      out AObj): HRESULT; stdcall;
    function SetItem(const AKey: TGUID;
      const AValue: TPropVariant): HRESULT; stdcall;
    function DeleteItem(const AKey: TGUID): HRESULT; stdcall;
    function DeleteAllItems(): HRESULT; stdcall;
    function SetUINT32(const AKey: TGUID;
      const AValue: UInt32): HRESULT; stdcall;
    function SetUINT64(const AKey: TGUID;
      const AValue: UInt64): HRESULT; stdcall;
    function SetDouble(const AKey: TGUID;
      const AValue: Double): HRESULT; stdcall;
    function SetGUID(const AKey: TGUID;
      const AValue: TGUID): HRESULT; stdcall;
    function SetString(const AKey: TGUID;
      const AValue: PWideChar): HRESULT; stdcall;
    function SetBlob(const AKey: TGUID; const ABuf: Pointer;
      const ASize: UInt32): HRESULT; stdcall;
    function SetUnknown(const AKey: TGUID;
      const AUnknown: IUnknown): HRESULT; stdcall;
    function LockStore(): HRESULT; stdcall;
    function UnlockStore(): HRESULT; stdcall;
    function GetCount(out ACount: UInt32): HRESULT; stdcall;
    function GetItemByIndex(const AIndex: UInt32; out AKey: TGUID;
      AValue: Pointer): HRESULT; stdcall;
    function CopyAllItems(const ADest: IMFAttributes): HRESULT; stdcall;
  end;

  { IMFMediaType }
  IMFMediaType = interface(IMFAttributes)
    ['{44AE0FA8-EA31-4109-8D2E-4CAE4997C555}']
    function GetMajorType(out AGuid: TGUID): HRESULT; stdcall;
    function IsCompressedFormat(out ACompressed: BOOL): HRESULT; stdcall;
    function IsEqual(const AType: IMFMediaType;
      out AFlags: UInt32): HRESULT; stdcall;
    function GetRepresentation(const AGuid: TGUID;
      out ARepresentation: Pointer): HRESULT; stdcall;
    function FreeRepresentation(const AGuid: TGUID;
      const ARepresentation: Pointer): HRESULT; stdcall;
  end;

  { IMFMediaBuffer }
  IMFMediaBuffer = interface(IUnknown)
    ['{045FA593-8799-42B8-BC8D-8968C6453507}']
    function Lock(out ABuffer: PByte; AMaxLength: PCardinal;
      ACurrentLength: PCardinal): HRESULT; stdcall;
    function Unlock(): HRESULT; stdcall;
    function GetCurrentLength(out ALength: UInt32): HRESULT; stdcall;
    function SetCurrentLength(const ALength: UInt32): HRESULT; stdcall;
    function GetMaxLength(out ALength: UInt32): HRESULT; stdcall;
  end;

  { IMFSample }
  IMFSample = interface(IMFAttributes)
    ['{C40A00F2-B93A-4D80-AE8C-5A1C634F58E4}']
    function GetSampleFlags(out AFlags: UInt32): HRESULT; stdcall;
    function SetSampleFlags(const AFlags: UInt32): HRESULT; stdcall;
    function GetSampleTime(out ATime: Int64): HRESULT; stdcall;
    function SetSampleTime(const ATime: Int64): HRESULT; stdcall;
    function GetSampleDuration(out ADuration: Int64): HRESULT; stdcall;
    function SetSampleDuration(const ADuration: Int64): HRESULT; stdcall;
    function GetBufferCount(out ACount: UInt32): HRESULT; stdcall;
    function GetBufferByIndex(const AIndex: UInt32;
      out ABuffer: IMFMediaBuffer): HRESULT; stdcall;
    function ConvertToContiguousBuffer(
      out ABuffer: IMFMediaBuffer): HRESULT; stdcall;
    function AddBuffer(const ABuffer: IMFMediaBuffer): HRESULT; stdcall;
    function RemoveBufferByIndex(const AIndex: UInt32): HRESULT; stdcall;
    function RemoveAllBuffers(): HRESULT; stdcall;
    function GetTotalLength(out ALength: UInt32): HRESULT; stdcall;
    function CopyToBuffer(const ABuffer: IMFMediaBuffer): HRESULT; stdcall;
  end;

  { IMFSourceReader }
  IMFSourceReader = interface(IUnknown)
    ['{70AE66F2-C809-4E4F-8915-BDCB406B7993}']
    function GetStreamSelection(const AStreamIndex: UInt32;
      out ASelected: BOOL): HRESULT; stdcall;
    function SetStreamSelection(const AStreamIndex: UInt32;
      const ASelected: BOOL): HRESULT; stdcall;
    function GetNativeMediaType(const AStreamIndex: UInt32;
      const ATypeIndex: UInt32; out AType: IMFMediaType): HRESULT; stdcall;
    function GetCurrentMediaType(const AStreamIndex: UInt32;
      out AType: IMFMediaType): HRESULT; stdcall;
    function SetCurrentMediaType(const AStreamIndex: UInt32;
      AReserved: PCardinal; const AType: IMFMediaType): HRESULT; stdcall;
    function SetCurrentPosition(const AGuidTimeFormat: TGUID;
      const APosition: TPropVariant): HRESULT; stdcall;
    function ReadSample(const AStreamIndex: UInt32; const AFlags: UInt32;
      AActualStreamIndex: PCardinal; AStreamFlags: PCardinal;
      ATimestamp: PInt64; out ASample: IMFSample): HRESULT; stdcall;
    function Flush(const AStreamIndex: UInt32): HRESULT; stdcall;
    function GetServiceForStream(const AStreamIndex: UInt32;
      const AGuidService: TGUID; const AIid: TGUID;
      out AObject): HRESULT; stdcall;
    function GetPresentationAttribute(const AStreamIndex: UInt32;
      const AGuidAttribute: TGUID;
      out AValue: TPropVariant): HRESULT; stdcall;
  end;

function MFStartup(const AVersion: UInt32;
  const AFlags: UInt32): HRESULT; stdcall;
  external 'mfplat.dll' name 'MFStartup';

function MFShutdown(): HRESULT; stdcall;
  external 'mfplat.dll' name 'MFShutdown';

function MFCreateAttributes(out AAttributes: IMFAttributes;
  const AInitialSize: UInt32): HRESULT; stdcall;
  external 'mfplat.dll' name 'MFCreateAttributes';

function MFCreateMediaType(out AType: IMFMediaType): HRESULT; stdcall;
  external 'mfplat.dll' name 'MFCreateMediaType';

function MFCreateSourceReaderFromURL(const AUrl: PWideChar;
  const AAttributes: IMFAttributes;
  out AReader: IMFSourceReader): HRESULT; stdcall;
  external 'mfreadwrite.dll' name 'MFCreateSourceReaderFromURL';

{ TVideo }

function TVideo.ExtractFrames(const AFileName: string;
  const ANumFrames: Integer; out AFrames: TArray<TVideoFrame>): Boolean;
var
  LComInit: HRESULT;
  LMfStarted: Boolean;
  LAttr: IMFAttributes;
  LReader: IMFSourceReader;
  LType: IMFMediaType;
  LCurType: IMFMediaType;
  LHr: HRESULT;
  LFrameSize: UInt64;
  LWidth: Integer;
  LHeight: Integer;
  LStrideU: UInt32;
  LStride: Integer;
  LVarDur: TPropVariant;
  LVarPos: TPropVariant;
  LDuration: Int64;
  LSample: IMFSample;
  LBuffer: IMFMediaBuffer;
  LPtr: PByte;
  LFlags: UInt32;
  LTimestamp: Int64;
  LTargetTime: Int64;
  LI: Integer;
  LRow: PByte;
  LDst: Integer;
  LGot: Integer;

  function ReadOneFrame(out AImage: TRGBImage;
    out ATimeSec: Double): Boolean;
  var
    LY: Integer;
    LX: Integer;
  begin
    Result := False;
    AImage.Width := 0;
    AImage.Height := 0;
    AImage.Pixels := nil;
    ATimeSec := 0.0;

    LSample := nil;
    LFlags := 0;
    LTimestamp := 0;
    // Loop past gaps/stream ticks until a real sample or end of stream
    repeat
      LHr := LReader.ReadSample(CMF_SOURCE_READER_FIRST_VIDEO_STREAM, 0,
        nil, @LFlags, @LTimestamp, LSample);
      if Failed(LHr) then
        Exit;
      if (LFlags and CMF_SOURCE_READERF_ENDOFSTREAM) <> 0 then
        Exit;
    until LSample <> nil;

    LBuffer := nil;
    if Failed(LSample.ConvertToContiguousBuffer(LBuffer)) then
      Exit;
    LPtr := nil;
    if Failed(LBuffer.Lock(LPtr, nil, nil)) then
      Exit;
    try
      AImage.Width := LWidth;
      AImage.Height := LHeight;
      SetLength(AImage.Pixels, LWidth * LHeight * 3);
      for LY := 0 to LHeight - 1 do
      begin
        // MF RGB32: positive stride = top-down rows from the start;
        // negative stride = bottom-up (first displayed row is last in
        // memory) -- both handled via signed row stepping
        if LStride >= 0 then
          LRow := LPtr + Int64(LY) * LStride
        else
          LRow := LPtr + Int64(LHeight - 1) * (-LStride) + Int64(LY) * LStride;
        LDst := LY * LWidth * 3;
        for LX := 0 to LWidth - 1 do
        begin
          // RGB32 memory bytes: B, G, R, X
          AImage.Pixels[LDst + LX * 3 + 0] := PByte(LRow + LX * 4 + 2)^;
          AImage.Pixels[LDst + LX * 3 + 1] := PByte(LRow + LX * 4 + 1)^;
          AImage.Pixels[LDst + LX * 3 + 2] := PByte(LRow + LX * 4 + 0)^;
        end;
      end;
    finally
      LBuffer.Unlock();
    end;
    ATimeSec := LTimestamp / 10000000.0;
    Result := True;
  end;

begin
  Result := False;
  AFrames := nil;

  if not FileExists(AFileName) then
  begin
    GetErrors().Add(esError, CVID_ERR_OPEN, 'File not found: ' + AFileName);
    Exit;
  end;
  if ANumFrames <= 0 then
  begin
    GetErrors().Add(esError, CVID_ERR_OPEN, 'Invalid frame count');
    Exit;
  end;

  // COM + Media Foundation lifetime is scoped to this call
  LComInit := CoInitializeEx(nil, COINIT_MULTITHREADED);
  LMfStarted := False;
  try
    LHr := MFStartup(CMF_VERSION, CMFSTARTUP_FULL);
    if Failed(LHr) then
    begin
      GetErrors().Add(esError, CVID_ERR_OPEN, 'MFStartup failed: %x', [LHr]);
      Exit;
    end;
    LMfStarted := True;

    // Enable the source reader's video processor so any decodable input
    // is color-converted to RGB32 for us
    if Failed(MFCreateAttributes(LAttr, 1)) then
      Exit;
    LAttr.SetUINT32(CMF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, 1);

    LHr := MFCreateSourceReaderFromURL(PWideChar(AFileName), LAttr, LReader);
    if Failed(LHr) then
    begin
      GetErrors().Add(esError, CVID_ERR_OPEN,
        'Cannot open video (hr=%x): %s', [LHr, AFileName]);
      Exit;
    end;

    if Failed(MFCreateMediaType(LType)) then
      Exit;
    LType.SetGUID(CMF_MT_MAJOR_TYPE, CMFMediaType_Video);
    LType.SetGUID(CMF_MT_SUBTYPE, CMFVideoFormat_RGB32);
    LHr := LReader.SetCurrentMediaType(CMF_SOURCE_READER_FIRST_VIDEO_STREAM,
      nil, LType);
    if Failed(LHr) then
    begin
      GetErrors().Add(esError, CVID_ERR_OPEN,
        'No RGB32-convertible video stream (hr=%x)', [LHr]);
      Exit;
    end;

    // Actual output frame size + stride
    if Failed(LReader.GetCurrentMediaType(
      CMF_SOURCE_READER_FIRST_VIDEO_STREAM, LCurType)) then
      Exit;
    if Failed(LCurType.GetUINT64(CMF_MT_FRAME_SIZE, LFrameSize)) then
      Exit;
    LWidth := Integer(LFrameSize shr 32);
    LHeight := Integer(LFrameSize and $FFFFFFFF);
    if Failed(LCurType.GetUINT32(CMF_MT_DEFAULT_STRIDE, LStrideU)) then
      LStride := LWidth * 4
    else
      LStride := Int32(LStrideU);
    if (LWidth <= 0) or (LHeight <= 0) then
    begin
      GetErrors().Add(esError, CVID_ERR_OPEN, 'Bad frame size %dx%d',
        [LWidth, LHeight]);
      Exit;
    end;

    // Duration (100 ns units) for uniform sampling
    FillChar(LVarDur, SizeOf(LVarDur), 0);
    LHr := LReader.GetPresentationAttribute(CMF_SOURCE_READER_MEDIASOURCE,
      CMF_PD_DURATION, LVarDur);
    if Failed(LHr) then
    begin
      GetErrors().Add(esError, CVID_ERR_OPEN, 'No duration (hr=%x)', [LHr]);
      Exit;
    end;
    LDuration := Int64(LVarDur.hVal.QuadPart);
    PropVariantClear(LVarDur);
    if LDuration <= 0 then
    begin
      GetErrors().Add(esError, CVID_ERR_OPEN, 'Empty video');
      Exit;
    end;

    Status('Video: %dx%d, %.2f s -> sampling %d frames',
      [LWidth, LHeight, LDuration / 10000000.0, ANumFrames]);

    SetLength(AFrames, ANumFrames);
    LGot := 0;
    for LI := 0 to ANumFrames - 1 do
    begin
      // Uniform mid-interval timestamps: (i + 0.5) * duration / n
      LTargetTime := Round((LI + 0.5) * (LDuration / ANumFrames));
      if LTargetTime >= LDuration then
        LTargetTime := LDuration - 1;

      FillChar(LVarPos, SizeOf(LVarPos), 0);
      LVarPos.vt := VT_I8;
      LVarPos.hVal.QuadPart := LTargetTime;
      LHr := LReader.SetCurrentPosition(CGUID_NULL, LVarPos);
      if Failed(LHr) then
      begin
        GetErrors().Add(esError, CVID_ERR_READ, 'Seek failed (hr=%x)', [LHr]);
        Exit;
      end;

      if not ReadOneFrame(AFrames[LGot].Image,
        AFrames[LGot].TimestampSec) then
      begin
        // Seek near the tail can run off the end; stop with what we have
        Break;
      end;
      // Use the uniform sampling position as the frame timestamp: HF
      // derives timestamps from sample indices, and the source reader's
      // video processor MFT is known to zero decoded sample times
      AFrames[LGot].TimestampSec := LTargetTime / 10000000.0;
      Inc(LGot);
    end;

    if LGot = 0 then
    begin
      GetErrors().Add(esError, CVID_ERR_READ, 'No frames decoded');
      SetLength(AFrames, 0);
      Exit;
    end;
    SetLength(AFrames, LGot);
    Status('Video frames extracted: %d', [LGot]);
    Result := True;
  finally
    LSample := nil;
    LBuffer := nil;
    LCurType := nil;
    LType := nil;
    LReader := nil;
    LAttr := nil;
    if LMfStarted then
      MFShutdown();
    if (LComInit = S_OK) or (LComInit = S_FALSE) then
      CoUninitialize();
  end;
end;

end.
