{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information

 -------------------------------------------------------------------------------

  Gemma4.Tools.Utils - Meta-tools, standard tool catalog, REST client, local ops

  Ready-to-use tools and helpers built on Gemma4.Tools:
    - TToolResponse: unified REST/local result (success, status, body Json())
    - TRestRequest: minimal HTTP client for tool implementations
    - TLocalOps: filesystem and script helpers (python via res\python)
    - Standard tools: ToolGeocode, ToolWeather, ToolWebSearch, ToolRunScript
    - RegisterMetaTools (find_tool/use_tool) and RegisterStandardTools catalogs
    - web_search uses the Tavily API (optional TAVILY_API_KEY)

  Dependencies: System.SysUtils, System.Classes, StdApp.Base, StdApp.JSON,
    Gemma4.Tools
===============================================================================}

unit Gemma4.Tools.Utils;

{$I StdApp.Defines.inc}

interface

uses
  System.SysUtils,
  System.Classes,
  StdApp.Base,
  StdApp.JSON,
  Gemma4.Tools;

const
  { Error codes }
  ERR_REST_REQUEST = 'TLU001';

  { Resource paths -- relative to app base folder }
  CResPythonExe          = 'res\python\python.exe';
  CResPythonSitePackages = 'res\python\Lib\site-packages';
  CResScripts            = 'res\scripts';

type
  { TToolResponse }
  // Unified response for REST and local operations -- caller-managed
  // (try/finally). Json() exposes the parsed body for deep access.
  TToolResponse = class(TBaseObject)
  private
    FSuccess: Boolean;
    FStatusCode: Integer;
    FErrorMsg: string;
    FRawJson: string;
    FJson: TJSON;
    procedure DoParseRaw();
  public
    constructor Create(); override;
    destructor Destroy(); override;
    procedure SetResult(const ASuccess: Boolean; const AStatusCode: Integer;
      const AErrorMsg: string; const ARawJson: string);
    function Success(): Boolean;
    function StatusCode(): Integer;
    function ErrorMsg(): string;
    function RawJson(): string;
    // Parsed JSON root, or nil when the body was not valid JSON. Owned
    // by this object -- borrow, do not free.
    function Json(): TJSON;
    // Top-level key conveniences
    function AsString(const AKey: string; const ADefault: string = ''): string;
    function AsInteger(const AKey: string; const ADefault: Integer = 0): Integer;
    function AsDouble(const AKey: string; const ADefault: Double = 0.0): Double;
    function AsBoolean(const AKey: string; const ADefault: Boolean = False): Boolean;
  end;

  { TRestRequest }
  // Fluent REST client -- wraps THTTPClient; caller owns request and
  // response objects
  TRestRequest = class(TBaseObject)
  private
    FBaseUrl: string;
    FHeaders: TStringList;
    FQueryParams: TStringList;
    FBody: string;
    function BuildUrl(): string;
    function DoRequest(const AMethod: string): TToolResponse;
  public
    constructor Create(); override;
    destructor Destroy(); override;
    procedure SetBaseUrl(const AUrl: string);
    function Header(const AKey: string; const AValue: string): TRestRequest;
    function Query(const AKey: string; const AValue: string): TRestRequest;
    function Body(const AJson: string): TRestRequest;
    function Bearer(const AToken: string): TRestRequest;
    function Get(): TToolResponse;
    function Post(): TToolResponse;
    function Put(): TToolResponse;
    function Delete(): TToolResponse;
  end;

  { TLocalOps }
  // Local filesystem and command operations -- class functions. RunCmd
  // executes arbitrary shell commands; only expose it to a model
  // deliberately.
  TLocalOps = class
    class function DirList(const APath: string;
      const APattern: string = '*'): TToolResponse;
    class function ReadFile(const APath: string;
      const AMaxChars: Integer = 8000): TToolResponse;
    class function WriteFile(const APath: string;
      const AContent: string): TToolResponse;
    class function FileInfo(const APath: string): TToolResponse;
    class function RunCmd(const ACommand: string): TToolResponse;
  end;

// Factory -- creates TRestRequest with base URL set
function RestRequest(const ABaseUrl: string): TRestRequest;

// Geocode a free-form location string -> lat/lon via Nominatim
// (OpenStreetMap). Returns True if resolved.
function ToolGeocode(const ALocation: string;
  out ALat: Double; out ALon: Double): Boolean;

// Current weather for a location name (Nominatim geocode + Open-Meteo).
// Caller frees result. Keys: location, temperature_f, conditions,
// wind_mph, humidity_pct
function ToolWeather(const ALocation: string): TToolResponse;

// Current local time formatted for display (e.g. "4:31 PM")
function ToolTimeNow(): string;

// Current local date formatted for display (e.g. "2026-06-11")
function ToolDateNow(): string;

// Web search via Tavily API. Caller frees result. Optional ATopic
// ('news'/'finance') and ATimeRange ('day'/'week'/'month'/'year')
// refine the search; empty omits them. Result is curated for model
// consumption. Keys: answer, results[N].title, results[N].url,
// results[N].content, results[N].score
function ToolWebSearch(const AApiKey: string; const AQuery: string;
  const AMaxResults: Integer = 1; const ATopic: string = '';
  const ATimeRange: string = '';
  const AMaxContentChars: Integer = 800): TToolResponse;

// Execute a Python script via the bundled embeddable interpreter.
// Writes ACode to a uniquely-named .py file in AScriptsDir, runs it
// with APythonExe, captures combined stdout+stderr. Caller frees
// result. Keys: output, exit_code, script_path
function ToolRunScript(const APythonExe: string; const AScriptsDir: string;
  const ACode: string): TToolResponse;

// Register the three bootstrap meta-tools (find_tool, use_tool,
// run_script) into ABootstrap. Real tools live in ACatalog -- searched
// by find_tool, executed by use_tool. run_script executes Python via
// the bundled interpreter at APythonExe, writing scripts to AScriptsDir.
procedure RegisterMetaTools(const ABootstrap: TToolRegistry;
  const ACatalog: TToolRegistry; const APythonExe: string;
  const AScriptsDir: string);

// Register the standard tool catalog (get_time, get_date, get_weather,
// web_search, dir_list, file_info, read_file, pip_install) into
// ARegistry. Each tool manages its own configuration (API keys, paths)
// internally.
procedure RegisterStandardTools(const ARegistry: TToolRegistry);

implementation

uses
  System.Types,
  System.IOUtils,
  System.DateUtils,
  System.Net.HttpClient,
  System.Net.URLClient,
  System.NetEncoding,
  Winapi.Windows,
  System.Hash,
  StdApp.Utils,
  StdApp.Resources;

{ RestRequest }

function RestRequest(const ABaseUrl: string): TRestRequest;
begin
  Result := TRestRequest.Create();
  Result.SetBaseUrl(ABaseUrl);
end;

{ TToolResponse }

constructor TToolResponse.Create();
begin
  inherited Create();
  FSuccess := False;
  FStatusCode := 0;
  FErrorMsg := '';
  FRawJson := '';
  FJson := nil;
end;

destructor TToolResponse.Destroy();
begin
  FJson.Free();
  inherited Destroy();
end;

procedure TToolResponse.SetResult(const ASuccess: Boolean;
  const AStatusCode: Integer; const AErrorMsg: string; const ARawJson: string);
begin
  FSuccess := ASuccess;
  FStatusCode := AStatusCode;
  FErrorMsg := AErrorMsg;
  FRawJson := ARawJson;
  FreeAndNil(FJson);
  DoParseRaw();
end;

procedure TToolResponse.DoParseRaw();
begin
  if FRawJson.Trim() = '' then
    Exit;
  try
    FJson := TJSON.FromString(FRawJson);
  except
    // Non-JSON response body -- RawJson() still works, Json() stays nil
    FJson := nil;
  end;
end;

function TToolResponse.Success(): Boolean;
begin
  Result := FSuccess;
end;

function TToolResponse.StatusCode(): Integer;
begin
  Result := FStatusCode;
end;

function TToolResponse.ErrorMsg(): string;
begin
  Result := FErrorMsg;
end;

function TToolResponse.RawJson(): string;
begin
  Result := FRawJson;
end;

function TToolResponse.Json(): TJSON;
begin
  Result := FJson;
end;

function TToolResponse.AsString(const AKey: string; const ADefault: string): string;
begin
  Result := ADefault;
  if (FJson <> nil) and FJson.Has(AKey) then
    Result := FJson.Get(AKey).AsString(ADefault);
end;

function TToolResponse.AsInteger(const AKey: string; const ADefault: Integer): Integer;
begin
  Result := ADefault;
  if (FJson <> nil) and FJson.Has(AKey) then
    Result := FJson.Get(AKey).AsInt32(ADefault);
end;

function TToolResponse.AsDouble(const AKey: string; const ADefault: Double): Double;
begin
  Result := ADefault;
  if (FJson <> nil) and FJson.Has(AKey) then
    Result := FJson.Get(AKey).AsDouble(ADefault);
end;

function TToolResponse.AsBoolean(const AKey: string; const ADefault: Boolean): Boolean;
begin
  Result := ADefault;
  if (FJson <> nil) and FJson.Has(AKey) then
    Result := FJson.Get(AKey).AsBoolean(ADefault);
end;

{ TRestRequest }

constructor TRestRequest.Create();
begin
  inherited Create();
  FHeaders := TStringList.Create();
  FQueryParams := TStringList.Create();
  FBody := '';
  FBaseUrl := '';
  FHeaders.Values['User-Agent'] := 'Gemma4/1.0';
end;

destructor TRestRequest.Destroy();
begin
  FQueryParams.Free();
  FHeaders.Free();
  inherited Destroy();
end;

procedure TRestRequest.SetBaseUrl(const AUrl: string);
begin
  FBaseUrl := AUrl;
end;

function TRestRequest.Header(const AKey: string; const AValue: string): TRestRequest;
begin
  FHeaders.Values[AKey] := AValue;
  Result := Self;
end;

function TRestRequest.Query(const AKey: string; const AValue: string): TRestRequest;
begin
  FQueryParams.Values[AKey] := AValue;
  Result := Self;
end;

function TRestRequest.Body(const AJson: string): TRestRequest;
begin
  FBody := AJson;
  Result := Self;
end;

function TRestRequest.Bearer(const AToken: string): TRestRequest;
begin
  FHeaders.Values['Authorization'] := 'Bearer ' + AToken;
  Result := Self;
end;

function TRestRequest.BuildUrl(): string;
var
  LI: Integer;
  LSep: string;
begin
  Result := FBaseUrl;
  if FQueryParams.Count > 0 then
  begin
    if Pos('?', Result) > 0 then
      LSep := '&'
    else
      LSep := '?';
    for LI := 0 to FQueryParams.Count - 1 do
    begin
      Result := Result + LSep +
        TNetEncoding.URL.Encode(FQueryParams.Names[LI]) + '=' +
        TNetEncoding.URL.Encode(FQueryParams.ValueFromIndex[LI]);
      LSep := '&';
    end;
  end;
end;

function TRestRequest.DoRequest(const AMethod: string): TToolResponse;
var
  LClient: THTTPClient;
  LHttpResponse: IHTTPResponse;
  LUrl: string;
  LI: Integer;
  LBodyStream: TStringStream;
  LResponseBody: string;
  LStatusCode: Integer;
begin
  Result := TToolResponse.Create();
  LUrl := BuildUrl();

  LClient := THTTPClient.Create();
  try
    // Apply headers
    for LI := 0 to FHeaders.Count - 1 do
      LClient.CustomHeaders[FHeaders.Names[LI]] := FHeaders.ValueFromIndex[LI];

    try
      if (AMethod = 'GET') or (AMethod = 'DELETE') then
      begin
        if AMethod = 'GET' then
          LHttpResponse := LClient.Get(LUrl)
        else
          LHttpResponse := LClient.Delete(LUrl);
      end
      else
      begin
        // POST or PUT with body
        if FHeaders.IndexOfName('Content-Type') < 0 then
          LClient.CustomHeaders['Content-Type'] := 'application/json';

        LBodyStream := TStringStream.Create(FBody, TEncoding.UTF8);
        try
          if AMethod = 'POST' then
            LHttpResponse := LClient.Post(LUrl, LBodyStream)
          else
            LHttpResponse := LClient.Put(LUrl, LBodyStream);
        finally
          LBodyStream.Free();
        end;
      end;

      LStatusCode := LHttpResponse.StatusCode;
      LResponseBody := LHttpResponse.ContentAsString();
      Result.SetResult(
        (LStatusCode >= 200) and (LStatusCode < 300),
        LStatusCode,
        '',
        LResponseBody
      );
    except
      on E: Exception do
      begin
        FErrors.Add(esError, ERR_REST_REQUEST, RSRestRequestFailed,
          [AMethod, LUrl, E.Message]);
        Result.SetResult(False, 0, E.Message, '');
      end;
    end;
  finally
    LClient.Free();
  end;
end;

function TRestRequest.Get(): TToolResponse;
begin
  Result := DoRequest('GET');
end;

function TRestRequest.Post(): TToolResponse;
begin
  Result := DoRequest('POST');
end;

function TRestRequest.Put(): TToolResponse;
begin
  Result := DoRequest('PUT');
end;

function TRestRequest.Delete(): TToolResponse;
begin
  Result := DoRequest('DELETE');
end;

{ TLocalOps }

class function TLocalOps.DirList(const APath: string;
  const APattern: string): TToolResponse;
var
  LFiles: TStringDynArray;
  LListing: string;
  LI: Integer;
begin
  Result := TToolResponse.Create();
  try
    if not TDirectory.Exists(APath) then
    begin
      Result.SetResult(False, 0, 'Directory not found: ' + APath, '');
      Exit;
    end;
    LFiles := TDirectory.GetFiles(APath, APattern);
    LListing := '';
    for LI := 0 to Length(LFiles) - 1 do
    begin
      if LListing <> '' then
        LListing := LListing + #10;
      LListing := LListing + ExtractFileName(LFiles[LI]);
    end;
    Result.SetResult(True, 0, '',
      '{"listing":' + ToolJsonStr(LListing) +
      ',"count":' + IntToStr(Length(LFiles)) + '}');
  except
    on E: Exception do
      Result.SetResult(False, 0, E.Message, '');
  end;
end;

class function TLocalOps.ReadFile(const APath: string;
  const AMaxChars: Integer): TToolResponse;
var
  LContent: string;
  LFullLen: Integer;
  LTruncated: Boolean;
  LTruncStr: string;
begin
  Result := TToolResponse.Create();
  try
    if not TFile.Exists(APath) then
    begin
      Result.SetResult(False, 0, 'File not found: ' + APath, '');
      Exit;
    end;
    LContent := TFile.ReadAllText(APath, TEncoding.UTF8);

    // Guard the context window -- an uncapped read of a large file can
    // flood the model with more text than its entire context holds.
    // AMaxChars <= 0 disables the guard.
    LFullLen := Length(LContent);
    LTruncated := (AMaxChars > 0) and (LFullLen > AMaxChars);
    if LTruncated then
      LContent := Copy(LContent, 1, AMaxChars);
    if LTruncated then
      LTruncStr := 'true'
    else
      LTruncStr := 'false';

    Result.SetResult(True, 0, '',
      '{"content":' + ToolJsonStr(LContent) +
      ',"size":' + IntToStr(LFullLen) +
      ',"truncated":' + LTruncStr + '}');
  except
    on E: Exception do
      Result.SetResult(False, 0, E.Message, '');
  end;
end;

class function TLocalOps.WriteFile(const APath: string;
  const AContent: string): TToolResponse;
begin
  Result := TToolResponse.Create();
  try
    TFile.WriteAllText(APath, AContent, TEncoding.UTF8);
    Result.SetResult(True, 0, '',
      '{"written":true,"size":' + IntToStr(Length(AContent)) + '}');
  except
    on E: Exception do
      Result.SetResult(False, 0, E.Message, '');
  end;
end;

class function TLocalOps.FileInfo(const APath: string): TToolResponse;
var
  LSize: Int64;
  LModified: TDateTime;
  LModStr: string;
begin
  Result := TToolResponse.Create();
  try
    if not TFile.Exists(APath) then
    begin
      Result.SetResult(False, 0, 'File not found: ' + APath, '');
      Exit;
    end;
    LSize := TFile.GetSize(APath);
    LModified := TFile.GetLastWriteTime(APath);
    LModStr := DateToISO8601(LModified, False);
    Result.SetResult(True, 0, '',
      '{"exists":true' +
      ',"size":' + IntToStr(LSize) +
      ',"modified":' + ToolJsonStr(LModStr) +
      ',"name":' + ToolJsonStr(ExtractFileName(APath)) + '}');
  except
    on E: Exception do
      Result.SetResult(False, 0, E.Message, '');
  end;
end;

class function TLocalOps.RunCmd(const ACommand: string): TToolResponse;
var
  LSA: TSecurityAttributes;
  LReadPipe: THandle;
  LWritePipe: THandle;
  LSI: TStartupInfo;
  LPI: TProcessInformation;
  LCmdLine: string;
  LBytesRead: DWORD;
  LBuffer: array[0..4095] of AnsiChar;
  LOutput: AnsiString;
  LChunk: AnsiString;
  LExitCode: DWORD;
begin
  Result := TToolResponse.Create();

  LSA.nLength := SizeOf(TSecurityAttributes);
  LSA.bInheritHandle := True;
  LSA.lpSecurityDescriptor := nil;

  if not CreatePipe(LReadPipe, LWritePipe, @LSA, 0) then
  begin
    Result.SetResult(False, 0, 'Failed to create pipe', '');
    Exit;
  end;

  try
    LSI := Default(TStartupInfo);
    LSI.cb := SizeOf(LSI);
    LSI.hStdOutput := LWritePipe;
    LSI.hStdError := LWritePipe;
    LSI.dwFlags := STARTF_USESTDHANDLES or STARTF_USESHOWWINDOW;
    LSI.wShowWindow := SW_HIDE;

    LCmdLine := 'cmd.exe /c ' + ACommand;
    LPI := Default(TProcessInformation);

    if not CreateProcess(nil, PChar(LCmdLine), nil, nil, True,
      CREATE_NO_WINDOW, nil, nil, LSI, LPI) then
    begin
      CloseHandle(LWritePipe);
      CloseHandle(LReadPipe);
      Result.SetResult(False, 0, 'Failed to create process', '');
      Exit;
    end;

    // Close write end in parent so ReadFile will eventually return 0
    CloseHandle(LWritePipe);
    LWritePipe := 0;

    // Read output
    LOutput := '';
    while Winapi.Windows.ReadFile(LReadPipe, LBuffer, SizeOf(LBuffer),
      LBytesRead, nil) and (LBytesRead > 0) do
    begin
      SetString(LChunk, PAnsiChar(@LBuffer[0]), LBytesRead);
      LOutput := LOutput + LChunk;
    end;

    WaitForSingleObject(LPI.hProcess, 5000);
    GetExitCodeProcess(LPI.hProcess, LExitCode);
    CloseHandle(LPI.hProcess);
    CloseHandle(LPI.hThread);

    Result.SetResult(LExitCode = 0, Integer(LExitCode), '',
      '{"output":' + ToolJsonStr(string(LOutput)) +
      ',"exitCode":' + IntToStr(LExitCode) + '}');
  finally
    if LWritePipe <> 0 then
      CloseHandle(LWritePipe);
    CloseHandle(LReadPipe);
  end;
end;

{ ToolGeocode }

function ToolGeocode(const ALocation: string;
  out ALat: Double; out ALon: Double): Boolean;
var
  LReq: TRestRequest;
  LResp: TToolResponse;
  LItems: TArray<TJSON>;
begin
  Result := False;
  ALat := 0;
  ALon := 0;

  LReq := RestRequest('https://nominatim.openstreetmap.org/search');
  try
    LResp := LReq
      .Query('q', ALocation)
      .Query('format', 'json')
      .Query('limit', '1')
      .Get();
    try
      if (not LResp.Success()) or (LResp.Json() = nil) then
        Exit;
      if not LResp.Json().IsArray() then
        Exit;
      LItems := LResp.Json().Items();
      if Length(LItems) = 0 then
        Exit;
      // Nominatim returns lat/lon as strings
      ALat := StrToFloatDef(LItems[0].Get('lat').AsString(), 0,
        TFormatSettings.Invariant);
      ALon := StrToFloatDef(LItems[0].Get('lon').AsString(), 0,
        TFormatSettings.Invariant);
      Result := (ALat <> 0) or (ALon <> 0);
    finally
      LResp.Free();
    end;
  finally
    LReq.Free();
  end;
end;

{ ToolWmoCodeToText }

// Open-Meteo reports conditions as WMO interpretation codes. Decode to
// plain text so the model never sees a bare numeric code.
function ToolWmoCodeToText(const ACode: Integer): string;
begin
  case ACode of
    0:          Result := 'clear sky';
    1:          Result := 'mainly clear';
    2:          Result := 'partly cloudy';
    3:          Result := 'overcast';
    45, 48:     Result := 'fog';
    51, 53, 55: Result := 'drizzle';
    56, 57:     Result := 'freezing drizzle';
    61, 63, 65: Result := 'rain';
    66, 67:     Result := 'freezing rain';
    71, 73, 75: Result := 'snow';
    77:         Result := 'snow grains';
    80, 81, 82: Result := 'rain showers';
    85, 86:     Result := 'snow showers';
    95:         Result := 'thunderstorm';
    96, 99:     Result := 'thunderstorm with hail';
  else
    Result := 'unknown (code ' + IntToStr(ACode) + ')';
  end;
end;

{ ToolWeather }

function ToolWeather(const ALocation: string): TToolResponse;
var
  LLat: Double;
  LLon: Double;
  LReq: TRestRequest;
  LJson: TJSON;
  LCleanJson: string;
begin
  // Geocode the location first
  if not ToolGeocode(ALocation, LLat, LLon) then
  begin
    Result := TToolResponse.Create();
    Result.SetResult(False, 0, 'Location not found: ' + ALocation, '');
    Exit;
  end;

  // Fetch current weather from Open-Meteo
  LReq := RestRequest('https://api.open-meteo.com/v1/forecast');
  try
    Result := LReq
      .Query('latitude', FormatFloat('0.####', LLat, TFormatSettings.Invariant))
      .Query('longitude', FormatFloat('0.####', LLon, TFormatSettings.Invariant))
      .Query('current',
        'temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m')
      .Query('temperature_unit', 'fahrenheit')
      .Query('wind_speed_unit', 'mph')
      .Get();
  finally
    LReq.Free();
  end;

  // Rewrite the raw Open-Meteo payload into a compact, model-friendly
  // result. The raw body reports conditions as a bare WMO integer
  // (weather_code) and carries coordinate/unit noise -- both invite
  // the model to fabricate. Falls through to the raw payload if the
  // body does not have the expected shape.
  if Result.Success() and (Result.Json() <> nil) and
     Result.Json().Has('current') then
  begin
    LJson := Result.Json();
    LCleanJson :=
      '{"location":' + ToolJsonStr(ALocation) + ',' +
      '"temperature_f":' + FormatFloat('0.#',
        LJson.Get('current.temperature_2m').AsDouble(),
        TFormatSettings.Invariant) + ',' +
      '"conditions":' + ToolJsonStr(ToolWmoCodeToText(
        LJson.Get('current.weather_code').AsInt32())) + ',' +
      '"wind_mph":' + FormatFloat('0.#',
        LJson.Get('current.wind_speed_10m').AsDouble(),
        TFormatSettings.Invariant) + ',' +
      '"humidity_pct":' + IntToStr(
        LJson.Get('current.relative_humidity_2m').AsInt32()) + '}';
    Result.SetResult(True, Result.StatusCode(), '', LCleanJson);
  end;
end;

{ ToolTimeNow / ToolDateNow }

function ToolTimeNow(): string;
begin
  Result := FormatDateTime('h:nn AMPM', Now());
end;

function ToolDateNow(): string;
begin
  Result := FormatDateTime('yyyy-mm-dd', Now());
end;

{ ToolWebSearch }

function ToolWebSearch(const AApiKey: string; const AQuery: string;
  const AMaxResults: Integer; const ATopic: string;
  const ATimeRange: string; const AMaxContentChars: Integer): TToolResponse;
var
  LReq: TRestRequest;
  LBody: string;
  LJson: TJSON;
  LItem: TJSON;
  LContent: string;
  LCleanJson: string;
  LFirst: Boolean;
begin
  if AApiKey = '' then
  begin
    Result := TToolResponse.Create();
    Result.SetResult(False, 0, 'Tavily API key not set', '');
    Exit;
  end;

  LBody :=
    '{"query":' + ToolJsonStr(AQuery) + ',' +
    '"include_answer":"advanced",' +
    '"include_images":false,' +
    '"include_image_descriptions":false,' +
    '"include_raw_content":false,' +
    '"max_results":' + IntToStr(AMaxResults) + ',';

  // Optional refinements -- only sent when supplied
  if ATopic <> '' then
    LBody := LBody + '"topic":' + ToolJsonStr(ATopic) + ',';
  if ATimeRange <> '' then
    LBody := LBody + '"time_range":' + ToolJsonStr(ATimeRange) + ',';

  LBody := LBody +
    '"include_domains":[],' +
    '"exclude_domains":[]}';

  LReq := RestRequest('https://api.tavily.com/search');
  try
    Result := LReq
      .Bearer(AApiKey)
      .Body(LBody)
      .Post();
  finally
    LReq.Free();
  end;

  // Curate the raw Tavily payload before it reaches the model. The raw
  // body carries response_time, image arrays, and untruncated snippets;
  // per Tavily's own skill guidance, raw search output pollutes the
  // context and degrades reasoning. Keep only the AI answer plus
  // title/url/content/score per result, content truncated. Falls
  // through to the raw payload if the body lacks the expected shape.
  if Result.Success() and (Result.Json() <> nil) and
     Result.Json().Has('results') then
  begin
    LJson := Result.Json();
    LCleanJson :=
      '{"answer":' + ToolJsonStr(LJson.Get('answer').AsString()) + ',' +
      '"results":[';
    LFirst := True;
    for LItem in LJson.Get('results').Items() do
    begin
      if not LFirst then
        LCleanJson := LCleanJson + ',';
      LFirst := False;

      LContent := LItem.Get('content').AsString();
      if Length(LContent) > AMaxContentChars then
        LContent := Copy(LContent, 1, AMaxContentChars) + '...';

      LCleanJson := LCleanJson +
        '{"title":' + ToolJsonStr(LItem.Get('title').AsString()) + ',' +
        '"url":' + ToolJsonStr(LItem.Get('url').AsString()) + ',' +
        '"content":' + ToolJsonStr(LContent) + ',' +
        '"score":' + FormatFloat('0.##',
          LItem.Get('score').AsDouble(), TFormatSettings.Invariant) + '}';
    end;
    LCleanJson := LCleanJson + ']}';
    Result.SetResult(True, Result.StatusCode(), '', LCleanJson);
  end;
end;

{ ToolRunScript }

function ToolRunScript(const APythonExe: string; const AScriptsDir: string;
  const ACode: string): TToolResponse;
var
  LHash: string;
  LScriptPath: string;
  LExitCode: DWORD;
  LOutput: TStringBuilder;
begin
  Result := TToolResponse.Create();

  // Hash-based filename -- identical code reuses the same script file
  LHash := Copy(THashSHA2.GetHashString(ACode), 1, 12);
  LScriptPath := IncludeTrailingPathDelimiter(AScriptsDir) + 'task_' + LHash + '.py';

  try
    ForceDirectories(AScriptsDir);
    if not TFile.Exists(LScriptPath) then
      TFile.WriteAllText(LScriptPath, ACode, TEncoding.UTF8);
  except
    on E: Exception do
    begin
      Result.SetResult(False, 0, 'Failed to write script: ' + E.Message, '');
      Exit;
    end;
  end;

  LOutput := TStringBuilder.Create();
  try
    LExitCode := 1;
    try
      TUtils.CaptureConsoleOutput(
        '',
        PChar(APythonExe),
        PChar('"' + LScriptPath + '"'),
        AScriptsDir,
        LExitCode,
        Pointer(LOutput),
        procedure(const ALine: string; const AUserData: Pointer)
        begin
          TStringBuilder(AUserData).AppendLine(ALine);
        end);
    except
      on E: Exception do
      begin
        Result.SetResult(False, 0, 'Failed to execute script: ' + E.Message, '');
        Exit;
      end;
    end;

    Result.SetResult(LExitCode = 0, Integer(LExitCode), '',
      '{"output":' + ToolJsonStr(LOutput.ToString().TrimRight()) +
      ',"exit_code":' + IntToStr(LExitCode) +
      ',"script_path":' + ToolJsonStr(LScriptPath) + '}');
  finally
    LOutput.Free();
  end;
end;

{ RegisterMetaTools }

procedure RegisterMetaTools(const ABootstrap: TToolRegistry;
  const ACatalog: TToolRegistry; const APythonExe: string;
  const AScriptsDir: string);
begin
  // find_tool -- search the catalog by keyword
  ABootstrap.DefineTool('find_tool')
    .Description(RSMetaFindTool)
    .Param('query', tptString,
      'What you are trying to accomplish (for logging; all tools are always returned)', False)
    .OnExecute(
      function(const AToolName: string; const AParams: TToolParams): string
      var
        LDefs: TArray<TToolDef>;
        LDef: TToolDef;
        LParam: TToolParamDef;
        LSb: TStringBuilder;
        LFirst: Boolean;
        LPFirst: Boolean;
      begin
        LDefs := ACatalog.GetDefs();

        LSb := TStringBuilder.Create();
        try
          LSb.Append('{"tools":[');
          LFirst := True;
          for LDef in LDefs do
          begin
            if not LFirst then
              LSb.Append(',');
            LFirst := False;

            LSb.Append('{"name":' + ToolJsonStr(LDef.ToolName));
            LSb.Append(',"description":' + ToolJsonStr(LDef.Description));
            LSb.Append(',"params":[');
            LPFirst := True;
            for LParam in LDef.Params do
            begin
              if not LPFirst then
                LSb.Append(',');
              LPFirst := False;
              LSb.Append('{"name":' + ToolJsonStr(LParam.ParamName));
              LSb.Append(',"type":' + ToolJsonStr(
                ACatalog.ParamTypeToStr(LParam.ParamType)));
              LSb.Append(',"description":' + ToolJsonStr(LParam.Description));
              if LParam.Required then
                LSb.Append(',"required":true')
              else
                LSb.Append(',"required":false');
              LSb.Append('}');
            end;
            LSb.Append(']}');
          end;
          LSb.Append(']}');
          Result := LSb.ToString();
        finally
          LSb.Free();
        end;
      end);

  // use_tool -- execute a catalog tool by name
  ABootstrap.DefineTool('use_tool')
    .Description(RSMetaUseTool)
    .Param('tool', tptString,
      'Name of the tool to execute (from find_tool results)', True)
    .Param('arguments', tptString,
      'JSON object of arguments to pass to the tool, e.g. {"location":"London"}', True)
    .OnExecute(
      function(const AToolName: string; const AParams: TToolParams): string
      var
        LCall: TToolCall;
      begin
        LCall := Default(TToolCall);
        LCall.ToolName := AParams.AsString('tool');
        LCall.Arguments := AParams.AsString('arguments');
        LCall.CallId := '';

        if not ACatalog.HasTool(LCall.ToolName) then
        begin
          Result := ToolResult(['error',
            'Tool not found in catalog: ' + LCall.ToolName +
            '. Use find_tool to discover available tools.']);
          Exit;
        end;

        Result := ACatalog.Execute(LCall);
      end);

  // run_script -- execute Python code
  ABootstrap.DefineTool('run_script')
    .Description(RSMetaRunScript)
    .Param('code', tptString, 'Complete Python source code to execute', True)
    .Param('description', tptString,
      'Brief description of what the script does (for logging)', False)
    .OnExecute(
      function(const AToolName: string; const AParams: TToolParams): string
      var
        LResp: TToolResponse;
      begin
        LResp := ToolRunScript(APythonExe, AScriptsDir,
          AParams.AsString('code'));
        try
          if LResp.Success() then
            Result := LResp.RawJson()
          else
            Result := ToolResult(['error', LResp.ErrorMsg()]);
        finally
          LResp.Free();
        end;
      end);
end;

{ RegisterStandardTools }

procedure RegisterStandardTools(const ARegistry: TToolRegistry);
begin
  ARegistry.DefineTool('get_time')
    .Description('Returns the current local time (hours, minutes, seconds). ' +
      'Use when the user asks what time it is. For the current date, use ' +
      'get_date instead.')
    .OnExecute(
      function(const AToolName: string; const AParams: TToolParams): string
      begin
        Result := ToolResult(['time', ToolTimeNow()]);
      end);

  ARegistry.DefineTool('get_date')
    .Description('Returns the current local date (year, month, day). Use when ' +
      'the user asks about today''s date or the day of the week. For the time ' +
      'of day, use get_time instead.')
    .OnExecute(
      function(const AToolName: string; const AParams: TToolParams): string
      begin
        Result := ToolResult(['date', ToolDateNow()]);
      end);

  ARegistry.DefineTool('get_weather')
    .Description('Gets the current weather conditions for a specific city. ' +
      'Always use this tool for any weather question. Do not use web_search ' +
      'for weather.')
    .Param('location', tptString, 'The city name, e.g. San Francisco', True)
    .OnExecute(
      function(const AToolName: string; const AParams: TToolParams): string
      var
        LResp: TToolResponse;
      begin
        LResp := ToolWeather(AParams.AsString('location'));
        try
          if LResp.Success() then
            Result := LResp.RawJson()
          else
            Result := ToolResult(['error', LResp.ErrorMsg()]);
        finally
          LResp.Free();
        end;
      end);

  ARegistry.DefineTool('web_search')
    .Description('Searches the web and returns up-to-date results with an ' +
      'AI-generated answer. Use for current events, news, recent releases, ' +
      'prices, or any facts that may have changed since your training data. ' +
      'Break complex questions into separate focused searches.')
    .Param('query', tptString,
      'A concise keyword-style search phrase', True)
    .Param('topic', tptString,
      'Result category: "news" for current events, "finance" for financial ' +
      'topics. Omit for general searches', False)
    .Param('time_range', tptString,
      'Restrict to recent results: "day", "week", "month", or "year". Omit ' +
      'for no restriction', False)
    .OnExecute(
      function(const AToolName: string; const AParams: TToolParams): string
      var
        LResp: TToolResponse;
      begin
        LResp := ToolWebSearch(TUtils.GetEnv('TAVILY_API_KEY'),
          AParams.AsString('query'), 3,
          AParams.AsString('topic'), AParams.AsString('time_range'));
        try
          if LResp.Success() then
            Result := LResp.RawJson()
          else
            Result := ToolResult(['error', LResp.ErrorMsg()]);
        finally
          LResp.Free();
        end;
      end);

  ARegistry.DefineTool('dir_list')
    .Description('Lists the file names in a local directory. Returns file ' +
      'names only.')
    .Param('path', tptString,
      'Absolute directory path, e.g. C:\Temp', True)
    .Param('pattern', tptString,
      'Optional wildcard filter, e.g. *.txt. Omit to list all files', False)
    .OnExecute(
      function(const AToolName: string; const AParams: TToolParams): string
      var
        LResp: TToolResponse;
        LPattern: string;
      begin
        LPattern := AParams.AsString('pattern', '*');
        if LPattern = '' then
          LPattern := '*';
        LResp := TLocalOps.DirList(AParams.AsString('path'), LPattern);
        try
          if LResp.Success() then
            Result := LResp.RawJson()
          else
            Result := ToolResult(['error', LResp.ErrorMsg()]);
        finally
          LResp.Free();
        end;
      end);

  ARegistry.DefineTool('file_info')
    .Description('Gets metadata about a local file: size in bytes, last ' +
      'modified time, and name. Does not read contents -- use read_file ' +
      'for that.')
    .Param('path', tptString,
      'Absolute file path, e.g. C:\Temp\notes.txt', True)
    .OnExecute(
      function(const AToolName: string; const AParams: TToolParams): string
      var
        LResp: TToolResponse;
      begin
        LResp := TLocalOps.FileInfo(AParams.AsString('path'));
        try
          if LResp.Success() then
            Result := LResp.RawJson()
          else
            Result := ToolResult(['error', LResp.ErrorMsg()]);
        finally
          LResp.Free();
        end;
      end);

  ARegistry.DefineTool('read_file')
    .Description('Reads the text content of a local file. For size or date ' +
      'only, use file_info instead.')
    .Param('path', tptString,
      'Absolute file path, e.g. C:\Temp\notes.txt', True)
    .OnExecute(
      function(const AToolName: string; const AParams: TToolParams): string
      var
        LResp: TToolResponse;
      begin
        LResp := TLocalOps.ReadFile(AParams.AsString('path'));
        try
          if LResp.Success() then
            Result := LResp.RawJson()
          else
            Result := ToolResult(['error', LResp.ErrorMsg()]);
        finally
          LResp.Free();
        end;
      end);

  ARegistry.DefineTool('pip_install')
    .Description('Installs a Python package into the embedded Python ' +
      'environment. Use this before run_script when your code needs a ' +
      'third-party package that is not yet installed (e.g. requests, ' +
      'beautifulsoup4, pandas). Installs one package per call.')
    .Param('package', tptString,
      'Package name to install, e.g. "requests" or "numpy"', True)
    .OnExecute(
      function(const AToolName: string; const AParams: TToolParams): string
      var
        LPkg: string;
        LPythonExe: string;
        LTarget: string;
        LCmd: string;
        LResp: TToolResponse;
        LI: Integer;
        LCh: Char;
        LSafe: Boolean;
      begin
        LPkg := AParams.AsString('package').Trim();
        if LPkg = '' then
        begin
          Result := ToolResult(['error', 'Package name is required']);
          Exit;
        end;

        // Reject shell metacharacters to prevent command injection
        LSafe := True;
        for LI := 1 to Length(LPkg) do
        begin
          LCh := LPkg[LI];
          if (LCh = '&') or (LCh = '|') or (LCh = ';') or
             (LCh = '`') or (LCh = '$') or (LCh = '(') or
             (LCh = ')') or (LCh = '"') or (LCh = '''') then
          begin
            LSafe := False;
            Break;
          end;
        end;
        if not LSafe then
        begin
          Result := ToolResult(['error', 'Invalid package name']);
          Exit;
        end;

        LPythonExe := TUtils.AppBasedPath(CResPythonExe);
        LTarget := TUtils.AppBasedPath(CResPythonSitePackages);
        LCmd := '"' + LPythonExe + '" -m pip install ' + LPkg +
          ' --target "' + LTarget + '"';
        LResp := TLocalOps.RunCmd(LCmd);
        try
          if LResp.Success() then
            Result := LResp.RawJson()
          else
            Result := ToolResult(['error', LResp.ErrorMsg()]);
        finally
          LResp.Free();
        end;
      end);
end;

end.
