{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information

 -------------------------------------------------------------------------------

  Gemma4.Tools - Tool registry, fluent builder, JSON schema generation, dispatch

  Function-calling infrastructure for the model:
    - TToolRegistry: register named tools, generate JSON schemas, dispatch calls
    - TToolBuilder: fluent construction of a tool (name, description, params)
    - TToolParams / TToolParamType: typed parameter declarations and access
    - Handler callbacks receive parsed params, return a result string
    - Error codes TOL001-TOL003 for unknown tool / handler / params failures

  Dependencies: System.SysUtils, System.Generics.Collections,
    StdApp.Base, StdApp.JSON
===============================================================================}

unit Gemma4.Tools;

{$I StdApp.Defines.inc}

interface

uses
  System.SysUtils,
  System.Generics.Collections,
  StdApp.Base,
  StdApp.JSON;

const
  { Error codes }
  ERR_TOOL_UNKNOWN = 'TOL001';
  ERR_TOOL_HANDLER = 'TOL002';
  ERR_TOOL_PARAMS  = 'TOL003';

type
  // Forward declarations
  TToolBuilder = class;
  TToolRegistry = class;

  { TToolParamType }
  TToolParamType = (
    tptString,
    tptInteger,
    tptNumber,
    tptBoolean
  );

  { TToolCall }
  TToolCall = record
    ToolName: string;
    Arguments: string;   // raw JSON object: {"location":"London"}
    CallId: string;      // optional id assigned by the template format
  end;

  { TToolParamDef }
  TToolParamDef = record
    ParamName: string;
    ParamType: TToolParamType;
    Description: string;
    Required: Boolean;
  end;

  { TToolParams }
  // JSON argument accessor -- registry-managed; handlers borrow it and
  // must not free or retain it past the handler call
  TToolParams = class(TBaseObject)
  private
    FRoot: TJSON;
  public
    constructor Create(); override;
    destructor Destroy(); override;
    procedure Parse(const AJson: string);
    function AsString(const AKey: string; const ADefault: string = ''): string;
    function AsInteger(const AKey: string; const ADefault: Integer = 0): Integer;
    function AsDouble(const AKey: string; const ADefault: Double = 0.0): Double;
    function AsBoolean(const AKey: string; const ADefault: Boolean = False): Boolean;
    function Has(const AKey: string): Boolean;
  end;

  { TToolHandler }
  // Handler returns the tool result as a JSON object string (use
  // ToolResult for quick assembly)
  TToolHandler = reference to function(const AToolName: string;
    const AParams: TToolParams): string;

  { TToolDef }
  TToolDef = record
    ToolName: string;
    Description: string;
    Params: TArray<TToolParamDef>;
    Handler: TToolHandler;
  end;

  { TToolBuilder }
  // Fluent builder -- owned by the registry, one reusable instance
  TToolBuilder = class
  private
    FRegistry: TToolRegistry;
    FDef: TToolDef;
  public
    constructor Create(const ARegistry: TToolRegistry);
    procedure Reset(const AToolName: string);
    function Description(const AText: string): TToolBuilder;
    function Param(const AName: string; const AType: TToolParamType;
      const ADescription: string; const ARequired: Boolean): TToolBuilder;
    function OnExecute(const AHandler: TToolHandler): TToolRegistry;
  end;

  { TToolRegistry }
  // Registered tool definitions: schema generation for the chat template
  // (ToOaiJson) and dispatch of parsed calls (Execute)
  TToolRegistry = class(TBaseObject)
  private
    FTools: TList<TToolDef>;
    FBuilder: TToolBuilder;
    function FindTool(const AToolName: string; out AIndex: Integer): Boolean;
  public
    constructor Create(); override;
    destructor Destroy(); override;
    function DefineTool(const AToolName: string): TToolBuilder;
    procedure AddDef(const ADef: TToolDef);
    function Count(): Integer;
    function HasTool(const AToolName: string): Boolean;
    function GetDefs(): TArray<TToolDef>;
    procedure Clear();
    // OpenAI-format tool declarations array consumed by the chat
    // template; '' when no tools are registered
    function ToOaiJson(): string;
    function Execute(const ACall: TToolCall): string;
    function ExecuteAll(const ACalls: TArray<TToolCall>): TArray<string>;
    function ParamTypeToStr(const AType: TToolParamType): string;
  end;

// Quick JSON result builder -- alternating key/value pairs
function ToolResult(const APairs: array of const): string;

// JSON-escape and quote a string value (includes the surrounding quotes)
function ToolJsonStr(const AValue: string): string;

implementation

uses
  System.JSON,
  StdApp.Resources;

{ ToolJsonStr }

function ToolJsonStr(const AValue: string): string;
var
  LStr: TJSONString;
begin
  // TJSONString.ToJSON emits the value quoted with all escaping applied
  LStr := TJSONString.Create(AValue);
  try
    Result := LStr.ToJSON();
  finally
    LStr.Free();
  end;
end;

{ ToolResult }

function ToolResult(const APairs: array of const): string;
var
  LI: Integer;
  LKey: string;
  LValue: string;
  LFirst: Boolean;
begin
  Result := '{';
  LFirst := True;

  // Process pairs: index 0=key, 1=value, 2=key, 3=value, ...
  LI := 0;
  while LI < Length(APairs) - 1 do
  begin
    // Extract key (must be a string type)
    case APairs[LI].VType of
      vtAnsiString:
        LKey := string(AnsiString(APairs[LI].VAnsiString));
      vtWideString:
        LKey := string(WideString(APairs[LI].VWideString));
      vtUnicodeString:
        LKey := string(APairs[LI].VUnicodeString);
      vtChar:
        LKey := string(APairs[LI].VChar);
    else
      begin
        Inc(LI, 2);
        Continue;
      end;
    end;

    // Extract value
    case APairs[LI + 1].VType of
      vtInteger:
        LValue := IntToStr(APairs[LI + 1].VInteger);
      vtInt64:
        LValue := IntToStr(APairs[LI + 1].VInt64^);
      vtExtended:
        LValue := FloatToStr(APairs[LI + 1].VExtended^, TFormatSettings.Invariant);
      vtBoolean:
        if APairs[LI + 1].VBoolean then
          LValue := 'true'
        else
          LValue := 'false';
      vtAnsiString:
        LValue := ToolJsonStr(string(AnsiString(APairs[LI + 1].VAnsiString)));
      vtWideString:
        LValue := ToolJsonStr(string(WideString(APairs[LI + 1].VWideString)));
      vtUnicodeString:
        LValue := ToolJsonStr(string(APairs[LI + 1].VUnicodeString));
      vtChar:
        LValue := ToolJsonStr(string(APairs[LI + 1].VChar));
    else
      LValue := 'null';
    end;

    if not LFirst then
      Result := Result + ',';
    Result := Result + ToolJsonStr(LKey) + ':' + LValue;
    LFirst := False;

    Inc(LI, 2);
  end;

  Result := Result + '}';
end;

{ TToolParams }

constructor TToolParams.Create();
begin
  inherited Create();
  FRoot := nil;
end;

destructor TToolParams.Destroy();
begin
  FRoot.Free();
  inherited Destroy();
end;

procedure TToolParams.Parse(const AJson: string);
begin
  FreeAndNil(FRoot);

  if AJson.Trim() = '' then
    Exit;

  try
    FRoot := TJSON.FromString(AJson);
    if (FRoot <> nil) and (not FRoot.IsObject()) then
    begin
      FreeAndNil(FRoot);
      FErrors.Add(esError, ERR_TOOL_PARAMS, RSToolParamsNotObject);
    end;
  except
    on E: Exception do
    begin
      FreeAndNil(FRoot);
      FErrors.Add(esError, ERR_TOOL_PARAMS, RSToolParamsParse, [E.Message]);
    end;
  end;
end;

function TToolParams.AsString(const AKey: string; const ADefault: string): string;
begin
  Result := ADefault;
  if (FRoot <> nil) and FRoot.Has(AKey) then
    Result := FRoot.Get(AKey).AsString(ADefault);
end;

function TToolParams.AsInteger(const AKey: string; const ADefault: Integer): Integer;
begin
  Result := ADefault;
  if (FRoot <> nil) and FRoot.Has(AKey) then
    Result := FRoot.Get(AKey).AsInt32(ADefault);
end;

function TToolParams.AsDouble(const AKey: string; const ADefault: Double): Double;
begin
  Result := ADefault;
  if (FRoot <> nil) and FRoot.Has(AKey) then
    Result := FRoot.Get(AKey).AsDouble(ADefault);
end;

function TToolParams.AsBoolean(const AKey: string; const ADefault: Boolean): Boolean;
begin
  Result := ADefault;
  if (FRoot <> nil) and FRoot.Has(AKey) then
    Result := FRoot.Get(AKey).AsBoolean(ADefault);
end;

function TToolParams.Has(const AKey: string): Boolean;
begin
  Result := (FRoot <> nil) and FRoot.Has(AKey);
end;

{ TToolBuilder }

constructor TToolBuilder.Create(const ARegistry: TToolRegistry);
begin
  inherited Create();
  FRegistry := ARegistry;
end;

procedure TToolBuilder.Reset(const AToolName: string);
begin
  FDef := Default(TToolDef);
  FDef.ToolName := AToolName;
end;

function TToolBuilder.Description(const AText: string): TToolBuilder;
begin
  FDef.Description := AText;
  Result := Self;
end;

function TToolBuilder.Param(const AName: string; const AType: TToolParamType;
  const ADescription: string; const ARequired: Boolean): TToolBuilder;
var
  LParam: TToolParamDef;
begin
  LParam.ParamName := AName;
  LParam.ParamType := AType;
  LParam.Description := ADescription;
  LParam.Required := ARequired;
  FDef.Params := FDef.Params + [LParam];
  Result := Self;
end;

function TToolBuilder.OnExecute(const AHandler: TToolHandler): TToolRegistry;
begin
  FDef.Handler := AHandler;
  FRegistry.AddDef(FDef);
  Result := FRegistry;
end;

{ TToolRegistry }

constructor TToolRegistry.Create();
begin
  inherited Create();
  FTools := TList<TToolDef>.Create();
  FBuilder := TToolBuilder.Create(Self);
end;

destructor TToolRegistry.Destroy();
begin
  FBuilder.Free();
  FTools.Free();
  inherited Destroy();
end;

function TToolRegistry.FindTool(const AToolName: string; out AIndex: Integer): Boolean;
var
  LI: Integer;
begin
  for LI := 0 to FTools.Count - 1 do
  begin
    if SameText(FTools[LI].ToolName, AToolName) then
    begin
      AIndex := LI;
      Exit(True);
    end;
  end;
  AIndex := -1;
  Result := False;
end;

function TToolRegistry.ParamTypeToStr(const AType: TToolParamType): string;
begin
  case AType of
    tptString:  Result := 'string';
    tptInteger: Result := 'integer';
    tptNumber:  Result := 'number';
    tptBoolean: Result := 'boolean';
  else
    Result := 'string';
  end;
end;

function TToolRegistry.DefineTool(const AToolName: string): TToolBuilder;
begin
  FBuilder.Reset(AToolName);
  Result := FBuilder;
end;

procedure TToolRegistry.AddDef(const ADef: TToolDef);
var
  LIndex: Integer;
begin
  // Replace if already registered
  if FindTool(ADef.ToolName, LIndex) then
    FTools[LIndex] := ADef
  else
    FTools.Add(ADef);
end;

function TToolRegistry.Count(): Integer;
begin
  Result := FTools.Count;
end;

function TToolRegistry.HasTool(const AToolName: string): Boolean;
var
  LIndex: Integer;
begin
  Result := FindTool(AToolName, LIndex);
end;

function TToolRegistry.GetDefs(): TArray<TToolDef>;
var
  LI: Integer;
begin
  SetLength(Result, FTools.Count);
  for LI := 0 to FTools.Count - 1 do
    Result[LI] := FTools[LI];
end;

procedure TToolRegistry.Clear();
begin
  FTools.Clear();
end;

function TToolRegistry.ToOaiJson(): string;
var
  LSb: TStringBuilder;
  LIdx: Integer;
  LP: Integer;
  LDef: TToolDef;
  LFirst: Boolean;
begin
  Result := '';
  if FTools.Count = 0 then
    Exit;

  LSb := TStringBuilder.Create();
  try
    LSb.Append('[');
    for LIdx := 0 to FTools.Count - 1 do
    begin
      LDef := FTools[LIdx];
      if LIdx > 0 then
        LSb.Append(',');
      LSb.Append('{"type":"function","function":{');
      LSb.Append('"name":' + ToolJsonStr(LDef.ToolName) + ',');
      LSb.Append('"description":' + ToolJsonStr(LDef.Description) + ',');
      LSb.Append('"parameters":{"type":"object","properties":{');
      for LP := 0 to Length(LDef.Params) - 1 do
      begin
        if LP > 0 then
          LSb.Append(',');
        LSb.Append(ToolJsonStr(LDef.Params[LP].ParamName) + ':{');
        LSb.Append('"type":' + ToolJsonStr(ParamTypeToStr(LDef.Params[LP].ParamType)) + ',');
        LSb.Append('"description":' + ToolJsonStr(LDef.Params[LP].Description));
        LSb.Append('}');
      end;
      LSb.Append('},"required":[');
      LFirst := True;
      for LP := 0 to Length(LDef.Params) - 1 do
      begin
        if not LDef.Params[LP].Required then
          Continue;
        if not LFirst then
          LSb.Append(',');
        LSb.Append(ToolJsonStr(LDef.Params[LP].ParamName));
        LFirst := False;
      end;
      LSb.Append(']}}}');
    end;
    LSb.Append(']');
    Result := LSb.ToString();
  finally
    LSb.Free();
  end;
end;

function TToolRegistry.Execute(const ACall: TToolCall): string;
var
  LIndex: Integer;
  LParams: TToolParams;
begin
  Result := '';

  if not FindTool(ACall.ToolName, LIndex) then
  begin
    FErrors.Add(esError, ERR_TOOL_UNKNOWN, RSToolUnknown, [ACall.ToolName]);
    Result := ToolResult(['error', 'Unknown tool: ' + ACall.ToolName]);
    Exit;
  end;

  LParams := TToolParams.Create();
  try
    LParams.SetErrors(FErrors);
    LParams.Parse(ACall.Arguments);
    try
      Result := FTools[LIndex].Handler(ACall.ToolName, LParams);
    except
      on E: Exception do
      begin
        FErrors.Add(esError, ERR_TOOL_HANDLER, RSToolHandlerError,
          [ACall.ToolName, E.Message]);
        Result := ToolResult(['error', E.Message]);
      end;
    end;
  finally
    LParams.Free();
  end;
end;

function TToolRegistry.ExecuteAll(const ACalls: TArray<TToolCall>): TArray<string>;
var
  LI: Integer;
begin
  SetLength(Result, Length(ACalls));
  for LI := 0 to Length(ACalls) - 1 do
    Result[LI] := Execute(ACalls[LI]);
end;

end.
