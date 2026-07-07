{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information

 -------------------------------------------------------------------------------

  Gemma4.Jinja -- Full Jinja template engine for Gemma 4 chat template
  rendering.

  Ported from research reference .claude\research\misc\Cognita.Jinja.pas with
  the following fixes applied (Atom 18):
    1. map('filtername') positional form implemented (template needs
       map('upper')); attribute= kwarg form retained.
    2. Undefined vs None distinction added (jvkUndefined): missing variables,
       missing dict keys via []/. access, out-of-range indexing and else-less
       inline-if yield Undefined (renders ''); explicit none renders 'None'
       to byte-match Python/Jinja2. dict.get() misses stay none (Python).
    3. default() filter now follows strict Jinja semantics: replaces
       Undefined only, unless the boolean flag (2nd arg) is truthy.
    4. Debug WriteLn calls removed from if/test evaluation and Parse.
    5. Macro kwarg binding: nil arg slots no longer shadow declared defaults.
    6. Test names normalized to lowercase at parse time ('is None' works).
    7. dictsort uses deterministic ordinal case-insensitive CompareText,
       matching Python's key.lower() sort for ASCII keys.
    8. Unknown statements, filters and tests now raise instead of being
       silently skipped / passed through (fail loud).
    9. Block set now verifies the closing % endset % exists.
===============================================================================}

unit Gemma4.Jinja;

{$I StdApp.Defines.inc}

interface

uses
  System.SysUtils,
  System.Classes,
  System.Math,
  System.StrUtils,
  System.Generics.Collections,
  System.Generics.Defaults,
  StdApp.Base;

type
  // Forward declarations
  TJinjaValue = class;
  TJinjaContext = class;
  TJinjaNode = class;
  TJinja = class;

  { TJinjaValueKind }
  // jvkUndefined is Jinja's Undefined (missing variable / missing key):
  // falsy, renders as '', is NOT none. jvkNone is an explicit none/None
  // value: falsy, renders as 'None' (Python str), IS none.
  TJinjaValueKind = (
    jvkUndefined,
    jvkNone,
    jvkBool,
    jvkInt,
    jvkFloat,
    jvkString,
    jvkArray,
    jvkDict,
    jvkCallable
  );

  { TJinjaCallableFunc }
  // Callable signature: takes array of values, returns a value
  // The pool parameter is used to allocate return values
  TJinjaCallableFunc = reference to function(
    const AArgs: TArray<TJinjaValue>;
    const APool: TObjectList<TJinjaValue>): TJinjaValue;

  { TJinjaValue }
  TJinjaValue = class
  private
    FKind: TJinjaValueKind;
    FBoolValue: Boolean;
    FIntValue: Int64;
    FFloatValue: Double;
    FStringValue: string;
    FArrayItems: TList<TJinjaValue>;
    FDictKeys: TStringList;
    FDictMap: TDictionary<string, TJinjaValue>;
    FCallable: TJinjaCallableFunc;
    FCallableName: string;
    FCallableParams: TStringList;  // macro parameter names for kwarg mapping
  public
    constructor Create(); virtual;
    destructor Destroy(); override;

    // Kind
    property Kind: TJinjaValueKind read FKind;

    // Truthiness (Jinja semantics)
    function IsTruthy(): Boolean;

    // Type checks
    function IsUndefined(): Boolean;
    function IsNone(): Boolean;
    function IsBool(): Boolean;
    function IsInt(): Boolean;
    function IsFloat(): Boolean;
    function IsNumber(): Boolean;
    function IsString(): Boolean;
    function IsArray(): Boolean;
    function IsDict(): Boolean;
    function IsCallable(): Boolean;
    function IsMapping(): Boolean;
    function IsIterable(): Boolean;

    // Accessors
    function AsBool(): Boolean;
    function AsInt(): Int64;
    function AsFloat(): Double;
    function AsString(): string;
    function ArrayCount(): Integer;
    function ArrayGet(const AIndex: Integer): TJinjaValue;
    procedure ArrayAdd(const AValue: TJinjaValue);
    function DictCount(): Integer;
    function DictGet(const AKey: string): TJinjaValue;
    function DictHas(const AKey: string): Boolean;
    procedure DictSet(const AKey: string; const AValue: TJinjaValue);
    function DictKeys(): TStringList;
    function CallableGet(): TJinjaCallableFunc;

    // Coerce to string for output
    function ToOutput(): string;

    // Coerce to JSON string (for tojson filter)
    function ToJSON(): string;

    // Comparison
    function Equals(const AOther: TJinjaValue): Boolean; reintroduce;
  end;

  { TJinjaValuePool }
  // Arena allocator for values -- owns all values created during rendering
  TJinjaValuePool = class
  private
    FValues: TObjectList<TJinjaValue>;
  public
    constructor Create(); virtual;
    destructor Destroy(); override;

    function NewUndefined(): TJinjaValue;
    function NewNone(): TJinjaValue;
    function NewBool(const AValue: Boolean): TJinjaValue;
    function NewInt(const AValue: Int64): TJinjaValue;
    function NewFloat(const AValue: Double): TJinjaValue;
    function NewString(const AValue: string): TJinjaValue;
    function NewArray(): TJinjaValue;
    function NewDict(): TJinjaValue;
    function NewCallable(const AFunc: TJinjaCallableFunc;
      const ANodeName: string = ''): TJinjaValue;

    // Clone a value into this pool
    function Clone(const ASource: TJinjaValue): TJinjaValue;

    function GetPool(): TObjectList<TJinjaValue>;
  end;

  { TJinjaContext }
  // Variable scope with parent chaining
  TJinjaContext = class
  private
    FVars: TDictionary<string, TJinjaValue>;
    FParent: TJinjaContext;
    FPool: TJinjaValuePool;
    FOwnsPool: Boolean;
  public
    constructor Create(const AParent: TJinjaContext = nil;
      const APool: TJinjaValuePool = nil);
    destructor Destroy(); override;

    function Get(const ANodeName: string): TJinjaValue;
    function Has(const ANodeName: string): Boolean;
    procedure SetVar(const ANodeName: string; const AValue: TJinjaValue);
    // Set in current scope only (for loop vars, macro params)
    procedure SetLocal(const ANodeName: string; const AValue: TJinjaValue);

    function GetPool(): TJinjaValuePool;
    function GetParent(): TJinjaContext;
  end;

  { TJinjaTokenKind }
  TJinjaTokenKind = (
    jtkText,         // raw text outside tags
    jtkIdent,        // identifier
    jtkString,       // string literal
    jtkInteger,      // integer literal
    jtkFloat,        // float literal
    jtkTrue,         // true
    jtkFalse,        // false
    jtkNone,         // none
    jtkLParen,       // (
    jtkRParen,       // )
    jtkLBracket,     // [
    jtkRBracket,     // ]
    jtkLBrace,       // {
    jtkRBrace,       // }
    jtkComma,        // ,
    jtkDot,          // .
    jtkPipe,         // |
    jtkColon,        // :
    jtkTilde,        // ~ (string concat)
    jtkAssign,       // =
    jtkEq,           // ==
    jtkNe,           // !=
    jtkLt,           // <
    jtkGt,           // >
    jtkLe,           // <=
    jtkGe,           // >=
    jtkPlus,         // +
    jtkMinus,        // -
    jtkMul,          // *
    jtkDiv,          // /
    jtkIntDiv,       // //
    jtkMod,          // %
    jtkNot,          // not
    jtkAnd,          // and
    jtkOr,           // or
    jtkIn,           // in
    jtkIs,           // is
    jtkIf,           // if (expression-level ternary)
    jtkElse,         // else
    jtkEOF           // end of input
  );

  { TJinjaToken }
  TJinjaToken = record
    Kind: TJinjaTokenKind;
    Value: string;
    Line: Integer;
    Col: Integer;
  end;

  { TJinjaSegmentKind }
  // Raw template segments before expression tokenization
  TJinjaSegmentKind = (
    jskText,       // raw text
    jskExpr,       // {{ ... }} content
    jskStmt,       // {% ... %} content
    jskComment     // {# ... #} content
  );

  { TJinjaSegment }
  TJinjaSegment = record
    Kind: TJinjaSegmentKind;
    Content: string;
    TrimLeft: Boolean;   // {%- or {{- or {#-
    TrimRight: Boolean;  // -%} or -}} or -#}
    Line: Integer;
  end;

  { TJinjaNodeKind }
  // AST node types
  TJinjaNodeKind = (
    jnkText,
    jnkOutput,
    jnkBlock,
    jnkIf,
    jnkFor,
    jnkSetStmt,
    jnkMacro,
    jnkCallBlock
  );

  { TJinjaExprKind }
  TJinjaExprKind = (
    jekLiteral,
    jekVar,
    jekBinary,
    jekUnary,
    jekMember,
    jekIndex,
    jekSlice,
    jekCall,
    jekFilter,
    jekTest,
    jekArray,
    jekDict,
    jekConditional,    // expr if cond else expr
    jekMethodCall      // obj.method(args)
  );

  { TJinjaExpr }
  TJinjaExpr = class
  public
    ExprKind: TJinjaExprKind;

    // jekLiteral
    LitValue: string;
    LitKind: TJinjaTokenKind; // jtkString, jtkInteger, jtkFloat, jtkTrue, jtkFalse, jtkNone

    // jekVar
    VarName: string;

    // jekBinary
    BinOp: TJinjaTokenKind;
    BinLeft: TJinjaExpr;
    BinRight: TJinjaExpr;

    // jekUnary
    UnaryOp: TJinjaTokenKind;
    UnaryOperand: TJinjaExpr;

    // jekMember / jekIndex
    MemberObj: TJinjaExpr;
    MemberName: string;       // for dot access
    IndexExpr: TJinjaExpr;  // for bracket access

    // jekSlice (arr[start:stop])
    SliceObj: TJinjaExpr;
    SliceStart: TJinjaExpr; // nil = from beginning
    SliceStop: TJinjaExpr;  // nil = to end
    SliceStep: TJinjaExpr;  // nil = step 1

    // jekCall / jekMethodCall
    CallTarget: TJinjaExpr;
    CallArgs: TObjectList<TJinjaExpr>;
    CallKwargs: TDictionary<string, TJinjaExpr>;

    // jekFilter
    FilterExpr: TJinjaExpr;
    FilterName: string;
    FilterArgs: TObjectList<TJinjaExpr>;
    FilterKwargs: TDictionary<string, TJinjaExpr>;

    // jekTest
    TestExpr: TJinjaExpr;
    TestName: string;
    TestArgs: TObjectList<TJinjaExpr>;
    TestNegated: Boolean;

    // jekArray
    ArrayItems: TObjectList<TJinjaExpr>;

    // jekDict
    DictPairs: TList<TPair<TJinjaExpr, TJinjaExpr>>;

    // jekConditional (ternary)
    CondTrue: TJinjaExpr;
    CondTest: TJinjaExpr;
    CondFalse: TJinjaExpr;

    constructor Create(const AKind: TJinjaExprKind);
    destructor Destroy(); override;
  end;

  { TJinjaIfBranch }
  TJinjaIfBranch = record
    Condition: TJinjaExpr; // nil for else
    Body: TJinjaNode;
  end;

  { TJinjaNode }
  TJinjaNode = class
  public
    NodeKind: TJinjaNodeKind;

    // jnkText
    Text: string;

    // jnkOutput
    OutputExpr: TJinjaExpr;

    // jnkBlock
    Children: TObjectList<TJinjaNode>;

    // jnkIf
    IfBranches: TList<TJinjaIfBranch>;

    // jnkFor
    ForVarName: string;
    ForVarName2: string;  // for key, value in dict.items()
    ForIterExpr: TJinjaExpr;
    ForBody: TJinjaNode;
    ForElseBody: TJinjaNode;
    ForRecursive: Boolean;

    // jnkSetStmt
    SetVarName: string;
    SetAttrName: string;  // for {% set ns.attr = expr %}
    SetExpr: TJinjaExpr;
    SetBlockBody: TJinjaNode;  // Block set: {% set x %}...{% endset %}

    // jnkMacro
    MacroName: string;
    MacroParams: TStringList;        // param names
    MacroDefaults: TDictionary<string, TJinjaExpr>; // default values
    MacroBody: TJinjaNode;

    // jnkCallBlock
    CallExpr: TJinjaExpr;   // the macro call expression
    CallBody: TJinjaNode;   // body rendered by caller()

    constructor Create(const AKind: TJinjaNodeKind);
    destructor Destroy(); override;
  end;

  { TJinjaLexer }
  // Phase 1: split template into segments
  // Phase 2: tokenize expression/statement content
  TJinjaLexer = class
  private
    class function SplitSegments(const ATemplate: string): TList<TJinjaSegment>; static;
    class function TokenizeExpr(const AContent: string; const ALine: Integer): TList<TJinjaToken>; static;
  end;

  { TJinjaParser }
  TJinjaParser = class(TBaseObject)
  private
    FSegments: TList<TJinjaSegment>;
    FSegIdx: Integer;
    // Current expression tokens (when inside a tag)
    FTokens: TList<TJinjaToken>;
    FTokIdx: Integer;

    // Segment-level navigation
    function AtEnd(): Boolean;
    function CurSegment(): TJinjaSegment;
    procedure NextSegment();

    // Token-level navigation
    procedure LoadTokens(const AContent: string; const ALine: Integer);
    {$HINTS OFF}
    function TokAtEnd(): Boolean;
    {$HINTS ON}
    function TokPeek(): TJinjaToken;
    function TokNext(): TJinjaToken;
    function TokExpect(const AKind: TJinjaTokenKind): TJinjaToken;
    function TokMatch(const AKind: TJinjaTokenKind): Boolean;
    function TokMatchIdent(const ANodeName: string): Boolean;

    // Statement-level parsing
    function IsStmt(const ANodeName: string): Boolean;
    {$HINTS OFF}
    procedure ExpectStmt(const ANodeName: string);
    {$HINTS ON}

    // Parse routines
    function ParseTemplate(): TJinjaNode;
    function ParseBlock(const AEndTags: TArray<string>): TJinjaNode;
    function ParseStatement(): TJinjaNode;
    function ParseIf(): TJinjaNode;
    function ParseFor(): TJinjaNode;
    function ParseSet(): TJinjaNode;
    function ParseMacro(): TJinjaNode;
    function ParseCallBlock(): TJinjaNode;

    // Expression parsing (precedence climbing)
    function ParseExpr(): TJinjaExpr;
    function ParseConditionalExpr(): TJinjaExpr;
    function ParseOrExpr(): TJinjaExpr;
    function ParseAndExpr(): TJinjaExpr;
    function ParseNotExpr(): TJinjaExpr;
    function ParseCompareExpr(): TJinjaExpr;
    function ParseAddExpr(): TJinjaExpr;
    function ParseMulExpr(): TJinjaExpr;
    function ParseUnaryExpr(): TJinjaExpr;
    function ParsePostfixExpr(): TJinjaExpr;
    function ParsePrimaryExpr(): TJinjaExpr;
    function ParseFilterChain(const AExpr: TJinjaExpr): TJinjaExpr;
  public
    constructor Create(); override;
    destructor Destroy(); override;

    function Parse(const ATemplate: string): TJinjaNode;
  end;

  { TJinjaRenderer }
  TJinjaRenderer = class(TBaseObject)
  private
    FPool: TJinjaValuePool;
    FOutput: TStringBuilder;

    // Built-in filters and globals
    procedure RegisterBuiltins(const ACtx: TJinjaContext);

    // Evaluation
    function Eval(const AExpr: TJinjaExpr;
      const ACtx: TJinjaContext): TJinjaValue;
    procedure Exec(const ANode: TJinjaNode;
      const ACtx: TJinjaContext);
    function ApplyFilter(const ANodeName: string; const AInput: TJinjaValue;
      const AArgs: TArray<TJinjaValue>;
      const AKwargs: TDictionary<string, TJinjaValue>;
      const ACtx: TJinjaContext): TJinjaValue;
    function ApplyTest(const ANodeName: string; const AInput: TJinjaValue;
      const AArgs: TArray<TJinjaValue>;
      const ACtx: TJinjaContext): Boolean;
    function CallValue(const ACallable: TJinjaValue;
      const AArgs: TArray<TJinjaValue>;
      const ACtx: TJinjaContext): TJinjaValue;
  public
    constructor Create(); override;
    destructor Destroy(); override;

    function Render(const ARoot: TJinjaNode;
      const ACtx: TJinjaContext): string;
  end;

  { TJinja }
  // Public API: parse template, render with context
  TJinja = class(TBaseObject)
  private
    FRoot: TJinjaNode;
    FParser: TJinjaParser;
  public
    constructor Create(); override;
    destructor Destroy(); override;

    function Parse(const ATemplate: string): Boolean;
    function Render(const AContext: TJinjaContext): string;
  end;

implementation

{ Helper routines }

// Escape a string as a double-quoted JSON-style literal.
function JinjaEscapeString(const AStr: string): string;
var
  LBuilder: TStringBuilder;
  I: Integer;
  LC: Char;
begin
  LBuilder := TStringBuilder.Create();
  try
    LBuilder.Append('"');
    for I := 1 to Length(AStr) do
    begin
      LC := AStr[I];
      if LC = '"' then
        LBuilder.Append('\"')
      else if LC = '\' then
        LBuilder.Append('\\')
      else if LC = #10 then
        LBuilder.Append('\n')
      else if LC = #13 then
        LBuilder.Append('\r')
      else if LC = #9 then
        LBuilder.Append('\t')
      else
        LBuilder.Append(LC);
    end;
    LBuilder.Append('"');
    Result := LBuilder.ToString();
  finally
    LBuilder.Free();
  end;
end;

// Sort callback for dictsort: deterministic ordinal case-insensitive
// compare (SysUtils.CompareText), matching Python's key.lower() sort for
// ASCII keys. Parameters are not const because the signature must match
// TStringListSortCompare exactly.
function JinjaDictSortCompare(AList: TStringList;
  AIndex1: Integer; AIndex2: Integer): Integer;
begin
  Result := CompareText(AList[AIndex1], AList[AIndex2]);
end;

{ TJinjaValue }

constructor TJinjaValue.Create();
begin
  inherited Create();
  FKind := jvkUndefined;
  FBoolValue := False;
  FIntValue := 0;
  FFloatValue := 0.0;
  FStringValue := '';
  FArrayItems := nil;
  FDictKeys := nil;
  FDictMap := nil;
  FCallable := nil;
  FCallableName := '';
  FCallableParams := nil;
end;

destructor TJinjaValue.Destroy();
begin
  // Array and dict sub-objects are owned by the pool, not by us
  FArrayItems.Free();
  FDictKeys.Free();
  FDictMap.Free();
  FCallableParams.Free();
  inherited Destroy();
end;

function TJinjaValue.IsTruthy(): Boolean;
begin
  case FKind of
    jvkUndefined: Result := False;
    jvkNone:      Result := False;
    jvkBool:      Result := FBoolValue;
    jvkInt:       Result := FIntValue <> 0;
    jvkFloat:     Result := FFloatValue <> 0.0;
    jvkString:    Result := FStringValue <> '';
    jvkArray:     Result := (FArrayItems <> nil) and (FArrayItems.Count > 0);
    jvkDict:      Result := (FDictMap <> nil) and (FDictMap.Count > 0);
    jvkCallable:  Result := True;
  else
    Result := False;
  end;
end;

function TJinjaValue.IsUndefined(): Boolean;
begin
  Result := FKind = jvkUndefined;
end;

function TJinjaValue.IsNone(): Boolean;
begin
  Result := FKind = jvkNone;
end;

function TJinjaValue.IsBool(): Boolean;
begin
  Result := FKind = jvkBool;
end;

function TJinjaValue.IsInt(): Boolean;
begin
  Result := FKind = jvkInt;
end;

function TJinjaValue.IsFloat(): Boolean;
begin
  Result := FKind = jvkFloat;
end;

function TJinjaValue.IsNumber(): Boolean;
begin
  Result := (FKind = jvkInt) or (FKind = jvkFloat);
end;

function TJinjaValue.IsString(): Boolean;
begin
  Result := FKind = jvkString;
end;

function TJinjaValue.IsArray(): Boolean;
begin
  Result := FKind = jvkArray;
end;

function TJinjaValue.IsDict(): Boolean;
begin
  Result := FKind = jvkDict;
end;

function TJinjaValue.IsCallable(): Boolean;
begin
  Result := FKind = jvkCallable;
end;

function TJinjaValue.IsMapping(): Boolean;
begin
  Result := FKind = jvkDict;
end;

function TJinjaValue.IsIterable(): Boolean;
begin
  Result := (FKind = jvkArray) or (FKind = jvkDict) or (FKind = jvkString);
end;

function TJinjaValue.AsBool(): Boolean;
begin
  Result := FBoolValue;
end;

function TJinjaValue.AsInt(): Int64;
begin
  if FKind = jvkFloat then
    Result := Trunc(FFloatValue)
  else
    Result := FIntValue;
end;

function TJinjaValue.AsFloat(): Double;
begin
  if FKind = jvkInt then
    Result := FIntValue
  else
    Result := FFloatValue;
end;

function TJinjaValue.AsString(): string;
begin
  Result := FStringValue;
end;

function TJinjaValue.ArrayCount(): Integer;
begin
  if FArrayItems <> nil then
    Result := FArrayItems.Count
  else
    Result := 0;
end;

function TJinjaValue.ArrayGet(const AIndex: Integer): TJinjaValue;
var
  LIdx: Integer;
begin
  LIdx := AIndex;
  // Support negative indexing
  if LIdx < 0 then
    LIdx := FArrayItems.Count + LIdx;
  if (LIdx < 0) or (LIdx >= FArrayItems.Count) then
    raise Exception.CreateFmt('Array index %d out of range', [AIndex]);
  Result := FArrayItems[LIdx];
end;

procedure TJinjaValue.ArrayAdd(const AValue: TJinjaValue);
begin
  if FArrayItems = nil then
    FArrayItems := TList<TJinjaValue>.Create();
  FArrayItems.Add(AValue);
end;

function TJinjaValue.DictCount(): Integer;
begin
  if FDictMap <> nil then
    Result := FDictMap.Count
  else
    Result := 0;
end;

function TJinjaValue.DictGet(const AKey: string): TJinjaValue;
begin
  if (FDictMap <> nil) and FDictMap.TryGetValue(AKey, Result) then
    Exit;
  Result := nil;
end;

function TJinjaValue.DictHas(const AKey: string): Boolean;
begin
  Result := (FDictMap <> nil) and FDictMap.ContainsKey(AKey);
end;

procedure TJinjaValue.DictSet(const AKey: string; const AValue: TJinjaValue);
begin
  if FDictMap = nil then
  begin
    FDictMap := TDictionary<string, TJinjaValue>.Create();
    FDictKeys := TStringList.Create();
  end;
  if not FDictMap.ContainsKey(AKey) then
    FDictKeys.Add(AKey);
  FDictMap.AddOrSetValue(AKey, AValue);
end;

function TJinjaValue.DictKeys(): TStringList;
begin
  Result := FDictKeys;
end;

function TJinjaValue.CallableGet(): TJinjaCallableFunc;
begin
  Result := FCallable;
end;

function TJinjaValue.ToOutput(): string;
var
  I: Integer;
  LBuilder: TStringBuilder;
begin
  case FKind of
    // Undefined renders as empty string; explicit none renders as 'None'
    // (Python str(None)) to byte-match Jinja2/HF output.
    jvkUndefined: Result := '';
    jvkNone:      Result := 'None';
    jvkBool:
      if FBoolValue then
        Result := 'True'
      else
        Result := 'False';
    jvkInt:      Result := IntToStr(FIntValue);
    jvkFloat:
    begin
      Result := FloatToStr(FFloatValue, TFormatSettings.Invariant);
      // Ensure decimal point
      if Pos('.', Result) = 0 then
        Result := Result + '.0';
    end;
    jvkString:   Result := FStringValue;
    jvkArray:
    begin
      LBuilder := TStringBuilder.Create();
      try
        LBuilder.Append('[');
        for I := 0 to ArrayCount() - 1 do
        begin
          if I > 0 then
            LBuilder.Append(', ');
          LBuilder.Append(FArrayItems[I].ToJSON());
        end;
        LBuilder.Append(']');
        Result := LBuilder.ToString();
      finally
        LBuilder.Free();
      end;
    end;
    jvkDict:
    begin
      LBuilder := TStringBuilder.Create();
      try
        LBuilder.Append('{');
        for I := 0 to FDictKeys.Count - 1 do
        begin
          if I > 0 then
            LBuilder.Append(', ');
          LBuilder.Append(JinjaEscapeString(FDictKeys[I]));
          LBuilder.Append(': ');
          LBuilder.Append(FDictMap[FDictKeys[I]].ToJSON());
        end;
        LBuilder.Append('}');
        Result := LBuilder.ToString();
      finally
        LBuilder.Free();
      end;
    end;
    jvkCallable: Result := '<callable>';
  else
    Result := '';
  end;
end;

function TJinjaValue.ToJSON(): string;
var
  I: Integer;
  LBuilder: TStringBuilder;
begin
  case FKind of
    jvkUndefined: Result := 'null';
    jvkNone:      Result := 'null';
    jvkBool:
      if FBoolValue then
        Result := 'true'
      else
        Result := 'false';
    jvkInt:    Result := IntToStr(FIntValue);
    jvkFloat:
    begin
      Result := FloatToStr(FFloatValue, TFormatSettings.Invariant);
      if Pos('.', Result) = 0 then
        Result := Result + '.0';
    end;
    jvkString: Result := JinjaEscapeString(FStringValue);
    jvkArray:
    begin
      LBuilder := TStringBuilder.Create();
      try
        LBuilder.Append('[');
        for I := 0 to ArrayCount() - 1 do
        begin
          if I > 0 then
            LBuilder.Append(', ');
          LBuilder.Append(FArrayItems[I].ToJSON());
        end;
        LBuilder.Append(']');
        Result := LBuilder.ToString();
      finally
        LBuilder.Free();
      end;
    end;
    jvkDict:
    begin
      LBuilder := TStringBuilder.Create();
      try
        LBuilder.Append('{');
        for I := 0 to FDictKeys.Count - 1 do
        begin
          if I > 0 then
            LBuilder.Append(', ');
          LBuilder.Append(JinjaEscapeString(FDictKeys[I]));
          LBuilder.Append(': ');
          LBuilder.Append(FDictMap[FDictKeys[I]].ToJSON());
        end;
        LBuilder.Append('}');
        Result := LBuilder.ToString();
      finally
        LBuilder.Free();
      end;
    end;
  else
    Result := 'null';
  end;
end;

function TJinjaValue.Equals(const AOther: TJinjaValue): Boolean;
var
  I: Integer;
begin
  if AOther = nil then
    Exit(FKind = jvkNone);
  if FKind <> AOther.FKind then
  begin
    // Allow int/float comparison
    if IsNumber() and AOther.IsNumber() then
      Exit(AsFloat() = AOther.AsFloat());
    Exit(False);
  end;
  case FKind of
    jvkUndefined: Result := True;
    jvkNone:      Result := True;
    jvkBool:      Result := FBoolValue = AOther.FBoolValue;
    jvkInt:       Result := FIntValue = AOther.FIntValue;
    jvkFloat:     Result := FFloatValue = AOther.FFloatValue;
    jvkString:    Result := FStringValue = AOther.FStringValue;
    jvkArray:
    begin
      if ArrayCount() <> AOther.ArrayCount() then
        Exit(False);
      for I := 0 to ArrayCount() - 1 do
      begin
        if not FArrayItems[I].Equals(AOther.FArrayItems[I]) then
          Exit(False);
      end;
      Result := True;
    end;
  else
    Result := False;
  end;
end;

{ TJinjaValuePool }

constructor TJinjaValuePool.Create();
begin
  inherited Create();
  FValues := TObjectList<TJinjaValue>.Create(True);
end;

destructor TJinjaValuePool.Destroy();
begin
  FValues.Free();
  inherited Destroy();
end;

function TJinjaValuePool.NewUndefined(): TJinjaValue;
begin
  Result := TJinjaValue.Create();
  Result.FKind := jvkUndefined;
  FValues.Add(Result);
end;

function TJinjaValuePool.NewNone(): TJinjaValue;
begin
  Result := TJinjaValue.Create();
  Result.FKind := jvkNone;
  FValues.Add(Result);
end;

function TJinjaValuePool.NewBool(const AValue: Boolean): TJinjaValue;
begin
  Result := TJinjaValue.Create();
  Result.FKind := jvkBool;
  Result.FBoolValue := AValue;
  FValues.Add(Result);
end;

function TJinjaValuePool.NewInt(const AValue: Int64): TJinjaValue;
begin
  Result := TJinjaValue.Create();
  Result.FKind := jvkInt;
  Result.FIntValue := AValue;
  FValues.Add(Result);
end;

function TJinjaValuePool.NewFloat(const AValue: Double): TJinjaValue;
begin
  Result := TJinjaValue.Create();
  Result.FKind := jvkFloat;
  Result.FFloatValue := AValue;
  FValues.Add(Result);
end;

function TJinjaValuePool.NewString(const AValue: string): TJinjaValue;
begin
  Result := TJinjaValue.Create();
  Result.FKind := jvkString;
  Result.FStringValue := AValue;
  FValues.Add(Result);
end;

function TJinjaValuePool.NewArray(): TJinjaValue;
begin
  Result := TJinjaValue.Create();
  Result.FKind := jvkArray;
  Result.FArrayItems := TList<TJinjaValue>.Create();
  FValues.Add(Result);
end;

function TJinjaValuePool.NewDict(): TJinjaValue;
begin
  Result := TJinjaValue.Create();
  Result.FKind := jvkDict;
  Result.FDictMap := TDictionary<string, TJinjaValue>.Create();
  Result.FDictKeys := TStringList.Create();
  FValues.Add(Result);
end;

function TJinjaValuePool.NewCallable(const AFunc: TJinjaCallableFunc;
  const ANodeName: string): TJinjaValue;
begin
  Result := TJinjaValue.Create();
  Result.FKind := jvkCallable;
  Result.FCallable := AFunc;
  Result.FCallableName := ANodeName;
  FValues.Add(Result);
end;

function TJinjaValuePool.Clone(const ASource: TJinjaValue): TJinjaValue;
var
  I: Integer;
  LKey: string;
begin
  case ASource.FKind of
    jvkUndefined: Result := NewUndefined();
    jvkNone:      Result := NewNone();
    jvkBool:      Result := NewBool(ASource.FBoolValue);
    jvkInt:       Result := NewInt(ASource.FIntValue);
    jvkFloat:     Result := NewFloat(ASource.FFloatValue);
    jvkString:    Result := NewString(ASource.FStringValue);
    jvkArray:
    begin
      Result := NewArray();
      for I := 0 to ASource.ArrayCount() - 1 do
        Result.ArrayAdd(Clone(ASource.FArrayItems[I]));
    end;
    jvkDict:
    begin
      Result := NewDict();
      if ASource.FDictKeys <> nil then
      begin
        for I := 0 to ASource.FDictKeys.Count - 1 do
        begin
          LKey := ASource.FDictKeys[I];
          Result.DictSet(LKey, Clone(ASource.FDictMap[LKey]));
        end;
      end;
    end;
    jvkCallable:
    begin
      Result := NewCallable(ASource.FCallable, ASource.FCallableName);
      if ASource.FCallableParams <> nil then
      begin
        Result.FCallableParams := TStringList.Create();
        Result.FCallableParams.Assign(ASource.FCallableParams);
      end;
    end;
  else
    Result := NewUndefined();
  end;
end;

function TJinjaValuePool.GetPool(): TObjectList<TJinjaValue>;
begin
  Result := FValues;
end;

{ TJinjaContext }

constructor TJinjaContext.Create(const AParent: TJinjaContext;
  const APool: TJinjaValuePool);
begin
  inherited Create();
  FVars := TDictionary<string, TJinjaValue>.Create();
  FParent := AParent;
  if APool <> nil then
  begin
    FPool := APool;
    FOwnsPool := False;
  end
  else if AParent <> nil then
  begin
    FPool := AParent.GetPool();
    FOwnsPool := False;
  end
  else
  begin
    FPool := TJinjaValuePool.Create();
    FOwnsPool := True;
  end;
end;

destructor TJinjaContext.Destroy();
begin
  FVars.Free();
  if FOwnsPool then
    FPool.Free();
  inherited Destroy();
end;

function TJinjaContext.Get(const ANodeName: string): TJinjaValue;
begin
  if FVars.TryGetValue(ANodeName, Result) then
    Exit;
  if FParent <> nil then
    Exit(FParent.Get(ANodeName));
  Result := nil;
end;

function TJinjaContext.Has(const ANodeName: string): Boolean;
begin
  if FVars.ContainsKey(ANodeName) then
    Exit(True);
  if FParent <> nil then
    Exit(FParent.Has(ANodeName));
  Result := False;
end;

procedure TJinjaContext.SetVar(const ANodeName: string;
  const AValue: TJinjaValue);
begin
  // Walk up to find where the var is defined, set it there
  if FVars.ContainsKey(ANodeName) then
  begin
    FVars.AddOrSetValue(ANodeName, AValue);
    Exit;
  end;
  if FParent <> nil then
  begin
    if FParent.Has(ANodeName) then
    begin
      FParent.SetVar(ANodeName, AValue);
      Exit;
    end;
  end;
  // Not found anywhere -- set in current scope
  FVars.AddOrSetValue(ANodeName, AValue);
end;

procedure TJinjaContext.SetLocal(const ANodeName: string;
  const AValue: TJinjaValue);
begin
  FVars.AddOrSetValue(ANodeName, AValue);
end;

function TJinjaContext.GetPool(): TJinjaValuePool;
begin
  Result := FPool;
end;

function TJinjaContext.GetParent(): TJinjaContext;
begin
  Result := FParent;
end;

{ TJinjaExpr }

constructor TJinjaExpr.Create(const AKind: TJinjaExprKind);
begin
  inherited Create();
  ExprKind := AKind;
  LitKind := jtkNone;
  BinLeft := nil;
  BinRight := nil;
  UnaryOperand := nil;
  MemberObj := nil;
  IndexExpr := nil;
  SliceObj := nil;
  SliceStart := nil;
  SliceStop := nil;
  SliceStep := nil;
  CallTarget := nil;
  CallArgs := nil;
  CallKwargs := nil;
  FilterExpr := nil;
  FilterArgs := nil;
  FilterKwargs := nil;
  TestExpr := nil;
  TestArgs := nil;
  TestNegated := False;
  ArrayItems := nil;
  DictPairs := nil;
  CondTrue := nil;
  CondTest := nil;
  CondFalse := nil;
end;

destructor TJinjaExpr.Destroy();
var
  LStrPair: TPair<string, TJinjaExpr>;
  LExprPair: TPair<TJinjaExpr, TJinjaExpr>;
begin
  BinLeft.Free();
  BinRight.Free();
  UnaryOperand.Free();
  MemberObj.Free();
  IndexExpr.Free();
  SliceObj.Free();
  SliceStart.Free();
  SliceStop.Free();
  SliceStep.Free();
  CallTarget.Free();
  CallArgs.Free();
  if CallKwargs <> nil then
  begin
    for LStrPair in CallKwargs do
      LStrPair.Value.Free();
    CallKwargs.Free();
  end;
  FilterExpr.Free();
  FilterArgs.Free();
  if FilterKwargs <> nil then
  begin
    for LStrPair in FilterKwargs do
      LStrPair.Value.Free();
    FilterKwargs.Free();
  end;
  TestExpr.Free();
  TestArgs.Free();
  ArrayItems.Free();
  if DictPairs <> nil then
  begin
    for LExprPair in DictPairs do
    begin
      LExprPair.Key.Free();
      LExprPair.Value.Free();
    end;
    DictPairs.Free();
  end;
  CondTrue.Free();
  CondTest.Free();
  CondFalse.Free();
  inherited Destroy();
end;

{ TJinjaNode }

constructor TJinjaNode.Create(const AKind: TJinjaNodeKind);
begin
  inherited Create();
  NodeKind := AKind;
  OutputExpr := nil;
  Children := nil;
  IfBranches := nil;
  ForIterExpr := nil;
  ForBody := nil;
  ForElseBody := nil;
  ForRecursive := False;
  SetExpr := nil;
  SetBlockBody := nil;
  MacroParams := nil;
  MacroDefaults := nil;
  MacroBody := nil;
  CallExpr := nil;
  CallBody := nil;
end;

destructor TJinjaNode.Destroy();
var
  I: Integer;
  LPair: TPair<string, TJinjaExpr>;
begin
  OutputExpr.Free();
  Children.Free();
  if IfBranches <> nil then
  begin
    for I := 0 to IfBranches.Count - 1 do
    begin
      IfBranches[I].Condition.Free();
      IfBranches[I].Body.Free();
    end;
    IfBranches.Free();
  end;
  ForIterExpr.Free();
  ForBody.Free();
  ForElseBody.Free();
  SetExpr.Free();
  SetBlockBody.Free();
  MacroParams.Free();
  if MacroDefaults <> nil then
  begin
    for LPair in MacroDefaults do
      LPair.Value.Free();
    MacroDefaults.Free();
  end;
  MacroBody.Free();
  CallExpr.Free();
  CallBody.Free();
  inherited Destroy();
end;

{ TJinjaLexer }

class function TJinjaLexer.SplitSegments(
  const ATemplate: string): TList<TJinjaSegment>;
var
  LLen: Integer;
  LPos: Integer;
  LLine: Integer;
  LSeg: TJinjaSegment;
  LTextStart: Integer;
  LContent: string;
  LTagEnd: string;
  LEndPos: Integer;
  LC, LC2: Char;

  procedure AddTextSeg(const AText: string);
  begin
    if AText <> '' then
    begin
      LSeg.Kind := jskText;
      LSeg.Content := AText;
      LSeg.TrimLeft := False;
      LSeg.TrimRight := False;
      LSeg.Line := LLine;
      Result.Add(LSeg);
    end;
  end;

  function CountNewlines(const AStr: string; AFrom, ATo: Integer): Integer;
  var
    LI: Integer;
  begin
    Result := 0;
    for LI := AFrom to ATo do
    begin
      if AStr[LI] = #10 then
        Inc(Result);
    end;
  end;

begin
  Result := TList<TJinjaSegment>.Create();
  LLen := Length(ATemplate);
  LPos := 1;
  LLine := 1;
  LTextStart := 1;

  while LPos <= LLen - 1 do
  begin
    LC := ATemplate[LPos];
    LC2 := ATemplate[LPos + 1];

    // Check for tag openers: {{, {%, {#
    if LC = '{' then
    begin
      if (LC2 = '{') or (LC2 = '%') or (LC2 = '#') then
      begin
        // Emit preceding text
        if LPos > LTextStart then
        begin
          AddTextSeg(Copy(ATemplate, LTextStart, LPos - LTextStart));
          LLine := LLine + CountNewlines(ATemplate, LTextStart, LPos - 1);
        end;

        // Determine tag type and end marker
        LSeg.Line := LLine;
        if LC2 = '{' then
        begin
          LSeg.Kind := jskExpr;
          LTagEnd := '}}';
        end
        else if LC2 = '%' then
        begin
          LSeg.Kind := jskStmt;
          LTagEnd := '%}';
        end
        else
        begin
          LSeg.Kind := jskComment;
          LTagEnd := '#}';
        end;

        // Check for left trim: {{- or {%- or {#-
        LSeg.TrimLeft := (LPos + 2 <= LLen) and (ATemplate[LPos + 2] = '-');

        // Find closing tag
        LEndPos := Pos(LTagEnd, ATemplate, LPos + 2);
        if LEndPos = 0 then
          raise Exception.CreateFmt(
            'Unclosed Jinja tag at line %d', [LLine]);

        // Check for right trim: -}} or -%} or -#}
        LSeg.TrimRight := (LEndPos > 1) and (ATemplate[LEndPos - 1] = '-');

        // Extract content between delimiters
        if LSeg.TrimLeft then
          LContent := Copy(ATemplate, LPos + 3, LEndPos - LPos - 3)
        else
          LContent := Copy(ATemplate, LPos + 2, LEndPos - LPos - 2);

        if LSeg.TrimRight then
        begin
          if (Length(LContent) > 0) and
             (LContent[Length(LContent)] = '-') then
            LContent := Copy(LContent, 1, Length(LContent) - 1);
        end;

        LSeg.Content := Trim(LContent);

        // Count newlines inside the tag
        LLine := LLine + CountNewlines(ATemplate, LPos, LEndPos + 1);

        Result.Add(LSeg);

        LPos := LEndPos + 2; // skip past closing tag
        LTextStart := LPos;
        Continue;
      end;
    end;

    Inc(LPos);
  end;

  // Remaining text
  if LTextStart <= LLen then
    AddTextSeg(Copy(ATemplate, LTextStart, LLen - LTextStart + 1));
end;

class function TJinjaLexer.TokenizeExpr(const AContent: string;
  const ALine: Integer): TList<TJinjaToken>;
var
  LLen: Integer;
  LPos: Integer;
  LTok: TJinjaToken;
  LC: Char;
  LStart: Integer;
  LIdent: string;
  LQuote: Char;
  LStrBuilder: TStringBuilder;

  procedure AddToken(const AKind: TJinjaTokenKind; const AValue: string);
  begin
    LTok.Kind := AKind;
    LTok.Value := AValue;
    LTok.Line := ALine;
    LTok.Col := LPos;
    Result.Add(LTok);
  end;

begin
  Result := TList<TJinjaToken>.Create();
  LLen := Length(AContent);
  LPos := 1;

  while LPos <= LLen do
  begin
    LC := AContent[LPos];

    // Skip whitespace
    if (LC = ' ') or (LC = #9) or (LC = #10) or (LC = #13) then
    begin
      Inc(LPos);
      Continue;
    end;

    // String literal
    if (LC = '''') or (LC = '"') then
    begin
      LQuote := LC;
      Inc(LPos);
      LStrBuilder := TStringBuilder.Create();
      try
        while (LPos <= LLen) and (AContent[LPos] <> LQuote) do
        begin
          if (AContent[LPos] = '\') and (LPos + 1 <= LLen) then
          begin
            Inc(LPos);
            case AContent[LPos] of
              'n': LStrBuilder.Append(#10);
              'r': LStrBuilder.Append(#13);
              't': LStrBuilder.Append(#9);
              '\': LStrBuilder.Append('\');
              '''': LStrBuilder.Append('''');
              '"': LStrBuilder.Append('"');
            else
              LStrBuilder.Append('\');
              LStrBuilder.Append(AContent[LPos]);
            end;
          end
          else
            LStrBuilder.Append(AContent[LPos]);
          Inc(LPos);
        end;
        if LPos <= LLen then
          Inc(LPos); // skip closing quote
        AddToken(jtkString, LStrBuilder.ToString());
      finally
        LStrBuilder.Free();
      end;
      Continue;
    end;

    // Numbers
    if CharInSet(LC, ['0'..'9']) then
    begin
      LStart := LPos;
      while (LPos <= LLen) and CharInSet(AContent[LPos], ['0'..'9']) do
        Inc(LPos);
      if (LPos <= LLen) and (AContent[LPos] = '.') and
         (LPos + 1 <= LLen) and CharInSet(AContent[LPos + 1], ['0'..'9']) then
      begin
        Inc(LPos);
        while (LPos <= LLen) and CharInSet(AContent[LPos], ['0'..'9']) do
          Inc(LPos);
        AddToken(jtkFloat, Copy(AContent, LStart, LPos - LStart));
      end
      else
        AddToken(jtkInteger, Copy(AContent, LStart, LPos - LStart));
      Continue;
    end;

    // Identifiers and keywords
    if CharInSet(LC, ['a'..'z', 'A'..'Z', '_']) then
    begin
      LStart := LPos;
      while (LPos <= LLen) and
            CharInSet(AContent[LPos], ['a'..'z', 'A'..'Z', '0'..'9', '_']) do
        Inc(LPos);
      LIdent := Copy(AContent, LStart, LPos - LStart);
      if LIdent = 'true' then
        AddToken(jtkTrue, LIdent)
      else if LIdent = 'True' then
        AddToken(jtkTrue, LIdent)
      else if LIdent = 'false' then
        AddToken(jtkFalse, LIdent)
      else if LIdent = 'False' then
        AddToken(jtkFalse, LIdent)
      else if LIdent = 'none' then
        AddToken(jtkNone, LIdent)
      else if LIdent = 'None' then
        AddToken(jtkNone, LIdent)
      else if LIdent = 'not' then
        AddToken(jtkNot, LIdent)
      else if LIdent = 'and' then
        AddToken(jtkAnd, LIdent)
      else if LIdent = 'or' then
        AddToken(jtkOr, LIdent)
      else if LIdent = 'in' then
        AddToken(jtkIn, LIdent)
      else if LIdent = 'is' then
        AddToken(jtkIs, LIdent)
      else if LIdent = 'if' then
        AddToken(jtkIf, LIdent)
      else if LIdent = 'else' then
        AddToken(jtkElse, LIdent)
      else
        AddToken(jtkIdent, LIdent);
      Continue;
    end;

    // Two-character operators
    if LPos + 1 <= LLen then
    begin
      if (LC = '=') and (AContent[LPos + 1] = '=') then
      begin AddToken(jtkEq, '=='); Inc(LPos, 2); Continue; end;
      if (LC = '!') and (AContent[LPos + 1] = '=') then
      begin AddToken(jtkNe, '!='); Inc(LPos, 2); Continue; end;
      if (LC = '<') and (AContent[LPos + 1] = '=') then
      begin AddToken(jtkLe, '<='); Inc(LPos, 2); Continue; end;
      if (LC = '>') and (AContent[LPos + 1] = '=') then
      begin AddToken(jtkGe, '>='); Inc(LPos, 2); Continue; end;
      if (LC = '/') and (AContent[LPos + 1] = '/') then
      begin AddToken(jtkIntDiv, '//'); Inc(LPos, 2); Continue; end;
    end;

    // Single-character operators/punctuation
    case LC of
      '(': begin AddToken(jtkLParen, '(');   Inc(LPos); Continue; end;
      ')': begin AddToken(jtkRParen, ')');   Inc(LPos); Continue; end;
      '[': begin AddToken(jtkLBracket, '['); Inc(LPos); Continue; end;
      ']': begin AddToken(jtkRBracket, ']'); Inc(LPos); Continue; end;
      '{': begin AddToken(jtkLBrace, '{');   Inc(LPos); Continue; end;
      '}': begin AddToken(jtkRBrace, '}');   Inc(LPos); Continue; end;
      ',': begin AddToken(jtkComma, ',');    Inc(LPos); Continue; end;
      '.': begin AddToken(jtkDot, '.');      Inc(LPos); Continue; end;
      '|': begin AddToken(jtkPipe, '|');     Inc(LPos); Continue; end;
      ':': begin AddToken(jtkColon, ':');    Inc(LPos); Continue; end;
      '~': begin AddToken(jtkTilde, '~');    Inc(LPos); Continue; end;
      '=': begin AddToken(jtkAssign, '=');   Inc(LPos); Continue; end;
      '<': begin AddToken(jtkLt, '<');       Inc(LPos); Continue; end;
      '>': begin AddToken(jtkGt, '>');       Inc(LPos); Continue; end;
      '+': begin AddToken(jtkPlus, '+');     Inc(LPos); Continue; end;
      '-': begin AddToken(jtkMinus, '-');    Inc(LPos); Continue; end;
      '*': begin AddToken(jtkMul, '*');      Inc(LPos); Continue; end;
      '/': begin AddToken(jtkDiv, '/');      Inc(LPos); Continue; end;
      '%': begin AddToken(jtkMod, '%');      Inc(LPos); Continue; end;
    end;

    // Unknown character -- skip
    Inc(LPos);
  end;

  // Add EOF
  AddToken(jtkEOF, '');
end;

{ TJinjaParser }

constructor TJinjaParser.Create();
begin
  inherited Create();
  FSegments := nil;
  FSegIdx := 0;
  FTokens := nil;
  FTokIdx := 0;
end;

destructor TJinjaParser.Destroy();
begin
  FTokens.Free();
  FSegments.Free();
  inherited Destroy();
end;

function TJinjaParser.AtEnd(): Boolean;
begin
  Result := FSegIdx >= FSegments.Count;
end;

function TJinjaParser.CurSegment(): TJinjaSegment;
begin
  Result := FSegments[FSegIdx];
end;

procedure TJinjaParser.NextSegment();
begin
  Inc(FSegIdx);
end;

procedure TJinjaParser.LoadTokens(const AContent: string;
  const ALine: Integer);
begin
  FTokens.Free();
  FTokens := TJinjaLexer.TokenizeExpr(AContent, ALine);
  FTokIdx := 0;
end;

function TJinjaParser.TokAtEnd(): Boolean;
begin
  Result := (FTokIdx >= FTokens.Count) or
            (FTokens[FTokIdx].Kind = jtkEOF);
end;

function TJinjaParser.TokPeek(): TJinjaToken;
begin
  if FTokIdx < FTokens.Count then
    Result := FTokens[FTokIdx]
  else
  begin
    Result.Kind := jtkEOF;
    Result.Value := '';
  end;
end;

function TJinjaParser.TokNext(): TJinjaToken;
begin
  Result := TokPeek();
  if FTokIdx < FTokens.Count then
    Inc(FTokIdx);
end;

function TJinjaParser.TokExpect(const AKind: TJinjaTokenKind): TJinjaToken;
begin
  Result := TokNext();
  if Result.Kind <> AKind then
    raise Exception.CreateFmt(
      'Expected token kind %d but got %d (%s)',
      [Ord(AKind), Ord(Result.Kind), Result.Value]);
end;

function TJinjaParser.TokMatch(const AKind: TJinjaTokenKind): Boolean;
begin
  if TokPeek().Kind = AKind then
  begin
    TokNext();
    Result := True;
  end
  else
    Result := False;
end;

function TJinjaParser.TokMatchIdent(const ANodeName: string): Boolean;
begin
  if (TokPeek().Kind = jtkIdent) and (TokPeek().Value = ANodeName) then
  begin
    TokNext();
    Result := True;
  end
  else
    Result := False;
end;

function TJinjaParser.IsStmt(const ANodeName: string): Boolean;
var
  LContent: string;
  LLen: Integer;
begin
  if AtEnd() then
    Exit(False);
  if CurSegment().Kind <> jskStmt then
    Exit(False);
  LContent := Trim(CurSegment().Content);
  LLen := Length(ANodeName);
  // Check keyword match with word boundary
  if not SameText(Copy(LContent, 1, LLen), ANodeName) then
    Exit(False);
  // Must be followed by whitespace, '(' or end of content
  Result := (Length(LContent) = LLen) or
            CharInSet(LContent[LLen + 1], [' ', #9, #10, #13, '(']);
end;

procedure TJinjaParser.ExpectStmt(const ANodeName: string);
begin
  if not IsStmt(ANodeName) then
    raise Exception.CreateFmt('Expected {%%%% %s %%%%}', [ANodeName]);
  LoadTokens(CurSegment().Content, CurSegment().Line);
  TokExpect(jtkIdent); // consume the keyword
  NextSegment();
end;

function TJinjaParser.Parse(const ATemplate: string): TJinjaNode;
begin
  FSegments.Free();
  FSegments := TJinjaLexer.SplitSegments(ATemplate);
  FSegIdx := 0;
  Result := ParseTemplate();
end;

function TJinjaParser.ParseTemplate(): TJinjaNode;
begin
  Result := ParseBlock([]);
end;

function TJinjaParser.ParseBlock(
  const AEndTags: TArray<string>): TJinjaNode;
var
  LBlock: TJinjaNode;
  LSeg: TJinjaSegment;
  LChild: TJinjaNode;
  LTextNode: TJinjaNode;
  LOutputNode: TJinjaNode;
  LTag: string;
  LFound: Boolean;
  LPrevSeg: TJinjaSegment;
  LNextSeg: TJinjaSegment;
  LText: string;
begin
  LBlock := TJinjaNode.Create(jnkBlock);
  LBlock.Children := TObjectList<TJinjaNode>.Create(True);

  while not AtEnd() do
  begin
    LSeg := CurSegment();

    // Check if this is an end tag we're looking for
    if (LSeg.Kind = jskStmt) and (Length(AEndTags) > 0) then
    begin
      LFound := False;
      for LTag in AEndTags do
      begin
        if SameText(Copy(Trim(LSeg.Content), 1, Length(LTag)), LTag) then
        begin
          LFound := True;
          Break;
        end;
      end;
      if LFound then
        Break;
    end;

    case LSeg.Kind of
      jskText:
      begin
        LText := LSeg.Content;
        // Handle whitespace trimming from adjacent tags
        // Check previous segment for TrimRight
        if (FSegIdx > 0) then
        begin
          LPrevSeg := FSegments[FSegIdx - 1];
          if LPrevSeg.TrimRight then
            LText := TrimLeft(LText);
        end;
        // Check next segment for TrimLeft
        if (FSegIdx + 1 < FSegments.Count) then
        begin
          LNextSeg := FSegments[FSegIdx + 1];
          if LNextSeg.TrimLeft then
            LText := TrimRight(LText);
        end;

        if LText <> '' then
        begin
          LTextNode := TJinjaNode.Create(jnkText);
          LTextNode.Text := LText;
          LBlock.Children.Add(LTextNode);
        end;
        NextSegment();
      end;

      jskExpr:
      begin
        LOutputNode := TJinjaNode.Create(jnkOutput);
        LoadTokens(LSeg.Content, LSeg.Line);
        LOutputNode.OutputExpr := ParseExpr();
        LBlock.Children.Add(LOutputNode);
        NextSegment();
      end;

      jskStmt:
      begin
        LChild := ParseStatement();
        if LChild <> nil then
          LBlock.Children.Add(LChild);
      end;

      jskComment:
        NextSegment(); // skip comments
    end;
  end;

  Result := LBlock;
end;

function TJinjaParser.ParseStatement(): TJinjaNode;
begin
  if IsStmt('if') then
    Result := ParseIf()
  else if IsStmt('for') then
    Result := ParseFor()
  else if IsStmt('set') then
    Result := ParseSet()
  else if IsStmt('macro') then
    Result := ParseMacro()
  else if IsStmt('call') then
    Result := ParseCallBlock()
  else
    // Unknown statement -- fail loud instead of silently skipping.
    // Stray end tags (endif/endfor/...) reaching here indicate template
    // structure errors; unsupported statements indicate missing features.
    raise Exception.CreateFmt('Unknown or unexpected statement: {%% %s %%}',
      [Trim(CurSegment().Content)]);
end;

function TJinjaParser.ParseIf(): TJinjaNode;
var
  LNode: TJinjaNode;
  LBranch: TJinjaIfBranch;
begin
  LNode := TJinjaNode.Create(jnkIf);
  LNode.IfBranches := TList<TJinjaIfBranch>.Create();

  // Parse {% if expr %}
  LoadTokens(CurSegment().Content, CurSegment().Line);
  TokExpect(jtkIf); // consume 'if' -- tokenized as jtkIf, not jtkIdent
  LBranch.Condition := ParseExpr();
  NextSegment();
  LBranch.Body := ParseBlock(['elif', 'else', 'endif']);
  LNode.IfBranches.Add(LBranch);

  // Parse {% elif expr %} chains
  while not AtEnd() and IsStmt('elif') do
  begin
    LoadTokens(CurSegment().Content, CurSegment().Line);
    TokExpect(jtkIdent); // consume 'elif'
    LBranch.Condition := ParseExpr();
    NextSegment();
    LBranch.Body := ParseBlock(['elif', 'else', 'endif']);
    LNode.IfBranches.Add(LBranch);
  end;

  // Parse {% else %}
  if not AtEnd() and IsStmt('else') then
  begin
    NextSegment();
    LBranch.Condition := nil;
    LBranch.Body := ParseBlock(['endif']);
    LNode.IfBranches.Add(LBranch);
  end;

  // Consume {% endif %}
  if not AtEnd() and IsStmt('endif') then
    NextSegment()
  else
    raise Exception.Create('Expected {% endif %}');

  Result := LNode;
end;

function TJinjaParser.ParseFor(): TJinjaNode;
var
  LNode: TJinjaNode;
  LTok: TJinjaToken;
begin
  LNode := TJinjaNode.Create(jnkFor);

  // Parse {% for var in expr %} or {% for key, value in expr %}
  LoadTokens(CurSegment().Content, CurSegment().Line);
  TokExpect(jtkIdent); // consume 'for'

  LTok := TokExpect(jtkIdent);
  LNode.ForVarName := LTok.Value;

  // Check for tuple unpacking: {% for key, value in ... %}
  // After this block: ForVarName = first name (key),
  // ForVarName2 = second name (value).
  if TokMatch(jtkComma) then
  begin
    LNode.ForVarName2 := TokExpect(jtkIdent).Value;
  end;

  TokExpect(jtkIn); // consume 'in'
  LNode.ForIterExpr := ParseExpr();

  // Check for 'recursive'
  if TokMatchIdent('recursive') then
    LNode.ForRecursive := True;

  NextSegment();
  LNode.ForBody := ParseBlock(['else', 'endfor']);

  // Optional {% else %}
  if not AtEnd() and IsStmt('else') then
  begin
    NextSegment();
    LNode.ForElseBody := ParseBlock(['endfor']);
  end;

  // Consume {% endfor %}
  if not AtEnd() and IsStmt('endfor') then
    NextSegment()
  else
    raise Exception.Create('Expected {% endfor %}');

  Result := LNode;
end;

function TJinjaParser.ParseSet(): TJinjaNode;
var
  LNode: TJinjaNode;
begin
  LNode := TJinjaNode.Create(jnkSetStmt);

  LoadTokens(CurSegment().Content, CurSegment().Line);
  TokExpect(jtkIdent); // consume 'set'

  LNode.SetVarName := TokExpect(jtkIdent).Value;

  // Check for dotted assignment: {% set ns.attr = expr %}
  if TokMatch(jtkDot) then
    LNode.SetAttrName := TokExpect(jtkIdent).Value;

  // Block set: {% set x %}...{% endset %} -- capture rendered output
  if not TokMatch(jtkAssign) then
  begin
    NextSegment();
    LNode.SetBlockBody := ParseBlock(['endset']);
    // Verify the closing tag actually exists before consuming it
    if AtEnd() or not IsStmt('endset') then
      raise Exception.Create('Expected {% endset %}');
    NextSegment(); // consume endset segment
    Result := LNode;
    Exit;
  end;

  LNode.SetExpr := ParseExpr();

  NextSegment();
  Result := LNode;
end;

function TJinjaParser.ParseMacro(): TJinjaNode;
var
  LNode: TJinjaNode;
  LParamName: string;
begin
  LNode := TJinjaNode.Create(jnkMacro);
  LNode.MacroParams := TStringList.Create();
  LNode.MacroDefaults := TDictionary<string, TJinjaExpr>.Create();

  LoadTokens(CurSegment().Content, CurSegment().Line);
  TokExpect(jtkIdent); // consume 'macro'

  LNode.MacroName := TokExpect(jtkIdent).Value;

  // Parse parameter list
  TokExpect(jtkLParen);
  if not TokMatch(jtkRParen) then
  begin
    repeat
      LParamName := TokExpect(jtkIdent).Value;
      LNode.MacroParams.Add(LParamName);
      if TokMatch(jtkAssign) then
        LNode.MacroDefaults.Add(LParamName, ParseExpr());
    until not TokMatch(jtkComma);
    TokExpect(jtkRParen);
  end;

  NextSegment();
  LNode.MacroBody := ParseBlock(['endmacro']);

  if not AtEnd() and IsStmt('endmacro') then
    NextSegment()
  else
    raise Exception.Create('Expected {% endmacro %}');

  Result := LNode;
end;

function TJinjaParser.ParseCallBlock(): TJinjaNode;
var
  LNode: TJinjaNode;
begin
  LNode := TJinjaNode.Create(jnkCallBlock);

  LoadTokens(CurSegment().Content, CurSegment().Line);
  TokExpect(jtkIdent); // consume 'call'

  // Parse the macro call expression (e.g., mymacro(arg1, arg2))
  LNode.CallExpr := ParseExpr();

  NextSegment();
  LNode.CallBody := ParseBlock(['endcall']);

  if not AtEnd() and IsStmt('endcall') then
    NextSegment()
  else
    raise Exception.Create('Expected {% endcall %}');

  Result := LNode;
end;

{ Expression parsing -- precedence climbing }

function TJinjaParser.ParseExpr(): TJinjaExpr;
begin
  Result := ParseConditionalExpr();
end;

function TJinjaParser.ParseConditionalExpr(): TJinjaExpr;
var
  LExpr: TJinjaExpr;
  LCond: TJinjaExpr;
  LFalse: TJinjaExpr;
  LResult: TJinjaExpr;
begin
  LExpr := ParseOrExpr();

  // Check for ternary: expr if cond else other
  if TokMatch(jtkIf) then
  begin
    LCond := ParseOrExpr();
    if TokMatch(jtkElse) then
      LFalse := ParseConditionalExpr()
    else
      LFalse := nil;

    LResult := TJinjaExpr.Create(jekConditional);
    LResult.CondTrue := LExpr;
    LResult.CondTest := LCond;
    LResult.CondFalse := LFalse;
    Result := LResult;
  end
  else
    Result := LExpr;
end;

function TJinjaParser.ParseOrExpr(): TJinjaExpr;
var
  LRight: TJinjaExpr;
  LBin: TJinjaExpr;
begin
  Result := ParseAndExpr();
  while TokMatch(jtkOr) do
  begin
    LRight := ParseAndExpr();
    LBin := TJinjaExpr.Create(jekBinary);
    LBin.BinOp := jtkOr;
    LBin.BinLeft := Result;
    LBin.BinRight := LRight;
    Result := LBin;
  end;
end;

function TJinjaParser.ParseAndExpr(): TJinjaExpr;
var
  LRight: TJinjaExpr;
  LBin: TJinjaExpr;
begin
  Result := ParseNotExpr();
  while TokMatch(jtkAnd) do
  begin
    LRight := ParseNotExpr();
    LBin := TJinjaExpr.Create(jekBinary);
    LBin.BinOp := jtkAnd;
    LBin.BinLeft := Result;
    LBin.BinRight := LRight;
    Result := LBin;
  end;
end;

function TJinjaParser.ParseNotExpr(): TJinjaExpr;
var
  LUnary: TJinjaExpr;
begin
  if TokMatch(jtkNot) then
  begin
    LUnary := TJinjaExpr.Create(jekUnary);
    LUnary.UnaryOp := jtkNot;
    LUnary.UnaryOperand := ParseNotExpr();
    Result := LUnary;
  end
  else
    Result := ParseCompareExpr();
end;

function TJinjaParser.ParseCompareExpr(): TJinjaExpr;
var
  LRight: TJinjaExpr;
  LBin: TJinjaExpr;
  LTest: TJinjaExpr;
  LOp: TJinjaTokenKind;
  LNegated: Boolean;
  LTestName: string;
  LTok: TJinjaToken;
begin
  Result := ParseAddExpr();

  while True do
  begin
    LOp := TokPeek().Kind;
    if LOp in [jtkEq, jtkNe, jtkLt, jtkGt, jtkLe, jtkGe] then
    begin
      TokNext();
      LRight := ParseAddExpr();
      LBin := TJinjaExpr.Create(jekBinary);
      LBin.BinOp := LOp;
      LBin.BinLeft := Result;
      LBin.BinRight := LRight;
      Result := LBin;
    end
    else if LOp = jtkIn then
    begin
      TokNext();
      LRight := ParseAddExpr();
      LBin := TJinjaExpr.Create(jekBinary);
      LBin.BinOp := jtkIn;
      LBin.BinLeft := Result;
      LBin.BinRight := LRight;
      Result := LBin;
    end
    else if (LOp = jtkNot) and (FTokIdx + 1 < FTokens.Count) and
            (FTokens[FTokIdx + 1].Kind = jtkIn) then
    begin
      // "not in" operator
      TokNext(); // consume 'not'
      TokNext(); // consume 'in'
      LRight := ParseAddExpr();
      LBin := TJinjaExpr.Create(jekUnary);
      LBin.UnaryOp := jtkNot;
      LBin.UnaryOperand := TJinjaExpr.Create(jekBinary);
      LBin.UnaryOperand.BinOp := jtkIn;
      LBin.UnaryOperand.BinLeft := Result;
      LBin.UnaryOperand.BinRight := LRight;
      Result := LBin;
    end
    else if LOp = jtkIs then
    begin
      TokNext(); // consume 'is'
      LNegated := False;
      if TokMatch(jtkNot) then
        LNegated := True;
      // Accept ident, none, true, false as test names
      // (Jinja `is none`, `is true`, `is false` use keywords as test names)
      LTok := TokNext();
      if LTok.Kind in [jtkIdent, jtkNone, jtkTrue, jtkFalse] then
        // Normalize to lowercase so 'is None' / 'is True' (Python
        // capitalization, common in HF templates) match the test table
        LTestName := LowerCase(LTok.Value)
      else
        raise Exception.CreateFmt(
          'Expected test name (ident/none/true/false) but got token kind %d (%s)',
          [Ord(LTok.Kind), LTok.Value]);

      LTest := TJinjaExpr.Create(jekTest);
      LTest.TestExpr := Result;
      LTest.TestName := LTestName;
      LTest.TestNegated := LNegated;
      LTest.TestArgs := TObjectList<TJinjaExpr>.Create(True);

      // Some tests take arguments in parens
      if TokMatch(jtkLParen) then
      begin
        if not TokMatch(jtkRParen) then
        begin
          LTest.TestArgs.Add(ParseExpr());
          while TokMatch(jtkComma) do
            LTest.TestArgs.Add(ParseExpr());
          TokExpect(jtkRParen);
        end;
      end;

      Result := LTest;
    end
    else
      Break;
  end;
end;

function TJinjaParser.ParseAddExpr(): TJinjaExpr;
var
  LRight: TJinjaExpr;
  LBin: TJinjaExpr;
  LOp: TJinjaTokenKind;
begin
  Result := ParseMulExpr();
  while TokPeek().Kind in [jtkPlus, jtkMinus, jtkTilde] do
  begin
    LOp := TokNext().Kind;
    LRight := ParseMulExpr();
    LBin := TJinjaExpr.Create(jekBinary);
    LBin.BinOp := LOp;
    LBin.BinLeft := Result;
    LBin.BinRight := LRight;
    Result := LBin;
  end;
end;

function TJinjaParser.ParseMulExpr(): TJinjaExpr;
var
  LRight: TJinjaExpr;
  LBin: TJinjaExpr;
  LOp: TJinjaTokenKind;
begin
  Result := ParseUnaryExpr();
  while TokPeek().Kind in [jtkMul, jtkDiv, jtkIntDiv, jtkMod] do
  begin
    LOp := TokNext().Kind;
    LRight := ParseUnaryExpr();
    LBin := TJinjaExpr.Create(jekBinary);
    LBin.BinOp := LOp;
    LBin.BinLeft := Result;
    LBin.BinRight := LRight;
    Result := LBin;
  end;
end;

function TJinjaParser.ParseUnaryExpr(): TJinjaExpr;
var
  LUnary: TJinjaExpr;
begin
  if TokPeek().Kind = jtkMinus then
  begin
    TokNext();
    LUnary := TJinjaExpr.Create(jekUnary);
    LUnary.UnaryOp := jtkMinus;
    LUnary.UnaryOperand := ParseUnaryExpr();
    Result := LUnary;
  end
  else
    Result := ParsePostfixExpr();
end;

function TJinjaParser.ParsePostfixExpr(): TJinjaExpr;
var
  LExpr: TJinjaExpr;
  LMember: TJinjaExpr;
  LCall: TJinjaExpr;
  LIndex: TJinjaExpr;
  LSlice: TJinjaExpr;
  LStart: TJinjaExpr;
  LStop: TJinjaExpr;
  LStep: TJinjaExpr;
  LName: string;
  LKwName: string;
begin
  LExpr := ParsePrimaryExpr();

  while True do
  begin
    // Dot member access
    if TokMatch(jtkDot) then
    begin
      LName := TokExpect(jtkIdent).Value;

      // Check if this is a method call: obj.method(args)
      if TokPeek().Kind = jtkLParen then
      begin
        TokNext(); // consume '('
        LCall := TJinjaExpr.Create(jekMethodCall);
        LCall.MemberObj := LExpr;
        LCall.MemberName := LName;
        LCall.CallArgs := TObjectList<TJinjaExpr>.Create(True);
        LCall.CallKwargs := TDictionary<string, TJinjaExpr>.Create();
        if not TokMatch(jtkRParen) then
        begin
          LCall.CallArgs.Add(ParseExpr());
          while TokMatch(jtkComma) do
            LCall.CallArgs.Add(ParseExpr());
          TokExpect(jtkRParen);
        end;
        LExpr := LCall;
      end
      else
      begin
        LMember := TJinjaExpr.Create(jekMember);
        LMember.MemberObj := LExpr;
        LMember.MemberName := LName;
        LExpr := LMember;
      end;
    end
    // Bracket index/slice
    else if TokMatch(jtkLBracket) then
    begin
      if TokPeek().Kind = jtkColon then
      begin
        // [:stop] or [:] or [::step] or [:stop:step]
        TokNext(); // consume ':'
        LStop := nil;
        LStep := nil;
        if (TokPeek().Kind <> jtkRBracket) and (TokPeek().Kind <> jtkColon) then
          LStop := ParseExpr();
        if TokMatch(jtkColon) then
        begin
          if TokPeek().Kind <> jtkRBracket then
            LStep := ParseExpr();
        end;
        TokExpect(jtkRBracket);

        LSlice := TJinjaExpr.Create(jekSlice);
        LSlice.SliceObj := LExpr;
        LSlice.SliceStart := nil;
        LSlice.SliceStop := LStop;
        LSlice.SliceStep := LStep;
        LExpr := LSlice;
      end
      else
      begin
        LStart := ParseExpr();
        if TokMatch(jtkColon) then
        begin
          // [start:stop] or [start:] or [start::step] or [start:stop:step]
          LStop := nil;
          LStep := nil;
          if (TokPeek().Kind <> jtkRBracket) and (TokPeek().Kind <> jtkColon) then
            LStop := ParseExpr();
          if TokMatch(jtkColon) then
          begin
            if TokPeek().Kind <> jtkRBracket then
              LStep := ParseExpr();
          end;
          TokExpect(jtkRBracket);

          LSlice := TJinjaExpr.Create(jekSlice);
          LSlice.SliceObj := LExpr;
          LSlice.SliceStart := LStart;
          LSlice.SliceStop := LStop;
          LSlice.SliceStep := LStep;
          LExpr := LSlice;
        end
        else
        begin
          TokExpect(jtkRBracket);
          LIndex := TJinjaExpr.Create(jekIndex);
          LIndex.MemberObj := LExpr;
          LIndex.IndexExpr := LStart;
          LExpr := LIndex;
        end;
      end;
    end
    // Function call
    else if TokPeek().Kind = jtkLParen then
    begin
      TokNext(); // consume '('
      LCall := TJinjaExpr.Create(jekCall);
      LCall.CallTarget := LExpr;
      LCall.CallArgs := TObjectList<TJinjaExpr>.Create(True);
      LCall.CallKwargs := TDictionary<string, TJinjaExpr>.Create();
      if not TokMatch(jtkRParen) then
      begin
        repeat
          // Check for keyword argument: name=expr
          if (TokPeek().Kind = jtkIdent) and
             (FTokIdx + 1 < FTokens.Count) and
             (FTokens[FTokIdx + 1].Kind = jtkAssign) then
          begin
            LKwName := TokNext().Value;
            TokNext(); // consume '='
            LCall.CallKwargs.Add(LKwName, ParseExpr());
          end
          else
            LCall.CallArgs.Add(ParseExpr());
        until not TokMatch(jtkComma);
        TokExpect(jtkRParen);
      end;
      LExpr := LCall;
    end
    // Filter pipe
    else if TokPeek().Kind = jtkPipe then
    begin
      LExpr := ParseFilterChain(LExpr);
    end
    else
      Break;
  end;

  Result := LExpr;
end;

function TJinjaParser.ParsePrimaryExpr(): TJinjaExpr;
var
  LTok: TJinjaToken;
  LResult: TJinjaExpr;
  LKey: TJinjaExpr;
  LVal: TJinjaExpr;
begin
  LTok := TokPeek();

  case LTok.Kind of
    jtkString:
    begin
      TokNext();
      LResult := TJinjaExpr.Create(jekLiteral);
      LResult.LitKind := jtkString;
      LResult.LitValue := LTok.Value;
      Result := LResult;
    end;

    jtkInteger:
    begin
      TokNext();
      LResult := TJinjaExpr.Create(jekLiteral);
      LResult.LitKind := jtkInteger;
      LResult.LitValue := LTok.Value;
      Result := LResult;
    end;

    jtkFloat:
    begin
      TokNext();
      LResult := TJinjaExpr.Create(jekLiteral);
      LResult.LitKind := jtkFloat;
      LResult.LitValue := LTok.Value;
      Result := LResult;
    end;

    jtkTrue:
    begin
      TokNext();
      LResult := TJinjaExpr.Create(jekLiteral);
      LResult.LitKind := jtkTrue;
      LResult.LitValue := 'true';
      Result := LResult;
    end;

    jtkFalse:
    begin
      TokNext();
      LResult := TJinjaExpr.Create(jekLiteral);
      LResult.LitKind := jtkFalse;
      LResult.LitValue := 'false';
      Result := LResult;
    end;

    jtkNone:
    begin
      TokNext();
      LResult := TJinjaExpr.Create(jekLiteral);
      LResult.LitKind := jtkNone;
      LResult.LitValue := 'none';
      Result := LResult;
    end;

    jtkIdent:
    begin
      TokNext();
      LResult := TJinjaExpr.Create(jekVar);
      LResult.VarName := LTok.Value;
      Result := LResult;
    end;

    jtkLParen:
    begin
      TokNext(); // consume '('
      Result := ParseExpr();
      TokExpect(jtkRParen);
    end;

    jtkLBracket:
    begin
      // Array literal: [a, b, c]
      TokNext(); // consume '['
      LResult := TJinjaExpr.Create(jekArray);
      LResult.ArrayItems := TObjectList<TJinjaExpr>.Create(True);
      if not TokMatch(jtkRBracket) then
      begin
        LResult.ArrayItems.Add(ParseExpr());
        while TokMatch(jtkComma) do
        begin
          if TokPeek().Kind = jtkRBracket then
            Break; // trailing comma
          LResult.ArrayItems.Add(ParseExpr());
        end;
        TokExpect(jtkRBracket);
      end;
      Result := LResult;
    end;

    jtkLBrace:
    begin
      // Dict literal: {key: value, ...}
      TokNext(); // consume '{'
      LResult := TJinjaExpr.Create(jekDict);
      LResult.DictPairs := TList<TPair<TJinjaExpr, TJinjaExpr>>.Create();
      if not TokMatch(jtkRBrace) then
      begin
        LKey := ParseExpr();
        TokExpect(jtkColon);
        LVal := ParseExpr();
        LResult.DictPairs.Add(TPair<TJinjaExpr, TJinjaExpr>.Create(LKey, LVal));
        while TokMatch(jtkComma) do
        begin
          if TokPeek().Kind = jtkRBrace then
            Break;
          LKey := ParseExpr();
          TokExpect(jtkColon);
          LVal := ParseExpr();
          LResult.DictPairs.Add(TPair<TJinjaExpr, TJinjaExpr>.Create(LKey, LVal));
        end;
        TokExpect(jtkRBrace);
      end;
      Result := LResult;
    end;

  else
    raise Exception.CreateFmt('Unexpected token: %s (%d)',
      [LTok.Value, Ord(LTok.Kind)]);
  end;
end;

function TJinjaParser.ParseFilterChain(
  const AExpr: TJinjaExpr): TJinjaExpr;
var
  LFilter: TJinjaExpr;
  LName: string;
  LKwName: string;
begin
  Result := AExpr;

  while TokMatch(jtkPipe) do
  begin
    LName := TokExpect(jtkIdent).Value;
    LFilter := TJinjaExpr.Create(jekFilter);
    LFilter.FilterExpr := Result;
    LFilter.FilterName := LName;
    LFilter.FilterArgs := TObjectList<TJinjaExpr>.Create(True);
    LFilter.FilterKwargs := TDictionary<string, TJinjaExpr>.Create();

    // Optional arguments
    if TokMatch(jtkLParen) then
    begin
      if not TokMatch(jtkRParen) then
      begin
        repeat
          // Check for keyword argument
          if (TokPeek().Kind = jtkIdent) and
             (FTokIdx + 1 < FTokens.Count) and
             (FTokens[FTokIdx + 1].Kind = jtkAssign) then
          begin
            LKwName := TokNext().Value;
            TokNext(); // consume '='
            LFilter.FilterKwargs.Add(LKwName, ParseExpr());
          end
          else
            LFilter.FilterArgs.Add(ParseExpr());
        until not TokMatch(jtkComma);
        TokExpect(jtkRParen);
      end;
    end;

    Result := LFilter;
  end;
end;

{ TJinjaRenderer }

constructor TJinjaRenderer.Create();
begin
  inherited Create();
  FPool := nil;
  FOutput := nil;
end;

destructor TJinjaRenderer.Destroy();
begin
  inherited Destroy();
end;

procedure TJinjaRenderer.RegisterBuiltins(const ACtx: TJinjaContext);
var
  LPool: TJinjaValuePool;
begin
  LPool := ACtx.GetPool();

  // range() function
  ACtx.SetLocal('range', LPool.NewCallable(
    function(const AArgs: TArray<TJinjaValue>;
      const APool: TObjectList<TJinjaValue>): TJinjaValue
    var
      LStart, LStop, LStep, LI: Int64;
      LResult: TJinjaValue;
      LItem: TJinjaValue;
    begin
      LStart := 0;
      LStop := 0;
      LStep := 1;
      if Length(AArgs) = 1 then
        LStop := AArgs[0].AsInt()
      else if Length(AArgs) >= 2 then
      begin
        LStart := AArgs[0].AsInt();
        LStop := AArgs[1].AsInt();
      end;
      if Length(AArgs) >= 3 then
        LStep := AArgs[2].AsInt();

      LResult := TJinjaValue.Create();
      LResult.FKind := jvkArray;
      LResult.FArrayItems := TList<TJinjaValue>.Create();
      APool.Add(LResult);

      LI := LStart;
      while ((LStep > 0) and (LI < LStop)) or
            ((LStep < 0) and (LI > LStop)) do
      begin
        LItem := TJinjaValue.Create();
        LItem.FKind := jvkInt;
        LItem.FIntValue := LI;
        APool.Add(LItem);
        LResult.ArrayAdd(LItem);
        Inc(LI, LStep);
      end;
      Result := LResult;
    end, 'range'));

  // namespace() function
  ACtx.SetLocal('namespace', LPool.NewCallable(
    function(const AArgs: TArray<TJinjaValue>;
      const APool: TObjectList<TJinjaValue>): TJinjaValue
    begin
      Result := TJinjaValue.Create();
      Result.FKind := jvkDict;
      Result.FDictMap := TDictionary<string, TJinjaValue>.Create();
      Result.FDictKeys := TStringList.Create();
      APool.Add(Result);
    end, 'namespace'));

  // joiner() function
  ACtx.SetLocal('joiner', LPool.NewCallable(
    function(const AArgs: TArray<TJinjaValue>;
      const APool: TObjectList<TJinjaValue>): TJinjaValue
    var
      LSep: string;
      LFirst: Boolean;
    begin
      if Length(AArgs) > 0 then
        LSep := AArgs[0].AsString()
      else
        LSep := ', ';
      LFirst := True;

      Result := TJinjaValue.Create();
      Result.FKind := jvkCallable;
      Result.FCallableName := 'joiner_instance';
      APool.Add(Result);

      Result.FCallable :=
        function(const AInnerArgs: TArray<TJinjaValue>;
          const AInnerPool: TObjectList<TJinjaValue>): TJinjaValue
        var
          LVal: TJinjaValue;
        begin
          LVal := TJinjaValue.Create();
          LVal.FKind := jvkString;
          AInnerPool.Add(LVal);
          if LFirst then
          begin
            LVal.FStringValue := '';
            LFirst := False;
          end
          else
            LVal.FStringValue := LSep;
          Result := LVal;
        end;
    end, 'joiner'));
end;

function TJinjaRenderer.Render(const ARoot: TJinjaNode;
  const ACtx: TJinjaContext): string;
begin
  FPool := ACtx.GetPool();
  FOutput := TStringBuilder.Create();
  try
    RegisterBuiltins(ACtx);
    Exec(ARoot, ACtx);
    Result := FOutput.ToString();
  finally
    FOutput.Free();
    FOutput := nil;
    FPool := nil;
  end;
end;

procedure TJinjaRenderer.Exec(const ANode: TJinjaNode;
  const ACtx: TJinjaContext);
var
  I: Integer;
  LVal: TJinjaValue;
  LIterVal: TJinjaValue;
  LLoopCtx: TJinjaContext;
  LLoopVar: TJinjaValue;
  LLen: Integer;
  LBranch: TJinjaIfBranch;
  LCondVal: TJinjaValue;
  LMacroBody: TJinjaNode;
  LMacroParams: TStringList;
  LMacroDefaults: TDictionary<string, TJinjaExpr>;
  LCallBody: TJinjaNode;
  LSavedOutput: TStringBuilder;
begin
  case ANode.NodeKind of
    jnkText:
      FOutput.Append(ANode.Text);

    jnkOutput:
    begin
      LVal := Eval(ANode.OutputExpr, ACtx);
      FOutput.Append(LVal.ToOutput());
    end;

    jnkBlock:
    begin
      for I := 0 to ANode.Children.Count - 1 do
        Exec(ANode.Children[I], ACtx);
    end;

    jnkIf:
    begin
      for I := 0 to ANode.IfBranches.Count - 1 do
      begin
        LBranch := ANode.IfBranches[I];
        if LBranch.Condition = nil then
        begin
          // else branch
          Exec(LBranch.Body, ACtx);
          Break;
        end;
        LCondVal := Eval(LBranch.Condition, ACtx);
        if LCondVal.IsTruthy() then
        begin
          Exec(LBranch.Body, ACtx);
          Break;
        end;
      end;
    end;

    jnkFor:
    begin
      LIterVal := Eval(ANode.ForIterExpr, ACtx);

      if LIterVal.IsArray() then
      begin
        LLen := LIterVal.ArrayCount();
        if LLen = 0 then
        begin
          if ANode.ForElseBody <> nil then
            Exec(ANode.ForElseBody, ACtx);
        end
        else
        begin
          for I := 0 to LLen - 1 do
          begin
            LLoopCtx := TJinjaContext.Create(ACtx);
            try
              LVal := LIterVal.ArrayGet(I);

              // Tuple unpacking for dict.items()
              if (ANode.ForVarName2 <> '') and LVal.IsArray() and
                 (LVal.ArrayCount() >= 2) then
              begin
                LLoopCtx.SetLocal(ANode.ForVarName, LVal.ArrayGet(0));
                LLoopCtx.SetLocal(ANode.ForVarName2, LVal.ArrayGet(1));
              end
              else
                LLoopCtx.SetLocal(ANode.ForVarName, LVal);

              // Set loop variable
              LLoopVar := FPool.NewDict();
              LLoopVar.DictSet('index', FPool.NewInt(I + 1));
              LLoopVar.DictSet('index0', FPool.NewInt(I));
              LLoopVar.DictSet('first', FPool.NewBool(I = 0));
              LLoopVar.DictSet('last', FPool.NewBool(I = LLen - 1));
              LLoopVar.DictSet('length', FPool.NewInt(LLen));
              LLoopVar.DictSet('revindex', FPool.NewInt(LLen - I));
              LLoopVar.DictSet('revindex0', FPool.NewInt(LLen - I - 1));
              LLoopCtx.SetLocal('loop', LLoopVar);

              Exec(ANode.ForBody, LLoopCtx);
            finally
              LLoopCtx.Free();
            end;
          end;
        end;
      end
      else if LIterVal.IsDict() then
      begin
        // Iterate over dict keys
        LLen := LIterVal.FDictKeys.Count;
        if LLen = 0 then
        begin
          if ANode.ForElseBody <> nil then
            Exec(ANode.ForElseBody, ACtx);
        end
        else
        begin
          for I := 0 to LLen - 1 do
          begin
            LLoopCtx := TJinjaContext.Create(ACtx);
            try
              LLoopCtx.SetLocal(ANode.ForVarName,
                FPool.NewString(LIterVal.FDictKeys[I]));

              LLoopVar := FPool.NewDict();
              LLoopVar.DictSet('index', FPool.NewInt(I + 1));
              LLoopVar.DictSet('index0', FPool.NewInt(I));
              LLoopVar.DictSet('first', FPool.NewBool(I = 0));
              LLoopVar.DictSet('last', FPool.NewBool(I = LLen - 1));
              LLoopVar.DictSet('length', FPool.NewInt(LLen));
              LLoopCtx.SetLocal('loop', LLoopVar);

              Exec(ANode.ForBody, LLoopCtx);
            finally
              LLoopCtx.Free();
            end;
          end;
        end;
      end;
    end;

    jnkSetStmt:
    begin
      // Block set: {% set x %}...{% endset %} -- capture rendered output
      if ANode.SetBlockBody <> nil then
      begin
        LSavedOutput := FOutput;
        FOutput := TStringBuilder.Create();
        try
          Exec(ANode.SetBlockBody, ACtx);
          LVal := FPool.NewString(FOutput.ToString());
        finally
          FOutput.Free();
          FOutput := LSavedOutput;
        end;
      end
      else
        LVal := Eval(ANode.SetExpr, ACtx);
      if ANode.SetAttrName <> '' then
      begin
        // Dotted assignment: {% set ns.attr = expr %}
        LIterVal := ACtx.Get(ANode.SetVarName);
        if (LIterVal <> nil) and LIterVal.IsDict() then
          LIterVal.DictSet(ANode.SetAttrName, LVal)
        else
          raise Exception.CreateFmt(
            'Cannot set attribute on non-dict: %s', [ANode.SetVarName]);
      end
      else
        ACtx.SetVar(ANode.SetVarName, LVal);
    end;

    jnkMacro:
    begin
      // Register macro as a callable in the context
      LMacroBody := ANode.MacroBody;
      LMacroParams := ANode.MacroParams;
      LMacroDefaults := ANode.MacroDefaults;

      LVal := FPool.NewCallable(
        function(const AArgs: TArray<TJinjaValue>;
          const APool: TObjectList<TJinjaValue>): TJinjaValue
        var
          LMacroCtx: TJinjaContext;
          LParamIdx: Integer;
          LParamName: string;
          LDefExpr: TJinjaExpr;
          LOldOutput: TStringBuilder;
        begin
          LMacroCtx := TJinjaContext.Create(ACtx);
          try
            // Bind parameters. A nil slot means the caller supplied kwargs
            // that extended the args array without filling this position,
            // so it must fall through to the declared default.
            for LParamIdx := 0 to LMacroParams.Count - 1 do
            begin
              LParamName := LMacroParams[LParamIdx];
              if (LParamIdx < Length(AArgs)) and
                 (AArgs[LParamIdx] <> nil) then
                LMacroCtx.SetLocal(LParamName, AArgs[LParamIdx])
              else if LMacroDefaults.TryGetValue(LParamName, LDefExpr) then
                LMacroCtx.SetLocal(LParamName, Eval(LDefExpr, ACtx))
              else
                LMacroCtx.SetLocal(LParamName, FPool.NewNone());
            end;

            // Render macro body to string
            LOldOutput := FOutput;
            FOutput := TStringBuilder.Create();
            try
              Exec(LMacroBody, LMacroCtx);
              Result := FPool.NewString(FOutput.ToString());
            finally
              FOutput.Free();
              FOutput := LOldOutput;
            end;
          finally
            LMacroCtx.Free();
          end;
        end, ANode.MacroName);
      LVal.FCallableParams := TStringList.Create();
      LVal.FCallableParams.Assign(ANode.MacroParams);
      ACtx.SetLocal(ANode.MacroName, LVal);
    end;

    jnkCallBlock:
    begin
      // Render the call body and make it available via caller()
      LCallBody := ANode.CallBody;

      ACtx.SetLocal('caller', FPool.NewCallable(
        function(const AArgs: TArray<TJinjaValue>;
          const APool: TObjectList<TJinjaValue>): TJinjaValue
        var
          LOldOutput: TStringBuilder;
        begin
          LOldOutput := FOutput;
          FOutput := TStringBuilder.Create();
          try
            Exec(LCallBody, ACtx);
            Result := FPool.NewString(FOutput.ToString());
          finally
            FOutput.Free();
            FOutput := LOldOutput;
          end;
        end, 'caller'));

      // Now evaluate the call expression (which will invoke the macro)
      LVal := Eval(ANode.CallExpr, ACtx);
      FOutput.Append(LVal.ToOutput());
    end;
  end;
end;

function TJinjaRenderer.Eval(const AExpr: TJinjaExpr;
  const ACtx: TJinjaContext): TJinjaValue;
var
  LLeft, LRight, LVal, LObj, LTarget: TJinjaValue;
  LKey: string;
  I: Integer;
  LArgs: TArray<TJinjaValue>;
  LBoolResult: Boolean;
  LResult: TJinjaValue;
  LStart, LStop, LStep, LLen: Integer;
  LSliceArr: TJinjaValue;
  LPair: TPair<TJinjaExpr, TJinjaExpr>;
  LStrPair: TPair<string, TJinjaExpr>;
  LKwargs: TDictionary<string, TJinjaValue>;
  LFilterArgs: TArray<TJinjaValue>;
  LTestArgs: TArray<TJinjaValue>;
  LParts: TArray<string>;
begin
  case AExpr.ExprKind of
    jekLiteral:
    begin
      case AExpr.LitKind of
        jtkString:  Result := FPool.NewString(AExpr.LitValue);
        jtkInteger: Result := FPool.NewInt(StrToInt64(AExpr.LitValue));
        jtkFloat:   Result := FPool.NewFloat(
          StrToFloat(AExpr.LitValue, TFormatSettings.Invariant));
        jtkTrue:    Result := FPool.NewBool(True);
        jtkFalse:   Result := FPool.NewBool(False);
        jtkNone:    Result := FPool.NewNone();
      else
        Result := FPool.NewNone();
      end;
    end;

    jekVar:
    begin
      // Missing variable is Jinja Undefined, not none
      LVal := ACtx.Get(AExpr.VarName);
      if LVal = nil then
        Result := FPool.NewUndefined()
      else
        Result := LVal;
    end;

    jekBinary:
    begin
      LLeft := Eval(AExpr.BinLeft, ACtx);

      // Short-circuit for and/or
      if AExpr.BinOp = jtkAnd then
      begin
        if not LLeft.IsTruthy() then
          Exit(LLeft);
        Exit(Eval(AExpr.BinRight, ACtx));
      end;
      if AExpr.BinOp = jtkOr then
      begin
        if LLeft.IsTruthy() then
          Exit(LLeft);
        Exit(Eval(AExpr.BinRight, ACtx));
      end;

      LRight := Eval(AExpr.BinRight, ACtx);

      case AExpr.BinOp of
        jtkPlus:
        begin
          if LLeft.IsString() and LRight.IsString() then
            Result := FPool.NewString(LLeft.AsString() + LRight.AsString())
          else if LLeft.IsInt() and LRight.IsInt() then
            Result := FPool.NewInt(LLeft.AsInt() + LRight.AsInt())
          else if LLeft.IsNumber() and LRight.IsNumber() then
            Result := FPool.NewFloat(LLeft.AsFloat() + LRight.AsFloat())
          else if LLeft.IsArray() and LRight.IsArray() then
          begin
            LResult := FPool.NewArray();
            for I := 0 to LLeft.ArrayCount() - 1 do
              LResult.ArrayAdd(LLeft.ArrayGet(I));
            for I := 0 to LRight.ArrayCount() - 1 do
              LResult.ArrayAdd(LRight.ArrayGet(I));
            Result := LResult;
          end
          else
            Result := FPool.NewString(LLeft.ToOutput() + LRight.ToOutput());
        end;

        jtkMinus:
        begin
          if LLeft.IsInt() and LRight.IsInt() then
            Result := FPool.NewInt(LLeft.AsInt() - LRight.AsInt())
          else
            Result := FPool.NewFloat(LLeft.AsFloat() - LRight.AsFloat());
        end;

        jtkMul:
        begin
          if LLeft.IsInt() and LRight.IsInt() then
            Result := FPool.NewInt(LLeft.AsInt() * LRight.AsInt())
          else
            Result := FPool.NewFloat(LLeft.AsFloat() * LRight.AsFloat());
        end;

        jtkDiv:
          Result := FPool.NewFloat(LLeft.AsFloat() / LRight.AsFloat());

        jtkIntDiv:
          Result := FPool.NewInt(LLeft.AsInt() div LRight.AsInt());

        jtkMod:
          Result := FPool.NewInt(LLeft.AsInt() mod LRight.AsInt());

        jtkTilde:
          Result := FPool.NewString(LLeft.ToOutput() + LRight.ToOutput());

        jtkEq:
          Result := FPool.NewBool(LLeft.Equals(LRight));

        jtkNe:
          Result := FPool.NewBool(not LLeft.Equals(LRight));

        jtkLt:
        begin
          if LLeft.IsString() and LRight.IsString() then
            Result := FPool.NewBool(LLeft.AsString() < LRight.AsString())
          else
            Result := FPool.NewBool(LLeft.AsFloat() < LRight.AsFloat());
        end;

        jtkGt:
        begin
          if LLeft.IsString() and LRight.IsString() then
            Result := FPool.NewBool(LLeft.AsString() > LRight.AsString())
          else
            Result := FPool.NewBool(LLeft.AsFloat() > LRight.AsFloat());
        end;

        jtkLe:
          Result := FPool.NewBool(LLeft.AsFloat() <= LRight.AsFloat());

        jtkGe:
          Result := FPool.NewBool(LLeft.AsFloat() >= LRight.AsFloat());

        jtkIn:
        begin
          LBoolResult := False;
          if LRight.IsArray() then
          begin
            for I := 0 to LRight.ArrayCount() - 1 do
            begin
              if LLeft.Equals(LRight.ArrayGet(I)) then
              begin
                LBoolResult := True;
                Break;
              end;
            end;
          end
          else if LRight.IsDict() then
            LBoolResult := LRight.DictHas(LLeft.AsString())
          else if LRight.IsString() then
            LBoolResult := Pos(LLeft.AsString(), LRight.AsString()) > 0;
          Result := FPool.NewBool(LBoolResult);
        end;
      else
        Result := FPool.NewNone();
      end;
    end;

    jekUnary:
    begin
      LVal := Eval(AExpr.UnaryOperand, ACtx);
      case AExpr.UnaryOp of
        jtkNot:   Result := FPool.NewBool(not LVal.IsTruthy());
        jtkMinus:
        begin
          if LVal.IsInt() then
            Result := FPool.NewInt(-LVal.AsInt())
          else
            Result := FPool.NewFloat(-LVal.AsFloat());
        end;
      else
        Result := LVal;
      end;
    end;

    jekMember:
    begin
      LObj := Eval(AExpr.MemberObj, ACtx);
      LKey := AExpr.MemberName;

      if LObj.IsDict() then
      begin
        LVal := LObj.DictGet(LKey);
        if LVal <> nil then
          Result := LVal
        else
          // Missing key via attribute access is Undefined
          Result := FPool.NewUndefined();
      end
      else if LObj.IsArray() and (LKey = 'length') then
        Result := FPool.NewInt(LObj.ArrayCount())
      else if LObj.IsString() and (LKey = 'length') then
        Result := FPool.NewInt(Length(LObj.AsString()))
      else
        Result := FPool.NewUndefined();
    end;

    jekIndex:
    begin
      LObj := Eval(AExpr.MemberObj, ACtx);
      LVal := Eval(AExpr.IndexExpr, ACtx);

      if LObj.IsArray() then
      begin
        // Out-of-range indexing yields Undefined (Jinja getitem semantics)
        LStart := LVal.AsInt();
        if LStart < 0 then
          LStart := LObj.ArrayCount() + LStart;
        if (LStart < 0) or (LStart >= LObj.ArrayCount()) then
          Result := FPool.NewUndefined()
        else
          Result := LObj.ArrayGet(LStart);
      end
      else if LObj.IsDict() then
      begin
        LResult := LObj.DictGet(LVal.AsString());
        if LResult <> nil then
          Result := LResult
        else
          // Missing key via subscript is Undefined
          Result := FPool.NewUndefined();
      end
      else if LObj.IsString() then
      begin
        LStart := LVal.AsInt();
        if LStart < 0 then
          LStart := Length(LObj.AsString()) + LStart;
        if (LStart < 0) or (LStart >= Length(LObj.AsString())) then
          Result := FPool.NewUndefined()
        else
          Result := FPool.NewString(LObj.AsString()[LStart + 1]);
      end
      else
        Result := FPool.NewUndefined();
    end;

    jekSlice:
    begin
      LObj := Eval(AExpr.SliceObj, ACtx);

      if AExpr.SliceStep <> nil then
        LStep := Eval(AExpr.SliceStep, ACtx).AsInt()
      else
        LStep := 1;

      if LObj.IsArray() then
      begin
        LLen := LObj.ArrayCount();

        // For negative step with default start/stop, reverse the defaults
        if LStep < 0 then
        begin
          if AExpr.SliceStart <> nil then
            LStart := Eval(AExpr.SliceStart, ACtx).AsInt()
          else
            LStart := LLen - 1;
          if AExpr.SliceStop <> nil then
            LStop := Eval(AExpr.SliceStop, ACtx).AsInt()
          else
            LStop := -(LLen + 1); // sentinel: go all the way to index 0
        end
        else
        begin
          if AExpr.SliceStart <> nil then
            LStart := Eval(AExpr.SliceStart, ACtx).AsInt()
          else
            LStart := 0;
          if AExpr.SliceStop <> nil then
            LStop := Eval(AExpr.SliceStop, ACtx).AsInt()
          else
            LStop := LLen;
        end;

        // Negative indexing
        if LStart < 0 then
          LStart := Max(0, LLen + LStart);
        if LStop < -(LLen) then
          LStop := -1  // sentinel for "include index 0"
        else if LStop < 0 then
          LStop := Max(0, LLen + LStop);

        LSliceArr := FPool.NewArray();
        if LStep > 0 then
        begin
          LStop := Min(LStop, LLen);
          I := LStart;
          while I < LStop do
          begin
            LSliceArr.ArrayAdd(LObj.ArrayGet(I));
            Inc(I, LStep);
          end;
        end
        else if LStep < 0 then
        begin
          LStart := Min(LStart, LLen - 1);
          I := LStart;
          while I >= Max(LStop + 1, 0) do
          begin
            LSliceArr.ArrayAdd(LObj.ArrayGet(I));
            Inc(I, LStep);
          end;
        end;
        Result := LSliceArr;
      end
      else if LObj.IsString() then
      begin
        if AExpr.SliceStart <> nil then
          LStart := Eval(AExpr.SliceStart, ACtx).AsInt()
        else
          LStart := 0;
        if AExpr.SliceStop <> nil then
          LStop := Eval(AExpr.SliceStop, ACtx).AsInt()
        else
          LStop := Length(LObj.AsString());
        if LStart < 0 then
          LStart := Max(0, Length(LObj.AsString()) + LStart);
        Result := FPool.NewString(
          Copy(LObj.AsString(), LStart + 1, LStop - LStart));
      end
      else
        Result := FPool.NewNone();
    end;

    jekCall:
    begin
      LTarget := Eval(AExpr.CallTarget, ACtx);
      SetLength(LArgs, AExpr.CallArgs.Count);
      for I := 0 to AExpr.CallArgs.Count - 1 do
        LArgs[I] := Eval(AExpr.CallArgs[I], ACtx);

      // Handle kwargs by mapping them onto the callable's parameter list
      if AExpr.CallKwargs.Count > 0 then
      begin
        LKwargs := TDictionary<string, TJinjaValue>.Create();
        try
          for LStrPair in AExpr.CallKwargs do
            LKwargs.Add(LStrPair.Key, Eval(LStrPair.Value, ACtx));

          // For namespace() with kwargs: namespace(found=false)
          if LTarget.FCallableName = 'namespace' then
          begin
            LResult := LTarget.FCallable(LArgs, FPool.GetPool());
            for LStrPair in AExpr.CallKwargs do
              LResult.DictSet(LStrPair.Key, LKwargs[LStrPair.Key]);
            Result := LResult;
          end
          else
          begin
            // Merge kwargs into positional args for macro calls
            if LTarget.FCallableParams <> nil then
            begin
              LLen := LTarget.FCallableParams.Count;
              if LLen > Length(LArgs) then
                SetLength(LArgs, LLen);
              for LStrPair in AExpr.CallKwargs do
              begin
                I := LTarget.FCallableParams.IndexOf(LStrPair.Key);
                if I >= 0 then
                  LArgs[I] := LKwargs[LStrPair.Key];
              end;
            end;
            Result := CallValue(LTarget, LArgs, ACtx);
          end;
        finally
          LKwargs.Free();
        end;
      end
      else
        Result := CallValue(LTarget, LArgs, ACtx);
    end;

    jekMethodCall:
    begin
      LObj := Eval(AExpr.MemberObj, ACtx);
      LKey := AExpr.MemberName;
      SetLength(LArgs, AExpr.CallArgs.Count);
      for I := 0 to AExpr.CallArgs.Count - 1 do
        LArgs[I] := Eval(AExpr.CallArgs[I], ACtx);

      // String methods
      if LObj.IsString() then
      begin
        if LKey = 'strip' then
          Result := FPool.NewString(Trim(LObj.AsString()))
        else if LKey = 'lstrip' then
          Result := FPool.NewString(TrimLeft(LObj.AsString()))
        else if LKey = 'rstrip' then
          Result := FPool.NewString(TrimRight(LObj.AsString()))
        else if LKey = 'startswith' then
          Result := FPool.NewBool(
            StartsStr(LArgs[0].AsString(), LObj.AsString()))
        else if LKey = 'endswith' then
          Result := FPool.NewBool(
            EndsStr(LArgs[0].AsString(), LObj.AsString()))
        else if LKey = 'split' then
        begin
          LResult := FPool.NewArray();
          if Length(LArgs) > 0 then
          begin
            // Split by delimiter (keeps empty parts, Python semantics)
            LParts := LObj.AsString().Split([LArgs[0].AsString()]);
            for I := 0 to Length(LParts) - 1 do
              LResult.ArrayAdd(FPool.NewString(LParts[I]));
          end
          else
          begin
            // Split by whitespace (drops empty parts, Python semantics)
            LParts := LObj.AsString().Split([' ', #9, #10, #13]);
            for I := 0 to Length(LParts) - 1 do
            begin
              if LParts[I] <> '' then
                LResult.ArrayAdd(FPool.NewString(LParts[I]));
            end;
          end;
          Result := LResult;
        end
        else if LKey = 'upper' then
          Result := FPool.NewString(UpperCase(LObj.AsString()))
        else if LKey = 'lower' then
          Result := FPool.NewString(LowerCase(LObj.AsString()))
        else if LKey = 'title' then
          Result := FPool.NewString(
            AnsiUpperCase(Copy(LObj.AsString(), 1, 1)) +
            Copy(LObj.AsString(), 2, MaxInt))
        else if LKey = 'replace' then
        begin
          if Length(LArgs) >= 2 then
            Result := FPool.NewString(
              StringReplace(LObj.AsString(), LArgs[0].AsString(),
                LArgs[1].AsString(), [rfReplaceAll]))
          else
            Result := LObj;
        end
        else
          Result := FPool.NewNone();
      end
      // Dict methods
      else if LObj.IsDict() then
      begin
        if LKey = 'items' then
        begin
          LResult := FPool.NewArray();
          if LObj.FDictKeys <> nil then
          begin
            for I := 0 to LObj.FDictKeys.Count - 1 do
            begin
              LVal := FPool.NewArray();
              LVal.ArrayAdd(FPool.NewString(LObj.FDictKeys[I]));
              LVal.ArrayAdd(LObj.FDictMap[LObj.FDictKeys[I]]);
              LResult.ArrayAdd(LVal);
            end;
          end;
          Result := LResult;
        end
        else if LKey = 'get' then
        begin
          // Python dict.get(): a missing key returns None (not Undefined),
          // or the supplied default when given
          if Length(LArgs) >= 1 then
          begin
            LVal := LObj.DictGet(LArgs[0].AsString());
            if LVal <> nil then
              Result := LVal
            else if Length(LArgs) >= 2 then
              Result := LArgs[1]
            else
              Result := FPool.NewNone();
          end
          else
            Result := FPool.NewNone();
        end
        else if LKey = 'keys' then
        begin
          LResult := FPool.NewArray();
          if LObj.FDictKeys <> nil then
          begin
            for I := 0 to LObj.FDictKeys.Count - 1 do
              LResult.ArrayAdd(FPool.NewString(LObj.FDictKeys[I]));
          end;
          Result := LResult;
        end
        else if LKey = 'values' then
        begin
          LResult := FPool.NewArray();
          if LObj.FDictKeys <> nil then
          begin
            for I := 0 to LObj.FDictKeys.Count - 1 do
              LResult.ArrayAdd(LObj.FDictMap[LObj.FDictKeys[I]]);
          end;
          Result := LResult;
        end
        else if LKey = 'update' then
        begin
          // dict.update(other_dict) - mutates in place
          if (Length(LArgs) >= 1) and LArgs[0].IsDict() and
             (LArgs[0].FDictKeys <> nil) then
          begin
            for I := 0 to LArgs[0].FDictKeys.Count - 1 do
              LObj.DictSet(LArgs[0].FDictKeys[I],
                LArgs[0].FDictMap[LArgs[0].FDictKeys[I]]);
          end;
          Result := FPool.NewNone();
        end
        else
          Result := FPool.NewNone();
      end
      // Array methods
      else if LObj.IsArray() then
      begin
        if LKey = 'append' then
        begin
          if Length(LArgs) >= 1 then
            LObj.ArrayAdd(LArgs[0]);
          Result := FPool.NewNone();
        end
        else
          Result := FPool.NewNone();
      end
      else
        Result := FPool.NewNone();
    end;

    jekFilter:
    begin
      LVal := Eval(AExpr.FilterExpr, ACtx);
      SetLength(LFilterArgs, AExpr.FilterArgs.Count);
      for I := 0 to AExpr.FilterArgs.Count - 1 do
        LFilterArgs[I] := Eval(AExpr.FilterArgs[I], ACtx);

      LKwargs := nil;
      if AExpr.FilterKwargs.Count > 0 then
      begin
        LKwargs := TDictionary<string, TJinjaValue>.Create();
        for LStrPair in AExpr.FilterKwargs do
          LKwargs.Add(LStrPair.Key, Eval(LStrPair.Value, ACtx));
      end;
      try
        Result := ApplyFilter(AExpr.FilterName, LVal, LFilterArgs, LKwargs, ACtx);
      finally
        LKwargs.Free();
      end;
    end;

    jekTest:
    begin
      // Special handling for defined/undefined -- check existence, not value
      if (AExpr.TestName = 'defined') or (AExpr.TestName = 'undefined') then
      begin
        if AExpr.TestExpr.ExprKind = jekVar then
          LBoolResult := ACtx.Has(AExpr.TestExpr.VarName)
        else if AExpr.TestExpr.ExprKind = jekMember then
        begin
          LVal := Eval(AExpr.TestExpr.MemberObj, ACtx);
          if LVal.IsDict() then
            LBoolResult := LVal.DictHas(AExpr.TestExpr.MemberName)
          else
            LBoolResult := False;
        end
        else
        begin
          LVal := Eval(AExpr.TestExpr, ACtx);
          LBoolResult := not LVal.IsUndefined();
        end;

        if AExpr.TestName = 'undefined' then
          LBoolResult := not LBoolResult;
      end
      else
      begin
        LVal := Eval(AExpr.TestExpr, ACtx);
        SetLength(LTestArgs, AExpr.TestArgs.Count);
        for I := 0 to AExpr.TestArgs.Count - 1 do
          LTestArgs[I] := Eval(AExpr.TestArgs[I], ACtx);
        LBoolResult := ApplyTest(AExpr.TestName, LVal, LTestArgs, ACtx);
      end;
      if AExpr.TestNegated then
        LBoolResult := not LBoolResult;
      Result := FPool.NewBool(LBoolResult);
    end;

    jekArray:
    begin
      LResult := FPool.NewArray();
      for I := 0 to AExpr.ArrayItems.Count - 1 do
        LResult.ArrayAdd(Eval(AExpr.ArrayItems[I], ACtx));
      Result := LResult;
    end;

    jekDict:
    begin
      LResult := FPool.NewDict();
      for I := 0 to AExpr.DictPairs.Count - 1 do
      begin
        LPair := AExpr.DictPairs[I];
        LResult.DictSet(
          Eval(LPair.Key, ACtx).AsString(),
          Eval(LPair.Value, ACtx));
      end;
      Result := LResult;
    end;

    jekConditional:
    begin
      LVal := Eval(AExpr.CondTest, ACtx);
      if LVal.IsTruthy() then
        Result := Eval(AExpr.CondTrue, ACtx)
      else if AExpr.CondFalse <> nil then
        Result := Eval(AExpr.CondFalse, ACtx)
      else
        // Else-less inline-if yields Undefined (renders as ''), matching
        // Jinja: {{ ',' if not loop.last }}
        Result := FPool.NewUndefined();
    end;

  else
    Result := FPool.NewNone();
  end;
end;

function TJinjaRenderer.ApplyFilter(const ANodeName: string;
  const AInput: TJinjaValue; const AArgs: TArray<TJinjaValue>;
  const AKwargs: TDictionary<string, TJinjaValue>;
  const ACtx: TJinjaContext): TJinjaValue;
var
  I: Integer;
  LResult: TJinjaValue;
  LBuilder: TStringBuilder;
  LSep: string;
  LAttr, LTestVal: string;
  LVal, LDefault: TJinjaValue;
  LInclude: Boolean;
  LItems: TStringList;
  LEmptyArgs: TArray<TJinjaValue>;
begin
  if ANodeName = 'tojson' then
    Result := FPool.NewString(AInput.ToJSON())
  else if ANodeName = 'trim' then
    Result := FPool.NewString(Trim(AInput.AsString()))
  else if (ANodeName = 'length') or (ANodeName = 'count') then
  begin
    if AInput.IsArray() then
      Result := FPool.NewInt(AInput.ArrayCount())
    else if AInput.IsDict() then
      Result := FPool.NewInt(AInput.DictCount())
    else if AInput.IsString() then
      Result := FPool.NewInt(Length(AInput.AsString()))
    else
      Result := FPool.NewInt(0);
  end
  else if ANodeName = 'join' then
  begin
    if Length(AArgs) > 0 then
      LSep := AArgs[0].AsString()
    else
      LSep := '';
    LBuilder := TStringBuilder.Create();
    try
      if AInput.IsArray() then
      begin
        for I := 0 to AInput.ArrayCount() - 1 do
        begin
          if I > 0 then
            LBuilder.Append(LSep);
          LBuilder.Append(AInput.ArrayGet(I).ToOutput());
        end;
      end;
      Result := FPool.NewString(LBuilder.ToString());
    finally
      LBuilder.Free();
    end;
  end
  else if ANodeName = 'items' then
  begin
    // dict | items -> array of [key, value] pairs
    LResult := FPool.NewArray();
    if AInput.IsDict() and (AInput.FDictKeys <> nil) then
    begin
      for I := 0 to AInput.FDictKeys.Count - 1 do
      begin
        LVal := FPool.NewArray();
        LVal.ArrayAdd(FPool.NewString(AInput.FDictKeys[I]));
        LVal.ArrayAdd(AInput.FDictMap[AInput.FDictKeys[I]]);
        LResult.ArrayAdd(LVal);
      end;
    end;
    Result := LResult;
  end
  else if ANodeName = 'dictsort' then
  begin
    // Sort keys with deterministic ordinal case-insensitive compare,
    // matching Python dictsort (case_sensitive=False) for ASCII keys
    LResult := FPool.NewArray();
    if AInput.IsDict() and (AInput.FDictKeys <> nil) then
    begin
      LItems := TStringList.Create();
      try
        LItems.Assign(AInput.FDictKeys);
        LItems.CustomSort(JinjaDictSortCompare);
        for I := 0 to LItems.Count - 1 do
        begin
          LVal := FPool.NewArray();
          LVal.ArrayAdd(FPool.NewString(LItems[I]));
          LVal.ArrayAdd(AInput.FDictMap[LItems[I]]);
          LResult.ArrayAdd(LVal);
        end;
      finally
        LItems.Free();
      end;
    end;
    Result := LResult;
  end
  else if ANodeName = 'default' then
  begin
    // Strict Jinja semantics: default replaces UNDEFINED only, unless the
    // boolean flag (2nd positional arg) is truthy, in which case any falsy
    // value is replaced. An explicit none is defined and passes through.
    LInclude := AInput.IsUndefined();
    if (not LInclude) and (Length(AArgs) >= 2) and AArgs[1].IsTruthy() then
      LInclude := not AInput.IsTruthy();
    if LInclude then
    begin
      if Length(AArgs) >= 1 then
        Result := AArgs[0]
      else
        Result := FPool.NewString('');
    end
    else
      Result := AInput;
  end
  else if ANodeName = 'first' then
  begin
    if AInput.IsArray() and (AInput.ArrayCount() > 0) then
      Result := AInput.ArrayGet(0)
    else
      Result := FPool.NewNone();
  end
  else if ANodeName = 'last' then
  begin
    if AInput.IsArray() and (AInput.ArrayCount() > 0) then
      Result := AInput.ArrayGet(AInput.ArrayCount() - 1)
    else
      Result := FPool.NewNone();
  end
  else if (ANodeName = 'e') or (ANodeName = 'escape') then
  begin
    // HTML escape -- in LLM context this is a no-op passthrough
    Result := AInput;
  end
  else if ANodeName = 'raise_exception' then
  begin
    raise Exception.Create(AInput.AsString());
  end
  else if ANodeName = 'selectattr' then
  begin
    // selectattr(attr, test, value) -- filter array of dicts
    LResult := FPool.NewArray();
    if (AInput.IsArray()) and (Length(AArgs) >= 1) then
    begin
      LAttr := AArgs[0].AsString();
      for I := 0 to AInput.ArrayCount() - 1 do
      begin
        LVal := AInput.ArrayGet(I);
        if LVal.IsDict() then
        begin
          LDefault := LVal.DictGet(LAttr);
          LInclude := False;
          if Length(AArgs) >= 3 then
          begin
            // selectattr('attr', 'equalto', value)
            if LDefault <> nil then
              LInclude := LDefault.Equals(AArgs[2]);
          end
          else if Length(AArgs) >= 2 then
          begin
            LTestVal := AArgs[1].AsString();
            if LTestVal = 'defined' then
              LInclude := LDefault <> nil
            else if LTestVal = 'undefined' then
              LInclude := LDefault = nil
            else if (LTestVal = 'truthy') or (LTestVal = 'true') then
              LInclude := (LDefault <> nil) and LDefault.IsTruthy()
            else if LDefault <> nil then
              LInclude := LDefault.IsTruthy();
          end
          else
          begin
            if LDefault <> nil then
              LInclude := LDefault.IsTruthy();
          end;

          if LInclude then
            LResult.ArrayAdd(LVal);
        end;
      end;
    end;
    Result := LResult;
  end
  else if ANodeName = 'rejectattr' then
  begin
    LResult := FPool.NewArray();
    if (AInput.IsArray()) and (Length(AArgs) >= 1) then
    begin
      LAttr := AArgs[0].AsString();
      for I := 0 to AInput.ArrayCount() - 1 do
      begin
        LVal := AInput.ArrayGet(I);
        if LVal.IsDict() then
        begin
          LDefault := LVal.DictGet(LAttr);
          LInclude := True;
          if Length(AArgs) >= 3 then
          begin
            if LDefault <> nil then
              LInclude := not LDefault.Equals(AArgs[2]);
          end
          else if Length(AArgs) >= 2 then
          begin
            LTestVal := AArgs[1].AsString();
            if LTestVal = 'defined' then
              LInclude := LDefault = nil
            else if (LDefault <> nil) then
              LInclude := not LDefault.IsTruthy();
          end
          else
          begin
            if LDefault <> nil then
              LInclude := not LDefault.IsTruthy();
          end;

          if LInclude then
            LResult.ArrayAdd(LVal);
        end;
      end;
    end;
    Result := LResult;
  end
  else if ANodeName = 'select' then
  begin
    LResult := FPool.NewArray();
    if AInput.IsArray() then
    begin
      for I := 0 to AInput.ArrayCount() - 1 do
      begin
        if AInput.ArrayGet(I).IsTruthy() then
          LResult.ArrayAdd(AInput.ArrayGet(I));
      end;
    end;
    Result := LResult;
  end
  else if ANodeName = 'reject' then
  begin
    LResult := FPool.NewArray();
    if AInput.IsArray() then
    begin
      for I := 0 to AInput.ArrayCount() - 1 do
      begin
        if not AInput.ArrayGet(I).IsTruthy() then
          LResult.ArrayAdd(AInput.ArrayGet(I));
      end;
    end;
    Result := LResult;
  end
  else if ANodeName = 'map' then
  begin
    // Two forms:
    //   map(attribute='name') -- extract an attribute from each item
    //   map('filtername')     -- apply a named filter to each item,
    //                            e.g. sizes | map('upper') | list
    LResult := FPool.NewArray();
    if AInput.IsArray() then
    begin
      if (AKwargs <> nil) and AKwargs.ContainsKey('attribute') then
      begin
        LAttr := AKwargs['attribute'].AsString();
        for I := 0 to AInput.ArrayCount() - 1 do
        begin
          LVal := AInput.ArrayGet(I);
          if LVal.IsDict() then
          begin
            LDefault := LVal.DictGet(LAttr);
            if LDefault <> nil then
              LResult.ArrayAdd(LDefault)
            else
              LResult.ArrayAdd(FPool.NewNone());
          end
          else
            LResult.ArrayAdd(FPool.NewNone());
        end;
      end
      else if (Length(AArgs) >= 1) and AArgs[0].IsString() then
      begin
        LEmptyArgs := nil;
        for I := 0 to AInput.ArrayCount() - 1 do
          LResult.ArrayAdd(ApplyFilter(AArgs[0].AsString(),
            AInput.ArrayGet(I), LEmptyArgs, nil, ACtx));
      end;
    end;
    Result := LResult;
  end
  else if ANodeName = 'upper' then
    Result := FPool.NewString(UpperCase(AInput.AsString()))
  else if ANodeName = 'lower' then
    Result := FPool.NewString(LowerCase(AInput.AsString()))
  else if ANodeName = 'replace' then
  begin
    if Length(AArgs) >= 2 then
      Result := FPool.NewString(
        StringReplace(AInput.AsString(), AArgs[0].AsString(),
          AArgs[1].AsString(), [rfReplaceAll]))
    else
      Result := AInput;
  end
  else if ANodeName = 'int' then
  begin
    if AInput.IsInt() then
      Result := AInput
    else if AInput.IsFloat() then
      Result := FPool.NewInt(Trunc(AInput.AsFloat()))
    else if AInput.IsString() then
    begin
      try
        Result := FPool.NewInt(StrToInt64(AInput.AsString()));
      except
        if Length(AArgs) > 0 then
          Result := AArgs[0]
        else
          Result := FPool.NewInt(0);
      end;
    end
    else
      Result := FPool.NewInt(0);
  end
  else if ANodeName = 'float' then
  begin
    if AInput.IsFloat() or AInput.IsInt() then
      Result := FPool.NewFloat(AInput.AsFloat())
    else if AInput.IsString() then
    begin
      try
        Result := FPool.NewFloat(
          StrToFloat(AInput.AsString(), TFormatSettings.Invariant));
      except
        Result := FPool.NewFloat(0.0);
      end;
    end
    else
      Result := FPool.NewFloat(0.0);
  end
  else if ANodeName = 'string' then
    Result := FPool.NewString(AInput.ToOutput())
  else if ANodeName = 'list' then
  begin
    LResult := FPool.NewArray();
    if AInput.IsArray() then
    begin
      for I := 0 to AInput.ArrayCount() - 1 do
        LResult.ArrayAdd(AInput.ArrayGet(I));
    end
    else if AInput.IsString() then
    begin
      for I := 1 to Length(AInput.AsString()) do
        LResult.ArrayAdd(FPool.NewString(AInput.AsString()[I]));
    end;
    Result := LResult;
  end
  else if ANodeName = 'min' then
  begin
    if AInput.IsArray() and (AInput.ArrayCount() > 0) then
    begin
      LResult := AInput.ArrayGet(0);
      for I := 1 to AInput.ArrayCount() - 1 do
      begin
        LVal := AInput.ArrayGet(I);
        if LVal.AsFloat() < LResult.AsFloat() then
          LResult := LVal;
      end;
      Result := LResult;
    end
    else
      Result := FPool.NewNone();
  end
  else if ANodeName = 'max' then
  begin
    if AInput.IsArray() and (AInput.ArrayCount() > 0) then
    begin
      LResult := AInput.ArrayGet(0);
      for I := 1 to AInput.ArrayCount() - 1 do
      begin
        LVal := AInput.ArrayGet(I);
        if LVal.AsFloat() > LResult.AsFloat() then
          LResult := LVal;
      end;
      Result := LResult;
    end
    else
      Result := FPool.NewNone();
  end
  else
    // Unknown filter -- fail loud instead of silently passing through
    raise Exception.CreateFmt('Unknown filter: %s', [ANodeName]);
end;

function TJinjaRenderer.ApplyTest(const ANodeName: string;
  const AInput: TJinjaValue; const AArgs: TArray<TJinjaValue>;
  const ACtx: TJinjaContext): Boolean;
begin
  // Test names arrive lowercased from the parser
  if ANodeName = 'defined' then
    Result := not AInput.IsUndefined()
  else if ANodeName = 'undefined' then
    Result := AInput.IsUndefined()
  else if ANodeName = 'none' then
    Result := AInput.IsNone()
  else if ANodeName = 'true' then
    Result := AInput.IsBool() and AInput.AsBool()
  else if ANodeName = 'false' then
    Result := AInput.IsBool() and not AInput.AsBool()
  else if ANodeName = 'boolean' then
    Result := AInput.IsBool()
  else if ANodeName = 'string' then
    Result := AInput.IsString()
  else if ANodeName = 'number' then
    Result := AInput.IsNumber()
  else if ANodeName = 'integer' then
    Result := AInput.IsInt()
  else if ANodeName = 'float' then
    Result := AInput.IsFloat()
  else if ANodeName = 'mapping' then
    Result := AInput.IsMapping()
  else if ANodeName = 'iterable' then
    Result := AInput.IsIterable()
  else if ANodeName = 'sequence' then
    Result := AInput.IsArray()
  else if ANodeName = 'callable' then
    Result := AInput.IsCallable()
  else if (ANodeName = 'equalto') or (ANodeName = 'eq') or
          (ANodeName = 'sameas') then
  begin
    if Length(AArgs) > 0 then
      Result := AInput.Equals(AArgs[0])
    else
      Result := False;
  end
  else if ANodeName = 'even' then
    Result := AInput.IsInt() and (AInput.AsInt() mod 2 = 0)
  else if ANodeName = 'odd' then
    Result := AInput.IsInt() and (AInput.AsInt() mod 2 <> 0)
  else
    // Unknown test -- fail loud instead of silently returning False
    raise Exception.CreateFmt('Unknown test: %s', [ANodeName]);
end;

function TJinjaRenderer.CallValue(const ACallable: TJinjaValue;
  const AArgs: TArray<TJinjaValue>;
  const ACtx: TJinjaContext): TJinjaValue;
begin
  if ACallable.IsCallable() then
    Result := ACallable.FCallable(AArgs, FPool.GetPool())
  else
    raise Exception.CreateFmt('Value is not callable: %s',
      [ACallable.ToOutput()]);
end;

{ TJinja }

constructor TJinja.Create();
begin
  inherited Create();
  FParser := TJinjaParser.Create();
  FParser.SetErrors(FErrors);
  FRoot := nil;
end;

destructor TJinja.Destroy();
begin
  FRoot.Free();
  FParser.Free();
  inherited Destroy();
end;

function TJinja.Parse(const ATemplate: string): Boolean;
begin
  FreeAndNil(FRoot);
  try
    FRoot := FParser.Parse(ATemplate);
    Result := FRoot <> nil;
  except
    on E: Exception do
    begin
      FErrors.Add(esError, 'JINJA', E.Message);
      FRoot := nil;
      Result := False;
    end;
  end;
end;

function TJinja.Render(const AContext: TJinjaContext): string;
var
  LRenderer: TJinjaRenderer;
begin
  LRenderer := TJinjaRenderer.Create();
  try
    LRenderer.SetErrors(FErrors);
    Result := LRenderer.Render(FRoot, AContext);
  finally
    LRenderer.Free();
  end;
end;

end.
