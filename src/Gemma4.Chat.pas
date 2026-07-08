{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information

 -------------------------------------------------------------------------------

  Gemma4.Chat - Abstract interactive chat loop + console frontend

  Front-end shell around TSession for interactive use:
    - TChat: abstract loop with virtual hooks (input, output, streaming)
    - Slash-command dispatch (/help /quit /clear /forget /system /addfact
      /addfile /stats /turns /tokens /compact /save /load /summary
      /dbsave /dbload /dblist /dbreset /state /restore)
    - Rebuild policy and memory-database management commands
    - TConsoleChat: concrete console implementation over StdApp.Console
    - Owns TMemory and TSession, borrows a shared TInference

  Dependencies: WinApi.Windows, System.SysUtils, System.IOUtils,
    System.Generics.Collections, StdApp.Base, StdApp.Console,
    StdApp.Console.Adapter, StdApp.Input, Gemma4.Types, Gemma4.Inference,
    Gemma4.Embeddings, Gemma4.Memory, Gemma4.Session, Gemma4.Tools
===============================================================================}

unit Gemma4.Chat;

{$I StdApp.Defines.inc}

interface

uses
  WinApi.Windows,
  System.SysUtils,
  System.IOUtils,
  System.Generics.Collections,
  StdApp.Base,
  StdApp.Console,
  StdApp.Console.Adapter,
  StdApp.Input,
  Gemma4.Types,
  Gemma4.Inference,
  Gemma4.Embeddings,
  Gemma4.Memory,
  Gemma4.Session,
  Gemma4.Tools;

const
  { Error codes }
  ERR_CHAT_NO_MODEL   = 'CHT001';
  ERR_CHAT_MODEL_LOAD = 'CHT002';
  ERR_CHAT_EMBEDDER   = 'CHT003';
  ERR_CHAT_MEMORY     = 'CHT004';
  ERR_CHAT_CONFIG     = 'CHT005';

type
  { TChat }
  // Layer 4: abstract chat loop over TSession. Owns the full stack --
  // TInference, optional TEmbeddings + TMemory, TSession -- built inside
  // Run() from path properties and torn down on exit. Derived classes
  // implement the four abstract I/O methods; everything else is policy
  // that lives here, including the proactive rebuild: after each turn,
  // when the token position crosses RebuildThreshold of the context,
  // CompactHistory evicts old turns (with summary-on-evict when memory
  // is attached). One full re-decode then buys many turns of clean KV
  // prefix reuse. All owned objects share this object's FErrors.
  TChat = class(TBaseObject)
  private
    FInference: TInference;
    FEmbeddings: TEmbeddings;
    FMemory: TMemory;
    FSession: TSession;
    FModelPath: string;
    FUseGPU: Boolean;
    FMemoryDbPath: string;
    FEnableThinking: Boolean;
    FShowThinking: Boolean;
    FThinkingOpenText: string;
    FThinkingCloseText: string;
    FThinkingPlaceholderText: string;
    FMaxTokens: Integer;
    FSystemPrompt: string;
    FRunning: Boolean;
    FRebuildThreshold: Single;
    FKeepRecentTurns: Integer;
    FDocChunkWords: Integer;
    FDocOverlapWords: Integer;
    FStripThinking: Boolean;
    FRecallTopK: Integer;
    FPinnedTopK: Integer;
    FRecallBudgetTokens: Integer;
    FMinRecallScore: Single;
    FMinTurnTokens: Integer;
    FToolRegistry: TToolRegistry;   // not owned -- app-provided
    FMaxToolRounds: Integer;
    FToolResultMaxChars: Integer;
    // Build the full stack from the configured paths. Reports through
    // FErrors and the Do* hooks; returns False on any failure. There
    // are no fallbacks -- a configured component that fails aborts Run.
    function DoBuildStack(): Boolean;
    // Tear down in reverse dependency order. Safe on a partial stack.
    procedure DoTeardownStack();
    // Forward everything accumulated in the shared FErrors to the
    // DoError/DoInfo hooks, then clear the collection.
    procedure DoDrainErrors();
    // The rebuild policy: compare the token position after a turn
    // against the threshold; compact when crossed.
    procedure DoCheckRebuild();
    // Parse and dispatch a /command.
    procedure ProcessCommand(const AInput: string);
    // One conversational turn via Session.Ask.
    procedure ProcessChat(const AInput: string);
    procedure DoCmdClear();
    procedure DoCmdForget();
    procedure DoCmdSystem(const AArgs: string);
    procedure DoCmdAddFact(const AArgs: string);
    procedure DoCmdAddFile(const AArgs: string);
    procedure DoCmdStats();
    procedure DoCmdTurns();
    procedure DoCmdTokens(const AArgs: string);
    procedure DoCmdCompact();
    procedure DoCmdSave(const AArgs: string);
    procedure DoCmdLoad(const AArgs: string);
    procedure DoCmdState(const AArgs: string);
    procedure DoCmdRestore(const AArgs: string);
    procedure DoCmdSummary();
    procedure DoCmdDbSave(const AArgs: string);
    procedure DoCmdDbLoad(const AArgs: string);
    procedure DoCmdDbList(const AArgs: string);
    procedure DoCmdDbReset();
    function DoResolveDbPath(const APath: string): string;
    procedure PrintHelp();
  protected
    // --- Abstract I/O (must override) ---

    // Read one line of input from the user. Blocks until available.
    function  DoGetInput(): string; virtual; abstract;

    // Display a complete output line (command result, info, error).
    procedure DoOutput(const AText: string); virtual; abstract;

    // Stream a single token during generation.
    procedure DoToken(const AToken: string); virtual; abstract;

    // Check if the user has requested cancellation.
    function  DoCancel(): Boolean; virtual; abstract;

    // --- Virtual hooks (override to customize) ---

    // Error messages.
    procedure DoError(const AText: string); virtual;

    // Informational messages.
    procedure DoInfo(const AText: string); virtual;

    // Called once before the stack is built.
    procedure DoStartup(); virtual;

    // Called once after teardown.
    procedure DoShutdown(); virtual;

    // Called after each generation completes.
    procedure DoGenerationComplete(); virtual;

    // One tool invocation is about to run (name + raw JSON arguments).
    procedure DoToolCall(const AToolName: string;
      const AArguments: string); virtual;

    // Model-load progress (wired to TInference.SetLoadProgressCallback).
    procedure DoLoadProgress(const AState: TProgressState;
      const AStep: Integer; const ATotal: Integer); virtual;

    // Called after the stack is built, before the input loop -- the
    // place for derived classes to touch the live objects (thinking
    // marker text, renderer width, ...).
    procedure DoConfigureStack(); virtual;

    // Extension point for custom commands. Return True if handled.
    function  DoCommand(const ACmd: string;
      const AArgs: string): Boolean; virtual;

    // Owned-object access for derived class customization. Valid only
    // between DoConfigureStack and teardown.
    function GetInference(): TInference;
    function GetSession(): TSession;
    function GetMemory(): TMemory;
  public
    constructor Create(); override;
    destructor Destroy(); override;

    // Main loop -- template method. Builds the stack, runs the input
    // loop, tears down on /quit.
    procedure Run();

    // Configure before calling Run()
    property ModelPath: string read FModelPath write FModelPath;
    property UseGPU: Boolean read FUseGPU write FUseGPU;
    property MemoryDbPath: string read FMemoryDbPath write FMemoryDbPath;
    property EnableThinking: Boolean read FEnableThinking write FEnableThinking;
    // False hides the reasoning span behind a placeholder ('Thinking...')
    property ShowThinking: Boolean read FShowThinking write FShowThinking;
    // Custom text emitted in place of the native thinking markers
    property ThinkingOpenText: string read FThinkingOpenText write FThinkingOpenText;
    property ThinkingCloseText: string read FThinkingCloseText write FThinkingCloseText;
    property ThinkingPlaceholderText: string read FThinkingPlaceholderText write FThinkingPlaceholderText;
    property MaxTokens: Integer read FMaxTokens write FMaxTokens;
    property SystemPrompt: string read FSystemPrompt write FSystemPrompt;
    // Rebuild policy: compaction triggers when the token position
    // crosses RebuildThreshold of the context; KeepRecentTurns user
    // exchanges survive a compaction.
    property RebuildThreshold: Single read FRebuildThreshold write FRebuildThreshold;
    property KeepRecentTurns: Integer read FKeepRecentTurns write FKeepRecentTurns;
    // /addfile chunking: words per chunk and the overlap carried between
    // adjacent chunks (TMemory.AddDocument measures whitespace words)
    property DocChunkWords: Integer read FDocChunkWords write FDocChunkWords;
    property DocOverlapWords: Integer read FDocOverlapWords write FDocOverlapWords;
    // Session passthroughs -- TSession is owned and built inside Run,
    // so its knobs surface here
    property StripThinking: Boolean read FStripThinking write FStripThinking;
    property RecallTopK: Integer read FRecallTopK write FRecallTopK;
    property PinnedTopK: Integer read FPinnedTopK write FPinnedTopK;
    // Token budget for the recall injection; entries that do not fit
    // are skipped, never clipped
    property RecallBudgetTokens: Integer read FRecallBudgetTokens write FRecallBudgetTokens;
    // Minimum cosine similarity for a vector recall hit to be injected;
    // below the floor a nearest-neighbor hit is noise, not a memory
    property MinRecallScore: Single read FMinRecallScore write FMinRecallScore;
    // Memory passthrough -- turns below this token count skip the archive
    property MinTurnTokens: Integer read FMinTurnTokens write FMinTurnTokens;
    // App-provided tool registry; assign before Run. nil = no tools.
    property ToolRegistry: TToolRegistry read FToolRegistry write FToolRegistry;
    property MaxToolRounds: Integer read FMaxToolRounds write FMaxToolRounds;
    // Maximum characters per tool result before truncation in the
    // agentic loop. 0 disables truncation.
    property ToolResultMaxChars: Integer read FToolResultMaxChars write FToolResultMaxChars;

    // Memory database management -- available to any frontend.

    // Snapshot the current memory DB to APath. Creates directories
    // along the path. The active DB remains unchanged after the call.
    function MemoryDbSave(const APath: string): Boolean;

    // Close the current memory DB and open APath instead. The embedder
    // (if any) is re-attached to the new DB automatically. Conversation
    // history in TSession is NOT cleared -- only the recall backing
    // changes. Returns False if memory is not enabled or open fails.
    function MemoryDbLoad(const APath: string): Boolean;

    // Recursively scan APath for *.db files. Returns full paths.
    // Returns nil if APath does not exist or contains no databases.
    function MemoryDbList(const APath: string): TArray<string>;

    // Reload the original MemoryDbPath configured before Run().
    function MemoryDbReset(): Boolean;
  end;

  { TConsoleChat }
  // Console front-end for TChat: markdown-rendered streaming via
  // TConsoleAdapter, ESC cancellation, ReadLn input.
  TConsoleChat = class(TChat)
  private
    FAdapter: TConsoleAdapter;
  protected
    function  DoGetInput(): string; override;
    procedure DoOutput(const AText: string); override;
    procedure DoToken(const AToken: string); override;
    function  DoCancel(): Boolean; override;
    procedure DoError(const AText: string); override;
    procedure DoInfo(const AText: string); override;
    procedure DoStartup(); override;
    procedure DoShutdown(); override;
    procedure DoGenerationComplete(); override;
    procedure DoLoadProgress(const AState: TProgressState;
      const AStep: Integer; const ATotal: Integer); override;
    procedure DoConfigureStack(); override;
  public
    constructor Create(); override;
    destructor Destroy(); override;
    property Adapter: TConsoleAdapter read FAdapter;
  end;

implementation

uses
  StdApp.JSON,
  StdApp.Utils,
  StdApp.Resources,
  Gemma4.Common;

{ TChat }

constructor TChat.Create();
begin
  inherited Create();
  FInference := nil;
  FEmbeddings := nil;
  FMemory := nil;
  FSession := nil;
  FModelPath := '';
  FUseGPU := True;
  FMemoryDbPath := '';
  FEnableThinking := True;
  FShowThinking := True;
  FThinkingOpenText := '';
  FThinkingCloseText := '';
  FThinkingPlaceholderText := '';
  FMaxTokens := 1024;
  FSystemPrompt := '';
  FRunning := False;
  FRebuildThreshold := 0.75;
  FKeepRecentTurns := 3;
  FDocChunkWords := 200;
  FDocOverlapWords := 40;
  FStripThinking := True;
  FRecallTopK := 5;
  FPinnedTopK := 10;
  FRecallBudgetTokens := 512;
  FMinRecallScore := 0.45;
  FMinTurnTokens := 2;
  FToolRegistry := nil;
  FMaxToolRounds := 4;
  FToolResultMaxChars := 4000;
end;

destructor TChat.Destroy();
begin
  // Normal teardown happens at the end of Run; this covers a Run that
  // never completed or was never called
  DoTeardownStack();
  inherited Destroy();
end;

procedure TChat.DoError(const AText: string);
begin
  DoOutput(AText);
end;

procedure TChat.DoInfo(const AText: string);
begin
  DoOutput(AText);
end;

procedure TChat.DoStartup();
begin
  // no-op -- override in derived class
end;

procedure TChat.DoShutdown();
begin
  // no-op -- override in derived class
end;

procedure TChat.DoGenerationComplete();
begin
  // no-op -- override to flush a renderer, etc.
end;

procedure TChat.DoToolCall(const AToolName: string;
  const AArguments: string);
const
  CMaxDisplayLen = 200;
var
  LDisplay: string;
  LJson: TJSON;
begin
  LDisplay := AArguments;

  // For long argument blocks, try to show a "description" field instead
  if Length(AArguments) > CMaxDisplayLen then
  begin
    LJson := nil;
    try
      LJson := TJSON.FromString(AArguments);
      if (LJson <> nil) and LJson.Has('description') then
        LDisplay := LJson.Get('description').AsString();
    except
      // Parse failed -- fall through to truncation
    end;
    LJson.Free();

    if Length(LDisplay) > CMaxDisplayLen then
      LDisplay := Copy(LDisplay, 1, CMaxDisplayLen) + '...';
  end;

  DoInfo(Format(RSChatToolCall, [AToolName, LDisplay]));
end;

procedure TChat.DoLoadProgress(const AState: TProgressState;
  const AStep: Integer; const ATotal: Integer);
begin
  // no-op -- override to show a progress bar
end;

procedure TChat.DoConfigureStack();
begin
  // no-op -- override in derived class
end;

function TChat.DoCommand(const ACmd: string;
  const AArgs: string): Boolean;
begin
  Result := False;
end;

function TChat.GetInference(): TInference;
begin
  Result := FInference;
end;

function TChat.GetSession(): TSession;
begin
  Result := FSession;
end;

function TChat.GetMemory(): TMemory;
begin
  Result := FMemory;
end;

function TChat.DoBuildStack(): Boolean;
begin
  Result := False;

  if FModelPath = '' then
  begin
    FErrors.Add(esError, ERR_CHAT_NO_MODEL, RSChatModelPathEmpty);
    DoDrainErrors();
    Exit;
  end;

  // Config sanity: MaxTokens is the slice of the context reserved for
  // the reply -- reserving the whole window leaves zero room for the
  // conversation and makes the trimmer warn on every turn.
  if FMaxTokens > 0 then
  begin
    // Defer this check until after model load when ContextSize is known
  end;

  // Inference engine -- all owned objects report into this object's FErrors
  FInference := TInference.Create();
  FInference.SetErrors(FErrors);

  FInference.SetTokenCallback(
    procedure(const AState: TProgressState; const AToken: string;
      const AUserData: Pointer)
    begin
      if AState = psInProgress then
        Self.DoToken(AToken);
    end, nil);

  FInference.SetCancelCallback(
    function(const AUserData: Pointer): Boolean
    begin
      Result := Self.DoCancel();
    end, nil);

  FInference.SetLoadProgressCallback(
    procedure(const AState: TProgressState; const AStep: Integer;
      const ATotal: Integer; const AUserData: Pointer)
    begin
      Self.DoLoadProgress(AState, AStep, ATotal);
    end, nil);

  if not FInference.LoadModel(FModelPath, FUseGPU) then
  begin
    FErrors.Add(esError, ERR_CHAT_MODEL_LOAD,
      Format(RSChatModelLoadFailed, [FModelPath]));
    DoDrainErrors();
    Exit;
  end;

  // Post-load budget check
  if (FMaxTokens > 0) and (FMaxTokens >= FInference.ContextSize()) then
  begin
    FErrors.Add(esError, ERR_CHAT_CONFIG,
      Format(RSChatBadTokenBudget, [FMaxTokens, FInference.ContextSize()]));
    DoDrainErrors();
    Exit;
  end;

  FInference.EnableThinking := FEnableThinking;
  FInference.ShowThinking := FShowThinking;
  if FThinkingOpenText <> '' then
    FInference.ThinkingOpenText := FThinkingOpenText;
  if FThinkingCloseText <> '' then
    FInference.ThinkingCloseText := FThinkingCloseText;
  if FThinkingPlaceholderText <> '' then
    FInference.ThinkingPlaceholderText := FThinkingPlaceholderText;

  // Embeddings -- always loaded from the same VPK (embeddings/ folder)
  FEmbeddings := TEmbeddings.Create();
  FEmbeddings.SetErrors(FErrors);

  if not FEmbeddings.Open(FModelPath) then
  begin
    FErrors.Add(esError, ERR_CHAT_EMBEDDER,
      Format(RSChatEmbedderFailed, [FModelPath]));
    DoDrainErrors();
    Exit;
  end;

  // Optional archive
  if FMemoryDbPath <> '' then
  begin
    FMemory := TMemory.Create();
    FMemory.SetErrors(FErrors);
    FMemory.MinTurnTokens := FMinTurnTokens;

    if not FMemory.OpenSession(DoResolveDbPath(FMemoryDbPath)) then
    begin
      FErrors.Add(esError, ERR_CHAT_MEMORY,
        Format(RSChatMemoryFailed, [FMemoryDbPath]));
      DoDrainErrors();
      Exit;
    end;

    if FEmbeddings <> nil then
    begin
      if not FMemory.AttachEmbeddings(FEmbeddings) then
      begin
        DoDrainErrors();
        Exit;
      end;
    end;
  end;

  // Conversation policy layer
  FSession := TSession.Create();
  FSession.SetErrors(FErrors);
  FSession.SetInference(FInference);
  FSession.StripThinking := FStripThinking;
  FSession.RecallTopK := FRecallTopK;
  FSession.PinnedTopK := FPinnedTopK;
  FSession.RecallBudgetTokens := FRecallBudgetTokens;
  FSession.MinRecallScore := FMinRecallScore;
  FSession.SetToolRegistry(FToolRegistry);
  FSession.MaxToolRounds := FMaxToolRounds;
  FSession.ToolResultMaxChars := FToolResultMaxChars;
  FSession.SetToolCallback(
    procedure(const AToolName: string; const AArguments: string;
      const AUserData: Pointer)
    begin
      DoToolCall(AToolName, AArguments);
    end, nil);
  if FMemory <> nil then
    FSession.SetMemory(FMemory);
  if FSystemPrompt <> '' then
    FSession.SetSystemPrompt(FSystemPrompt);

  Result := True;
end;

procedure TChat.DoTeardownStack();
begin
  // Reverse dependency order: TSession references TInference + TMemory;
  // TMemory holds a non-owning reference to TEmbeddings
  FreeAndNil(FSession);
  FreeAndNil(FMemory);
  FreeAndNil(FEmbeddings);
  FreeAndNil(FInference);
end;

procedure TChat.DoDrainErrors();
var
  LItems: TList<TError>;
  LI: Integer;
begin
  LItems := FErrors.GetItems();
  for LI := 0 to LItems.Count - 1 do
  begin
    if LItems[LI].Severity in [esError, esFatal] then
      DoError(Format(RSChatErrorFmt, [LItems[LI].Code, LItems[LI].Message]))
    else
      DoInfo(Format(RSChatErrorFmt, [LItems[LI].Code, LItems[LI].Message]));
  end;
  FErrors.Clear();
end;

procedure TChat.DoCheckRebuild();
var
  LUsed: Integer;
  LCtx: Integer;
  LRemoved: Integer;
begin
  // KV occupancy after the last turn. PrefillTokenCount is suffix-only
  // under prefix caching (PLAN-kv-prefix); Position is the true absolute
  // position (prompt + generated) in the KV cache.
  LUsed := FInference.Stats.Position;
  LCtx := FInference.ContextSize();
  if (LCtx <= 0) or (LUsed < Round(FRebuildThreshold * LCtx)) then
    Exit;

  DoInfo(RSChatCompressing);
  LRemoved := FSession.CompactHistory(FKeepRecentTurns);
  if LRemoved > 0 then
    DoInfo(Format(RSChatCompacted, [LRemoved]));
end;

procedure TChat.Run();
var
  LInput: string;
begin
  DoStartup();

  if not DoBuildStack() then
  begin
    DoTeardownStack();
    DoShutdown();
    Exit;
  end;

  DoConfigureStack();

  FRunning := True;
  while FRunning do
  begin
    LInput := DoGetInput();
    if LInput = '' then
      Continue;

    if LInput.StartsWith('/') then
      ProcessCommand(LInput)
    else
      ProcessChat(LInput);
  end;

  DoTeardownStack();
  DoShutdown();
end;

procedure TChat.ProcessChat(const AInput: string);
begin
  // Token output streams through the DoToken callback wired in
  // DoBuildStack; the returned reply is not needed here
  FSession.Ask(AInput, FMaxTokens);
  DoGenerationComplete();

  if FErrors.Count() > 0 then
    DoDrainErrors();

  DoCheckRebuild();
end;

procedure TChat.ProcessCommand(const AInput: string);
var
  LSpacePos: Integer;
  LCmd: string;
  LArgs: string;
begin
  // Split at the first space: /command args
  LSpacePos := Pos(' ', AInput);
  if LSpacePos > 0 then
  begin
    LCmd := LowerCase(Copy(AInput, 1, LSpacePos - 1));
    LArgs := Copy(AInput, LSpacePos + 1, MaxInt).Trim();
  end
  else
  begin
    LCmd := LowerCase(AInput);
    LArgs := '';
  end;

  if LCmd = '/quit' then
    FRunning := False
  else if LCmd = '/clear' then
    DoCmdClear()
  else if LCmd = '/forget' then
    DoCmdForget()
  else if LCmd = '/system' then
    DoCmdSystem(LArgs)
  else if LCmd = '/addfact' then
    DoCmdAddFact(LArgs)
  else if LCmd = '/addfile' then
    DoCmdAddFile(LArgs)
  else if LCmd = '/stats' then
    DoCmdStats()
  else if LCmd = '/turns' then
    DoCmdTurns()
  else if LCmd = '/tokens' then
    DoCmdTokens(LArgs)
  else if LCmd = '/compact' then
    DoCmdCompact()
  else if LCmd = '/save' then
    DoCmdSave(LArgs)
  else if LCmd = '/load' then
    DoCmdLoad(LArgs)
  else if LCmd = '/state' then
    DoCmdState(LArgs)
  else if LCmd = '/restore' then
    DoCmdRestore(LArgs)
  else if LCmd = '/summary' then
    DoCmdSummary()
  else if LCmd = '/dbsave' then
    DoCmdDbSave(LArgs)
  else if LCmd = '/dbload' then
    DoCmdDbLoad(LArgs)
  else if LCmd = '/dblist' then
    DoCmdDbList(LArgs)
  else if LCmd = '/dbreset' then
    DoCmdDbReset()
  else if LCmd = '/help' then
    PrintHelp()
  else
  begin
    // Fall through to derived class custom commands
    if not DoCommand(LCmd, LArgs) then
      DoInfo(Format(RSChatUnknownCommand, [LCmd]));
  end;
end;

procedure TChat.DoCmdClear();
begin
  FSession.ClearHistory();
  DoInfo(RSChatCleared);
end;

procedure TChat.DoCmdForget();
begin
  FSession.ClearHistory();

  if FMemory <> nil then
    FMemory.PurgeDatabase();

  DoInfo(RSChatForgot);
end;

procedure TChat.DoCmdSystem(const AArgs: string);
begin
  if AArgs = '' then
  begin
    DoInfo(RSChatUsageSystem);
    Exit;
  end;

  FSystemPrompt := AArgs;
  FSession.SetSystemPrompt(AArgs);
  DoInfo(RSChatSystemUpdated);
end;

procedure TChat.DoCmdAddFact(const AArgs: string);
begin
  if AArgs = '' then
  begin
    DoInfo(RSChatUsageAddFact);
    Exit;
  end;

  if FMemory = nil then
  begin
    DoInfo(RSChatNoMemory);
    Exit;
  end;

  FMemory.AddFact(AArgs);
  DoDrainErrors();
  DoInfo(RSChatFactAdded);
end;

procedure TChat.DoCmdAddFile(const AArgs: string);
var
  LText: string;
begin
  if AArgs = '' then
  begin
    DoInfo(Format(RSChatUsageFile, ['/addfile']));
    Exit;
  end;

  if FMemory = nil then
  begin
    DoInfo(RSChatNoMemory);
    Exit;
  end;

  if not TFile.Exists(AArgs) then
  begin
    DoInfo(Format(RSChatFileNotFound, [AArgs]));
    Exit;
  end;

  try
    LText := TFile.ReadAllText(AArgs);
  except
    on E: Exception do
    begin
      DoError(Format(RSChatFileReadFailed, [E.Message]));
      Exit;
    end;
  end;

  FMemory.AddDocument(AArgs, TPath.GetFileName(AArgs), LText,
    FDocChunkWords, FDocOverlapWords);
  DoDrainErrors();
  DoInfo(Format(RSChatDocAdded, [AArgs]));
end;

procedure TChat.DoCmdStats();
var
  LStats: TInferenceStats;
  LUsed: Integer;
  LCtx: Integer;
begin
  LStats := FInference.Stats;
  DoInfo(Format(RSChatStatsPrefill,
    [LStats.PrefillTokenCount, LStats.PrefillTokensPerSec,
     LStats.PrefillSec]));
  DoInfo(Format(RSChatStatsGenerate,
    [LStats.TokenCount, LStats.TokensPerSec, LStats.ElapsedSec]));

  // Context occupancy: Position is the true absolute KV position under
  // prefix caching (PrefillTokenCount is suffix-only, see PLAN-kv-prefix)
  LUsed := LStats.Position;
  LCtx := FInference.ContextSize();
  if LCtx > 0 then
    DoInfo(Format(RSChatStatsContext,
      [LUsed, LCtx, 100.0 * LUsed / LCtx]));
end;

procedure TChat.DoCmdTurns();
begin
  DoInfo(Format(RSChatHistoryCount, [FInference.Messages.Count]));
  if FMemory <> nil then
    DoInfo(Format(RSChatArchivedCount, [FMemory.GetTurnCount()]));
end;

procedure TChat.DoCmdTokens(const AArgs: string);
begin
  FMaxTokens := StrToIntDef(AArgs, FMaxTokens);
  DoInfo(Format(RSChatMaxTokensSet, [FMaxTokens]));
end;

procedure TChat.DoCmdCompact();
var
  LRemoved: Integer;
begin
  LRemoved := FSession.CompactHistory(FKeepRecentTurns);
  if LRemoved > 0 then
    DoInfo(Format(RSChatCompacted, [LRemoved]))
  else
    DoInfo(RSChatNothingToCompact);
end;

procedure TChat.DoCmdSave(const AArgs: string);
var
  LPath: string;
begin
  if AArgs = '' then
  begin
    DoInfo(Format(RSChatUsageFile, ['/save']));
    Exit;
  end;

  LPath := TPath.ChangeExtension(AArgs.DeQuotedString('"'), CExtJson);
  TUtils.CreateDirInPath(LPath);

  if FSession.SaveHistory(LPath) then
    DoInfo(Format(RSChatHistorySaved, [LPath]))
  else
  begin
    DoDrainErrors();
    DoError(RSChatOpFailed);
  end;
end;

procedure TChat.DoCmdLoad(const AArgs: string);
var
  LPath: string;
begin
  if AArgs = '' then
  begin
    DoInfo(Format(RSChatUsageFile, ['/load']));
    Exit;
  end;

  LPath := TPath.ChangeExtension(AArgs.DeQuotedString('"'), CExtJson);

  if FSession.LoadHistory(LPath) then
    DoInfo(Format(RSChatHistoryLoaded, [LPath]))
  else
  begin
    DoDrainErrors();
    DoError(RSChatOpFailed);
  end;
end;

procedure TChat.DoCmdState(const AArgs: string);
var
  LPath: string;
begin
  if AArgs = '' then
  begin
    DoInfo(Format(RSChatUsageFile, ['/state']));
    Exit;
  end;

  LPath := TPath.ChangeExtension(AArgs.DeQuotedString('"'), CExtState);
  TUtils.CreateDirInPath(LPath);

  if FSession.SaveState(LPath) then
    DoInfo(Format(RSChatStateSaved, [LPath]))
  else
  begin
    DoDrainErrors();
    DoError(RSChatOpFailed);
  end;
end;

procedure TChat.DoCmdRestore(const AArgs: string);
var
  LPath: string;
begin
  if AArgs = '' then
  begin
    DoInfo(Format(RSChatUsageFile, ['/restore']));
    Exit;
  end;

  LPath := TPath.ChangeExtension(AArgs.DeQuotedString('"'), CExtState);

  if FSession.LoadState(LPath) then
    DoInfo(Format(RSChatStateRestored, [LPath]))
  else
  begin
    DoDrainErrors();
    DoError(RSChatOpFailed);
  end;
end;

procedure TChat.DoCmdSummary();
var
  LSummary: string;
begin
  if FMemory = nil then
  begin
    DoInfo(RSChatNoMemory);
    Exit;
  end;

  LSummary := FMemory.GetSummary();
  if LSummary = '' then
    DoInfo(RSChatSummaryEmpty)
  else
    DoInfo(LSummary);
end;

function TChat.MemoryDbSave(const APath: string): Boolean;
begin
  Result := False;
  if FMemory = nil then
  begin
    DoInfo(RSChatNoMemory);
    Exit;
  end;
  Result := FMemory.SnapshotTo(DoResolveDbPath(APath.DeQuotedString('"')));
end;

function TChat.MemoryDbLoad(const APath: string): Boolean;
var
  LPath: string;
begin
  Result := False;
  if FMemory = nil then
  begin
    DoInfo(RSChatNoMemory);
    Exit;
  end;

  // Ensure correct extension
  LPath := TPath.ChangeExtension(APath.DeQuotedString('"'), CExtDb);
  LPath := DoResolveDbPath(LPath);

  if not FileExists(LPath) then
  begin
    DoError(Format(RSChatFileNotFound, [LPath]));
    Exit;
  end;

  // Close current session and open the new path. OpenSession is
  // idempotent -- it calls CloseSession internally if already open.
  if not FMemory.OpenSession(LPath) then
  begin
    DoDrainErrors();
    Exit;
  end;

  // Re-attach embedder if present
  if FEmbeddings <> nil then
  begin
    if not FMemory.AttachEmbeddings(FEmbeddings) then
    begin
      DoDrainErrors();
      Exit;
    end;
  end;

  // Reset session to clean state -- the loaded memory should not be
  // contaminated by the current conversation's KV cache or history.
  FSession.ClearHistory();

  Result := True;
end;

function TChat.MemoryDbList(const APath: string): TArray<string>;
var
  LPath: string;
begin
  Result := nil;
  LPath := DoResolveDbPath(APath.DeQuotedString('"'));
  if not TDirectory.Exists(LPath) then
    Exit;
  Result := TDirectory.GetFiles(LPath, '*.' + CExtDb,
    TSearchOption.soAllDirectories);
end;

function TChat.MemoryDbReset(): Boolean;
begin
  Result := MemoryDbLoad(FMemoryDbPath);
end;

function TChat.DoResolveDbPath(const APath: string): string;
begin
  Result := TGemma4.ResPath(CResDatabase + PathDelim + APath);
end;

procedure TChat.DoCmdDbSave(const AArgs: string);
begin
  if AArgs = '' then
  begin
    DoInfo(Format(RSChatUsageFile, ['/dbsave']));
    Exit;
  end;

  if MemoryDbSave(AArgs) then
    DoInfo(Format(RSChatDbSaved, [AArgs]))
  else
  begin
    DoDrainErrors();
    DoError(RSChatOpFailed);
  end;
end;

procedure TChat.DoCmdDbLoad(const AArgs: string);
begin
  if AArgs = '' then
  begin
    DoInfo(Format(RSChatUsageFile, ['/dbload']));
    Exit;
  end;

  if MemoryDbLoad(AArgs) then
    DoInfo(Format(RSChatDbLoaded, [AArgs]))
  else
  begin
    DoDrainErrors();
    DoError(RSChatOpFailed);
  end;
end;

procedure TChat.DoCmdDbList(const AArgs: string);
var
  LFiles: TArray<string>;
  LFile: string;
begin
  if AArgs = '' then
  begin
    DoInfo(Format(RSChatUsageFile, ['/dblist']));
    Exit;
  end;

  LFiles := MemoryDbList(AArgs);
  if Length(LFiles) = 0 then
  begin
    DoInfo(Format(RSChatDbNoneFound, [AArgs]));
    Exit;
  end;

  DoInfo(Format(RSChatDbListHeader, [Length(LFiles), AArgs]));
  for LFile in LFiles do
    DoInfo('  ' + LFile);
end;

procedure TChat.DoCmdDbReset();
begin
  if MemoryDbReset() then
    DoInfo(Format(RSChatDbLoaded, [FMemoryDbPath]))
  else
  begin
    DoDrainErrors();
    DoError(RSChatOpFailed);
  end;
end;

procedure TChat.PrintHelp();
begin
  DoInfo(RSChatHelpHeader);
  DoInfo(RSChatHelpQuit);
  DoInfo(RSChatHelpClear);
  DoInfo(RSChatHelpForget);
  DoInfo(RSChatHelpSystem);
  DoInfo(RSChatHelpAddFact);
  DoInfo(RSChatHelpAddFile);
  DoInfo(RSChatHelpStats);
  DoInfo(RSChatHelpTurns);
  DoInfo(RSChatHelpTokens);
  DoInfo(RSChatHelpCompact);
  DoInfo(RSChatHelpSave);
  DoInfo(RSChatHelpLoad);
  DoInfo(RSChatHelpState);
  DoInfo(RSChatHelpRestore);
  DoInfo(RSChatHelpSummary);
  DoInfo(RSChatHelpDbSave);
  DoInfo(RSChatHelpDbLoad);
  DoInfo(RSChatHelpDbList);
  DoInfo(RSChatHelpDbReset);
  DoInfo(RSChatHelpHelp);
end;

{ TConsoleChat }

constructor TConsoleChat.Create();
begin
  inherited Create();
  FAdapter := TConsoleAdapter.Create();
  FAdapter.LineWidth := 100;
  FAdapter.SetOutputCallback(
    procedure(const AText: string; const AUserData: Pointer)
    begin
      TConsole.Print(AText, False);
    end, nil);
end;

destructor TConsoleChat.Destroy();
begin
  FreeAndNil(FAdapter);
  inherited Destroy();
end;

function TConsoleChat.DoGetInput(): string;
var
  LInput: string;
begin
  TConsole.PrintLn('');
  TConsole.Print(COLOR_GREEN + RSChatPrompt + COLOR_RESET, False);
  ReadLn(LInput);
  Result := LInput.Trim();
end;

procedure TConsoleChat.DoOutput(const AText: string);
begin
  TConsole.PrintLn(AText);
end;

procedure TConsoleChat.DoToken(const AToken: string);
begin
  FAdapter.Write(AToken);
end;

function TConsoleChat.DoCancel(): Boolean;
begin
  Result := TInput.KeyDown(VK_ESCAPE);
end;

procedure TConsoleChat.DoError(const AText: string);
begin
  TConsole.PrintLn(COLOR_RED + AText + COLOR_RESET);
end;

procedure TConsoleChat.DoInfo(const AText: string);
begin
  TConsole.PrintLn(COLOR_CYAN + AText + COLOR_RESET);
end;

procedure TConsoleChat.DoStartup();
begin
  TConsole.PrintLn(COLOR_BOLD + COLOR_CYAN + RSChatBanner + COLOR_RESET);
  TConsole.PrintLn(COLOR_CYAN + RSChatBannerHint + COLOR_RESET);
  TConsole.PrintLn('');
end;

procedure TConsoleChat.DoShutdown();
begin
  TConsole.PrintLn('');
  TConsole.PrintLn(COLOR_CYAN + RSChatGoodbye + COLOR_RESET);
end;

procedure TConsoleChat.DoGenerationComplete();
begin
  FAdapter.Flush();
  TConsole.PrintLn('');
end;

procedure TConsoleChat.DoLoadProgress(const AState: TProgressState;
  const AStep: Integer; const ATotal: Integer);
begin
  case AState of
    psStart:
      TConsole.Print(COLOR_GREEN + RSChatLoadingModel, False);

    psInProgress:
      begin
        if ATotal > 0 then
          TConsole.ProgressBar(AStep, ATotal, 40, COLOR_CYAN);
      end;

    psEnd:
      TConsole.ClearLine(True);
  end;
end;

procedure TConsoleChat.DoConfigureStack();
begin
  inherited DoConfigureStack();

  // Colored thinking markers
  GetInference().ThinkingOpenText :=
    COLOR_GREEN + '<thinking>' + COLOR_RESET + #10;
  GetInference().ThinkingCloseText :=
    COLOR_GREEN + #10 + '</thinking>' + COLOR_RESET + #10;
end;

end.
