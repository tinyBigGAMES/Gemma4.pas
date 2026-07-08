{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information

 -------------------------------------------------------------------------------

  Gemma4.Session - Conversation policy, recall injection, tool loop, infinite memory

  Policy layer between raw inference (TInference) and a chat frontend:
    - Ask() drives a turn: build prompt, generate, dispatch tool calls, repeat
    - Recall injection: pulls relevant turns/facts from TMemory into context
    - History management with token budgeting and context-overflow handling
    - Infinite memory: DoSummarizeAndEvict summarizes old turns before dropping
    - SaveState/LoadState persists combined KV blob + history JSON

  Dependencies: System.SysUtils, System.Classes, System.IOUtils,
    System.Generics.Collections/Defaults, StdApp.Base, StdApp.JSON,
    Gemma4.Inference, Gemma4.Memory, Gemma4.Tools
===============================================================================}

unit Gemma4.Session;

{$I StdApp.Defines.inc}

interface

uses
  System.SysUtils,
  System.Classes,
  System.IOUtils,
  System.Generics.Collections,
  System.Generics.Defaults,
  StdApp.Base,
  StdApp.JSON,
  Gemma4.Inference,
  Gemma4.Memory,
  Gemma4.Tools;

const
  { Error codes }
  ERR_SES_NO_INFERENCE = 'SES001';
  ERR_SES_GENERATE     = 'SES002';
  ERR_SES_OVERFLOW     = 'SES003';
  ERR_SES_SAVE         = 'SES004';
  ERR_SES_LOAD         = 'SES005';
  ERR_SES_TOOL_ROUNDS  = 'SES008';
  ERR_SES_STATE_SAVE   = 'SES009';
  ERR_SES_STATE_LOAD   = 'SES010';

type
  { TToolNotifyCallback }
  // Fired once per tool invocation, before the handler runs
  TToolNotifyCallback = reference to procedure(const AToolName: string;
    const AArguments: string; const AUserData: Pointer);

  { TSession }
  // Pure-Delphi conversation policy layer over TInference. Owns: the
  // system-prompt invariant (always message 0), assistant-turn
  // bookkeeping, context-budget trimming, thinking-span stripping,
  // tool dispatch rounds, memory recall injection, summary-on-evict
  // infinite memory, and JSON history persistence.
  TSession = class(TBaseObject)
  private
    FInference: TInference;         // not owned
    FSystemPrompt: string;
    FStripThinking: Boolean;
    FMemory: TMemory;               // not owned
    FRecallTopK: Integer;
    FPinnedTopK: Integer;
    FRecallBudgetTokens: Integer;
    FMinRecallScore: Single;
    FToolRegistry: TToolRegistry;   // not owned
    FMaxToolRounds: Integer;
    FToolResultMaxChars: Integer;
    FToolCallback: TToolNotifyCallback;
    FToolCallbackUserData: Pointer;
    FSummaryMaxTokens: Integer;
    FSummarizeOnEvict: Boolean;
    function DoEstimateTokens(): Integer;
    function DoTrimHistory(const AMaxTokens: Integer;
      const AProtectTail: Integer = 1): Boolean;
    function DoHistoryToJson(): TJSON;
    function DoHistoryFromJson(const AJson: TJSON): Boolean;
    procedure DoCaptureTurn(const ARole: string; const AText: string);
    function DoIsLiveText(const AText: string): Boolean;
    function DoBuildRecall(const AQuery: string): string;
    function DoExtractToolCalls(
      const AParsedJson: string): TArray<TToolCall>;
    procedure DoFireToolCallback(const AToolName: string;
      const AArguments: string);
    function DoSummarizeAndEvict(
      const AKeepRecentTurns: Integer): Integer;
    function DoSummarizeSpan(const AFirst: Integer;
      const AKeepFrom: Integer): Boolean;
  public
    constructor Create(); override;
    destructor Destroy(); override;
    procedure SetInference(const AInference: TInference);
    procedure SetMemory(const AMemory: TMemory);
    procedure SetToolRegistry(const ARegistry: TToolRegistry);
    procedure SetToolCallback(const ACallback: TToolNotifyCallback;
      const AUserData: Pointer);
    procedure SetSystemPrompt(const APrompt: string);
    function Ask(const AText: string; const AMaxTokens: Integer): string;
    procedure ClearHistory();
    // Proactive compaction: remove every message older than the last
    // AKeepRecentTurns user-led exchanges (system prompt survives).
    // When SummarizeOnEvict is True and Memory is attached, evicted
    // turns are summarized first (infinite memory). Returns the
    // number of messages removed.
    function CompactHistory(const AKeepRecentTurns: Integer): Integer;
    function SaveHistory(const AFileName: string): Boolean;
    function LoadHistory(const AFileName: string): Boolean;
    // Combined state: KV cache blob + history JSON in one file
    function SaveState(const AFileName: string): Boolean;
    function LoadState(const AFileName: string): Boolean;
    property Inference: TInference read FInference;
    property Memory: TMemory read FMemory;
    property SystemPrompt: string read FSystemPrompt;
    // Remove the thinking span from replies before they are stored in
    // history (the caller still receives the full reply from Ask).
    property StripThinking: Boolean read FStripThinking write FStripThinking;
    // Recall caps
    property RecallTopK: Integer read FRecallTopK write FRecallTopK;
    property PinnedTopK: Integer read FPinnedTopK write FPinnedTopK;
    // Total token budget for the recall injection
    property RecallBudgetTokens: Integer read FRecallBudgetTokens
      write FRecallBudgetTokens;
    // Minimum cosine similarity for a vector recall hit to be injected.
    // Top-K nearest neighbors are returned regardless of relevance, so
    // without a floor even a greeting pulls in unrelated memories.
    // FTS5 keyword hits are unaffected.
    property MinRecallScore: Single read FMinRecallScore
      write FMinRecallScore;
    property ToolRegistry: TToolRegistry read FToolRegistry;
    // Ceiling on generate->execute->regenerate rounds per Ask
    property MaxToolRounds: Integer read FMaxToolRounds write FMaxToolRounds;
    // Maximum characters per tool result before truncation
    property ToolResultMaxChars: Integer read FToolResultMaxChars
      write FToolResultMaxChars;
    // Infinite memory: max tokens for the summarization generation
    property SummaryMaxTokens: Integer read FSummaryMaxTokens
      write FSummaryMaxTokens;
    // Master switch: when False, falls back to dumb CompactHistory drop
    property SummarizeOnEvict: Boolean read FSummarizeOnEvict
      write FSummarizeOnEvict;
  end;

implementation

uses
  System.Math,
  StdApp.Resources;

const
  { MSG_OVERHEAD_TOKENS }
  // Estimated per-message template wrapper cost (role markers, separators)
  MSG_OVERHEAD_TOKENS = 8;

{ TSession }

constructor TSession.Create();
begin
  inherited Create();
  FInference := nil;
  FMemory := nil;
  FSystemPrompt := '';
  FStripThinking := True;
  FRecallTopK := 5;
  FPinnedTopK := 10;
  FRecallBudgetTokens := 512;
  FMinRecallScore := 0.45;
  FToolRegistry := nil;
  FMaxToolRounds := 4;
  FToolResultMaxChars := 4000;
  FToolCallback := nil;
  FToolCallbackUserData := nil;
  FSummaryMaxTokens := 512;
  FSummarizeOnEvict := True;
end;

destructor TSession.Destroy();
begin
  // FInference, FMemory, FToolRegistry are not owned
  inherited;
end;

procedure TSession.SetInference(const AInference: TInference);
begin
  FInference := AInference;
end;

procedure TSession.SetMemory(const AMemory: TMemory);
begin
  FMemory := AMemory;
end;

procedure TSession.SetToolRegistry(const ARegistry: TToolRegistry);
begin
  FToolRegistry := ARegistry;
end;

procedure TSession.SetToolCallback(const ACallback: TToolNotifyCallback;
  const AUserData: Pointer);
begin
  FToolCallback := ACallback;
  FToolCallbackUserData := AUserData;
end;

procedure TSession.DoFireToolCallback(const AToolName: string;
  const AArguments: string);
begin
  if Assigned(FToolCallback) then
    FToolCallback(AToolName, AArguments, FToolCallbackUserData);
end;

function TSession.DoExtractToolCalls(
  const AParsedJson: string): TArray<TToolCall>;
var
  LJson: TJSON;
  LItem: TJSON;
  LFunc: TJSON;
  LCall: TToolCall;
begin
  Result := nil;

  if AParsedJson = '' then
    Exit;

  LJson := TJSON.FromString(AParsedJson);
  if LJson = nil then
    Exit;
  try
    // ParseToolCalls returns a bare JSON array, not a wrapper object
    if LJson.IsNull() or (not LJson.IsArray()) then
      Exit;

    for LItem in LJson.Items() do
    begin
      if not LItem.Has('function') then
        Continue;
      LFunc := LItem.Get('function');

      LCall := Default(TToolCall);
      LCall.ToolName := LFunc.Get('name').AsString();
      // arguments is a JSON object -- stringify it for TToolRegistry
      if LFunc.Has('arguments') then
        LCall.Arguments := LFunc.Get('arguments').ToString();
      if LItem.Has('id') then
        LCall.CallId := LItem.Get('id').AsString();
      if LCall.ToolName <> '' then
        Result := Result + [LCall];
    end;
  finally
    LJson.Free();
  end;
end;

procedure TSession.SetSystemPrompt(const APrompt: string);
var
  LMsg: TChatMessage;
begin
  FSystemPrompt := APrompt;

  if FInference = nil then
    Exit;

  // Maintain the invariant: the system prompt is message 0, or absent
  if (FInference.Messages.Count > 0) and
     (FInference.Messages[0].Role = CRoleSystem) then
  begin
    if APrompt = '' then
      FInference.Messages.Delete(0)
    else
    begin
      LMsg := FInference.Messages[0];
      LMsg.Content := APrompt;
      FInference.Messages[0] := LMsg;
    end;
  end
  else if APrompt <> '' then
  begin
    LMsg := Default(TChatMessage);
    LMsg.Role := CRoleSystem;
    LMsg.Content := APrompt;
    FInference.Messages.Insert(0, LMsg);
  end;
end;

function TSession.Ask(const AText: string;
  const AMaxTokens: Integer): string;
var
  LRecall: string;
  LInjTokens: Integer;
  LInjected: Boolean;
  LUserIndex: Integer;
  LOrigUserText: string;
  LMsg: TChatMessage;
  LStored: string;
  LRounds: Integer;
  LParsed: string;
  LCalls: TArray<TToolCall>;
  LIdx: Integer;
  LToolOut: string;
  LFloorIndex: Integer;
  LCountBefore: Integer;
  LRemoved: Integer;
begin
  Result := '';

  if FInference = nil then
  begin
    FErrors.Add(esError, ERR_SES_NO_INFERENCE, RSSesNoInference);
    Exit;
  end;

  // Tool declarations refresh per turn
  if (FToolRegistry <> nil) and (FToolRegistry.Count() > 0) then
    FInference.SetTools(FToolRegistry.ToOaiJson())
  else
    FInference.SetTools('');

  FInference.AddMessage(CRoleUser, AText);

  // Capture at the moment of utterance
  DoCaptureTurn(CRoleUser, AText);

  // Recall BEFORE trimming so its token cost can be reserved
  LRecall := DoBuildRecall(AText);
  LInjTokens := 0;
  if LRecall <> '' then
    LInjTokens := FInference.TokenCount(LRecall) + MSG_OVERHEAD_TOKENS;

  // Hard valve: recall must not break the turn it serves
  if (LInjTokens > 0) and
     (FInference.ContextSize() - AMaxTokens - LInjTokens <= 0) then
  begin
    LRecall := '';
    LInjTokens := 0;
  end;

  // Drop oldest exchange pairs until the conversation fits the budget
  if not DoTrimHistory(AMaxTokens + LInjTokens) then
    FErrors.Add(esWarning, ERR_SES_OVERFLOW, RSSesOverflow);

  // LATE injection: prepend recall context into the user message,
  // labelling the live message so the model cannot mistake recalled
  // turns for the current question
  LInjected := LRecall <> '';
  LUserIndex := FInference.Messages.Count - 1;
  if LInjected then
  begin
    LOrigUserText := FInference.Messages[LUserIndex].Content;
    LMsg := FInference.Messages[LUserIndex];
    LMsg.Content := LRecall + sLineBreak + sLineBreak +
      RSSesCurrentMessage + sLineBreak + LOrigUserText;
    FInference.Messages[LUserIndex] := LMsg;
  end;

  // The current user message is the floor -- everything from here to
  // the end is protected from in-loop eviction
  LFloorIndex := LUserIndex;

  // Agentic loop: generate, execute tool calls, feed results back,
  // regenerate until the model answers in plain content or the round
  // guard trips
  LRounds := 0;
  while True do
  begin
    if not FInference.Generate(AMaxTokens) then
    begin
      // Restore the original user message content
      if LInjected then
      begin
        LMsg := FInference.Messages[LUserIndex];
        LMsg.Content := LOrigUserText;
        FInference.Messages[LUserIndex] := LMsg;
      end;
      FErrors.Add(esError, ERR_SES_GENERATE, RSSesGenerateFailed);
      Exit;
    end;

    if (FToolRegistry = nil) or (FToolRegistry.Count() = 0) then
      Break;

    LParsed := FInference.ParseToolCalls();
    LCalls := DoExtractToolCalls(LParsed);
    if Length(LCalls) = 0 then
      Break;

    if LRounds >= FMaxToolRounds then
    begin
      FErrors.Add(esWarning, ERR_SES_TOOL_ROUNDS, RSSesToolRounds);
      Break;
    end;

    // Store the assistant's tool-call turn
    FInference.AddToolCallTurn(LParsed);

    // Execute each tool and add results
    for LIdx := 0 to Length(LCalls) - 1 do
    begin
      DoFireToolCallback(LCalls[LIdx].ToolName, LCalls[LIdx].Arguments);
      LToolOut := FToolRegistry.Execute(LCalls[LIdx]);

      // Truncate oversized tool results
      if (FToolResultMaxChars > 0) and
         (Length(LToolOut) > FToolResultMaxChars) then
        LToolOut := Copy(LToolOut, 1, FToolResultMaxChars) +
          '...[truncated from ' + IntToStr(Length(LToolOut)) + ' chars]';

      FInference.AddToolResult(LCalls[LIdx].ToolName, LToolOut);
    end;

    // Tool payloads grew the prompt -- re-trim before regenerating
    LCountBefore := FInference.Messages.Count;
    if not DoTrimHistory(AMaxTokens, LCountBefore - LFloorIndex) then
      FErrors.Add(esWarning, ERR_SES_OVERFLOW, RSSesOverflow);
    LRemoved := LCountBefore - FInference.Messages.Count;
    if LRemoved > 0 then
    begin
      Dec(LFloorIndex, LRemoved);
      if LInjected then
        Dec(LUserIndex, LRemoved);
    end;

    Inc(LRounds);
  end;

  // Restore original user message content before it enters history
  if LInjected then
  begin
    LMsg := FInference.Messages[LUserIndex];
    LMsg.Content := LOrigUserText;
    FInference.Messages[LUserIndex] := LMsg;
  end;

  Result := FInference.Response;

  // Bookkeeping: the reply joins the history as visible text only.
  // ResponseText excludes thinking spans AND tool-call spans -- neither
  // may be replayed into the model's context nor enter the archive.
  if FStripThinking then
    LStored := FInference.ResponseText
  else
    LStored := Result;

  // A turn with no visible text (e.g. an unparsed tool call) stores
  // nothing -- raw tool-call blobs must not pollute history or memory
  if LStored.Trim() <> '' then
  begin
    FInference.AddMessage(CRoleAssistant, LStored);
    DoCaptureTurn(CRoleAssistant, LStored);
  end;
end;

procedure TSession.ClearHistory();
begin
  if FInference = nil then
    Exit;

  FInference.ClearMessages();

  // Re-establish the system-prompt invariant
  if FSystemPrompt <> '' then
    FInference.AddMessage(CRoleSystem, FSystemPrompt);
end;

function TSession.CompactHistory(
  const AKeepRecentTurns: Integer): Integer;
var
  LFirst: Integer;
  LKeepFrom: Integer;
  LTurns: Integer;
  LI: Integer;
begin
  // When infinite memory is active, summarize before evicting
  if FSummarizeOnEvict and (FMemory <> nil) and FMemory.IsOpen() then
  begin
    Result := DoSummarizeAndEvict(AKeepRecentTurns);
    Exit;
  end;

  Result := 0;

  if (FInference = nil) or (AKeepRecentTurns <= 0) then
    Exit;

  // First droppable message: skip the system prompt at index 0
  if (FInference.Messages.Count > 0) and
     (FInference.Messages[0].Role = CRoleSystem) then
    LFirst := 1
  else
    LFirst := 0;

  // Walk backward counting user messages
  LKeepFrom := LFirst;
  LTurns := 0;
  for LI := FInference.Messages.Count - 1 downto LFirst do
  begin
    if FInference.Messages[LI].Role = CRoleUser then
    begin
      Inc(LTurns);
      if LTurns = AKeepRecentTurns then
      begin
        LKeepFrom := LI;
        Break;
      end;
    end;
  end;

  // Remove [LFirst, LKeepFrom)
  while LFirst < LKeepFrom do
  begin
    FInference.Messages.Delete(LFirst);
    Dec(LKeepFrom);
    Inc(Result);
  end;
end;

function TSession.DoSummarizeAndEvict(
  const AKeepRecentTurns: Integer): Integer;
var
  LFirst: Integer;
  LKeepFrom: Integer;
  LTurns: Integer;
  LI: Integer;
begin
  Result := 0;

  if (FInference = nil) or (AKeepRecentTurns <= 0) then
    Exit;

  // First droppable message: skip the system prompt at index 0
  if (FInference.Messages.Count > 0) and
     (FInference.Messages[0].Role = CRoleSystem) then
    LFirst := 1
  else
    LFirst := 0;

  // Walk backward counting user messages to find the keep window
  LKeepFrom := LFirst;
  LTurns := 0;
  for LI := FInference.Messages.Count - 1 downto LFirst do
  begin
    if FInference.Messages[LI].Role = CRoleUser then
    begin
      Inc(LTurns);
      if LTurns = AKeepRecentTurns then
      begin
        LKeepFrom := LI;
        Break;
      end;
    end;
  end;

  // Nothing to evict
  if LFirst >= LKeepFrom then
    Exit;

  // Summarize the span into the persistent memory summary before it drops
  DoSummarizeSpan(LFirst, LKeepFrom);

  // Now do the actual drop
  while LFirst < LKeepFrom do
  begin
    FInference.Messages.Delete(LFirst);
    Dec(LKeepFrom);
    Inc(Result);
  end;
end;

function TSession.DoSummarizeSpan(const AFirst: Integer;
  const AKeepFrom: Integer): Boolean;
var
  LI: Integer;
  LSb: TStringBuilder;
  LExistingSummary: string;
  LSavedMessages: TList<TChatMessage>;
  LNewSummary: string;
begin
  Result := False;

  if (FInference = nil) or (FMemory = nil) or (not FMemory.IsOpen()) then
    Exit;
  if AFirst >= AKeepFrom then
    Exit;

  // Collect the text of messages about to be evicted
  LSb := TStringBuilder.Create();
  try
    for LI := AFirst to AKeepFrom - 1 do
    begin
      LSb.Append('[');
      LSb.Append(FInference.Messages[LI].Role);
      LSb.Append('] ');
      LSb.AppendLine(FInference.Messages[LI].Content);
    end;

    // Read the existing summary (may be empty on first eviction)
    LExistingSummary := FMemory.GetSummary();

    // Temporarily replace inference messages to generate the summary
    LSavedMessages := TList<TChatMessage>.Create();
    try
      // Save current messages
      for LI := 0 to FInference.Messages.Count - 1 do
        LSavedMessages.Add(FInference.Messages[LI]);

      FInference.ClearMessages();

      // Build the summarization prompt (tuned for Gemma 4 E4B)
      FInference.AddMessage(CRoleSystem,
        'Summarize the following conversation concisely. Preserve key ' +
        'facts, decisions, names, numbers, and any commitments made. ' +
        'If a previous summary exists, merge it with the new content ' +
        'into one cohesive summary. Output ONLY the summary, no preamble.');

      if LExistingSummary <> '' then
        FInference.AddMessage(CRoleUser,
          'Previous summary:' + sLineBreak + LExistingSummary +
          sLineBreak + sLineBreak +
          'New conversation to incorporate:' + sLineBreak +
          LSb.ToString())
      else
        FInference.AddMessage(CRoleUser,
          'Conversation to summarize:' + sLineBreak + LSb.ToString());

      // Internal generation: mute streaming so the summary never
      // reaches the display callback
      FInference.MuteTokens := True;
      try
        if FInference.Generate(FSummaryMaxTokens) then
          LNewSummary := FInference.ResponseText
        else
          LNewSummary := '';
      finally
        FInference.MuteTokens := False;
      end;

      // Restore the original messages
      FInference.ClearMessages();
      for LI := 0 to LSavedMessages.Count - 1 do
        FInference.Messages.Add(LSavedMessages[LI]);
    finally
      LSavedMessages.Free();
    end;
  finally
    LSb.Free();
  end;

  // Store the new summary (an empty result stores nothing)
  if LNewSummary <> '' then
  begin
    FMemory.SetSummary(LNewSummary);
    Result := True;
  end;
end;

function TSession.DoEstimateTokens(): Integer;
var
  LI: Integer;
begin
  Result := 0;
  for LI := 0 to FInference.Messages.Count - 1 do
    Result := Result +
      FInference.TokenCount(FInference.Messages[LI].Content) +
      MSG_OVERHEAD_TOKENS;
end;

function TSession.DoTrimHistory(const AMaxTokens: Integer;
  const AProtectTail: Integer): Boolean;
var
  LBudget: Integer;
  LFirst: Integer;
  LKeepFrom: Integer;
  LCount: Integer;
  LEstimate: Integer;
  LDropped: Integer;
  LFits: Boolean;

  // Estimated cost of one message (mirrors DoEstimateTokens per-item)
  function MsgCost(const AIndex: Integer): Integer;
  begin
    Result := FInference.TokenCount(FInference.Messages[AIndex].Content) +
      MSG_OVERHEAD_TOKENS;
  end;

begin
  Result := False;

  // Reserve the requested reply length out of the context window
  LBudget := FInference.ContextSize() - AMaxTokens;
  if LBudget <= 0 then
    Exit;

  // First droppable message: skip the system prompt at index 0
  if (FInference.Messages.Count > 0) and
     (FInference.Messages[0].Role = CRoleSystem) then
    LFirst := 1
  else
    LFirst := 0;

  // Phase 1 -- simulate the eviction to find the span [LFirst, LKeepFrom)
  // that must go, WITHOUT deleting yet, so the span can be summarized
  // first. Pairing and protected-tail semantics mirror the old
  // drop-as-you-go loop exactly.
  LCount := FInference.Messages.Count;
  LKeepFrom := LFirst;
  LEstimate := DoEstimateTokens();
  LFits := True;

  while LEstimate > LBudget do
  begin
    LDropped := LKeepFrom - LFirst;

    // Never drop into the protected tail (live count = LCount - LDropped)
    if LFirst > (LCount - LDropped) - 1 - AProtectTail then
    begin
      LFits := False;
      Break;
    end;

    // Drop the oldest message of the exchange
    LEstimate := LEstimate - MsgCost(LKeepFrom);
    Inc(LKeepFrom);
    Inc(LDropped);

    // ...and its matching assistant reply if present (same guard as the
    // old loop: the reply must not be the last live message)
    if (LKeepFrom < LCount) and
       (LFirst < (LCount - LDropped) - 1) and
       (FInference.Messages[LKeepFrom].Role = CRoleAssistant) then
    begin
      LEstimate := LEstimate - MsgCost(LKeepFrom);
      Inc(LKeepFrom);
    end;
  end;

  // Nothing must go: fits as-is
  if LKeepFrom <= LFirst then
  begin
    Result := LFits;
    Exit;
  end;

  // Phase 2 -- summarize the doomed span into persistent memory BEFORE
  // dropping it, so trim evictions feed the infinite-memory summary the
  // same way CompactHistory evictions do
  if FSummarizeOnEvict and (FMemory <> nil) and FMemory.IsOpen() then
    DoSummarizeSpan(LFirst, LKeepFrom);

  // Phase 3 -- the actual drop of [LFirst, LKeepFrom)
  while LFirst < LKeepFrom do
  begin
    FInference.Messages.Delete(LFirst);
    Dec(LKeepFrom);
  end;

  Result := LFits;
end;

procedure TSession.DoCaptureTurn(const ARole: string;
  const AText: string);
begin
  if (FMemory = nil) or (not FMemory.IsOpen()) then
    Exit;

  FMemory.AppendTurn(ARole, AText, FInference.TokenCount(AText));
end;

function TSession.DoIsLiveText(const AText: string): Boolean;
var
  LI: Integer;
begin
  Result := False;
  for LI := 0 to FInference.Messages.Count - 1 do
  begin
    if FInference.Messages[LI].Content = AText then
      Exit(True);
  end;
end;

function TSession.DoBuildRecall(const AQuery: string): string;
var
  LHits: TList<TMemoryTurn>;
  LPinned: TArray<TMemoryTurn>;
  LVec: TArray<TMemoryTurn>;
  LFts: TArray<TMemoryTurn>;
  LFound: Boolean;
  LI: Integer;
  LJ: Integer;
  LSb: TStringBuilder;
  LLine: string;
  LCost: Integer;
  LUsed: Integer;
  LCount: Integer;
  LSummary: string;
begin
  Result := '';

  if (FMemory = nil) or (not FMemory.IsOpen()) then
    Exit;
  if FMemory.GetTurnCount() = 0 then
    Exit;

  LHits := TList<TMemoryTurn>.Create();
  try
    // Standing profile: pinned facts ride along on every turn
    LPinned := FMemory.GetPinnedTurns(FPinnedTopK);

    // Semantic hits first; keyword hits fill the remaining slots.
    // Hits that echo a message still in the live context are skipped;
    // hits below the similarity floor are noise, not memories
    if FMemory.HasEmbedder() then
    begin
      LVec := FMemory.SearchVector(AQuery, FRecallTopK);
      for LI := 0 to High(LVec) do
      begin
        if (LVec[LI].CosineScore >= FMinRecallScore) and
           (not DoIsLiveText(LVec[LI].Text)) then
          LHits.Add(LVec[LI]);
      end;
    end;

    LFts := FMemory.SearchFTS5(AQuery, FRecallTopK);
    for LI := 0 to High(LFts) do
    begin
      if LHits.Count >= FRecallTopK then
        Break;
      if DoIsLiveText(LFts[LI].Text) then
        Continue;

      LFound := False;
      for LJ := 0 to LHits.Count - 1 do
      begin
        if LHits[LJ].TurnId = LFts[LI].TurnId then
        begin
          LFound := True;
          Break;
        end;
      end;

      if not LFound then
        LHits.Add(LFts[LI]);
    end;

    // Drop search hits that duplicate a pinned fact
    for LI := LHits.Count - 1 downto 0 do
    begin
      for LJ := 0 to High(LPinned) do
      begin
        if LHits[LI].TurnId = LPinned[LJ].TurnId then
        begin
          LHits.Delete(LI);
          Break;
        end;
      end;
    end;

    // Get the conversation summary (infinite memory)
    LSummary := FMemory.GetSummary();

    if (LHits.Count = 0) and (Length(LPinned) = 0) and
       (LSummary = '') then
      Exit;

    // Chronological order reads naturally when injected
    LHits.Sort(TComparer<TMemoryTurn>.Construct(
      function(const ALeft, ARight: TMemoryTurn): Integer
      begin
        Result := ALeft.TurnIndex - ARight.TurnIndex;
      end));

    LSb := TStringBuilder.Create();
    try
      LUsed := 0;
      LCount := 0;

      // INFINITE MEMORY: inject the conversation summary first
      // (always, not query-dependent). Gets priority over search hits.
      if LSummary <> '' then
      begin
        LLine := RSSesSummaryHeader + sLineBreak + LSummary;
        LCost := FInference.TokenCount(LLine);
        if LCost <= FRecallBudgetTokens then
        begin
          LSb.AppendLine(LLine);
          LSb.AppendLine('');
          LUsed := LCost;
          Inc(LCount);
        end;
      end;

      // Recall header
      if (Length(LPinned) > 0) or (LHits.Count > 0) then
      begin
        LLine := RSMemRecallHeader;
        LCost := FInference.TokenCount(LLine);
        if LUsed + LCost <= FRecallBudgetTokens then
        begin
          LSb.AppendLine(LLine);
          LUsed := LUsed + LCost;
        end;
      end;

      // Token-budgeted injection: pinned facts first, then search hits
      for LI := 0 to High(LPinned) do
      begin
        LLine := Format('[%s] %s', [LPinned[LI].Role, LPinned[LI].Text]);
        LCost := FInference.TokenCount(LLine);
        if LUsed + LCost > FRecallBudgetTokens then
          Continue;
        LSb.AppendLine(LLine);
        LUsed := LUsed + LCost;
        Inc(LCount);
      end;

      for LI := 0 to LHits.Count - 1 do
      begin
        LLine := Format('[%s] %s', [LHits[LI].Role, LHits[LI].Text]);
        LCost := FInference.TokenCount(LLine);
        if LUsed + LCost > FRecallBudgetTokens then
          Continue;
        LSb.AppendLine(LLine);
        LUsed := LUsed + LCost;
        Inc(LCount);
      end;

      if LCount > 0 then
        Result := LSb.ToString().Trim()
      else
        Result := '';
    finally
      LSb.Free();
    end;
  finally
    LHits.Free();
  end;
end;

function TSession.DoHistoryToJson(): TJSON;
var
  LI: Integer;
begin
  Result := TJSON.Create();
  Result.Add('version', 1);
  Result.BeginArray('messages');
  for LI := 0 to FInference.Messages.Count - 1 do
  begin
    Result.BeginObject()
      .Add('role', FInference.Messages[LI].Role)
      .Add('content', FInference.Messages[LI].Content);
    if FInference.Messages[LI].ToolName <> '' then
      Result.Add('toolname', FInference.Messages[LI].ToolName);
    if FInference.Messages[LI].ToolCallsJson <> '' then
      Result.Add('toolcalls', FInference.Messages[LI].ToolCallsJson);
    Result.EndObject();
  end;
  Result.EndArray();
end;

function TSession.DoHistoryFromJson(const AJson: TJSON): Boolean;
var
  LItem: TJSON;
  LRole: string;
begin
  Result := False;

  if AJson.IsNull() or (not AJson.Get('messages').IsArray()) then
  begin
    FErrors.Add(esError, ERR_SES_LOAD, RSSesInvalidFormat);
    Exit;
  end;

  // Format validated -- replace the conversation
  FInference.ClearMessages();
  FSystemPrompt := '';

  for LItem in AJson.Get('messages').Items() do
  begin
    if not (LItem.Has('role') and LItem.Has('content')) then
      Continue;

    LRole := LItem.Get('role').AsString();
    if LRole = '' then
      Continue;

    if LItem.Has('toolcalls') then
    begin
      // Tool-call assistant turn
      FInference.AddToolCallTurn(LItem.Get('toolcalls').AsString());
    end
    else if LItem.Has('toolname') then
      FInference.AddToolResult(LItem.Get('toolname').AsString(),
        LItem.Get('content').AsString())
    else
      FInference.AddMessage(LRole, LItem.Get('content').AsString());

    // Restore the system-prompt invariant
    if (LRole = CRoleSystem) and (FSystemPrompt = '') then
      FSystemPrompt := LItem.Get('content').AsString();
  end;

  Result := True;
end;

function TSession.SaveHistory(const AFileName: string): Boolean;
var
  LJson: TJSON;
begin
  Result := False;

  if FInference = nil then
  begin
    FErrors.Add(esError, ERR_SES_NO_INFERENCE, RSSesNoInference);
    Exit;
  end;

  LJson := DoHistoryToJson();
  try
    try
      LJson.SaveToFile(AFileName);
      Result := True;
    except
      on E: Exception do
        FErrors.Add(esError, ERR_SES_SAVE, RSSesSaveFailed, [E.Message]);
    end;
  finally
    LJson.Free();
  end;
end;

function TSession.LoadHistory(const AFileName: string): Boolean;
var
  LJson: TJSON;
begin
  Result := False;

  if FInference = nil then
  begin
    FErrors.Add(esError, ERR_SES_NO_INFERENCE, RSSesNoInference);
    Exit;
  end;

  if not TFile.Exists(AFileName) then
  begin
    FErrors.Add(esError, ERR_SES_LOAD, RSSesLoadFailed, [AFileName]);
    Exit;
  end;

  LJson := TJSON.FromFile(AFileName);
  try
    Result := DoHistoryFromJson(LJson);
  finally
    LJson.Free();
  end;
end;

function TSession.SaveState(const AFileName: string): Boolean;
var
  LKvPath: string;
  LJsonBytes: TBytes;
  LJson: TJSON;
  LStream: TFileStream;
  LKvStream: TFileStream;
  LKvSize: Int64;
  LBuf: TBytes;
  LRead: Integer;
begin
  Result := False;

  if FInference = nil then
  begin
    FErrors.Add(esError, ERR_SES_NO_INFERENCE, RSSesNoInference);
    Exit;
  end;

  // Step 1: save KV cache to a temp file
  LKvPath := AFileName + '.kvtmp';
  try
    if not FInference.SaveKVState(LKvPath) then
    begin
      FErrors.Add(esError, ERR_SES_STATE_SAVE, 'KV state save failed');
      Exit;
    end;

    // Step 2: build history JSON bytes
    LJson := DoHistoryToJson();
    try
      LJsonBytes := TEncoding.UTF8.GetBytes(LJson.ToString());
    finally
      LJson.Free();
    end;

    // Step 3: write combined file: [kv_size:Int64][kv_blob][json_bytes]
    try
      LStream := TFileStream.Create(AFileName, fmCreate);
      try
        LKvStream := TFileStream.Create(LKvPath, fmOpenRead or fmShareDenyWrite);
        try
          LKvSize := LKvStream.Size;
          LStream.WriteBuffer(LKvSize, SizeOf(Int64));

          // Copy KV blob in chunks
          SetLength(LBuf, 65536);
          repeat
            LRead := LKvStream.Read(LBuf[0], Length(LBuf));
            if LRead > 0 then
              LStream.WriteBuffer(LBuf[0], LRead);
          until LRead = 0;
        finally
          LKvStream.Free();
        end;

        // Append JSON
        if Length(LJsonBytes) > 0 then
          LStream.WriteBuffer(LJsonBytes[0], Length(LJsonBytes));
      finally
        LStream.Free();
      end;

      Result := True;
    except
      on E: Exception do
        FErrors.Add(esError, ERR_SES_STATE_SAVE,
          'Failed to write state file: ' + E.Message);
    end;
  finally
    if TFile.Exists(LKvPath) then
      TFile.Delete(LKvPath);
  end;
end;

function TSession.LoadState(const AFileName: string): Boolean;
var
  LKvPath: string;
  LStream: TFileStream;
  LKvStream: TFileStream;
  LKvSize: Int64;
  LBuf: TBytes;
  LRead: Integer;
  LRemaining: Int64;
  LJsonSize: Int64;
  LJsonBytes: TBytes;
  LJsonStr: string;
  LJson: TJSON;
begin
  Result := False;

  if FInference = nil then
  begin
    FErrors.Add(esError, ERR_SES_NO_INFERENCE, RSSesNoInference);
    Exit;
  end;

  if not TFile.Exists(AFileName) then
  begin
    FErrors.Add(esError, ERR_SES_STATE_LOAD,
      'State file not found: ' + AFileName);
    Exit;
  end;

  LKvPath := AFileName + '.kvtmp';
  try
    try
      LStream := TFileStream.Create(AFileName, fmOpenRead or fmShareDenyWrite);
      try
        // Read KV blob size
        LStream.ReadBuffer(LKvSize, SizeOf(Int64));

        // Extract KV blob to temp file
        LKvStream := TFileStream.Create(LKvPath, fmCreate);
        try
          SetLength(LBuf, 65536);
          LRemaining := LKvSize;
          while LRemaining > 0 do
          begin
            LRead := LStream.Read(LBuf[0], Min(Length(LBuf), LRemaining));
            if LRead = 0 then Break;
            LKvStream.WriteBuffer(LBuf[0], LRead);
            Dec(LRemaining, LRead);
          end;
        finally
          LKvStream.Free();
        end;

        // Read remaining bytes as JSON
        LJsonSize := LStream.Size - LStream.Position;
        if LJsonSize > 0 then
        begin
          SetLength(LJsonBytes, LJsonSize);
          LStream.ReadBuffer(LJsonBytes[0], LJsonSize);
        end
        else
          SetLength(LJsonBytes, 0);
      finally
        LStream.Free();
      end;

      // Step 2: load KV cache from temp file
      if not FInference.LoadKVState(LKvPath) then
      begin
        FErrors.Add(esError, ERR_SES_STATE_LOAD, 'KV state load failed');
        Exit;
      end;

      // Step 3: restore history from JSON
      if Length(LJsonBytes) > 0 then
      begin
        LJsonStr := TEncoding.UTF8.GetString(LJsonBytes);
        LJson := TJSON.FromString(LJsonStr);
        try
          DoHistoryFromJson(LJson);
        finally
          LJson.Free();
        end;
      end;

      Result := True;
    except
      on E: Exception do
        FErrors.Add(esError, ERR_SES_STATE_LOAD,
          'Failed to read state file: ' + E.Message);
    end;
  finally
    if TFile.Exists(LKvPath) then
      TFile.Delete(LKvPath);
  end;
end;

end.
