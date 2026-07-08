unit UTemp;

interface

procedure Test_Tools();
procedure Test_ToolsUtils();
procedure Test_HNSW();
procedure Test_Memory();
procedure Test_Session();
procedure Test_Chat();
procedure Test_KVState();

implementation

uses
  System.SysUtils,
  System.IOUtils,
  StdApp.Base,
  StdApp.JSON,
  StdApp.Console,
  Gemma4.Tools,
  Gemma4.Tools.Utils,
  Gemma4.HNSW,
  Gemma4.Common,
  Gemma4.Memory,
  Gemma4.Session,
  Gemma4.Inference,
  Gemma4.Chat,
  UTestbed.Common;

{ Test_Tools }

procedure Test_Tools();
var
  LRegistry: TToolRegistry;
  LParams: TToolParams;
  LCall: TToolCall;
  LJson: string;
  LResult: string;
  LPass: Boolean;
begin
  LPass := True;
  TConsole.PrintLn('--- Test_Tools ---', []);

  LRegistry := TToolRegistry.Create();
  try
    // Define a tool: get_weather(location: string required, unit: string optional)
    LRegistry
      .DefineTool('get_weather')
        .Description('Get current weather for a location')
        .Param('location', tptString, 'City name', True)
        .Param('unit', tptString, 'Temperature unit (C or F)', False)
        .OnExecute(
          function(const AToolName: string; const AParams: TToolParams): string
          begin
            Result := ToolResult(['temperature', '22C', 'location', AParams.AsString('location')]);
          end
        );

    // 1. Count
    if LRegistry.Count() <> 1 then
    begin
      TConsole.PrintLn('  FAIL: Count = %d, expected 1', [LRegistry.Count()]);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: Count = 1', []);

    // 2. HasTool
    if not LRegistry.HasTool('get_weather') then
    begin
      TConsole.PrintLn('  FAIL: HasTool(get_weather) = False', []);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: HasTool(get_weather) = True', []);

    // 3. ToOaiJson -- verify non-empty and contains expected fragments
    LJson := LRegistry.ToOaiJson();
    if (LJson = '') or (not LJson.Contains('"get_weather"')) or
       (not LJson.Contains('"location"')) then
    begin
      TConsole.PrintLn('  FAIL: ToOaiJson missing expected content', []);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: ToOaiJson contains expected schema', []);

    // 4. TToolParams -- parse and read
    LParams := TToolParams.Create();
    try
      LParams.Parse('{"location":"London","count":5,"active":true}');
      if LParams.AsString('location') <> 'London' then
      begin
        TConsole.PrintLn('  FAIL: AsString(location) = %s', [LParams.AsString('location')]);
        LPass := False;
      end
      else
        TConsole.PrintLn('  OK: AsString(location) = London', []);

      if LParams.AsInteger('count') <> 5 then
      begin
        TConsole.PrintLn('  FAIL: AsInteger(count) = %d', [LParams.AsInteger('count')]);
        LPass := False;
      end
      else
        TConsole.PrintLn('  OK: AsInteger(count) = 5', []);

      if LParams.AsBoolean('active') <> True then
      begin
        TConsole.PrintLn('  FAIL: AsBoolean(active) = False', []);
        LPass := False;
      end
      else
        TConsole.PrintLn('  OK: AsBoolean(active) = True', []);

      if LParams.Has('missing') then
      begin
        TConsole.PrintLn('  FAIL: Has(missing) = True', []);
        LPass := False;
      end
      else
        TConsole.PrintLn('  OK: Has(missing) = False', []);
    finally
      LParams.Free();
    end;

    // 5. Execute known tool
    LCall.ToolName := 'get_weather';
    LCall.Arguments := '{"location":"London"}';
    LCall.CallId := '';
    LResult := LRegistry.Execute(LCall);
    if (LResult = '') or (not LResult.Contains('22C')) then
    begin
      TConsole.PrintLn('  FAIL: Execute result = %s', [LResult]);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: Execute(get_weather) returned expected result', []);

    // 6. Execute unknown tool -- should add error, return error JSON
    LCall.ToolName := 'nonexistent';
    LCall.Arguments := '{}';
    LResult := LRegistry.Execute(LCall);
    if not LResult.Contains('error') then
    begin
      TConsole.PrintLn('  FAIL: Execute(nonexistent) did not return error', []);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: Execute(nonexistent) returned error JSON', []);
  finally
    LRegistry.Free();
  end;

  if LPass then
    TConsole.PrintLn('Test_Tools: PASS', [])
  else
    TConsole.PrintLn('Test_Tools: FAIL', []);
  TConsole.PrintLn('', []);
end;

procedure Test_ToolsUtils();
var
  LResp: TToolResponse;
  LReq: TRestRequest;
  LResult: string;
  LPass: Boolean;
begin
  LPass := True;
  TConsole.PrintLn('--- Test_ToolsUtils ---', []);

  // 1. TToolResponse -- create, set result, verify accessors
  LResp := TToolResponse.Create();
  try
    LResp.SetResult(True, 200, '',
      '{"temperature":"22C","humidity":65,"active":true}');

    if not LResp.Success() then
    begin
      TConsole.PrintLn('  FAIL: Success() = False', []);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: Success() = True', []);

    if LResp.StatusCode() <> 200 then
    begin
      TConsole.PrintLn('  FAIL: StatusCode() = %d', [LResp.StatusCode()]);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: StatusCode() = 200', []);

    if LResp.Json() = nil then
    begin
      TConsole.PrintLn('  FAIL: Json() = nil', []);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: Json() parsed', []);

    if LResp.AsString('temperature') <> '22C' then
    begin
      TConsole.PrintLn('  FAIL: AsString(temperature) = %s',
        [LResp.AsString('temperature')]);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: AsString(temperature) = 22C', []);

    if LResp.AsInteger('humidity') <> 65 then
    begin
      TConsole.PrintLn('  FAIL: AsInteger(humidity) = %d',
        [LResp.AsInteger('humidity')]);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: AsInteger(humidity) = 65', []);

    if LResp.AsBoolean('active') <> True then
    begin
      TConsole.PrintLn('  FAIL: AsBoolean(active) = False', []);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: AsBoolean(active) = True', []);

    // Default for missing key
    if LResp.AsString('missing', 'default') <> 'default' then
    begin
      TConsole.PrintLn('  FAIL: AsString(missing) did not return default', []);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: AsString(missing) = default', []);
  finally
    LResp.Free();
  end;

  // 2. TToolResponse with empty body -- Json() should be nil
  LResp := TToolResponse.Create();
  try
    LResp.SetResult(True, 200, '', '');
    if LResp.Json() <> nil then
    begin
      TConsole.PrintLn('  FAIL: Json() should be nil for empty body', []);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: Json() = nil for empty body', []);

    if LResp.RawJson() <> '' then
    begin
      TConsole.PrintLn('  FAIL: RawJson() should be empty', []);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: RawJson() = empty', []);
  finally
    LResp.Free();
  end;

  // 3. TToolResponse with error
  LResp := TToolResponse.Create();
  try
    LResp.SetResult(False, 0, 'connection refused', '');
    if LResp.Success() then
    begin
      TConsole.PrintLn('  FAIL: Success() should be False for error', []);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: Success() = False for error', []);

    if LResp.ErrorMsg() <> 'connection refused' then
    begin
      TConsole.PrintLn('  FAIL: ErrorMsg() = %s', [LResp.ErrorMsg()]);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: ErrorMsg() = connection refused', []);
  finally
    LResp.Free();
  end;

  // 4. ToolResult helper
  LResult := ToolResult(['city', 'London', 'temp', '22C']);
  if (not LResult.Contains('"city"')) or (not LResult.Contains('"London"')) or
     (not LResult.Contains('"temp"')) or (not LResult.Contains('"22C"')) then
  begin
    TConsole.PrintLn('  FAIL: ToolResult = %s', [LResult]);
    LPass := False;
  end
  else
    TConsole.PrintLn('  OK: ToolResult produced valid JSON', []);

  // 5. TRestRequest -- verify URL building (no network call)
  LReq := TRestRequest.Create();
  try
    LReq.SetBaseUrl('https://example.com/api');
    LReq.Query('q', 'hello world');
    LReq.Query('limit', '10');
    LReq.Header('X-Custom', 'test');
    LReq.Bearer('my_token');
    // BuildUrl is private, so we just verify the object was constructed
    // without error. Actual HTTP is tested manually.
    TConsole.PrintLn('  OK: TRestRequest constructed with fluent API', []);
  finally
    LReq.Free();
  end;

  // 6. ToolTimeNow / ToolDateNow -- verify non-empty
  if ToolTimeNow() = '' then
  begin
    TConsole.PrintLn('  FAIL: ToolTimeNow() empty', []);
    LPass := False;
  end
  else
    TConsole.PrintLn('  OK: ToolTimeNow() = %s', [ToolTimeNow()]);

  if ToolDateNow() = '' then
  begin
    TConsole.PrintLn('  FAIL: ToolDateNow() empty', []);
    LPass := False;
  end
  else
    TConsole.PrintLn('  OK: ToolDateNow() = %s', [ToolDateNow()]);

  if LPass then
    TConsole.PrintLn('Test_ToolsUtils: PASS', [])
  else
    TConsole.PrintLn('Test_ToolsUtils: FAIL', []);
  TConsole.PrintLn('', []);
end;

procedure Test_HNSW();

  // Build a unit-length vector with one hot dimension for easy cosine math
  function MakeVec(const AD0, AD1, AD2, AD3: Single): TArray<Single>;
  var
    LNorm: Single;
  begin
    SetLength(Result, 4);
    Result[0] := AD0; Result[1] := AD1;
    Result[2] := AD2; Result[3] := AD3;
    // L2 normalize so cosine distance = 1 - dot
    LNorm := Sqrt(AD0*AD0 + AD1*AD1 + AD2*AD2 + AD3*AD3);
    if LNorm > 0 then
    begin
      Result[0] := Result[0] / LNorm;
      Result[1] := Result[1] / LNorm;
      Result[2] := Result[2] / LNorm;
      Result[3] := Result[3] / LNorm;
    end;
  end;

var
  LIndex: THNSWIndex;
  LConfig: THNSWConfig;
  LResults: TArray<THNSWSearchResult>;
  LBytes: TBytes;
  LPass: Boolean;
  LQuery: TArray<Single>;
begin
  LPass := True;
  TConsole.PrintLn('--- Test_HNSW ---', []);

  LConfig := Default(THNSWConfig);
  LConfig.Dim := 4;
  LConfig.M := 4;
  LConfig.EfConstruction := 20;
  LConfig.EfSearch := 10;

  LIndex := THNSWIndex.Create();
  try
    LIndex.Init(LConfig);
    // Seed RNG for reproducible levels
    RandSeed := 42;

    // Insert 10 vectors with distinct directions
    LIndex.Insert(1,  MakeVec(1.0, 0.0, 0.0, 0.0));
    LIndex.Insert(2,  MakeVec(0.0, 1.0, 0.0, 0.0));
    LIndex.Insert(3,  MakeVec(0.0, 0.0, 1.0, 0.0));
    LIndex.Insert(4,  MakeVec(0.0, 0.0, 0.0, 1.0));
    LIndex.Insert(5,  MakeVec(1.0, 1.0, 0.0, 0.0));
    LIndex.Insert(6,  MakeVec(0.0, 1.0, 1.0, 0.0));
    LIndex.Insert(7,  MakeVec(0.0, 0.0, 1.0, 1.0));
    LIndex.Insert(8,  MakeVec(1.0, 0.0, 0.0, 1.0));
    LIndex.Insert(9,  MakeVec(1.0, 1.0, 1.0, 0.0));
    LIndex.Insert(10, MakeVec(1.0, 1.0, 1.0, 1.0));

    // 1. NodeCount
    if LIndex.NodeCount() <> 10 then
    begin
      TConsole.PrintLn('  FAIL: NodeCount = %d, expected 10', [LIndex.NodeCount()]);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: NodeCount = 10', []);

    // 2. Search -- query near (1,0,0,0), top-1 should be node 1
    LQuery := MakeVec(1.0, 0.1, 0.0, 0.0);
    LResults := LIndex.Search(LQuery, 3);
    if Length(LResults) < 1 then
    begin
      TConsole.PrintLn('  FAIL: Search returned 0 results', []);
      LPass := False;
    end
    else if LResults[0].NodeId <> 1 then
    begin
      TConsole.PrintLn('  FAIL: Top-1 = %d, expected 1 (dist=%.4f)',
        [LResults[0].NodeId, LResults[0].Distance]);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: Top-1 = node 1 (dist=%.4f)', [LResults[0].Distance]);

    // 3. Delete node 1, search again -- node 1 should be excluded
    LIndex.Delete(1);
    if LIndex.NodeCount() <> 9 then
    begin
      TConsole.PrintLn('  FAIL: NodeCount after delete = %d, expected 9',
        [LIndex.NodeCount()]);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: NodeCount after delete = 9', []);

    LResults := LIndex.Search(LQuery, 3);
    if (Length(LResults) >= 1) and (LResults[0].NodeId = 1) then
    begin
      TConsole.PrintLn('  FAIL: Deleted node 1 still in search results', []);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: Deleted node 1 excluded from results (top=%d)',
        [LResults[0].NodeId]);

    // 4. Save/Load round-trip
    LBytes := LIndex.SaveToBytes();
    TConsole.PrintLn('  OK: SaveToBytes = %d bytes', [Length(LBytes)]);

    LIndex.Clear();
    if LIndex.NodeCount() <> 0 then
    begin
      TConsole.PrintLn('  FAIL: NodeCount after Clear = %d', [LIndex.NodeCount()]);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: Clear -> NodeCount = 0', []);

    LIndex.LoadFromBytes(LBytes);
    if LIndex.NodeCount() <> 9 then
    begin
      TConsole.PrintLn('  FAIL: NodeCount after Load = %d, expected 9',
        [LIndex.NodeCount()]);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: LoadFromBytes -> NodeCount = 9', []);

    // 5. Search after load -- same query, same expected top-1 (not node 1)
    LResults := LIndex.Search(LQuery, 3);
    if (Length(LResults) >= 1) and (LResults[0].NodeId = 1) then
    begin
      TConsole.PrintLn('  FAIL: Deleted node 1 in results after load', []);
      LPass := False;
    end
    else if Length(LResults) < 1 then
    begin
      TConsole.PrintLn('  FAIL: Search after load returned 0 results', []);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: Search after load top=%d (dist=%.4f)',
        [LResults[0].NodeId, LResults[0].Distance]);
  finally
    LIndex.Free();
  end;

  if LPass then
    TConsole.PrintLn('Test_HNSW: PASS', [])
  else
    TConsole.PrintLn('Test_HNSW: FAIL', []);
  TConsole.PrintLn('', []);
end;

procedure Test_Memory();
var
  LMem: TMemory;
  LDbPath: string;
  LId1: Int64;
  LId2: Int64;
  LFactId: Int64;
  LTurns: TArray<TMemoryTurn>;
  LFts: TArray<TMemoryTurn>;
  LPinned: TArray<TMemoryTurn>;
  LPass: Boolean;
begin
  LPass := True;
  TConsole.PrintLn('--- Test_Memory ---', []);

  LDbPath := TGemma4.ResPath(CResDatabase + PathDelim + 'test_memory.db');

  // Clean up any leftover from previous run
  if TFile.Exists(LDbPath) then
    TFile.Delete(LDbPath);

  LMem := TMemory.Create();
  try
    // 1. Open session
    if not LMem.OpenSession(LDbPath) then
    begin
      TConsole.PrintLn('  FAIL: OpenSession returned False', []);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: OpenSession', []);

    // 2. AppendTurn
    LId1 := LMem.AppendTurn('user', 'Hello world, this is a test message', 7);
    if LId1 <= 0 then
    begin
      TConsole.PrintLn('  FAIL: AppendTurn returned %d', [LId1]);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: AppendTurn returned turn_id=%d', [LId1]);

    // 3. Dedup -- same text should return same turn_id
    LId2 := LMem.AppendTurn('user', 'Hello world, this is a test message', 7);
    if LId2 <> LId1 then
    begin
      TConsole.PrintLn('  FAIL: Dedup failed, got %d expected %d', [LId2, LId1]);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: Dedup returned same turn_id=%d', [LId2]);

    // 4. GetRecentTurns
    LTurns := LMem.GetRecentTurns(10);
    if Length(LTurns) <> 1 then
    begin
      TConsole.PrintLn('  FAIL: GetRecentTurns count=%d, expected 1', [Length(LTurns)]);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: GetRecentTurns count=1', []);

    // 5. SearchFTS5
    LFts := LMem.SearchFTS5('hello', 5);
    if Length(LFts) < 1 then
    begin
      TConsole.PrintLn('  FAIL: SearchFTS5(hello) returned 0 results', []);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: SearchFTS5(hello) found %d result(s)', [Length(LFts)]);

    // 6. AddFact (pinned)
    LFactId := LMem.AddFact('The sky is blue', True);
    if LFactId <= 0 then
    begin
      TConsole.PrintLn('  FAIL: AddFact returned %d', [LFactId]);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: AddFact returned turn_id=%d', [LFactId]);

    // 7. GetPinnedTurns
    LPinned := LMem.GetPinnedTurns(10);
    if Length(LPinned) < 1 then
    begin
      TConsole.PrintLn('  FAIL: GetPinnedTurns returned 0', []);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: GetPinnedTurns found %d pinned turn(s)', [Length(LPinned)]);

    // 8. SetSummary / GetSummary round-trip
    LMem.SetSummary('User discussed weather and greetings');
    if LMem.GetSummary() <> 'User discussed weather and greetings' then
    begin
      TConsole.PrintLn('  FAIL: GetSummary mismatch', []);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: SetSummary/GetSummary round-trip', []);

    // 9. GetTurnCount before purge
    TConsole.PrintLn('  INFO: TurnCount before purge = %d', [LMem.GetTurnCount()]);

    // 10. PurgeAll
    LMem.PurgeAll();
    if LMem.GetTurnCount() <> 0 then
    begin
      TConsole.PrintLn('  FAIL: PurgeAll did not clear turns (count=%d)', [LMem.GetTurnCount()]);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: PurgeAll -> TurnCount=0', []);

    // 11. Summary survives PurgeAll (session_meta is not cleared)
    if LMem.GetSummary() <> 'User discussed weather and greetings' then
    begin
      TConsole.PrintLn('  FAIL: Summary lost after PurgeAll', []);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: Summary survives PurgeAll', []);

    // 12. SetMeta / GetMeta
    LMem.SetMeta('test_key', 'test_value');
    if LMem.GetMeta('test_key') <> 'test_value' then
    begin
      TConsole.PrintLn('  FAIL: GetMeta mismatch', []);
      LPass := False;
    end
    else
      TConsole.PrintLn('  OK: SetMeta/GetMeta round-trip', []);

    LMem.CloseSession();
    TConsole.PrintLn('  OK: CloseSession', []);
  finally
    LMem.Free();
  end;

  // Clean up temp file
  if TFile.Exists(LDbPath) then
    TFile.Delete(LDbPath);

  if LPass then
    TConsole.PrintLn('Test_Memory: PASS', [])
  else
    TConsole.PrintLn('Test_Memory: FAIL', []);
  TConsole.PrintLn('', []);
end;

procedure Test_Session();
var
  LInf: TInference;
  LSes: TSession;
  LMem: TMemory;
  LDbPath: string;
  LHistFile: string;
  LReply: string;
  LRemoved: Integer;
  LPass: Boolean;
begin
  LPass := True;
  TConsole.PrintLn('--- Test_Session ---', []);

  LInf := TInference.Create();
  try
    // 1. Load model
    TConsole.PrintLn('  Loading model...', []);
    if not LInf.LoadModel(CVpkOutputFile) then
    begin
      TConsole.PrintLn('  FAIL: LoadModel failed', []);
      LInf.PrintErrors();
      LPass := False;
    end
    else
    begin
      TConsole.PrintLn('  OK: Model loaded', []);

      LSes := TSession.Create();
      try
        LSes.SetInference(LInf);
        LSes.SetSystemPrompt('You are a helpful assistant.');

        // 2. Verify system prompt is message 0
        if (LInf.Messages.Count <> 1) or
           (LInf.Messages[0].Role <> CRoleSystem) then
        begin
          TConsole.PrintLn('  FAIL: System prompt not at index 0 (count=%d)',
            [LInf.Messages.Count]);
          LPass := False;
        end
        else
          TConsole.PrintLn('  OK: System prompt set', []);

        // 3. Ask a simple question
        TConsole.PrintLn('  Generating reply...', []);
        LReply := LSes.Ask('What is 2+2? Answer in one word.', 64);
        if LReply = '' then
        begin
          TConsole.PrintLn('  FAIL: Ask returned empty', []);
          LPass := False;
        end
        else
          TConsole.PrintLn('  OK: Ask returned %d chars', [Length(LReply)]);

        // 4. Verify history: system + user + assistant = 3 messages
        if LInf.Messages.Count <> 3 then
        begin
          TConsole.PrintLn('  FAIL: History count=%d, expected 3',
            [LInf.Messages.Count]);
          LPass := False;
        end
        else
          TConsole.PrintLn('  OK: History has 3 messages', []);

        // 5. SaveHistory / LoadHistory round-trip
        LHistFile := TGemma4.ResPath(CResDatabase + PathDelim +
          'test_session_hist.json');
        if not LSes.SaveHistory(LHistFile) then
        begin
          TConsole.PrintLn('  FAIL: SaveHistory returned False', []);
          LPass := False;
        end
        else
        begin
          TConsole.PrintLn('  OK: SaveHistory', []);
          LSes.ClearHistory();
          if LInf.Messages.Count <> 1 then
          begin
            TConsole.PrintLn('  FAIL: ClearHistory count=%d, expected 1',
              [LInf.Messages.Count]);
            LPass := False;
          end
          else
            TConsole.PrintLn('  OK: ClearHistory (system prompt kept)', []);

          if not LSes.LoadHistory(LHistFile) then
          begin
            TConsole.PrintLn('  FAIL: LoadHistory returned False', []);
            LPass := False;
          end
          else
          begin
            if LInf.Messages.Count <> 3 then
            begin
              TConsole.PrintLn('  FAIL: After LoadHistory count=%d, expected 3',
                [LInf.Messages.Count]);
              LPass := False;
            end
            else
              TConsole.PrintLn('  OK: LoadHistory restored 3 messages', []);
          end;

          if TFile.Exists(LHistFile) then
            TFile.Delete(LHistFile);
        end;

        // 6. CompactHistory (dumb drop, SummarizeOnEvict=False)
        LSes.SummarizeOnEvict := False;
        // Add extra turns to compact
        LSes.Ask('What color is the sky?', 64);
        LSes.Ask('What is the capital of France?', 64);
        // Should have system + 3*(user+assistant) = 7
        TConsole.PrintLn('  History before compact: %d messages',
          [LInf.Messages.Count]);
        LRemoved := LSes.CompactHistory(1); // keep only last exchange
        TConsole.PrintLn('  CompactHistory removed %d, remaining %d',
          [LRemoved, LInf.Messages.Count]);
        if LRemoved <= 0 then
        begin
          TConsole.PrintLn('  FAIL: CompactHistory removed nothing', []);
          LPass := False;
        end
        else
          TConsole.PrintLn('  OK: CompactHistory dropped old turns', []);

        // 7. Test with Memory + SummarizeOnEvict
        LDbPath := TGemma4.ResPath(CResDatabase + PathDelim +
          'test_session_mem.db');
        if TFile.Exists(LDbPath) then
          TFile.Delete(LDbPath);

        LMem := TMemory.Create();
        try
          if LMem.OpenSession(LDbPath) then
          begin
            LSes.SetMemory(LMem);
            LSes.SummarizeOnEvict := True;
            LSes.ClearHistory();

            // Build up history
            LSes.Ask('My favorite color is blue.', 64);
            LSes.Ask('I have two cats named Luna and Mochi.', 64);
            LSes.Ask('What day is it?', 64);

            TConsole.PrintLn('  History before summarize: %d messages',
              [LInf.Messages.Count]);

            LRemoved := LSes.CompactHistory(1);
            TConsole.PrintLn('  SummarizeAndEvict removed %d, remaining %d',
              [LRemoved, LInf.Messages.Count]);

            LReply := LMem.GetSummary();
            if LReply <> '' then
              TConsole.PrintLn('  OK: Summary stored (%d chars): %s',
                [Length(LReply), Copy(LReply, 1, 120)])
            else
            begin
              TConsole.PrintLn('  FAIL: Summary is empty after eviction', []);
              LPass := False;
            end;

            LSes.SetMemory(nil);
            LMem.CloseSession();
          end
          else
          begin
            TConsole.PrintLn('  FAIL: Memory OpenSession failed', []);
            LPass := False;
          end;
        finally
          LMem.Free();
        end;

        if TFile.Exists(LDbPath) then
          TFile.Delete(LDbPath);
      finally
        LSes.Free();
      end;
    end;
  finally
    LInf.Free();
  end;

  if LPass then
    TConsole.PrintLn('Test_Session: PASS', [])
  else
    TConsole.PrintLn('Test_Session: FAIL', []);
  TConsole.PrintLn('', []);
end;

type
  { TScriptedChat }
  // Test subclass that feeds canned inputs and captures outputs
  TScriptedChat = class(TChat)
  private
    FInputQueue: TArray<string>;
    FInputIdx: Integer;
    FOutputLog: string;
    FTokenLog: string;
  protected
    function DoGetInput(): string; override;
    procedure DoOutput(const AText: string); override;
    procedure DoToken(const AToken: string); override;
    function DoCancel(): Boolean; override;
  public
    constructor Create(); override;
  end;

{ TScriptedChat }

constructor TScriptedChat.Create();
begin
  inherited Create();
  FInputIdx := 0;
  FOutputLog := '';
  FTokenLog := '';
end;

function TScriptedChat.DoGetInput(): string;
begin
  if FInputIdx <= High(FInputQueue) then
  begin
    Result := FInputQueue[FInputIdx];
    Inc(FInputIdx);
  end
  else
    Result := '/quit';
end;

procedure TScriptedChat.DoOutput(const AText: string);
begin
  FOutputLog := FOutputLog + AText + sLineBreak;
end;

procedure TScriptedChat.DoToken(const AToken: string);
begin
  FTokenLog := FTokenLog + AToken;
end;

function TScriptedChat.DoCancel(): Boolean;
begin
  Result := False;
end;

procedure Test_Chat();
var
  LChat: TScriptedChat;
  LPass: Boolean;
begin
  TConsole.PrintLn('--- Test_Chat ---', []);
  LPass := True;

  LChat := TScriptedChat.Create();
  try
    LChat.ModelPath := CVpkOutputFile;
    LChat.UseGPU := True;
    LChat.EnableThinking := False;
    LChat.ShowThinking := False;
    LChat.MaxTokens := 128;

    // Script: ask one question, run /help, run /stats, then /quit
    LChat.FInputQueue := TArray<string>.Create(
      'What is 2+2?',
      '/help',
      '/stats',
      '/quit'
    );

    LChat.Run();

    // Verify we got token output from the generation
    if LChat.FTokenLog = '' then
    begin
      TConsole.PrintLn('  FAIL: No tokens generated', []);
      LPass := False;
    end
    else
      TConsole.PrintLn('  Tokens received: %d chars', [Length(LChat.FTokenLog)]);

    // Verify /help produced output containing 'Available commands'
    if not LChat.FOutputLog.Contains('Available commands') then
    begin
      TConsole.PrintLn('  FAIL: /help did not produce expected output', []);
      LPass := False;
    end
    else
      TConsole.PrintLn('  /help output: OK', []);

    // Verify /stats produced output containing 'Generation'
    if not LChat.FOutputLog.Contains('Generation') then
    begin
      TConsole.PrintLn('  FAIL: /stats did not produce expected output', []);
      LPass := False;
    end
    else
      TConsole.PrintLn('  /stats output: OK', []);

  finally
    LChat.Free();
  end;

  if LPass then
    TConsole.PrintLn('Test_Chat: PASS', [])
  else
    TConsole.PrintLn('Test_Chat: FAIL', []);
  TConsole.PrintLn('', []);
end;

procedure Test_KVState();
var
  LInf: TInference;
  LStatePath: string;
  LPass: Boolean;
  LFirstResponse: string;
  LFollowUp: string;
  LMsg: TChatMessage;
  LI: Integer;
  LSavedMessages: TArray<TChatMessage>;
begin
  TConsole.PrintLn('--- Test_KVState ---', []);
  LPass := True;
  LStatePath := 'gemma4_test.g4kv';

  LInf := TInference.Create();
  try
    LInf.EnableThinking := False;

    if not LInf.LoadModel(CVpkOutputFile) then
    begin
      TConsole.PrintLn('  FAIL: LoadModel failed', []);
      LInf.Free();
      Exit;
    end;

    // Step 1: generate a response that establishes a fact
    LInf.AddMessage('user',
      'My secret code word is BANANA. Remember it. What is 2+2?');
    LInf.Generate(128);
    LFirstResponse := LInf.ResponseText;
    TConsole.PrintLn('  First response: %s',
      [Copy(LFirstResponse, 1, 80)]);

    // Step 2: save KV state + snapshot messages
    if not LInf.SaveKVState(LStatePath) then
    begin
      TConsole.PrintLn('  FAIL: SaveKVState failed', []);
      LPass := False;
    end
    else
    begin
      TConsole.PrintLn('  SaveKVState: OK (%d bytes)',
        [TFile.GetSize(LStatePath)]);

      // Snapshot current messages before reset
      SetLength(LSavedMessages, LInf.Messages.Count);
      for LI := 0 to LInf.Messages.Count - 1 do
        LSavedMessages[LI] := LInf.Messages[LI];

      // Step 3: full reset -- KV cache zeroed, messages cleared
      LInf.ResetConversation();
      LInf.ClearMessages();

      // Step 4: restore KV state + messages
      if not LInf.LoadKVState(LStatePath) then
      begin
        TConsole.PrintLn('  FAIL: LoadKVState failed', []);
        LPass := False;
      end
      else
      begin
        TConsole.PrintLn('  LoadKVState: OK', []);

        // Restore messages
        for LI := 0 to High(LSavedMessages) do
          LInf.Messages.Add(LSavedMessages[LI]);

        // Add the first assistant response back
        LMsg := Default(TChatMessage);
        LMsg.Role := 'assistant';
        LMsg.Content := LFirstResponse;
        LInf.Messages.Add(LMsg);

        // Step 5: ask a follow-up that requires the KV cache
        LInf.AddMessage('user', 'What was my secret code word?');
        LInf.Generate(64);
        LFollowUp := LInf.ResponseText;
        TConsole.PrintLn('  Follow-up: %s',
          [Copy(LFollowUp, 1, 80)]);

        // Verify the model remembers BANANA
        if LFollowUp.ToUpper().Contains('BANANA') then
          TConsole.PrintLn(
            '  KV restore verified: model remembers BANANA', [])
        else
        begin
          TConsole.PrintLn(
            '  FAIL: model did not recall BANANA', []);
          LPass := False;
        end;
      end;
    end;

    // Clean up temp file
    if TFile.Exists(LStatePath) then
      TFile.Delete(LStatePath);
  finally
    LInf.Free();
  end;

  if LPass then
    TConsole.PrintLn('Test_KVState: PASS', [])
  else
    TConsole.PrintLn('Test_KVState: FAIL', []);
  TConsole.PrintLn('', []);
end;

end.
