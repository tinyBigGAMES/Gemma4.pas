{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information

 -------------------------------------------------------------------------------

  Gemma4.Memory - SQLite conversation archive with FTS5 + HNSW hybrid retrieval

  Persistent long-term store for conversation turns, facts and ingested
  documents:
    - FireDAC/SQLite archive of turns (role, text, metadata, embeddings)
    - Hybrid retrieval: FTS5 keyword match fused with HNSW vector search
    - Fact storage, document chunking/ingest, dedup and maintenance
    - Summary slot (SetSummary/GetSummary) powering infinite-memory eviction
    - Embeddings supplied by an attached TEmbeddings encoder

  Dependencies: System.SysUtils, System.Classes, System.DateUtils,
    System.Generics.Collections/Defaults, System.Hash, Data.DB, FireDAC.*,
    StdApp.Base, Gemma4.Common, Gemma4.HNSW, Gemma4.Embeddings
===============================================================================}

unit Gemma4.Memory;

{$I StdApp.Defines.inc}

interface

uses
  System.SysUtils,
  System.Classes,
  System.DateUtils,
  System.Generics.Collections,
  System.Generics.Defaults,
  System.Hash,
  Data.DB,
  FireDAC.Stan.Intf,
  FireDAC.Stan.Option,
  FireDAC.Stan.Error,
  FireDAC.Stan.Def,
  FireDAC.Stan.Async,
  FireDAC.DApt,
  FireDAC.Phys,
  FireDAC.Phys.SQLite,
  FireDAC.Phys.SQLiteWrapper,
  FireDAC.Phys.SQLiteWrapper.Stat,
  FireDAC.Comp.Client,
  FireDAC.VCLUI.Wait,
  StdApp.Base,
  Gemma4.Common,
  Gemma4.HNSW,
  Gemma4.Embeddings;

const
  { Error codes }
  ERR_MEM_NOT_OPEN            = 'MEM001';
  ERR_MEM_DBPATH_EMPTY        = 'MEM002';
  ERR_MEM_EMBEDDER_NIL        = 'MEM003';
  ERR_MEM_EMBEDDER_NOT_LOADED = 'MEM004';
  ERR_MEM_EMBEDDER_DETACHED   = 'MEM005';
  ERR_MEM_NO_EMBEDDER         = 'MEM006';
  ERR_MEM_EMBEDDING_MISMATCH  = 'MEM007';
  ERR_MEM_WHERE_EMPTY         = 'MEM008';
  ERR_MEM_CHUNK_INVALID       = 'MEM009';
  ERR_MEM_OVERLAP_INVALID     = 'MEM010';
  ERR_MEM_OPEN                = 'MEM011';
  ERR_MEM_EMBED               = 'MEM012';
  ERR_MEM_FTS5                = 'MEM013';
  ERR_MEM_SNAPSHOT            = 'MEM014';

  { Archive roles }
  crFact  = 'fact';
  crChunk = 'chunk';

type
  { TMemoryTurn }
  TMemoryTurn = record
    TurnId      : Int64;
    TurnIndex   : Integer;
    Role        : string;
    Text        : string;
    TokenCount  : Integer;
    CreatedAt   : Int64;
    Score       : Single;
    CosineScore : Single;
    Pinned      : Boolean;
  end;

  { TMemory }
  TMemory = class(TBaseObject)
  private
    FLink: TFDPhysSQLiteDriverLink;
    FConn: TFDConnection;
    FDbPath: string;
    FNextTurnIndex: Integer;
    FEmbedder: TEmbeddings;
    FEmbedderDim: Integer;
    FIndex: THNSWIndex;
    FMinTurnTokens: Integer;
    procedure CreateSchemaIfNeeded();
    procedure MigrateSchema();
    procedure LoadNextTurnIndex();
    function SanitizeFTSQuery(const AQuery: string): string;
    function ReadTurnFromQuery(const AQuery: TFDQuery): TMemoryTurn;
    function SingleArrayToBytes(const AVec: TArray<Single>): TBytes;
    function BytesToSingleArray(const ABytes: TBytes): TArray<Single>;
    function CosineDot(const AVecA, AVecB: TArray<Single>): Single;
    function NormalizeText(const AText: string): string;
    function ComputeContentHash(const AText: string): TBytes;
    procedure LoadHNSWIndex();
    procedure PersistHNSWIndex();
    procedure RebuildHNSWIndex();
  public
    constructor Create(); override;
    destructor Destroy(); override;
    function OpenSession(const ADbPath: string): Boolean;
    procedure CloseSession();
    function IsOpen(): Boolean;
    procedure SetMeta(const AKey: string; const AValue: string);
    function GetMeta(const AKey: string): string;

    // Conversation summary convenience (session_meta backed)
    procedure SetSummary(const AText: string);
    function GetSummary(): string;

    function AttachEmbeddings(const AEmbedder: TEmbeddings): Boolean;
    procedure DetachEmbeddings();
    function HasEmbedder(): Boolean;
    property MinTurnTokens: Integer read FMinTurnTokens write FMinTurnTokens;
    function AppendTurn(const ARole: string; const AText: string;
      const ATokenCount: Integer): Int64;
    function GetTurn(const ATurnId: Int64): TMemoryTurn;
    function GetRecentTurns(const ACount: Integer): TArray<TMemoryTurn>;
    function GetTurnCount(): Integer;
    function GetPinnedTurns(const AMax: Integer): TArray<TMemoryTurn>;
    function SearchFTS5(const AQuery: string;
      const ATopK: Integer): TArray<TMemoryTurn>;
    function SearchVector(const AQuery: string;
      const ATopK: Integer): TArray<TMemoryTurn>;
    function AddFact(const AText: string;
      const APinned: Boolean = True): Int64;
    procedure PurgeTurn(const ATurnId: Int64);
    procedure PurgeAll();
    procedure PurgeDatabase();
    procedure PurgeWhere(const AWhereClause: string);
    function AddDocument(const ASource, ATitle, AText: string;
      const AChunkTokens, AOverlapTokens: Integer;
      const APinned: Boolean = False): Int64;
    procedure PurgeDocument(const ADocumentId: Int64);
    function GetDbPath(): string;
    function SnapshotTo(const APath: string): Boolean;
    property Embedder: TEmbeddings read FEmbedder;
  end;

implementation

uses
  System.IOUtils,
  StdApp.Utils,
  StdApp.Resources;

const
  CMemFuzzyDupThreshold = 0.97;
  CMemFuzzyDupScanLimit = 50;
  CMemSummaryKey = 'conversation_summary';

  { CMemDDLTurns }
  CMemDDLTurns =
    'CREATE TABLE IF NOT EXISTS turns (' + sLineBreak +
    '  turn_id     INTEGER PRIMARY KEY AUTOINCREMENT,' + sLineBreak +
    '  turn_index  INTEGER NOT NULL,' + sLineBreak +
    '  role        TEXT    NOT NULL,' + sLineBreak +
    '  text        TEXT    NOT NULL,' + sLineBreak +
    '  token_count INTEGER NOT NULL,' + sLineBreak +
    '  created_at  INTEGER NOT NULL' + sLineBreak +
    ')';

  { CMemDDLTurnsIndex }
  CMemDDLTurnsIndex =
    'CREATE INDEX IF NOT EXISTS idx_turns_index ON turns(turn_index)';

  { CMemDDLMeta }
  CMemDDLMeta =
    'CREATE TABLE IF NOT EXISTS session_meta (' + sLineBreak +
    '  key   TEXT PRIMARY KEY,' + sLineBreak +
    '  value TEXT' + sLineBreak +
    ')';

  { CMemDDLFTS }
  CMemDDLFTS =
    'CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(' + sLineBreak +
    '  text,' + sLineBreak +
    '  content=''turns'',' + sLineBreak +
    '  content_rowid=''turn_id''' + sLineBreak +
    ')';

  { CMemDDLTrigInsert }
  CMemDDLTrigInsert =
    'CREATE TRIGGER IF NOT EXISTS turns_ai AFTER INSERT ON turns BEGIN' + sLineBreak +
    '  INSERT INTO turns_fts(rowid, text) VALUES (new.turn_id, new.text);' + sLineBreak +
    'END';

  { CMemDDLTrigDelete }
  CMemDDLTrigDelete =
    'CREATE TRIGGER IF NOT EXISTS turns_ad AFTER DELETE ON turns BEGIN' + sLineBreak +
    '  INSERT INTO turns_fts(turns_fts, rowid, text)' + sLineBreak +
    '    VALUES(''delete'', old.turn_id, old.text);' + sLineBreak +
    'END';

  { CMemDDLTrigUpdate }
  CMemDDLTrigUpdate =
    'CREATE TRIGGER IF NOT EXISTS turns_au AFTER UPDATE ON turns BEGIN' + sLineBreak +
    '  INSERT INTO turns_fts(turns_fts, rowid, text)' + sLineBreak +
    '    VALUES(''delete'', old.turn_id, old.text);' + sLineBreak +
    '  INSERT INTO turns_fts(rowid, text) VALUES (new.turn_id, new.text);' + sLineBreak +
    'END';

  { CMemDDLDocuments }
  CMemDDLDocuments =
    'CREATE TABLE IF NOT EXISTS documents (' + sLineBreak +
    '  id          INTEGER PRIMARY KEY AUTOINCREMENT,' + sLineBreak +
    '  source      TEXT    NOT NULL,' + sLineBreak +
    '  title       TEXT    NOT NULL,' + sLineBreak +
    '  ingested_at INTEGER NOT NULL,' + sLineBreak +
    '  pinned      INTEGER DEFAULT 0' + sLineBreak +
    ')';

  { CMemDDLTrigDocCascade }
  CMemDDLTrigDocCascade =
    'CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN' + sLineBreak +
    '  DELETE FROM turns WHERE document_id = old.id;' + sLineBreak +
    'END';

  { CMemDDLHNSWData }
  CMemDDLHNSWData =
    'CREATE TABLE IF NOT EXISTS hnsw_data (' +
    '  key  TEXT PRIMARY KEY,' +
    '  data BLOB' +
    ')';

  { CMemFTSOperatorSet }
  CMemFTSOperatorSet: TSysCharSet = ['"', '*', '(', ')', ':', '-', '+', '^',
    '?', '!', ',', '.', ';', '[', ']', '{', '}', '<', '>', '@', '#', '$',
    '%', '&', '=', '~', '\', '/', '|', ''''];

{ TMemory }
constructor TMemory.Create();
begin
  inherited Create();
  FLink := nil;
  FConn := nil;
  FDbPath := '';
  FNextTurnIndex := 0;
  FEmbedder := nil;
  FEmbedderDim := 0;
  FIndex := nil;
  FMinTurnTokens := 2;
end;

destructor TMemory.Destroy();
begin
  FreeAndNil(FIndex);
  CloseSession();
  inherited Destroy();
end;

function TMemory.IsOpen(): Boolean;
begin
  Result := (FConn <> nil) and FConn.Connected;
end;

function TMemory.GetDbPath(): string;
begin
  Result := FDbPath;
end;

function TMemory.OpenSession(const ADbPath: string): Boolean;
begin
  Result := False;

  if ADbPath.IsEmpty() then
  begin
    FErrors.Add(esError, ERR_MEM_DBPATH_EMPTY, RSMemDbPathEmpty);
    Exit;
  end;

  if IsOpen() then
    CloseSession();

  FDbPath := ADbPath;
  TUtils.CreateDirInPath(FDbPath);

  try
    FLink := TFDPhysSQLiteDriverLink.Create(nil);
    FLink.EngineLinkage := slStatic;

    FConn := TFDConnection.Create(nil);
    FConn.DriverName := 'SQLite';
    FConn.Params.Values['Database'] := ADbPath;
    FConn.LoginPrompt := False;
    FConn.Open();

    CreateSchemaIfNeeded();
    LoadNextTurnIndex();

    Result := True;
  except
    on E: Exception do
    begin
      CloseSession();
      FErrors.Add(esError, ERR_MEM_OPEN, RSMemOpenFailed,
        [ADbPath, E.Message]);
    end;
  end;
end;

procedure TMemory.CloseSession();
begin
  PersistHNSWIndex();

  if FConn <> nil then
  begin
    if FConn.Connected then
      FConn.Close();
    FConn.Free();
    FConn := nil;
  end;
  if FLink <> nil then
  begin
    FLink.Free();
    FLink := nil;
  end;
  FNextTurnIndex := 0;
end;

function TMemory.SnapshotTo(const APath: string): Boolean;
var
  LPath: string;
begin
  Result := False;

  if not IsOpen() then
  begin
    FErrors.Add(esError, ERR_MEM_NOT_OPEN, RSMemSessionNotOpen);
    Exit;
  end;

  if APath.IsEmpty() then
  begin
    FErrors.Add(esError, ERR_MEM_DBPATH_EMPTY, RSMemDbPathEmpty);
    Exit;
  end;

  LPath := TPath.ChangeExtension(APath, CExtDb);
  TUtils.CreateDirInPath(LPath);

  try
    CloseSession();
    TFile.Copy(FDbPath, LPath, True);
  except
    on E: Exception do
    begin
      OpenSession(FDbPath);
      FErrors.Add(esError, ERR_MEM_SNAPSHOT,
        Format(RSMemSnapshotFailed, [LPath, E.Message]));
      Exit;
    end;
  end;

  if not OpenSession(FDbPath) then
    Exit;

  Result := True;
end;

procedure TMemory.CreateSchemaIfNeeded();
begin
  FConn.ExecSQL(CMemDDLTurns);
  FConn.ExecSQL(CMemDDLTurnsIndex);
  FConn.ExecSQL(CMemDDLMeta);
  FConn.ExecSQL(CMemDDLFTS);
  FConn.ExecSQL(CMemDDLTrigInsert);
  FConn.ExecSQL(CMemDDLTrigDelete);
  FConn.ExecSQL(CMemDDLTrigUpdate);
  FConn.ExecSQL(CMemDDLDocuments);
  FConn.ExecSQL(CMemDDLTrigDocCascade);
  FConn.ExecSQL(CMemDDLHNSWData);
  MigrateSchema();
end;

procedure TMemory.MigrateSchema();
var
  LQuery: TFDQuery;
  LHasEmbedding: Boolean;
  LHasContentHash: Boolean;
  LHasPinned: Boolean;
  LHasDocumentId: Boolean;
  LColName: string;
begin
  LHasEmbedding := False;
  LHasContentHash := False;
  LHasPinned := False;
  LHasDocumentId := False;
  LQuery := TFDQuery.Create(nil);
  try
    LQuery.Connection := FConn;
    LQuery.SQL.Text := 'PRAGMA table_info(turns)';
    LQuery.Open();
    while not LQuery.Eof do
    begin
      LColName := LQuery.FieldByName('name').AsString;
      if SameText(LColName, 'embedding') then
        LHasEmbedding := True
      else if SameText(LColName, 'content_hash') then
        LHasContentHash := True
      else if SameText(LColName, 'pinned') then
        LHasPinned := True
      else if SameText(LColName, 'document_id') then
        LHasDocumentId := True;
      LQuery.Next();
    end;
    LQuery.Close();
  finally
    LQuery.Free();
  end;

  if not LHasEmbedding then
    FConn.ExecSQL('ALTER TABLE turns ADD COLUMN embedding BLOB');
  if not LHasContentHash then
    FConn.ExecSQL('ALTER TABLE turns ADD COLUMN content_hash BLOB');
  if not LHasPinned then
    FConn.ExecSQL('ALTER TABLE turns ADD COLUMN pinned INTEGER DEFAULT 0');
  if not LHasDocumentId then
    FConn.ExecSQL('ALTER TABLE turns ADD COLUMN document_id INTEGER');
end;

procedure TMemory.LoadNextTurnIndex();
var
  LQuery: TFDQuery;
begin
  LQuery := TFDQuery.Create(nil);
  try
    LQuery.Connection := FConn;
    LQuery.SQL.Text := 'SELECT COALESCE(MAX(turn_index), -1) + 1 FROM turns';
    LQuery.Open();
    FNextTurnIndex := LQuery.Fields[0].AsInteger;
    LQuery.Close();
  finally
    LQuery.Free();
  end;
end;

procedure TMemory.SetMeta(const AKey: string; const AValue: string);
var
  LQuery: TFDQuery;
begin
  if not IsOpen() then
  begin
    FErrors.Add(esError, ERR_MEM_NOT_OPEN, RSMemSessionNotOpen);
    Exit;
  end;

  LQuery := TFDQuery.Create(nil);
  try
    LQuery.Connection := FConn;
    LQuery.SQL.Text :=
      'INSERT INTO session_meta(key, value) VALUES (:k, :v) ' +
      'ON CONFLICT(key) DO UPDATE SET value = excluded.value';
    LQuery.ParamByName('k').AsString := AKey;
    LQuery.ParamByName('v').AsString := AValue;
    LQuery.ExecSQL();
  finally
    LQuery.Free();
  end;
end;

function TMemory.GetMeta(const AKey: string): string;
var
  LQuery: TFDQuery;
begin
  Result := '';
  if not IsOpen() then
  begin
    FErrors.Add(esError, ERR_MEM_NOT_OPEN, RSMemSessionNotOpen);
    Exit;
  end;

  LQuery := TFDQuery.Create(nil);
  try
    LQuery.Connection := FConn;
    LQuery.SQL.Text := 'SELECT value FROM session_meta WHERE key = :k';
    LQuery.ParamByName('k').AsString := AKey;
    LQuery.Open();
    if not LQuery.Eof then
      Result := LQuery.Fields[0].AsString;
    LQuery.Close();
  finally
    LQuery.Free();
  end;
end;

procedure TMemory.SetSummary(const AText: string);
begin
  SetMeta(CMemSummaryKey, AText);
end;

function TMemory.GetSummary(): string;
begin
  Result := GetMeta(CMemSummaryKey);
end;

function TMemory.AttachEmbeddings(const AEmbedder: TEmbeddings): Boolean;
var
  LConfig: THNSWConfig;
begin
  Result := False;

  if AEmbedder = nil then
  begin
    FErrors.Add(esError, ERR_MEM_EMBEDDER_NIL, RSMemEmbedderNil);
    Exit;
  end;

  if not AEmbedder.IsLoaded() then
  begin
    FErrors.Add(esError, ERR_MEM_EMBEDDER_NOT_LOADED, RSMemEmbedderNotLoaded);
    Exit;
  end;

  FEmbedder := AEmbedder;
  FEmbedderDim := AEmbedder.EmbeddingDim();

  FreeAndNil(FIndex);
  LConfig := Default(THNSWConfig);
  LConfig.Dim := FEmbedderDim;
  LConfig.M := 16;
  LConfig.EfConstruction := 200;
  LConfig.EfSearch := 50;
  FIndex := THNSWIndex.Create();
  FIndex.Init(LConfig);
  if IsOpen() then
    LoadHNSWIndex();

  Result := True;
end;

function TMemory.HasEmbedder(): Boolean;
begin
  Result := (FEmbedder <> nil) and FEmbedder.IsLoaded();
end;

procedure TMemory.DetachEmbeddings();
begin
  PersistHNSWIndex();
  FreeAndNil(FIndex);
  FEmbedder := nil;
  FEmbedderDim := 0;
end;

function TMemory.AppendTurn(const ARole: string; const AText: string;
  const ATokenCount: Integer): Int64;
var
  LQuery: TFDQuery;
  LNowUnix: Int64;
  LVec: TArray<Single>;
  LBytes: TBytes;
  LStream: TBytesStream;
  LHash: TBytes;
  LHashStream: TBytesStream;
  LDupQuery: TFDQuery;
  LFuzzyQuery: TFDQuery;
  LFuzzyStream: TStream;
  LFuzzyBlob: TBytes;
  LFuzzyRowVec: TArray<Single>;
  LFuzzyHit: Boolean;
  LFuzzyHitId: Int64;
begin
  Result := 0;
  if not IsOpen() then
  begin
    FErrors.Add(esError, ERR_MEM_NOT_OPEN, RSMemSessionNotOpen);
    Exit;
  end;

  if (FEmbedder <> nil) and (not FEmbedder.IsLoaded()) then
  begin
    FErrors.Add(esError, ERR_MEM_EMBEDDER_DETACHED, RSMemEmbedderDetached);
    Exit;
  end;

  if ATokenCount < FMinTurnTokens then
    Exit;

  // Exact-duplicate suppression via SHA-256 content hash
  LHash := ComputeContentHash(AText);
  LDupQuery := TFDQuery.Create(nil);
  try
    LDupQuery.Connection := FConn;
    LDupQuery.SQL.Text :=
      'SELECT turn_id FROM turns WHERE content_hash = :h LIMIT 1';
    LHashStream := TBytesStream.Create(LHash);
    try
      LDupQuery.ParamByName('h').LoadFromStream(LHashStream, ftBlob);
    finally
      LHashStream.Free();
    end;
    LDupQuery.Open();
    if not LDupQuery.Eof then
    begin
      Result := LDupQuery.Fields[0].AsLargeInt;
      LDupQuery.Close();
      Exit;
    end;
    LDupQuery.Close();
  finally
    LDupQuery.Free();
  end;

  LNowUnix := DateTimeToUnix(Now(), False);

  LQuery := TFDQuery.Create(nil);
  try
    LQuery.Connection := FConn;
    LQuery.SQL.Text :=
      'INSERT INTO turns ' +
      '  (turn_index, role, text, token_count, created_at, embedding, ' +
      '   content_hash, pinned) ' +
      'VALUES (:idx, :role, :txt, :tok, :ts, :emb, :hash, :pin)';
    LQuery.ParamByName('idx').AsInteger := FNextTurnIndex;
    LQuery.ParamByName('role').AsString := ARole;
    LQuery.ParamByName('txt').AsString  := AText;
    LQuery.ParamByName('tok').AsInteger := ATokenCount;
    LQuery.ParamByName('ts').AsLargeInt := LNowUnix;

    LHashStream := TBytesStream.Create(LHash);
    try
      LQuery.ParamByName('hash').LoadFromStream(LHashStream, ftBlob);
    finally
      LHashStream.Free();
    end;

    LQuery.ParamByName('pin').AsInteger := 0;

    if FEmbedder <> nil then
    begin
      LVec := FEmbedder.EmbedDocument(AText);

      if Length(LVec) = 0 then
      begin
        FErrors.Add(esError, ERR_MEM_EMBED, RSMemEmbedFailed);
        Exit;
      end;

      // Fuzzy semantic dedup
      LFuzzyHit := False;
      LFuzzyHitId := 0;
      LFuzzyQuery := TFDQuery.Create(nil);
      try
        LFuzzyQuery.Connection := FConn;
        LFuzzyQuery.SQL.Text :=
          'SELECT turn_id, embedding FROM turns ' +
          'WHERE embedding IS NOT NULL ' +
          'ORDER BY turn_id DESC LIMIT :lim';
        LFuzzyQuery.ParamByName('lim').AsInteger := CMemFuzzyDupScanLimit;
        LFuzzyQuery.Open();
        while (not LFuzzyQuery.Eof) and (not LFuzzyHit) do
        begin
          LFuzzyStream := LFuzzyQuery.CreateBlobStream(
            LFuzzyQuery.FieldByName('embedding'), bmRead);
          try
            SetLength(LFuzzyBlob, LFuzzyStream.Size);
            if LFuzzyStream.Size > 0 then
              LFuzzyStream.ReadBuffer(LFuzzyBlob[0], LFuzzyStream.Size);
          finally
            LFuzzyStream.Free();
          end;

          if Length(LFuzzyBlob) = FEmbedderDim * SizeOf(Single) then
          begin
            LFuzzyRowVec := BytesToSingleArray(LFuzzyBlob);
            if CosineDot(LVec, LFuzzyRowVec) >= CMemFuzzyDupThreshold then
            begin
              LFuzzyHit := True;
              LFuzzyHitId :=
                LFuzzyQuery.FieldByName('turn_id').AsLargeInt;
            end;
          end;

          if not LFuzzyHit then
            LFuzzyQuery.Next();
        end;
        LFuzzyQuery.Close();
      finally
        LFuzzyQuery.Free();
      end;

      if LFuzzyHit then
      begin
        Result := LFuzzyHitId;
        Exit;
      end;

      LBytes := SingleArrayToBytes(LVec);
      LStream := TBytesStream.Create(LBytes);
      try
        LQuery.ParamByName('emb').LoadFromStream(LStream, ftBlob);
      finally
        LStream.Free();
      end;
    end
    else
    begin
      LQuery.ParamByName('emb').DataType := ftBlob;
      LQuery.ParamByName('emb').Clear();
    end;

    LQuery.ExecSQL();

    LQuery.SQL.Text := 'SELECT last_insert_rowid()';
    LQuery.Open();
    Result := LQuery.Fields[0].AsLargeInt;
    LQuery.Close();

    if (FIndex <> nil) and (Length(LVec) > 0) then
      FIndex.Insert(Result, LVec);

    Inc(FNextTurnIndex);
  finally
    LQuery.Free();
  end;
end;

function TMemory.ReadTurnFromQuery(
  const AQuery: TFDQuery): TMemoryTurn;
begin
  Result.TurnId      := AQuery.FieldByName('turn_id').AsLargeInt;
  Result.TurnIndex   := AQuery.FieldByName('turn_index').AsInteger;
  Result.Role        := AQuery.FieldByName('role').AsString;
  Result.Text        := AQuery.FieldByName('text').AsString;
  Result.TokenCount  := AQuery.FieldByName('token_count').AsInteger;
  Result.CreatedAt   := AQuery.FieldByName('created_at').AsLargeInt;
  Result.Score       := 0.0;
  Result.CosineScore := 0.0;
  Result.Pinned      := AQuery.FieldByName('pinned').AsInteger <> 0;
end;

function TMemory.SingleArrayToBytes(
  const AVec: TArray<Single>): TBytes;
var
  LLen: Integer;
begin
  LLen := Length(AVec) * SizeOf(Single);
  SetLength(Result, LLen);
  if LLen > 0 then
    Move(AVec[0], Result[0], LLen);
end;

function TMemory.BytesToSingleArray(
  const ABytes: TBytes): TArray<Single>;
var
  LCount: Integer;
begin
  LCount := Length(ABytes) div SizeOf(Single);
  SetLength(Result, LCount);
  if LCount > 0 then
    Move(ABytes[0], Result[0], LCount * SizeOf(Single));
end;

function TMemory.CosineDot(const AVecA, AVecB: TArray<Single>): Single;
var
  LI: Integer;
  LSum: Single;
  LN: Integer;
begin
  LSum := 0.0;
  LN := Length(AVecA);
  if Length(AVecB) < LN then
    LN := Length(AVecB);
  for LI := 0 to LN - 1 do
    LSum := LSum + AVecA[LI] * AVecB[LI];
  Result := LSum;
end;

function TMemory.NormalizeText(const AText: string): string;
var
  LI: Integer;
  LLen: Integer;
  LCh: Char;
  LInSpace: Boolean;
  LBuf: TStringBuilder;
begin
  LBuf := TStringBuilder.Create(Length(AText));
  try
    LInSpace := True;
    LLen := Length(AText);
    for LI := 1 to LLen do
    begin
      LCh := AText[LI];
      if (LCh = ' ') or (LCh = #9) or (LCh = #10) or (LCh = #13) then
      begin
        if not LInSpace then
        begin
          LBuf.Append(' ');
          LInSpace := True;
        end;
      end
      else if ((LCh >= 'A') and (LCh <= 'Z')) or
              ((LCh >= 'a') and (LCh <= 'z')) or
              ((LCh >= '0') and (LCh <= '9')) then
      begin
        LBuf.Append(LowerCase(LCh));
        LInSpace := False;
      end;
    end;
    Result := Trim(LBuf.ToString());
  finally
    LBuf.Free();
  end;
end;

function TMemory.ComputeContentHash(const AText: string): TBytes;
var
  LNorm: string;
  LUtf8: TBytes;
  LHash: THashSHA2;
begin
  LNorm := NormalizeText(AText);
  LUtf8 := TEncoding.UTF8.GetBytes(LNorm);
  LHash := THashSHA2.Create();
  LHash.Update(LUtf8, Length(LUtf8));
  Result := LHash.HashAsBytes;
end;

procedure TMemory.LoadHNSWIndex();
var
  LQuery: TFDQuery;
  LFieldStream: TStream;
  LData: TBytes;
begin
  if (FIndex = nil) or (not IsOpen()) then
    Exit;

  LQuery := TFDQuery.Create(nil);
  try
    LQuery.Connection := FConn;
    LQuery.SQL.Text := 'SELECT data FROM hnsw_data WHERE key = :k';
    LQuery.ParamByName('k').AsString := 'index';
    LQuery.Open();

    if not LQuery.Eof then
    begin
      LFieldStream := LQuery.CreateBlobStream(
        LQuery.FieldByName('data'), bmRead);
      try
        SetLength(LData, LFieldStream.Size);
        if LFieldStream.Size > 0 then
          LFieldStream.ReadBuffer(LData[0], LFieldStream.Size);
      finally
        LFieldStream.Free();
      end;
      LQuery.Close();

      if Length(LData) > 0 then
        FIndex.LoadFromBytes(LData);
    end
    else
    begin
      LQuery.Close();
      RebuildHNSWIndex();
    end;
  finally
    LQuery.Free();
  end;
end;

procedure TMemory.PersistHNSWIndex();
var
  LData: TBytes;
  LQuery: TFDQuery;
  LStream: TBytesStream;
begin
  if (FIndex = nil) or (not IsOpen()) then
    Exit;

  LData := FIndex.SaveToBytes();

  LQuery := TFDQuery.Create(nil);
  try
    LQuery.Connection := FConn;
    LQuery.SQL.Text :=
      'INSERT OR REPLACE INTO hnsw_data (key, data) VALUES (:k, :d)';
    LQuery.ParamByName('k').AsString := 'index';
    LStream := TBytesStream.Create(LData);
    try
      LQuery.ParamByName('d').LoadFromStream(LStream, ftBlob);
    finally
      LStream.Free();
    end;
    LQuery.ExecSQL();
  finally
    LQuery.Free();
  end;
end;

procedure TMemory.RebuildHNSWIndex();
var
  LQuery: TFDQuery;
  LFieldStream: TStream;
  LBlob: TBytes;
  LVec: TArray<Single>;
  LTurnId: Int64;
begin
  if (FIndex = nil) or (not IsOpen()) then
    Exit;

  FIndex.Clear();

  LQuery := TFDQuery.Create(nil);
  try
    LQuery.Connection := FConn;
    LQuery.SQL.Text :=
      'SELECT turn_id, embedding FROM turns WHERE embedding IS NOT NULL';
    LQuery.Open();

    while not LQuery.Eof do
    begin
      LTurnId := LQuery.FieldByName('turn_id').AsLargeInt;

      LFieldStream := LQuery.CreateBlobStream(
        LQuery.FieldByName('embedding'), bmRead);
      try
        SetLength(LBlob, LFieldStream.Size);
        if LFieldStream.Size > 0 then
          LFieldStream.ReadBuffer(LBlob[0], LFieldStream.Size);
      finally
        LFieldStream.Free();
      end;

      LVec := BytesToSingleArray(LBlob);
      if Length(LVec) = FIndex.Config.Dim then
        FIndex.Insert(LTurnId, LVec);

      LQuery.Next();
    end;
    LQuery.Close();
  finally
    LQuery.Free();
  end;
end;

function TMemory.GetTurn(const ATurnId: Int64): TMemoryTurn;
var
  LQuery: TFDQuery;
begin
  Result.TurnId      := 0;
  Result.TurnIndex   := 0;
  Result.Role        := '';
  Result.Text        := '';
  Result.TokenCount  := 0;
  Result.CreatedAt   := 0;
  Result.Score       := 0.0;
  Result.CosineScore := 0.0;
  Result.Pinned      := False;

  if not IsOpen() then
  begin
    FErrors.Add(esError, ERR_MEM_NOT_OPEN, RSMemSessionNotOpen);
    Exit;
  end;

  LQuery := TFDQuery.Create(nil);
  try
    LQuery.Connection := FConn;
    LQuery.SQL.Text :=
      'SELECT turn_id, turn_index, role, text, token_count, created_at, pinned ' +
      'FROM turns WHERE turn_id = :id';
    LQuery.ParamByName('id').AsLargeInt := ATurnId;
    LQuery.Open();
    if not LQuery.Eof then
      Result := ReadTurnFromQuery(LQuery);
    LQuery.Close();
  finally
    LQuery.Free();
  end;
end;

function TMemory.GetTurnCount(): Integer;
var
  LQuery: TFDQuery;
begin
  Result := 0;
  if not IsOpen() then
  begin
    FErrors.Add(esError, ERR_MEM_NOT_OPEN, RSMemSessionNotOpen);
    Exit;
  end;

  LQuery := TFDQuery.Create(nil);
  try
    LQuery.Connection := FConn;
    LQuery.SQL.Text := 'SELECT COUNT(*) FROM turns';
    LQuery.Open();
    Result := LQuery.Fields[0].AsInteger;
    LQuery.Close();
  finally
    LQuery.Free();
  end;
end;

function TMemory.GetRecentTurns(
  const ACount: Integer): TArray<TMemoryTurn>;
var
  LQuery: TFDQuery;
  LList: TList<TMemoryTurn>;
  LI: Integer;
begin
  Result := nil;
  if not IsOpen() then
  begin
    FErrors.Add(esError, ERR_MEM_NOT_OPEN, RSMemSessionNotOpen);
    Exit;
  end;
  if ACount <= 0 then
    Exit;

  LList := TList<TMemoryTurn>.Create();
  LQuery := TFDQuery.Create(nil);
  try
    LQuery.Connection := FConn;
    LQuery.SQL.Text :=
      'SELECT turn_id, turn_index, role, text, token_count, created_at, pinned ' +
      'FROM turns ORDER BY turn_index DESC LIMIT :lim';
    LQuery.ParamByName('lim').AsInteger := ACount;
    LQuery.Open();
    while not LQuery.Eof do
    begin
      LList.Add(ReadTurnFromQuery(LQuery));
      LQuery.Next();
    end;
    LQuery.Close();

    SetLength(Result, LList.Count);
    for LI := 0 to LList.Count - 1 do
      Result[LI] := LList[LList.Count - 1 - LI];
  finally
    LQuery.Free();
    LList.Free();
  end;
end;

function TMemory.GetPinnedTurns(const AMax: Integer): TArray<TMemoryTurn>;
var
  LQuery: TFDQuery;
  LList: TList<TMemoryTurn>;
begin
  Result := nil;
  if not IsOpen() then
  begin
    FErrors.Add(esError, ERR_MEM_NOT_OPEN, RSMemSessionNotOpen);
    Exit;
  end;
  if AMax <= 0 then
    Exit;

  LList := TList<TMemoryTurn>.Create();
  LQuery := TFDQuery.Create(nil);
  try
    LQuery.Connection := FConn;
    LQuery.SQL.Text :=
      'SELECT turn_id, turn_index, role, text, token_count, created_at, pinned ' +
      'FROM turns WHERE pinned = 1 ORDER BY turn_index LIMIT :lim';
    LQuery.ParamByName('lim').AsInteger := AMax;
    LQuery.Open();
    while not LQuery.Eof do
    begin
      LList.Add(ReadTurnFromQuery(LQuery));
      LQuery.Next();
    end;
    LQuery.Close();

    Result := LList.ToArray();
  finally
    LQuery.Free();
    LList.Free();
  end;
end;

function TMemory.SanitizeFTSQuery(const AQuery: string): string;
var
  LCleaned: string;
  LI: Integer;
  LCh: Char;
  LTokens: TArray<string>;
  LOut: TStringBuilder;
  LTok: string;
begin
  SetLength(LCleaned, Length(AQuery));
  for LI := 1 to Length(AQuery) do
  begin
    LCh := AQuery[LI];
    if CharInSet(LCh, CMemFTSOperatorSet) then
      LCleaned[LI] := ' '
    else
      LCleaned[LI] := LCh;
  end;

  LTokens := LCleaned.Split([' ', #9, #10, #13],
    TStringSplitOptions.ExcludeEmpty);

  if Length(LTokens) = 0 then
    Exit('');

  LOut := TStringBuilder.Create();
  try
    for LI := 0 to High(LTokens) do
    begin
      LTok := Trim(LTokens[LI]);
      if LTok = '' then
        Continue;
      if (LTok = 'NOT') or (LTok = 'AND') or (LTok = 'OR') or
         (LTok = 'NEAR') then
        Continue;
      if LOut.Length > 0 then
        LOut.Append(' OR ');
      LOut.Append(LTok);
    end;
    Result := LOut.ToString();
  finally
    LOut.Free();
  end;
end;

function TMemory.SearchFTS5(const AQuery: string;
  const ATopK: Integer): TArray<TMemoryTurn>;
var
  LQuery: TFDQuery;
  LList: TList<TMemoryTurn>;
  LMatch: string;
  LTurn: TMemoryTurn;
begin
  Result := nil;
  if not IsOpen() then
  begin
    FErrors.Add(esError, ERR_MEM_NOT_OPEN, RSMemSessionNotOpen);
    Exit;
  end;
  if ATopK <= 0 then
    Exit;

  LMatch := SanitizeFTSQuery(AQuery);
  if LMatch = '' then
    Exit;

  LList := TList<TMemoryTurn>.Create();
  LQuery := TFDQuery.Create(nil);
  try
    LQuery.Connection := FConn;
    LQuery.SQL.Text :=
      'SELECT turns.turn_id, turns.turn_index, turns.role, turns.text, ' +
      '       turns.token_count, turns.created_at, turns.pinned, ' +
      '       bm25(turns_fts) AS score ' +
      'FROM turns_fts ' +
      'JOIN turns ON turns.turn_id = turns_fts.rowid ' +
      'WHERE turns_fts MATCH :q ' +
      'ORDER BY bm25(turns_fts) ' +
      'LIMIT :lim';
    LQuery.ParamByName('q').AsString    := LMatch;
    LQuery.ParamByName('lim').AsInteger := ATopK;
    try
      LQuery.Open();
      while not LQuery.Eof do
      begin
        LTurn := ReadTurnFromQuery(LQuery);
        LTurn.Score := Single(LQuery.FieldByName('score').AsFloat);
        LList.Add(LTurn);
        LQuery.Next();
      end;
      LQuery.Close();

      Result := LList.ToArray();
    except
      on E: Exception do
      begin
        FErrors.Add(esWarning, ERR_MEM_FTS5,
          Format(RSMemFTS5Failed, [E.Message]));
        Result := nil;
      end;
    end;
  finally
    LQuery.Free();
    LList.Free();
  end;
end;

function TMemory.SearchVector(const AQuery: string;
  const ATopK: Integer): TArray<TMemoryTurn>;
var
  LQueryVec: TArray<Single>;
  LHNSWResults: TArray<THNSWSearchResult>;
  LTurn: TMemoryTurn;
  LI: Integer;
  LCount: Integer;
begin
  Result := nil;

  if not IsOpen() then
  begin
    FErrors.Add(esError, ERR_MEM_NOT_OPEN, RSMemSessionNotOpen);
    Exit;
  end;

  if (FEmbedder = nil) or (not FEmbedder.IsLoaded()) then
  begin
    FErrors.Add(esError, ERR_MEM_NO_EMBEDDER, RSMemNoEmbedder);
    Exit;
  end;

  if FIndex = nil then
  begin
    FErrors.Add(esError, ERR_MEM_NO_EMBEDDER, RSMemNoEmbedder);
    Exit;
  end;

  if ATopK <= 0 then
    Exit;

  LQueryVec := FEmbedder.EmbedQuery(AQuery);
  if Length(LQueryVec) = 0 then
  begin
    FErrors.Add(esError, ERR_MEM_EMBED, RSMemEmbedFailed);
    Exit;
  end;

  LHNSWResults := FIndex.Search(LQueryVec, ATopK);

  LCount := Length(LHNSWResults);
  SetLength(Result, LCount);
  for LI := 0 to LCount - 1 do
  begin
    LTurn := GetTurn(LHNSWResults[LI].NodeId);
    LTurn.CosineScore := 1.0 - LHNSWResults[LI].Distance;
    Result[LI] := LTurn;
  end;
end;

function TMemory.AddFact(const AText: string;
  const APinned: Boolean): Int64;
var
  LQuery: TFDQuery;
  LNowUnix: Int64;
  LVec: TArray<Single>;
  LBytes: TBytes;
  LStream: TBytesStream;
  LHash: TBytes;
  LHashStream: TBytesStream;
  LDupQuery: TFDQuery;
begin
  Result := 0;
  if not IsOpen() then
  begin
    FErrors.Add(esError, ERR_MEM_NOT_OPEN, RSMemSessionNotOpen);
    Exit;
  end;

  if (FEmbedder <> nil) and (not FEmbedder.IsLoaded()) then
  begin
    FErrors.Add(esError, ERR_MEM_EMBEDDER_DETACHED, RSMemEmbedderDetached);
    Exit;
  end;

  // Dedup
  LHash := ComputeContentHash(AText);
  LDupQuery := TFDQuery.Create(nil);
  try
    LDupQuery.Connection := FConn;
    LDupQuery.SQL.Text :=
      'SELECT turn_id FROM turns WHERE content_hash = :h LIMIT 1';
    LHashStream := TBytesStream.Create(LHash);
    try
      LDupQuery.ParamByName('h').LoadFromStream(LHashStream, ftBlob);
    finally
      LHashStream.Free();
    end;
    LDupQuery.Open();
    if not LDupQuery.Eof then
    begin
      Result := LDupQuery.Fields[0].AsLargeInt;
      LDupQuery.Close();
      Exit;
    end;
    LDupQuery.Close();
  finally
    LDupQuery.Free();
  end;

  LNowUnix := DateTimeToUnix(Now(), False);

  LQuery := TFDQuery.Create(nil);
  try
    LQuery.Connection := FConn;
    LQuery.SQL.Text :=
      'INSERT INTO turns ' +
      '  (turn_index, role, text, token_count, created_at, embedding, ' +
      '   content_hash, pinned) ' +
      'VALUES (:idx, :role, :txt, :tok, :ts, :emb, :hash, :pin)';
    LQuery.ParamByName('idx').AsInteger := FNextTurnIndex;
    LQuery.ParamByName('role').AsString := crFact;
    LQuery.ParamByName('txt').AsString  := AText;
    LQuery.ParamByName('tok').AsInteger := 0;
    LQuery.ParamByName('ts').AsLargeInt := LNowUnix;

    LHashStream := TBytesStream.Create(LHash);
    try
      LQuery.ParamByName('hash').LoadFromStream(LHashStream, ftBlob);
    finally
      LHashStream.Free();
    end;

    if APinned then
      LQuery.ParamByName('pin').AsInteger := 1
    else
      LQuery.ParamByName('pin').AsInteger := 0;

    if FEmbedder <> nil then
    begin
      LVec := FEmbedder.EmbedDocument(AText);

      if Length(LVec) = 0 then
      begin
        FErrors.Add(esError, ERR_MEM_EMBED, RSMemEmbedFailed);
        Exit;
      end;

      LBytes := SingleArrayToBytes(LVec);
      LStream := TBytesStream.Create(LBytes);
      try
        LQuery.ParamByName('emb').LoadFromStream(LStream, ftBlob);
      finally
        LStream.Free();
      end;
    end
    else
    begin
      LQuery.ParamByName('emb').DataType := ftBlob;
      LQuery.ParamByName('emb').Clear();
    end;

    LQuery.ExecSQL();

    LQuery.SQL.Text := 'SELECT last_insert_rowid()';
    LQuery.Open();
    Result := LQuery.Fields[0].AsLargeInt;
    LQuery.Close();

    if (FIndex <> nil) and (Length(LVec) > 0) then
      FIndex.Insert(Result, LVec);

    Inc(FNextTurnIndex);
  finally
    LQuery.Free();
  end;
end;

procedure TMemory.PurgeTurn(const ATurnId: Int64);
var
  LQuery: TFDQuery;
begin
  if not IsOpen() then
  begin
    FErrors.Add(esError, ERR_MEM_NOT_OPEN, RSMemSessionNotOpen);
    Exit;
  end;

  LQuery := TFDQuery.Create(nil);
  try
    LQuery.Connection := FConn;
    LQuery.SQL.Text := 'DELETE FROM turns WHERE turn_id = :id';
    LQuery.ParamByName('id').AsLargeInt := ATurnId;
    LQuery.ExecSQL();
  finally
    LQuery.Free();
  end;

  if FIndex <> nil then
    FIndex.Delete(ATurnId);
end;

procedure TMemory.PurgeAll();
begin
  if not IsOpen() then
  begin
    FErrors.Add(esError, ERR_MEM_NOT_OPEN, RSMemSessionNotOpen);
    Exit;
  end;

  FConn.ExecSQL('DELETE FROM turns');
  FNextTurnIndex := 0;

  if FIndex <> nil then
    FIndex.Clear();
end;

procedure TMemory.PurgeDatabase();
begin
  if not IsOpen() then
  begin
    FErrors.Add(esError, ERR_MEM_NOT_OPEN, RSMemSessionNotOpen);
    Exit;
  end;

  FConn.ExecSQL('DELETE FROM turns');
  FConn.ExecSQL('DELETE FROM documents');
  FConn.ExecSQL('DELETE FROM session_meta');
  FConn.ExecSQL('DELETE FROM hnsw_data');
  FConn.ExecSQL('VACUUM');
  FNextTurnIndex := 0;

  if FIndex <> nil then
    FIndex.Clear();
end;

procedure TMemory.PurgeWhere(const AWhereClause: string);
begin
  if not IsOpen() then
  begin
    FErrors.Add(esError, ERR_MEM_NOT_OPEN, RSMemSessionNotOpen);
    Exit;
  end;
  if AWhereClause = '' then
  begin
    FErrors.Add(esError, ERR_MEM_WHERE_EMPTY, RSMemWhereEmpty);
    Exit;
  end;

  FConn.ExecSQL('DELETE FROM turns WHERE ' + AWhereClause);
  LoadNextTurnIndex();
end;

function TMemory.AddDocument(const ASource, ATitle, AText: string;
  const AChunkTokens, AOverlapTokens: Integer;
  const APinned: Boolean): Int64;
var
  LDocQuery: TFDQuery;
  LUpdateQuery: TFDQuery;
  LDocId: Int64;
  LNowUnix: Int64;
  LParagraphs: TArray<string>;
  LChunks: TList<string>;
  LWords: TArray<string>;
  LCurrentWords: TList<string>;
  LParaWords: TArray<string>;
  LChunkText: string;
  LOverlapStart: Integer;
  LTurnId: Int64;
  LI: Integer;
  LJ: Integer;
  LK: Integer;
  LPinInt: Integer;
begin
  Result := 0;
  if not IsOpen() then
  begin
    FErrors.Add(esError, ERR_MEM_NOT_OPEN, RSMemSessionNotOpen);
    Exit;
  end;
  if AChunkTokens <= 0 then
  begin
    FErrors.Add(esError, ERR_MEM_CHUNK_INVALID, RSMemChunkInvalid);
    Exit;
  end;
  if AOverlapTokens >= AChunkTokens then
  begin
    FErrors.Add(esError, ERR_MEM_OVERLAP_INVALID, RSMemOverlapInvalid);
    Exit;
  end;

  LNowUnix := DateTimeToUnix(Now(), False);
  if APinned then
    LPinInt := 1
  else
    LPinInt := 0;

  LDocQuery := TFDQuery.Create(nil);
  try
    LDocQuery.Connection := FConn;
    LDocQuery.SQL.Text :=
      'INSERT INTO documents (source, title, ingested_at, pinned) ' +
      'VALUES (:src, :ttl, :ts, :pin)';
    LDocQuery.ParamByName('src').AsString := ASource;
    LDocQuery.ParamByName('ttl').AsString := ATitle;
    LDocQuery.ParamByName('ts').AsLargeInt := LNowUnix;
    LDocQuery.ParamByName('pin').AsInteger := LPinInt;
    LDocQuery.ExecSQL();

    LDocQuery.SQL.Text := 'SELECT last_insert_rowid()';
    LDocQuery.Open();
    LDocId := LDocQuery.Fields[0].AsLargeInt;
    LDocQuery.Close();
  finally
    LDocQuery.Free();
  end;

  // Paragraph-aware chunking
  LParagraphs := AText.Replace(#13#10, #10).Replace(#13, #10)
    .Split([#10#10], TStringSplitOptions.ExcludeEmpty);

  LChunks := TList<string>.Create();
  LCurrentWords := TList<string>.Create();
  try
    for LI := 0 to High(LParagraphs) do
    begin
      LParaWords := LParagraphs[LI].Split([' ', #9, #10],
        TStringSplitOptions.ExcludeEmpty);

      if (LCurrentWords.Count > 0) and
         (LCurrentWords.Count + Length(LParaWords) > AChunkTokens) then
      begin
        LChunks.Add(string.Join(' ', LCurrentWords.ToArray()));

        if AOverlapTokens > 0 then
        begin
          LOverlapStart := LCurrentWords.Count - AOverlapTokens;
          if LOverlapStart < 0 then
            LOverlapStart := 0;
          LWords := LCurrentWords.ToArray();
          LCurrentWords.Clear();
          for LJ := LOverlapStart to High(LWords) do
            LCurrentWords.Add(LWords[LJ]);
        end
        else
          LCurrentWords.Clear();
      end;

      if Length(LParaWords) > AChunkTokens then
      begin
        for LJ := 0 to High(LParaWords) do
        begin
          LCurrentWords.Add(LParaWords[LJ]);
          if LCurrentWords.Count >= AChunkTokens then
          begin
            LChunks.Add(string.Join(' ', LCurrentWords.ToArray()));
            if AOverlapTokens > 0 then
            begin
              LOverlapStart := LCurrentWords.Count - AOverlapTokens;
              if LOverlapStart < 0 then
                LOverlapStart := 0;
              LWords := LCurrentWords.ToArray();
              LCurrentWords.Clear();
              for LK := LOverlapStart to High(LWords) do
                LCurrentWords.Add(LWords[LK]);
            end
            else
              LCurrentWords.Clear();
          end;
        end;
      end
      else
      begin
        for LJ := 0 to High(LParaWords) do
          LCurrentWords.Add(LParaWords[LJ]);
      end;
    end;

    if LCurrentWords.Count > 0 then
      LChunks.Add(string.Join(' ', LCurrentWords.ToArray()));

    LUpdateQuery := TFDQuery.Create(nil);
    try
      LUpdateQuery.Connection := FConn;
      for LI := 0 to LChunks.Count - 1 do
      begin
        LChunkText := LChunks[LI];
        LWords := LChunkText.Split([' '], TStringSplitOptions.ExcludeEmpty);
        LTurnId := AppendTurn(crChunk, LChunkText, Length(LWords));

        LUpdateQuery.SQL.Text :=
          'UPDATE turns SET document_id = :did, pinned = :pin ' +
          'WHERE turn_id = :tid';
        LUpdateQuery.ParamByName('did').AsLargeInt := LDocId;
        LUpdateQuery.ParamByName('pin').AsInteger := LPinInt;
        LUpdateQuery.ParamByName('tid').AsLargeInt := LTurnId;
        LUpdateQuery.ExecSQL();
      end;
    finally
      LUpdateQuery.Free();
    end;
  finally
    LCurrentWords.Free();
    LChunks.Free();
  end;

  Result := LDocId;
end;

procedure TMemory.PurgeDocument(const ADocumentId: Int64);
var
  LQuery: TFDQuery;
begin
  if not IsOpen() then
  begin
    FErrors.Add(esError, ERR_MEM_NOT_OPEN, RSMemSessionNotOpen);
    Exit;
  end;

  LQuery := TFDQuery.Create(nil);
  try
    LQuery.Connection := FConn;
    LQuery.SQL.Text := 'DELETE FROM documents WHERE id = :did';
    LQuery.ParamByName('did').AsLargeInt := ADocumentId;
    LQuery.ExecSQL();
  finally
    LQuery.Free();
  end;

  LoadNextTurnIndex();
end;

end.
