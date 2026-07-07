{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information

 -------------------------------------------------------------------------------

  Gemma4.Tokenizer - BPE tokenizer for Gemma 4

  Implements a byte-pair encoding (BPE) tokenizer compatible with the
  HuggingFace tokenizer.json format used by Gemma 4. Handles:
    - Vocab loading (token string -> ID mapping)
    - Merge rules (ordered pair merges)
    - SentencePiece-style normalization (space -> unicode block char)
    - Byte fallback for unknown characters
    - Special token handling (BOS, EOS, turn markers)
    - Encode (text -> token IDs) and Decode (token IDs -> text)
    - Chat template formatting

  The tokenizer.json is loaded from a VPK archive or from disk as
  a JSON string. The 32MB file is parsed once at startup.

  Dependencies: StdApp.Base, StdApp.JSON, Gemma4.Types
===============================================================================}

unit Gemma4.Tokenizer;

{$I StdApp.Defines.inc}

interface

uses
  System.SysUtils,
  System.Classes,
  System.Generics.Collections,
  StdApp.Base,
  StdApp.JSON,
  Gemma4.Types;

const
  CTK_ERR_PARSE = 'TK01';
  CTK_ERR_VOCAB = 'TK02';
  CTK_ERR_MERGES = 'TK03';

  // SentencePiece uses this unicode character (U+2581, LOWER ONE EIGHTH BLOCK)
  // in place of space. Delphi strings are UTF-16: this MUST be the single
  // code point #$2581. Writing the UTF-8 byte sequence (#$E2#$96#$81) creates
  // a 3-character string that never matches vocab entries parsed from JSON.
  CSentencePieceSpace = #$2581; // "▁" as one UTF-16 char

type
  { TMergePair }
  TMergePair = record
    Left: string;
    Right: string;
  end;

  { TTokenizer }
  // BPE tokenizer for Gemma 4 E4B
  TTokenizer = class(TBaseObject)
  private
    // Token string -> ID
    FVocab: TDictionary<string, Integer>;
    // ID -> Token string
    FIdToToken: TDictionary<Integer, string>;
    // Merge pair -> priority (lower = higher priority)
    FMerges: TDictionary<string, Integer>;
    // Special tokens
    FAddedTokens: TDictionary<string, Integer>;

    FBosTokenId: Integer;
    FEosTokenId: Integer;
    FPadTokenId: Integer;
    FUnkTokenId: Integer;
    FVocabSize: Integer;
    FIsLoaded: Boolean;

    function DoParseVocab(const AModel: TJSON): Boolean;
    function DoParseMerges(const AModel: TJSON): Boolean;
    function DoParseAddedTokens(const ARoot: TJSON): Boolean;
    function DoNormalize(const AText: string): string;
    function DoDenormalize(const AText: string): string;
    function DoMergePairKey(const ALeft: string;
      const ARight: string): string;
    procedure DoBpeMerge(var ATokens: TList<string>);
    function DoEncodeWord(const AWord: string): TArray<Integer>;
    function DoTokenToId(const AToken: string): Integer;
  public
    constructor Create(); override;
    destructor Destroy(); override;

    // Load from JSON string (e.g. read from VPK)
    function LoadFromString(const AJsonStr: string): Boolean;

    // Load from file
    function LoadFromFile(const AFilename: string): Boolean;

    function IsLoaded(): Boolean;

    // Encode text to token IDs
    function Encode(const AText: string;
      const AAddBos: Boolean = False): TArray<Integer>;

    // Decode token IDs to text
    function Decode(const ATokenIds: TArray<Integer>): string;

    // Single token lookup
    function TokenToId(const AToken: string): Integer;
    function IdToToken(const AId: Integer): string;

    // Encode a raw formatted prompt string that contains special tokens
    // (e.g. "<bos><|turn>user\nhi there<turn|>\n<|turn>model\n")
    // Splits on special token boundaries, maps them to IDs directly,
    // and BPE-encodes non-special segments.
    function EncodeRaw(const AText: string): TArray<Integer>;

    // Format a chat prompt with turn markers
    function FormatChat(const ARole: string;
      const AContent: string): string;
    function FormatUserTurn(const AContent: string): string;
    function FormatModelTurn(): string;

    // Properties
    function GetVocabSize(): Integer;
    function GetBosTokenId(): Integer;
    function GetEosTokenId(): Integer;
  end;

implementation

uses
  System.IOUtils;

{ TTokenizer }

constructor TTokenizer.Create();
begin
  inherited Create();
  FVocab := TDictionary<string, Integer>.Create();
  FIdToToken := TDictionary<Integer, string>.Create();
  FMerges := TDictionary<string, Integer>.Create();
  FAddedTokens := TDictionary<string, Integer>.Create();
  FBosTokenId := CBosTokenId;
  FEosTokenId := CEosTokenId1;
  FPadTokenId := CPadTokenId;
  FUnkTokenId := 3;
  FVocabSize := 0;
  FIsLoaded := False;
end;

destructor TTokenizer.Destroy();
begin
  FAddedTokens.Free();
  FMerges.Free();
  FIdToToken.Free();
  FVocab.Free();
  inherited;
end;

function TTokenizer.LoadFromString(const AJsonStr: string): Boolean;
var
  LJSON: TJSON;
  LModel: TJSON;
begin
  Result := False;
  FIsLoaded := False;

  LJSON := TJSON.FromString(AJsonStr);
  try
    if LJSON.IsNull() then
    begin
      GetErrors().Add(esError, CTK_ERR_PARSE,
        'Failed to parse tokenizer JSON');
      Exit;
    end;

    // Parse added_tokens (special tokens with IDs)
    DoParseAddedTokens(LJSON);

    // Parse model section (vocab + merges)
    LModel := LJSON.Get('model');
    if LModel.IsNull() then
    begin
      GetErrors().Add(esError, CTK_ERR_PARSE,
        'Missing model section in tokenizer.json');
      Exit;
    end;

    if not DoParseVocab(LModel) then
      Exit;

    if not DoParseMerges(LModel) then
      Exit;

    FVocabSize := FVocab.Count;
    FIsLoaded := True;
    Result := True;

    Status('Tokenizer loaded: %d vocab, %d merges, %d special tokens', [
      FVocab.Count, FMerges.Count, FAddedTokens.Count]);
  finally
    LJSON.Free();
  end;
end;

function TTokenizer.LoadFromFile(const AFilename: string): Boolean;
var
  LText: string;
begin
  Result := False;
  try
    LText := TFile.ReadAllText(AFilename, TEncoding.UTF8);
  except
    on E: Exception do
    begin
      GetErrors().Add(esError, CTK_ERR_PARSE,
        'Failed to read tokenizer file: %s', [E.Message]);
      Exit;
    end;
  end;
  Result := LoadFromString(LText);
end;

function TTokenizer.IsLoaded(): Boolean;
begin
  Result := FIsLoaded;
end;

function TTokenizer.DoParseVocab(const AModel: TJSON): Boolean;
var
  LVocab: TJSON;
  LPairs: TArray<TJSONPair>;
  LI: Integer;
  LId: Integer;
begin
  Result := False;

  LVocab := AModel.Get('vocab');
  if LVocab.IsNull() then
  begin
    GetErrors().Add(esError, CTK_ERR_VOCAB,
      'Missing vocab in tokenizer model');
    Exit;
  end;

  LPairs := LVocab.Pairs();
  for LI := 0 to High(LPairs) do
  begin
    LId := LPairs[LI].Value.AsInt32(-1);
    if LId >= 0 then
    begin
      FVocab.AddOrSetValue(LPairs[LI].NodeName, LId);
      FIdToToken.AddOrSetValue(LId, LPairs[LI].NodeName);
    end;
  end;

  Result := FVocab.Count > 0;
  if not Result then
    GetErrors().Add(esError, CTK_ERR_VOCAB, 'Vocab is empty');
end;

function TTokenizer.DoParseMerges(const AModel: TJSON): Boolean;
var
  LMergesArr: TJSON;
  LItems: TArray<TJSON>;
  LI: Integer;
  LLeft: string;
  LRight: string;
  LKey: string;
begin
  Result := False;

  LMergesArr := AModel.Get('merges');
  if LMergesArr.IsNull() then
  begin
    GetErrors().Add(esError, CTK_ERR_MERGES,
      'Missing merges in tokenizer model');
    Exit;
  end;

  LItems := LMergesArr.Items();
  for LI := 0 to High(LItems) do
  begin
    // Each merge can be either:
    //   - a string "tok1 tok2" (standard HF format)
    //   - an array ["tok1", "tok2"] (Gemma 4 format)
    if LItems[LI].IsArray() then
    begin
      if LItems[LI].Count() >= 2 then
      begin
        LLeft := LItems[LI].Get('[0]').AsString('');
        LRight := LItems[LI].Get('[1]').AsString('');
        if (LLeft <> '') and (LRight <> '') then
        begin
          LKey := LLeft + ' ' + LRight;
          FMerges.AddOrSetValue(LKey, LI);
        end;
      end;
    end
    else
    begin
      // String format: "tok1 tok2"
      LKey := LItems[LI].AsString('');
      if (LKey <> '') and (Pos(' ', LKey) > 0) then
        FMerges.AddOrSetValue(LKey, LI);
    end;
  end;

  Result := FMerges.Count > 0;
  if not Result then
    GetErrors().Add(esError, CTK_ERR_MERGES, 'Merges list is empty');
end;

function TTokenizer.DoParseAddedTokens(const ARoot: TJSON): Boolean;
var
  LAddedArr: TJSON;
  LItems: TArray<TJSON>;
  LI: Integer;
  LContent: string;
  LId: Integer;
begin
  Result := False;

  LAddedArr := ARoot.Get('added_tokens');
  if LAddedArr.IsNull() then
    Exit;

  LItems := LAddedArr.Items();
  for LI := 0 to High(LItems) do
  begin
    LContent := LItems[LI].Get('content').AsString('');
    LId := LItems[LI].Get('id').AsInt32(-1);
    if (LContent <> '') and (LId >= 0) then
    begin
      FAddedTokens.AddOrSetValue(LContent, LId);
      FVocab.AddOrSetValue(LContent, LId);
      FIdToToken.AddOrSetValue(LId, LContent);
    end;
  end;

  Result := True;
end;

function TTokenizer.DoNormalize(const AText: string): string;
begin
  // SentencePiece normalization: replace spaces with the block character
  Result := AText.Replace(' ', CSentencePieceSpace);
end;

function TTokenizer.DoDenormalize(const AText: string): string;
begin
  // Reverse SentencePiece normalization
  Result := AText.Replace(CSentencePieceSpace, ' ');
end;

function TTokenizer.DoMergePairKey(const ALeft: string;
  const ARight: string): string;
begin
  Result := ALeft + ' ' + ARight;
end;

procedure TTokenizer.DoBpeMerge(var ATokens: TList<string>);
var
  LBestIdx: Integer;
  LBestPriority: Integer;
  LI: Integer;
  LKey: string;
  LPriority: Integer;
  LMerged: string;
begin
  // Repeatedly find and apply the highest-priority merge
  while ATokens.Count >= 2 do
  begin
    LBestIdx := -1;
    LBestPriority := MaxInt;

    // Find the pair with the lowest merge index (highest priority)
    for LI := 0 to ATokens.Count - 2 do
    begin
      LKey := DoMergePairKey(ATokens[LI], ATokens[LI + 1]);
      if FMerges.TryGetValue(LKey, LPriority) then
      begin
        if LPriority < LBestPriority then
        begin
          LBestPriority := LPriority;
          LBestIdx := LI;
        end;
      end;
    end;

    // No more merges possible
    if LBestIdx < 0 then
      Break;

    // Apply the merge: combine tokens[bestIdx] and tokens[bestIdx+1]
    LMerged := ATokens[LBestIdx] + ATokens[LBestIdx + 1];
    ATokens[LBestIdx] := LMerged;
    ATokens.Delete(LBestIdx + 1);
  end;
end;

function TTokenizer.DoTokenToId(const AToken: string): Integer;
begin
  if not FVocab.TryGetValue(AToken, Result) then
    Result := FUnkTokenId;
end;

function TTokenizer.DoEncodeWord(const AWord: string): TArray<Integer>;
var
  LChars: TList<string>;
  LIds: TList<Integer>;
  LI: Integer;
  LJ: Integer;
  LId: Integer;
  LUtf8: TBytes;
  LByteTok: string;
begin
  // SentencePiece-style BPE with byte fallback (tokenizer.json:
  // model.type=BPE, byte_fallback=true):
  //   1. Split into Unicode CHARACTERS (code points), not bytes.
  //   2. Apply BPE merges over the character strings.
  //   3. Convert merged tokens to IDs; any token still not in the vocab
  //      falls back to its UTF-8 bytes as <0xNN> byte tokens.
  // Starting from byte tokens is WRONG: merge keys are character strings,
  // so byte tokens would never merge and everything degrades to bytes.
  LChars := TList<string>.Create();
  LIds := TList<Integer>.Create();
  try
    // Step 1: split into code points. Delphi strings are UTF-16, so a
    // surrogate pair (e.g. emoji) is kept together as one element.
    LI := 1;
    while LI <= Length(AWord) do
    begin
      if (AWord[LI] >= #$D800) and (AWord[LI] <= #$DBFF) and
         (LI < Length(AWord)) then
      begin
        LChars.Add(Copy(AWord, LI, 2));
        Inc(LI, 2);
      end
      else
      begin
        LChars.Add(AWord[LI]);
        Inc(LI);
      end;
    end;

    // Step 2: apply BPE merges
    DoBpeMerge(LChars);

    // Step 3: tokens to IDs, byte fallback for out-of-vocab tokens
    for LI := 0 to LChars.Count - 1 do
    begin
      if FVocab.TryGetValue(LChars[LI], LId) then
        LIds.Add(LId)
      else
      begin
        LUtf8 := TEncoding.UTF8.GetBytes(LChars[LI]);
        for LJ := 0 to High(LUtf8) do
        begin
          LByteTok := Format('<0x%s>', [IntToHex(LUtf8[LJ], 2)]);
          LIds.Add(DoTokenToId(LByteTok));
        end;
      end;
    end;

    Result := LIds.ToArray();
  finally
    LIds.Free();
    LChars.Free();
  end;
end;

function TTokenizer.Encode(const AText: string;
  const AAddBos: Boolean): TArray<Integer>;
var
  LNormalized: string;
  LResult: TList<Integer>;
  LWordIds: TArray<Integer>;
  LJ: Integer;
  LAddedId: Integer;
begin
  if not FIsLoaded then
  begin
    SetLength(Result, 0);
    Exit;
  end;

  LResult := TList<Integer>.Create();
  try
    // Add BOS if requested
    if AAddBos then
      LResult.Add(FBosTokenId);

    // Check if the entire text is a special/added token
    if FAddedTokens.TryGetValue(AText, LAddedId) then
    begin
      LResult.Add(LAddedId);
      Result := LResult.ToArray();
      Exit;
    end;

    // Normalize: replace ALL spaces with the SentencePiece marker.
    // Per tokenizer.json normalizer: {type: Replace, " " -> U+2581}.
    // NOTHING is prepended to the start of the text.
    LNormalized := DoNormalize(AText);

    // BPE over the ENTIRE normalized text as one sequence.
    // Per tokenizer.json the pre_tokenizer splits on " ", but no spaces
    // remain after normalization, so the whole string is a single BPE
    // segment (SentencePiece-style). Token boundaries emerge from the
    // learned merges, not from pre-splitting.
    LWordIds := DoEncodeWord(LNormalized);
    for LJ := 0 to High(LWordIds) do
      LResult.Add(LWordIds[LJ]);

    Result := LResult.ToArray();
  finally
    LResult.Free();
  end;
end;

function TTokenizer.Decode(const ATokenIds: TArray<Integer>): string;
var
  LBuilder: TStringBuilder;
  LI: Integer;
  LToken: string;
  LByteBuf: TBytes;
  LByteCount: Integer;
  LByteVal: Integer;

  // Flush accumulated byte-fallback bytes as UTF-8 decoded text
  procedure FlushBytes();
  begin
    if LByteCount > 0 then
    begin
      LBuilder.Append(TEncoding.UTF8.GetString(LByteBuf, 0, LByteCount));
      LByteCount := 0;
    end;
  end;

begin
  // Decoder pipeline per tokenizer.json:
  //   1. Replace U+2581 -> " " (done via DoDenormalize at the end)
  //   2. ByteFallback: consecutive <0xNN> tokens fuse into a byte run
  //      which is decoded as UTF-8
  //   3. Fuse: all pieces concatenate into one string
  LBuilder := TStringBuilder.Create();
  try
    SetLength(LByteBuf, Length(ATokenIds));
    LByteCount := 0;

    for LI := 0 to High(ATokenIds) do
    begin
      if FIdToToken.TryGetValue(ATokenIds[LI], LToken) then
      begin
        // Byte-fallback token <0xNN>: accumulate the raw byte
        if (Length(LToken) = 6) and LToken.StartsWith('<0x') and
           LToken.EndsWith('>') and
           TryStrToInt('$' + Copy(LToken, 4, 2), LByteVal) then
        begin
          LByteBuf[LByteCount] := Byte(LByteVal);
          Inc(LByteCount);
          Continue;
        end;

        FlushBytes();

        // Skip special tokens in decode output
        if FAddedTokens.ContainsKey(LToken) then
          Continue;
        LBuilder.Append(LToken);
      end;
    end;

    FlushBytes();

    // Denormalize: replace SentencePiece block chars back to spaces
    Result := DoDenormalize(LBuilder.ToString());
  finally
    LBuilder.Free();
  end;
end;

function TTokenizer.TokenToId(const AToken: string): Integer;
begin
  Result := DoTokenToId(AToken);
end;

function TTokenizer.IdToToken(const AId: Integer): string;
begin
  if not FIdToToken.TryGetValue(AId, Result) then
    Result := '<unk>';
end;

function TTokenizer.EncodeRaw(const AText: string): TArray<Integer>;
var
  LResult: TList<Integer>;
  LRemaining: string;
  LFoundPos: Integer;
  LBestPos: Integer;
  LBestToken: string;
  LBestId: Integer;
  LSegment: string;
  LSegmentIds: TArray<Integer>;
  LI: Integer;
  LPair: TPair<string, Integer>;
begin
  if not FIsLoaded then
  begin
    SetLength(Result, 0);
    Exit;
  end;

  LResult := TList<Integer>.Create();
  try
    LRemaining := AText;

    while LRemaining <> '' do
    begin
      // Find the earliest occurring special token in the remaining text
      LBestPos := MaxInt;
      LBestToken := '';
      LBestId := -1;

      for LPair in FAddedTokens do
      begin
        LFoundPos := Pos(LPair.Key, LRemaining);
        if (LFoundPos > 0) and (LFoundPos < LBestPos) then
        begin
          LBestPos := LFoundPos;
          LBestToken := LPair.Key;
          LBestId := LPair.Value;
        end;
      end;

      if LBestToken = '' then
      begin
        // No more special tokens -- BPE-encode the rest
        if LRemaining <> '' then
        begin
          LSegmentIds := Encode(LRemaining, False);
          for LI := 0 to High(LSegmentIds) do
            LResult.Add(LSegmentIds[LI]);
        end;
        Break;
      end;

      // BPE-encode the text before the special token
      LSegment := Copy(LRemaining, 1, LBestPos - 1);
      if LSegment <> '' then
      begin
        LSegmentIds := Encode(LSegment, False);
        for LI := 0 to High(LSegmentIds) do
          LResult.Add(LSegmentIds[LI]);
      end;

      // Add the special token ID directly
      LResult.Add(LBestId);

      // Advance past the special token
      LRemaining := Copy(LRemaining, LBestPos + Length(LBestToken));
    end;

    Result := LResult.ToArray();
  finally
    LResult.Free();
  end;
end;

function TTokenizer.FormatChat(const ARole: string;
  const AContent: string): string;
begin
  // Gemma 4 chat format:
  // <|turn>role
  // content<turn|>
  Result := '<|turn>' + ARole + #10 + AContent + '<turn|>' + #10;
end;

function TTokenizer.FormatUserTurn(const AContent: string): string;
begin
  Result := FormatChat('user', AContent);
end;

function TTokenizer.FormatModelTurn(): string;
begin
  // Start of model turn (no content -- model generates from here)
  Result := '<|turn>model' + #10;
end;

function TTokenizer.GetVocabSize(): Integer;
begin
  Result := FVocabSize;
end;

function TTokenizer.GetBosTokenId(): Integer;
begin
  Result := FBosTokenId;
end;

function TTokenizer.GetEosTokenId(): Integer;
begin
  Result := FEosTokenId;
end;

end.
