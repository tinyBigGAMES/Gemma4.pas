{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information
===============================================================================}

unit UDemo.Embedding;

interface

uses
  System.SysUtils,
  StdApp.Console,
  Gemma4.Embeddings,
  UTestbed.Common;

type
  { TDemoEmbedding }
  // Semantic search: embeds a small document corpus once, then ranks the
  // corpus against several queries by cosine similarity. Each query has a
  // known expected top document, so the demo self-checks.
  TDemoEmbedding = class(TBaseDemoCase)
  public
    constructor Create(); override;
    procedure OnRender(); override;
  end;

implementation

const
  { CDocs }
  CDocs: array[0..5] of string = (
    'Mars, known for its reddish appearance, is often referred to as the Red Planet.',
    'Jupiter, the largest planet in our solar system, has a prominent red spot.',
    'Photosynthesis converts sunlight, water and carbon dioxide into glucose and oxygen.',
    'The Great Wall of China stretches thousands of kilometers across northern China.',
    'Vulkan is a low-overhead, cross-platform graphics and compute API.',
    'A sourdough starter is a fermented culture of flour and water used to leaven bread.'
  );

  { CQueries }
  CQueries: array[0..2] of string = (
    'Which planet is known as the Red Planet?',
    'How do plants make food from sunlight?',
    'What is a low-level GPU programming interface?'
  );

  { CExpectedTop }
  // Index into CDocs of the document each query must rank first
  CExpectedTop: array[0..2] of Integer = (0, 2, 4);

{ TDemoEmbedding }

constructor TDemoEmbedding.Create();
begin
  inherited;
  Title := 'Embeddings: semantic search (EmbeddingGemma-300m)';
end;

procedure TDemoEmbedding.OnRender();
var
  LEmb: TEmbeddings;
  LDocVecs: TArray<TArray<Single>>;
  LQuery: TArray<Single>;
  LSims: TArray<Single>;
  LOrder: TArray<Integer>;
  LQ: Integer;
  LI: Integer;
  LJ: Integer;
  LTmp: Integer;
  LAllPassed: Boolean;
begin
  LEmb := TEmbeddings.Create();
  try
    //LEmb.SetStatusCallback(StatusCallback, Self);
    LEmb.SetLoadProgressCallback(LoadProgressCallback, Self);

    if not LEmb.Open(CVpkOutputFile) then
    begin
      PrintLn(COLOR_RED + 'Open failed: %s', [CVpkOutputFile]);
      LEmb.PrintErrors();
      Exit;
    end;

    PrintLn('', []);
    PrintLn(COLOR_WHITE + 'Embedding %d documents...', [Length(CDocs)]);
    SetLength(LDocVecs, Length(CDocs));
    for LI := 0 to High(CDocs) do
    begin
      LDocVecs[LI] := LEmb.EmbedDocument(CDocs[LI]);
      if Length(LDocVecs[LI]) = 0 then
      begin
        PrintLn(COLOR_RED + 'EmbedDocument failed on doc %d', [LI]);
        LEmb.PrintErrors();
        Exit;
      end;
    end;
    PrintLn(COLOR_WHITE + 'dim = %d', [Length(LDocVecs[0])]);

    LAllPassed := True;
    SetLength(LSims, Length(CDocs));
    SetLength(LOrder, Length(CDocs));
    for LQ := 0 to High(CQueries) do
    begin
      LQuery := LEmb.EmbedQuery(CQueries[LQ]);
      if Length(LQuery) = 0 then
      begin
        PrintLn(COLOR_RED + 'EmbedQuery failed on query %d', [LQ]);
        LEmb.PrintErrors();
        Exit;
      end;

      for LI := 0 to High(CDocs) do
      begin
        LSims[LI] := TEmbeddings.Similarity(LQuery, LDocVecs[LI]);
        LOrder[LI] := LI;
      end;

      // Selection sort descending by similarity
      for LI := 0 to High(LOrder) - 1 do
        for LJ := LI + 1 to High(LOrder) do
          if LSims[LOrder[LJ]] > LSims[LOrder[LI]] then
          begin
            LTmp := LOrder[LI];
            LOrder[LI] := LOrder[LJ];
            LOrder[LJ] := LTmp;
          end;

      PrintLn('', []);
      PrintLn(COLOR_CYAN + 'query> %s', [CQueries[LQ]]);
      for LI := 0 to High(LOrder) do
        PrintLn(COLOR_WHITE + '  %.4f  [doc %d] %s',
          [LSims[LOrder[LI]], LOrder[LI],
           Copy(CDocs[LOrder[LI]], 1, 60)]);

      if LOrder[0] = CExpectedTop[LQ] then
        PrintLn(COLOR_GREEN + '  top hit correct (doc %d)', [LOrder[0]])
      else
      begin
        PrintLn(COLOR_RED + '  WRONG top hit: doc %d, expected doc %d',
          [LOrder[0], CExpectedTop[LQ]]);
        LAllPassed := False;
      end;
    end;

    PrintLn('', []);
    if LAllPassed then
      PrintLn(COLOR_GREEN + 'All %d queries retrieved the expected document',
        [Length(CQueries)])
    else
      PrintLn(COLOR_RED + 'Retrieval mismatches -- see above', []);

    LEmb.PrintErrors();
  finally
    LEmb.Free();
  end;
end;

end.
