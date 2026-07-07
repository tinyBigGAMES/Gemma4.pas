{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information
===============================================================================}

unit UDemo.Multimedia;

interface

uses
  System.SysUtils,
  StdApp.Console,
  Gemma4.Types,
  Gemma4.Inference,
  UTestbed.Common;

type
  { TDemoMultimedia }
  // Three-turn multimodal conversation: image description, audio
  // description, video description -- each modality in a separate turn
  // through TInference.Generate with TMessagePart-based messages.
  // Media files are resolved relative to the exe: res\image.png, etc.
  TDemoMultimedia = class(TBaseDemoCase)
  public
    constructor Create(); override;
    procedure OnRender(); override;
  end;

implementation

{ TDemoMultimedia }

constructor TDemoMultimedia.Create();
begin
  inherited;
  Title := 'Multimedia: image + audio + video via TInference';
end;

procedure TDemoMultimedia.OnRender();
var
  LInf: TInference;
  LResDir: string;
  LImagePath: string;
  LAudioPath: string;
  LVideoPath: string;
  LStats: TInferenceStats;
begin
  LResDir := ExtractFilePath(ParamStr(0)) + 'res' + PathDelim;
  LImagePath := LResDir + 'image.png';
  LAudioPath := LResDir + 'audio.wav';
  LVideoPath := LResDir + 'video.mp4';

  // Verify all media files exist before loading the model
  if not FileExists(LImagePath) then
  begin
    PrintLn(COLOR_RED + 'Missing: %s', [LImagePath]);
    Exit;
  end;
  if not FileExists(LAudioPath) then
  begin
    PrintLn(COLOR_RED + 'Missing: %s', [LAudioPath]);
    Exit;
  end;
  if not FileExists(LVideoPath) then
  begin
    PrintLn(COLOR_RED + 'Missing: %s', [LVideoPath]);
    Exit;
  end;

  PrintLn(COLOR_WHITE + 'Image: %s', [LImagePath]);
  PrintLn(COLOR_WHITE + 'Audio: %s', [LAudioPath]);
  PrintLn(COLOR_WHITE + 'Video: %s', [LVideoPath]);
  PrintLn('', []);

  LInf := TInference.Create();
  try
    //LInf.SetStatusCallback(StatusCallback, Self);
    LInf.SetTokenCallback(TokenCallback, Self);
    LInf.SetLoadProgressCallback(LoadProgressCallback, Self);

    if not LInf.LoadModel(CVpkOutputFile) then
    begin
      PrintLn(COLOR_RED + 'LoadModel failed', []);
      LInf.PrintErrors();
      Exit;
    end;

    LInf.EnableThinking := False;

    // Turn 1: image (image before text per model card)
    ResetAdapter();
    PrintLn(COLOR_CYAN + '--- Turn 1: Image ---', []);
    Print(COLOR_WHITE + 'model> ', []);
    LInf.AddMessage(CRoleUser, [
      TMessagePart.ImagePart(LImagePath),
      TMessagePart.TextPart('Describe this image in one sentence.')
    ]);
    if not LInf.Generate(512) then
    begin
      PrintLn('', []);
      PrintLn(COLOR_RED + 'Image generation failed', []);
      LInf.PrintErrors();
      Exit;
    end;
    FlushAdapter();
    LInf.AddMessage(CRoleAssistant, LInf.ResponseText);
    PrintLn('', []);
    LStats := LInf.Stats;
    TConsole.PrintLn(LStats.FormatText(sdkBasic, True));

    // Turn 2: audio (audio after text per model card)
    ResetAdapter();
    PrintLn('', []);
    PrintLn(COLOR_CYAN + '--- Turn 2: Audio ---', []);
    Print(COLOR_WHITE + 'model> ', []);
    LInf.AddMessage(CRoleUser, [
      TMessagePart.TextPart('Describe this audio in one sentence.'),
      TMessagePart.AudioPart(LAudioPath)
    ]);
    if not LInf.Generate(512) then
    begin
      PrintLn('', []);
      PrintLn(COLOR_RED + 'Audio generation failed', []);
      LInf.PrintErrors();
      Exit;
    end;
    FlushAdapter();
    LInf.AddMessage(CRoleAssistant, LInf.ResponseText);
    PrintLn('', []);
    LStats := LInf.Stats;
    TConsole.PrintLn(LStats.FormatText(sdkBasic, True));

    // Turn 3: video (video before text per model card)
    ResetAdapter();
    PrintLn('', []);
    PrintLn(COLOR_CYAN + '--- Turn 3: Video ---', []);
    Print(COLOR_WHITE + 'model> ', []);
    LInf.AddMessage(CRoleUser, [
      TMessagePart.VideoPart(LVideoPath),
      TMessagePart.TextPart('Describe this video in one sentence.')
    ]);
    if not LInf.Generate(512) then
    begin
      PrintLn('', []);
      PrintLn(COLOR_RED + 'Video generation failed', []);
      LInf.PrintErrors();
      Exit;
    end;
    FlushAdapter();
    PrintLn('', []);
    LStats := LInf.Stats;
    TConsole.PrintLn(LStats.FormatText(sdkBasic, True));

    PrintLn('', []);
    PrintLn(COLOR_GREEN + 'DONE -- verify image, audio AND video ' +
      'outputs are grounded', []);

    LInf.PrintErrors();
  finally
    LInf.Free();
  end;
end;

end.
