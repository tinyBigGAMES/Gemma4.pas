{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information
===============================================================================}

unit UDemo.Inference;

interface

uses
  WinApi.Windows,
  System.SysUtils,
  StdApp.Input,
  StdApp.Console,
  Gemma4.Types,
  Gemma4.Inference,
  UTestbed.Common;

type
  { TDemoInference }
  // Streaming chat: loads Gemma4.vpk, runs a two-turn conversation with
  // the thinking channel visible, prints generation statistics per turn.
  // Inherits TBaseDemoCase for markdown/LaTeX rendering via TConsoleAdapter.
  TDemoInference = class(TBaseDemoCase)
  private
    function DoRunTurn(
      const AInf: TInference;
      const APrompt: string;
      const AMaxTokens: Integer
    ): Boolean;
  public
    constructor Create(); override;
    procedure OnRender(); override;
  end;

implementation

const
  { CPrompt1 }
  CPrompt1 = 'In two sentences, what makes the Vulkan API different from OpenGL?';

  { CPrompt2 }
  // Second turn deliberately depends on the first -- proves history works
  CPrompt2 = 'Give one concrete example of that difference in practice.';

{ TDemoInference }

constructor TDemoInference.Create();
begin
  inherited;
  Title := 'Inference: streaming chat (thinking visible)';
end;

function TDemoInference.DoRunTurn(
  const AInf: TInference;
  const APrompt: string;
  const AMaxTokens: Integer): Boolean;
var
  LStats: TInferenceStats;
begin
  Result := False;

  ResetAdapter();

  PrintLn('', []);
  PrintLn(COLOR_CYAN + 'user> %s', [APrompt]);
  //Print(COLOR_WHITE + 'model> ', []);

  AInf.AddMessage(CRoleUser, APrompt);
  if not AInf.Generate(AMaxTokens) then
  begin
    FlushAdapter();
    PrintLn('', []);
    PrintLn(COLOR_RED + 'Generation failed', []);
    AInf.PrintErrors();
    Exit;
  end;

  FlushAdapter();

  // Feed the visible reply back into the history so the next turn
  // renders a complete conversation
  AInf.AddMessage(CRoleAssistant, AInf.ResponseText);

  PrintLn('', []);
  PrintLn('', []);
  LStats := AInf.Stats;
  TConsole.PrintLn(LStats.FormatText(sdkBasic, True));
  Result := True;
end;

procedure TDemoInference.OnRender();
var
  LInf: TInference;
begin
  LInf := TInference.Create();
  try
    LInf.ThinkingOpenText := COLOR_GREEN + '<thinking>'#10;
    Linf.ThinkingCloseText := COLOR_GREEN + #10'</thinking>'#10;
    LInf.ThinkingPlaceholderText := COLOR_GREEN + 'Thinking...' + sLineBreak;;

    LInf.SetCancelCallback(
      function(const AUserData: Pointer): Boolean
      begin
        Result := TInput.KeyPressed(VK_ESCAPE);
      end,
      nil
    );

    //LInf.SetStatusCallback(StatusCallback, Self);
    LInf.SetTokenCallback(TokenCallback, Self);
    LInf.SetLoadProgressCallback(LoadProgressCallback, Self);

    if not LInf.LoadModel(CVpkOutputFile) then
    begin
      PrintLn(COLOR_RED + 'LoadModel failed: %s', [CVpkOutputFile]);
      LInf.PrintErrors();
      Exit;
    end;

    LInf.EnableThinking := True;
    LInf.ShowThinking := True;

    if not DoRunTurn(LInf, CPrompt1, 1024) then
      Exit;
    if not DoRunTurn(LInf, CPrompt2, 1024) then
      Exit;
  finally
    LInf.Free();
  end;
end;

end.
