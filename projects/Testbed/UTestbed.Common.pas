{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information
===============================================================================}

unit UTestbed.Common;

interface

uses
  StdApp.Console,
  StdApp.Console.Adapter,
  StdApp.TestDemo,
  Gemma4.Types,
  Gemma4.Inference,
  Gemma4.Chat;

const
  CSafetensorsFile = 'C:\Dev\LLM\SAFETENSORS\Huihui-gemma-4-E4B-it-qat-q4_0-unquantized-abliterated\model.safetensors';
  CVpkBuildDir = 'C:\Dev\LLM\VPK\Gemma4\E4B';
  CEmbSafetensorsDir = 'C:\Dev\LLM\SAFETENSORS\embeddinggemma-300m';
  CEmbBuildDir = 'C:\Dev\LLM\VPK\Gemma4\embeddings';
  CVpkOutputFile = 'C:\Dev\LLM\VPK\Gemma4.vpk';

  { Chat demo constants }
  CMaxTokens  = 2048;
  CChatMemory = 'chat_memory.db';
  CChatSystemPrompt = 'You are Gemma4, a helpful assistant powered by Gemma4.pas!';

type
  { TBaseDemoCase }
  // Common base for all demos. Owns a TConsoleAdapter for streaming
  // markdown/LaTeX rendering of LLM output. Provides DoOnStatus and
  // DoOnToken methods that the standalone callbacks delegate to via
  // the AUserData pointer (which carries the demo instance).
  TBaseDemoCase = class(TTestDemo)
  private
    FAdapter: TConsoleAdapter;
  public
    constructor Create(); override;
    destructor Destroy(); override;

    // Called by StatusCallback -- prints a status line to the console
    procedure OnStatus(const AText: string);

    // Called by TokenCallback -- routes tokens through the adapter
    procedure OnToken(const AState: TProgressState;
      const AToken: string);

    // Called by LoadProgressCallback -- drives a console progress bar
    procedure OnLoadProgress(const AState: TProgressState;
      const AStep: Integer; const ATotal: Integer);

    // Flush any buffered adapter output (call at end of generation)
    procedure FlushAdapter();

    // Reset adapter state (call before a new generation turn)
    procedure ResetAdapter();

    // Returns True if safetensors source files exist for VPK packing
    class function CheckForVPKFiles(): Boolean;

    property Adapter: TConsoleAdapter read FAdapter;
  end;

  { TBaseDemoChat }
  // Common base for chat demos. Creates a TConsoleChat, applies shared
  // config (VPK path, system prompt, thinking, max tokens), calls the
  // DoConfigureChat hook for subclass-specific setup (tools, etc.),
  // then runs the interactive loop.
  TBaseDemoChat = class(TTestDemo)
  private
    FChat: TConsoleChat;
  protected
    // Override to configure tools or other chat-specific settings.
    // FChat is created and common config applied before this fires.
    procedure DoConfigureChat(); virtual;
  public
    constructor Create(); override;
    procedure OnRender(); override;
    property Chat: TConsoleChat read FChat;
  end;

// Standalone callbacks -- cast AUserData to TBaseDemoCase and delegate.
// Pass Self as AUserData when wiring these to SetStatusCallback /
// SetTokenCallback.
procedure StatusCallback(
  const AText: string;
  const AUserData: Pointer
);

procedure TokenCallback(
  const AState: TProgressState;
  const AToken: string;
  const AUserData: Pointer
);

procedure LoadProgressCallback(
  const AState: TProgressState;
  const AStep: Integer;
  const ATotal: Integer;
  const AUserData: Pointer
);

implementation

uses
  System.SysUtils,
  System.IOUtils;

{ TBaseDemoCase }

constructor TBaseDemoCase.Create();
begin
  inherited;
  FAdapter := TConsoleAdapter.Create();
  FAdapter.Markdown := True;
  FAdapter.SetOutputCallback(
    procedure(const AText: string; const AUserData: Pointer)
    begin
      TConsole.Print(AText, []);
    end, nil);
end;

destructor TBaseDemoCase.Destroy();
begin
  FAdapter.Free();
  inherited;
end;

procedure TBaseDemoCase.OnStatus(const AText: string);
begin
  TConsole.PrintLn(AText);
end;

procedure TBaseDemoCase.OnToken(const AState: TProgressState;
  const AToken: string);
begin
  case AState of
    psInProgress:
      FAdapter.Write(AToken);
    psEnd:
      FAdapter.Flush();
  end;
end;

procedure TBaseDemoCase.OnLoadProgress(const AState: TProgressState;
  const AStep: Integer; const ATotal: Integer);
begin
  case AState of
    psStart:
      TConsole.PrintLn('Loading model...');
    psInProgress:
      TConsole.ProgressBar(AStep, ATotal);
    psEnd:
    begin
      TConsole.ClearLine(True);
      TConsole.CursorUp();
      TConsole.ClearLine(True);
    end;
  end;
end;

procedure TBaseDemoCase.FlushAdapter();
begin
  FAdapter.Flush();
end;

procedure TBaseDemoCase.ResetAdapter();
begin
  FAdapter.Reset();
end;

{ StatusCallback }

procedure StatusCallback(
  const AText: string;
  const AUserData: Pointer);
begin
  if AUserData <> nil then
    TBaseDemoCase(AUserData).OnStatus(AText);
end;

{ TokenCallback }

procedure TokenCallback(
  const AState: TProgressState;
  const AToken: string;
  const AUserData: Pointer);
begin
  if AUserData <> nil then
    TBaseDemoCase(AUserData).OnToken(AState, AToken);
end;

{ LoadProgressCallback }

procedure LoadProgressCallback(
  const AState: TProgressState;
  const AStep: Integer;
  const ATotal: Integer;
  const AUserData: Pointer);
begin
  if AUserData <> nil then
    TBaseDemoCase(AUserData).OnLoadProgress(AState, AStep, ATotal);
end;

{ CheckForVPKFiles }

class function TBaseDemoCase.CheckForVPKFiles(): Boolean;
begin
  Result := FileExists(CSafetensorsFile) and
            TDirectory.Exists(CEmbSafetensorsDir);
end;

{ TBaseDemoChat }

constructor TBaseDemoChat.Create();
begin
  inherited;
  FChat := nil;
end;

procedure TBaseDemoChat.DoConfigureChat();
begin
  // no-op -- override to add tools, etc.
end;

procedure TBaseDemoChat.OnRender();
begin
  FChat := TConsoleChat.Create();
  try
    FChat.ModelPath := CVpkOutputFile;
    FChat.MemoryDbPath := CChatMemory;
    FChat.MaxTokens := CMaxTokens;
    FChat.EnableThinking := True;
    FChat.ShowThinking := True;
    FChat.SystemPrompt := CChatSystemPrompt;

    DoConfigureChat();

    FChat.Run();
  finally
    FChat.Free();
    FChat := nil;
  end;

  Terminate();
end;

end.
