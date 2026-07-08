{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information
===============================================================================}

unit UDemo.ChatWithDynTools;

interface

uses
  Gemma4.Tools,
  UTestbed.Common;

type
  { TDemoChatWithDynTools }
  // Interactive console chat with dynamic tool discovery. Two registries:
  // a catalog holding the standard tools, and a bootstrap holding the
  // three meta-tools (find_tool, use_tool, run_script). The model sees
  // only the meta-tools in its context and discovers the catalog at
  // runtime via find_tool.
  TDemoChatWithDynTools = class(TBaseDemoChat)
  private
    FCatalog: TToolRegistry;
    FBootstrap: TToolRegistry;
  protected
    procedure DoConfigureChat(); override;
  public
    constructor Create(); override;
    destructor Destroy(); override;
  end;

implementation

uses
  Gemma4.Tools.Utils,
  Gemma4.Common;

{ TDemoChatWithDynTools }

constructor TDemoChatWithDynTools.Create();
begin
  inherited;
  Title := 'Chat: with dynamic tool discovery';
  FCatalog := TToolRegistry.Create();
  FBootstrap := TToolRegistry.Create();
end;

destructor TDemoChatWithDynTools.Destroy();
begin
  FBootstrap.Free();
  FCatalog.Free();
  inherited;
end;

procedure TDemoChatWithDynTools.DoConfigureChat();
begin
  RegisterStandardTools(FCatalog);
  RegisterMetaTools(FBootstrap, FCatalog,
    TGemma4.ResPath(CResPythonExe),
    TGemma4.ResPath(CResScripts));

  Chat.ToolRegistry := FBootstrap;
  Chat.MaxToolRounds := 10;
  Chat.SystemPrompt :=
    'You are Gemma4, a helpful assistant powered by Gemma4.pas. ' +
    'You have access to three meta-tools: find_tool to discover available ' +
    'tools, use_tool to execute them, and run_script to run Python code. ' +
    'Always use find_tool first to discover what tools are available before ' +
    'trying use_tool.';
end;

end.