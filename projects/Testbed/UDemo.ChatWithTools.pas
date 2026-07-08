{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information
===============================================================================}

unit UDemo.ChatWithTools;

interface

uses
  Gemma4.Tools,
  UTestbed.Common;

type
  { TDemoChatWithTools }
  // Interactive console chat with the standard tool catalog registered.
  // The model sees all tools in its context and can call them directly.
  TDemoChatWithTools = class(TBaseDemoChat)
  private
    FTools: TToolRegistry;
  protected
    procedure DoConfigureChat(); override;
  public
    constructor Create(); override;
    destructor Destroy(); override;
  end;

implementation

uses
  Gemma4.Tools.Utils;

{ TDemoChatWithTools }

constructor TDemoChatWithTools.Create();
begin
  inherited;
  Title := 'Chat: with standard tools';
  FTools := TToolRegistry.Create();
end;

destructor TDemoChatWithTools.Destroy();
begin
  FTools.Free();
  inherited;
end;

procedure TDemoChatWithTools.DoConfigureChat();
begin
  RegisterStandardTools(FTools);
  Chat.ToolRegistry := FTools;
end;

end.