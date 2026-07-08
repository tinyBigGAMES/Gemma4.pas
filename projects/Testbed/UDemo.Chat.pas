{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information
===============================================================================}

unit UDemo.Chat;

interface

uses
  UTestbed.Common;

type
  { TDemoChat }
  // Interactive console chat with no tools. Loads the model, opens
  // the chat loop, and exits on /quit.
  TDemoChat = class(TBaseDemoChat)
  public
    constructor Create(); override;
  end;

implementation

{ TDemoChat }

constructor TDemoChat.Create();
begin
  inherited;
  Title := 'Chat: interactive console';
end;

end.