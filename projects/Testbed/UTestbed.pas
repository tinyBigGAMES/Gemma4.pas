{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information
===============================================================================}

unit UTestbed;

interface

procedure RunTestbed();

implementation

uses
  System.SysUtils,
  StdApp.Utils,
  StdApp.Console,
  StdApp.Console.Menu,
  StdApp.TestDemo,
  Gemma4.Inference,
  UTestbed.Common,
  UDemo.Pack,
  UDemo.Inference,
  UDemo.Embedding,
  UDemo.Multimedia,
  UDemo.Chat,
  UDemo.ChatWithTools,
  UDemo.ChatWithDynTools;

procedure Menu();
var
  LMenu: TConsoleMenu;
begin
  LMenu := TConsoleMenu.Create();
  try
    LMenu.Title(Format('Gemma4.pas v%s | Testbed', [TInference.GetVersionStr()]));
    LMenu.Pause := True;
    if TBaseDemoCase.CheckForVPKFiles() then
      LMenu.AddTestDemo(TDemoPack);
    LMenu.AddTestDemo(TDemoInference);
    LMenu.AddTestDemo(TDemoEmbedding);
    LMenu.AddTestDemo(TDemoMultimedia);
    LMenu.AddTestDemo(TDemoChat);
    LMenu.AddTestDemo(TDemoChatWithTools);
    LMenu.AddTestDemo(TDemoChatWithDynTools);
    LMenu.Run();
  finally
    LMenu.Free();
  end;
end;

procedure RunTest(const AProc: TProc);
begin
  AProc();
  if TUtils.RunFromIDE() then
    TConsole.Pause();
end;

procedure RunTestbed();
begin
  try
    Menu();
  except
    on E: Exception do
    begin
      TConsole.PrintLn('');
      TConsole.PrintLn(COLOR_RED + 'EXCEPTION: %s', [E.Message]);

      if TUtils.RunFromIDE() then
        TConsole.Pause();
    end;
  end;
end;

end.
