{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information

 -------------------------------------------------------------------------------

  Gemma4.Common - Shared resource paths and application constants

  Foundation helpers shared across the application layer (tools, memory,
  session, chat):
    - File-extension and resource-path constants (database, python, scripts)
    - Conversation-state file extension (g4state)
    - TGemma4 static class: resolves relative resource paths against the
      executable directory or an explicit base folder (SetResFolder / ResPath)

  Dependencies: StdApp.Utils
===============================================================================}

unit Gemma4.Common;

{$I StdApp.Defines.inc}

interface

uses
  StdApp.Utils;

const
  { File extensions }
  CExtDb    = 'db';
  CExtJson  = 'json';
  CExtState = 'g4state';

  { Resource paths (relative to exe directory) }
  CResDatabase  = 'res\database';
  CResPythonExe = 'res\python\python.exe';
  CResScripts   = 'res\scripts';

type
  { TGemma4 }
  // Shared path resolution for the Gemma4 runtime. All resource paths
  // are resolved relative to the executable directory (or an explicit
  // base folder supplied to SetResFolder).
  TGemma4 = class
  private class var
    FResFolder: string;
  public
    // Set the resource base folder. Empty = exe directory (default).
    class procedure SetResFolder(const AFolder: string); static;

    // Resolve a relative resource path to an absolute path.
    class function ResPath(const ARelativePath: string): string; static;
  end;

implementation

{ TGemma4 }

class procedure TGemma4.SetResFolder(const AFolder: string);
begin
  FResFolder := AFolder;
end;

class function TGemma4.ResPath(const ARelativePath: string): string;
begin
  Result := TUtils.AppBasedPath(ARelativePath, FResFolder);
end;

end.
