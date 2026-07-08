{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information
===============================================================================}

program Testbed;

{$APPTYPE CONSOLE}

{$R *.res}

uses
  StdApp.VMM,
  System.SysUtils,
  Gemma4.Attention in '..\..\src\Gemma4.Attention.pas',
  Gemma4.Audio in '..\..\src\Gemma4.Audio.pas',
  Gemma4.Common in '..\..\src\Gemma4.Common.pas',
  Gemma4.Compute in '..\..\src\Gemma4.Compute.pas',
  Gemma4.Config in '..\..\src\Gemma4.Config.pas',
  Gemma4.Embeddings in '..\..\src\Gemma4.Embeddings.pas',
  Gemma4.HNSW in '..\..\src\Gemma4.HNSW.pas',
  Gemma4.Image in '..\..\src\Gemma4.Image.pas',
  Gemma4.Inference in '..\..\src\Gemma4.Inference.pas',
  Gemma4.Jinja in '..\..\src\Gemma4.Jinja.pas',
  Gemma4.Layers in '..\..\src\Gemma4.Layers.pas',
  Gemma4.Memory in '..\..\src\Gemma4.Memory.pas',
  Gemma4.Model in '..\..\src\Gemma4.Model.pas',
  Gemma4.Packer in '..\..\src\Gemma4.Packer.pas',
  Gemma4.Quant in '..\..\src\Gemma4.Quant.pas',
  Gemma4.Safetensors in '..\..\src\Gemma4.Safetensors.pas',
  Gemma4.Shaders in '..\..\src\Gemma4.Shaders.pas',
  Gemma4.Tensors in '..\..\src\Gemma4.Tensors.pas',
  Gemma4.Tokenizer in '..\..\src\Gemma4.Tokenizer.pas',
  Gemma4.Tools in '..\..\src\Gemma4.Tools.pas',
  Gemma4.Tools.Utils in '..\..\src\Gemma4.Tools.Utils.pas',
  Gemma4.Types in '..\..\src\Gemma4.Types.pas',
  Gemma4.Video in '..\..\src\Gemma4.Video.pas',
  Gemma4.Vision in '..\..\src\Gemma4.Vision.pas',
  Gemma4.Vulkan in '..\..\src\Gemma4.Vulkan.pas',
  StdApp.Resources in '..\..\src\StdApp.Resources.pas',
  Gemma4.Session in '..\..\src\Gemma4.Session.pas',
  Gemma4.Chat in '..\..\src\Gemma4.Chat.pas',
  UDemo.Chat in 'UDemo.Chat.pas',
  UDemo.ChatWithDynTools in 'UDemo.ChatWithDynTools.pas',
  UDemo.ChatWithTools in 'UDemo.ChatWithTools.pas',
  UDemo.Embedding in 'UDemo.Embedding.pas',
  UDemo.Inference in 'UDemo.Inference.pas',
  UDemo.Multimedia in 'UDemo.Multimedia.pas',
  UDemo.Pack in 'UDemo.Pack.pas',
  UTestbed.Common in 'UTestbed.Common.pas',
  UTestbed in 'UTestbed.pas';

begin
  RunTestbed();
end.
