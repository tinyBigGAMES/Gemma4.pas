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
  Gemma4.Compute in '..\..\src\Gemma4.Compute.pas',
  Gemma4.Config in '..\..\src\Gemma4.Config.pas',
  Gemma4.Embeddings in '..\..\src\Gemma4.Embeddings.pas',
  Gemma4.Image in '..\..\src\Gemma4.Image.pas',
  Gemma4.Inference in '..\..\src\Gemma4.Inference.pas',
  Gemma4.Jinja in '..\..\src\Gemma4.Jinja.pas',
  Gemma4.Layers in '..\..\src\Gemma4.Layers.pas',
  Gemma4.Model in '..\..\src\Gemma4.Model.pas',
  Gemma4.Packer in '..\..\src\Gemma4.Packer.pas',
  Gemma4.Quant in '..\..\src\Gemma4.Quant.pas',
  Gemma4.Safetensors in '..\..\src\Gemma4.Safetensors.pas',
  Gemma4.Shaders in '..\..\src\Gemma4.Shaders.pas',
  Gemma4.Tensors in '..\..\src\Gemma4.Tensors.pas',
  Gemma4.Tokenizer in '..\..\src\Gemma4.Tokenizer.pas',
  Gemma4.Types in '..\..\src\Gemma4.Types.pas',
  Gemma4.Video in '..\..\src\Gemma4.Video.pas',
  Gemma4.Vision in '..\..\src\Gemma4.Vision.pas',
  Gemma4.Vulkan in '..\..\src\Gemma4.Vulkan.pas',
  StdApp.Resources in '..\..\src\StdApp.Resources.pas',
  UDemo.Embedding in 'UDemo.Embedding.pas',
  UDemo.Inference in 'UDemo.Inference.pas',
  UDemo.Multimedia in 'UDemo.Multimedia.pas',
  UDemo.Pack in 'UDemo.Pack.pas',
  UTestbed in 'UTestbed.pas',
  UTestbed.Common in 'UTestbed.Common.pas';

begin
  RunTestbed();
end.
