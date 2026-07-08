<div align="center">

![Gemma4.pas](media/logo.jpg)

[![Discord](https://img.shields.io/discord/1457450179254026250?style=for-the-badge&logo=discord&label=Discord)](https://discord.gg/Wb6z8Wam7p) [![Follow on Bluesky](https://img.shields.io/badge/Bluesky-tinyBigGAMES-blue?style=for-the-badge&logo=bluesky)](https://bsky.app/profile/tinybiggames.com)

</div>

## What is Gemma4.pas?

**A complete multimodal AI inference engine for Google's Gemma 4 E4B, written entirely in Delphi.** No Python, no C/C++, no llama.cpp underneath -- zero third-party dependencies. Load a single VPK archive, call `TInference.Generate`, and stream tokens from your GPU.

```pascal
uses
  Gemma4.Types,
  Gemma4.Inference;

var
  LInf: TInference;
begin
  LInf := TInference.Create();
  try
    LInf.SetTokenCallback(
      procedure(const AState: TProgressState;
        const AToken: string; const AUserData: Pointer)
      begin
        if AState = psInProgress then
          Write(AToken);
      end, nil);

    LInf.LoadModel('C:\Dev\LLM\VPK\Gemma4.vpk');
    LInf.EnableThinking := True;
    LInf.AddMessage(CRoleUser, 'What makes Vulkan different from OpenGL?');
    LInf.Generate(1024);
  finally
    LInf.Free();
  end;
end.
```

Gemma4.pas runs the full Gemma 4 E4B model: text generation with a thinking/reasoning channel, vision, audio, and video, all accelerated through Vulkan compute shaders on the GPU. It also bundles an [EmbeddingGemma-300m](https://huggingface.co/google/embeddinggemma-300m) encoder for semantic search and RAG. The engine loads an abliterated checkpoint -- refusal behavior removed -- so the model stays genuinely helpful for any legitimate use. Everything runs locally. Nothing leaves your machine.

## 🎬 Media

<div align="center">
<br/>  

![Gemma4.pas Infographic](media/infographic.jpg)
  

https://github.com/user-attachments/assets/22e43766-2bc3-478e-b928-633f4ad202b2

</div>

## 🎯 Who is Gemma4.pas For?

- **Delphi and Pascal developers**: integrate local LLM inference into your applications without leaving the Delphi ecosystem or depending on external runtimes.
- **Application developers**: a self-contained, single-file AI engine that runs on consumer hardware with full multimodal support -- text, images, audio, video, and embeddings.
- **Privacy-conscious users**: a model that runs entirely on your own machine with no network calls, no telemetry, and no content filtering.

## ✨ Key Features

- **Text generation**: streaming token output with configurable sampling (temperature, top-k, top-p) and a thinking/reasoning channel where the model reasons privately before answering.
- **Tool use**: the model can call functions via structured tool declarations and a complete Jinja chat template engine.
- **Vision**: describe images at variable detail levels (70 to 1120 soft tokens). VCL Graphics decode -- no third-party image libraries.
- **Audio**: understand speech and sound from WAV files (PCM16/F32, any sample rate). 12-layer conformer encoder.
- **Video**: sample frames across a clip and describe temporal content. Windows Media Foundation frame extraction, reuses the vision tower.
- **Embeddings and RAG**: bundled EmbeddingGemma-300m (24-layer bidirectional, 768-dim output) for semantic search, document retrieval, and memory.
- **Single-file deployment**: one VPK archive holds both models (Gemma 4 E4B + EmbeddingGemma-300m), ready to memory-map and run.
- **4-bit quantization**: Q4_0 decoder weights with DP4A integer dot products; vision/audio/embedding encoders stay at full F32 precision.
- **Vulkan compute**: all matrix math, attention, softmax, and normalization run on the GPU as compiled SPIR-V shaders.
- **Zero dependencies**: pure Delphi -- no DLLs, no Python, no C runtime.

### Application layer (optional)

Built on top of the engine, and easy to ignore if you only need generation:

- **Tool calling**: a fluent `TToolRegistry` plus a ready-made catalog (weather, web search, files, Python) so the model can call functions and act on the results.
- **Persistent memory**: `TMemory` -- a SQLite conversation archive with FTS5 keyword search fused with HNSW vector recall (backed by the bundled embeddings model).
- **Infinite memory**: old turns are summarized before eviction, so a conversation can run indefinitely without the context simply falling off a cliff.
- **Conversation policy**: `TSession` drives the generate -> execute -> regenerate agentic loop, recall injection, and history budgeting in a single `Ask` call.
- **Interactive chat**: `TChat` / `TConsoleChat` -- an abstract chat loop with a console frontend and a full set of slash commands.
- **KV state save/load**: persist and restore the GPU KV cache (`.g4kv`) to resume a conversation with a warm prefix.

## 🚀 Getting Started

### Download the Model

Download the pre-packed VPK archive from HuggingFace:

**[📦 Download Gemma4.vpk](https://huggingface.co/buckets/tinybiggames/Gemma4.pas/resolve/Gemma4.vpk?download=true)**

Place the file at `C:\Dev\LLM\VPK\Gemma4.vpk`. This is the default path the testbed demos look for. If you place it somewhere else, change the path constant in your code.

### Run the Testbed

1. Clone the repository: `git clone https://github.com/tinyBigGAMES/Gemma4.pas.git`
2. Open `projects\Testbed\Testbed.dproj` in Delphi 12 Athens or higher
3. Build and run (Win64 target)
4. The testbed menu offers four demos: Pack, Inference, Embedding, and Multimedia

### Minimal Example

```pascal
LInf := TInference.Create();
try
  LInf.LoadModel('C:\Dev\LLM\VPK\Gemma4.vpk');
  LInf.EnableThinking := True;
  LInf.AddMessage(CRoleUser, 'Hello, world!');
  LInf.Generate(1024);
  WriteLn(LInf.ResponseText);
finally
  LInf.Free();
end;
```

The model's chat template is applied automatically -- you never construct prompt text by hand. For multi-turn conversations, feed the response back:

```pascal
LInf.AddMessage(CRoleAssistant, LInf.ResponseText);
LInf.AddMessage(CRoleUser, 'Tell me more.');
LInf.Generate(1024);
```

### Multimodal

Image, audio, and video are first-class message parts:

```pascal
// Image (image before text per model card)
LInf.AddMessage(CRoleUser, [
  TMessagePart.ImagePart('photo.png'),
  TMessagePart.TextPart('Describe this image.')
]);
LInf.Generate(512);

// Audio (text before audio per model card)
LInf.AddMessage(CRoleUser, [
  TMessagePart.TextPart('Describe this audio.'),
  TMessagePart.AudioPart('speech.wav')
]);
LInf.Generate(512);

// Video (video before text per model card)
LInf.AddMessage(CRoleUser, [
  TMessagePart.VideoPart('clip.mp4'),
  TMessagePart.TextPart('Describe this video.')
]);
LInf.Generate(512);
```

### Embeddings

```pascal
LEmb := TEmbeddings.Create();
try
  LEmb.Open('C:\Dev\LLM\VPK\Gemma4.vpk');
  LQuery := LEmb.EmbedQuery('Which planet is the Red Planet?');
  LDoc := LEmb.EmbedDocument('Mars is often called the Red Planet.');
  WriteLn(TEmbeddings.Similarity(LQuery, LDoc):0:4);
finally
  LEmb.Free();
end;
```

## 🧰 Application Layer

Turn the raw engine into an agent or an assistant. Register tools, hand them to a `TSession`, and one `Ask` call runs the full generate -> execute -> regenerate loop:

```pascal
uses
  Gemma4.Inference, Gemma4.Tools, Gemma4.Tools.Utils, Gemma4.Session;

LInf := TInference.Create();
LTools := TToolRegistry.Create();
LSession := TSession.Create();
try
  LInf.LoadModel('C:\Dev\LLM\VPK\Gemma4.vpk');
  RegisterStandardTools(LTools);          // weather, web_search, files, ...

  LSession.SetInference(LInf);
  LSession.SetToolRegistry(LTools);
  WriteLn(LSession.Ask('What is the weather in Tokyo?', 1024));
finally
  LSession.Free(); LTools.Free(); LInf.Free();
end;
```

Attach a `TMemory` for persistent recall (SQLite + FTS5 + HNSW), or skip straight to `TConsoleChat` for a full interactive assistant with slash commands (`/addfact`, `/addfile`, `/summary`, `/state`, `/restore`, ...):

```pascal
LChat := TConsoleChat.Create();
try
  LChat.ModelPath := 'C:\Dev\LLM\VPK\Gemma4.vpk';
  LChat.MemoryDbPath := 'res\database\chat.db';
  LChat.Run();          // blocks until the user types /quit
finally
  LChat.Free();
end;
```

> [!NOTE]
> The `web_search` tool uses the [Tavily](https://tavily.com) API. It is optional -- everything else works with no key. To enable it, create a free Tavily key (1000 credits/month; see [pricing](https://tavily.com/#pricing)) and set the `TAVILY_API_KEY` environment variable.

## ⚡ Performance

Measured on an NVIDIA RTX 3060 (12 GB VRAM):

| Metric | Value |
|--------|-------|
| Text generation | ~57 tokens/sec |
| Prefill (5972-token prompt) | ~223 tokens/sec |
| Vision encoder parity | ~9.5e-5 max abs diff vs HuggingFace |
| Audio encoder parity | ~1.96e-4 max abs diff vs HuggingFace |

## 📖 Documentation

The full API reference, architecture guide, and how-to recipes are in a single document:

| Document | Description |
|----------|-------------|
| **[Gemma4.pas Documentation](docs/Gemma4-pas.md)** | Complete tour: `TInference` API (loading, messages, generation, thinking, multimodal, stats), `TEmbeddings` API, the application layer (tools, `TMemory`, `THNSW`, `TSession`, `TChat`, KV state), architecture (42-layer decoder, Vulkan pipeline, Q4_0 quantization, VPK format, vision/audio/video encoders, embeddings model), and practical how-to recipes with verified testbed code. |

## 🗂️ Module Map

**Inference engine**

| Unit | Role |
|------|------|
| `Gemma4.Types` | Shared types, records, constants, enums |
| `Gemma4.Config` | Model config loader (config.json) |
| `Gemma4.Tokenizer` | BPE tokenizer (262144-token vocabulary) |
| `Gemma4.Tensors` | Tensor storage, views, basic ops |
| `Gemma4.Quant` | Q4_0 quantization and dequantization |
| `Gemma4.Safetensors` | Safetensors header/tensor parser |
| `Gemma4.Packer` | Offline packing tool: safetensors -> VPK |
| `Gemma4.Attention` | RoPE, GQA, sliding + full attention, KV cache |
| `Gemma4.Layers` | RMSNorm, GeLU MLP, residual connections |
| `Gemma4.Model` | Forward pass orchestration, generation loop |
| `Gemma4.Vulkan` | Vulkan device, queues, buffer management |
| `Gemma4.Shaders` | SPIR-V loading and pipeline cache |
| `Gemma4.Compute` | GPU kernel dispatch (GEMM, softmax, RoPE) |
| `Gemma4.Jinja` | Full Jinja chat template engine |
| `Gemma4.Image` | Image decode, resize, patchify (VCL Graphics) |
| `Gemma4.Vision` | SigLIP vision encoder (16 layers, F32 GPU) |
| `Gemma4.Audio` | Conformer audio encoder (12 layers, F32 GPU) |
| `Gemma4.Video` | Video frame extraction (Windows Media Foundation) |
| `Gemma4.Embeddings` | EmbeddingGemma-300m bidirectional encoder |
| `Gemma4.Inference` | Top-level API: messages, template, generate, stream |

**Application layer** (optional, builds on `Gemma4.Inference`)

| Unit | Role |
|------|------|
| `Gemma4.Common` | Shared resource-path resolution and constants |
| `Gemma4.Tools` | Tool registry, fluent builder, schema generation, dispatch |
| `Gemma4.Tools.Utils` | Standard tool catalog, meta-tools, REST client, local ops |
| `Gemma4.HNSW` | HNSW approximate nearest-neighbor vector index |
| `Gemma4.Memory` | SQLite conversation archive (FTS5 + HNSW retrieval) |
| `Gemma4.Session` | Conversation policy, recall, tool loop, infinite memory |
| `Gemma4.Chat` | Abstract interactive chat loop + console frontend |

## 🔨 Building from Source

### Prerequisites

| | Requirement |
|---|---|
| **Host OS** | Windows x64 |
| **GPU** | NVIDIA with Vulkan compute (tested on RTX 3060 12 GB) |
| **Compiler** | Delphi 12 Athens or higher |

### Get the Source

```bash
git clone https://github.com/tinyBigGAMES/Gemma4.pas.git
```

### Compile

1. Open `projects\Testbed\Testbed.dproj` in Delphi 12 Athens or higher
2. Build (Win64 target)
3. Download [Gemma4.vpk](https://huggingface.co/buckets/tinybiggames/Gemma4.pas/resolve/Gemma4.vpk?download=true) to `C:\Dev\LLM\VPK\Gemma4.vpk`
4. Run the testbed -- it offers Inference, Embedding, Multimedia, and Pack demos

## 🤝 Contributing

Gemma4.pas is an open project and contributions are welcome at every level:

- **Report bugs**: open an issue with a minimal reproduction, the exact code used, and your GPU model.
- **Suggest features**: describe the use case first, then the proposed API or behavior.
- **Submit pull requests**: bug fixes, documentation improvements, new test cases, and well-scoped features.

Join our [Discord](https://discord.gg/Wb6z8Wam7p) to discuss development, ask questions, or share what you are building with Gemma4.pas.

## 💙 Support the Project

If Gemma4.pas saves you time, sparks an idea, or becomes part of something you ship:

- :star: **Star the repo**: it costs nothing and helps others find the project
- :speech_balloon: **Spread the word**: write a post, mention it in a community you are part of
- :busts_in_silhouette: **[Join us on Discord](https://discord.gg/Wb6z8Wam7p)**: share what you are building and help shape what comes next
- :sparkling_heart: **[Become a sponsor](https://github.com/sponsors/tinyBigGAMES)**: sponsorship directly funds development and documentation
- :butterfly: **[Follow on Bluesky](https://bsky.app/profile/tinybiggames.com)**: stay in the loop on releases and development

## 📄 License

Gemma4.pas is licensed under the **Apache License 2.0**. See [LICENSE](https://github.com/tinyBigGAMES/Gemma4.pas?tab=License-1-ov-file#) for details.

Apache 2.0 is a permissive open source license that lets you use, modify, and distribute StdApp freely in both open source and commercial projects. You are not required to release your own source code. The license includes an explicit patent grant. Attribution is required; keep the copyright notice and license file in place.

## 🔗 Links

- [GitHub](https://github.com/tinyBigGAMES/Gemma4.pas)
- [Discord](https://discord.gg/Wb6z8Wam7p)
- [Bluesky](https://bsky.app/profile/tinybiggames.com)
- [tinyBigGAMES](https://tinybiggames.com)

<div align="center">

**Gemma4.pas**&#8482; - Local LLM inference in Pascal.

Copyright &copy; 2026-present tinyBigGAMES&#8482; LLC
All Rights Reserved.

</div>
