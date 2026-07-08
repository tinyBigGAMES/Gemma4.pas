<div align="center">

![Gemma4.pas](../media/logo.jpg)

</div>

<a id="what-is-gemma4pas"></a>

## 🚀 What is Gemma4.pas?

**Gemma4.pas** is a complete inference engine for Google's [Gemma 4 E4B](https://ai.google.dev/gemma) multimodal model, written entirely in Delphi (Object Pascal). No Python, no C/C++, no llama.cpp underneath -- zero third-party dependencies. Just Delphi and your GPU.

It runs the full model: text generation with a thinking/reasoning channel, vision (image description), audio (speech understanding), and video (temporal scene description), with all heavy math accelerated through Vulkan compute shaders. It also ships a bundled [EmbeddingGemma-300m](https://huggingface.co/google/embeddinggemma-300m) encoder for semantic search and retrieval-augmented generation (RAG).

The engine loads an abliterated Gemma 4 E4B checkpoint -- the trained-in refusal behavior has been removed, so the model stays genuinely helpful for any legitimate use. Everything runs locally on your machine. Nothing you ask ever leaves your computer.

```
Download Gemma4.vpk  ->  TInference.LoadModel  ->  AddMessage  ->  Generate  ->  tokens
```

> [!TIP]
> 💡 **Fast path:** read [Getting Started](#getting-started), then skim the [API Reference](#api-reference) and [How-To Guide](#how-to-guide).

### 🚦 Documentation Roadmap

| Reader Goal | Start Here | Why |
|-------------|------------|-----|
| 🚀 Run your first inference | [Getting Started](#getting-started) | Download the model, load it, generate your first response |
| 🔌 Learn the API | [API Reference](#api-reference) | `TInference` public surface: loading, messages, generation, thinking, multimodal, stats |
| 🧰 Build an agent or assistant | [Application Layer](#application-layer) | Tools, persistent memory, conversation policy, interactive chat, KV state |
| 🧪 Solve a task | [How-To Guide](#how-to-guide) | Practical recipes: streaming chat, multi-turn, image/audio/video, embeddings, tools, memory |
| 🏗️ Understand the internals | [Architecture](#architecture) | Model structure, Vulkan pipeline, quantization, VPK format, encoders |
### 💡 Why Gemma4.pas?

Almost every serious language model engine is written in C or C++. The Delphi ecosystem had nothing. Gemma4.pas proves it can be done: a production-quality, competitive inference engine built natively in Delphi, with no external libraries. Every piece -- the tokenizer, the quantizer, the Vulkan compute layer, the Jinja template engine -- was built from scratch in Pascal.

The bar was set high on purpose: match llama.cpp, the tool everyone benchmarks against.

> [!IMPORTANT]
> 🧱 This is not a toy and not a thin wrapper around someone else's work. Every component was hand-built in Delphi from the ground up.

### ✨ Key Features

| Feature | What It Means |
|---------|---------------|
| **💬 Text generation** | Streaming token output with configurable sampling (temperature, top-k, top-p) |
| **🧠 Thinking channel** | The model reasons privately before answering; reasoning is optionally visible |
| **🔧 Tool use** | The model can call functions via structured tool declarations |
| **👁️ Vision** | Describe images at variable detail levels (70 to 1120 soft tokens) |
| **🎙️ Audio** | Understand speech and sound from WAV files |
| **🎬 Video** | Sample frames across a clip and describe temporal content |
| **🔍 Embeddings and RAG** | EmbeddingGemma-300m for semantic search, document retrieval, and memory |
| **🧾 Jinja chat template** | Every prompt matches exactly how the model was trained |
| **📦 4-bit quantization** | Q4_0 decoder weights for compact storage; encoders stay at full F32 precision |
| **📁 Single-file deployment** | One VPK archive holds both models, ready to memory-map and run |
| **⚡ Vulkan compute** | All matrix math, attention, softmax, and normalization run on the GPU |
| **🧰 Zero dependencies** | Pure Delphi -- no DLLs, no Python, no C runtime |
| **🗂️ Tool catalog** | Ready-made tools (weather, web search, files, Python) plus a fluent registry for your own |
| **🗄️ Persistent memory** | SQLite conversation archive with FTS5 keyword + HNSW vector recall |
| **♾️ Infinite memory** | Old turns are summarized before eviction, so context never simply falls off a cliff |
| **💻 Interactive chat** | Abstract chat loop with a console frontend and a full set of slash commands |
| **💾 KV state save/load** | Persist and restore the GPU KV cache to resume a conversation with a warm prefix |

### 🏗️ Architecture at a Glance

```
Gemma4.vpk (single archive)
    |
    +-- E4B/ (Gemma 4 text + multimodal)
    |     weights.bin      Q4_0 decoder (42 layers, 2560 hidden)
    |     encoders.bin     F32 vision (16L) + audio (12L conformer)
    |     tokenizer.json   262144-token BPE vocabulary
    |     chat_template.jinja
    |
    +-- embeddings/ (EmbeddingGemma-300m)
          weights.bin      F32 bidirectional encoder (24L, 768 hidden)
          tokenizer.json

    TInference.LoadModel('Gemma4.vpk')
        |
        v
    Memory-map weights --> Vulkan GPU upload
        |
        v
    AddMessage (text / image / audio / video parts)
        |
        v
    Generate --> Jinja template --> tokenize --> GPU forward pass
        |
        v
    Stream tokens via callback
```

The text decoder is a 42-layer transformer with a hybrid attention pattern: five sliding-window layers followed by one full-attention layer, repeated seven times. It uses grouped-query attention (8 query heads, 2 KV heads) with a shared KV cache across 18 layers. Per-layer embeddings (PLE) provide richer representations at each layer.

On top of the language model sit the encoders: a 16-layer SigLIP vision tower for images, a 12-layer conformer for audio, and frame extraction for video that reuses the vision tower.

> [!NOTE]
> 🧩 The two models -- Gemma 4 E4B (text + multimodal) and EmbeddingGemma-300m (semantic embeddings) -- are bundled into a single VPK archive. One file to download, one file to load.
### ⚡ Performance

Measured on an NVIDIA RTX 3060 (12 GB VRAM):

| Metric | Value |
|--------|-------|
| 💬 Text generation | ~57 tokens/sec |
| 📥 Prefill (long prompt) | ~223 tokens/sec (5972-token input) |
| 👁️ Vision encoder parity | max abs diff ~9.5e-5 vs HuggingFace reference |
| 🎙️ Audio encoder parity | max abs diff ~1.96e-4 vs HuggingFace reference |

### 🎯 Who Is This For?

- **🖥️ Delphi and Pascal developers** who want to integrate local LLM inference into their applications without leaving the Delphi ecosystem or depending on external runtimes.
- **📦 Application developers** who need a self-contained, single-file AI engine that runs on consumer hardware with full multimodal support.
- **🔒 Privacy-conscious users** who want a model that runs entirely on their own machine with no network calls, no telemetry, and no content filtering.

### 📌 Current Status

The engine is feature-complete across all planned modalities:

- 💬 Text generation with thinking/reasoning channel
- 🔧 Tool use via structured tool declarations and the Jinja template engine
- 👁️ Vision (image description at variable detail levels)
- 🎙️ Audio (speech and sound understanding)
- 🎬 Video (temporal scene description via frame sampling)
- 🔍 Embeddings (EmbeddingGemma-300m semantic search and retrieval)
- ⚡ Batched prefill (~223 t/s) and optimized generation (~57 t/s)
- ✅ Byte-exact template rendering verified against HuggingFace `apply_chat_template`
- 🧰 Application layer: tool registry and catalog, persistent memory (SQLite + FTS5 + HNSW), conversation policy with infinite memory, interactive chat, and KV state save/load

> [!TIP]
> 💡 The exact public API always lives in `Gemma4.Inference.pas`. When the docs and the source ever seem to disagree, the source is authoritative.

### 💻 System Requirements

| Area | Requirement |
|------|-------------|
| **🪶 Operating system** | Windows x64 |
| **🖥️ GPU** | NVIDIA GPU with Vulkan compute support (tested on RTX 3060 12 GB) |
| **💾 VRAM** | 12 GB recommended |
| **📦 Runtime dependencies** | None |
| **🔧 Building from source** | Delphi 12 Athens or higher |

### 🗺️ Table of Contents

- 🚀 [Getting Started](#getting-started): download the model, load it, run your first inference
- 🔌 [API Reference](#api-reference): `TInference` public surface, `TEmbeddings`, types, callbacks, constants
- 🧰 [Application Layer](#application-layer): tools, `TMemory`, `THNSW`, `TSession`, `TChat`, KV state save/load
- 🏗️ [Architecture](#architecture): model internals, Vulkan pipeline, quantization, VPK format, encoders
- 🧪 [How-To Guide](#how-to-guide): practical recipes with complete code from the testbed

<a id="getting-started"></a>

## 🚀 Getting Started

Gemma4.pas ships as Delphi source code. You compile it with Delphi 12 Athens or higher, download the pre-packed model archive, and run inference with a few lines of code.

> [!NOTE]
> 🪶 Gemma4.pas targets Windows x64. The Vulkan compute shaders run on NVIDIA GPUs with up-to-date drivers -- no separate Vulkan SDK install is needed at runtime.

### ⚙️ Prerequisites

| Requirement | Details |
|-------------|---------|
| **🔧 Delphi** | 12 Athens or higher |
| **🖥️ GPU** | NVIDIA GPU with Vulkan compute support (tested on RTX 3060 12 GB) |
| **💾 Disk space** | ~5 GB for the Gemma4.vpk model archive |
| **📦 Vulkan driver** | Ships with modern NVIDIA drivers -- no separate SDK install needed at runtime |
| **🔑 Tavily API key** | Optional. Only needed for the `web_search` tool in the application layer (see the note below). |

> [!NOTE]
> 🔑 The `web_search` standard tool uses the [Tavily](https://tavily.com) API. It is entirely optional -- everything else (text, vision, audio, video, embeddings, memory, and every other tool) works with no API key. If you want web search, create a free Tavily key (the free tier includes 1000 credits/month; see [tavily.com/#pricing](https://tavily.com/#pricing)) and set the `TAVILY_API_KEY` environment variable before running.

### ⬇️ Step 1: Get the Source

Clone the repository:

```
git clone https://github.com/tinyBigGAMES/Gemma4.pas.git
```

### 📥 Step 2: Download the Model

Download the pre-packed VPK archive from HuggingFace:

[📦 Download Gemma4.vpk](https://huggingface.co/buckets/tinybiggames/Gemma4.pas/resolve/Gemma4.vpk?download=true)

Place the file at `C:\Dev\LLM\VPK\Gemma4.vpk`. This is the default path the testbed demos look for. If you place it somewhere else, change the path constant in your code.

> [!IMPORTANT]
> 🧱 The VPK archive contains both models: Gemma 4 E4B (text + multimodal) and EmbeddingGemma-300m (semantic embeddings). One file, one download.

### ✍️ Step 3: Your First Inference

Open the testbed project at `projects\Testbed\Testbed.dproj` in the Delphi IDE and build it. Or write the following in your own project:

```pascal
uses
  Gemma4.Types,
  Gemma4.Inference;

var
  LInf: TInference;
  LStats: TInferenceStats;
begin
  LInf := TInference.Create();
  try
    // Wire up streaming output
    LInf.SetTokenCallback(
      procedure(const AState: TProgressState;
        const AToken: string; const AUserData: Pointer)
      begin
        if AState = psInProgress then
          Write(AToken);
      end, nil);

    // Load the model
    if not LInf.LoadModel('C:\Dev\LLM\VPK\Gemma4.vpk') then
    begin
      WriteLn('LoadModel failed');
      LInf.PrintErrors();
      Exit;
    end;

    // Enable thinking (model reasons before answering)
    LInf.EnableThinking := True;
    LInf.ShowThinking := True;

    // Add a user message and generate
    LInf.AddMessage(CRoleUser,
      'In two sentences, what makes the Vulkan API different from OpenGL?');

    if not LInf.Generate(1024) then
    begin
      WriteLn('Generation failed');
      LInf.PrintErrors();
      Exit;
    end;

    WriteLn;

    // Print generation statistics
    LStats := LInf.Stats;
    WriteLn(LStats.FormatText(sdkBasic));
  finally
    LInf.Free();
  end;
end.
```

That is the whole loop: load the VPK, add a message, generate, read the streamed tokens. The model's chat template is applied automatically -- you never construct prompt text by hand.

> [!TIP]
> 💡 `Generate()` applies the model's real Jinja chat template over your message history. You work with messages, not raw prompt strings.

### 🔁 Step 4: Multi-Turn Conversation

To continue a conversation, feed the model's response back into the history:

```pascal
// First turn
LInf.AddMessage(CRoleUser, 'What is Vulkan?');
LInf.Generate(1024);
LInf.AddMessage(CRoleAssistant, LInf.ResponseText);

// Second turn (depends on the first)
LInf.AddMessage(CRoleUser, 'Give one concrete example.');
LInf.Generate(1024);
```

`ResponseText` contains the visible reply only (thinking stripped). `Response` contains the full text including thinking markers.

### 🔍 Step 5: Your First Embedding

```pascal
uses
  Gemma4.Embeddings;

var
  LEmb: TEmbeddings;
  LQueryVec: TArray<Single>;
  LDocVec: TArray<Single>;
  LSim: Single;
begin
  LEmb := TEmbeddings.Create();
  try
    if not LEmb.Open('C:\Dev\LLM\VPK\Gemma4.vpk') then
    begin
      WriteLn('Open failed');
      LEmb.PrintErrors();
      Exit;
    end;

    LQueryVec := LEmb.EmbedQuery('Which planet is the Red Planet?');
    LDocVec := LEmb.EmbedDocument(
      'Mars is often referred to as the Red Planet.');
    LSim := TEmbeddings.Similarity(LQueryVec, LDocVec);

    WriteLn('Similarity: ', LSim:0:4);
  finally
    LEmb.Free();
  end;
end.
```

`EmbedQuery` and `EmbedDocument` apply the model's exact trained prompt prefixes. `Similarity` computes cosine similarity between two normalized embeddings.

### 🧯 Common First-Run Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| ❌ `LoadModel failed` | VPK file not found at the specified path | Verify the path to `Gemma4.vpk`; the default is `C:\Dev\LLM\VPK\Gemma4.vpk` |
| ❌ Vulkan initialization failure | GPU driver does not support Vulkan compute | Update your NVIDIA driver to the latest version |
| ❌ Out of VRAM | GPU has less than ~4 GB free | Close other GPU-intensive applications; 12 GB VRAM recommended |
| ⚠️ Very slow generation | Weights fell back to host-visible memory | Ensure enough contiguous VRAM is free for the ~2.7 GB weight upload |
| ⚠️ No output from token callback | Callback not wired before `Generate` | Call `SetTokenCallback` before `Generate` |

<a id="api-reference"></a>

## 🔌 API Reference

The primary user-facing class is `TInference` in `Gemma4.Inference.pas`. It handles model loading, conversation management, chat template rendering, generation, and streaming output. For simple use, everything flows through this single class.

On top of the engine sits an optional [Application Layer](#application-layer): tool calling (`TToolRegistry`), persistent memory (`TMemory` + `THNSW`), conversation policy (`TSession`), an interactive chat loop (`TChat` / `TConsoleChat`), and KV-state save/load. Skip it if all you need is generation; reach for it when you want agents, recall, or an interactive assistant.

### 💬 TInference

Declared in `Gemma4.Inference.pas`. Extends `TBaseObject`.

#### 📦 Loading and Lifecycle

```pascal
function LoadModel(const AVpkPath: string;
  const AUseGPU: Boolean = True): Boolean;
```

Opens a VPK archive, loads the model config, tokenizer, weights, chat template, and initializes Vulkan compute. Returns `True` on success. On failure, call `PrintErrors()` for diagnostics.

```pascal
procedure UnloadModel();
function IsLoaded(): Boolean;
```

Release all resources, or check whether a model is currently loaded.

#### 📝 Conversation History

```pascal
procedure AddMessage(const ARole: string; const AContent: string);
```

Appends a text-only message to the conversation history. Use the role constants `CRoleUser`, `CRoleAssistant`, `CRoleSystem`, or `CRoleTool`.

```pascal
procedure AddMessage(const ARole: string;
  const AParts: TArray<TMessagePart>);
```

Appends a multimodal message. Each `TMessagePart` is a text, image, audio, or video segment. Parts render in order through the chat template.

```pascal
procedure AddToolResult(const AToolName: string; const AContent: string);
```

Appends a tool-result turn (role `'tool'`). `AContent` is the tool's output text.

```pascal
procedure ClearMessages();
procedure ResetConversation();
```

`ClearMessages` empties the message history. `ResetConversation` also clears the KV cache.

#### 🔧 Tool Definitions

```pascal
procedure SetTools(const AToolsJson: string);
```

Registers tool schemas for the template's `tools` block. Pass a JSON array of HuggingFace-style tool schemas. Pass `''` to clear.

#### ⚡ Generation

```pascal
function Generate(const AMaxTokens: Integer = 2048): Boolean;
```

Renders the chat template over the full message history, tokenizes, runs the GPU forward pass, and streams tokens through the token callback. Returns `True` on success.

After generation:
- `Response` -- the full normalized text, including `<thinking>`/`</thinking>` markers
- `ResponseText` -- visible text only, with thinking content stripped

> [!TIP]
> 💡 Feed `ResponseText` back into the history with `AddMessage(CRoleAssistant, LInf.ResponseText)` to maintain multi-turn context.

```pascal
function RenderPrompt(): string;
```

Renders the chat template without generating -- the exact prompt text that `Generate` would tokenize. Useful for debugging and template verification.

#### 🔔 Callbacks

```pascal
procedure SetTokenCallback(const ACallback: TTokenCallback;
  const AUserData: Pointer = nil);
```

Streaming callback for generated display text. The callback signature:

```pascal
TTokenCallback = reference to procedure(
  const AState: TProgressState;
  const AToken: string;
  const AUserData: Pointer);
```

| State | When It Fires |
|-------|---------------|
| `psPrep` | Before template rendering |
| `psStart` | Before the generation loop |
| `psInProgress` | Per emitted display fragment |
| `psEnd` | Generation complete |

```pascal
procedure SetCancelCallback(const ACallback: TCancelCallback;
  const AUserData: Pointer = nil);
```

Polled during generation. Return `True` from the callback to stop cleanly:

```pascal
TCancelCallback = reference to function(
  const AUserData: Pointer): Boolean;
```

```pascal
procedure SetLoadProgressCallback(
  const ACallback: TLoadProgressCallback;
  const AUserData: Pointer = nil);
```

Reports model loading progress (weight upload to GPU). The callback signature:

```pascal
TLoadProgressCallback = reference to procedure(
  const AState: TProgressState;
  const AStep: Integer;
  const ATotal: Integer;
  const AUserData: Pointer);
```

#### 🎲 Sampling Parameters

```pascal
procedure SetTemperature(const AValue: Single);
procedure SetTopK(const AValue: Integer);
procedure SetTopP(const AValue: Single);
property SamplingParams: TSamplingParams
  read FSamplingParams write FSamplingParams;
```

#### 🧠 Thinking Channel

```pascal
property EnableThinking: Boolean
  read FEnableThinking write FEnableThinking;
```

When `True`, the chat template injects the `<|think|>` token that activates the model's reasoning mode. The model generates a private reasoning span before the visible answer.

```pascal
property ShowThinking: Boolean
  read FShowThinking write FShowThinking;
```

When `False`, the reasoning span is hidden from the token callback behind `ThinkingPlaceholderText` (displayed once per span). `Response` and `ResponseText` are unaffected -- they always contain the full text.

```pascal
property ThinkingOpenText: string
  read FThinkOpenText write FThinkOpenText;
property ThinkingCloseText: string
  read FThinkCloseText write FThinkCloseText;
property ThinkingPlaceholderText: string
  read FThinkPlaceholder write FThinkPlaceholder;
```

Customize the markers emitted around thinking spans. Defaults: `<thinking>\n` / `\n</thinking>\n` / `Thinking...\n`. Set both open and close to `''` to strip markers entirely.

> [!NOTE]
> 🧩 The thinking channel normalizes Gemma 4's native markers (`<|channel>thought` / `<channel|>`) to the standard `<thinking>` / `</thinking>` format automatically.

#### 📊 Info and Stats

```pascal
function GetModelName(): string;
function GetDeviceName(): string;
function GetVocabSize(): Integer;
property Stats: TInferenceStats read FStats;
```

`Stats` is populated after each `Generate` call. Use `Stats.FormatText(sdkBasic)` for the standard one-line report or `Stats.FormatText(sdkDetailed)` for the full CPU/GPU phase split.

---

### 🧩 TMessagePart

Declared in `Gemma4.Inference.pas`. One content segment of a multimodal message.

```pascal
TPartKind = (pkText, pkImage, pkAudio, pkVideo);

TMessagePart = record
  Kind: TPartKind;
  Text: string;         // text content (pkText only)
  SourcePath: string;   // file path (media parts only)
  SoftBudget: Integer;  // image: soft-token count; video: frame count
end;
```

Static constructors:

```pascal
class function TextPart(const AText: string): TMessagePart; static;
class function ImagePart(const APath: string;
  const ASoftBudget: Integer = 1120): TMessagePart; static;
class function AudioPart(const APath: string): TMessagePart; static;
class function VideoPart(const APath: string;
  const AFrameCount: Integer = 32): TMessagePart; static;
```

> [!TIP]
> 💡 Image soft-token budgets: 70, 140, 280, 560, 1120. Higher budgets give more detail at the cost of more tokens. Video frame counts range from 1 to 60.

---

### 🔍 TEmbeddings

Declared in `Gemma4.Embeddings.pas`. Text embedding engine using the bundled EmbeddingGemma-300m model.

```pascal
function Open(const AVpkPath: string): Boolean;
procedure Close();
function IsLoaded(): Boolean;
```

Opens the VPK, loads the embeddings model (24-layer bidirectional encoder, F32), and initializes GPU compute.

```pascal
function Embed(const AText: string): TArray<Single>;
function EmbedQuery(const AText: string): TArray<Single>;
function EmbedDocument(const AText: string): TArray<Single>;
```

`Embed` returns a raw 768-dimensional L2-normalized vector. `EmbedQuery` and `EmbedDocument` prepend the model's trained prompt prefixes (`task: search result | query: ` and `title: none | text: ` respectively) before embedding.

```pascal
class function Similarity(const AA: TArray<Single>;
  const AB: TArray<Single>): Single; static;
```

Cosine similarity between two normalized embeddings. Range: -1.0 to 1.0.

---

### 📋 Key Constants

Declared in `Gemma4.Inference.pas`:

| Constant | Value | Purpose |
|----------|-------|---------|
| `CRoleUser` | `'user'` | User message role |
| `CRoleAssistant` | `'assistant'` | Assistant message role |
| `CRoleSystem` | `'system'` | System message role |
| `CRoleTool` | `'tool'` | Tool result role |

Declared in `Gemma4.Types.pas`:

| Constant | Value | Purpose |
|----------|-------|---------|
| `CHiddenSize` | 2560 | Text decoder hidden dimension |
| `CNumHiddenLayers` | 42 | Number of decoder layers |
| `CVocabSize` | 262144 | Vocabulary size |
| `CSlidingWindow` | 512 | Sliding attention window size |

### 📋 Key Types

`TProgressState` (declared in `Gemma4.Types.pas`):

| Value | Meaning |
|-------|---------|
| `psPrep` | Preparation phase |
| `psStart` | Operation starting |
| `psInProgress` | Data arriving (tokens, progress steps) |
| `psEnd` | Operation complete |

`TInferenceStats` (declared in `Gemma4.Types.pas`):

| Field | Type | Meaning |
|-------|------|---------|
| `TokenCount` | `Integer` | Generated token count |
| `ElapsedSec` | `Double` | Generation wall-clock time |
| `TokensPerSec` | `Double` | Generation speed |
| `PrefillTokenCount` | `Integer` | Prompt tokens processed |
| `PrefillSec` | `Double` | Prefill wall-clock time |
| `PrefillTokensPerSec` | `Double` | Prefill speed |

> [!TIP]
> 💡 Use `FormatText(sdkBasic)` for a formatted one-line summary or `FormatText(sdkDetailed)` for the full breakdown including per-phase CPU and GPU timing.


---

<a id="application-layer"></a>

## 🧰 Application Layer

The classes above are the raw inference engine. On top of them sits an optional application layer that adds tool calling, persistent memory, conversation policy, and an interactive chat loop. These build strictly downward: `TChat` -> `TSession` -> `TInference`, with `TMemory` (SQLite + FTS5 + HNSW) and `TToolRegistry` as siblings.

- 🔧 [Tool System](#tool-system) -- `TToolRegistry`, `TToolBuilder`, standard catalog
- 🗄️ [TMemory](#tmemory) -- SQLite conversation archive with hybrid retrieval
- 🧭 [THNSW](#thnsw) -- vector index behind semantic recall
- 🗣️ [TSession](#tsession) -- conversation policy, recall, tool loop, infinite memory
- 💻 [TChat / TConsoleChat](#tchat) -- interactive chat loop and console frontend
- 📁 [Gemma4.Common](#gemma4-common) -- shared resource paths
- 💾 [KV State Save/Load](#kv-state) -- persist and restore the GPU KV cache

---

<a id="tool-system"></a>

### 🔧 Tool System

Declared in `Gemma4.Tools.pas` and `Gemma4.Tools.Utils.pas`. Function-calling infrastructure: register tools, generate schemas for the chat template, and dispatch parsed calls.

#### 🏗️ TToolRegistry

Declared in `Gemma4.Tools.pas`. Extends `TBaseObject`.

```pascal
function DefineTool(const AToolName: string): TToolBuilder;
procedure AddDef(const ADef: TToolDef);
function Count(): Integer;
function HasTool(const AToolName: string): Boolean;
function GetDefs(): TArray<TToolDef>;
procedure Clear();
function ToOaiJson(): string;
function Execute(const ACall: TToolCall): string;
function ExecuteAll(const ACalls: TArray<TToolCall>): TArray<string>;
```

`DefineTool` returns a fluent `TToolBuilder` for the named tool. `ToOaiJson` emits the OpenAI-format tool declarations array consumed by the chat template (feed it to `TInference.SetTools`); it returns `''` when no tools are registered. `Execute` dispatches one parsed `TToolCall` to its handler and returns the handler's JSON result.

#### 🧱 TToolBuilder

Fluent construction, owned by the registry. `OnExecute` closes the chain and returns the owning registry.

```pascal
function Description(const AText: string): TToolBuilder;
function Param(const AName: string; const AType: TToolParamType;
  const ADescription: string; const ARequired: Boolean): TToolBuilder;
function OnExecute(const AHandler: TToolHandler): TToolRegistry;
```

```pascal
TToolParamType = (tptString, tptInteger, tptNumber, tptBoolean);

TToolHandler = reference to function(const AToolName: string;
  const AParams: TToolParams): string;
```

Inside a handler, read arguments through `TToolParams` (`AsString`/`AsInteger`/`AsDouble`/`AsBoolean`/`Has`) and assemble the result with the `ToolResult` helper.

```pascal
function ToolResult(const APairs: array of const): string;
function ToolJsonStr(const AValue: string): string;
```

`ToolResult` builds a JSON object from alternating key/value pairs. `ToolJsonStr` JSON-escapes and quotes a single string value.

#### 🗂️ Standard and Meta Tools

Declared in `Gemma4.Tools.Utils.pas`. `RegisterStandardTools` populates a registry with a ready-made catalog; `RegisterMetaTools` installs a two-tier bootstrap (`find_tool` / `use_tool` / `run_script`) so a small model can search a large catalog on demand.

```pascal
procedure RegisterStandardTools(const ARegistry: TToolRegistry);
procedure RegisterMetaTools(const ABootstrap: TToolRegistry;
  const ACatalog: TToolRegistry; const APythonExe: string;
  const AScriptsDir: string);
```

| Tool | Purpose |
|------|---------|
| `get_time` | Current local time (display format) |
| `get_date` | Current local date |
| `get_weather` | Current weather (Nominatim geocode + Open-Meteo) |
| `web_search` | Web search via the Tavily API |
| `dir_list` | List a directory |
| `file_info` | File metadata |
| `read_file` | Read a text file (bounded) |
| `pip_install` | Install a Python package into the bundled interpreter |

Lower-level helpers are also exposed directly: `RestRequest` (fluent HTTP client via `TRestRequest`), `TLocalOps` (filesystem + `RunCmd`), and the standalone `ToolGeocode`, `ToolWeather`, `ToolWebSearch`, `ToolRunScript`, `ToolTimeNow`, `ToolDateNow`. Each returns a caller-owned `TToolResponse` (`Success`/`StatusCode`/`ErrorMsg`/`RawJson`/`Json`).

> [!IMPORTANT]
> 🔑 `web_search` uses the [Tavily](https://tavily.com) API. Set the `TAVILY_API_KEY` environment variable before registering the standard tools. Tavily's free tier includes 1000 credits/month; see [tavily.com/#pricing](https://tavily.com/#pricing) for details. Without a key, `web_search` returns an error result; every other standard tool works unchanged.

#### 📋 Error Codes

| Code | Meaning |
|------|---------|
| `TOL001` | Unknown tool |
| `TOL002` | Handler failure |
| `TOL003` | Parameter error |
| `TLU001` | REST request failure |

```pascal
LReg := TToolRegistry.Create();
LReg.DefineTool('get_weather')
  .Description('Current weather for a city')
  .Param('location', tptString, 'City name', True)
  .OnExecute(
    function(const AToolName: string; const AParams: TToolParams): string
    var
      LResp: TToolResponse;
    begin
      LResp := ToolWeather(AParams.AsString('location'));
      try
        Result := LResp.RawJson();
      finally
        LResp.Free();
      end;
    end);
LInf.SetTools(LReg.ToOaiJson());
```

---

<a id="tmemory"></a>

### 🗄️ TMemory

Declared in `Gemma4.Memory.pas`. Extends `TBaseObject`. A FireDAC/SQLite archive of conversation turns, facts, and ingested documents, with hybrid keyword (FTS5) plus vector (HNSW) retrieval.

#### 📂 Lifecycle and Embeddings

```pascal
function OpenSession(const ADbPath: string): Boolean;
procedure CloseSession();
function IsOpen(): Boolean;
function AttachEmbeddings(const AEmbedder: TEmbeddings): Boolean;
procedure DetachEmbeddings();
function HasEmbedder(): Boolean;
```

`OpenSession` opens (or creates) the SQLite database and its FTS5/HNSW schema. A loaded `TEmbeddings` is required: turns are embedded on append, and retrieval fuses the HNSW vector hits with the FTS5 keyword index.

#### 📝 Turns, Facts, and Documents

```pascal
function AppendTurn(const ARole: string; const AText: string;
  const ATokenCount: Integer): Int64;
function GetTurn(const ATurnId: Int64): TMemoryTurn;
function GetRecentTurns(const ACount: Integer): TArray<TMemoryTurn>;
function GetTurnCount(): Integer;
function GetPinnedTurns(const AMax: Integer): TArray<TMemoryTurn>;
function AddFact(const AText: string; const APinned: Boolean = True): Int64;
function AddDocument(const ASource, ATitle, AText: string;
  const AChunkTokens, AOverlapTokens: Integer;
  const APinned: Boolean = False): Int64;
```

`AddDocument` chunks and embeds a document for retrieval. Facts default to pinned (always eligible for recall).

```pascal
TMemoryTurn = record
  TurnId      : Int64;
  TurnIndex   : Integer;
  Role        : string;
  Text        : string;
  TokenCount  : Integer;
  CreatedAt   : Int64;
  Score       : Single;       // fused retrieval score
  CosineScore : Single;       // vector similarity component
  Pinned      : Boolean;
end;
```

#### 🔎 Retrieval

```pascal
function SearchFTS5(const AQuery: string;
  const ATopK: Integer): TArray<TMemoryTurn>;
function SearchVector(const AQuery: string;
  const ATopK: Integer): TArray<TMemoryTurn>;
```

`SearchFTS5` is keyword match; `SearchVector` embeds the query and searches the HNSW index. `TSession` fuses both for recall injection.

#### 🧹 Maintenance, Metadata, and Summary

```pascal
procedure PurgeTurn(const ATurnId: Int64);
procedure PurgeAll();
procedure PurgeDatabase();
procedure PurgeWhere(const AWhereClause: string);
procedure PurgeDocument(const ADocumentId: Int64);
function SnapshotTo(const APath: string): Boolean;
function GetDbPath(): string;
procedure SetMeta(const AKey: string; const AValue: string);
function GetMeta(const AKey: string): string;
procedure SetSummary(const AText: string);
function GetSummary(): string;
property MinTurnTokens: Integer read FMinTurnTokens write FMinTurnTokens;
```

`SetSummary`/`GetSummary` back the infinite-memory feature: when `TSession` evicts old turns it summarizes them and stores the running summary here. `MinTurnTokens` filters trivially short turns out of the archive.

#### 📋 Error Codes

`MEM001`-`MEM014`: not open, empty DB path, embedder nil/not-loaded/detached, no embedder, embedding-dimension mismatch, empty WHERE, invalid chunk/overlap, open failure, embed failure, FTS5 failure, snapshot failure. Archive role constants: `crFact`, `crChunk`.

> [!NOTE]
> 🧩 `TMemory` opens a real SQLite file through FireDAC's bundled SQLite driver. No external database server is required.

---

<a id="thnsw"></a>

### 🧭 THNSW

Declared in `Gemma4.HNSW.pas`. Extends `TBaseObject`. A Hierarchical Navigable Small World graph for approximate nearest-neighbor search over embedding vectors. `TMemory` owns one internally; it is also usable standalone.

```pascal
constructor Create(); override;
procedure Init(const AConfig: THNSWConfig);
procedure Insert(const ANodeId: Int64; const AVector: TArray<Single>);
function Search(const AQuery: TArray<Single>;
  const ATopK: Integer): TArray<THNSWSearchResult>;
procedure Delete(const ANodeId: Int64);
function NodeCount(): Integer;
function TotalNodeCount(): Integer;
function SaveToBytes(): TBytes;
procedure LoadFromBytes(const AData: TBytes);
procedure Clear();
property Config: THNSWConfig read FConfig;
```

`THNSWIndex` follows the `TBaseObject` pattern: parameterless `Create()` then `Init` with a config record.

```pascal
THNSWConfig = record
  Dim: Integer;             // vector dimensionality (e.g. 768)
  M: Integer;               // max connections per node per layer (default 16)
  EfConstruction: Integer;  // beam width during insert (default 200)
  EfSearch: Integer;        // beam width during search (default 50)
  ML: Double;               // level multiplier = 1/ln(M), auto-computed
end;

THNSWSearchResult = record
  NodeId: Int64;
  Distance: Single;         // 1 - cosine (0 = identical, 2 = opposite)
end;
```

`SaveToBytes`/`LoadFromBytes` serialize the graph, metadata, and vectors to a binary blob (used by `TMemory` for SQLite BLOB storage). Load validates against `HNSW_MAGIC`/`HNSW_VERSION`.

#### 📋 Error Codes

| Code | Meaning |
|------|---------|
| `HNS001` | Bad magic number on load |
| `HNS002` | Unsupported version |
| `HNS003` | Vector dimension mismatch |

```pascal
LCfg.Dim := 768;
LCfg.M := 16;
LCfg.EfConstruction := 200;
LCfg.EfSearch := 50;

LIndex := THNSWIndex.Create();
LIndex.Init(LCfg);
LIndex.Insert(1, LEmb.EmbedDocument('the sky is blue'));
LHits := LIndex.Search(LEmb.EmbedQuery('what color is the sky'), 5);
```

---

<a id="tsession"></a>

### 🗣️ TSession

Declared in `Gemma4.Session.pas`. Extends `TBaseObject`. The policy layer over `TInference`: it owns the system-prompt invariant, assistant-turn bookkeeping, context-budget trimming, thinking-span stripping, the tool-dispatch loop, memory-recall injection, summarize-on-evict infinite memory, and state persistence. `TInference`, `TMemory`, and `TToolRegistry` are borrowed, not owned.

#### 🔌 Wiring

```pascal
procedure SetInference(const AInference: TInference);
procedure SetMemory(const AMemory: TMemory);
procedure SetToolRegistry(const ARegistry: TToolRegistry);
procedure SetToolCallback(const ACallback: TToolNotifyCallback;
  const AUserData: Pointer);
procedure SetSystemPrompt(const APrompt: string);
```

`SetInference` is required; the rest are optional. `TToolNotifyCallback` fires once per tool invocation, before the handler runs.

#### 💬 Asking

```pascal
function Ask(const AText: string; const AMaxTokens: Integer): string;
```

Drives one full turn: build the recall-augmented prompt, generate, extract and dispatch any tool calls, feed results back, and regenerate -- up to `MaxToolRounds` times. Returns the assistant's reply (the full reply; `StripThinking` only affects what is stored in history).

#### 📚 History and State

```pascal
procedure ClearHistory();
function CompactHistory(const AKeepRecentTurns: Integer): Integer;
function SaveHistory(const AFileName: string): Boolean;
function LoadHistory(const AFileName: string): Boolean;
function SaveState(const AFileName: string): Boolean;
function LoadState(const AFileName: string): Boolean;
```

`SaveHistory`/`LoadHistory` persist the message history as JSON. `SaveState`/`LoadState` persist a combined file: the KV-cache blob plus the history JSON, so a conversation resumes with warm KV. `CompactHistory` proactively evicts old turns; when `SummarizeOnEvict` is set and memory is attached, evicted turns are summarized first.

#### 🎛️ Policy Properties

| Property | Default | Purpose |
|----------|---------|---------|
| `StripThinking` | `True` | Drop the thinking span before storing a reply |
| `RecallTopK` | `5` | Vector/keyword recall hits to consider |
| `PinnedTopK` | `10` | Pinned facts to consider |
| `RecallBudgetTokens` | `512` | Token budget for the recall injection |
| `MinRecallScore` | `0.45` | Minimum cosine similarity for a vector hit |
| `MaxToolRounds` | `4` | Generate->execute->regenerate rounds per `Ask` |
| `ToolResultMaxChars` | `4000` | Truncate long tool results |
| `SummaryMaxTokens` | `512` | Token budget for eviction summaries |
| `SummarizeOnEvict` | `True` | Summarize evicted turns (infinite memory) |

> [!TIP]
> 💡 `MinRecallScore` matters: top-K nearest neighbors are returned regardless of relevance, so without a floor even a greeting pulls in unrelated memories. FTS5 keyword hits are unaffected.

#### 📋 Error Codes

`SES001` no inference, `SES002` generate failure, `SES003` context overflow, `SES004`/`SES005` history save/load, `SES008` tool-round ceiling, `SES009`/`SES010` state save/load.

---

<a id="tchat"></a>

### 💻 TChat / TConsoleChat

Declared in `Gemma4.Chat.pas`. `TChat` extends `TBaseObject` and is an abstract chat loop that owns the full stack (`TInference`, a required `TEmbeddings`, an optional `TMemory` archive, and `TSession`), builds it inside `Run()` from path properties, and tears it down on exit. `TConsoleChat` is the concrete console frontend (markdown-rendered streaming, ESC cancellation, `ReadLn` input).

#### ▶️ Running

```pascal
procedure Run();  // template method: build stack, input loop, teardown
```

Configure via properties before calling `Run()`:

| Property | Default | Purpose |
|----------|---------|---------|
| `ModelPath` | `''` | VPK path (required) |
| `UseGPU` | `True` | Vulkan acceleration |
| `MemoryDbPath` | `''` | SQLite memory DB (empty disables memory) |
| `SystemPrompt` | `''` | Initial system prompt |
| `MaxTokens` | `1024` | Max generation tokens |
| `EnableThinking` | `True` | Activate the reasoning channel |
| `ShowThinking` | `True` | Show or hide the reasoning span |
| `RebuildThreshold` | `0.75` | Compact when token position crosses this fraction of context |
| `KeepRecentTurns` | `3` | Exchanges that survive a compaction |
| `DocChunkWords` / `DocOverlapWords` | `200` / `40` | `/addfile` chunking |
| `ToolRegistry` | `nil` | App-provided tools (assign before `Run`) |

`RecallTopK`, `PinnedTopK`, `RecallBudgetTokens`, `MinRecallScore`, `MinTurnTokens`, `StripThinking`, `MaxToolRounds`, and `ToolResultMaxChars` are passed through to the owned `TSession`/`TMemory`.

#### 🧩 Extending

Derived classes must override four abstract I/O methods -- `DoGetInput`, `DoOutput`, `DoToken`, `DoCancel` -- and may override virtual hooks: `DoError`, `DoInfo`, `DoStartup`, `DoShutdown`, `DoGenerationComplete`, `DoToolCall`, `DoLoadProgress`, `DoConfigureStack`, and `DoCommand` (custom slash commands; return `True` if handled). `GetInference`/`GetSession`/`GetMemory` expose the live objects between `DoConfigureStack` and teardown.

#### ⌨️ Slash Commands

| Command | Action |
|---------|--------|
| `/quit` | Exit the chat |
| `/clear` | Clear conversation history |
| `/forget` | Clear history AND wipe the memory database |
| `/system <text>` | Set the system prompt |
| `/addfact <text>` | Add a fact to memory |
| `/addfile <path>` | Add a document to memory |
| `/stats` | Show inference statistics |
| `/turns` | Show history and archive counts |
| `/tokens <n>` | Set max generation tokens |
| `/compact` | Archive all but recent turns |
| `/save <path>` | Save history JSON |
| `/load <path>` | Load history JSON |
| `/state <path>` | Save full state (KV cache + history) |
| `/restore <path>` | Restore full state |
| `/summary` | Show the conversation summary |
| `/dbsave <path>` | Snapshot the memory DB to a path |
| `/dbload <path>` | Load a memory DB from a path |
| `/dblist <path>` | List memory DB files under a path |
| `/dbreset` | Reload the original memory DB |
| `/help` | Show this help |

#### 🗄️ Memory DB Management

```pascal
function MemoryDbSave(const APath: string): Boolean;
function MemoryDbLoad(const APath: string): Boolean;
function MemoryDbList(const APath: string): TArray<string>;
function MemoryDbReset(): Boolean;
```

Snapshot, swap, enumerate, or reset the backing memory database at runtime. Swapping changes only the recall backing; the in-flight `TSession` history is preserved.

#### 📋 Error Codes

`CHT001` no model, `CHT002` model load, `CHT003` embedder, `CHT004` memory, `CHT005` config.

```pascal
LChat := TConsoleChat.Create();
try
  LChat.ModelPath := 'C:\Dev\LLM\VPK\Gemma4.vpk';
  LChat.MemoryDbPath := TGemma4.ResPath('res\database\chat.db');
  LChat.SystemPrompt := 'You are a helpful assistant.';
  LChat.Run();
finally
  LChat.Free();
end;
```

---

<a id="gemma4-common"></a>

### 📁 Gemma4.Common

Declared in `Gemma4.Common.pas`. Shared resource-path resolution for the application layer.

```pascal
class procedure TGemma4.SetResFolder(const AFolder: string); static;
class function TGemma4.ResPath(const ARelativePath: string): string; static;
```

`ResPath` resolves a relative path against the executable directory (or the folder set by `SetResFolder`). Key constants:

| Constant | Value | Purpose |
|----------|-------|---------|
| `CExtDb` | `'db'` | Memory database extension |
| `CExtState` | `'g4state'` | Saved conversation-state extension |
| `CResDatabase` | `'res\database'` | Default database folder |
| `CResPythonExe` | `'res\python\python.exe'` | Bundled Python interpreter |
| `CResScripts` | `'res\scripts'` | Tool script folder |

---

<a id="kv-state"></a>

### 💾 KV State Save/Load

Declared in `Gemma4.Inference.pas` (delegating to `TModel`). Persist and restore the GPU KV cache so a conversation resumes with a warm prefix instead of re-decoding the whole prompt.

```pascal
function SaveKVState(const AFileName: string): Boolean;
function LoadKVState(const AFileName: string): Boolean;
```

`SaveKVState` downloads the KV cache from GPU buffers and writes a `.g4kv` file (magic, version, slot count, position, and per-slot K/V blobs). `LoadKVState` uploads it back. `TSession.SaveState`/`LoadState` combine this blob with the history JSON into a single resumable file.

Session-support helpers used by the policy layer are also public:

```pascal
function TokenCount(const AText: string): Integer;
function ContextSize(): Integer;
function ParseToolCalls(): string;
procedure AddToolCallTurn(const AToolCallsJson: string);
```

`ParseToolCalls` extracts `<|tool_call>...<tool_call|>` markers from the last response and returns a JSON array of tool_call objects (`''` when none). `AddToolCallTurn` re-injects that array as an assistant turn so the template re-renders the markers on the next `Generate`.

> [!NOTE]
> 🧩 The `.g4kv` format is versioned; a KV file written by an incompatible engine build is rejected on load rather than silently misread.

<a id="architecture"></a>

## 🏗️ Architecture

This section covers the internals of Gemma4.pas: the model architecture, the Vulkan compute pipeline, quantization, the VPK file format, and the multimodal encoders. You do not need any of this to use `TInference` -- it is here for developers who want to understand what happens beneath the API.

### 🗺️ Unit Map

```
Gemma4.Types.pas        Shared types, records, constants, enums
Gemma4.Config.pas       Model config loader (config.json)
Gemma4.Tokenizer.pas    BPE tokenizer (tokenizer.json, 262144 tokens)
Gemma4.Tensors.pas      Tensor storage, views, basic ops
Gemma4.Quant.pas        Q4_0 quantization and dequantization
Gemma4.Safetensors.pas  Safetensors file parser (header + tensor map)
Gemma4.Packer.pas       Offline packing tool: safetensors -> VPK
Gemma4.Attention.pas    RoPE, GQA, sliding window + full attention, KV cache
Gemma4.Layers.pas       RMSNorm, GeLU MLP, residual connections
Gemma4.Model.pas        Forward pass orchestration, generation loop
Gemma4.Vulkan.pas       Vulkan instance, device, queues, buffer management
Gemma4.Shaders.pas      SPIR-V shader loading and pipeline cache
Gemma4.Compute.pas      GPU kernel dispatch (GEMM, softmax, RoPE, etc.)
Gemma4.Jinja.pas        Full Jinja template engine for chat formatting
Gemma4.Image.pas        Image decode, resize, patchify (VCL Graphics)
Gemma4.Vision.pas       SigLIP vision encoder (16 layers, F32 GPU)
Gemma4.Audio.pas        Conformer audio encoder (12 layers, F32 GPU)
Gemma4.Video.pas        Video frame extraction (Windows Media Foundation)
Gemma4.Embeddings.pas   EmbeddingGemma-300m bidirectional encoder
Gemma4.Inference.pas    Top-level API: messages, template, generate, stream

  --- Application layer (optional, builds on Inference) ---
Gemma4.Common.pas       Shared resource-path resolution and constants
Gemma4.Tools.pas        Tool registry, fluent builder, schema generation, dispatch
Gemma4.Tools.Utils.pas  Standard tool catalog, meta-tools, REST client, local ops
Gemma4.HNSW.pas         HNSW approximate nearest-neighbor vector index
Gemma4.Memory.pas       SQLite conversation archive (FTS5 + HNSW retrieval)
Gemma4.Session.pas      Conversation policy, recall, tool loop, infinite memory
Gemma4.Chat.pas         Abstract interactive chat loop + console frontend
```

> [!NOTE]
> 🧩 Dependencies flow downward. `Inference` depends on `Model`, which depends on `Attention` and `Layers`, which depend on `Tensors` and `Quant`, which depend on `Types`. No circular references, no lateral dependencies between peers.

The application layer sits above the engine and never below it: `Chat` -> `Session` -> `Inference`, with `Memory` (which owns an `HNSW` index and borrows a `TEmbeddings`) and `Tools` as siblings. `Common` is a leaf shared by the whole layer. `Session`, `Memory`, and `Tools` are borrowed by the layers above them, never owned, so a single `TInference` and `TEmbeddings` can back several higher-level objects.

### 💬 Text Decoder

The Gemma 4 E4B text decoder is a 42-layer transformer with the following specifications:

| Parameter | Value |
|-----------|-------|
| 📏 Hidden size | 2560 |
| 📚 Layers | 42 |
| 🧠 Query heads | 8 |
| 🔑 KV heads | 2 (grouped-query attention) |
| 📐 Head dimension | 256 (sliding), 512 (full) |
| ⚙️ Intermediate size | 10240 |
| 📖 Vocabulary | 262144 tokens |
| 🗔️ Sliding window | 512 positions |
| ⚡ Activation | GeLU (PyTorch tanh approximation) |
| 📏 Normalization | RMSNorm (epsilon 1e-6) |
| 🔐 Logit softcapping | 30.0 |

The layer pattern repeats seven times: five sliding-window attention layers followed by one full-attention layer. Sliding layers attend only to the nearest 512 positions using a ring buffer. Full layers attend to the entire sequence using a standard growing KV cache.

**🔄 Grouped-query attention (GQA):** Each layer has 8 query heads but only 2 KV heads. Four query heads share one KV head, reducing memory and bandwidth.

**🔗 Shared KV cache:** 18 of the 42 layers share KV caches with other layers, leaving 24 unique KV cache slots. This further reduces VRAM usage.

**📊 Per-layer embeddings (PLE):** Each layer receives a 256-dimensional per-layer embedding that is added to the residual stream before the attention block. The embedding is looked up from a learned table indexed by token ID.

**🔁 RoPE:** Sliding layers use standard rotary position embeddings (theta=10000). Full layers use proportional RoPE (theta=1000000) with a partial rotary factor of 0.25 -- only the first quarter of the head dimension is rotated.

### ⚡ Vulkan Compute Pipeline

All heavy math runs on the GPU as compiled SPIR-V compute shaders. The pipeline:

1. **📤 Weight upload:** Q4_0-quantized decoder weights (~2.7 GB) are uploaded to device-local VRAM at model load time. Encoder weights (F32) are uploaded separately.
2. **🔢 Embedding lookup:** Token IDs are mapped to embedding vectors on the GPU.
3. **🔄 Layer loop:** For each of the 42 layers: RMSNorm, attention (Q/K/V projection, RoPE, score computation, softmax, weighted sum, output projection), RMSNorm, MLP (gate + up projection, GeLU activation, down projection), residual connections.
4. **🏁 Final norm + LM head:** RMSNorm the final hidden state, project to vocabulary logits (tied weights -- reuses the embedding matrix), apply logit softcapping.
5. **🎲 Sampling:** Logits are read back to the CPU for top-k/top-p/temperature sampling.

> [!NOTE]
> 🧩 **Batched prefill:** Long prompts are processed in 256-token chunks. Each chunk uses matrix-matrix multiply shaders (matmat_q4q8 with DP4A integer dot products) and batched attention shaders. Generation (one token at a time) uses matrix-vector multiply shaders.

### 📦 Quantization

The text decoder uses Q4_0 quantization for weight matrices. Each Q4_0 block stores 32 weights:

- 2 bytes: fp16 scale factor
- 16 bytes: 32 x 4-bit values packed two per byte

Norms, scalars, and biases stay at F32. Encoder weights (vision, audio, embeddings) are stored at full F32 precision -- they are precision-sensitive and relatively small.

> [!IMPORTANT]
> 🧱 At inference time, activations are quantized to Q8_1 format on the fly. The GEMM shaders use `dotPacked4x8EXT` integer dot products (DP4A) to multiply Q4_0 weights against Q8_1 activations, which is the same approach used by llama.cpp's Vulkan backend.

### 📁 VPK File Format

A VPK (Virtual Pack) is a flat archive file containing both models. It is created by `TPacker` and memory-mapped at runtime by `TInference`.

```
Gemma4.vpk
  E4B/
    manifest.json           Tensor name -> offset/size/dtype map
    weights.bin             Q4_0 decoder weights (~2.7 GB)
    encoders.bin            F32 vision + audio encoder weights (~1.8 GB)
    encoders_manifest.json  Encoder tensor map
    config.json             Model configuration
    tokenizer.json          BPE vocabulary and merge rules
    tokenizer_config.json   Tokenizer settings
    generation_config.json  Default generation parameters
    chat_template.jinja     Trained chat template
    processor_config.json   Image/audio processor settings
  embeddings/
    manifest.json           Tensor map
    weights.bin             F32 embedding model weights (~1.2 GB)
    config.json             Embedding model configuration
    tokenizer.json          Vocabulary (same family as E4B)
```

> [!TIP]
> 💡 Because the archive is memory-mapped, startup is fast and memory usage stays low -- weights are paged in on demand by the OS, not loaded into heap memory.

### 👁️ Vision Encoder

The vision encoder is a 16-layer SigLIP-based transformer that converts an image into soft tokens the text decoder can attend to.

**🖼️ Image processing pipeline** (`Gemma4.Image.pas`):
1. Decode the image via VCL `Graphics` (BMP, JPEG, PNG -- no third-party libraries)
2. Compute the optimal tile grid based on the soft-token budget (70/140/280/560/1120)
3. Bicubic resize each tile to 224x224
4. Rescale pixels to [-1, 1] range
5. Extract 16x16 patches (196 patches per tile)

**🧠 Vision forward pass** (`Gemma4.Vision.pas`):
1. Linear patch embedding (3x16x16 -> 768) + learned 2D position table
2. 16 transformer layers with 2D RoPE (theta=100)
3. 3x3 average pooling with sqrt(768) scaling
4. Layer normalization + linear projection (768 -> 2560)
5. Output: soft-token embeddings injected into the text decoder's residual stream

### 🎙️ Audio Encoder

The audio encoder is a 12-layer conformer that converts a WAV file into soft tokens.

**🎵 Audio processing pipeline** (`Gemma4.Audio.pas`):
1. Load WAV (PCM16 or F32, mono or stereo, any sample rate)
2. Downmix to mono, resample to 16 kHz
3. Compute 80-band log-mel spectrogram (25 ms windows, 10 ms hop)
4. Subsample time dimension (stack 2 frames, halving sequence length)

**🧠 Conformer forward pass:**
1. Linear input projection
2. 12 conformer blocks: feed-forward, self-attention (sliding window=12), depthwise convolution, feed-forward
3. Output: soft-token embeddings (one per 40 ms of audio)

### 🎬 Video Processing

Video reuses the vision encoder. `Gemma4.Video.pas` extracts frames using Windows Media Foundation (OS-shipped COM API):

1. Open the video file via `IMFSourceReader`
2. Sample N frames uniformly across the duration (default 32, max 60)
3. Each frame is processed through the image pipeline (resize, patchify) and the vision encoder
4. Soft tokens for all frames are concatenated and injected into the text decoder

> [!NOTE]
> 🧩 Each frame produces 66-70 soft tokens (at budget=70), so a 32-frame video generates ~2100 soft tokens.

### 🔍 Embeddings Model

The EmbeddingGemma-300m model is a separate 24-layer bidirectional encoder based on the Gemma 3 architecture:

| Parameter | Value |
|-----------|-------|
| 📏 Hidden size | 768 |
| 📚 Layers | 24 |
| 🧠 Query heads | 3 |
| 🔑 KV heads | 1 |
| 📐 Head dimension | 256 |
| ⚙️ Intermediate size | 1152 |
| 📊 Output dimension | 768 |
| 📏 Max positions | 2048 |

**Key differences from the text decoder:**
- 🔁 Bidirectional attention (not causal): full layers attend to all positions unconditionally; sliding layers use a symmetric window (attend if abs(q_pos - k_pos) < 512)
- 🔄 Full rotary embeddings (no partial factor)
- ❌ No per-layer embeddings (PLE)
- 📏 F32 throughout (no quantization)

**🧠 Head pipeline (CPU):**
1. Mean pooling over all positions
2. Dense 768 -> 3072 (no bias)
3. Dense 3072 -> 768 (no bias)
4. L2 normalization
5. Output: 768-dimensional unit vector

> [!TIP]
> 💡 The model uses trained prompt prefixes for retrieval: `task: search result | query: ` for queries and `title: none | text: ` for documents. These are applied automatically by `TEmbeddings.EmbedQuery` and `TEmbeddings.EmbedDocument`.


### 🧰 Application Layer

Above the inference engine sits an optional application layer -- the classes that turn `TInference` into an agent or an interactive assistant. It is strictly additive: nothing in the engine depends on it, and you can ignore it entirely if all you need is generation.

```
              TChat  (owns the stack, built inside Run)
                |
                v
             TSession  (conversation policy, recall, tool loop)
              /  |  \
             /   |   \
            v    v    v
   TInference  TMemory  TToolRegistry
   (borrowed)    |         |
                 |         +-- TToolBuilder / standard tool catalog
                 |
                 +-- THNSW (vector index)
                 +-- TEmbeddings (borrowed, for vector recall)
                 +-- SQLite (FTS5 keyword index)

   TGemma4 (Gemma4.Common) -- leaf: resource-path resolution, shared by all
```

**🔗 Ownership.** `TChat` owns the whole stack: it constructs `TInference` and a `TEmbeddings` (always loaded), plus an optional `TMemory` archive, and `TSession` inside `Run()` from its path properties, then tears them down in reverse order on exit. `TSession` borrows `TInference`, `TMemory`, and `TToolRegistry` -- it never frees them. `TMemory` owns its `THNSW` index but borrows the `TEmbeddings` encoder. This borrowing lets one loaded model back several higher-level objects without duplicating GPU weights.

**♾️ Infinite memory.** When history approaches the context budget, `TSession` evicts the oldest turns. With `SummarizeOnEvict` enabled, those turns are first summarized by the model itself and the running summary is stored in `TMemory` (via `SetSummary`), then re-injected ahead of recall on later turns. The conversation can therefore run indefinitely without the context simply falling off a cliff.

**💾 KV state.** `TModel` can serialize its GPU KV cache to a versioned `.g4kv` file (`SaveKVState`/`LoadKVState`, surfaced on `TInference`). `TSession.SaveState`/`LoadState` bundle that blob with the history JSON, so a full conversation -- warm KV included -- can be saved and restored across process runs.

> [!NOTE]
> 🧩 The application layer is pure Delphi and in-process: there is no subprocess, no server, and no external database.

<a id="how-to-guide"></a>

## 🧪 How-To Guide

Practical recipes built from the verified testbed examples in `projects\Testbed\`. Each compiles and runs as shown.

### 🗺️ Recipe Map

| Need | Recipe |
|------|--------|
| 💬 Stream text with thinking | [Streaming Chat](#-streaming-chat-with-thinking) |
| 🔁 Continue a conversation | [Multi-Turn Conversation](#-multi-turn-conversation) |
| 🖼️ Describe an image | [Image Description](#️-image-description) |
| 🎙️ Understand audio | [Audio Description](#️-audio-description) |
| 🎬 Describe a video | [Video Description](#-video-description) |
| 🔍 Semantic search | [Embeddings and Retrieval](#-embeddings-and-retrieval) |
| 🔧 Give the model tools | [Tool Calling and the Agentic Loop](#-tool-calling-and-the-agentic-loop) |
| 🗄️ Remember across turns | [Persistent Memory and Recall](#️-persistent-memory-and-recall) |
| ♾️ Never run out of context | [Session with Infinite Memory](#️-session-with-infinite-memory) |
| 💻 Build an interactive assistant | [Interactive Console Chat](#-interactive-console-chat) |
| 💾 Save and resume a conversation | [KV State Save and Restore](#-kv-state-save-and-restore) |
| 📊 Show load progress | [Load Progress Bar](#-load-progress-bar) |
| ✋ Cancel generation | [Cancellation](#-cancellation) |

> [!TIP]
> 💡 All code examples are sourced directly from the working testbed demos. If the prose and the testbed ever seem to disagree, the testbed is authoritative.

### 💬 Streaming Chat with Thinking

From `UDemo.Inference.pas`. Loads the model, enables the thinking channel, and streams a response with the reasoning visible:

```pascal
LInf := TInference.Create();
try
  LInf.ThinkingOpenText := '<thinking>'#10;
  LInf.ThinkingCloseText := #10'</thinking>'#10;
  LInf.SetTokenCallback(TokenCallback, Self);
  LInf.SetLoadProgressCallback(LoadProgressCallback, Self);

  if not LInf.LoadModel(CVpkOutputFile) then
  begin
    LInf.PrintErrors();
    Exit;
  end;

  LInf.EnableThinking := True;
  LInf.ShowThinking := True;

  LInf.AddMessage(CRoleUser,
    'In two sentences, what makes the Vulkan API different from OpenGL?');

  if not LInf.Generate(1024) then
  begin
    LInf.PrintErrors();
    Exit;
  end;

  // Feed the visible reply back for multi-turn
  LInf.AddMessage(CRoleAssistant, LInf.ResponseText);
finally
  LInf.Free();
end;
```

The token callback receives `psInProgress` with each display fragment. When `ShowThinking` is `True`, the thinking content streams through normally. When `False`, only a placeholder string appears.

### 🔁 Multi-Turn Conversation

From `UDemo.Inference.pas`. The second turn deliberately depends on the first, proving history works:

```pascal
// Turn 1
LInf.AddMessage(CRoleUser,
  'In two sentences, what makes the Vulkan API different from OpenGL?');
LInf.Generate(1024);
LInf.AddMessage(CRoleAssistant, LInf.ResponseText);

// Turn 2 -- references turn 1
LInf.AddMessage(CRoleUser,
  'Give one concrete example of that difference in practice.');
LInf.Generate(1024);
```

> [!IMPORTANT]
> 🧱 Always feed `ResponseText` (not `Response`) back into the history. `ResponseText` is the visible answer with thinking stripped. `Response` includes the full text with thinking markers.

### 🖼️ Image Description

From `UDemo.Multimedia.pas`. Image parts go before the text prompt (per model card):

```pascal
LInf.AddMessage(CRoleUser, [
  TMessagePart.ImagePart(LImagePath),
  TMessagePart.TextPart('Describe this image in one sentence.')
]);

if not LInf.Generate(512) then
begin
  LInf.PrintErrors();
  Exit;
end;
```

> [!TIP]
> 💡 `ImagePart` accepts an optional soft-token budget (default 1120, maximum detail). Lower budgets (70, 140, 280, 560) use fewer tokens and process faster at the cost of detail.

### 🎙️ Audio Description

From `UDemo.Multimedia.pas`. Audio parts go after the text prompt (per model card):

```pascal
LInf.AddMessage(CRoleUser, [
  TMessagePart.TextPart('Describe this audio in one sentence.'),
  TMessagePart.AudioPart(LAudioPath)
]);

if not LInf.Generate(512) then
begin
  LInf.PrintErrors();
  Exit;
end;
```

> [!NOTE]
> 🧩 The audio encoder accepts WAV files in PCM16 or F32 format, mono or stereo, at any sample rate. The engine handles downmixing and resampling to 16 kHz internally.

### 🎬 Video Description

From `UDemo.Multimedia.pas`. Video parts go before the text prompt (per model card):

```pascal
LInf.AddMessage(CRoleUser, [
  TMessagePart.VideoPart(LVideoPath),
  TMessagePart.TextPart('Describe this video in one sentence.')
]);

if not LInf.Generate(512) then
begin
  LInf.PrintErrors();
  Exit;
end;
```

> [!TIP]
> 💡 `VideoPart` accepts an optional frame count (default 32, max 60). The engine extracts frames uniformly across the video duration using Windows Media Foundation, processes each frame through the vision encoder, and injects all soft tokens into the text decoder.

### 🔍 Embeddings and Retrieval

From `UDemo.Embedding.pas`. Embeds a document corpus, then ranks documents against queries by cosine similarity:

```pascal
const
  CDocs: array[0..5] of string = (
    'Mars, known for its reddish appearance, is often referred to as the Red Planet.',
    'Jupiter, the largest planet in our solar system, has a prominent red spot.',
    'Photosynthesis converts sunlight, water and carbon dioxide into glucose and oxygen.',
    'The Great Wall of China stretches thousands of kilometers across northern China.',
    'Vulkan is a low-overhead, cross-platform graphics and compute API.',
    'A sourdough starter is a fermented culture of flour and water used to leaven bread.'
  );

var
  LEmb: TEmbeddings;
  LDocVecs: TArray<TArray<Single>>;
  LQuery: TArray<Single>;
  LI: Integer;
  LSim: Single;
begin
  LEmb := TEmbeddings.Create();
  try
    if not LEmb.Open(CVpkOutputFile) then
    begin
      LEmb.PrintErrors();
      Exit;
    end;

    // Embed all documents once
    SetLength(LDocVecs, Length(CDocs));
    for LI := 0 to High(CDocs) do
      LDocVecs[LI] := LEmb.EmbedDocument(CDocs[LI]);

    // Embed a query and rank
    LQuery := LEmb.EmbedQuery('Which planet is known as the Red Planet?');
    for LI := 0 to High(CDocs) do
    begin
      LSim := TEmbeddings.Similarity(LQuery, LDocVecs[LI]);
      WriteLn(Format('  %.4f  [doc %d] %s',
        [LSim, LI, Copy(CDocs[LI], 1, 60)]));
    end;
  finally
    LEmb.Free();
  end;
end;
```

> [!NOTE]
> 🧩 `EmbedQuery` and `EmbedDocument` prepend the model's trained prompt prefixes automatically. The output is a 768-dimensional L2-normalized vector. `Similarity` computes cosine similarity.

### 📊 Load Progress Bar

From `UTestbed.Common.pas`. Wire a load progress callback to show a progress bar during weight upload:

```pascal
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
```

Wire it before calling `LoadModel`:

```pascal
LInf.SetLoadProgressCallback(LoadProgressCallback, Self);
```

### ✋ Cancellation

Poll a cancel callback during generation to stop cleanly:

```pascal
LInf.SetCancelCallback(
  function(const AUserData: Pointer): Boolean
  begin
    Result := TInput.KeyPressed(VK_ESCAPE);  // return True to stop
  end, nil);
```

> [!TIP]
> 💡 The callback is polled after each token. When it returns `True`, generation stops and `Generate` returns `True` (not an error -- the partial response is available in `Response` and `ResponseText`).

### 🔧 Tool Calling and the Agentic Loop

The application layer turns tool use into a single call. Register tools in a `TToolRegistry`, hand it to a `TSession`, and `Ask` runs the whole generate -> execute -> regenerate loop for you:

```pascal
uses
  Gemma4.Inference,
  Gemma4.Tools,
  Gemma4.Tools.Utils,
  Gemma4.Session;

var
  LInf: TInference;
  LTools: TToolRegistry;
  LSession: TSession;
  LReply: string;
begin
  LInf := TInference.Create();
  LTools := TToolRegistry.Create();
  LSession := TSession.Create();
  try
    LInf.LoadModel(CVpkOutputFile);

    // Ready-made catalog: get_time, get_date, get_weather, web_search,
    // dir_list, file_info, read_file, pip_install
    RegisterStandardTools(LTools);

    LSession.SetInference(LInf);
    LSession.SetToolRegistry(LTools);
    LSession.SetSystemPrompt('You are a helpful assistant with tools.');

    // Ask drives the agentic loop internally, up to MaxToolRounds
    LReply := LSession.Ask('What is the weather in Tokyo right now?', 1024);
    WriteLn(LReply);
  finally
    LSession.Free();
    LTools.Free();
    LInf.Free();
  end;
end;
```

Defining your own tool is a fluent chain that ends with the handler:

```pascal
LTools.DefineTool('roll_dice')
  .Description('Roll an N-sided die')
  .Param('sides', tptInteger, 'Number of sides', True)
  .OnExecute(
    function(const AToolName: string; const AParams: TToolParams): string
    begin
      Result := ToolResult(['result', Random(AParams.AsInteger('sides')) + 1]);
    end);
```

> [!IMPORTANT]
> 🔑 The `web_search` tool needs a Tavily API key. Every other standard tool works without any key. See [Getting Started](#getting-started) for the one-time environment setup.

> [!TIP]
> 💡 For a large tool catalog, use `RegisterMetaTools` instead: the model gets three bootstrap tools (`find_tool`, `use_tool`, `run_script`) and searches the full catalog on demand rather than seeing every schema up front.

### 🗄️ Persistent Memory and Recall

Attach a `TMemory` (SQLite + FTS5 + HNSW) to a `TSession` and past turns are recalled automatically on later questions. `TMemory` requires a loaded `TEmbeddings` -- turns are embedded on append and retrieved by fused keyword + vector search:

```pascal
uses
  Gemma4.Inference,
  Gemma4.Embeddings,
  Gemma4.Memory,
  Gemma4.Session,
  Gemma4.Common;

var
  LInf: TInference;
  LEmb: TEmbeddings;
  LMem: TMemory;
  LSession: TSession;
begin
  LInf := TInference.Create();
  LEmb := TEmbeddings.Create();
  LMem := TMemory.Create();
  LSession := TSession.Create();
  try
    LInf.LoadModel(CVpkOutputFile);
    LEmb.Open(CVpkOutputFile);

    LMem.OpenSession(TGemma4.ResPath('res\database\chat.db'));
    LMem.AttachEmbeddings(LEmb);        // required: memory embeds turns on append

    LSession.SetInference(LInf);
    LSession.SetMemory(LMem);
    LSession.MinRecallScore := 0.45;    // reject weak vector hits

    // Store a durable fact, then ask something that needs it
    LMem.AddFact('The user''s favorite fruit is a banana.');
    WriteLn(LSession.Ask('What is my favorite fruit?', 256));
  finally
    LSession.Free();
    LMem.Free();
    LEmb.Free();
    LInf.Free();
  end;
end;
```

> [!NOTE]
> 🧩 `TSession` fuses keyword (FTS5) and vector (HNSW) hits, drops anything below `MinRecallScore`, fits the rest inside `RecallBudgetTokens`, and injects it as background context ahead of the live message.

### ♾️ Session with Infinite Memory

With memory attached and `SummarizeOnEvict` on (the default), a long conversation never simply loses its oldest turns: they are summarized by the model and the running summary is re-injected on later turns. `CompactHistory` triggers eviction proactively:

```pascal
LSession.SetInference(LInf);
LSession.SetMemory(LMem);
LSession.SummarizeOnEvict := True;   // default
LSession.SummaryMaxTokens := 512;

// ... many turns later, when history grows large ...

// Keep the last 3 exchanges live; summarize and evict the rest
LSession.CompactHistory(3);

// The summary is retrievable at any time
WriteLn('Summary so far: ', LMem.GetSummary());
```

`TChat` does this automatically after each turn once the token position crosses `RebuildThreshold` of the context window, so an interactive assistant runs indefinitely without manual compaction.

### 💻 Interactive Console Chat

`TConsoleChat` wires the entire stack together behind a REPL. Configure paths and knobs, then call `Run()`:

```pascal
uses
  Gemma4.Chat,
  Gemma4.Common;

var
  LChat: TConsoleChat;
begin
  LChat := TConsoleChat.Create();
  try
    LChat.ModelPath := CVpkOutputFile;
    LChat.MemoryDbPath := TGemma4.ResPath('res\database\chat.db');
    LChat.SystemPrompt := 'You are a helpful assistant.';
    LChat.EnableThinking := True;
    LChat.ShowThinking := False;       // hide reasoning behind a placeholder
    LChat.Run();                       // blocks until the user types /quit
  finally
    LChat.Free();
  end;
end;
```

At the prompt, slash commands manage the session: `/addfact`, `/addfile`, `/compact`, `/summary`, `/state` and `/restore` (full KV + history), `/dbsave` / `/dbload` / `/dblist` / `/dbreset` (memory databases), and `/help` for the full list. To add your own tools, assign a `TToolRegistry` to `LChat.ToolRegistry` before `Run()`.

### 💾 KV State Save and Restore

Saving the KV cache lets a conversation resume with a warm prefix instead of re-decoding the whole prompt. At the engine level, `TInference` exposes it directly:

```pascal
LInf.AddMessage(CRoleUser, 'Remember the codeword BANANA.');
LInf.Generate(64);

LInf.SaveKVState('session.g4kv');   // write GPU KV cache to disk

// ... reset clears the cache, as if starting over ...
LInf.ResetConversation();

LInf.LoadKVState('session.g4kv');   // restore the cache
LInf.AddMessage(CRoleUser, 'What was the codeword?');
LInf.Generate(64);                  // recalls BANANA from restored KV
```

At the application level, `TSession.SaveState`/`LoadState` bundle the KV blob and the history JSON into one file, so both the cache and the message list travel together:

```pascal
LSession.SaveState('conversation.g4state');
// ... new process ...
LSession.LoadState('conversation.g4state');
WriteLn(LSession.Ask('Continue where we left off.', 256));
```

> [!NOTE]
> 🧩 The `.g4kv` format is versioned. A KV file written by an incompatible engine build is rejected on load rather than silently misread.

<a id="contributing"></a>

## 🤝 Contributing

Gemma4.pas is developed by tinyBigGAMES. Whether you are fixing a bug, improving documentation, sharpening examples, or proposing a feature, contributions are welcome.

| Contribution | Best Way to Help |
|--------------|------------------|
| 🐞 Bug report | Open an issue with a minimal reproduction, the exact code used, and the GPU model |
| 💡 Feature idea | Describe the real use case first, then the proposed API or behavior |
| 🧾 Documentation fix | Point to the section and explain what was unclear or missing |
| 🧪 Test case | Include the smallest code that demonstrates the behavior |
| 🔧 Pull request | Keep the change focused and explain the before/after behavior |

> [!TIP]
> 🚀 Small, focused contributions are the easiest to review and the fastest to land.

## 💖 Support the Project

If Gemma4.pas saves you time, helps you learn, or sparks something useful:

- ⭐ **Star the repo**: it costs nothing and helps others find the project
- 🗣️ **Spread the word**: write a post, mention it in a community, or share a screenshot
- 💬 **Join the community**: show what you are building and help shape what comes next
- 🧪 **Try examples**: real usage finds issues that synthetic tests miss
- 💖 **[Become a sponsor](https://github.com/sponsors/tinyBigGAMES)**: sponsorship directly funds development, examples, and documentation

## 📜 License

Gemma4.pas is licensed under the **Apache License, Version 2.0**. See [LICENSE](https://github.com/tinyBigGAMES/Gemma4.pas?tab=License-1-ov-file#) for details.

Apache 2.0 is a permissive open source license that lets you use, modify, and distribute Gemma4.pas freely in both open source and commercial projects. You are not required to release your own source code. Attribution is required: keep the copyright notice and license file in place.

## 🔗 Links

- 🧑‍💻 [GitHub](https://github.com/tinyBigGAMES/Gemma4.pas)
- 💬 [Discord](https://discord.gg/Wb6z8Wam7p)
- 🦋 [Bluesky](https://bsky.app/profile/tinybiggames.com)
- 🎮 [tinyBigGAMES](https://tinybiggames.com)

<div align="center">

**🚀 Gemma4.pas™** - Local LLM inference in Pascal

Copyright © 2026-present tinyBigGAMES™ LLC<br/>All Rights Reserved.

</div>