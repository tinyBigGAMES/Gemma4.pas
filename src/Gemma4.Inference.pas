{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information

 -------------------------------------------------------------------------------

  Gemma4.Inference - Top-level inference API (replaces Gemma4.Engine)

  Single entry point for loading a Gemma 4 E4B model from a VPK archive
  and running chat inference, modeled on the Cognita TInference pattern
  (.claude\research\congita\Cognita.Inference.pas). Where Cognita
  delegates chat templating to llama.cpp's llama_chat_ex_*, this unit
  renders the model's real chat template (meta/chat_template.jinja in
  the VPK) with the Gemma4.Jinja engine.

  Usage:
    var LInf := TInference.Create();
    LInf.LoadModel('Gemma4.vpk');
    LInf.SetTokenCallback(
      procedure(const AState: TProgressState; const AToken: string;
        const AUserData: Pointer)
      begin
        if AState = psInProgress then
          Write(AToken);
      end, nil);
    LInf.AddMessage(CRoleUser, 'Hello, world!');
    LInf.Generate();
    // LInf.Response      -- full normalized text (with thinking)
    // LInf.ResponseText  -- visible text only (feed back into history)
    LInf.Free();

  Generate() pipeline:
    1. Build a Jinja context from the message history:
       messages, bos_token, add_generation_prompt, enable_thinking
    2. Render the parsed chat template -> exact trained prompt format
    3. Tokenize with special-token awareness (EncodeRaw: <bos>=2,
       <|think|>=98, <|channel>=100, <channel|>=101, ...)
    4. TModel.Generate streams tokens through the thinking rewriter:
       native Gemma 4 markers (<|channel>thought ... <channel|>) are
       normalized to <thinking>/</thinking>, optionally hidden behind a
       placeholder, and an unmatched end tag stops the turn

  Thinking model support:
    EnableThinking=True sets the template's enable_thinking variable,
    which injects <|think|> into the system turn -- the prompt-side
    switch the model was trained on. The reasoning arrives in the
    generated stream inside native markers; the rewriter (ported from
    Cognita.DoRewriteThinking) handles display and normalization.

  Dependencies: StdApp.Base, StdApp.VFS, StdApp.VirtualMemory,
    Gemma4.Types, Gemma4.Config, Gemma4.Tokenizer, Gemma4.Tensors,
    Gemma4.Model, Gemma4.Vulkan, Gemma4.Shaders, Gemma4.Compute,
    Gemma4.Jinja
===============================================================================}

unit Gemma4.Inference;

{$I StdApp.Defines.inc}

interface

uses
  System.SysUtils,
  System.IOUtils,
  System.Classes,
  System.Generics.Collections,
  StdApp.Base,
  StdApp.Console,
  StdApp.JSON,
  StdApp.Resources,
  StdApp.VFS,
  StdApp.VirtualMemory,
  Gemma4.Types,
  Gemma4.Config,
  Gemma4.Tokenizer,
  Gemma4.Tensors,
  Gemma4.Model,
  Gemma4.Attention,
  Gemma4.Vulkan,
  Gemma4.Shaders,
  Gemma4.Compute,
  Gemma4.Jinja,
  Gemma4.Image,
  Gemma4.Vision,
  Gemma4.Audio,
  Gemma4.Video;

const
  // Version info
  CGM4_VERSION_MAJOR = '0';
  CGM4_VERSION_MINOR = '1';
  CGM4_VERSION_PATCH = '0';
  CGM4_VERSION       = CGM4_VERSION_MAJOR + '.' + CGM4_VERSION_MINOR + '.' +
                       CGM4_VERSION_PATCH;

  // Error Codes
  CINF_ERR_CONFIG = 'IN01';
  CINF_ERR_TOKENIZER = 'IN02';
  CINF_ERR_NOT_LOADED = 'IN03';
  CINF_ERR_TEMPLATE = 'IN04';
  CINF_ERR_NO_MESSAGES = 'IN05';
  CINF_ERR_RENDER = 'IN06';
  CINF_ERR_MEDIA = 'IN07';

  // <boi> begin-of-image (also opens each video frame's soft run)
  CTokBoi = 255999;

  // <boa> begin-of-audio
  CTokBoa = 256000;

  // <|image|> -- template marker AND per-row image soft-token slot
  CTokImageSoft = 258880;

  // <|audio|> -- template marker AND per-row audio soft-token slot
  CTokAudioSoft = 258881;

  // <image|> end-of-image
  CTokEoi = 258882;

  // <audio|> end-of-audio
  CTokEoa = 258883;

  // <|video|> -- template marker AND per-row video soft-token slot
  CTokVideoSoft = 258884;

  // Official image soft-token budgets (HF _SUPPORTED_SOFT_TOKENS)
  CImgBudgets: array[0..4] of Integer = (70, 140, 280, 560, 1120);

  // Upper bound for a caller-supplied video frame count (~60 s at 1 fps
  // per the model card guidance; HF ref samples uniformly to N frames)
  CVidMaxFrames = 60;

  CRoleSystem = 'system';
  CRoleUser = 'user';
  CRoleAssistant = 'assistant';
  CRoleTool = 'tool';

  // Literal <bos> marker rendered by the template; EncodeRaw maps it to
  // token ID 2 via the tokenizer's added_tokens table
  CBosTokenText = '<bos>';

  // Native marker opening a reasoning span in the generated stream.
  // Derived from the chat template's own rendering of prior reasoning:
  // '<|channel>thought\n' + text + '\n<channel|>' (strip_thinking macro).
  // Verify against the first live thinking run; a variant emission is a
  // one-constant fix here.
  CThinkNativeStart = '<|channel>thought'#10;

  // Native marker closing a reasoning span (also the end-of-turn signal
  // when it appears unmatched -- Gemma 4 re-emission quirk)
  CThinkNativeEnd = '<channel|>';

  // Bare start-marker text re-injected when the special token arrives:
  // Tokenizer.Decode intentionally skips special tokens, so <|channel>
  // would vanish before the rewriter sees it; "thought\n" follows as
  // ordinary text tokens completing CThinkNativeStart
  CChannelStartText = '<|channel>';

  // Normalized marker emitted in place of the native start marker
  CThinkOpen = '<thinking>'#10;

  // Normalized marker emitted in place of the native end marker
  CThinkClose = #10'</thinking>'#10;

  { CToolQuoteMark }
  // Special-token quote delimiter around string values inside
  // <|tool_call> bodies. Decode drops special tokens, so the Generate
  // bridge re-injects this marker by ID; DoGemma4ArgsToJson treats it
  // as the string delimiter when converting arguments to JSON.
  CToolQuoteMark = '<|"|>';

  // Default display stand-in for a hidden reasoning span (ShowThinking
  // off). Shown once at span start; span content never reaches the
  // token callback. Response always keeps the full normalized text.
  CThinkPlaceholder = 'Thinking...' + sLineBreak;

type
  { TTokenCallback }
  // Streaming callback for generated display text. psPrep fires before
  // template rendering, psStart before the generation loop, psInProgress
  // per emitted display fragment, psEnd when generation completes.
  TTokenCallback = reference to procedure(
    const AState: TProgressState;
    const AToken: string;
    const AUserData: Pointer);

  { TCancelCallback }
  // Polled during generation; return True to stop cleanly
  TCancelCallback = reference to function(
    const AUserData: Pointer): Boolean;

  { TPartKind }
  // Modality of one multimodal message part
  TPartKind = (
    pkText,
    pkImage,
    pkAudio,
    pkVideo
  );

  { TMediaDetail }
  // Level of detail for the convenience AddMessage overload.
  // Maps to image soft-token budgets or video frame counts;
  // ignored for audio.
  TMediaDetail = (
    mdDefault,    // image: 280, video: 32
    mdMinimal,    // image: 70,  video: 8
    mdLow,        // image: 140, video: 16
    mdMedium,     // image: 280, video: 32
    mdHigh,       // image: 560, video: 48
    mdMaximum     // image: 1120, video: 60
  );

  { TMessagePart }
  // One content part of a multimodal turn. Text parts carry Text;
  // media parts carry SourcePath (the file fed to the matching
  // encoder at generation time).
  TMessagePart = record
    Kind: TPartKind;
    Text: string;
    SourcePath: string;
    // Image parts: soft-token budget (one of CImgBudgets).
    // Video parts: frame count (1..CVidMaxFrames).
    // Text/audio parts: 0 = not applicable.
    SoftBudget: Integer;
    class function TextPart(const AText: string): TMessagePart; static;
    class function ImagePart(const APath: string;
      const ASoftBudget: Integer = CImgMaxSoftTokens): TMessagePart; static;
    class function AudioPart(const APath: string): TMessagePart; static;
    class function VideoPart(const APath: string;
      const AFrameCount: Integer = CVidNumFrames): TMessagePart; static;
  end;

  { TChatMessage }
  // One conversation turn. ToolName marks a tool-result turn (role
  // 'tool'); plain turns leave it empty. Tool-call declarations and
  // structured call turns are deferred -- the template skips its tools
  // block when the context variable is unset.
  TChatMessage = record
    Role: string;
    Content: string;
    ToolName: string;
    // Multimodal parts; when non-empty, the template receives content
    // as a sequence of {type[,text]} dicts instead of the Content string
    Parts: TArray<TMessagePart>;
    // Tool-call assistant turn: JSON array of tool_calls objects
    // ([{"function":{"name":"...","arguments":{...}}}]). When non-empty
    // the template receives a tool_calls variable on this message dict.
    ToolCallsJson: string;
  end;

  { TInference }
  // Top-level inference API for Gemma 4 E4B: VPK loading, config,
  // tokenizer, weights, Vulkan compute, chat session state, Jinja chat
  // templating, generation, and thinking-channel normalization.
  TInference = class(TBaseObject)
  private
    // Engine core (transplanted verbatim from Gemma4.Engine.TEngine)
    FConfig: TModelConfig;
    FConfigLoader: TConfigLoader;
    FTokenizer: TTokenizer;
    FWeightStore: TWeightStore;
    FModel: TModel;
    FVulkanDevice: TVulkanDevice;
    FComputeKernels: TComputeKernels;
    FSamplingParams: TSamplingParams;
    FIsLoaded: Boolean;
    FUseGPU: Boolean;

    // Chat session state
    FJinja: TJinja;
    FMessages: TList<TChatMessage>;
    FTemplateLoaded: Boolean;
    // JSON array of tool schemas for the template's tools block
    // ('' = no tools; the template's {%- if tools -%} stays falsy)
    FToolsJson: string;
    FResponse: string;           // full normalized text (with thinking)
    FResponseText: string;       // visible text only (thinking stripped)
    FStats: TInferenceStats;
    FTokenCallback: TCallback<TTokenCallback>;
    FCancelCallback: TCallback<TCancelCallback>;
    FLoadProgress: TCallback<TLoadProgressCallback>;

    // Thinking channel state
    FEnableThinking: Boolean;
    FShowThinking: Boolean;
    FThinkOpenText: string;      // emitted in place of the native start
    FThinkCloseText: string;     // emitted in place of the native end
    FThinkPlaceholder: string;   // display stand-in for a hidden span
    FThinkHold: string;          // hold-back buffer for split markers
    FInThinking: Boolean;        // currently inside a reasoning span
    FThinkTurnDone: Boolean;     // unmatched end tag seen -- turn complete
    FTokChannelStart: Integer;   // <|channel> special token ID
    FTokChannelEnd: Integer;     // <channel|> special token ID
    FTokQuote: Integer;          // <|"|> special token ID (tool-call string delimiter)
    FTokToolCallStart: Integer;  // <|tool_call> special token ID
    FTokToolCallEnd: Integer;    // <tool_call|> special token ID
    FInToolCall: Boolean;        // inside a tool-call span (suppress display)
    FCachedTokenIds: TArray<Integer>; // ids materialized in the KV cache (prefix reuse)
    FMuteTokens: Boolean;        // silence the token callback (internal generations)

    // Multimodal encoders -- created/opened lazily on the first media
    // part of the matching kind; share this instance's compute device
    FVpkPath: string;
    FImagePipeline: TImagePipeline;
    FVision: TVision;
    FAudio: TAudio;

    function DoLoadConfig(const AVpkPath: string): Boolean;
    function DoLoadTokenizer(const AVpkPath: string): Boolean;
    function DoLoadWeights(const AVpkPath: string): Boolean;
    function DoInitModel(): Boolean;
    function DoInitVulkan(): Boolean;
    function DoLoadTemplate(const AVpkPath: string): Boolean;
    function DoReadVpkFileAsString(const AVpkPath: string;
      const AInternalPath: string): string;
    function DoApplyTemplate(): string;
    function DoEnsureVision(): Boolean;
    function DoEnsureAudio(): Boolean;
    function DoExpandImagePart(const APath: string;
      const AIds: TList<Integer>; const ABlocks: TList<TSoftTokenBlock>;
      const ASoftId: Integer; const AMaxSoftTokens: Integer): Boolean;
    function DoExpandAudioPart(const APath: string;
      const AIds: TList<Integer>;
      const ABlocks: TList<TSoftTokenBlock>): Boolean;
    function DoExpandVideoPart(const APath: string;
      const AIds: TList<Integer>;
      const ABlocks: TList<TSoftTokenBlock>;
      const AFrameCount: Integer): Boolean;
    function DoExpandMultimodal(var ATokenIds: TArray<Integer>;
      out ASoftBlocks: TArray<TSoftTokenBlock>): Boolean;
    function DoJsonToJinja(const ANode: TJSON;
      const APool: TJinjaValuePool): TJinjaValue;
    procedure DoResetThinkingState();
    function DoRewriteThinking(const AChunk: string;
      out ADisplay: string; out APlain: string): string;
    procedure DoFireTokenCallback(const AState: TProgressState;
      const AToken: string);
    function IsCancelled(): Boolean;
    function DoGemma4ArgsToJson(const ABody: string): string;
  public
    constructor Create(); override;
    destructor Destroy(); override;

    class function  GetVersionStr(): string;

    procedure SetStatusCallback(const ACallback: TStatusCallback;
      const AUserData: Pointer = nil); override;

    // Load model + chat template from a VPK archive file
    function LoadModel(const AVpkPath: string;
      const AUseGPU: Boolean = True): Boolean;

    // Load and parse ONLY the chat template from a VPK (no weights, no
    // Vulkan) -- for template verification and tooling
    function LoadTemplate(const AVpkPath: string): Boolean;

    // Unload model and free resources
    procedure UnloadModel();

    // Check if model is loaded and ready
    function IsLoaded(): Boolean;

    // Streaming and cancellation callbacks
    procedure SetTokenCallback(const ACallback: TTokenCallback;
      const AUserData: Pointer = nil);
    procedure SetCancelCallback(const ACallback: TCancelCallback;
      const AUserData: Pointer = nil);
    procedure SetLoadProgressCallback(const ACallback: TLoadProgressCallback;
      const AUserData: Pointer = nil);

    // Conversation history
    procedure AddMessage(const ARole: string;
      const AContent: string); overload;
    // Multimodal turn -- parts render in order through the template's
    // content-as-sequence path (text | <|image|> | <|audio|> | <|video|>)
    procedure AddMessage(const ARole: string;
      const AParts: TArray<TMessagePart>); overload;
    // Single-media convenience -- enforces correct part ordering per
    // model card and maps ADetail to the appropriate budget/frame count.
    // pkImage/pkVideo before text; pkAudio after text.
    procedure AddMessage(const ARole: string; const AMediaKind: TPartKind;
      const AMediaPath, AText: string;
      const ADetail: TMediaDetail = mdDefault); overload;
    // Tool-result turn (role 'tool') -- AContent is the tool's output
    procedure AddToolResult(const AToolName: string; const AContent: string);
    procedure ClearMessages();

    // Tool definitions for the template's tools block: a JSON array of
    // HF-style tool schemas ([{"type":"function","function":{...}}, ...]).
    // Pass '' to clear. Invalid or non-array JSON is rejected with an error.
    procedure SetTools(const AToolsJson: string);

    // Render the chat template over the full history and generate.
    // Streams display text via the token callback; the full normalized
    // text lands in Response, the visible-only text in ResponseText.
    function Generate(const AMaxTokens: Integer = 2048): Boolean;

    // Render the chat template over the current message history without
    // generating -- the exact prompt text Generate() tokenizes
    function RenderPrompt(): string;

    // Reset conversation state (clears KV cache)
    procedure ResetConversation();

    // Configure sampling parameters
    procedure SetTemperature(const AValue: Single);
    procedure SetTopK(const AValue: Integer);
    procedure SetTopP(const AValue: Single);

    // Info
    function GetModelName(): string;
    function GetDeviceName(): string;
    function GetVocabSize(): Integer;

    // Session support: token counting and context budget
    function TokenCount(const AText: string): Integer;
    function ContextSize(): Integer;

    // KV state persistence -- save/load the GPU KV cache to/from a file
    function SaveKVState(const AFileName: string): Boolean;
    function LoadKVState(const AFileName: string): Boolean;

    // Tool-call parsing: extract <|tool_call>...<tool_call|> markers
    // from the last response and return a JSON array of tool_call
    // objects ([{"function":{"name":"...","arguments":{...}}}]).
    // Returns '' when no tool calls are present.
    function ParseToolCalls(): string;

    // Add an assistant turn that carries tool_calls. AToolCallsJson is
    // the JSON array returned by ParseToolCalls. The template will
    // re-render the <|tool_call> markers from this data on the next
    // Generate.
    procedure AddToolCallTurn(const AToolCallsJson: string);

    property Response: string read FResponse;
    property ResponseText: string read FResponseText;
    property Messages: TList<TChatMessage> read FMessages;
    // Statistics from the last Generate call (see TInferenceStats in
    // Gemma4.Types; use Stats.FormatText() for the standard report)
    property Stats: TInferenceStats read FStats;
    property SamplingParams: TSamplingParams read FSamplingParams write FSamplingParams;

    // Thinking/reasoning control (Gemma 4 thinking channel)
    property EnableThinking: Boolean read FEnableThinking write FEnableThinking;
    // False hides the reasoning span from the token callback behind
    // ThinkingPlaceholderText (shown once per span). Display-only --
    // Response and ResponseText are unaffected.
    property ShowThinking: Boolean read FShowThinking write FShowThinking;
    // Runtime-configurable text emitted in place of the native thinking
    // markers. Set both to '' to strip the markers entirely (reasoning
    // content still streams when ShowThinking is on).
    property ThinkingOpenText: string read FThinkOpenText write FThinkOpenText;
    property ThinkingCloseText: string read FThinkCloseText write FThinkCloseText;
    property ThinkingPlaceholderText: string read FThinkPlaceholder write FThinkPlaceholder;
    // True silences the token callback entirely -- used for internal
    // generations (e.g. history summarization) that must not stream to
    // the display. The registered callback itself is left untouched.
    property MuteTokens: Boolean read FMuteTokens write FMuteTokens;
  end;

implementation


{ TInference }

constructor TInference.Create();
begin
  inherited Create();
  FConfigLoader := TConfigLoader.Create();
  FTokenizer := TTokenizer.Create();
  FWeightStore := TWeightStore.Create();
  FModel := TModel.Create();
  FVulkanDevice := TVulkanDevice.Create();
  FComputeKernels := TComputeKernels.Create();
  FIsLoaded := False;
  FUseGPU := True;
  FSamplingParams.SetDefaults();

  FJinja := TJinja.Create();
  FJinja.SetErrors(GetErrors());
  FMessages := TList<TChatMessage>.Create();
  FTemplateLoaded := False;
  FToolsJson := '';
  FResponse := '';
  FResponseText := '';
  FStats := Default(TInferenceStats);
  FTokenCallback := Default(TCallback<TTokenCallback>);
  FCancelCallback := Default(TCallback<TCancelCallback>);

  FEnableThinking := True;
  FShowThinking := True;
  FThinkOpenText := CThinkOpen;
  FThinkCloseText := CThinkClose;
  FThinkPlaceholder := CThinkPlaceholder;
  FTokChannelStart := -1;
  FTokChannelEnd := -1;
  FTokQuote := -1;
  FTokToolCallStart := -1;
  FTokToolCallEnd := -1;
  FInToolCall := False;
  FCachedTokenIds := nil;
  FMuteTokens := False;
  DoResetThinkingState();
end;

destructor TInference.Destroy();
begin
  UnloadModel();
  FMessages.Free();
  FJinja.Free();
  // Reverse dependency order: FModel holds a non-owned reference to
  // FComputeKernels and releases its GPU buffers in its destructor, so it
  // MUST be freed while the kernels object is still alive. Freeing the
  // kernels/device first leaves FModel with a dangling pointer -> AV.
  FModel.Free();
  FComputeKernels.Free();
  FVulkanDevice.Free();
  FWeightStore.Free();
  FTokenizer.Free();
  FConfigLoader.Free();
  inherited;
end;

class function TInference.GetVersionStr: string;
begin
  Result := CGM4_VERSION;
end;

procedure TInference.SetStatusCallback(const ACallback: TStatusCallback;
  const AUserData: Pointer);
begin
  inherited SetStatusCallback(ACallback, AUserData);
  FConfigLoader.SetStatusCallback(ACallback, AUserData);
  FTokenizer.SetStatusCallback(ACallback, AUserData);
  FWeightStore.SetStatusCallback(ACallback, AUserData);
  FModel.SetStatusCallback(ACallback, AUserData);
  FVulkanDevice.SetStatusCallback(ACallback, AUserData);
  FComputeKernels.SetStatusCallback(ACallback, AUserData);
end;

function TInference.DoReadVpkFileAsString(const AVpkPath: string;
  const AInternalPath: string): string;
var
  LVFS: TVFS;
  LView: TVirtualMemoryView<Byte>;
  LBytes: TBytes;
begin
  Result := '';
  LVFS := TVFS.Create();
  try
    LVFS.SetErrors(GetErrors());
    if not LVFS.Open(AVpkPath) then
      Exit;

    if not LVFS.FileExists(AInternalPath) then
    begin
      LVFS.Close();
      Exit;
    end;

    LView := LVFS.OpenFile(AInternalPath);
    try
      SetLength(LBytes, LView.Size);
      if LView.Size > 0 then
        LView.Read(LBytes[0], LView.Size);
      Result := TEncoding.UTF8.GetString(LBytes);
    finally
      LView.Free();
    end;

    LVFS.Close();
  finally
    LVFS.Free();
  end;
end;

function TInference.DoLoadConfig(const AVpkPath: string): Boolean;
var
  LConfigStr: string;
begin
  Result := False;

  LConfigStr := DoReadVpkFileAsString(AVpkPath, CVpkConfigPath);
  if LConfigStr = '' then
  begin
    GetErrors().Add(esError, CINF_ERR_CONFIG,
      'Failed to read config.json from VPK');
    Exit;
  end;

  FConfigLoader.SetErrors(GetErrors());
  Result := FConfigLoader.LoadFromString(LConfigStr, FConfig);

  if Result then
    Status('Config loaded: %d layers, hidden=%d, vocab=%d', [
      FConfig.NumHiddenLayers, FConfig.HiddenSize, FConfig.VocabSize]);
end;

function TInference.DoLoadTokenizer(const AVpkPath: string): Boolean;
var
  LTokenizerStr: string;
begin
  Result := False;

  LTokenizerStr := DoReadVpkFileAsString(AVpkPath, CVpkTokenizerPath);
  if LTokenizerStr = '' then
  begin
    GetErrors().Add(esError, CINF_ERR_TOKENIZER,
      'Failed to read tokenizer.json from VPK');
    Exit;
  end;

  FTokenizer.SetErrors(GetErrors());
  Result := FTokenizer.LoadFromString(LTokenizerStr);

  if Result then
  begin
    // Resolve the thinking-channel marker IDs once: Decode skips special
    // tokens, so Generate's bridge re-injects these by ID
    FTokChannelStart := FTokenizer.TokenToId(CChannelStartText);
    FTokChannelEnd := FTokenizer.TokenToId(CThinkNativeEnd);
    FTokToolCallStart := FTokenizer.TokenToId('<|tool_call>');
    FTokToolCallEnd := FTokenizer.TokenToId('<tool_call|>');
    FTokQuote := FTokenizer.TokenToId(CToolQuoteMark);
    Status('Tokenizer loaded: %d vocab', [FTokenizer.GetVocabSize()]);
  end;
end;

function TInference.DoLoadWeights(const AVpkPath: string): Boolean;
begin
  FWeightStore.SetErrors(GetErrors());
  Result := FWeightStore.Open(AVpkPath);

  if Result then
    Status('Weights loaded: %d tensors', [FWeightStore.TensorCount()]);
end;

function TInference.DoInitModel(): Boolean;
begin
  FModel.SetErrors(GetErrors());
  if FLoadProgress.IsAssigned() then
    FModel.SetLoadProgressCallback(FLoadProgress.Callback,
      FLoadProgress.UserData);
  Result := FModel.Load(FWeightStore, FConfig, False, FComputeKernels);

  if Result then
  begin
    if FModel.UseGPU then
      Status('Model initialized with GPU: %s', [CModelName])
    else
      Status('Model initialized (CPU): %s', [CModelName]);
  end;
end;

function TInference.DoInitVulkan(): Boolean;
begin
  if not FUseGPU then
  begin
    Status('GPU disabled, using CPU inference');
    Result := True;
    Exit;
  end;

  FVulkanDevice.SetErrors(GetErrors());
  if not FVulkanDevice.Init() then
  begin
    Status('Vulkan init failed, falling back to CPU');
    FUseGPU := False;
    Result := True; // non-fatal, CPU fallback
    Exit;
  end;

  Status('Vulkan initialized: %s', [FVulkanDevice.DeviceName]);

  FComputeKernels.SetErrors(GetErrors());
  if not FComputeKernels.Init(FVulkanDevice) then
  begin
    Status('Compute kernel init failed, falling back to CPU');
    FVulkanDevice.Shutdown();
    FUseGPU := False;
    Result := True; // non-fatal
    Exit;
  end;

  Status('Compute kernels initialized');
  Result := True;
end;

function TInference.DoLoadTemplate(const AVpkPath: string): Boolean;
var
  LTemplate: string;
begin
  Result := False;
  FTemplateLoaded := False;

  LTemplate := DoReadVpkFileAsString(AVpkPath, CVpkChatTemplatePath);
  if LTemplate = '' then
  begin
    GetErrors().Add(esError, CINF_ERR_TEMPLATE,
      'Failed to read %s from VPK', [CVpkChatTemplatePath]);
    Exit;
  end;

  if not FJinja.Parse(LTemplate) then
  begin
    GetErrors().Add(esError, CINF_ERR_TEMPLATE,
      'Failed to parse chat template');
    Exit;
  end;

  FTemplateLoaded := True;
  Status('Chat template loaded and parsed (%d chars)', [Length(LTemplate)]);
  Result := True;
end;

function TInference.LoadModel(const AVpkPath: string;
  const AUseGPU: Boolean): Boolean;
begin
  Result := False;
  UnloadModel();
  FUseGPU := AUseGPU;
  // Kept for the lazy multimodal encoder opens (encoders.bin lives in
  // the same VPK)
  FVpkPath := AVpkPath;

  Status('Loading model from: %s', [AVpkPath]);

  if FLoadProgress.IsAssigned() then
    FLoadProgress.Callback(psStart, 0, 0, FLoadProgress.UserData);

  if not DoLoadConfig(AVpkPath) then Exit;
  if not DoLoadTokenizer(AVpkPath) then Exit;
  if not DoLoadWeights(AVpkPath) then Exit;
  if not DoInitVulkan() then Exit;
  if not DoInitModel() then Exit;
  if not DoLoadTemplate(AVpkPath) then Exit;

  FIsLoaded := True;
  Status('Model ready: %s', [CModelName]);

  if FLoadProgress.IsAssigned() then
    FLoadProgress.Callback(psEnd, 0, 0, FLoadProgress.UserData);

  Result := True;
end;

procedure TInference.UnloadModel();
begin
  // Multimodal encoders share this instance's compute device -- close
  // and free them BEFORE the kernels/device shut down
  if FVision <> nil then
  begin
    FVision.Close();
    FreeAndNil(FVision);
  end;
  if FAudio <> nil then
  begin
    FAudio.Close();
    FreeAndNil(FAudio);
  end;
  FreeAndNil(FImagePipeline);
  FVpkPath := '';

  // Free the model's GPU buffers while the device is still initialized.
  // FComputeKernels.Shutdown() below nils its device reference, after which
  // TModel.DoFreeGpuBuffers early-exits and the buffers would leak (the ~4.3 GB
  // weight buffer plus working/KV/batch buffers). Free first, then shut down.
  FModel.FreeGpuResources();

  if FComputeKernels.IsInitialized() then
    FComputeKernels.Shutdown();

  if FVulkanDevice.IsInitialized() then
    FVulkanDevice.Shutdown();

  FModel.Reset();
  FCachedTokenIds := nil;
  FWeightStore.Close();
  FTemplateLoaded := False;
  FIsLoaded := False;
end;

function TInference.IsLoaded(): Boolean;
begin
  Result := FIsLoaded;
end;

procedure TInference.SetTokenCallback(const ACallback: TTokenCallback;
  const AUserData: Pointer);
begin
  FTokenCallback.Callback := ACallback;
  FTokenCallback.UserData := AUserData;
end;

procedure TInference.SetCancelCallback(const ACallback: TCancelCallback;
  const AUserData: Pointer);
begin
  FCancelCallback.Callback := ACallback;
  FCancelCallback.UserData := AUserData;
end;

procedure TInference.SetLoadProgressCallback(
  const ACallback: TLoadProgressCallback;
  const AUserData: Pointer);
begin
  FLoadProgress.Callback := ACallback;
  FLoadProgress.UserData := AUserData;
end;

procedure TInference.DoFireTokenCallback(const AState: TProgressState;
  const AToken: string);
begin
  // Internal generations (summarization) run muted -- nothing streams
  if FMuteTokens then
    Exit;
  if FTokenCallback.IsAssigned() then
    FTokenCallback.Callback(AState, AToken, FTokenCallback.UserData);
end;

function TInference.IsCancelled(): Boolean;
begin
  Result := FCancelCallback.IsAssigned() and
    FCancelCallback.Callback(FCancelCallback.UserData);
end;

{ TMessagePart }

class function TMessagePart.TextPart(const AText: string): TMessagePart;
begin
  Result := Default(TMessagePart);
  Result.Kind := pkText;
  Result.Text := AText;
end;

class function TMessagePart.ImagePart(const APath: string;
  const ASoftBudget: Integer): TMessagePart;
begin
  Result := Default(TMessagePart);
  Result.Kind := pkImage;
  Result.SourcePath := APath;
  Result.SoftBudget := ASoftBudget;
end;

class function TMessagePart.AudioPart(const APath: string): TMessagePart;
begin
  Result := Default(TMessagePart);
  Result.Kind := pkAudio;
  Result.SourcePath := APath;
end;

class function TMessagePart.VideoPart(const APath: string;
  const AFrameCount: Integer): TMessagePart;
begin
  Result := Default(TMessagePart);
  Result.Kind := pkVideo;
  Result.SourcePath := APath;
  // Same slot as the image budget; Kind disambiguates
  Result.SoftBudget := AFrameCount;
end;

procedure TInference.AddMessage(const ARole: string; const AContent: string);
var
  LMsg: TChatMessage;
begin
  LMsg := Default(TChatMessage);
  LMsg.Role := ARole;
  LMsg.Content := AContent;
  FMessages.Add(LMsg);
end;

procedure TInference.AddMessage(const ARole: string;
  const AParts: TArray<TMessagePart>);
var
  LMsg: TChatMessage;
begin
  LMsg := Default(TChatMessage);
  LMsg.Role := ARole;
  LMsg.Parts := AParts;
  FMessages.Add(LMsg);
end;

procedure TInference.AddMessage(const ARole: string;
  const AMediaKind: TPartKind; const AMediaPath, AText: string;
  const ADetail: TMediaDetail);
const
  CImgDetailBudgets: array[TMediaDetail] of Integer =
    (280, 70, 140, 280, 560, 1120);
  CVidDetailFrames: array[TMediaDetail] of Integer =
    (32, 8, 16, 32, 48, 60);
begin
  case AMediaKind of
    pkText:
      AddMessage(ARole, AText);
    pkImage:
      AddMessage(ARole, [
        TMessagePart.ImagePart(AMediaPath, CImgDetailBudgets[ADetail]),
        TMessagePart.TextPart(AText)
      ]);
    pkAudio:
      AddMessage(ARole, [
        TMessagePart.TextPart(AText),
        TMessagePart.AudioPart(AMediaPath)
      ]);
    pkVideo:
      AddMessage(ARole, [
        TMessagePart.VideoPart(AMediaPath, CVidDetailFrames[ADetail]),
        TMessagePart.TextPart(AText)
      ]);
  else
    begin
      FErrors.RaiseOnError := True;
      FErrors.Add(esError, CINF_ERR_MEDIA, RSInfUnsupportedMediaKind);
    end;
  end;
end;

procedure TInference.AddToolResult(const AToolName: string;
  const AContent: string);
var
  LMsg: TChatMessage;
begin
  LMsg := Default(TChatMessage);
  LMsg.Role := CRoleTool;
  LMsg.ToolName := AToolName;
  LMsg.Content := AContent;
  FMessages.Add(LMsg);
end;

procedure TInference.ClearMessages();
begin
  FMessages.Clear();
end;

function TInference.TokenCount(const AText: string): Integer;
begin
  Result := 0;
  if (AText = '') or (not FTokenizer.IsLoaded()) then
    Exit;
  Result := Length(FTokenizer.EncodeRaw(AText));
end;

function TInference.ContextSize(): Integer;
begin
  Result := CGpuMaxSeq;
end;

function TInference.SaveKVState(const AFileName: string): Boolean;
begin
  Result := False;
  if not FIsLoaded then
  begin
    FErrors.Add(esError, CINF_ERR_NOT_LOADED, 'Model not loaded');
    Exit;
  end;
  Result := FModel.SaveKVState(AFileName);
end;

function TInference.LoadKVState(const AFileName: string): Boolean;
begin
  Result := False;
  if not FIsLoaded then
  begin
    FErrors.Add(esError, CINF_ERR_NOT_LOADED, 'Model not loaded');
    Exit;
  end;
  Result := FModel.LoadKVState(AFileName);
  // A restored blob's token ids are unknown -- the next Generate must
  // re-prefill fully rather than diff against a stale id list
  FCachedTokenIds := nil;
end;

procedure TInference.AddToolCallTurn(const AToolCallsJson: string);
var
  LMsg: TChatMessage;
begin
  LMsg := Default(TChatMessage);
  LMsg.Role := CRoleAssistant;
  LMsg.ToolCallsJson := AToolCallsJson;
  FMessages.Add(LMsg);
end;

function TInference.DoGemma4ArgsToJson(const ABody: string): string;
var
  LSb: TStringBuilder;
  LLen: Integer;
  LI: Integer;
  LStart: Integer;
  LClose: Integer;
  LDepth: Integer;
  LKey: string;
  LValue: string;
  LFirst: Boolean;

  function EscapeJson(const AText: string): string;
  var
    LEsc: TStringBuilder;
    LJ: Integer;
    LCh: Char;
  begin
    LEsc := TStringBuilder.Create();
    try
      for LJ := 0 to Length(AText) - 1 do
      begin
        LCh := AText.Chars[LJ];
        case LCh of
          '"': LEsc.Append('\"');
          '\': LEsc.Append('\\');
          #8:  LEsc.Append('\b');
          #9:  LEsc.Append('\t');
          #10: LEsc.Append('\n');
          #12: LEsc.Append('\f');
          #13: LEsc.Append('\r');
        else
          if Ord(LCh) < $20 then
            LEsc.Append(Format('\u%.4x', [Ord(LCh)]))
          else
            LEsc.Append(LCh);
        end;
      end;
      Result := LEsc.ToString();
    finally
      LEsc.Free();
    end;
  end;

  function IsJsonLiteral(const AValue: string): Boolean;
  var
    LNum: Double;
  begin
    // Bare JSON keywords and numbers pass through unquoted
    Result := (AValue = 'true') or (AValue = 'false') or (AValue = 'null') or
      TryStrToFloat(AValue, LNum, TFormatSettings.Invariant);
  end;

  procedure SkipWhitespace();
  begin
    while (LI < LLen) and (ABody.Chars[LI] <= ' ') do
      Inc(LI);
  end;

begin
  // Convert the Gemma 4 tool-call argument format to strict JSON.
  //   Input:  location:<|"|>London<|"|>,unit:<|"|>celsius<|"|>,count:3
  //   Output: {"location":"London","unit":"celsius","count":3}
  // The <|"|> special-token markers delimit string values (re-injected
  // by ID in the Generate bridge -- Decode drops special tokens). Their
  // raw content is emitted as a properly escaped JSON string, so
  // interior quotes, newlines, and backslashes in payloads such as
  // run_script code survive. Bare values are emitted unquoted only when
  // they are valid JSON literals (number/true/false/null); anything
  // else is escaped as a string.
  LSb := TStringBuilder.Create();
  try
    LSb.Append('{');
    LFirst := True;
    LLen := Length(ABody);
    LI := 0;
    while LI < LLen do
    begin
      SkipWhitespace();
      if LI >= LLen then
        Break;

      // Key: everything up to the next ':' (tolerate quoted keys)
      LStart := LI;
      while (LI < LLen) and (ABody.Chars[LI] <> ':') do
        Inc(LI);
      LKey := ABody.Substring(LStart, LI - LStart).Trim();
      LKey := LKey.Replace(CToolQuoteMark, '').Replace('"', '').Trim();

      // Dangling key with no value -- drop it and stop
      if LI >= LLen then
        Break;
      Inc(LI); // consume ':'
      SkipWhitespace();

      if (LI + Length(CToolQuoteMark) <= LLen) and
         (ABody.Substring(LI, Length(CToolQuoteMark)) = CToolQuoteMark) then
      begin
        // Marker-delimited string: raw content up to the closing marker
        LI := LI + Length(CToolQuoteMark);
        LClose := ABody.IndexOf(CToolQuoteMark, LI);
        if LClose < 0 then
          LClose := LLen; // unterminated -- take the remainder
        LValue := '"' + EscapeJson(ABody.Substring(LI, LClose - LI)) + '"';
        LI := LClose + Length(CToolQuoteMark);
        if LI > LLen then
          LI := LLen;
        SkipWhitespace();
        if (LI < LLen) and (ABody.Chars[LI] = ',') then
          Inc(LI); // consume the pair separator
      end
      else
      begin
        // Bare value: scan to the next top-level comma, tracking bracket
        // nesting so nested payloads stay intact
        LStart := LI;
        LDepth := 0;
        while LI < LLen do
        begin
          case ABody.Chars[LI] of
            '{', '[', '(': Inc(LDepth);
            '}', ']', ')': Dec(LDepth);
            ',': if LDepth <= 0 then
                   Break;
          end;
          Inc(LI);
        end;
        LValue := ABody.Substring(LStart, LI - LStart).Trim();
        if (LI < LLen) and (ABody.Chars[LI] = ',') then
          Inc(LI); // consume the pair separator

        // Already double-quoted values are re-escaped for safety; other
        // non-literal values become escaped JSON strings
        if (LValue.Length >= 2) and LValue.StartsWith('"') and
           LValue.EndsWith('"') then
          LValue := '"' +
            EscapeJson(LValue.Substring(1, LValue.Length - 2)) + '"'
        else if not IsJsonLiteral(LValue) then
          LValue := '"' + EscapeJson(LValue) + '"';
      end;

      if LKey <> '' then
      begin
        if not LFirst then
          LSb.Append(',');
        LSb.Append('"');
        LSb.Append(EscapeJson(LKey));
        LSb.Append('":');
        LSb.Append(LValue);
        LFirst := False;
      end;
    end;
    LSb.Append('}');
    Result := LSb.ToString();
  finally
    LSb.Free();
  end;
end;

function TInference.ParseToolCalls(): string;
var
  LText: string;
  LOpenTag: string;
  LCloseTag: string;
  LStart: Integer;
  LEnd: Integer;
  LBody: string;
  LName: string;
  LArgs: string;
  LArgsJson: string;
  LBraceStart: Integer;
  LBraceEnd: Integer;
  LCallIdx: Integer;
  LSb: TStringBuilder;
begin
  Result := '';
  LText := FResponse;
  if LText = '' then
    Exit;

  LOpenTag := '<|tool_call>';
  LCloseTag := '<tool_call|>';

  // Check if any tool calls exist
  if not LText.Contains(LOpenTag) then
    Exit;

  LSb := TStringBuilder.Create();
  try
    LSb.Append('[');
    LCallIdx := 0;
    LStart := LText.IndexOf(LOpenTag);
    while LStart >= 0 do
    begin
      LEnd := LText.IndexOf(LCloseTag, LStart + Length(LOpenTag));
      if LEnd < 0 then
        Break;

      LBody := LText.Substring(LStart + Length(LOpenTag),
        LEnd - LStart - Length(LOpenTag));

      // Expected format: call:funcname{key:val,...}
      if LBody.StartsWith('call:') then
      begin
        LBody := LBody.Substring(5); // skip 'call:'
        LBraceStart := LBody.IndexOf('{');
        if LBraceStart >= 0 then
        begin
          LName := LBody.Substring(0, LBraceStart);
          // Find matching closing brace (last '}')
          LBraceEnd := LBody.LastIndexOf('}');
          if LBraceEnd > LBraceStart then
          begin
            LArgs := LBody.Substring(LBraceStart + 1,
              LBraceEnd - LBraceStart - 1);
            LArgsJson := DoGemma4ArgsToJson(LArgs);
          end
          else
            LArgsJson := '{}';
        end
        else
        begin
          LName := LBody;
          LArgsJson := '{}';
        end;

        if LCallIdx > 0 then
          LSb.Append(',');
        LSb.Append('{"function":{"name":"');
        LSb.Append(LName);
        LSb.Append('","arguments":');
        LSb.Append(LArgsJson);
        LSb.Append('},"id":"call_');
        LSb.Append(IntToStr(LCallIdx));
        LSb.Append('"}');
        Inc(LCallIdx);
      end;

      LStart := LText.IndexOf(LOpenTag, LEnd + Length(LCloseTag));
    end;
    LSb.Append(']');

    if LCallIdx > 0 then
      Result := LSb.ToString();
  finally
    LSb.Free();
  end;
end;

function TInference.DoApplyTemplate(): string;
var
  LCtx: TJinjaContext;
  LPool: TJinjaValuePool;
  LMsgs: TJinjaValue;
  LDict: TJinjaValue;
  LParts: TJinjaValue;
  LPart: TJinjaValue;
  LIdx: Integer;
  LPartIdx: Integer;
  LToolsJson: TJSON;
begin
  // Fresh root context (owns its value pool) per render; the rendered
  // string is copied out, so freeing the context afterwards is safe
  LCtx := TJinjaContext.Create();
  try
    LPool := LCtx.GetPool();

    // messages: array of {role, content} dicts. Tool-result turns carry
    // their name (the template resolves it via message.get('name')).
    LMsgs := LPool.NewArray();
    for LIdx := 0 to FMessages.Count - 1 do
    begin
      LDict := LPool.NewDict();
      LDict.DictSet('role', LPool.NewString(FMessages[LIdx].Role));
      if Length(FMessages[LIdx].Parts) > 0 then
      begin
        // Multimodal turn: content as a sequence of {type[,text]} dicts.
        // The template renders text parts via item['text'] | trim and
        // media parts as <|image|>/<|audio|>/<|video|> markers in order
        LParts := LPool.NewArray();
        for LPartIdx := 0 to High(FMessages[LIdx].Parts) do
        begin
          LPart := LPool.NewDict();
          case FMessages[LIdx].Parts[LPartIdx].Kind of
            pkText:
            begin
              LPart.DictSet('type', LPool.NewString('text'));
              LPart.DictSet('text',
                LPool.NewString(FMessages[LIdx].Parts[LPartIdx].Text));
            end;
            pkImage:
              LPart.DictSet('type', LPool.NewString('image'));
            pkAudio:
              LPart.DictSet('type', LPool.NewString('audio'));
            pkVideo:
              LPart.DictSet('type', LPool.NewString('video'));
          end;
          LParts.ArrayAdd(LPart);
        end;
        LDict.DictSet('content', LParts);
      end
      else
        LDict.DictSet('content', LPool.NewString(FMessages[LIdx].Content));
      if FMessages[LIdx].ToolName <> '' then
        LDict.DictSet('name', LPool.NewString(FMessages[LIdx].ToolName));
      // Tool-call assistant turn: inject parsed tool_calls array so
      // the template can re-render <|tool_call> markers on replay
      if FMessages[LIdx].ToolCallsJson <> '' then
      begin
        LToolsJson := TJSON.FromString(FMessages[LIdx].ToolCallsJson);
        try
          LDict.DictSet('tool_calls', DoJsonToJinja(LToolsJson, LPool));
        finally
          LToolsJson.Free();
        end;
      end;
      LMsgs.ArrayAdd(LDict);
    end;

    LCtx.SetLocal('messages', LMsgs);
    LCtx.SetLocal('bos_token', LPool.NewString(CBosTokenText));
    LCtx.SetLocal('add_generation_prompt', LPool.NewBool(True));
    // The template's prompt-side thinking switch: when truthy it injects
    // <|think|> into the system turn ({%- if enable_thinking is defined
    // and enable_thinking -%})
    LCtx.SetLocal('enable_thinking', LPool.NewBool(FEnableThinking));
    // tools: parsed fresh per render into pool-owned Jinja values.
    // Left unset when empty -- the template's tools block is gated on
    // {%- if tools -%} and an undefined variable is falsy
    if FToolsJson <> '' then
    begin
      LToolsJson := TJSON.FromString(FToolsJson);
      try
        LCtx.SetLocal('tools', DoJsonToJinja(LToolsJson, LPool));
      finally
        LToolsJson.Free();
      end;
    end;

    Result := FJinja.Render(LCtx);
  finally
    LCtx.Free();
  end;
end;

function TInference.DoEnsureVision(): Boolean;
begin
  Result := True;
  if FImagePipeline = nil then
    FImagePipeline := TImagePipeline.Create();
  if FVision = nil then
  begin
    FVision := TVision.Create();
    if not FVision.Open(FVpkPath, FComputeKernels) then
    begin
      GetErrors().Add(esError, CINF_ERR_MEDIA,
        'Failed to open vision encoder from VPK');
      FVision.PrintErrors();
      FreeAndNil(FVision);
      Exit(False);
    end;
  end;
end;

function TInference.DoEnsureAudio(): Boolean;
begin
  Result := True;
  if FAudio = nil then
  begin
    FAudio := TAudio.Create();
    if not FAudio.Open(FVpkPath, FComputeKernels) then
    begin
      GetErrors().Add(esError, CINF_ERR_MEDIA,
        'Failed to open audio encoder from VPK');
      FAudio.PrintErrors();
      FreeAndNil(FAudio);
      Exit(False);
    end;
  end;
end;

function TInference.DoExpandImagePart(const APath: string;
  const AIds: TList<Integer>; const ABlocks: TList<TSoftTokenBlock>;
  const ASoftId: Integer; const AMaxSoftTokens: Integer): Boolean;
var
  LPatches: TArray<Single>;
  LPositions: TArray<Integer>;
  LNumSoft: Integer;
  LSoftTokens: TArray<Single>;
  LBlock: TSoftTokenBlock;
  LI: Integer;
  LValidBudget: Boolean;
  LBudget: Integer;
begin
  Result := False;

  // Only the official Gemma 4 budgets are supported (HF enforces the
  // same set); anything else is a caller error, not a resize target
  LValidBudget := False;
  for LBudget in CImgBudgets do
  begin
    if LBudget = AMaxSoftTokens then
    begin
      LValidBudget := True;
      Break;
    end;
  end;
  if not LValidBudget then
  begin
    GetErrors().Add(esError, CINF_ERR_MEDIA,
      'Invalid image soft-token budget %d (allowed: 70/140/280/560/1120)',
      [AMaxSoftTokens]);
    Exit;
  end;

  if not DoEnsureVision() then
    Exit;

  if not FImagePipeline.ProcessFile(APath, LPatches, LPositions, LNumSoft,
    AMaxSoftTokens) then
  begin
    GetErrors().Add(esError, CINF_ERR_MEDIA,
      'Image processing failed: %s', [APath]);
    FImagePipeline.PrintErrors();
    Exit;
  end;

  if not FVision.EncodeImage(LPatches, LPositions, LSoftTokens) then
  begin
    GetErrors().Add(esError, CINF_ERR_MEDIA,
      'Vision encoding failed: %s', [APath]);
    FVision.PrintErrors();
    Exit;
  end;

  // <boi> + N soft-token slots + <eoi>; rows substitute the slot run
  AIds.Add(CTokBoi);
  LBlock := Default(TSoftTokenBlock);
  LBlock.Position := AIds.Count;
  LBlock.RowCount := LNumSoft;
  LBlock.Rows := LSoftTokens;
  for LI := 0 to LNumSoft - 1 do
    AIds.Add(ASoftId);
  AIds.Add(CTokEoi);
  ABlocks.Add(LBlock);
  Result := True;
end;

function TInference.DoExpandAudioPart(const APath: string;
  const AIds: TList<Integer>;
  const ABlocks: TList<TSoftTokenBlock>): Boolean;
var
  LSamples: TArray<Single>;
  LSoftTokens: TArray<Single>;
  LNumTokens: Integer;
  LBlock: TSoftTokenBlock;
  LI: Integer;
begin
  Result := False;
  if not DoEnsureAudio() then
    Exit;

  if not TAudio.LoadWaveFile(APath, LSamples) then
  begin
    GetErrors().Add(esError, CINF_ERR_MEDIA,
      'WAV load failed (missing or unsupported format): %s', [APath]);
    Exit;
  end;

  if not FAudio.EncodeAudio(LSamples, LSoftTokens, LNumTokens) then
  begin
    GetErrors().Add(esError, CINF_ERR_MEDIA,
      'Audio encoding failed: %s', [APath]);
    FAudio.PrintErrors();
    Exit;
  end;

  // <boa> + N soft-token slots + <eoa>
  AIds.Add(CTokBoa);
  LBlock := Default(TSoftTokenBlock);
  LBlock.Position := AIds.Count;
  LBlock.RowCount := LNumTokens;
  LBlock.Rows := LSoftTokens;
  for LI := 0 to LNumTokens - 1 do
    AIds.Add(CTokAudioSoft);
  AIds.Add(CTokEoa);
  ABlocks.Add(LBlock);
  Result := True;
end;

function TInference.DoExpandVideoPart(const APath: string;
  const AIds: TList<Integer>;
  const ABlocks: TList<TSoftTokenBlock>;
  const AFrameCount: Integer): Boolean;
var
  LVideo: TVideo;
  LFrames: TArray<TVideoFrame>;
  LFrame: Integer;
  LMins: Integer;
  LSecs: Integer;
  LFragText: string;
  LPatches: TArray<Single>;
  LPositions: TArray<Integer>;
  LNumSoft: Integer;
  LSoftTokens: TArray<Single>;
  LBlock: TSoftTokenBlock;
  LI: Integer;
begin
  Result := False;

  // Frame count bounded by the model card's ~60-frame (1 fps) guidance
  if (AFrameCount < 1) or (AFrameCount > CVidMaxFrames) then
  begin
    GetErrors().Add(esError, CINF_ERR_MEDIA,
      'Invalid video frame count %d (allowed: 1..%d)',
      [AFrameCount, CVidMaxFrames]);
    Exit;
  end;

  if not DoEnsureVision() then
    Exit;

  LVideo := TVideo.Create();
  try
    if not LVideo.ExtractFrames(APath, AFrameCount, LFrames) then
    begin
      GetErrors().Add(esError, CINF_ERR_MEDIA,
        'Video frame extraction failed: %s', [APath]);
      LVideo.PrintErrors();
      Exit;
    end;
  finally
    LVideo.Free();
  end;

  // Canonical HF video expansion (space-joined per frame):
  //   frame 0: "MM:SS <boi>" + soft*N + "<eoi>"
  //   frame>0: " MM:SS <boi>" + soft*N + "<eoi>"
  for LFrame := 0 to High(LFrames) do
  begin
    LMins := Trunc(LFrames[LFrame].TimestampSec) div 60;
    LSecs := Trunc(LFrames[LFrame].TimestampSec) mod 60;
    if LFrame > 0 then
      LFragText := Format(' %2.2d:%2.2d ', [LMins, LSecs])
    else
      LFragText := Format('%2.2d:%2.2d ', [LMins, LSecs]);
    for LI in FTokenizer.EncodeRaw(LFragText) do
      AIds.Add(LI);

    if not FImagePipeline.ProcessRGB(LFrames[LFrame].Image, LPatches,
      LPositions, LNumSoft, CVidMaxSoftTokens) then
    begin
      GetErrors().Add(esError, CINF_ERR_MEDIA,
        'Video frame %d processing failed: %s', [LFrame, APath]);
      FImagePipeline.PrintErrors();
      Exit;
    end;

    if not FVision.EncodeImage(LPatches, LPositions, LSoftTokens) then
    begin
      GetErrors().Add(esError, CINF_ERR_MEDIA,
        'Video frame %d encoding failed: %s', [LFrame, APath]);
      FVision.PrintErrors();
      Exit;
    end;

    AIds.Add(CTokBoi);
    LBlock := Default(TSoftTokenBlock);
    LBlock.Position := AIds.Count;
    LBlock.RowCount := LNumSoft;
    LBlock.Rows := LSoftTokens;
    for LI := 0 to LNumSoft - 1 do
      AIds.Add(CTokVideoSoft);
    AIds.Add(CTokEoi);
    ABlocks.Add(LBlock);
  end;
  Result := True;
end;

function TInference.DoExpandMultimodal(var ATokenIds: TArray<Integer>;
  out ASoftBlocks: TArray<TSoftTokenBlock>): Boolean;
var
  LImagePaths: TList<string>;
  LImageBudgets: TList<Integer>;
  LAudioPaths: TList<string>;
  LVideoPaths: TList<string>;
  LVideoFrames: TList<Integer>;
  LIds: TList<Integer>;
  LBlocks: TList<TSoftTokenBlock>;
  LMsgIdx: Integer;
  LPartIdx: Integer;
  LImageNext: Integer;
  LAudioNext: Integer;
  LVideoNext: Integer;
  LI: Integer;
  LId: Integer;
begin
  Result := False;
  ASoftBlocks := nil;

  LImagePaths := TList<string>.Create();
  LImageBudgets := TList<Integer>.Create();
  LAudioPaths := TList<string>.Create();
  LVideoPaths := TList<string>.Create();
  LVideoFrames := TList<Integer>.Create();
  LIds := TList<Integer>.Create();
  LBlocks := TList<TSoftTokenBlock>.Create();
  try
    // Collect media parts in message/part order -- the template renders
    // markers in exactly this order, so the k-th marker of a kind pairs
    // with the k-th part of that kind
    for LMsgIdx := 0 to FMessages.Count - 1 do
      for LPartIdx := 0 to High(FMessages[LMsgIdx].Parts) do
      begin
        case FMessages[LMsgIdx].Parts[LPartIdx].Kind of
          pkImage:
          begin
            LImagePaths.Add(FMessages[LMsgIdx].Parts[LPartIdx].SourcePath);
            LImageBudgets.Add(FMessages[LMsgIdx].Parts[LPartIdx].SoftBudget);
          end;
          pkAudio:
            LAudioPaths.Add(FMessages[LMsgIdx].Parts[LPartIdx].SourcePath);
          pkVideo:
          begin
            LVideoPaths.Add(FMessages[LMsgIdx].Parts[LPartIdx].SourcePath);
            LVideoFrames.Add(FMessages[LMsgIdx].Parts[LPartIdx].SoftBudget);
          end;
        end;
      end;

    // Text-only history: nothing to expand
    if (LImagePaths.Count = 0) and (LAudioPaths.Count = 0) and
       (LVideoPaths.Count = 0) then
      Exit(True);

    LImageNext := 0;
    LAudioNext := 0;
    LVideoNext := 0;

    // Rebuild the id list, expanding each modality marker in place
    for LI := 0 to High(ATokenIds) do
    begin
      LId := ATokenIds[LI];
      if LId = CTokImageSoft then
      begin
        if LImageNext >= LImagePaths.Count then
        begin
          GetErrors().Add(esError, CINF_ERR_MEDIA,
            'Prompt has more <|image|> markers than image parts');
          Exit;
        end;
        if not DoExpandImagePart(LImagePaths[LImageNext], LIds, LBlocks,
          CTokImageSoft, LImageBudgets[LImageNext]) then
          Exit;
        Inc(LImageNext);
      end
      else if LId = CTokAudioSoft then
      begin
        if LAudioNext >= LAudioPaths.Count then
        begin
          GetErrors().Add(esError, CINF_ERR_MEDIA,
            'Prompt has more <|audio|> markers than audio parts');
          Exit;
        end;
        if not DoExpandAudioPart(LAudioPaths[LAudioNext], LIds, LBlocks) then
          Exit;
        Inc(LAudioNext);
      end
      else if LId = CTokVideoSoft then
      begin
        if LVideoNext >= LVideoPaths.Count then
        begin
          GetErrors().Add(esError, CINF_ERR_MEDIA,
            'Prompt has more <|video|> markers than video parts');
          Exit;
        end;
        if not DoExpandVideoPart(LVideoPaths[LVideoNext], LIds, LBlocks,
          LVideoFrames[LVideoNext]) then
          Exit;
        Inc(LVideoNext);
      end
      else
        LIds.Add(LId);
    end;

    if (LImageNext <> LImagePaths.Count) or
       (LAudioNext <> LAudioPaths.Count) or
       (LVideoNext <> LVideoPaths.Count) then
    begin
      GetErrors().Add(esError, CINF_ERR_MEDIA,
        'Unconsumed media parts: template rendered fewer markers than parts');
      Exit;
    end;

    ATokenIds := LIds.ToArray();
    ASoftBlocks := LBlocks.ToArray();
    Result := True;
  finally
    LBlocks.Free();
    LIds.Free();
    LVideoFrames.Free();
    LVideoPaths.Free();
    LAudioPaths.Free();
    LImageBudgets.Free();
    LImagePaths.Free();
  end;
end;

function TInference.DoJsonToJinja(const ANode: TJSON;
  const APool: TJinjaValuePool): TJinjaValue;
var
  LPair: TJSONPair;
  LItem: TJSON;
  LNum: Double;
begin
  if (ANode = nil) or ANode.IsNull() then
    Exit(APool.NewNone());

  if ANode.IsObject() then
  begin
    Result := APool.NewDict();
    for LPair in ANode.Pairs() do
      Result.DictSet(LPair.NodeName, DoJsonToJinja(LPair.Value, APool));
    Exit;
  end;

  if ANode.IsArray() then
  begin
    Result := APool.NewArray();
    for LItem in ANode.Items() do
      Result.ArrayAdd(DoJsonToJinja(LItem, APool));
    Exit;
  end;

  if ANode.IsString() then
    Exit(APool.NewString(ANode.AsString()));
  if ANode.IsBoolean() then
    Exit(APool.NewBool(ANode.AsBoolean()));

  if ANode.IsNumber() then
  begin
    LNum := ANode.AsDouble();
    // Integral JSON numbers become Jinja ints so they render without a
    // decimal point, matching Python's json -> int parsing
    if (Frac(LNum) = 0.0) and (Abs(LNum) < 9007199254740992.0) then
      Exit(APool.NewInt(Trunc(LNum)));
    Exit(APool.NewFloat(LNum));
  end;

  Result := APool.NewNone();
end;

procedure TInference.SetTools(const AToolsJson: string);
var
  LJson: TJSON;
begin
  FToolsJson := '';
  if AToolsJson.Trim() = '' then
    Exit;

  // Validate shape now so a bad schema fails loudly at set time, not
  // silently inside a render
  LJson := TJSON.FromString(AToolsJson);
  try
    if (LJson = nil) or (not LJson.IsArray()) then
    begin
      GetErrors().Add(esError, CINF_ERR_TEMPLATE,
        'SetTools: expected a JSON array of tool schemas');
      Exit;
    end;
  finally
    LJson.Free();
  end;

  FToolsJson := AToolsJson;
end;

function TInference.LoadTemplate(const AVpkPath: string): Boolean;
begin
  Result := DoLoadTemplate(AVpkPath);
end;

function TInference.RenderPrompt(): string;
begin
  Result := '';
  if not FTemplateLoaded then
  begin
    GetErrors().Add(esError, CINF_ERR_TEMPLATE, 'No chat template loaded');
    Exit;
  end;
  if FMessages.Count = 0 then
  begin
    GetErrors().Add(esError, CINF_ERR_NO_MESSAGES, 'No messages to render');
    Exit;
  end;
  Result := DoApplyTemplate();
end;

procedure TInference.DoResetThinkingState();
begin
  FThinkHold := '';
  FInThinking := False;
  FThinkTurnDone := False;
  FInToolCall := False;
end;

function TInference.DoRewriteThinking(const AChunk: string;
  out ADisplay: string; out APlain: string): string;
var
  LOut: TStringBuilder;
  LDisp: TStringBuilder;
  LPlain: TStringBuilder;
  LMarker: string;
  LPos: Integer;
  LPosStart: Integer;
  LPosEnd: Integer;
  LK: Integer;
  LMaxK: Integer;
  LHeld: Integer;
  LDone: Boolean;
  LText: string;
begin
  // Pass-through when thinking is disabled: the prompt carried no
  // <|think|> switch, so the model emits no reasoning markers
  if not FEnableThinking then
  begin
    Result := AChunk;
    ADisplay := AChunk;
    APlain := AChunk;
    Exit;
  end;

  // Ported from Cognita.Inference.DoRewriteThinking with a third output:
  // APlain accumulates only visible text (no span content, no markers,
  // no placeholder) -- the source of ResponseText.
  FThinkHold := FThinkHold + AChunk;
  LOut := TStringBuilder.Create();
  LDisp := TStringBuilder.Create();
  LPlain := TStringBuilder.Create();
  try
    LDone := False;
    while not LDone do
    begin
      if FInThinking then
      begin
        // Inside a span: only the end tag matters and it is a legit close
        LMarker := CThinkNativeEnd;
        LPos := FThinkHold.IndexOf(LMarker); // 0-based; -1 = not found
        if LPos >= 0 then
        begin
          LText := Copy(FThinkHold, 1, LPos);
          LOut.Append(LText);
          // Display side: span content stays hidden when ShowThinking off
          if FShowThinking then
            LDisp.Append(LText);
          LOut.Append(FThinkCloseText);
          // Hidden span: the placeholder was shown at span start --
          // nothing more on close
          if FShowThinking then
            LDisp.Append(FThinkCloseText);
          FInThinking := False;
          FThinkHold := Copy(FThinkHold, LPos + Length(LMarker) + 1, MaxInt);
          Continue;
        end;

        // No full end tag: hold the longest suffix that is a prefix of it
        LHeld := 0;
        LMaxK := Length(LMarker) - 1; // full match already excluded above
        if Length(FThinkHold) < LMaxK then
          LMaxK := Length(FThinkHold);
        for LK := LMaxK downto 1 do
        begin
          if Copy(FThinkHold, Length(FThinkHold) - LK + 1, LK) =
             Copy(LMarker, 1, LK) then
          begin
            LHeld := LK;
            Break;
          end;
        end;
        LText := Copy(FThinkHold, 1, Length(FThinkHold) - LHeld);
        LOut.Append(LText);
        if FShowThinking then
          LDisp.Append(LText);
        FThinkHold := Copy(FThinkHold, Length(FThinkHold) - LHeld + 1, LHeld);
        LDone := True;
      end
      else
      begin
        // Outside a span: watch the start tag AND an unmatched end tag.
        // An unmatched end tag is the model's end-of-turn signal (Gemma 4
        // re-emission quirk; Cognita's parser stops there too).
        LPosStart := FThinkHold.IndexOf(CThinkNativeStart);
        LPosEnd := FThinkHold.IndexOf(CThinkNativeEnd);

        if (LPosEnd >= 0) and ((LPosStart < 0) or (LPosEnd < LPosStart)) then
        begin
          // Unmatched end tag: emit the content before it, consume the
          // tag, discard the degenerate remainder, signal turn done
          LText := Copy(FThinkHold, 1, LPosEnd);
          LOut.Append(LText);
          LDisp.Append(LText);
          LPlain.Append(LText);
          FThinkHold := '';
          FThinkTurnDone := True;
          LDone := True;
          Continue;
        end;

        if LPosStart >= 0 then
        begin
          // Span opens: emit preceding content, swap in the open text /
          // placeholder, enter the span, keep scanning
          LText := Copy(FThinkHold, 1, LPosStart);
          LOut.Append(LText);
          LDisp.Append(LText);
          LPlain.Append(LText);
          LOut.Append(FThinkOpenText);
          if FShowThinking then
            LDisp.Append(FThinkOpenText)
          else
            LDisp.Append(FThinkPlaceholder);
          FInThinking := True;
          FThinkHold := Copy(FThinkHold,
            LPosStart + Length(CThinkNativeStart) + 1, MaxInt);
          Continue;
        end;

        // No full tag: hold the longest suffix that is a prefix of
        // EITHER tag (it may complete on the next chunk)
        LHeld := 0;
        LMaxK := Length(CThinkNativeStart) - 1;
        if Length(FThinkHold) < LMaxK then
          LMaxK := Length(FThinkHold);
        for LK := LMaxK downto 1 do
        begin
          if Copy(FThinkHold, Length(FThinkHold) - LK + 1, LK) =
             Copy(CThinkNativeStart, 1, LK) then
          begin
            LHeld := LK;
            Break;
          end;
        end;
        LMaxK := Length(CThinkNativeEnd) - 1;
        if Length(FThinkHold) < LMaxK then
          LMaxK := Length(FThinkHold);
        for LK := LMaxK downto LHeld + 1 do
        begin
          if Copy(FThinkHold, Length(FThinkHold) - LK + 1, LK) =
             Copy(CThinkNativeEnd, 1, LK) then
          begin
            LHeld := LK;
            Break;
          end;
        end;
        LText := Copy(FThinkHold, 1, Length(FThinkHold) - LHeld);
        LOut.Append(LText);
        LDisp.Append(LText);
        LPlain.Append(LText);
        FThinkHold := Copy(FThinkHold, Length(FThinkHold) - LHeld + 1, LHeld);
        LDone := True;
      end;
    end;
    Result := LOut.ToString();
    ADisplay := LDisp.ToString();
    APlain := LPlain.ToString();
  finally
    LPlain.Free();
    LDisp.Free();
    LOut.Free();
  end;
end;

function TInference.Generate(const AMaxTokens: Integer): Boolean;
var
  LPrompt: string;
  LTokenIds: TArray<Integer>;
  LSoftBlocks: TArray<TSoftTokenBlock>;
  LNormOut: TStringBuilder;
  LPlainOut: TStringBuilder;
  LPrefillFrom: Integer;
  LSuffixIds: TArray<Integer>;
  LGeneratedIds: TArray<Integer>;
begin
  Result := False;
  FResponse := '';
  FResponseText := '';
  FStats := Default(TInferenceStats);

  if not FIsLoaded then
  begin
    GetErrors().Add(esError, CINF_ERR_NOT_LOADED,
      'Cannot generate: model not loaded');
    Exit;
  end;

  if not FTemplateLoaded then
  begin
    GetErrors().Add(esError, CINF_ERR_TEMPLATE,
      'Cannot generate: chat template not loaded');
    Exit;
  end;

  if FMessages.Count = 0 then
  begin
    GetErrors().Add(esError, CINF_ERR_NO_MESSAGES,
      'Cannot generate: no messages');
    Exit;
  end;

  DoResetThinkingState();

  // Signal preparation (template render + tokenize)
  DoFireTokenCallback(psPrep, '');

  // Render the chat template over the full message history. The result
  // is the exact trained prompt format including <bos>, turn markers,
  // and (when enabled) the <|think|> switch.
  LPrompt := DoApplyTemplate();
  if LPrompt = '' then
  begin
    GetErrors().Add(esError, CINF_ERR_RENDER,
      'Chat template rendered to an empty prompt');
    Exit;
  end;

  // Tokenize with special-token awareness: the rendered prompt contains
  // special markers (<bos>, <|turn>, <|think|>, ...) which plain Encode
  // would BPE-encode as literal text; EncodeRaw maps them to their IDs.
  LTokenIds := FTokenizer.EncodeRaw(LPrompt);

  // Expand multimodal markers in place: each <|image|>/<|audio|>/<|video|>
  // id becomes its trained begin/soft-run/end layout, and the matching
  // encoder's rows are queued for substitution after embedding lookup
  LSoftBlocks := nil;
  if not DoExpandMultimodal(LTokenIds, LSoftBlocks) then
    Exit;

  LNormOut := TStringBuilder.Create();
  LPlainOut := TStringBuilder.Create();
  try
    // Signal generation start
    DoFireTokenCallback(psStart, '');

    // KV prefix reuse (PLAN-kv-prefix): keep the cache across calls and
    // prefill only the divergent suffix. Requires GPU mode (CPU cache
    // untested for rewind) and a soft-token-free prompt (rolled-back
    // soft rows cannot be replayed from ids alone).
    LPrefillFrom := 0;
    if FModel.UseGPU and (Length(LSoftBlocks) = 0) and
       (Length(FCachedTokenIds) > 0) then
    begin
      while (LPrefillFrom < Length(FCachedTokenIds)) and
            (LPrefillFrom < Length(LTokenIds)) and
            (FCachedTokenIds[LPrefillFrom] = LTokenIds[LPrefillFrom]) do
        Inc(LPrefillFrom);
      // Prompt fully cached: re-decode the last token for fresh logits
      if LPrefillFrom >= Length(LTokenIds) then
        LPrefillFrom := Length(LTokenIds) - 1;
      FModel.TruncateTo(LPrefillFrom);
    end
    else
      FModel.Reset();
    FCachedTokenIds := nil; // set on success below; any early exit re-prefills
    LSuffixIds := Copy(LTokenIds, LPrefillFrom,
      Length(LTokenIds) - LPrefillFrom);

    // TModel.Generate passes empty token text -- the bridge decodes each
    // ID, rewrites the thinking channel, and streams display text.
    // Anonymous-method capture keeps the builders live inside the bridge.
    LGeneratedIds := FModel.Generate(
      LSuffixIds,
      AMaxTokens,
      FSamplingParams,
      function(const ATokenId: Integer;
        const ATokenTextInner: string;
        const AUserData: Pointer): Boolean
      var
        LText: string;
        LNorm: string;
        LDisp: string;
        LPlain: string;
      begin
        // Decode skips special tokens, so the channel markers would
        // vanish before the rewriter sees them -- re-inject their
        // literal text by ID
        if ATokenId = FTokToolCallStart then
        begin
          // Tool-call open: append marker to FResponse only, suppress
          // from display and plain text
          LNormOut.Append('<|tool_call>');
          FInToolCall := True;
          Result := not IsCancelled();
          Exit;
        end
        else if ATokenId = FTokToolCallEnd then
        begin
          // Tool-call close: append marker to FResponse only
          LNormOut.Append('<tool_call|>');
          FInToolCall := False;
          Result := not IsCancelled();
          Exit;
        end
        else if ATokenId = FTokChannelStart then
          LText := CChannelStartText
        else if ATokenId = FTokChannelEnd then
          LText := CThinkNativeEnd
        else if ATokenId = FTokQuote then
          // Tool-call string delimiter: Decode drops it as a special
          // token; re-inject so ParseToolCalls sees the value boundaries
          LText := CToolQuoteMark
        else
          LText := FTokenizer.Decode(TArray<Integer>.Create(ATokenId));

        // Inside a tool-call span: body goes to FResponse only
        if FInToolCall then
        begin
          LNormOut.Append(LText);
          Result := not IsCancelled();
          Exit;
        end;

        LNorm := DoRewriteThinking(LText, LDisp, LPlain);
        if LNorm <> '' then
          LNormOut.Append(LNorm);
        if LPlain <> '' then
          LPlainOut.Append(LPlain);
        if LDisp <> '' then
          DoFireTokenCallback(psInProgress, LDisp);

        Result := not IsCancelled();

        // Unmatched thinking-end tag: the model signalled end of turn --
        // stop generation cleanly
        if FThinkTurnDone then
          Result := False;
      end,
      nil,
      LSoftBlocks
    );

    // Flush any held marker-prefix text left in the rewriter
    if FThinkHold <> '' then
    begin
      LNormOut.Append(FThinkHold);
      if not FInThinking then
      begin
        LPlainOut.Append(FThinkHold);
        DoFireTokenCallback(psInProgress, FThinkHold);
      end
      else if FShowThinking then
        DoFireTokenCallback(psInProgress, FThinkHold);
      FThinkHold := '';
    end;

    // Close an open thinking span (cancel or max tokens hit mid-reasoning)
    if FInThinking then
    begin
      FInThinking := False;
      if FThinkCloseText <> '' then
      begin
        LNormOut.Append(FThinkCloseText);
        // Hidden span: placeholder already covers it -- no close on screen
        if FShowThinking then
          DoFireTokenCallback(psInProgress, FThinkCloseText);
      end;
    end;

    FModel.GetStats(FStats);
    FResponse := LNormOut.ToString();
    FResponseText := LPlainOut.ToString().Trim();

    // Record exactly the ids whose KV rows exist for the next prefix
    // diff: the full prompt plus the FORWARDED generated ids. The final
    // element of TModel.Generate's result is only Forward()ed on the
    // max-tokens exit path (EOS/callback breaks skip its Forward), so
    // it is always dropped -- shortening is safe, lengthening corrupts.
    if Length(LGeneratedIds) > 0 then
      FCachedTokenIds := Concat(LTokenIds,
        Copy(LGeneratedIds, 0, Length(LGeneratedIds) - 1))
    else
      FCachedTokenIds := Copy(LTokenIds, 0, Length(LTokenIds));

    Result := True;

    // Signal end
    DoFireTokenCallback(psEnd, '');
  finally
    LPlainOut.Free();
    LNormOut.Free();
  end;
end;

procedure TInference.ResetConversation();
begin
  if FIsLoaded then
    FModel.Reset();
  FCachedTokenIds := nil;
end;

procedure TInference.SetTemperature(const AValue: Single);
begin
  FSamplingParams.Temperature := AValue;
end;

procedure TInference.SetTopK(const AValue: Integer);
begin
  FSamplingParams.TopK := AValue;
end;

procedure TInference.SetTopP(const AValue: Single);
begin
  FSamplingParams.TopP := AValue;
end;

function TInference.GetModelName(): string;
begin
  Result := CModelName;
end;

function TInference.GetDeviceName(): string;
begin
  if FVulkanDevice.IsInitialized() then
    Result := FVulkanDevice.DeviceName
  else
    Result := 'CPU';
end;

function TInference.GetVocabSize(): Integer;
begin
  if FTokenizer.IsLoaded() then
    Result := FTokenizer.GetVocabSize()
  else
    Result := 0;
end;

end.
