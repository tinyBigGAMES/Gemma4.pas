{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information

 -------------------------------------------------------------------------------

  Gemma4.Config - Model configuration loader

  Parses config.json (either from disk or from a VPK archive) and
  extracts the text_config section into a TModelConfig record. Also
  derives the layer type map (sliding vs full attention per layer)
  and the KV cache sharing map (which layers share KV caches).

  Key types:
  - TRopeConfig: RoPE parameters for one attention type
  - TModelConfig: Complete text decoder configuration record
  - TConfigLoader: Parses config.json via TJSON into TModelConfig

  Dependencies: StdApp.Base, StdApp.JSON, Gemma4.Types
===============================================================================}

unit Gemma4.Config;

{$I StdApp.Defines.inc}

interface

uses
  System.SysUtils,
  System.IOUtils,
  System.Math,
  System.Generics.Collections,
  StdApp.Base,
  StdApp.JSON,
  Gemma4.Types;

const
  CCF_ERR_PARSE = 'CF01';
  CCF_ERR_TEXT_CONFIG = 'CF02';

type
  { TRopeConfig }
  // RoPE parameters for a single attention type
  TRopeConfig = record
    Theta: Single;
    RopeKind: string;        // "default" or "proportional"
    PartialRotaryFactor: Single; // 1.0 for sliding, 0.25 for full
  end;

  { TKvCacheMap }
  // Maps each layer index to its unique KV cache index.
  // Layers that share a KV cache get the same cache index.
  TKvCacheMap = array[0..CNumHiddenLayers - 1] of Integer;

  { TModelConfig }
  // Complete text decoder configuration derived from config.json
  TModelConfig = record
    HiddenSize: Integer;
    NumHiddenLayers: Integer;
    NumAttentionHeads: Integer;
    NumKeyValueHeads: Integer;
    HeadDim: Integer;
    GlobalHeadDim: Integer;
    IntermediateSize: Integer;
    VocabSize: Integer;
    HiddenSizePerLayerInput: Integer;
    SlidingWindow: Integer;
    NumKvSharedLayers: Integer;
    RmsNormEps: Single;
    FinalLogitSoftcapping: Single;
    TieWordEmbeddings: Boolean;
    MaxPositionEmbeddings: Integer;
    VocabSizePerLayerInput: Integer;

    // Derived
    LayerTypes: TLayerTypeArray;
    KvCacheMap: TKvCacheMap;
    NumUniqueKvCaches: Integer;
    RopeSliding: TRopeConfig;
    RopeFull: TRopeConfig;

    // EOS token IDs (from top-level config)
    EosTokenIds: TArray<Integer>;
    BosTokenId: Integer;
    PadTokenId: Integer;
  end;

  { TConfigLoader }
  // Loads and parses config.json into a TModelConfig record
  TConfigLoader = class(TBaseObject)
  private
    function DoParseTextConfig(const ATextConfig: TJSON;
      out AConfig: TModelConfig): Boolean;
    procedure DoBuildLayerTypes(const ATextConfig: TJSON;
      var AConfig: TModelConfig);
    procedure DoBuildKvCacheMap(var AConfig: TModelConfig);
    procedure DoParseRope(const ATextConfig: TJSON;
      var AConfig: TModelConfig);
    procedure DoParseEosTokens(const ARoot: TJSON;
      var AConfig: TModelConfig);
  public
    constructor Create(); override;
    destructor Destroy(); override;

    // Load from a JSON string (e.g. read from VPK)
    function LoadFromString(const AJsonStr: string;
      out AConfig: TModelConfig): Boolean;

    // Load from a file path
    function LoadFromFile(const AFilename: string;
      out AConfig: TModelConfig): Boolean;
  end;

implementation

{ TConfigLoader }

constructor TConfigLoader.Create();
begin
  inherited Create();
end;

destructor TConfigLoader.Destroy();
begin
  inherited;
end;

function TConfigLoader.LoadFromString(const AJsonStr: string;
  out AConfig: TModelConfig): Boolean;
var
  LJSON: TJSON;
  LTextConfig: TJSON;
begin
  Result := False;
  AConfig := Default(TModelConfig);

  LJSON := TJSON.FromString(AJsonStr);
  try
    if LJSON.IsNull() then
    begin
      GetErrors().Add(esError, CCF_ERR_PARSE,
        'Failed to parse config JSON');
      Exit;
    end;

    LTextConfig := LJSON.Get('text_config');
    if LTextConfig.IsNull() then
    begin
      GetErrors().Add(esError, CCF_ERR_TEXT_CONFIG,
        'Missing text_config section in config.json');
      Exit;
    end;

    if not DoParseTextConfig(LTextConfig, AConfig) then
      Exit;

    DoBuildLayerTypes(LTextConfig, AConfig);
    DoBuildKvCacheMap(AConfig);
    DoParseRope(LTextConfig, AConfig);
    DoParseEosTokens(LJSON, AConfig);

    Result := True;
  finally
    LJSON.Free();
  end;
end;

function TConfigLoader.LoadFromFile(const AFilename: string;
  out AConfig: TModelConfig): Boolean;
var
  LText: string;
begin
  Result := False;
  AConfig := Default(TModelConfig);

  try
    LText := TFile.ReadAllText(AFilename, TEncoding.UTF8);
  except
    on E: Exception do
    begin
      GetErrors().Add(esError, CCF_ERR_PARSE,
        'Failed to read config file: %s', [E.Message]);
      Exit;
    end;
  end;

  Result := LoadFromString(LText, AConfig);
end;

function TConfigLoader.DoParseTextConfig(const ATextConfig: TJSON;
  out AConfig: TModelConfig): Boolean;
begin
  AConfig.HiddenSize := ATextConfig.Get('hidden_size').AsInt32(CHiddenSize);
  AConfig.NumHiddenLayers := ATextConfig.Get('num_hidden_layers').AsInt32(CNumHiddenLayers);
  AConfig.NumAttentionHeads := ATextConfig.Get('num_attention_heads').AsInt32(CNumAttentionHeads);
  AConfig.NumKeyValueHeads := ATextConfig.Get('num_key_value_heads').AsInt32(CNumKeyValueHeads);
  AConfig.HeadDim := ATextConfig.Get('head_dim').AsInt32(CHeadDim);
  AConfig.GlobalHeadDim := ATextConfig.Get('global_head_dim').AsInt32(CGlobalHeadDim);
  AConfig.IntermediateSize := ATextConfig.Get('intermediate_size').AsInt32(CIntermediateSize);
  AConfig.VocabSize := ATextConfig.Get('vocab_size').AsInt32(CVocabSize);
  AConfig.HiddenSizePerLayerInput := ATextConfig.Get('hidden_size_per_layer_input').AsInt32(CHiddenSizePerLayerInput);
  AConfig.SlidingWindow := ATextConfig.Get('sliding_window').AsInt32(CSlidingWindow);
  AConfig.NumKvSharedLayers := ATextConfig.Get('num_kv_shared_layers').AsInt32(CNumKvSharedLayers);
  AConfig.RmsNormEps := ATextConfig.Get('rms_norm_eps').AsSingle(CRmsNormEps);
  AConfig.FinalLogitSoftcapping := ATextConfig.Get('final_logit_softcapping').AsSingle(CLogitSoftcap);
  AConfig.TieWordEmbeddings := ATextConfig.Get('tie_word_embeddings').AsBoolean(True);
  AConfig.MaxPositionEmbeddings := ATextConfig.Get('max_position_embeddings').AsInt32(131072);
  AConfig.VocabSizePerLayerInput := ATextConfig.Get('vocab_size_per_layer_input').AsInt32(CVocabSize);
  AConfig.BosTokenId := ATextConfig.Get('bos_token_id').AsInt32(CBosTokenId);
  AConfig.PadTokenId := ATextConfig.Get('pad_token_id').AsInt32(CPadTokenId);

  Result := True;
end;

procedure TConfigLoader.DoBuildLayerTypes(const ATextConfig: TJSON;
  var AConfig: TModelConfig);
var
  LLayerTypesArr: TJSON;
  LItems: TArray<TJSON>;
  LI: Integer;
begin
  LLayerTypesArr := ATextConfig.Get('layer_types');
  if (not LLayerTypesArr.IsNull()) and (LLayerTypesArr.Count() > 0) then
  begin
    // Parse from config.json layer_types array
    LItems := LLayerTypesArr.Items();
    for LI := 0 to Min(High(LItems), CNumHiddenLayers - 1) do
      AConfig.LayerTypes[LI] := AttentionKindFromString(LItems[LI].AsString(''));
  end
  else
  begin
    // Fall back to computed pattern
    AConfig.LayerTypes := BuildLayerTypeMap();
  end;
end;

procedure TConfigLoader.DoBuildKvCacheMap(var AConfig: TModelConfig);
var
  LI: Integer;
  LJ: Integer;
  LNumShared: Integer;
  LCacheIdx: Integer;
  LTotalLayers: Integer;
  LNonSharedStart: Integer;
  LLastSlidingIdx: Integer;
  LLastFullIdx: Integer;
begin
  // Gemma 4 E4B: the LAST num_kv_shared_layers layers (24..41) have no K/V
  // projection weights. Per HF Gemma4TextAttention, each shared layer reuses
  // the K/V states of the LAST non-shared layer OF ITS OWN LAYER TYPE:
  //   sliding-type shared layers -> layer 22 (last non-shared sliding layer)
  //   full-type shared layers    -> layer 23 (last non-shared full layer)
  //
  // Non-shared layers (0..23) each get their own unique KV cache index
  // (cache index == layer index for these).

  LTotalLayers := AConfig.NumHiddenLayers;
  LNumShared := AConfig.NumKvSharedLayers;
  LNonSharedStart := LTotalLayers - LNumShared;
  AConfig.NumUniqueKvCaches := LNonSharedStart;

  // First: assign unique cache indices to non-shared layers
  LCacheIdx := 0;
  for LI := 0 to LNonSharedStart - 1 do
  begin
    AConfig.KvCacheMap[LI] := LCacheIdx;
    Inc(LCacheIdx);
  end;

  // Find the last non-shared layer of each attention type. These are the
  // KV publishers that all shared layers of the matching type consume.
  LLastSlidingIdx := -1;
  LLastFullIdx := -1;
  for LJ := 0 to LNonSharedStart - 1 do
  begin
    if AConfig.LayerTypes[LJ] = akFull then
      LLastFullIdx := LJ
    else
      LLastSlidingIdx := LJ;
  end;

  // Shared layers map to the last non-shared layer of their own type
  for LI := LNonSharedStart to LTotalLayers - 1 do
  begin
    if AConfig.LayerTypes[LI] = akFull then
      AConfig.KvCacheMap[LI] := AConfig.KvCacheMap[LLastFullIdx]
    else
      AConfig.KvCacheMap[LI] := AConfig.KvCacheMap[LLastSlidingIdx];
  end;
end;

procedure TConfigLoader.DoParseRope(const ATextConfig: TJSON;
  var AConfig: TModelConfig);
var
  LRopeParams: TJSON;
  LSlidingRope: TJSON;
  LFullRope: TJSON;
begin
  // Defaults
  AConfig.RopeSliding.Theta := CRopeThetaSliding;
  AConfig.RopeSliding.RopeKind := 'default';
  AConfig.RopeSliding.PartialRotaryFactor := 1.0;

  AConfig.RopeFull.Theta := CRopeThetaFull;
  AConfig.RopeFull.RopeKind := 'proportional';
  AConfig.RopeFull.PartialRotaryFactor := CRopePartialRotaryFactor;

  LRopeParams := ATextConfig.Get('rope_parameters');
  if LRopeParams.IsNull() then
    Exit;

  // Sliding attention RoPE
  LSlidingRope := LRopeParams.Get('sliding_attention');
  if not LSlidingRope.IsNull() then
  begin
    AConfig.RopeSliding.Theta := LSlidingRope.Get('rope_theta').AsSingle(CRopeThetaSliding);
    AConfig.RopeSliding.RopeKind := LSlidingRope.Get('rope_type').AsString('default');
  end;

  // Full attention RoPE
  LFullRope := LRopeParams.Get('full_attention');
  if not LFullRope.IsNull() then
  begin
    AConfig.RopeFull.Theta := LFullRope.Get('rope_theta').AsSingle(CRopeThetaFull);
    AConfig.RopeFull.RopeKind := LFullRope.Get('rope_type').AsString('proportional');
    AConfig.RopeFull.PartialRotaryFactor := LFullRope.Get('partial_rotary_factor').AsSingle(CRopePartialRotaryFactor);
  end;
end;

procedure TConfigLoader.DoParseEosTokens(const ARoot: TJSON;
  var AConfig: TModelConfig);
var
  LEos: TJSON;
  LItems: TArray<TJSON>;
  LI: Integer;
begin
  LEos := ARoot.Get('eos_token_id');
  if LEos.IsNull() then
  begin
    // Default EOS tokens
    AConfig.EosTokenIds := TArray<Integer>.Create(
      CEosTokenId1, CEosTokenId2, CEosTokenId3);
    Exit;
  end;

  if LEos.IsArray() then
  begin
    LItems := LEos.Items();
    SetLength(AConfig.EosTokenIds, Length(LItems));
    for LI := 0 to High(LItems) do
      AConfig.EosTokenIds[LI] := LItems[LI].AsInt32(0);
  end
  else
  begin
    // Single integer
    SetLength(AConfig.EosTokenIds, 1);
    AConfig.EosTokenIds[0] := LEos.AsInt32(CEosTokenId1);
  end;
end;

end.
