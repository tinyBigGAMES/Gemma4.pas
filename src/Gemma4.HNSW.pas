{===============================================================================
  Gemma4.pas™ - Local LLM inference in Pascal

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information

 -------------------------------------------------------------------------------

  Gemma4.HNSW - Hierarchical Navigable Small World vector index

  Approximate nearest-neighbour search over embedding vectors, backing the
  semantic-recall path of TMemory:
    - Multi-layer navigable graph with configurable M / efConstruction / efSearch
    - Cosine-distance search returning ranked (id, score) hits
    - Incremental add plus binary save/load (HNSW_MAGIC / HNSW_VERSION)
    - TBaseObject base: parameterless Create() then Init(AConfig)

  Dependencies: System.SysUtils, System.Math, System.Classes,
    System.Generics.Collections, System.Generics.Defaults,
    StdApp.Base, StdApp.VirtualMemory, StdApp.Utils
===============================================================================}

unit Gemma4.HNSW;

interface

uses
  System.SysUtils,
  System.Math,
  System.Classes,
  System.Generics.Collections,
  System.Generics.Defaults,
  StdApp.Base,
  StdApp.VirtualMemory,
  StdApp.Utils;

const

  HNSW_MAGIC: UInt32 = $484E5357;
  HNSW_VERSION: UInt32 = 1;
  HNSW_INITIAL_VECTOR_CAPACITY = 4096;

  { Error Codes }
  ERR_HNSW_MAGIC   = 'HNS001';
  ERR_HNSW_VERSION = 'HNS002';
  ERR_HNSW_DIM     = 'HNS003';

type
  { THNSWConfig }
  THNSWConfig = record
    Dim: Integer;             // vector dimensionality (e.g. 384, 768)
    M: Integer;               // max connections per node per layer (default 16)
    EfConstruction: Integer;  // beam width during insert (default 200)
    EfSearch: Integer;        // beam width during search (default 50)
    ML: Double;               // level multiplier = 1/ln(M), computed automatically
  end;

  { THNSWNeighbor }
  THNSWNeighbor = record
    NodeId: Int64;
    Distance: Single;         // 1 - cosine (0 = identical, 2 = opposite)
  end;

  { THNSWNodeInfo }
  THNSWNodeInfo = record
    VectorSlot: UInt64;       // index into the vector store
    MaxLayer: Integer;
    Deleted: Boolean;
  end;

  { THNSWSearchResult }
  THNSWSearchResult = record
    NodeId: Int64;
    Distance: Single;         // 1 - cosine
  end;

  { THNSWHeapKind }
  THNSWHeapKind = (
    hkMin,  // pop returns smallest distance (candidates)
    hkMax   // pop returns largest distance (results)
  );

  { THNSWHeap }
  THNSWHeap = record
  private
    FItems: TArray<THNSWNeighbor>;
    FCount: Integer;
    FKind: THNSWHeapKind;
    procedure SiftUp(AIndex: Integer);
    procedure SiftDown(AIndex: Integer);
    function HasHigherPriority(const AA, AB: Single): Boolean; inline;
  public
    procedure Init(const AKind: THNSWHeapKind; const ACapacity: Integer);
    procedure Push(const AItem: THNSWNeighbor);
    function Pop(): THNSWNeighbor;
    function Peek(): THNSWNeighbor; inline;
    function IsEmpty(): Boolean; inline;
    property Count: Integer read FCount;
  end;

  { THNSWIndex }
  THNSWIndex = class(TBaseObject)
  private
    FConfig: THNSWConfig;

    // Vector storage -- OS-paged via sparse temp file.
    // Vector N lives at offset N * FConfig.Dim in the flat array.
    FVectors: TVirtualMemory<Single>;
    FNextSlot: UInt64;

    // Node directory -- maps node ID to metadata.
    FDirectory: TVirtualDirectory<Int64, THNSWNodeInfo>;

    // Graph edges -- neighbor lists per node per layer.
    FNeighbors: TDictionary<Int64, TArray<TArray<THNSWNeighbor>>>;

    FEntryPointId: Int64;
    FMaxLevel: Integer;
    FActiveCount: Integer;

    procedure EnsureVectorCapacity(const ASlot: UInt64);
    procedure StoreVector(const ASlot: UInt64;
      const AVector: TArray<Single>);
    function GetVectorBySlot(const ASlot: UInt64): TArray<Single>;
    {$HINTS OFF}
    function GetVector(const ANodeId: Int64): TArray<Single>;
    function CosineDistance(const AVecA, AVecB: TArray<Single>): Single;
    {$HINTS ON}
    function CosineDistanceToSlot(
      const AVec: TArray<Single>; const ASlot: UInt64): Single;
    function RandomLevel(): Integer;
    function GetMaxConnections(const ALayer: Integer): Integer;
    function SearchLayer(const AQuery: TArray<Single>;
      const AEntryId: Int64; const AEf: Integer;
      const ALayer: Integer): TArray<THNSWNeighbor>;
    procedure ConnectNeighbors(const ANodeId: Int64;
      const ANeighbors: TArray<THNSWNeighbor>;
      const ALayer: Integer);
    procedure ShrinkNeighbors(const ANodeId: Int64;
      const ALayer: Integer; const AMaxConn: Integer);
  public
    constructor Create(); override;
    destructor Destroy(); override;

    procedure Init(const AConfig: THNSWConfig);

    procedure Insert(const ANodeId: Int64;
      const AVector: TArray<Single>);
    function Search(const AQuery: TArray<Single>;
      const ATopK: Integer): TArray<THNSWSearchResult>;
    procedure Delete(const ANodeId: Int64);

    function NodeCount(): Integer;
    function TotalNodeCount(): Integer;

    // Serialization -- binary format for SQLite BLOB storage.
    // Saves graph structure, metadata, and vectors.
    function SaveToBytes(): TBytes;
    procedure LoadFromBytes(const AData: TBytes);
    procedure Clear();

    property Config: THNSWConfig read FConfig;
  end;

implementation

{ THNSWHeap }

function THNSWHeap.HasHigherPriority(const AA, AB: Single): Boolean;
begin
  // Min-heap: smaller distance wins. Max-heap: larger distance wins.
  if FKind = hkMin then
    Result := AA < AB
  else
    Result := AA > AB;
end;

procedure THNSWHeap.Init(const AKind: THNSWHeapKind;
  const ACapacity: Integer);
begin
  FKind := AKind;
  FCount := 0;
  SetLength(FItems, ACapacity);
end;

procedure THNSWHeap.SiftUp(AIndex: Integer);
var
  LParent: Integer;
  LTemp: THNSWNeighbor;
begin
  while AIndex > 0 do
  begin
    LParent := (AIndex - 1) div 2;
    if HasHigherPriority(FItems[AIndex].Distance,
      FItems[LParent].Distance) then
    begin
      LTemp := FItems[AIndex];
      FItems[AIndex] := FItems[LParent];
      FItems[LParent] := LTemp;
      AIndex := LParent;
    end
    else
      Break;
  end;
end;

procedure THNSWHeap.SiftDown(AIndex: Integer);
var
  LLeft, LRight, LBest: Integer;
  LTemp: THNSWNeighbor;
begin
  while True do
  begin
    LLeft := AIndex * 2 + 1;
    LRight := AIndex * 2 + 2;
    LBest := AIndex;

    if (LLeft < FCount) and
       HasHigherPriority(FItems[LLeft].Distance,
         FItems[LBest].Distance) then
      LBest := LLeft;

    if (LRight < FCount) and
       HasHigherPriority(FItems[LRight].Distance,
         FItems[LBest].Distance) then
      LBest := LRight;

    if LBest = AIndex then
      Break;

    LTemp := FItems[AIndex];
    FItems[AIndex] := FItems[LBest];
    FItems[LBest] := LTemp;
    AIndex := LBest;
  end;
end;

procedure THNSWHeap.Push(const AItem: THNSWNeighbor);
begin
  if FCount >= Length(FItems) then
    SetLength(FItems, Length(FItems) * 2 + 1);
  FItems[FCount] := AItem;
  SiftUp(FCount);
  Inc(FCount);
end;

function THNSWHeap.Pop(): THNSWNeighbor;
begin
  Result := FItems[0];
  Dec(FCount);
  if FCount > 0 then
  begin
    FItems[0] := FItems[FCount];
    SiftDown(0);
  end;
end;

function THNSWHeap.Peek(): THNSWNeighbor;
begin
  Result := FItems[0];
end;

function THNSWHeap.IsEmpty(): Boolean;
begin
  Result := FCount = 0;
end;

{ THNSWIndex }

constructor THNSWIndex.Create();
begin
  inherited Create();
  FVectors := nil;
  FDirectory := nil;
  FNeighbors := nil;
  FEntryPointId := -1;
  FMaxLevel := -1;
  FActiveCount := 0;
  FNextSlot := 0;
end;

procedure THNSWIndex.Init(const AConfig: THNSWConfig);
var
  LAllocSize: UInt64;
begin
  FConfig := AConfig;

  // Apply defaults for zero values
  if FConfig.M < 2 then
    FConfig.M := 16;
  if FConfig.EfConstruction < 1 then
    FConfig.EfConstruction := 200;
  if FConfig.EfSearch < 1 then
    FConfig.EfSearch := 50;
  FConfig.ML := 1.0 / Ln(FConfig.M);

  // Allocate vector storage (sparse file -- no upfront RAM cost)
  FVectors := TVirtualMemory<Single>.Create();
  LAllocSize := UInt64(HNSW_INITIAL_VECTOR_CAPACITY) *
    UInt64(FConfig.Dim) * UInt64(SizeOf(Single));
  FVectors.Allocate(LAllocSize);
  FNextSlot := 0;

  FDirectory := TVirtualDirectory<Int64, THNSWNodeInfo>.Create();
  FNeighbors := TDictionary<Int64, TArray<TArray<THNSWNeighbor>>>.Create();

  FEntryPointId := -1;
  FMaxLevel := -1;
  FActiveCount := 0;
end;

destructor THNSWIndex.Destroy();
begin
  FNeighbors.Free();
  FDirectory.Free();
  FVectors.Free();
  inherited;
end;

procedure THNSWIndex.EnsureVectorCapacity(const ASlot: UInt64);
var
  LNeeded: UInt64;
begin
  LNeeded := (ASlot + 1) * UInt64(FConfig.Dim);
  if LNeeded > FVectors.Capacity then
    FVectors.Grow(LNeeded * UInt64(SizeOf(Single)) * 2);
end;

procedure THNSWIndex.StoreVector(const ASlot: UInt64;
  const AVector: TArray<Single>);
var
  LI: Integer;
  LBase: UInt64;
  LData: PSingleArray;
begin
  EnsureVectorCapacity(ASlot);
  LBase := ASlot * UInt64(FConfig.Dim);
  LData := PSingleArray(FVectors.Memory);
  for LI := 0 to FConfig.Dim - 1 do
    LData^[LBase + UInt64(LI)] := AVector[LI];
end;

function THNSWIndex.GetVectorBySlot(const ASlot: UInt64): TArray<Single>;
var
  LI: Integer;
  LBase: UInt64;
  LData: PSingleArray;
begin
  SetLength(Result, FConfig.Dim);
  LBase := ASlot * UInt64(FConfig.Dim);
  LData := PSingleArray(FVectors.Memory);
  for LI := 0 to FConfig.Dim - 1 do
    Result[LI] := LData^[LBase + UInt64(LI)];
end;

function THNSWIndex.GetVector(const ANodeId: Int64): TArray<Single>;
var
  LInfo: THNSWNodeInfo;
begin
  if FDirectory.TryGetValue(ANodeId, LInfo) then
    Result := GetVectorBySlot(LInfo.VectorSlot)
  else
    SetLength(Result, 0);
end;

function THNSWIndex.CosineDistance(
  const AVecA, AVecB: TArray<Single>): Single;
var
  LI: Integer;
  LDot: Single;
begin
  LDot := 0.0;
  for LI := 0 to FConfig.Dim - 1 do
    LDot := LDot + AVecA[LI] * AVecB[LI];
  Result := 1.0 - LDot;
end;

function THNSWIndex.CosineDistanceToSlot(
  const AVec: TArray<Single>; const ASlot: UInt64): Single;
var
  LI: Integer;
  LDot: Single;
  LBase: UInt64;
  LData: PSingleArray;
begin
  LDot := 0.0;
  LBase := ASlot * UInt64(FConfig.Dim);
  LData := PSingleArray(FVectors.Memory);
  for LI := 0 to FConfig.Dim - 1 do
    LDot := LDot + AVec[LI] * LData^[LBase + UInt64(LI)];
  Result := 1.0 - LDot;
end;

function THNSWIndex.RandomLevel(): Integer;
var
  LR: Double;
begin
  LR := Random();
  if LR < 1e-10 then
    LR := 1e-10;
  Result := Floor(-Ln(LR) * FConfig.ML);
end;

function THNSWIndex.GetMaxConnections(const ALayer: Integer): Integer;
begin
  // Layer 0 gets 2*M connections, higher layers get M
  if ALayer = 0 then
    Result := FConfig.M * 2
  else
    Result := FConfig.M;
end;

function THNSWIndex.SearchLayer(const AQuery: TArray<Single>;
  const AEntryId: Int64; const AEf: Integer;
  const ALayer: Integer): TArray<THNSWNeighbor>;
var
  LVisited: TDictionary<Int64, Boolean>;
  LCandidates: THNSWHeap;  // min-heap: pop nearest
  LResults: THNSWHeap;     // max-heap: pop/peek furthest
  LC, LN: THNSWNeighbor;
  LNeighborLayers: TArray<TArray<THNSWNeighbor>>;
  LLayerNeighbors: TArray<THNSWNeighbor>;
  LI, LCount: Integer;
  LDist: Single;
  LInfo: THNSWNodeInfo;
begin
  LVisited := TDictionary<Int64, Boolean>.Create();
  try
    LCandidates.Init(hkMin, AEf + 1);
    LResults.Init(hkMax, AEf + 1);

    // Seed with entry point
    FDirectory.TryGetValue(AEntryId, LInfo);
    LDist := CosineDistanceToSlot(AQuery, LInfo.VectorSlot);
    LC.NodeId := AEntryId;
    LC.Distance := LDist;
    LVisited.Add(AEntryId, True);
    LCandidates.Push(LC);
    LResults.Push(LC);

    while not LCandidates.IsEmpty() do
    begin
      // Pop nearest candidate -- O(log N)
      LC := LCandidates.Pop();

      // If nearest candidate is further than furthest result and we
      // have enough results, stop -- no more improvement possible
      if (LC.Distance > LResults.Peek().Distance) and
         (LResults.Count >= AEf) then
        Break;

      // Expand neighbors of current candidate
      if not FNeighbors.TryGetValue(LC.NodeId, LNeighborLayers) then
        Continue;
      if ALayer >= Length(LNeighborLayers) then
        Continue;

      LLayerNeighbors := LNeighborLayers[ALayer];
      for LI := 0 to Length(LLayerNeighbors) - 1 do
      begin
        if LVisited.ContainsKey(LLayerNeighbors[LI].NodeId) then
          Continue;
        LVisited.Add(LLayerNeighbors[LI].NodeId, True);

        // Skip nodes that no longer exist in directory
        if not FDirectory.TryGetValue(LLayerNeighbors[LI].NodeId,
          LInfo) then
          Continue;

        LDist := CosineDistanceToSlot(AQuery, LInfo.VectorSlot);

        if (LDist < LResults.Peek().Distance) or
           (LResults.Count < AEf) then
        begin
          LN.NodeId := LLayerNeighbors[LI].NodeId;
          LN.Distance := LDist;
          LCandidates.Push(LN);
          LResults.Push(LN);

          // Trim results if over capacity -- O(log N)
          if LResults.Count > AEf then
            LResults.Pop();
        end;
      end;
    end;

    // Extract results into sorted array (ascending by distance)
    LCount := LResults.Count;
    SetLength(Result, LCount);
    // Pop from max-heap gives descending order, fill from end
    for LI := LCount - 1 downto 0 do
      Result[LI] := LResults.Pop();
  finally
    LVisited.Free();
  end;
end;

procedure THNSWIndex.ConnectNeighbors(const ANodeId: Int64;
  const ANeighbors: TArray<THNSWNeighbor>;
  const ALayer: Integer);
var
  LLayers: TArray<TArray<THNSWNeighbor>>;
begin
  LLayers := FNeighbors[ANodeId];
  LLayers[ALayer] := Copy(ANeighbors);
  FNeighbors[ANodeId] := LLayers;
end;

procedure THNSWIndex.ShrinkNeighbors(const ANodeId: Int64;
  const ALayer: Integer; const AMaxConn: Integer);
var
  LLayers: TArray<TArray<THNSWNeighbor>>;
  LArr: TArray<THNSWNeighbor>;
  LI, LJ: Integer;
  LTemp: THNSWNeighbor;
begin
  LLayers := FNeighbors[ANodeId];
  if ALayer >= Length(LLayers) then
    Exit;
  LArr := LLayers[ALayer];
  if Length(LArr) <= AMaxConn then
    Exit;

  // Sort by distance ascending (insertion sort -- arrays are small)
  for LI := 1 to Length(LArr) - 1 do
  begin
    LTemp := LArr[LI];
    LJ := LI - 1;
    while (LJ >= 0) and (LArr[LJ].Distance > LTemp.Distance) do
    begin
      LArr[LJ + 1] := LArr[LJ];
      Dec(LJ);
    end;
    LArr[LJ + 1] := LTemp;
  end;

  // Keep only the closest AMaxConn neighbors
  SetLength(LArr, AMaxConn);
  LLayers[ALayer] := LArr;
  FNeighbors[ANodeId] := LLayers;
end;

procedure THNSWIndex.Insert(const ANodeId: Int64;
  const AVector: TArray<Single>);
var
  LLevel, LLayer, LI: Integer;
  LInfo: THNSWNodeInfo;
  LLayers: TArray<TArray<THNSWNeighbor>>;
  LCandidates: TArray<THNSWNeighbor>;
  LSelected: TArray<THNSWNeighbor>;
  LEp: Int64;
  LMaxConn: Integer;
  LNeighborLayers: TArray<TArray<THNSWNeighbor>>;
  LNewNeighbor: THNSWNeighbor;
  LLayerArr: TArray<THNSWNeighbor>;
begin
  LLevel := RandomLevel();

  // Store vector in virtual memory
  LInfo.VectorSlot := FNextSlot;
  LInfo.MaxLayer := LLevel;
  LInfo.Deleted := False;
  StoreVector(FNextSlot, AVector);
  Inc(FNextSlot);

  // Register in directory and neighbor map
  FDirectory.Add(ANodeId, LInfo);
  Inc(FActiveCount);

  SetLength(LLayers, LLevel + 1);
  for LLayer := 0 to LLevel do
    SetLength(LLayers[LLayer], 0);
  FNeighbors.Add(ANodeId, LLayers);

  // First node in the index -- set as entry point
  if FEntryPointId = -1 then
  begin
    FEntryPointId := ANodeId;
    FMaxLevel := LLevel;
    Exit;
  end;

  LEp := FEntryPointId;

  // Phase 1: Greedy descent from top level to insertion level + 1
  for LLayer := FMaxLevel downto LLevel + 1 do
  begin
    LCandidates := SearchLayer(AVector, LEp, 1, LLayer);
    if Length(LCandidates) > 0 then
      LEp := LCandidates[0].NodeId;
  end;

  // Phase 2: Insert at each layer from min(LLevel, FMaxLevel) down to 0
  for LLayer := Min(LLevel, FMaxLevel) downto 0 do
  begin
    LCandidates := SearchLayer(AVector, LEp, FConfig.EfConstruction, LLayer);
    LMaxConn := GetMaxConnections(LLayer);

    // Take at most M best candidates (already sorted by distance)
    if Length(LCandidates) > LMaxConn then
      LSelected := Copy(LCandidates, 0, LMaxConn)
    else
      LSelected := LCandidates;

    // Set these as neighbors of the new node at this layer
    ConnectNeighbors(ANodeId, LSelected, LLayer);

    // Add bidirectional connections
    for LI := 0 to Length(LSelected) - 1 do
    begin
      if not FNeighbors.TryGetValue(LSelected[LI].NodeId,
        LNeighborLayers) then
        Continue;
      if LLayer >= Length(LNeighborLayers) then
        Continue;

      // Append new node as neighbor of the candidate
      LNewNeighbor.NodeId := ANodeId;
      LNewNeighbor.Distance := LSelected[LI].Distance;
      LLayerArr := LNeighborLayers[LLayer];
      SetLength(LLayerArr, Length(LLayerArr) + 1);
      LLayerArr[High(LLayerArr)] := LNewNeighbor;
      LNeighborLayers[LLayer] := LLayerArr;
      FNeighbors[LSelected[LI].NodeId] := LNeighborLayers;

      // Trim if over max connections
      if Length(LLayerArr) > LMaxConn then
        ShrinkNeighbors(LSelected[LI].NodeId, LLayer, LMaxConn);
    end;

    // Use nearest candidate as entry for next layer down
    if Length(LCandidates) > 0 then
      LEp := LCandidates[0].NodeId;
  end;

  // Update entry point if new node has a higher level
  if LLevel > FMaxLevel then
  begin
    FEntryPointId := ANodeId;
    FMaxLevel := LLevel;
  end;
end;

function THNSWIndex.Search(const AQuery: TArray<Single>;
  const ATopK: Integer): TArray<THNSWSearchResult>;
var
  LLayer: Integer;
  LEp: Int64;
  LCandidates: TArray<THNSWNeighbor>;
  LI, LCount: Integer;
  LEf: Integer;
  LInfo: THNSWNodeInfo;
begin
  SetLength(Result, 0);
  if FEntryPointId = -1 then
    Exit;

  LEp := FEntryPointId;
  LEf := Max(FConfig.EfSearch, ATopK);

  // Phase 1: Greedy descent from top layer to layer 1
  for LLayer := FMaxLevel downto 1 do
  begin
    LCandidates := SearchLayer(AQuery, LEp, 1, LLayer);
    if Length(LCandidates) > 0 then
      LEp := LCandidates[0].NodeId;
  end;

  // Phase 2: Full beam search at layer 0
  LCandidates := SearchLayer(AQuery, LEp, LEf, 0);

  // Collect top K non-deleted results (already sorted by distance)
  LCount := 0;
  SetLength(Result, ATopK);
  for LI := 0 to Length(LCandidates) - 1 do
  begin
    if LCount >= ATopK then
      Break;
    if FDirectory.TryGetValue(LCandidates[LI].NodeId, LInfo) then
    begin
      if not LInfo.Deleted then
      begin
        Result[LCount].NodeId := LCandidates[LI].NodeId;
        Result[LCount].Distance := LCandidates[LI].Distance;
        Inc(LCount);
      end;
    end;
  end;
  SetLength(Result, LCount);
end;

procedure THNSWIndex.Delete(const ANodeId: Int64);
var
  LInfo: THNSWNodeInfo;
begin
  if FDirectory.TryGetValue(ANodeId, LInfo) then
  begin
    if not LInfo.Deleted then
    begin
      LInfo.Deleted := True;
      FDirectory[ANodeId] := LInfo;
      Dec(FActiveCount);
    end;
  end;
end;

function THNSWIndex.NodeCount(): Integer;
begin
  Result := FActiveCount;
end;

function THNSWIndex.TotalNodeCount(): Integer;
begin
  Result := FDirectory.Count;
end;

function THNSWIndex.SaveToBytes(): TBytes;
var
  LStream: TMemoryStream;
  LMagic: UInt32;
  LVersion: UInt32;
begin
  LStream := TMemoryStream.Create();
  try
    // Header
    LMagic := HNSW_MAGIC;
    LVersion := HNSW_VERSION;
    LStream.WriteData(LMagic);
    LStream.WriteData(LVersion);
    LStream.WriteData(Int32(FConfig.Dim));
    LStream.WriteData(Int32(FConfig.M));
    LStream.WriteData(Int32(FConfig.EfConstruction));
    LStream.WriteData(Int32(FConfig.EfSearch));
    LStream.WriteData(Int32(FDirectory.Count));
    LStream.WriteData(FEntryPointId);
    LStream.WriteData(Int32(FMaxLevel));

    // Per-node data via ForEach on the directory
    FDirectory.ForEach(
      procedure(const AKey: Int64; const AValue: THNSWNodeInfo)
      var
        LVec: TArray<Single>;
        LNeighborLayers: TArray<TArray<THNSWNeighbor>>;
        LLayer, LI: Integer;
        LCount: Int32;
        LDeleted: Byte;
      begin
        // Node ID
        LStream.WriteData(AKey);

        // Max layer
        LStream.WriteData(Int32(AValue.MaxLayer));

        // Deleted flag
        if AValue.Deleted then
          LDeleted := 1
        else
          LDeleted := 0;
        LStream.WriteData(LDeleted);

        // Vector data (Dim floats)
        LVec := GetVectorBySlot(AValue.VectorSlot);
        for LI := 0 to Length(LVec) - 1 do
          LStream.WriteData(LVec[LI]);

        // Neighbor lists for each layer
        if FNeighbors.TryGetValue(AKey, LNeighborLayers) then
        begin
          for LLayer := 0 to AValue.MaxLayer do
          begin
            if LLayer < Length(LNeighborLayers) then
            begin
              LCount := Length(LNeighborLayers[LLayer]);
              LStream.WriteData(LCount);
              for LI := 0 to LCount - 1 do
              begin
                LStream.WriteData(LNeighborLayers[LLayer][LI].NodeId);
                LStream.WriteData(LNeighborLayers[LLayer][LI].Distance);
              end;
            end
            else
            begin
              LCount := 0;
              LStream.WriteData(LCount);
            end;
          end;
        end
        else
        begin
          for LLayer := 0 to AValue.MaxLayer do
          begin
            LCount := 0;
            LStream.WriteData(LCount);
          end;
        end;
      end
    );

    SetLength(Result, LStream.Size);
    if LStream.Size > 0 then
      Move(LStream.Memory^, Result[0], LStream.Size);
  finally
    LStream.Free();
  end;
end;

procedure THNSWIndex.LoadFromBytes(const AData: TBytes);
var
  LStream: TMemoryStream;
  LMagic, LVersion: UInt32;
  LDim, LM, LEfC, LEfS, LNodeCount, LMaxLvl: Int32;
  LEntryId, LNodeId, LNeighborId: Int64;
  LMaxLayer: Int32;
  LDeleted: Byte;
  LVec: TArray<Single>;
  LInfo: THNSWNodeInfo;
  LLayers: TArray<TArray<THNSWNeighbor>>;
  LNeighborCount: Int32;
  LI, LJ, LLayer: Integer;
  LDist: Single;
begin
  Clear();

  FErrors.RaiseOnError := True;

  LStream := TMemoryStream.Create();
  try
    LStream.Write(AData[0], Length(AData));
    LStream.Position := 0;

    // Read and validate header
    LStream.ReadData(LMagic);
    if LMagic <> HNSW_MAGIC then
    begin
      FErrors.Add(esError, ERR_HNSW_MAGIC, 'Invalid HNSW data: bad magic');
      Exit;
    end;

    LStream.ReadData(LVersion);
    if LVersion <> HNSW_VERSION then
    begin
      FErrors.Add(esError, ERR_HNSW_VERSION, 'Invalid HNSW data: unsupported version');
      Exit;
    end;

    LStream.ReadData(LDim);
    LStream.ReadData(LM);
    LStream.ReadData(LEfC);
    LStream.ReadData(LEfS);
    LStream.ReadData(LNodeCount);
    LStream.ReadData(LEntryId);
    LStream.ReadData(LMaxLvl);

    if LDim <> FConfig.Dim then
    begin
      FErrors.Add(esError, ERR_HNSW_DIM,
        'HNSW dim mismatch: expected %d, got %d', [FConfig.Dim, LDim]);
      Exit;
    end;

    FEntryPointId := LEntryId;
    FMaxLevel := LMaxLvl;

    // Read each node
    SetLength(LVec, FConfig.Dim);
    for LI := 0 to LNodeCount - 1 do
    begin
      LStream.ReadData(LNodeId);
      LStream.ReadData(LMaxLayer);
      LStream.ReadData(LDeleted);

      // Read vector data
      for LJ := 0 to FConfig.Dim - 1 do
        LStream.ReadData(LVec[LJ]);

      // Store vector and create node info
      LInfo.VectorSlot := FNextSlot;
      LInfo.MaxLayer := LMaxLayer;
      LInfo.Deleted := (LDeleted <> 0);
      StoreVector(FNextSlot, LVec);
      Inc(FNextSlot);

      FDirectory.Add(LNodeId, LInfo);
      if not LInfo.Deleted then
        Inc(FActiveCount);

      // Read neighbor lists per layer
      SetLength(LLayers, LMaxLayer + 1);
      for LLayer := 0 to LMaxLayer do
      begin
        LStream.ReadData(LNeighborCount);
        SetLength(LLayers[LLayer], LNeighborCount);
        for LJ := 0 to LNeighborCount - 1 do
        begin
          LStream.ReadData(LNeighborId);
          LStream.ReadData(LDist);
          LLayers[LLayer][LJ].NodeId := LNeighborId;
          LLayers[LLayer][LJ].Distance := LDist;
        end;
      end;
      FNeighbors.Add(LNodeId, LLayers);
    end;
  finally
    LStream.Free();
  end;
end;

procedure THNSWIndex.Clear();
var
  LAllocSize: UInt64;
begin
  FNeighbors.Clear();
  FDirectory.Clear();

  // Re-allocate vector storage
  FVectors.Close();
  LAllocSize := UInt64(HNSW_INITIAL_VECTOR_CAPACITY) *
    UInt64(FConfig.Dim) * UInt64(SizeOf(Single));
  FVectors.Allocate(LAllocSize);

  FNextSlot := 0;
  FEntryPointId := -1;
  FMaxLevel := -1;
  FActiveCount := 0;
end;

end.
