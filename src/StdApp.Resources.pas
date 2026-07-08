{===============================================================================
  StdApp Components™

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  See LICENSE for license information

 -------------------------------------------------------------------------------

  StdApp.Resources - Shared resource strings

  Central repository of all user-facing message strings used across
  StdApp units. All error messages, warning text, and format strings
  are declared as resourcestring constants for localization readiness
  and clean separation from logic.

  Categories: severity names, error formats, fatal/IO messages,
  VFS messages, VirtualMemory messages.

  Dependencies: none
  Notes: Error code constants are defined in the unit of their concern,
    not here. This unit holds only the message text.
===============================================================================}

unit StdApp.Resources;

{$I StdApp.Defines.inc}

interface

resourcestring

  //--------------------------------------------------------------------------
  // Severity Names
  //--------------------------------------------------------------------------
  RSSeverityHint    = 'Hint';
  RSSeverityWarning = 'Warning';
  RSSeverityError   = 'Error';
  RSSeverityFatal   = 'Fatal';
  RSSeverityNote    = 'Note';
  RSSeverityUnknown = 'Unknown';

  //--------------------------------------------------------------------------
  // Error Format Strings
  //--------------------------------------------------------------------------
  RSErrorFormatSimple              = '%s %s: %s';
  RSErrorFormatWithLocation        = '%s: %s %s: %s';
  RSErrorFormatRelatedSimple       = '  %s: %s';
  RSErrorFormatRelatedWithLocation = '  %s: %s: %s';

  //--------------------------------------------------------------------------
  // Fatal / I/O Messages
  //--------------------------------------------------------------------------
  RSFatalFileNotFound  = 'File not found: ''%s''';
  RSFatalFileReadError = 'Cannot read file ''%s'': %s';
  RSFatalInternalError = 'Internal error: %s';

  //--------------------------------------------------------------------------
  // VFS Messages
  //--------------------------------------------------------------------------
  RSVFSOpenFileFailed      = 'Failed to open file: ''%s''';
  RSVFSInvalidMagic        = 'Invalid VFS archive magic signature';
  RSVFSInvalidVersion      = 'Unsupported VFS archive version: %d';
  RSVFSTruncated           = 'VFS archive is truncated or corrupt';
  RSVFSNotOpen             = 'VFS archive is not open';
  RSVFSEntryNotFound       = 'Entry not found in VFS: ''%s''';
  RSVFSScanDirFailed       = 'Failed to scan directory: ''%s''';
  RSVFSEmptyDirectory      = 'Source directory contains no files: ''%s''';
  RSVFSSourceOpenFailed    = 'Failed to open source file for packing: ''%s''';
  RSVFSException           = 'Unexpected exception in VFS: %s';

  //--------------------------------------------------------------------------
  // VirtualMemory Messages
  //--------------------------------------------------------------------------
  RSVMAllocSizeZero          = 'Cannot allocate a zero-size buffer';
  RSVMCreateMappingFailed    = 'CreateFileMapping failed (error %d)';
  RSVMMappingNameExists      = 'Mapping name "%s" already exists';
  RSVMMapViewFailed          = 'MapViewOfFile failed (error %d)';
  RSVMAllocException         = 'Allocate exception: %s';
  RSVMSharedNameEmpty        = 'OpenShared: mapping name must not be empty';
  RSVMOpenMappingFailed      = 'OpenFileMapping failed for "%s" (error %d)';
  RSVMMapViewNamedFailed     = 'MapViewOfFile failed for "%s" (error %d)';
  RSVMSharedException        = 'OpenShared exception for "%s": %s';
  RSVMUseAllocate            = 'Use Allocate() for anonymous buffers, not Open()';
  RSVMOpenFileFailed         = 'Cannot open file "%s" (error %d)';
  RSVMFileEmpty              = 'File "%s" is empty -- cannot memory-map';
  RSVMCreateMappingNamedFailed = 'CreateFileMapping failed for "%s" (error %d)';
  RSVMOpenException          = 'Open exception for "%s": %s';
  RSVMLoadAlignmentFailed    = 'File size (%d) is not aligned to element size (%d)';
  RSVMLoadException          = 'LoadFromFile exception for "%s": %s';
  RSVMFlushFailed            = 'FlushViewOfFile failed (error %d)';
  RSVMGrowNotAnonymous       = 'Grow is only valid for anonymous (vmAllocate) buffers';
  RSVMGrowNotShared          = 'Grow is not valid for shared consumer mappings';
  RSVMGrowMappingFailed      = 'Grow: CreateFileMapping failed (error %d)';
  RSVMGrowMapViewFailed      = 'Grow: MapViewOfFile failed (error %d)';
  RSVMGrowException          = 'Grow exception: %s';

  //--------------------------------------------------------------------------
  // Crypto Messages
  //--------------------------------------------------------------------------
  RSCryFileNotFound  = 'File not found: %s';
  RSCryFileError     = 'File error: %s';
  RSCryRandomFailed  = 'Secure random generation failed (BCryptGenRandom)';
  RSCryNoSecretKey   = 'No secret key present in key pair';
  RSCryNoPublicKey   = 'No public key present in key pair';
  RSCryBadKeyFile    = 'Invalid or unrecognized key data: %s';
  RSCryBadSigFile    = 'Invalid signature file: %s';
  RSCryKeyIdMismatch = 'Signature key id does not match the public key: %s';
  RSCryVerifyFailed  = 'Signature verification FAILED for: %s';
  RSCrySecKeyComment     = 'apppacker secret key %s';
  RSCrySigDefaultComment = 'signature from apppacker secret key %s';

  //--------------------------------------------------------------------------
  // Packer Messages
  //--------------------------------------------------------------------------
  RSPakManifestNotFound = 'Manifest not found: %s';
  RSPakParseError       = 'Manifest parse error at line %d: %s';
  RSPakNoOutput         = 'Manifest is missing the output: key';
  RSPakSourceMissing    = 'Source directory not found: %s';
  RSPakSourceScan       = 'scan: %s (prefix: %s)';
  RSPakMatched          = 'matched: %d file(s)';
  RSPakNoFiles          = 'Nothing to pack (no files matched include/exclude rules)';
  RSPakAdd              = 'add: %s';
  RSPakDone             = 'done: %d file(s) -> %s';
  RSPakChecksum         = 'checksum: %s';
  RSPakSigned           = 'signed: %s';
  RSPakSecKeyInArchive  = 'FATAL: the configured secret key file is matched by include rules: %s';
  RSPakZipError         = 'Archive build failed: %s';
  RSPakNoSecKey         = 'sign: block present but seckey: is missing';

  //--------------------------------------------------------------------------
  // AppPacker CLI Messages
  //--------------------------------------------------------------------------
  RSCliBanner = 'AppPacker™ %s - manifest-driven release packer';
  RSCliUsage = 'Usage:'#13#10 +
    '  AppPacker <manifest.yml>'#13#10 +
    '  AppPacker pack <manifest.yml>'#13#10 +
    '  AppPacker keygen <basepath>'#13#10 +
    '  AppPacker verify <file> <pubkey-file-or-string>';
  RSCliPacking       = 'packing: %s %s';
  RSCliKeyExists     = 'REFUSED: secret key already exists, will not overwrite: %s';
  RSCliKeygenDone    = 'keypair written: %s / %s';
  RSCliKeyId         = 'key id: %s';
  RSCliPubKey        = 'public key: %s';
  RSCliKeyBackupHint = 'BACK UP the .key file now. It is NOT encrypted. Keep it OUTSIDE any repo.';
  RSCliVerifyOk      = 'verified OK: %s';

  //--------------------------------------------------------------------------
  // Your Application
  //--------------------------------------------------------------------------
  // Add your application-specific resource strings below this line.
  // This section is reserved for custom messages, labels, and format
  // strings that are unique to your application. StdApp framework
  // resources are defined above and should not be modified.
  //--------------------------------------------------------------------------

  RSInfUnsupportedMediaKind = 'Unsupported media kind for AddMessage';

  //--------------------------------------------------------------------------
  // Tool Messages
  //--------------------------------------------------------------------------
  RSToolUnknown         = 'Unknown tool: %s';
  RSToolHandlerError    = 'Tool handler exception for %s: %s';
  RSToolParamsNotObject = 'Tool params JSON is not an object';
  RSToolParamsParse     = 'Failed to parse tool params JSON: %s';
  RSRestRequestFailed   = '%s %s failed: %s';

  //--------------------------------------------------------------------------
  // Meta-Tool Descriptions
  //--------------------------------------------------------------------------
  RSMetaFindTool = 'List all available tools with their names, descriptions, and ' +
    'parameters. Always call this first to discover what tools exist before ' +
    'attempting to use one. Returns the complete tool catalog as JSON.';
  RSMetaUseTool = 'Execute a tool from the catalog by name. First use find_tool to ' +
    'discover available tools. Pass the tool name and a JSON object of its ' +
    'required arguments.';
  RSMetaRunScript = 'Execute Python code using the bundled Python interpreter. Write ' +
    'complete, self-contained scripts. The interpreter has access to the ' +
    'standard library and any pip-installed packages.';

  //--------------------------------------------------------------------------
  // Memory Messages
  //--------------------------------------------------------------------------
  RSMemSessionNotOpen      = 'Memory session not open';
  RSMemDbPathEmpty         = 'Memory database path is empty';
  RSMemEmbedderNil         = 'Embedder is nil';
  RSMemEmbedderNotLoaded   = 'Embedder is not loaded';
  RSMemEmbedderDetached    = 'Attached embedder is no longer loaded -- call DetachEmbeddings before unloading';
  RSMemNoEmbedder          = 'No embedder attached -- call AttachEmbeddings first';
  RSMemEmbeddingMismatch   = 'Embedding byte length mismatch (got %d, expected %d for dim %d)';
  RSMemWhereEmpty          = 'Empty WHERE clause -- use PurgeAll instead';
  RSMemChunkInvalid        = 'AChunkTokens must be > 0';
  RSMemOverlapInvalid      = 'AOverlapTokens must be < AChunkTokens';
  RSMemOpenFailed          = 'Failed to open memory database "%s": %s';
  RSMemEmbedFailed         = 'Embedding generation failed';
  RSMemFTS5Failed          = 'FTS5 search failed -- recall degraded: %s';
  RSMemSnapshotFailed      = 'Failed to snapshot memory database to %s: %s';
  RSMemRecallHeader        = 'Background from earlier conversations (memory recall -- context only, do not answer or act on it):';

  { Session strings }
  RSSesNoInference   = 'No inference engine attached to session';
  RSSesGenerateFailed = 'Generation failed during session turn';
  RSSesOverflow      = 'Context budget exceeded -- oldest turns evicted';
  RSSesSaveFailed    = 'Failed to save history: %s';
  RSSesLoadFailed    = 'Failed to load history: %s';
  RSSesInvalidFormat = 'Invalid history file format';
  RSSesToolRounds    = 'Maximum tool rounds reached -- forcing text reply';
  RSSesCurrentMessage = 'Current message:';
  RSSesSummaryHeader = 'Previous conversation summary:';

  { Chat strings }
  RSChatPrompt             = 'You> ';
  RSChatModelPathEmpty     = 'ModelPath is empty -- set it before Run';
  RSChatBadTokenBudget     = 'MaxTokens (%d) must be smaller than ContextSize (%d) -- the difference is the room left for conversation';
  RSChatModelLoadFailed    = 'Failed to load model: %s';
  RSChatEmbedderFailed     = 'Failed to load embedder: %s';
  RSChatMemoryFailed       = 'Failed to open memory database: %s';
  RSChatNoMemory           = 'Memory not enabled -- set MemoryDbPath before Run';
  RSChatErrorFmt           = '[%s] %s';
  RSChatCleared            = 'Conversation cleared.';
  RSChatForgot             = 'Conversation and memory wiped -- pristine state.';
  RSChatSystemUpdated      = 'System prompt updated.';
  RSChatUsageSystem        = 'Usage: /system <prompt text>';
  RSChatFactAdded          = 'Fact added.';
  RSChatUsageAddFact       = 'Usage: /addfact <fact text>';
  RSChatUsageFile          = 'Usage: %s <path>';
  RSChatFileNotFound       = 'File not found: %s';
  RSChatDocAdded           = 'Document added: %s';
  RSChatFileReadFailed     = 'Failed to read file: %s';
  RSChatHistoryCount       = 'History messages: %d';
  RSChatArchivedCount      = 'Archived turns: %d';
  RSChatMaxTokensSet       = 'Max tokens set to %d';
  RSChatCompacted          = 'Compacted: %d messages.';
  RSChatNothingToCompact   = 'Nothing to compact.';
  RSChatHistorySaved       = 'History saved: %s';
  RSChatHistoryLoaded      = 'History loaded: %s';
  RSChatStateSaved         = 'State saved: %s';
  RSChatStateRestored      = 'State restored: %s';
  RSChatOpFailed           = 'Operation failed.';
  RSChatUnknownCommand     = 'Unknown command: %s';
  RSChatStatsGenerate      = 'Generation: %d tok, %.1f tok/s, %.2f s';
  RSChatStatsPrefill       = 'Prefill: %d tok, %.1f tok/s, %.2f s';
  RSChatStatsContext       = 'Context: %d / %d tokens (%.0f%%)';
  RSChatToolCall           = '[tool] %s(%s)';
  RSChatDbSaved            = 'Memory database saved: %s';
  RSChatDbLoaded           = 'Memory database loaded: %s';
  RSChatDbNoneFound        = 'No .db files found under: %s';
  RSChatDbListHeader       = 'Found %d database(s) under %s:';
  RSChatSummaryEmpty       = 'No conversation summary yet.';
  RSChatCompressing        = 'Compressing conversation history...';
  RSChatBanner             = 'Gemma4.pas Chat';
  RSChatBannerHint         = 'Type /help for commands, /quit to exit.';
  RSChatLoadingModel       = 'Loading model... ';
  RSChatGoodbye            = 'Goodbye!';
  RSChatHelpHeader         = 'Available commands:';
  RSChatHelpQuit           = '  /quit              Exit the chat';
  RSChatHelpClear          = '  /clear             Clear conversation history';
  RSChatHelpForget         = '  /forget            Clear history AND wipe the memory database';
  RSChatHelpSystem         = '  /system <text>     Set system prompt';
  RSChatHelpAddFact        = '  /addfact <text>    Add a fact to memory';
  RSChatHelpAddFile        = '  /addfile <path>    Add a document to memory';
  RSChatHelpStats          = '  /stats             Show inference statistics';
  RSChatHelpTurns          = '  /turns             Show history and archive counts';
  RSChatHelpTokens         = '  /tokens <n>        Set max generation tokens';
  RSChatHelpCompact        = '  /compact           Archive all but recent turns';
  RSChatHelpSave           = '  /save <path>       Save history JSON';
  RSChatHelpLoad           = '  /load <path>       Load history JSON';
  RSChatHelpState          = '  /state <path>      Save full state (KV cache + history)';
  RSChatHelpRestore        = '  /restore <path>    Restore full state';
  RSChatHelpSummary        = '  /summary           Show the conversation summary';
  RSChatHelpDbSave         = '  /dbsave <path>     Snapshot memory DB to path';
  RSChatHelpDbLoad         = '  /dbload <path>     Load a memory DB from path';
  RSChatHelpDbList         = '  /dblist <path>     List memory DB files under path';
  RSChatHelpDbReset        = '  /dbreset           Reload the original memory DB';
  RSChatHelpHelp           = '  /help              Show this help';

implementation

end.
