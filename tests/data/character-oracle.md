`character-oracle.json` was generated independently by the public C# character
box estimator at the revision recorded in the fixture. No models or Rust output
are involved. It covers padding correction, midpoint gaps, perspective quads,
vertical crop rotation, classifier 180-degree rotation, multi-codepoint/Unicode
tokens, empty alignment, and invalid geometry/alignment arguments. The reference
assembly hash and .NET runtime are retained in the JSON.

Six additional fixtures call the reference public packed-text/hash helpers on
empty results, empty lines, Unicode and combining tokens, multilingual text,
embedded NUL/newline characters, and repeated empty lines. They retain exact
UTF-8 bytes, text order, trailing NULs, and unsigned FNV-1a hashes.

Regenerate deliberately after reviewing the reference identity:

```powershell
dotnet build tools/reference-exporter/ReferenceExporter.csproj -c Release -p:ReferenceRoot=D:\simd-paddle-ocr-workspace\SimdPaddleOCR
dotnet tools/reference-exporter/bin/Release/net10.0/Sdcb.SimdPaddleOCR.Tests.dll --character-oracle --output tests/data
```

The source reference is Apache-2.0 licensed; see the repository license and
`NOTICE`. Do not replace oracle outputs with Rust estimates.
