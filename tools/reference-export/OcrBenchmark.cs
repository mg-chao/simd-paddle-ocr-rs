using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace SimdPaddleOCR.ReferenceExport;

internal static class OcrBenchmark
{
    private sealed record Row(string CaseId, string ArtifactId, int[] Shape, string Sha256, string Stage, Func<double> Call);
    private static double Elapsed(long start) => (Stopwatch.GetTimestamp()-start)*1000.0/Stopwatch.Frequency;
    private static double Percentile(double[] values,double q) {double[] v=values.Order().ToArray();double p=(v.Length-1)*q;int a=(int)Math.Floor(p),b=(int)Math.Ceiling(p);return v[a]+(v[b]-v[a])*(p-a);}
    public static int Run(string[] args)
    {
        Dictionary<string,string> options=[];
        for(int i=0;i<args.Length;i+=2) if(i+1==args.Length||args[i] is not("--reference" or "--mode" or "--warmup" or "--samples" or "--case" or "--profile")||!options.TryAdd(args[i],args[i+1]))throw new ArgumentException("Invalid benchmark arguments");
        string root=Path.GetFullPath(options.GetValueOrDefault("--reference")??throw new ArgumentException("Missing --reference")),mode=options.GetValueOrDefault("--mode","rec_graph");
        string profile=options.GetValueOrDefault("--profile","canonical");
        int warmup=int.Parse(options.GetValueOrDefault("--warmup","30"),CultureInfo.InvariantCulture),samples=int.Parse(options.GetValueOrDefault("--samples","100"),CultureInfo.InvariantCulture);
        if(warmup<0||samples<1||mode is not("rec_graph" or "rec_public_owned" or "ocr_owned"))throw new ArgumentException("Invalid mode/count");
        RequireProfile(profile);byte[] manifestBytes=File.ReadAllBytes(Path.Combine(root,"manifest.json"));using JsonDocument manifest=JsonDocument.Parse(manifestBytes);JsonElement document=manifest.RootElement,contract=document.GetProperty("contract");
        string family=contract.GetProperty("model").GetString()!;_=Program.ModelPreset.For(family);
        if(contract.GetProperty("profile").GetString()!="canonical"||contract.GetProperty("capture_scope").GetString()!="ocr")throw new InvalidDataException("A canonical full OCR reference is required for frozen benchmark inputs");
        PaddleOcrOptions effectiveOptions=Program.ReadOptions(contract);
        JsonElement[] assets=document.GetProperty("models").EnumerateArray().ToArray();
        foreach(JsonElement asset in assets)_=Program.ReadBundleFile(root,asset);
        foreach(JsonElement asset in document.GetProperty("auxiliary_files").EnumerateArray())_=Program.ReadBundleFile(root,asset);
        JsonElement[] cases=document.GetProperty("cases").EnumerateArray().ToArray();foreach(JsonElement item in cases)foreach(JsonElement artifact in item.GetProperty("artifacts").EnumerateArray())_=Program.ReadBundleFile(root,artifact);
        cases=cases.Where(c=>!options.TryGetValue("--case",out string? selected)||c.GetProperty("id").GetString()==selected).ToArray();if(cases.Length==0)throw new ArgumentException("Empty selected corpus");
        JsonElement Asset(string id)=>assets.Single(a=>a.GetProperty("id").GetString()==id);
        string ModelPath(string id)=>BundleWriter.DestinationPath(root,Asset(id).GetProperty("path").GetString()!);
        GC.Collect();GC.WaitForPendingFinalizers();GC.Collect();long loadBaseline=GC.GetAllocatedBytesForCurrentThread(),started=Stopwatch.GetTimestamp();
        using FileStream recStream=File.OpenRead(ModelPath("rec"));using Model rec=Model.Load(recStream);byte[] dictionary=Program.ReadBundleFile(root,Asset("dictionary"));
        Model? det=null,cls=null;if(mode=="ocr_owned"){using FileStream ds=File.OpenRead(ModelPath("det"));using FileStream cs=File.OpenRead(ModelPath("cls"));det=Model.Load(ds);cls=Model.Load(cs);}
        double loadMs=Elapsed(started);long loadAllocationBytes=GC.GetAllocatedBytesForCurrentThread()-loadBaseline,prepareBaseline=GC.GetAllocatedBytesForCurrentThread();started=Stopwatch.GetTimestamp();
        using PaddleOcrRecognizer recognizer=new(rec,dictionary,effectiveOptions.Recognizer);
        PaddleOcrAll? ocr=null;
        if(mode=="ocr_owned") {
            ocr=new(det!,cls,rec,dictionary,effectiveOptions);
        }
        List<IDisposable> sessions=[];List<Row> rows=[];
        try {
            foreach(JsonElement item in cases) {
                string caseId=item.GetProperty("id").GetString()!;Dictionary<string,JsonElement> artifacts=item.GetProperty("artifacts").EnumerateArray().ToDictionary(a=>a.GetProperty("id").GetString()!);
                foreach(var (id,a) in artifacts) {
                    if(!(mode=="rec_graph"?id.EndsWith(".rec.input",StringComparison.Ordinal):mode=="rec_public_owned"?id.EndsWith(".rec.pixels",StringComparison.Ordinal):id=="input.pixels"))continue;
                    int[] shape=a.GetProperty("shape").EnumerateArray().Select(n=>n.GetInt32()).ToArray();string hash=a.GetProperty("sha256").GetString()!;byte[] raw=Program.ReadBundleFile(root,a);
                    if(mode=="rec_graph") {float[] input=new float[raw.Length/4];for(int i=0;i<input.Length;i++)input[i]=System.Buffers.Binary.BinaryPrimitives.ReadSingleLittleEndian(raw.AsSpan(i*4,4));InferenceSession session=new(rec,shape,1);sessions.Add(session);rows.Add(new(caseId,id,shape,hash,mode,()=>session.RunInternal(input.AsSpan())[0]));}
                    else if(mode=="rec_public_owned") {int width=shape[1],height=shape[0];rows.Add(new(caseId,id,shape,hash,mode,()=>{var r=recognizer.Recognize(raw,width,height,true);return r.Score+r.Text.Length;}));}
                    else {using JsonDocument meta=JsonDocument.Parse(Program.ReadBundleFile(root,artifacts["input.metadata"]));int width=meta.RootElement.GetProperty("width").GetInt32(),height=meta.RootElement.GetProperty("height").GetInt32(),stride=meta.RootElement.GetProperty("stride").GetInt32();rows.Add(new(caseId,id,shape,hash,mode,()=>{var r=ocr!.Run(raw,width,height,true,stride);return r.Lines.Length+r.Text.Length;}));}
                }
            }
            if(rows.Count==0)throw new InvalidDataException("No benchmark rows");double prepareMs=Elapsed(started);long prepareAllocationBytes=GC.GetAllocatedBytesForCurrentThread()-prepareBaseline;
            double[] first=new double[rows.Count],corpus=new double[samples];double[][] times=rows.Select(_=>new double[samples]).ToArray();long[] allocated=new long[rows.Count];double checksum=0;
            for(int i=0;i<rows.Count;i++) {started=Stopwatch.GetTimestamp();checksum+=rows[i].Call();first[i]=Elapsed(started);for(int j=0;j<warmup;j++)checksum+=rows[i].Call();long baseline=GC.GetAllocatedBytesForCurrentThread();for(int j=0;j<samples;j++){started=Stopwatch.GetTimestamp();checksum+=rows[i].Call();times[i][j]=Elapsed(started);corpus[j]+=times[i][j];}allocated[i]=GC.GetAllocatedBytesForCurrentThread()-baseline;}
            if(!double.IsFinite(checksum))throw new InvalidDataException("Benchmark produced a nonfinite checksum");
            Console.WriteLine(Encoding.UTF8.GetString(BundleWriter.SerializeJson(new{schemaVersion=2,kind="ocr_benchmark",runtime="csharp",modelFamily=family,profile,referenceProfile="canonical",effectiveOptions,mode,modelSha256=Asset("rec").GetProperty("sha256").GetString(),dictionarySha256=Asset("dictionary").GetProperty("sha256").GetString(),manifestSha256=BundleWriter.Hash(manifestBytes),warmup,samples,loadMs,loadAllocationBytes,prepareMs,prepareAllocationBytes,instrumented=false,checksum,
                executableSha256=BundleWriter.Hash(File.ReadAllBytes(typeof(OcrBenchmark).Assembly.Location)),csharpNativeSha256=BundleWriter.Hash(File.ReadAllBytes(typeof(Model).Assembly.Location)),detModelSha256=mode=="ocr_owned"?Asset("det").GetProperty("sha256").GetString():null,clsModelSha256=mode=="ocr_owned"?Asset("cls").GetProperty("sha256").GetString():null,
                measurement="Per-row contiguous blocks; corpus sums matching sample indexes. Frozen graphs copy input through span overload. Public APIs return owned strings/arrays. Managed current-thread allocation volume only; native and retained heap unavailable. Validation excluded. Cold boundaries are engine-specific. Graph sessions are all retained; Rust sequentially creates/drops one, so RSS is not a matched memory scope.",
                inputs=rows.Select((r,i)=>new{r.CaseId,r.ArtifactId,r.Shape,r.Sha256,r.Stage,firstCallMs=first[i],samplesMs=times[i],p50Ms=Percentile(times[i],.5),p95Ms=Percentile(times[i],.95),meanMs=times[i].Average(),warmAllocationCount=(long?)null,warmAllocationBytes=allocated[i],warmRetainedHeapBytes=(long?)null,warmPeakHeapGrowthBytes=(long?)null}),
                corpus=new{samplesMs=corpus,p50Ms=Percentile(corpus,.5),p95Ms=Percentile(corpus,.95),meanMs=corpus.Average()}})));return 0;
        }finally{foreach(var session in sessions)session.Dispose();ocr?.Dispose();det?.Dispose();cls?.Dispose();}
    }

    private static void RequireProfile(string profile)
    {
        if(profile is not ("canonical" or "cpu"))throw new ArgumentException("Benchmark profile must be canonical or cpu");
        if(profile=="canonical")Step12Tools.RequireCanonical();
        else
        {
            foreach((string key,string expected) in new[]{("DOTNET_PROCESSOR_COUNT","1"),("DOTNET_EnableHWIntrinsic","1"),("PPOCR_NHWC","1"),("SIMD_OCR_NOFUSE","0"),("PPOCR_DET_THREADS","1")})
                if(Environment.GetEnvironmentVariable(key)!=expected)throw new InvalidOperationException($"{key} must be {expected} for optimized single-worker CPU measurements");
            if(Environment.ProcessorCount!=1)throw new InvalidOperationException("Optimized CPU benchmark requires one visible processor");
            foreach(string key in new[]{"PPOCR_DUMP_CONV","PPOCR_DUMP_NODES","PPOCR_DUMP_PLAN","PPOCR_DUMP_GRAPH"})
                if(Environment.GetEnvironmentVariable(key)is not null)throw new InvalidOperationException($"{key} must be absent during timing");
        }
        foreach(string key in new[]{"DOTNET_TieredCompilation","DOTNET_TieredPGO"})
            if(Environment.GetEnvironmentVariable(key)!=(profile=="canonical"?"0":"1"))throw new InvalidOperationException($"{key} must be {(profile=="canonical"?"0":"1")} for the selected benchmark profile");
    }
}
