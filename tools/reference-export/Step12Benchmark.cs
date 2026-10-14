using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.OnnxSharp;

namespace SimdPaddleOCR.ReferenceExport;

internal static class Step12Benchmark
{
    private sealed record Row(string CaseId, string ArtifactId, int[] Shape, string Sha256, string Stage, Func<double> Call);
    private static double Elapsed(long start) => (Stopwatch.GetTimestamp() - start) * 1000.0 / Stopwatch.Frequency;
    private static double Percentile(double[] samples, double q) {
        double[] sorted = samples.Order().ToArray(); double position = (sorted.Length - 1) * q;
        int left = (int)Math.Floor(position), right = (int)Math.Ceiling(position);
        return sorted[left] + (sorted[right] - sorted[left]) * (position - left);
    }
    public static int Run(string[] args)
    {
        Dictionary<string, string> options = [];
        for (int i = 0; i < args.Length; i += 2) {
            if (i + 1 == args.Length || args[i] is not ("--reference" or "--mode" or "--warmup" or "--samples" or "--case") || !options.TryAdd(args[i], args[i + 1]))
                throw new ArgumentException("Invalid, missing or duplicate benchmark argument.");
        }
        string root = Path.GetFullPath(options.GetValueOrDefault("--reference") ?? throw new ArgumentException("Missing --reference"));
        string mode = options.GetValueOrDefault("--mode", "det_graph");
        int warmup = int.Parse(options.GetValueOrDefault("--warmup", "30"), CultureInfo.InvariantCulture), samples = int.Parse(options.GetValueOrDefault("--samples", "100"), CultureInfo.InvariantCulture);
        if (warmup < 0 || samples < 1 || mode is not ("det_graph" or "mixed_shapes" or "pipeline" or "stages")) throw new ArgumentException("Invalid benchmark mode/count.");
        Step12Tools.RequireCanonical();
        byte[] manifestBytes = File.ReadAllBytes(Path.Combine(root, "manifest.json"));
        using JsonDocument manifest = JsonDocument.Parse(manifestBytes);
        JsonElement document = manifest.RootElement, contract = document.GetProperty("contract");
        if (document.GetProperty("schema_version").GetInt32() != 1 || contract.GetProperty("profile").GetString() != "canonical" || contract.GetProperty("backend").GetString() != "cpu") throw new InvalidDataException("Canonical reference required.");
        JsonElement[] models = document.GetProperty("models").EnumerateArray().ToArray();
        foreach (JsonElement model in models) _ = Program.ReadBundleFile(root, model);
        foreach (JsonElement file in document.GetProperty("auxiliary_files").EnumerateArray()) _ = Program.ReadBundleFile(root, file);
        JsonElement[] allCases = document.GetProperty("cases").EnumerateArray().ToArray();
        foreach (JsonElement item in allCases) foreach (JsonElement artifact in item.GetProperty("artifacts").EnumerateArray()) _ = Program.ReadBundleFile(root, artifact);
        JsonElement[] cases = allCases.Where(item => !options.TryGetValue("--case", out string? selected) || item.GetProperty("id").GetString() == selected).ToArray();
        if (cases.Length == 0) throw new ArgumentException("Selected corpus is empty.");
        JsonElement detAsset = models.Single(item => item.GetProperty("id").GetString() == "det"), clsAsset = models.Single(item => item.GetProperty("id").GetString() == "cls");
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        long loadAllocationBaseline=GC.GetAllocatedBytesForCurrentThread();
        long started = Stopwatch.GetTimestamp();
        using FileStream detStream = File.OpenRead(BundleWriter.DestinationPath(root, detAsset.GetProperty("path").GetString()!));
        using Model det = Model.Load(detStream);
        using FileStream clsStream = File.OpenRead(BundleWriter.DestinationPath(root, clsAsset.GetProperty("path").GetString()!));
        using Model cls = Model.Load(clsStream);
        double loadMs = Elapsed(started);long loadAllocationBytes=GC.GetAllocatedBytesForCurrentThread()-loadAllocationBaseline;
        long prepareAllocationBaseline=GC.GetAllocatedBytesForCurrentThread();started = Stopwatch.GetTimestamp();
        List<IDisposable> owned = []; List<Row> rows = []; int[][]? shapeSequence=null;
        double[] reshapeSamples=mode=="mixed_shapes" ? new double[samples] : [];
        JsonElement configured=contract.GetProperty("options"), detOptions=configured.GetProperty("detector");
        var detectorOptions = new PaddleOcrDetectorOptions {Backend=OcrBackend.Cpu,MaxPooledSessions=1,
            LimitSideLength=detOptions.GetProperty("limit_side_length").GetInt32(),MaxCandidates=detOptions.GetProperty("max_candidates").GetInt32(),
            BitmapThreshold=detOptions.GetProperty("bitmap_threshold").GetSingle(),BoxThreshold=detOptions.GetProperty("box_threshold").GetSingle(),
            UnclipRatio=detOptions.GetProperty("unclip_ratio").GetSingle(),UseDilation=detOptions.GetProperty("use_dilation").GetBoolean(),MaxImagePixels=detOptions.GetProperty("max_image_pixels").GetInt64()};
        float classifierThreshold=configured.GetProperty("classifier_threshold").GetSingle();
        using PaddleOcrDetector detector = new(det, detectorOptions, 1);
        using PaddleOcrClassifier classifier = new(cls, new() { Backend = OcrBackend.Cpu, MaxPooledSessions = 1 });
        try
        {
            if (mode == "mixed_shapes") {
                int[][] dims = [[32,64],[64,32],[64,128],[128,64],[32,64],[64,128],[32,32],[32,96],[96,32],[64,64],[64,96],[96,64],[96,96],[128,32]];
                shapeSequence=dims;
                InferenceSession session = new(det, [1,3,32,64]); owned.Add(session);
                List<(int[] Shape,float[] Input)> sequence=[];using MemoryStream corpusBytes=new();
                foreach ((int[] hw, int index) in dims.Select((hw, index) => (hw, index))) {
                    int[] shape = [1,3,hw[0],hw[1]];
                    float[] input = Enumerable.Range(0, checked(3 * hw[0] * hw[1])).Select(i => ((i * 17 % 251) - 125) / 125f).ToArray();
                    byte[] encoded = new byte[input.Length * 4]; for (int i=0;i<input.Length;i++) System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(encoded.AsSpan(i*4,4),input[i]);
                    corpusBytes.Write(encoded);sequence.Add((shape,input));
                }
                int mixedCallIndex=0;
                rows.Add(new("mixed-shapes","det.input",[],BundleWriter.Hash(corpusBytes.ToArray()),"mixed_shapes",()=> {
                    double sum=0,reshapeMs=0;
                    foreach(var entry in sequence){long reshapeStart=Stopwatch.GetTimestamp();session.Reshape(entry.Shape);reshapeMs+=Elapsed(reshapeStart);sum+=session.RunInternal(entry.Input.AsSpan())[0];}
                    int measured=mixedCallIndex++-(1+warmup);if(measured>=0 && measured<samples)reshapeSamples[measured]=reshapeMs;
                    return sum;
                }));
            } else foreach (JsonElement item in cases) {
                string caseId = item.GetProperty("id").GetString()!;
                Dictionary<string, JsonElement> artifacts = item.GetProperty("artifacts").EnumerateArray().ToDictionary(a=>a.GetProperty("id").GetString()!,StringComparer.Ordinal);
                JsonElement Artifact(string id) => artifacts.TryGetValue(id,out JsonElement a) ? a : throw new InvalidDataException($"Missing {id}");
                int[] Shape(string id) => Artifact(id).GetProperty("shape").EnumerateArray().Select(d=>d.GetInt32()).ToArray();
                string Hash(string id) => Artifact(id).GetProperty("sha256").GetString()!;
                byte[] Bytes(string id) => Program.ReadBundleFile(root,Artifact(id));
                float[] Floats(string id) {byte[] bytes=Bytes(id);float[] result=new float[bytes.Length/4];for(int i=0;i<result.Length;i++)result[i]=System.Buffers.Binary.BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i*4,4));return result;}
                if (mode == "det_graph") {
                    int[] shape = Shape("det.input"); float[] input = Floats("det.input");
                    InferenceSession session = new(det, shape, 1); owned.Add(session);
                    rows.Add(new(caseId,"det.input",shape,Hash("det.input"),"det_graph",()=>session.RunInternal(input.AsSpan())[0])); continue;
                }
                using JsonDocument meta = JsonDocument.Parse(Bytes("input.metadata")); JsonElement fields=meta.RootElement;
                int width=fields.GetProperty("width").GetInt32(),height=fields.GetProperty("height").GetInt32(),stride=fields.GetProperty("stride").GetInt32();
                byte[] pixels=Bytes("input.pixels");
                if (mode == "pipeline") {
                    byte[] cropBuffer=[];
                    rows.Add(new(caseId,"input.pixels",Shape("input.pixels"),Hash("input.pixels"),"pipeline",()=> {
                        PaddleOcrDetectionResult found=detector.Detect(pixels,width,height,stride);double sum=found.Boxes.Length;
                        foreach(PaddleOcrDetectionBox box in found.Boxes) {
                            var size=PPOCRCrop.GetSize(box);if(cropBuffer.Length<size.ByteCount)cropBuffer=new byte[size.ByteCount];
                            Span<byte> crop=cropBuffer.AsSpan(0,size.ByteCount);PPOCRCrop.ExtractInto(pixels,width,height,stride,box,crop,out int cw,out int ch);
                            PaddleOcrClassificationResult result=classifier.Classify(crop,cw,ch);if(result.Label==1 && result.Score>classifierThreshold)PPOCRCrop.Rotate180(crop,cw,ch);
                            sum+=result.Score+crop[0];
                        }return sum;
                    })); continue;
                }
                // Frozen inputs isolate stage cost and keep discontinuous geometry independent of graph rounding.
                byte[] detPixels=pixels;int dw=width,dh=height,ds=stride;
                if(width+(long)height<64){dw=Math.Max(32,width);dh=Math.Max(32,height);ds=dw*3;detPixels=new byte[ds*dh];for(int y=0;y<height;y++)pixels.AsSpan(y*stride,width*3).CopyTo(detPixels.AsSpan(y*ds));}
                int[] detShape=Shape("det.input");float[] normalized=new float[3*detShape[2]*detShape[3]];ResizeWorkspace resize=new();
                rows.Add(new(caseId,"det.input",detShape,Hash("det.input"),"det_preprocess",()=> {PPOCRPreprocess.Det(detPixels,dw,dh,ds,detShape[3],detShape[2],normalized,resize);return normalized[0];}));
                float[] probability=Floats("det.probabilities");DbPostprocess.Workspace db=new();
                rows.Add(new(caseId,"det.probabilities",Shape("det.probabilities"),Hash("det.probabilities"),"db_postprocess",()=> DbPostprocess.Run(probability,detShape[3],detShape[2],detectorOptions,width,height,1,1,db).Length));
                using JsonDocument boxDoc=JsonDocument.Parse(Bytes("det.boxes"));JsonElement[] boxes=boxDoc.RootElement.GetProperty("boxes").EnumerateArray().ToArray();
                for(int index=0;index<boxes.Length;index++) {
                    string prefix=$"line.{index:D3}";JsonElement[] p=boxes[index].GetProperty("points").EnumerateArray().ToArray();
                    PaddleOcrDetectionBox box=new(p[0][0].GetSingle(),p[0][1].GetSingle(),p[1][0].GetSingle(),p[1][1].GetSingle(),p[2][0].GetSingle(),p[2][1].GetSingle(),p[3][0].GetSingle(),p[3][1].GetSingle(),boxes[index].GetProperty("score").GetSingle());
                    int[] cropShape=Shape(prefix+".crop");byte[] crop=new byte[cropShape.Aggregate(1,(x,y)=>checked(x*y))];
                    rows.Add(new(caseId,prefix+".crop",cropShape,Hash(prefix+".crop"),"crop",()=> {PPOCRCrop.ExtractInto(pixels,width,height,stride,box,crop,out _,out _);return crop[0];}));
                    byte[] frozenCrop=Bytes(prefix+".crop");float[] clsInput=new float[3*80*160];ResizeWorkspace clsResize=new();
                    rows.Add(new(caseId,prefix+".cls.input",[1,3,80,160],Hash(prefix+".cls.input"),"cls_preprocess",()=> {PPOCRPreprocess.Cls(frozenCrop,cropShape[1],cropShape[0],cropShape[1]*3,clsInput,clsResize);return clsInput[0];}));
                    float[] frozenCls=Floats(prefix+".cls.input");InferenceSession clsSession=new(cls,[1,3,80,160]);owned.Add(clsSession);
                    rows.Add(new(caseId,prefix+".cls.probabilities",Shape(prefix+".cls.probabilities"),Hash(prefix+".cls.probabilities"),"cls_graph",()=>clsSession.RunInternal(frozenCls.AsSpan())[0]));
                    rows.Add(new(caseId,prefix+".cls.public",Shape(prefix+".cls.public"),Hash(prefix+".cls.public"),"cls_public",()=> {PaddleOcrClassificationResult result=classifier.Classify(frozenCrop,cropShape[1],cropShape[0]);return result.Label+result.Score;}));
                    byte[] rotated=frozenCrop.ToArray();rows.Add(new(caseId,prefix+".rec.pixels",cropShape,Hash(prefix+".rec.pixels"),"rotate180",()=> {PPOCRCrop.Rotate180(rotated,cropShape[1],cropShape[0]);return rotated[0];}));
                }
            }
            if(rows.Count==0)throw new InvalidDataException("No benchmark rows.");
            double prepareMs=Elapsed(started),checksum=0;long prepareAllocationBytes=GC.GetAllocatedBytesForCurrentThread()-prepareAllocationBaseline;double[] firstCalls=new double[rows.Count];
            double[][] times=rows.Select(_=>new double[samples]).ToArray();double[] sweeps=new double[samples];long[] rowAllocations=new long[rows.Count];long allocated=0;
            for(int i=0;i<rows.Count;i++) {
                long call=Stopwatch.GetTimestamp();checksum+=rows[i].Call();firstCalls[i]=Elapsed(call);
                for(int repeat=0;repeat<warmup;repeat++)checksum+=rows[i].Call();
                long allocatedBefore=GC.GetAllocatedBytesForCurrentThread();
                for(int repeat=0;repeat<samples;repeat++){call=Stopwatch.GetTimestamp();checksum+=rows[i].Call();double ms=Elapsed(call);times[i][repeat]=ms;sweeps[repeat]+=ms;}
                rowAllocations[i]=GC.GetAllocatedBytesForCurrentThread()-allocatedBefore;allocated+=rowAllocations[i];
            }
            double firstSweepMs=firstCalls.Sum();
            Console.WriteLine(Encoding.UTF8.GetString(BundleWriter.SerializeJson(new {
                schemaVersion=1,runtime="csharp",profile="canonical",mode,reference=root,shapeSequence,reshapeSamplesMs=reshapeSamples,
                modelSha256=detAsset.GetProperty("sha256").GetString(),clsModelSha256=clsAsset.GetProperty("sha256").GetString(),manifestSha256=BundleWriter.Hash(manifestBytes),
                loadMs,prepareMs,loadAllocationBytes,prepareAllocationBytes,firstSweepMs,warmup,samples,checksum,activationBytes=(long?)null,retainedActivationBytes=(long?)null,
                warmAllocationCount=(long?)null,warmAllocationBytes=allocated,warmPeakHeapGrowthBytes=(long?)null,
                instrumentation="Current-thread managed allocation volume; native allocations and retained heap are not measured.",
                measurement="Per-row contiguous first/warmup/measured blocks; corpus samples sum corresponding indexes, not wall sweeps. Canonical serial scalar. Graph calls copy external inputs through the span overload. Mixed calls include reshape on one session. Pipeline uses public C# detector/classifier (classifier uses its compatible prefix head), reused crop bytes, and owning detection boxes. Load covers model stream read/import; prepare covers benchmark row inputs/sessions and detector/classifier setup. Validation is outside both phases. No tracing, hashes, file I/O or serialization during warm timing.",
                inputs=rows.Select((row,i)=>new {row.CaseId,row.ArtifactId,row.Shape,row.Sha256,row.Stage,firstCallMs=firstCalls[i],samplesMs=times[i],p50Ms=Percentile(times[i],.5),p95Ms=Percentile(times[i],.95),meanMs=times[i].Average(),warmAllocationCount=(long?)null,warmAllocationBytes=rowAllocations[i]}).ToArray(),
                corpus=new {samplesMs=sweeps,p50Ms=Percentile(sweeps,.5),p95Ms=Percentile(sweeps,.95),meanMs=sweeps.Average()} })));
            return 0;
        }finally{foreach(IDisposable item in owned)item.Dispose();}
    }
}
