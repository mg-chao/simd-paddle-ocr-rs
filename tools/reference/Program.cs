// Developer tensor oracle. Inputs and outputs are little-endian logical F32.
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Sdcb.SimdPaddleOCR.OnnxSharp;
using Sdcb.SimdPaddleOCR.Backends.Vulkan;

if (args.Length != 8) throw new ArgumentException("MODEL INPUT.f32 OUTPUT_PREFIX N C H W cpu|gpu");
byte[] source=File.ReadAllBytes(args[0]);
using var model=Model.Load(source);
var compiled=new CompiledModel(model,intraOpThreads:1);
int[] shape=args.Skip(3).Take(4).Select(int.Parse).ToArray();
float[] input=MemoryMarshal.Cast<byte,float>(File.ReadAllBytes(args[1])).ToArray();
if(input.Length!=shape.Aggregate(1,(a,b)=>checked(a*b))) throw new ArgumentException("input size mismatch");
float[] output;
int[] outputShape;
string adapter;
using var cpu=new InferenceSession(compiled);
cpu.Reshape(shape);
outputShape=cpu.OutputShape.Dimensions;
if(args[7]=="gpu") {
    using var device=VkDevice.Create();
    using var gpu=new GpuDetGraph(device,compiled);
    output=gpu.Run(shape,input).ToArray();
    adapter=device.DeviceName;
} else if(args[7]=="cpu") {output=cpu.Run(input);adapter=System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture.ToString();}
else throw new ArgumentException("backend must be cpu or gpu");
File.WriteAllBytes(args[2]+".f32",MemoryMarshal.AsBytes(output.AsSpan()).ToArray());
File.WriteAllText(args[2]+".json",JsonSerializer.Serialize(new {
    model=Path.GetFullPath(args[0]),sha256=Convert.ToHexString(SHA256.HashData(source)).ToLowerInvariant(),
    input_sha256=Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(args[1]))).ToLowerInvariant(),
    shape,output_shape=outputShape,backend="csharp-"+args[7],adapter,
    precision=args[7]=="gpu"?"mixed_f16":"f32",boundary="graph-output",threads=1
}));
