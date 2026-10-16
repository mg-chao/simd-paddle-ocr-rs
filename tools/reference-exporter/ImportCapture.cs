using System.Buffers.Binary;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using Sdcb.SimdPaddleOCR.OnnxSharp;

// A diagnostic adapter over the pinned reader and pre-layout Model records.
// It deliberately does not patch or copy the reference importer implementation.
internal static class ImportCapture
{
    private static readonly JsonSerializerOptions Json = new() { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower, WriteIndented = true };
    private static readonly MethodInfo TensorBytes = typeof(Model).GetMethod("TensorBytes", BindingFlags.Static | BindingFlags.NonPublic)
        ?? throw new MissingMethodException("Pinned Model.TensorBytes contract is missing");

    internal static void Write(string path, string output, string name)
    {
        if (Environment.GetEnvironmentVariable("PPOCR_NHWC") != "0")
            throw new InvalidOperationException("Import contracts must be captured before layout planning");
        using FileStream stream = File.OpenRead(path);
        OnnxModelData parsed = OnnxProtoReader.Parse(stream);
        OnnxGraphData graph = parsed.Graph!;
        object RawTensor(OnnxTensorData tensor)
        {
            var (dtype, bytes) = ((DType, byte[]))TensorBytes.Invoke(null, [tensor])!;
            return new { Dtype = dtype.ToString().ToLowerInvariant(), Shape = tensor.Dims,
                DataBytes = bytes.LongLength, DataSha256 = Hash(bytes) };
        }
        object Attribute(OnnxAttributeData attribute) => attribute.Type switch
        {
            1 => new { Kind = "float", Bits = BitConverter.SingleToUInt32Bits(attribute.Float) },
            2 => new { Kind = "int", Value = attribute.Int },
            3 => new { Kind = "bytes", Hex = Convert.ToHexStringLower(attribute.String) },
            4 => new { Kind = "tensor", Value = RawTensor(attribute.Tensor ?? throw new InvalidDataException("Missing tensor attribute")) },
            6 => new { Kind = "floats", Bits = attribute.Floats.Select(BitConverter.SingleToUInt32Bits).ToArray() },
            7 => new { Kind = "ints", Values = attribute.Ints },
            8 => new { Kind = "strings", Hex = attribute.Strings.Select(Convert.ToHexStringLower).ToArray() },
            _ => throw new InvalidDataException($"Unsupported raw attribute type {attribute.Type}")
        };
        object Value(OnnxValueInfoData value) => new { value.Name, Dtype = value.ElementType switch
            { 1 => "f32", 6 => "i32", 7 => "i64", 2 => "u8", _ => "unknown" }, value.Shape };
        object raw = new
        {
            parsed.IrVersion,
            Opsets = parsed.Opsets.OrderBy(x => x.Domain, StringComparer.Ordinal).Select(x => new { Domain = x.Domain == "ai.onnx" ? "" : x.Domain, x.Version }).ToArray(),
            Inputs = graph.Inputs.Select(Value).ToArray(), Outputs = graph.Outputs.Select(Value).ToArray(),
            ValueInfo = graph.ValueInfo.OrderBy(x => x.Name, StringComparer.Ordinal).Select(Value).ToArray(),
            Initializers = graph.Initializers.OrderBy(x => x.Name, StringComparer.Ordinal).Select(x => new { x.Name, Tensor = RawTensor(x) }).ToArray(),
            Nodes = graph.Nodes.Select(node => new { node.Name, node.OpType, node.Domain, node.Inputs, node.Outputs,
                Attributes = node.Attributes.ToDictionary(a => a.Name, Attribute, StringComparer.Ordinal) }).ToArray()
        };
        // Load again through the production importer. Every edge is identified
        // by its graph input, producing node/slot, or constant content, avoiding
        // dependencies on implementation-specific tensor numbering and names.
        using FileStream normalizedStream = File.OpenRead(path);
        using Model model = Model.Load(normalizedStream);
        Dictionary<uint, (int Node, int Slot)> producers = [];
        for (int node = 0; node < model.Nodes.Length; node++)
            for (int slot = 0; slot < model.Nodes[node].Outputs.Length; slot++)
                producers.Add(model.Nodes[node].Outputs[slot], (node, slot));
        object? Edge(uint index)
        {
            if (index == uint.MaxValue) return null;
            TensorRecord tensor = model.Tensors[index];
            object origin;
            if ((tensor.Flags & Model.TensorConstant) != 0)
                origin = new { Kind = "constant", DataBytes = model.GetTensorBytes((int)index).Length, DataSha256 = Hash(model.GetTensorBytes((int)index)) };
            else if (producers.TryGetValue(index, out var producer))
                origin = new { Kind = "node", Index = producer.Node, Slot = producer.Slot };
            else
            {
                int input = model.GraphInputs.ToList().IndexOf(index);
                if (input < 0) throw new InvalidDataException("Unbound normalized edge");
                origin = new { Kind = "input", Index = input };
            }
            return new { Dtype = tensor.DType.ToString().ToLowerInvariant(), Shape = tensor.Dimensions.Take((int)tensor.Rank).ToArray(), Origin = origin };
        }
        object normalized = new
        {
            Inputs = model.GraphInputs.Select(Edge).ToArray(), Outputs = model.GraphOutputs.Select(Edge).ToArray(),
            Nodes = model.Nodes.Select(node => new { node.Name, node.OpType, node.Domain, node.Opset,
                Attributes = Parameters(model, node),
                Inputs = node.Inputs.Select(Edge).ToArray(), Outputs = node.Outputs.Select(Edge).ToArray() }).ToArray()
        };
        string directory = Path.Combine(output, "imports");
        Directory.CreateDirectory(directory);
        File.WriteAllText(Path.Combine(directory, name + ".json"), JsonSerializer.Serialize(new { SchemaVersion = 1, Raw = raw, Normalized = normalized }, Json));
        Console.WriteLine($"import {name}: raw and normalized");
    }

    private static string Hash(ReadOnlySpan<byte> bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));

    private static object Parameters(Model model, NodeRecord node)
    {
        byte[] data = model.GetParameters(node).ToArray();
        Dictionary<string, object> attributes = [];
        int I(int offset) => BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(offset, 4));
        void Integer(string name, int offset) => attributes.Add(name, new { Kind = "int", Value = I(offset) });
        void Float(string name, int offset) => attributes.Add(name, new { Kind = "float", Bits = unchecked((uint)I(offset)) });
        void Integers(string name, int offset, int count) => attributes.Add(name, new { Kind = "ints", Values = Enumerable.Range(0, count).Select(i => I(offset + i * 4)).ToArray() });
        switch (node.OpType)
        {
            case "Conv": case "ConvTranspose":
                Integer("group", 4); Integers("kernel_shape", 8, 2); Integers("strides", 16, 2); Integers("dilations", 24, 2); Integers("pads", 32, 4); break;
            case "AveragePool": case "MaxPool":
                Integers("kernel_shape", 8, 2); Integers("strides", 16, 2); Integers("pads", 24, 4); Integer("ceil_mode", 40); Integer("count_include_pad", 44); break;
            case "BatchNormalization": Float("epsilon", 4); Float("momentum", 8); Integer("training_mode", 12); break;
            case "HardSigmoid": Float("alpha", 4); Float("beta", 8); break;
            case "Squeeze": case "Unsqueeze": case "Transpose":
                Integers(node.OpType == "Transpose" ? "perm" : "axes", 4, BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(2, 2))); break;
            case "Softmax": case "Concat": Integer("axis", 4); break;
            case "ReduceMean":
                Integer("keepdims", 4); Integer("noop_with_empty_axes", 8); Integers("axes", 12, BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(2, 2))); break;
            case "Resize":
                attributes.Add("scales", new { Kind = "floats", Bits = Enumerable.Range(0, 4).Select(i => unchecked((uint)I(4 + i * 4))).ToArray() }); break;
        }
        return attributes;
    }
}
