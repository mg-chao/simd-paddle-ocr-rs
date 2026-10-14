using System.Text.Json;

namespace SimdPaddleOCR.ReferenceExport;

/// <summary>Model-free checks for the public gate's exact and tolerant field boundaries.</summary>
internal static class PublicSemanticContractTests
{
    public static int Run()
    {
        int checks = 0;
        void Check(string expected, string actual, bool accepted, string detail)
        {
            using JsonDocument reference = JsonDocument.Parse(expected), candidate = JsonDocument.Parse(actual);
            try { PublicBenchmarkValidation.Compare(reference.RootElement, candidate.RootElement, "contract"); }
            catch (InvalidDataException) when (!accepted) { checks++; return; }
            if (!accepted) throw new InvalidDataException("Public gate accepted " + detail);
            checks++;
        }
        Check("{\"text\":\"AB\",\"emitted_count\":2}", "{\"text\":\"AB\",\"emitted_count\":2}", true, "identical fields");
        Check("{\"text\":\"AB\"}", "{\"text\":\"CD\"}", false, "different text with identical length");
        Check("{\"score\":-2}", "{\"score\":-2.0001}", true, "compact raw-max score is not a probability");
        Check("{\"score\":1}", "{\"score\":1.00005}", true, "declared score tolerance");
        Check("{\"score\":1}", "{\"score\":1.01}", false, "excessive score error");
        Check("{\"score\":1e400}", "{\"score\":1e400}", false, "nonfinite scores");
        Check("{\"box\":{\"points\":[[100,200],[300,400]]}}", "{\"box\":{\"points\":[[100.00005,200],[300,400]]}}", true, "box coordinate tolerance");
        Check("{\"box\":{\"points\":[[100,200],[300,400]]}}", "{\"box\":{\"points\":[[300,400],[100,200]]}}", false, "corner reordering");
        Check("{\"tensor_width\":100000}", "{\"tensor_width\":100001}", false, "dimensions must remain exact even inside relative float tolerance");
        Check("{\"applied_rotation_degrees\":180,\"classification_label\":1}", "{\"applied_rotation_degrees\":0,\"classification_label\":1}", false, "rotation changes");
        Check("{\"ctc_spans\":[{\"text\":\"A\",\"score\":2,\"start_column\":0,\"end_column\":2}]}", "{\"ctc_spans\":[{\"text\":\"A\",\"score\":2,\"start_column\":0,\"end_column\":3}]}", false, "half-open CTC bounds");
        Check("{\"lines\":[{\"text\":\"A\"},{\"text\":\"B\"}]}", "{\"lines\":[{\"text\":\"B\"},{\"text\":\"A\"}]}", false, "line order");
        Check("{\"ctc_spans\":[]}", "{\"ctc_spans\":null}", false, "alignment null versus empty");
        Check("{\"text\":\"A\",\"emitted_count\":1}", "{\"text\":\"A\"}", false, "missing fields");
        Check("{\"text\":\"A\"}", "{\"text\":\"A\",\"text\":\"B\"}", false, "duplicate fields");
        Console.WriteLine($"Public semantic contract regressions passed: {checks} checks.");
        return 0;
    }
}
