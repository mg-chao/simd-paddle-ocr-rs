using Sdcb.SimdPaddleOCR;

namespace SimdPaddleOCR.ReferenceExport;

/// <summary>Small model-free regression cases for native exporter decode validation.</summary>
internal static class RecContractTests
{
    public static int Run(Action<PaddleOcrRecognitionResult, PaddleOcrRecognitionResult, int, int, int, string> verify)
    {
        int checkedForms = 0;
        PaddleOcrRecognitionResult Empty(int width = 32) => new("", 0, 0, width, 48)
        {
            TensorWidth = width,
            CtcSpans = []
        };
        PaddleOcrRecognitionResult Token(string text = "a", float score = 0.5f, int width = 32) => new(text, score, 1, width, 48)
        {
            TensorWidth = width,
            CtcSpans = [new(text, score, 0, 48)]
        };
        void Accept(PaddleOcrRecognitionResult dense, PaddleOcrRecognitionResult compact, string name, int width = 32)
        {
            verify(dense, compact, width, width, 48, name);
            checkedForms += 2;
        }
        void Reject(PaddleOcrRecognitionResult result, string name)
        {
            foreach (bool dense in new[] { true, false })
            {
                try
                {
                    verify(dense ? result : Empty(), dense ? Token() : result, 32, 32, 48, name);
                }
                catch (InvalidDataException)
                {
                    checkedForms++;
                    continue;
                }
                throw new InvalidDataException($"REC exporter accepted invalid {(dense ? "dense" : "compact")} form: {name}");
            }
        }

        // Bias-first compact projection can emit a token while a separate
        // zero-first dense MatMul plus Add chooses blank. Both are legitimate.
        Accept(Empty(), Token(), "distinct-native-decodes");
        Accept(Token("", 0.4f), Token("", -2f), "emitted-empty-dictionary-token");
        Accept(Token(), Token(score: 2f), "uncalibrated-public-score");
        Accept(Token(width: 1), Token(width: 1), "fixed-width-one", 1);
        Accept(Token(width: 33), Token(width: 33), "fixed-width-thirty-three", 33);

        Reject(Token() with { CtcSpans = null }, "missing-captured-spans");
        Reject(Token() with { EmittedCount = 2 }, "count-span-length");
        Reject(Token() with { EmittedCount = 49 }, "count-exceeds-time");
        Reject(Token() with { Score = float.NaN }, "nonfinite-mean-score");
        Reject(Token() with { Score = float.PositiveInfinity }, "infinite-mean-score");
        Reject(Token() with { CtcSpans = [new("a", float.NaN, 0, 48)] }, "nonfinite-span-score");
        Reject(Token() with { CtcSpans = [new("a", 0.5f, -1, 48)] }, "negative-start");
        Reject(Token() with { CtcSpans = [new("a", 0.5f, 0, 0)] }, "empty-span");
        Reject(Token() with { CtcSpans = [new("a", 0.5f, 0, 49)] }, "end-outside-time");
        Reject(Token("aa") with { EmittedCount = 2, CtcSpans = [new("a", 0.5f, 0, 25), new("a", 0.5f, 24, 48)] }, "overlapping-spans");
        Reject(Token() with { CtcSpans = [new("b", 0.5f, 0, 48)] }, "token-reconstruction");
        Reject(Token() with { Text = null! }, "null-result-text");
        Reject(Token() with { CtcSpans = [new(null!, 0.5f, 0, 48)] }, "null-token-text");
        Reject(Empty() with { Text = "a" }, "zero-emissions-with-text");
        Reject(Empty() with { Score = 1 }, "zero-emissions-with-score");
        Reject(Token() with { ResizedWidth = 31 }, "independent-content-width");
        Reject(Token() with { TensorWidth = 33 }, "independent-tensor-width");
        Reject(Token() with { TimeSteps = 47 }, "independent-time-steps");
        Console.WriteLine($"REC exporter decode contract regressions passed: {checkedForms} forms.");
        return 0;
    }
}
