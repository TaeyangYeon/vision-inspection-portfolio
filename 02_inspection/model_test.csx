#r "nuget: Microsoft.ML.OnnxRuntime, 1.20.1"
using Microsoft.ML.OnnxRuntime;

var modelPath = "/Users/geseuteu/vision-inspection-portfolio/02_inspection/models/best.onnx";
Console.WriteLine($"File exists: {System.IO.File.Exists(modelPath)}");
Console.WriteLine($"File size: {new System.IO.FileInfo(modelPath).Length / 1024 / 1024.0:F1} MB");

try
{
    var session = new InferenceSession(modelPath);
    Console.WriteLine("Model loaded successfully");
    Console.WriteLine($"Input: {session.InputMetadata.Keys.First()}");
    Console.WriteLine($"Output shape: {string.Join(", ", session.OutputMetadata.Values.First().Dimensions)}");
    session.Dispose();
}
catch (Exception ex)
{
    Console.WriteLine($"Load failed: {ex.Message}");
}