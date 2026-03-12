#r "bin/Debug/net10.0/InspectionSystem.Core.dll"
#r "nuget: OpenCvSharp4, 4.9.0.20240103"
#r "nuget: Microsoft.ML.OnnxRuntime, 1.20.1"
#r "nuget: Microsoft.Extensions.Logging.Abstractions, 10.0.4"

using System;
using System.IO;
using InspectionSystem.Core.Services;
using InspectionSystem.Core.Models;
using Microsoft.Extensions.Logging.Abstractions;

try 
{
    var imageProcessor = new ImageProcessor();
    var engine = new OnnxInferenceEngine(NullLogger<OnnxInferenceEngine>.Instance, imageProcessor);
    
    // Load model
    var modelPath = "/Users/geseuteu/vision-inspection-portfolio/02_inspection/models/best.onnx";
    await engine.LoadModelAsync(modelPath);
    Console.WriteLine($"Model loaded: {engine.IsModelLoaded}");
    
    // Load test image
    var imagePath = "/Users/geseuteu/vision-inspection-portfolio/01_training/data/processed/bottle/images/val/broken_small_010.jpg";
    var imageBytes = File.ReadAllBytes(imagePath);
    Console.WriteLine($"Image loaded: {imageBytes.Length} bytes");
    
    // Run inference
    var options = new InferenceOptions 
    {
        ConfidenceThreshold = 0.005f,
        IouThreshold = 0.45f,
        ImageSize = 640
    };
    
    var result = await engine.RunInferenceAsync(imageBytes, 640, 640, options);
    Console.WriteLine($"Inference complete:");
    Console.WriteLine($"  Time: {result.InferenceTimeMs:F1}ms");
    Console.WriteLine($"  Detections: {result.Detections.Count}");
    Console.WriteLine($"  IsNG: {result.IsNG}");
    
    foreach (var det in result.Detections)
    {
        Console.WriteLine($"  Detection: {det.ClassName} conf={det.Confidence:F4} box=[{det.Box.X1},{det.Box.Y1},{det.Box.X2},{det.Box.Y2}]");
    }
    
    engine.Dispose();
}
catch (Exception ex)
{
    Console.WriteLine($"Error: {ex.Message}");
    Console.WriteLine($"Stack: {ex.StackTrace}");
}