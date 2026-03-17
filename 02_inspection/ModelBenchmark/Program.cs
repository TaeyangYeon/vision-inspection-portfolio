using System;
using System.IO;
using System.Text;
using System.Diagnostics;
using InspectionSystem.Core.Services;
using InspectionSystem.Core.Models;
using Microsoft.Extensions.Logging.Abstractions;

namespace ModelBenchmark;

public class Program
{
    private static readonly int WarmupRuns = 5;
    private static readonly int BenchmarkRuns = 50;
    private static readonly int ImageSize = 640;
    
    public static async Task Main(string[] args)
    {
        Console.WriteLine("Model Benchmark Tool");
        Console.WriteLine("===================");
        
        var models = new[]
        {
            new ModelInfo { Name = "bottle_n", Path = "models/bottle/best.onnx", Description = "YOLOv8n Bottle Detection" },
            new ModelInfo { Name = "bottle_s", Path = "models/bottle_s/best.onnx", Description = "YOLOv8s Bottle Detection" },
            new ModelInfo { Name = "tile_n", Path = "models/tile/best.onnx", Description = "YOLOv8n Tile Detection" }
        };
        
        var results = new List<BenchmarkResult>();
        
        foreach (var model in models)
        {
            Console.WriteLine($"\nBenchmarking {model.Name} ({model.Description})...");
            
            if (!File.Exists(model.Path))
            {
                Console.WriteLine($"  ❌ Model not found: {model.Path}");
                continue;
            }
            
            var result = await BenchmarkModelAsync(model);
            if (result != null)
            {
                results.Add(result);
                Console.WriteLine($"  ✅ Avg: {result.AvgTimeMs:F1}ms, FPS: {result.Fps:F1}, Min: {result.MinTimeMs:F1}ms, Max: {result.MaxTimeMs:F1}ms");
            }
        }
        
        // Export results to CSV
        await ExportResultsAsync(results);
        
        // Print summary table
        PrintSummaryTable(results);
    }
    
    private static async Task<BenchmarkResult?> BenchmarkModelAsync(ModelInfo model)
    {
        try
        {
            var imageProcessor = new ImageProcessor();
            var engine = new OnnxInferenceEngine(NullLogger<OnnxInferenceEngine>.Instance, imageProcessor);
            
            await engine.LoadModelAsync(model.Path);
            if (!engine.IsModelLoaded)
            {
                Console.WriteLine($"  ❌ Failed to load model: {model.Path}");
                return null;
            }
            
            // Create dummy image data
            var dummyImage = CreateDummyImage(ImageSize, ImageSize);
            var options = new InferenceOptions
            {
                ConfidenceThreshold = 0.25f,
                IouThreshold = 0.45f,
                ImageSize = ImageSize
            };
            
            // Warmup runs
            Console.WriteLine($"  Warming up ({WarmupRuns} runs)...");
            for (int i = 0; i < WarmupRuns; i++)
            {
                await engine.RunInferenceAsync(dummyImage, ImageSize, ImageSize, options);
            }
            
            // Benchmark runs
            Console.WriteLine($"  Running benchmark ({BenchmarkRuns} runs)...");
            var times = new List<double>();
            
            for (int i = 0; i < BenchmarkRuns; i++)
            {
                var sw = Stopwatch.StartNew();
                var result = await engine.RunInferenceAsync(dummyImage, ImageSize, ImageSize, options);
                sw.Stop();
                
                times.Add(result.InferenceTimeMs);
            }
            
            engine.Dispose();
            
            var avgTime = times.Average();
            var minTime = times.Min();
            var maxTime = times.Max();
            var fps = 1000.0 / avgTime;
            
            return new BenchmarkResult
            {
                ModelName = model.Name,
                ModelDescription = model.Description,
                AvgTimeMs = avgTime,
                MinTimeMs = minTime,
                MaxTimeMs = maxTime,
                Fps = fps,
                WarmupRuns = WarmupRuns,
                BenchmarkRuns = BenchmarkRuns
            };
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  ❌ Error benchmarking {model.Name}: {ex.Message}");
            return null;
        }
    }
    
    private static byte[] CreateDummyImage(int width, int height)
    {
        var data = new byte[width * height * 3];
        var rng = new Random(42); // Seed for reproducible results
        rng.NextBytes(data);
        return data;
    }
    
    private static async Task ExportResultsAsync(List<BenchmarkResult> results)
    {
        var csvPath = "TestResults/benchmark_results.csv";
        Directory.CreateDirectory(Path.GetDirectoryName(csvPath)!);
        
        var csv = new StringBuilder();
        csv.AppendLine("Model,Description,AvgTimeMs,MinTimeMs,MaxTimeMs,FPS,WarmupRuns,BenchmarkRuns");
        
        foreach (var result in results)
        {
            csv.AppendLine($"{result.ModelName},{result.ModelDescription},{result.AvgTimeMs:F2},{result.MinTimeMs:F2},{result.MaxTimeMs:F2},{result.Fps:F2},{result.WarmupRuns},{result.BenchmarkRuns}");
        }
        
        await File.WriteAllTextAsync(csvPath, csv.ToString());
        Console.WriteLine($"\n📊 Results exported to: {csvPath}");
    }
    
    private static void PrintSummaryTable(List<BenchmarkResult> results)
    {
        Console.WriteLine("\n📊 Benchmark Summary:");
        Console.WriteLine("┌─────────────┬─────────────────────────────┬─────────────┬─────────────┬─────────────┬─────────────┐");
        Console.WriteLine("│    Model    │         Description         │   Avg (ms)  │   Min (ms)  │   Max (ms)  │     FPS     │");
        Console.WriteLine("├─────────────┼─────────────────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤");
        
        foreach (var result in results)
        {
            Console.WriteLine($"│ {result.ModelName,-11} │ {result.ModelDescription,-27} │ {result.AvgTimeMs,11:F1} │ {result.MinTimeMs,11:F1} │ {result.MaxTimeMs,11:F1} │ {result.Fps,11:F1} │");
        }
        
        Console.WriteLine("└─────────────┴─────────────────────────────┴─────────────┴─────────────┴─────────────┴─────────────┘");
    }
}

public class ModelInfo
{
    public required string Name { get; init; }
    public required string Path { get; init; }
    public required string Description { get; init; }
}

public class BenchmarkResult
{
    public required string ModelName { get; init; }
    public required string ModelDescription { get; init; }
    public required double AvgTimeMs { get; init; }
    public required double MinTimeMs { get; init; }
    public required double MaxTimeMs { get; init; }
    public required double Fps { get; init; }
    public required int WarmupRuns { get; init; }
    public required int BenchmarkRuns { get; init; }
}