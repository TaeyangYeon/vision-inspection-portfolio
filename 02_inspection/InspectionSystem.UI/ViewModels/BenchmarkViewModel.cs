using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InspectionSystem.Core.Interfaces;
using InspectionSystem.Core.Models;

namespace InspectionSystem.UI.ViewModels
{
    public partial class BenchmarkViewModel : ObservableObject
    {
        private readonly IInferenceEngine _inferenceEngine;
        private readonly IImageProcessor _imageProcessor;
        private readonly ISessionLogger _sessionLogger;
        private readonly ISettingsService _settingsService;

        [ObservableProperty]
        private string _statusMessage = "Load a model and run benchmark.";

        [ObservableProperty]
        private bool _isRunning = false;

        [ObservableProperty]
        private double _minTimeMs = 0;

        [ObservableProperty]
        private double _maxTimeMs = 0;

        [ObservableProperty]
        private double _avgTimeMs = 0;

        [ObservableProperty]
        private double _fps = 0;

        [ObservableProperty]
        private int _totalRuns = 0;

        [ObservableProperty]
        private int _warmupRuns = 5;

        [ObservableProperty]
        private int _benchmarkRuns = 20;

        [ObservableProperty]
        private string _csvExportPath = string.Empty;

        [ObservableProperty]
        private string _csvStatusMessage = string.Empty;

        [ObservableProperty]
        private ObservableCollection<BenchmarkRunItem> _runResults = new();

        [ObservableProperty]
        private ObservableCollection<SessionRecordItem> _sessionRecords = new();

        public BenchmarkViewModel(
            IInferenceEngine inferenceEngine,
            IImageProcessor imageProcessor,
            ISessionLogger sessionLogger,
            ISettingsService settingsService)
        {
            _inferenceEngine = inferenceEngine;
            _imageProcessor = imageProcessor;
            _sessionLogger = sessionLogger;
            _settingsService = settingsService;

            var settings = settingsService.Current;
            CsvExportPath = Path.Combine(settings.SavePath, "session_log.csv");
        }

        [RelayCommand]
        private async Task RunBenchmarkAsync()
        {
            if (!_inferenceEngine.IsModelLoaded)
            {
                StatusMessage = "Model not loaded. Load model in Inspection tab first.";
                return;
            }

            try
            {
                IsRunning = true;
                RunResults.Clear();
                StatusMessage = $"Warming up ({WarmupRuns} runs)...";

                byte[] dummyImage = CreateDummyImage(640, 640);
                var options = new InferenceOptions
                {
                    ConfidenceThreshold = 0.25f,
                    IouThreshold = 0.45f,
                    ImageSize = 640,
                };

                // warmup
                for (int i = 0; i < WarmupRuns; i++)
                {
                    await Task.Run(() =>
                        _inferenceEngine.RunInferenceAsync(
                            dummyImage, 640, 640, options));
                }

                StatusMessage = $"Running benchmark ({BenchmarkRuns} runs)...";

                double totalMs = 0;
                double minMs = double.MaxValue;
                double maxMs = 0;

                for (int i = 0; i < BenchmarkRuns; i++)
                {
                    var result = await Task.Run(() =>
                        _inferenceEngine.RunInferenceAsync(
                            dummyImage, 640, 640, options));

                    double ms = result.InferenceTimeMs;
                    totalMs += ms;
                    if (ms < minMs) minMs = ms;
                    if (ms > maxMs) maxMs = ms;

                    RunResults.Add(new BenchmarkRunItem
                    {
                        RunNumber = i + 1,
                        TimeMs = ms,
                    });
                }

                MinTimeMs = minMs;
                MaxTimeMs = maxMs;
                AvgTimeMs = totalMs / BenchmarkRuns;
                Fps = 1000.0 / AvgTimeMs;
                TotalRuns = BenchmarkRuns;

                StatusMessage = $"Benchmark complete. Avg: {AvgTimeMs:F1}ms / {Fps:F1} FPS";
            }
            catch (Exception ex)
            {
                StatusMessage = $"Benchmark failed: {ex.Message}";
            }
            finally
            {
                IsRunning = false;
            }
        }

        [RelayCommand]
        private void RefreshSessionRecords()
        {
            SessionRecords.Clear();
            var stats = _sessionLogger.GetStats();
            var records = _sessionLogger.GetHistory();

            foreach (var r in records)
            {
                SessionRecords.Add(new SessionRecordItem
                {
                    Timestamp = r.Timestamp.ToString("HH:mm:ss"),
                    ImageName = Path.GetFileName(r.ImagePath),
                    Result = r.IsNG ? "NG" : "OK",
                    ResultColor = r.IsNG ? "#F38BA8" : "#A6E3A1",
                    DefectCount = r.DefectCount,
                    InferenceTimeMs = r.InferenceTimeMs,
                    Confidence = r.Confidence,
                });
            }

            StatusMessage = $"Session: {stats.TotalInspected} total, " +
                            $"{stats.NgCount} NG, " +
                            $"avg {stats.AverageInferenceMs:F1}ms";
        }

        [RelayCommand]
        private async Task ExportCsvAsync()
        {
            try
            {
                await _sessionLogger.ExportToCsvAsync(CsvExportPath);
                CsvStatusMessage = $"Exported to {CsvExportPath}";
            }
            catch (Exception ex)
            {
                CsvStatusMessage = $"Export failed: {ex.Message}";
            }
        }

        private byte[] CreateDummyImage(int width, int height)
        {
            var data = new byte[width * height * 3];
            var rng = new Random(42);
            rng.NextBytes(data);
            return data;
        }
    }

    public class BenchmarkRunItem
    {
        public int RunNumber { get; set; }
        public double TimeMs { get; set; }
    }

    public class SessionRecordItem
    {
        public string Timestamp { get; set; } = string.Empty;
        public string ImageName { get; set; } = string.Empty;
        public string Result { get; set; } = string.Empty;
        public string ResultColor { get; set; } = string.Empty;
        public int DefectCount { get; set; }
        public double InferenceTimeMs { get; set; }
        public float Confidence { get; set; }
    }
}