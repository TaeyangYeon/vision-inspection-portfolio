using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InspectionSystem.Core.Interfaces;
using InspectionSystem.Core.Models;
using InspectionSystem.Core.Services;
using Microsoft.Extensions.Logging;

namespace InspectionSystem.UI.ViewModels
{
    public partial class InspectionViewModel : ObservableObject
    {
        private readonly IInferenceEngine _inferenceEngine;
        private readonly IImageProcessor _imageProcessor;
        private readonly ISessionLogger _sessionLogger;
        private readonly ISettingsService _settingsService;
        private readonly NgImageSaver _ngImageSaver;
        private readonly ILogger<InspectionViewModel> _logger;

        private byte[]? _currentImageData;
        private int _currentImageWidth;
        private int _currentImageHeight;
        private string _currentImagePath = string.Empty;
        private CancellationTokenSource? _cts;
        private SemaphoreSlim _inferenceSemaphore = new(1, 1);

        [ObservableProperty]
        private Bitmap? _displayImage;

        [ObservableProperty]
        private Bitmap? _resultImage;

        [ObservableProperty]
        private bool _isInspecting = false;

        [ObservableProperty]
        private bool _isModelLoaded = false;

        [ObservableProperty]
        private string _statusMessage = "Load a model to start inspection.";

        [ObservableProperty]
        private string _resultLabel = "NO RESULT";

        [ObservableProperty]
        private string _resultColor = "#6C7086";

        [ObservableProperty]
        private double _inferenceTimeMs = 0;

        [ObservableProperty]
        private int _detectionCount = 0;

        [ObservableProperty]
        private float _confidenceThreshold = 0.25f;

        [ObservableProperty]
        private float _iouThreshold = 0.45f;

        [ObservableProperty]
        private bool _autoSaveNg = true;

        [ObservableProperty]
        private ObservableCollection<DetectionItemViewModel> _detections = new();

        public InspectionViewModel(
            IInferenceEngine inferenceEngine,
            IImageProcessor imageProcessor,
            ISessionLogger sessionLogger,
            ISettingsService settingsService,
            NgImageSaver ngImageSaver,
            ILogger<InspectionViewModel> logger)
        {
            _inferenceEngine = inferenceEngine;
            _imageProcessor = imageProcessor;
            _sessionLogger = sessionLogger;
            _settingsService = settingsService;
            _ngImageSaver = ngImageSaver;
            _logger = logger;

            var settings = settingsService.Current;
            ConfidenceThreshold = settings.ConfidenceThreshold;
            IouThreshold = settings.IouThreshold;
            AutoSaveNg = settings.AutoSaveNg;
        }

        [RelayCommand]
        private async Task LoadModelAsync()
        {
            try
            {
                StatusMessage = "Loading model...";
                var modelPath = _settingsService.Current.ModelPath;

                if (!File.Exists(modelPath))
                {
                    StatusMessage = $"Model not found: {modelPath}";
                    return;
                }

                await Task.Run(() => _inferenceEngine.LoadModelAsync(modelPath));
                IsModelLoaded = true;
                StatusMessage = $"Model loaded: {Path.GetFileName(modelPath)}";
                _logger.LogInformation("Model loaded: {Path}", modelPath);
            }
            catch (Exception ex)
            {
                StatusMessage = $"Failed to load model: {ex.Message}";
                IsModelLoaded = false;
                _logger.LogError(ex, "Failed to load model");
            }
        }

        [RelayCommand]
        private async Task OpenImageAsync()
        {
            try
            {
                var topLevel = Avalonia.Application.Current?.ApplicationLifetime
                    as Avalonia.Controls.ApplicationLifetimes.IClassicDesktopStyleApplicationLifetime;

                if (topLevel?.MainWindow == null) return;

                var dialog = new Avalonia.Platform.Storage.FilePickerOpenOptions
                {
                    Title = "Select Image",
                    AllowMultiple = false,
                    FileTypeFilter = new[]
                    {
                        new Avalonia.Platform.Storage.FilePickerFileType("Images")
                        {
                            Patterns = new[] { "*.jpg", "*.jpeg", "*.png", "*.bmp" }
                        }
                    }
                };

                var files = await topLevel.MainWindow.StorageProvider.OpenFilePickerAsync(dialog);
                if (files == null || files.Count == 0) return;

                var filePath = files[0].Path.LocalPath;
                await LoadImageFromPathAsync(filePath);
            }
            catch (Exception ex)
            {
                StatusMessage = $"Failed to open image: {ex.Message}";
            }
        }

        public async Task LoadImageFromPathAsync(string filePath)
        {
            try
            {
                var bytes = await File.ReadAllBytesAsync(filePath);
                using var ms = new MemoryStream(bytes);
                var bitmap = new Bitmap(ms);

                _currentImageWidth = bitmap.PixelSize.Width;
                _currentImageHeight = bitmap.PixelSize.Height;
                _currentImageData = ConvertBitmapToRgbBytes(bitmap);
                _currentImagePath = filePath;

                DisplayImage = bitmap;
                ResultImage = null;
                ResultLabel = "READY";
                ResultColor = "#6C7086";
                Detections.Clear();
                StatusMessage = $"Image loaded: {Path.GetFileName(filePath)} ({_currentImageWidth}x{_currentImageHeight})";
            }
            catch (Exception ex)
            {
                StatusMessage = $"Failed to load image: {ex.Message}";
            }
        }

        [RelayCommand]
        private async Task RunInspectionAsync()
        {
            if (!IsModelLoaded)
            {
                StatusMessage = "Model not loaded. Click Load Model first.";
                return;
            }

            if (_currentImageData == null)
            {
                StatusMessage = "No image loaded. Click Open Image first.";
                return;
            }

            if (!await _inferenceSemaphore.WaitAsync(0))
            {
                StatusMessage = "Inspection already in progress.";
                return;
            }

            _cts = new CancellationTokenSource();

            try
            {
                IsInspecting = true;
                StatusMessage = "Running inspection...";

                var options = new InferenceOptions
                {
                    ConfidenceThreshold = ConfidenceThreshold,
                    IouThreshold = IouThreshold,
                    ImageSize = 640,
                };

                var imageDataSnapshot = _currentImageData;
                var widthSnapshot = _currentImageWidth;
                var heightSnapshot = _currentImageHeight;

                var result = await Task.Run(async () =>
                    await _inferenceEngine.RunInferenceAsync(
                        imageDataSnapshot,
                        widthSnapshot,
                        heightSnapshot,
                        options,
                        _cts.Token
                    ), _cts.Token
                );

                _cts.Token.ThrowIfCancellationRequested();

                var drawOptions = new DrawOptions
                {
                    ShowLabel = true,
                    ShowConfidence = true,
                    LineThickness = 2,
                    FontScale = 0.5,
                };

                var resultBytes = await Task.Run(() =>
                    _imageProcessor.DrawDetections(
                        imageDataSnapshot,
                        widthSnapshot,
                        heightSnapshot,
                        result,
                        drawOptions
                    )
                );

                await Dispatcher.UIThread.InvokeAsync(() =>
                {
                    InferenceTimeMs = result.InferenceTimeMs;
                    DetectionCount = result.Detections.Count;

                    ResultImage = ConvertRgbBytesToBitmap(
                        resultBytes, widthSnapshot, heightSnapshot
                    );

                    Detections.Clear();
                    foreach (var det in result.Detections)
                    {
                        Detections.Add(new DetectionItemViewModel
                        {
                            ClassName = det.ClassName,
                            Confidence = det.Confidence,
                            BBox = $"[{det.Box.X1},{det.Box.Y1},{det.Box.X2},{det.Box.Y2}]",
                        });
                    }

                    if (result.IsNG)
                    {
                        ResultLabel = "NG";
                        ResultColor = "#F38BA8";
                        StatusMessage = $"NG - {result.Detections.Count} defect(s) in {result.InferenceTimeMs:F1}ms";
                    }
                    else
                    {
                        ResultLabel = "OK";
                        ResultColor = "#A6E3A1";
                        StatusMessage = $"OK - No defects in {result.InferenceTimeMs:F1}ms";
                    }
                });

                _sessionLogger.Log(new InspectionRecord
                {
                    Timestamp = DateTime.Now,
                    ImagePath = _currentImagePath,
                    IsNG = result.IsNG,
                    DefectCount = result.Detections.Count,
                    InferenceTimeMs = result.InferenceTimeMs,
                    Confidence = result.Detections.Count > 0
                        ? result.Detections[0].Confidence : 0f,
                });

                if (AutoSaveNg && result.IsNG)
                {
                    var savePath = _settingsService.Current.SavePath;
                    await Task.Run(async () =>
                        await _ngImageSaver.SaveAsync(
                            resultBytes, widthSnapshot, heightSnapshot,
                            result, savePath
                        )
                    );
                    _logger.LogInformation(
                        "NG image auto-saved to {Path}", savePath);
                }
            }
            catch (OperationCanceledException)
            {
                await Dispatcher.UIThread.InvokeAsync(() =>
                {
                    StatusMessage = "Inspection cancelled.";
                    IsInspecting = false;
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Inspection failed");
                await Dispatcher.UIThread.InvokeAsync(() =>
                {
                    StatusMessage = $"Inspection failed: {ex.Message}";
                });
            }
            finally
            {
                _inferenceSemaphore.Release();
                await Dispatcher.UIThread.InvokeAsync(() =>
                {
                    IsInspecting = false;
                });
            }
        }

        [RelayCommand]
        private void CancelInspection()
        {
            _cts?.Cancel();
            StatusMessage = "Cancelling...";
        }

        private byte[] ConvertBitmapToRgbBytes(Bitmap bitmap)
        {
            using var ms = new MemoryStream();
            bitmap.Save(ms);
            return ms.ToArray();
        }

        private Bitmap ConvertRgbBytesToBitmap(byte[] data, int width, int height)
        {
            using var ms = new MemoryStream(data);
            return new Bitmap(ms);
        }
    }

    public class DetectionItemViewModel
    {
        public string ClassName { get; set; } = string.Empty;
        public float Confidence { get; set; }
        public string BBox { get; set; } = string.Empty;
    }
}