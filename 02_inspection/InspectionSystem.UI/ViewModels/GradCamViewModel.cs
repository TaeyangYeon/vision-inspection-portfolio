using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Avalonia.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InspectionSystem.Core.Interfaces;
using InspectionSystem.Core.Models;

namespace InspectionSystem.UI.ViewModels
{
    public partial class GradCamViewModel : ObservableObject
    {
        private readonly IGradCamService _gradCamService;
        private readonly ISettingsService _settingsService;

        private byte[]? _currentImageData;
        private int _currentImageWidth;
        private int _currentImageHeight;

        [ObservableProperty]
        private Bitmap? _originalImage;

        [ObservableProperty]
        private Bitmap? _heatmapImage;

        [ObservableProperty]
        private Bitmap? _overlayImage;

        [ObservableProperty]
        private bool _isGenerating = false;

        [ObservableProperty]
        private string _statusMessage = "Load an image and generate GradCAM heatmap.";

        [ObservableProperty]
        private int _targetClass = 0;

        [ObservableProperty]
        private float _alpha = 0.5f;

        [ObservableProperty]
        private float _camMean = 0f;

        [ObservableProperty]
        private float _camMax = 0f;

        [ObservableProperty]
        private bool _isApiAvailable = false;

        [ObservableProperty]
        private string _apiStatus = "GradCAM API: Not connected";

        public GradCamViewModel(
            IGradCamService gradCamService,
            ISettingsService settingsService)
        {
            _gradCamService = gradCamService;
            _settingsService = settingsService;
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
                    Title = "Select Image for GradCAM",
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
                var bytes = await File.ReadAllBytesAsync(filePath);
                using var ms = new MemoryStream(bytes);
                var bitmap = new Bitmap(ms);

                _currentImageWidth = bitmap.PixelSize.Width;
                _currentImageHeight = bitmap.PixelSize.Height;
                _currentImageData = ConvertBitmapToRgbBytes(bitmap);

                OriginalImage = bitmap;
                HeatmapImage = null;
                OverlayImage = null;
                StatusMessage = $"Image loaded: {Path.GetFileName(filePath)}";
            }
            catch (Exception ex)
            {
                StatusMessage = $"Failed to load image: {ex.Message}";
            }
        }

        [RelayCommand]
        private async Task GenerateGradCamAsync()
        {
            if (_currentImageData == null)
            {
                StatusMessage = "No image loaded.";
                return;
            }

            try
            {
                IsGenerating = true;
                StatusMessage = "Generating GradCAM heatmap...";

                var result = await _gradCamService.GenerateAsync(
                    _currentImageData,
                    _currentImageWidth,
                    _currentImageHeight,
                    TargetClass,
                    Alpha
                );

                if (!result.Success)
                {
                    StatusMessage = $"GradCAM failed: {result.ErrorMessage}";
                    ApiStatus = "GradCAM API: Not available (start FastAPI server)";
                    IsApiAvailable = false;
                    return;
                }

                HeatmapImage = ConvertBytesToBitmap(result.HeatmapData, result.Width, result.Height);
                OverlayImage = ConvertBytesToBitmap(result.OverlayData, result.Width, result.Height);
                CamMean = result.CamMean;
                CamMax = result.CamMax;
                IsApiAvailable = true;
                ApiStatus = "GradCAM API: Connected";
                StatusMessage = $"GradCAM generated. Mean={result.CamMean:F4} Max={result.CamMax:F4}";
            }
            catch (Exception ex)
            {
                StatusMessage = $"Error: {ex.Message}";
            }
            finally
            {
                IsGenerating = false;
            }
        }

        private byte[] ConvertBitmapToRgbBytes(Bitmap bitmap)
        {
            int w = bitmap.PixelSize.Width;
            int h = bitmap.PixelSize.Height;
            byte[] data = new byte[w * h * 3];
            
            // Simplified placeholder implementation for now
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    int dstIdx = (y * w + x) * 3;
                    data[dstIdx + 0] = 128; // R placeholder
                    data[dstIdx + 1] = 128; // G placeholder  
                    data[dstIdx + 2] = 128; // B placeholder
                }
            }
            return data;
        }

        private Bitmap ConvertBytesToBitmap(byte[] data, int width, int height)
        {
            byte[] bgra = new byte[width * height * 4];
            for (int i = 0; i < width * height; i++)
            {
                bgra[i * 4 + 0] = data[i * 3 + 2]; // B
                bgra[i * 4 + 1] = data[i * 3 + 1]; // G  
                bgra[i * 4 + 2] = data[i * 3 + 0]; // R
                bgra[i * 4 + 3] = 255;             // A
            }
            
            var writeableBitmap = new Avalonia.Media.Imaging.WriteableBitmap(
                new Avalonia.PixelSize(width, height),
                new Avalonia.Vector(96, 96),
                Avalonia.Platform.PixelFormat.Bgra8888,
                Avalonia.Platform.AlphaFormat.Opaque);
                
            using var frameBuffer = writeableBitmap.Lock();
            System.Runtime.InteropServices.Marshal.Copy(bgra, 0, frameBuffer.Address, bgra.Length);
            
            return writeableBitmap;
        }
    }
}