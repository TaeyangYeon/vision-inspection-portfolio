using System;
using System.Net.Http;
using System.Net.Http.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using InspectionSystem.Core.Interfaces;
using InspectionSystem.Core.Models;

namespace InspectionSystem.Core.Services
{
    public class GradCamService : IGradCamService, IDisposable
    {
        private readonly ILogger<GradCamService> _logger;
        private readonly HttpClient _httpClient;
        private readonly string _apiUrl;
        private bool _disposed = false;

        public bool IsAvailable { get; private set; } = false;

        public GradCamService(ILogger<GradCamService> logger, string apiUrl = "http://localhost:8000")
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            
            if (string.IsNullOrEmpty(apiUrl))
                throw new ArgumentNullException(nameof(apiUrl));
                
            _apiUrl = apiUrl;
            _httpClient = new HttpClient
            {
                Timeout = TimeSpan.FromSeconds(30)
            };
        }

        public async Task<GradCamResult> GenerateAsync(
            byte[] imageData,
            int width,
            int height,
            int targetClass,
            float alpha = 0.5f,
            CancellationToken cancellationToken = default)
        {
            if (imageData == null || imageData.Length == 0)
                throw new ArgumentNullException(nameof(imageData));
            
            if (width <= 0)
                throw new ArgumentOutOfRangeException(nameof(width), "Width must be greater than zero.");
            
            if (height <= 0)
                throw new ArgumentOutOfRangeException(nameof(height), "Height must be greater than zero.");

            try
            {
                var imageBase64 = Convert.ToBase64String(imageData);
                var request = new
                {
                    image_base64 = imageBase64,
                    width = width,
                    height = height,
                    target_class = targetClass,
                    alpha = alpha,
                };

                var response = await _httpClient.PostAsJsonAsync(
                    $"{_apiUrl}/gradcam",
                    request,
                    cancellationToken
                );

                if (!response.IsSuccessStatusCode)
                {
                    var error = await response.Content.ReadAsStringAsync(cancellationToken);
                    _logger.LogWarning("GradCAM API error: {Status} {Error}",
                        response.StatusCode, error);
                    return new GradCamResult
                    {
                        Success = false,
                        ErrorMessage = $"API returned {response.StatusCode}: {error}"
                    };
                }

                var result = await response.Content.ReadFromJsonAsync<GradCamApiResponse>(
                    cancellationToken: cancellationToken
                );

                if (result == null)
                {
                    return new GradCamResult
                    {
                        Success = false,
                        ErrorMessage = "Empty response from GradCAM API"
                    };
                }

                IsAvailable = true;

                return new GradCamResult
                {
                    Success = true,
                    HeatmapData = Convert.FromBase64String(result.HeatmapBase64),
                    OverlayData = Convert.FromBase64String(result.OverlayBase64),
                    Width = result.Width,
                    Height = result.Height,
                    CamMean = result.CamMean,
                    CamMax = result.CamMax,
                };
            }
            catch (HttpRequestException ex)
            {
                _logger.LogWarning(ex, "GradCAM API not reachable at {Url}", _apiUrl);
                IsAvailable = false;
                return new GradCamResult
                {
                    Success = false,
                    ErrorMessage = $"GradCAM API not available: {ex.Message}"
                };
            }
            catch (TaskCanceledException)
            {
                return new GradCamResult
                {
                    Success = false,
                    ErrorMessage = "GradCAM request timed out."
                };
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _httpClient.Dispose();
                _disposed = true;
            }
        }

        private class GradCamApiResponse
        {
            public string HeatmapBase64 { get; set; } = string.Empty;
            public string OverlayBase64 { get; set; } = string.Empty;
            public int Width { get; set; }
            public int Height { get; set; }
            public float CamMean { get; set; }
            public float CamMax { get; set; }
        }
    }
}