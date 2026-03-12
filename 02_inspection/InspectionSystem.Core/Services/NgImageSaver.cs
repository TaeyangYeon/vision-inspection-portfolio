using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using InspectionSystem.Core.Models;

namespace InspectionSystem.Core.Services
{
    public class NgImageSaver
    {
        private readonly ILogger<NgImageSaver> _logger;

        public NgImageSaver(ILogger<NgImageSaver> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        public async Task SaveAsync(
            byte[] imageData,
            int width,
            int height,
            DetectionResult result,
            string savePath)
        {
            if (imageData == null || imageData.Length == 0)
                throw new ArgumentNullException(nameof(imageData));

            if (string.IsNullOrWhiteSpace(savePath))
                throw new ArgumentNullException(nameof(savePath));

            if (!result.IsNG) return;

            Directory.CreateDirectory(savePath);

            string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff");
            string defectTypes = string.Join("_",
                GetUniqueDefectTypes(result));
            string fileName = $"NG_{timestamp}_{defectTypes}.jpg";
            string filePath = Path.Combine(savePath, fileName);

            await File.WriteAllBytesAsync(filePath, imageData);
            _logger.LogInformation("NG image saved: {Path}", filePath);
        }

        private System.Collections.Generic.List<string> GetUniqueDefectTypes(
            DetectionResult result)
        {
            var types = new System.Collections.Generic.HashSet<string>();
            foreach (var det in result.Detections)
                types.Add(det.ClassName);
            return new System.Collections.Generic.List<string>(types);
        }
    }
}