using InspectionSystem.Core.Models;

namespace InspectionSystem.Core.Interfaces
{
    public interface IImageProcessor
    {
        ProcessedImage Preprocess(byte[] imageData, int width, int height, int targetSize = 640);
        byte[] DrawDetections(byte[] imageData, int width, int height, DetectionResult result, DrawOptions options);
        byte[] ResizeImage(byte[] imageData, int width, int height, int targetWidth, int targetHeight);
    }
}