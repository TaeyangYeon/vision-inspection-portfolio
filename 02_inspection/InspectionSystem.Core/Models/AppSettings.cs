namespace InspectionSystem.Core.Models
{
    public class AppSettings
    {
        public string ModelPath { get; set; } = "models/best.onnx";
        public float ConfidenceThreshold { get; set; } = 0.25f;
        public float IouThreshold { get; set; } = 0.45f;
        public int ImageSize { get; set; } = 640;
        public string SavePath { get; set; } = "outputs/ng_images";
        public bool AutoSaveNg { get; set; } = true;
        public string GradCamApiUrl { get; set; } = "http://localhost:8000";
        public bool EnableGradCam { get; set; } = true;
        public string Theme { get; set; } = "Dark";
    }
}