namespace InspectionSystem.Core.Models
{
    public class AppSettings
    {
        public string ModelPath { get; set; } = GetDefaultModelPath();
        public string Category { get; set; } = "bottle";
        public string CategoryModelPath { get; set; } = string.Empty;

        private static string GetDefaultModelPath()
        {
            var baseDir = AppContext.BaseDirectory;
            var candidates = new[]
            {
                System.IO.Path.Combine(baseDir, "models", "bottle", "best.onnx"),
                System.IO.Path.Combine(baseDir, "models", "best.onnx"),
                System.IO.Path.GetFullPath(System.IO.Path.Combine(baseDir,
                    "..", "..", "..", "..", "models", "best.onnx")),
                System.IO.Path.GetFullPath(System.IO.Path.Combine(baseDir,
                    "..", "..", "..", "..", "..", "02_inspection", "models", "best.onnx")),
            };
            foreach (var path in candidates)
            {
                if (System.IO.File.Exists(path))
                    return System.IO.Path.GetFullPath(path);
            }
            return "models/best.onnx";
        }
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