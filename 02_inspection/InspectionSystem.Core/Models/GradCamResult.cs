namespace InspectionSystem.Core.Models
{
    public class GradCamResult
    {
        public bool Success { get; set; }
        public byte[] HeatmapData { get; set; } = System.Array.Empty<byte>();
        public byte[] OverlayData { get; set; } = System.Array.Empty<byte>();
        public int Width { get; set; }
        public int Height { get; set; }
        public float CamMean { get; set; }
        public float CamMax { get; set; }
        public string ErrorMessage { get; set; } = string.Empty;
    }
}