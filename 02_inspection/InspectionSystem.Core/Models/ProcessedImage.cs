namespace InspectionSystem.Core.Models
{
    public class ProcessedImage
    {
        public float[] Tensor { get; set; } = System.Array.Empty<float>();
        public int TensorWidth { get; set; }
        public int TensorHeight { get; set; }
        public float Scale { get; set; }
        public int OriginalWidth { get; set; }
        public int OriginalHeight { get; set; }
    }
}