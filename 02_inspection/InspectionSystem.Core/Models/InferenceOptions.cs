namespace InspectionSystem.Core.Models
{
    public class InferenceOptions
    {
        public float ConfidenceThreshold { get; set; } = 0.25f;
        public float IouThreshold { get; set; } = 0.45f;
        public int ImageSize { get; set; } = 640;
    }
}