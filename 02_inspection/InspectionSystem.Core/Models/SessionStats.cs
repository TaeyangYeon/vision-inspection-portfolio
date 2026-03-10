namespace InspectionSystem.Core.Models
{
    public class SessionStats
    {
        public int TotalInspected { get; set; }
        public int OkCount { get; set; }
        public int NgCount { get; set; }
        public double NgRate => TotalInspected > 0 ? (double)NgCount / TotalInspected * 100 : 0;
        public double AverageInferenceMs { get; set; }
        public double EstimatedFps => AverageInferenceMs > 0 ? 1000.0 / AverageInferenceMs : 0;
    }
}