using System;

namespace InspectionSystem.Core.Models
{
    public class InspectionRecord
    {
        public DateTime Timestamp { get; set; } = DateTime.Now;
        public string ImagePath { get; set; } = string.Empty;
        public bool IsNG { get; set; }
        public int DefectCount { get; set; }
        public string DefectTypes { get; set; } = string.Empty;
        public double InferenceTimeMs { get; set; }
        public float Confidence { get; set; }
    }
}