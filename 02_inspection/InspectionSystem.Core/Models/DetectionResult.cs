using System.Collections.Generic;

namespace InspectionSystem.Core.Models
{
    public class DetectionResult
    {
        public bool IsNG => Detections.Count > 0;
        public List<Detection> Detections { get; set; } = new();
        public double InferenceTimeMs { get; set; }
        public int ImageWidth { get; set; }
        public int ImageHeight { get; set; }
    }

    public class Detection
    {
        public int ClassId { get; set; }
        public string ClassName { get; set; } = string.Empty;
        public float Confidence { get; set; }
        public BoundingBox Box { get; set; } = new();
    }

    public class BoundingBox
    {
        public int X1 { get; set; }
        public int Y1 { get; set; }
        public int X2 { get; set; }
        public int Y2 { get; set; }
        public int Width => X2 - X1;
        public int Height => Y2 - Y1;
    }
}