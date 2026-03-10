namespace InspectionSystem.Core.Models
{
    public class DrawOptions
    {
        public bool ShowLabel { get; set; } = true;
        public bool ShowConfidence { get; set; } = true;
        public int LineThickness { get; set; } = 2;
        public double FontScale { get; set; } = 0.5;
    }
}