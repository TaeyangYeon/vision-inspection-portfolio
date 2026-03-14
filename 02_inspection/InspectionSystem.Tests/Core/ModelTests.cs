using NUnit.Framework;
using InspectionSystem.Core.Models;

namespace InspectionSystem.Tests.Core
{
    [TestFixture]
    public class ModelTests
    {
        // --- BoundingBox ---

        [Test]
        public void BoundingBox_DefaultValues_AreZero()
        {
            var box = new BoundingBox();
            Assert.That(box.X1, Is.EqualTo(0f));
            Assert.That(box.Y1, Is.EqualTo(0f));
            Assert.That(box.X2, Is.EqualTo(0f));
            Assert.That(box.Y2, Is.EqualTo(0f));
        }

        [Test]
        public void BoundingBox_SetValues_ReturnsCorrectly()
        {
            var box = new BoundingBox { X1=10, Y1=20, X2=100, Y2=200 };
            Assert.That(box.X1, Is.EqualTo(10f));
            Assert.That(box.Y1, Is.EqualTo(20f));
            Assert.That(box.X2, Is.EqualTo(100f));
            Assert.That(box.Y2, Is.EqualTo(200f));
        }

        [Test]
        public void BoundingBox_Width_IsX2MinusX1()
        {
            var box = new BoundingBox { X1=10, Y1=20, X2=110, Y2=220 };
            Assert.That(box.Width, Is.EqualTo(100f));
        }

        [Test]
        public void BoundingBox_Height_IsY2MinusY1()
        {
            var box = new BoundingBox { X1=10, Y1=20, X2=110, Y2=220 };
            Assert.That(box.Height, Is.EqualTo(200f));
        }

        [Test]
        public void BoundingBox_Area_IsWidthTimesHeight()
        {
            var box = new BoundingBox { X1=0, Y1=0, X2=10, Y2=20 };
            Assert.That(box.Area, Is.EqualTo(200f));
        }

        // --- GradCamResult ---

        [Test]
        public void GradCamResult_DefaultSuccess_IsFalse()
        {
            var result = new GradCamResult();
            Assert.That(result.Success, Is.False);
        }

        [Test]
        public void GradCamResult_SetSuccess_ReturnsTrue()
        {
            var result = new GradCamResult { Success = true };
            Assert.That(result.Success, Is.True);
        }

        [Test]
        public void GradCamResult_ErrorMessage_DefaultIsNullOrEmpty()
        {
            var result = new GradCamResult();
            Assert.That(result.ErrorMessage,
                Is.Null.Or.Empty);
        }

        [Test]
        public void GradCamResult_CamMean_DefaultIsZero()
        {
            var result = new GradCamResult();
            Assert.That(result.CamMean, Is.EqualTo(0f));
        }

        [Test]
        public void GradCamResult_CamMax_DefaultIsZero()
        {
            var result = new GradCamResult();
            Assert.That(result.CamMax, Is.EqualTo(0f));
        }

        [Test]
        public void GradCamResult_SetValues_ReturnsCorrectly()
        {
            var result = new GradCamResult
            {
                Success = true,
                CamMean = 0.35f,
                CamMax = 0.92f,
                Width = 640,
                Height = 640,
                HeatmapData = new byte[640 * 640 * 3],
                OverlayData = new byte[640 * 640 * 3],
            };

            Assert.That(result.Success, Is.True);
            Assert.That(result.CamMean, Is.EqualTo(0.35f).Within(0.001f));
            Assert.That(result.CamMax, Is.EqualTo(0.92f).Within(0.001f));
            Assert.That(result.Width, Is.EqualTo(640));
            Assert.That(result.Height, Is.EqualTo(640));
            Assert.That(result.HeatmapData, Has.Length.EqualTo(640 * 640 * 3));
            Assert.That(result.OverlayData, Has.Length.EqualTo(640 * 640 * 3));
        }

        // --- DetectionResult.IsNG ---

        [Test]
        public void DetectionResult_NoDetections_IsNGFalse()
        {
            var result = new DetectionResult
            {
                Detections = new System.Collections.Generic.List<Detection>()
            };
            Assert.That(result.IsNG, Is.False);
        }

        [Test]
        public void DetectionResult_WithDetections_IsNGTrue()
        {
            var result = new DetectionResult
            {
                Detections = new System.Collections.Generic.List<Detection>
                {
                    new Detection
                    {
                        ClassId = 0,
                        ClassName = "broken_large",
                        Confidence = 0.9f,
                        Box = new BoundingBox { X1=0, Y1=0, X2=50, Y2=50 }
                    }
                }
            };
            Assert.That(result.IsNG, Is.True);
        }

        // --- InferenceOptions ---

        [Test]
        public void InferenceOptions_DefaultValues_AreReasonable()
        {
            var opts = new InferenceOptions();
            Assert.That(opts.ConfidenceThreshold, Is.GreaterThan(0f));
            Assert.That(opts.IouThreshold, Is.GreaterThan(0f));
            Assert.That(opts.ImageSize, Is.GreaterThan(0));
        }

        // --- AppSettings ---

        [Test]
        public void AppSettings_DefaultModelPath_IsNotEmpty()
        {
            var settings = new AppSettings();
            Assert.That(settings.ModelPath, Is.Not.Null.And.Not.Empty);
        }

        [Test]
        public void AppSettings_DefaultConfidence_IsInValidRange()
        {
            var settings = new AppSettings();
            Assert.That(settings.ConfidenceThreshold,
                Is.GreaterThan(0f).And.LessThan(1f));
        }

        [Test]
        public void AppSettings_DefaultIou_IsInValidRange()
        {
            var settings = new AppSettings();
            Assert.That(settings.IouThreshold,
                Is.GreaterThan(0f).And.LessThan(1f));
        }
    }
}