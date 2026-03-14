using System;
using NUnit.Framework;
using InspectionSystem.Core.Models;
using InspectionSystem.Core.Services;

namespace InspectionSystem.Tests.Core
{
    [TestFixture]
    public class ImageProcessorTests
    {
        private ImageProcessor _processor = null!;

        [SetUp]
        public void SetUp()
        {
            _processor = new ImageProcessor();
        }

        private byte[] CreateDummyImage(int width, int height)
        {
            var data = new byte[width * height * 3];
            var rng = new Random(42);
            rng.NextBytes(data);
            return data;
        }

        private bool IsOpenCvAvailable()
        {
            try
            {
                var dummyImage = CreateDummyImage(1, 1);
                _processor.ResizeImage(dummyImage, 1, 1, 1, 1);
                return true;
            }
            catch (Exception ex) when (ex is System.DllNotFoundException ||
                                       ex is System.TypeInitializationException ||
                                       ex.InnerException is System.DllNotFoundException)
            {
                return false;
            }
        }

        [Test]
        public void Preprocess_ValidImage_ReturnsTensorWithCorrectSize()
        {
            if (!IsOpenCvAvailable())
                Assert.Ignore("OpenCV native libraries not found. Skipping.");

            var imageData = CreateDummyImage(320, 240);
            var result = _processor.Preprocess(imageData, 320, 240, 640);

            Assert.That(result, Is.Not.Null);
            Assert.That(result.Tensor.Length, Is.EqualTo(3 * 640 * 640));
            Assert.That(result.TensorWidth, Is.EqualTo(640));
            Assert.That(result.TensorHeight, Is.EqualTo(640));
            Assert.That(result.OriginalWidth, Is.EqualTo(320));
            Assert.That(result.OriginalHeight, Is.EqualTo(240));
        }

        [Test]
        public void Preprocess_ValidImage_ScaleIsCorrect()
        {
            if (!IsOpenCvAvailable())
                Assert.Ignore("OpenCV native libraries not found. Skipping.");

            var imageData = CreateDummyImage(320, 240);
            var result = _processor.Preprocess(imageData, 320, 240, 640);

            float expectedScale = 640f / 320f;
            Assert.That(result.Scale, Is.EqualTo(expectedScale).Within(0.001f));
        }

        [Test]
        public void Preprocess_ValidImage_TensorValuesNormalized()
        {
            if (!IsOpenCvAvailable())
                Assert.Ignore("OpenCV native libraries not found. Skipping.");

            var imageData = CreateDummyImage(64, 64);
            var result = _processor.Preprocess(imageData, 64, 64, 64);

            foreach (var val in result.Tensor)
            {
                Assert.That(val, Is.GreaterThanOrEqualTo(0f));
                Assert.That(val, Is.LessThanOrEqualTo(1f));
            }
        }

        [Test]
        public void Preprocess_NullImageData_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(
                () => _processor.Preprocess(null!, 320, 240)
            );
        }

        [Test]
        public void Preprocess_EmptyImageData_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(
                () => _processor.Preprocess(Array.Empty<byte>(), 320, 240)
            );
        }

        [Test]
        public void DrawDetections_NullImageData_ThrowsArgumentNullException()
        {
            var result = new DetectionResult();
            var options = new DrawOptions();

            Assert.Throws<ArgumentNullException>(
                () => _processor.DrawDetections(null!, 320, 240, result, options)
            );
        }

        [Test]
        public void DrawDetections_NullDetectionResult_ThrowsArgumentNullException()
        {
            var imageData = CreateDummyImage(320, 240);
            var options = new DrawOptions();

            Assert.Throws<ArgumentNullException>(
                () => _processor.DrawDetections(imageData, 320, 240, null!, options)
            );
        }

        [Test]
        public void DrawDetections_EmptyDetections_ReturnsSameSizeImage()
        {
            if (!IsOpenCvAvailable())
                Assert.Ignore("OpenCV native libraries not found. Skipping.");

            var imageData = CreateDummyImage(320, 240);
            var result = new DetectionResult { Detections = new() };
            var options = new DrawOptions();

            var output = _processor.DrawDetections(imageData, 320, 240, result, options);

            Assert.That(output, Is.Not.Null);
            Assert.That(output.Length, Is.EqualTo(imageData.Length));
        }

        [Test]
        public void ResizeImage_ValidInput_ReturnsCorrectSize()
        {
            if (!IsOpenCvAvailable())
                Assert.Ignore("OpenCV native libraries not found. Skipping.");

            var imageData = CreateDummyImage(320, 240);
            var result = _processor.ResizeImage(imageData, 320, 240, 160, 120);

            Assert.That(result.Length, Is.EqualTo(160 * 120 * 3));
        }

        [Test]
        public void ResizeImage_NullImageData_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(
                () => _processor.ResizeImage(null!, 320, 240, 160, 120)
            );
        }

        [Test]
        public void Preprocess_ZeroWidth_ThrowsArgumentOutOfRangeException()
        {
            var data = new byte[100 * 100 * 3];
            Assert.Throws<ArgumentOutOfRangeException>(
                () => _processor.Preprocess(data, 0, 100)
            );
        }

        [Test]
        public void Preprocess_ZeroHeight_ThrowsArgumentOutOfRangeException()
        {
            var data = new byte[100 * 100 * 3];
            Assert.Throws<ArgumentOutOfRangeException>(
                () => _processor.Preprocess(data, 100, 0)
            );
        }

        [Test]
        public void Preprocess_ValidInput_ReturnsTensorWithCorrectLength()
        {
            if (!IsOpenCvAvailable())
                Assert.Ignore("OpenCV native libraries not found. Skipping.");

            var data = new byte[100 * 100 * 3];
            var result = _processor.Preprocess(data, 100, 100);
            Assert.That(result, Is.Not.Null);
            Assert.That(result.Tensor, Has.Length.EqualTo(3 * 640 * 640));
        }

        [Test]
        public void Preprocess_ValidInput_TensorValuesNormalized()
        {
            if (!IsOpenCvAvailable())
                Assert.Ignore("OpenCV native libraries not found. Skipping.");

            var data = new byte[100 * 100 * 3];
            for (int i = 0; i < data.Length; i++) data[i] = 128;
            var result = _processor.Preprocess(data, 100, 100);
            foreach (var v in result.Tensor)
            {
                Assert.That(v, Is.GreaterThanOrEqualTo(0.0f));
                Assert.That(v, Is.LessThanOrEqualTo(1.0f));
            }
        }

        [Test]
        public void DrawDetections_EmptyDetections_ReturnsImageSameSize()
        {
            if (!IsOpenCvAvailable())
                Assert.Ignore("OpenCV native libraries not found. Skipping.");

            var data = new byte[100 * 100 * 3];
            var result = new InspectionSystem.Core.Models.DetectionResult
            {
                Detections = new System.Collections.Generic.List<InspectionSystem.Core.Models.Detection>(),
                ImageWidth = 100,
                ImageHeight = 100,
            };
            var output = _processor.DrawDetections(data, 100, 100, result, new InspectionSystem.Core.Models.DrawOptions());
            Assert.That(output, Has.Length.EqualTo(100 * 100 * 3));
        }
    }
}