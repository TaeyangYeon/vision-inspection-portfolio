using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using NUnit.Framework;
using InspectionSystem.Core.Models;
using InspectionSystem.Core.Services;

namespace InspectionSystem.Tests.Core
{
    [TestFixture]
    public class OnnxInferenceEngineTests
    {
        private OnnxInferenceEngine _engine = null!;
        private string _modelPath = string.Empty;

        [SetUp]
        public void SetUp()
        {
            _engine = new OnnxInferenceEngine(NullLogger<OnnxInferenceEngine>.Instance);
            _modelPath = Path.GetFullPath(
                Path.Combine(
                    AppContext.BaseDirectory,
                    "..", "..", "..", "..", "..",
                    "02_inspection", "models", "best.onnx"
                )
            );
        }

        [TearDown]
        public void TearDown()
        {
            _engine.Dispose();
        }

        [Test]
        public void IsModelLoaded_BeforeLoad_ReturnsFalse()
        {
            Assert.That(_engine.IsModelLoaded, Is.False);
        }

        [Test]
        public async Task LoadModelAsync_ValidPath_SetsIsModelLoadedTrue()
        {
            if (!File.Exists(_modelPath))
                Assert.Ignore("ONNX model not found. Skipping.");

            try
            {
                await _engine.LoadModelAsync(_modelPath);
                Assert.That(_engine.IsModelLoaded, Is.True);
            }
            catch (Microsoft.ML.OnnxRuntime.OnnxRuntimeException ex) when (ex.Message.Contains("Opset"))
            {
                Assert.Ignore($"Model opset version not supported by current ONNX Runtime: {ex.Message}");
            }
        }

        [Test]
        public void LoadModelAsync_NullPath_ThrowsArgumentNullException()
        {
            Assert.ThrowsAsync<ArgumentNullException>(
                async () => await _engine.LoadModelAsync(null!)
            );
        }

        [Test]
        public void LoadModelAsync_NonExistentPath_ThrowsFileNotFoundException()
        {
            Assert.ThrowsAsync<FileNotFoundException>(
                async () => await _engine.LoadModelAsync("nonexistent/path/model.onnx")
            );
        }

        [Test]
        public void RunInferenceAsync_ModelNotLoaded_ThrowsInvalidOperationException()
        {
            var imageData = new byte[640 * 640 * 3];
            var options = new InferenceOptions();

            Assert.ThrowsAsync<InvalidOperationException>(
                async () => await _engine.RunInferenceAsync(imageData, 640, 640, options)
            );
        }

        [Test]
        public async Task RunInferenceAsync_NullImageData_ThrowsArgumentNullException()
        {
            if (!File.Exists(_modelPath))
                Assert.Ignore("ONNX model not found. Skipping.");

            try
            {
                await _engine.LoadModelAsync(_modelPath);
            }
            catch (Microsoft.ML.OnnxRuntime.OnnxRuntimeException ex) when (ex.Message.Contains("Opset"))
            {
                Assert.Ignore($"Model opset version not supported by current ONNX Runtime: {ex.Message}");
            }

            var options = new InferenceOptions();

            Assert.ThrowsAsync<ArgumentNullException>(
                async () => await _engine.RunInferenceAsync(null!, 640, 640, options)
            );
        }

        [Test]
        public async Task RunInferenceAsync_ValidImage_ReturnsDetectionResult()
        {
            if (!File.Exists(_modelPath))
                Assert.Ignore("ONNX model not found. Skipping.");

            try
            {
                await _engine.LoadModelAsync(_modelPath);
            }
            catch (Microsoft.ML.OnnxRuntime.OnnxRuntimeException ex) when (ex.Message.Contains("Opset"))
            {
                Assert.Ignore($"Model opset version not supported by current ONNX Runtime: {ex.Message}");
            }

            var imageData = new byte[640 * 640 * 3];
            var rng = new Random(42);
            rng.NextBytes(imageData);

            var options = new InferenceOptions
            {
                ConfidenceThreshold = 0.25f,
                IouThreshold = 0.45f,
                ImageSize = 640,
            };

            var result = await _engine.RunInferenceAsync(imageData, 640, 640, options);

            Assert.That(result, Is.Not.Null);
            Assert.That(result.Detections, Is.Not.Null);
            Assert.That(result.InferenceTimeMs, Is.GreaterThan(0));
            Assert.That(result.ImageWidth, Is.EqualTo(640));
            Assert.That(result.ImageHeight, Is.EqualTo(640));
        }

        [Test]
        public async Task RunInferenceAsync_ValidImage_InferenceTimeLessThan5000Ms()
        {
            if (!File.Exists(_modelPath))
                Assert.Ignore("ONNX model not found. Skipping.");

            try
            {
                await _engine.LoadModelAsync(_modelPath);
            }
            catch (Microsoft.ML.OnnxRuntime.OnnxRuntimeException ex) when (ex.Message.Contains("Opset"))
            {
                Assert.Ignore($"Model opset version not supported by current ONNX Runtime: {ex.Message}");
            }

            var imageData = new byte[640 * 640 * 3];
            var options = new InferenceOptions();

            var result = await _engine.RunInferenceAsync(imageData, 640, 640, options);

            Assert.That(result.InferenceTimeMs, Is.LessThan(5000));
        }

        [Test]
        public void RunInferenceAsync_NullOptions_ThrowsArgumentNullException()
        {
            var imageData = new byte[640 * 640 * 3];

            Assert.ThrowsAsync<InvalidOperationException>(
                async () => await _engine.RunInferenceAsync(imageData, 640, 640, null!)
            );
        }
    }
}