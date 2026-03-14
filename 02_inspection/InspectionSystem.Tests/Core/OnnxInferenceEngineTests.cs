using System;
using System.Threading;
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
        private ImageProcessor _imageProcessor = null!;

        [SetUp]
        public void SetUp()
        {
            _imageProcessor = new ImageProcessor();
            _engine = new OnnxInferenceEngine(
                NullLogger<OnnxInferenceEngine>.Instance,
                _imageProcessor
            );
        }

        [TearDown]
        public void TearDown()
        {
            _engine.Dispose();
        }

        private byte[] CreateDummyImage(int width = 100, int height = 100)
        {
            return new byte[width * height * 3];
        }

        private InferenceOptions DefaultOptions() => new InferenceOptions
        {
            ConfidenceThreshold = 0.25f,
            IouThreshold = 0.45f,
            ImageSize = 640,
        };

        // --- IsModelLoaded ---

        [Test]
        public void IsModelLoaded_BeforeLoad_ReturnsFalse()
        {
            Assert.That(_engine.IsModelLoaded, Is.False);
        }

        [Test]
        public void LoadModelAsync_InvalidPath_ThrowsFileNotFoundException()
        {
            Assert.ThrowsAsync<System.IO.FileNotFoundException>(async () =>
                await _engine.LoadModelAsync("nonexistent/best.onnx"));
        }

        // --- RunInferenceAsync edge cases ---

        [Test]
        public void RunInferenceAsync_ModelNotLoaded_ThrowsInvalidOperationException()
        {
            Assert.ThrowsAsync<InvalidOperationException>(async () =>
                await _engine.RunInferenceAsync(
                    CreateDummyImage(), 100, 100, DefaultOptions())
            );
        }

        [Test]
        public void RunInferenceAsync_NullImageData_ThrowsArgumentNullException()
        {
            Assert.ThrowsAsync<ArgumentNullException>(async () =>
                await _engine.RunInferenceAsync(
                    null!, 100, 100, DefaultOptions())
            );
        }

        [Test]
        public void RunInferenceAsync_EmptyImageData_ThrowsArgumentNullException()
        {
            Assert.ThrowsAsync<ArgumentNullException>(async () =>
                await _engine.RunInferenceAsync(
                    Array.Empty<byte>(), 100, 100, DefaultOptions())
            );
        }

        [Test]
        public void RunInferenceAsync_ZeroWidth_ThrowsArgumentOutOfRangeException()
        {
            Assert.ThrowsAsync<ArgumentOutOfRangeException>(async () =>
                await _engine.RunInferenceAsync(
                    CreateDummyImage(), 0, 100, DefaultOptions())
            );
        }

        [Test]
        public void RunInferenceAsync_ZeroHeight_ThrowsArgumentOutOfRangeException()
        {
            Assert.ThrowsAsync<ArgumentOutOfRangeException>(async () =>
                await _engine.RunInferenceAsync(
                    CreateDummyImage(), 100, 0, DefaultOptions())
            );
        }

        [Test]
        public void RunInferenceAsync_NegativeWidth_ThrowsArgumentOutOfRangeException()
        {
            Assert.ThrowsAsync<ArgumentOutOfRangeException>(async () =>
                await _engine.RunInferenceAsync(
                    CreateDummyImage(), -1, 100, DefaultOptions())
            );
        }

        [Test]
        public void RunInferenceAsync_CancelledToken_ThrowsOperationCanceledException()
        {
            var cts = new CancellationTokenSource();
            cts.Cancel();

            Assert.ThrowsAsync<OperationCanceledException>(async () =>
                await _engine.RunInferenceAsync(
                    CreateDummyImage(), 100, 100, DefaultOptions(), cts.Token)
            );
        }

        // --- LoadModelAsync ---

        [Test]
        public void LoadModelAsync_CalledTwice_ThrowsFileNotFoundException()
        {
            Assert.ThrowsAsync<System.IO.FileNotFoundException>(async () =>
                await _engine.LoadModelAsync("nonexistent1.onnx"));
            Assert.ThrowsAsync<System.IO.FileNotFoundException>(async () =>
                await _engine.LoadModelAsync("nonexistent2.onnx"));
        }

        [Test]
        public void Constructor_NullImageProcessor_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() =>
                new OnnxInferenceEngine(
                    NullLogger<OnnxInferenceEngine>.Instance,
                    null!)
            );
        }

        [Test]
        public void Constructor_NullLogger_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() =>
                new OnnxInferenceEngine(
                    null!,
                    _imageProcessor)
            );
        }
    }
}