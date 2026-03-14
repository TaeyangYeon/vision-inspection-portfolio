using System;
using System.IO;
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
        private string _tempDir = string.Empty;

        [SetUp]
        public void SetUp()
        {
            _imageProcessor = new ImageProcessor();
            _engine = new OnnxInferenceEngine(
                NullLogger<OnnxInferenceEngine>.Instance,
                _imageProcessor
            );
            _tempDir = Path.Combine(Path.GetTempPath(), $"onnx_test_{Guid.NewGuid()}");
            Directory.CreateDirectory(_tempDir);
        }

        [TearDown]
        public void TearDown()
        {
            _engine.Dispose();
            if (Directory.Exists(_tempDir))
                Directory.Delete(_tempDir, recursive: true);
        }

        private byte[] CreateDummyImage(int width = 100, int height = 100)
        {
            var data = new byte[width * height * 3];
            new Random(42).NextBytes(data);
            return data;
        }

        private InferenceOptions DefaultOptions() => new InferenceOptions
        {
            ConfidenceThreshold = 0.25f,
            IouThreshold = 0.45f,
            ImageSize = 640,
        };

        private string CreateFakeOnnxFile()
        {
            var path = Path.Combine(_tempDir, "fake.onnx");
            File.WriteAllBytes(path, new byte[] { 0x08, 0x01 });
            return path;
        }

        // --- Constructor ---

        [Test]
        public void Constructor_NullImageProcessor_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() =>
                new OnnxInferenceEngine(
                    NullLogger<OnnxInferenceEngine>.Instance,
                    null!));
        }

        [Test]
        public void Constructor_NullLogger_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() =>
                new OnnxInferenceEngine(null!, _imageProcessor));
        }

        [Test]
        public void Constructor_ValidArgs_IsModelLoadedFalse()
        {
            Assert.That(_engine.IsModelLoaded, Is.False);
        }

        // --- LoadModelAsync ---

        [Test]
        public async Task LoadModelAsync_NullPath_IsModelLoadedFalse()
        {
            await _engine.LoadModelAsync(null!);
            Assert.That(_engine.IsModelLoaded, Is.False);
        }

        [Test]
        public async Task LoadModelAsync_EmptyPath_IsModelLoadedFalse()
        {
            await _engine.LoadModelAsync(string.Empty);
            Assert.That(_engine.IsModelLoaded, Is.False);
        }

        [Test]
        public async Task LoadModelAsync_NonExistentPath_IsModelLoadedFalse()
        {
            await _engine.LoadModelAsync("nonexistent/best.onnx");
            Assert.That(_engine.IsModelLoaded, Is.False);
        }

        [Test]
        public async Task LoadModelAsync_InvalidOnnxFile_IsModelLoadedFalse()
        {
            var fakePath = CreateFakeOnnxFile();
            await _engine.LoadModelAsync(fakePath);
            Assert.That(_engine.IsModelLoaded, Is.False);
        }

        [Test]
        public async Task LoadModelAsync_CalledTwiceWithInvalidPaths_IsModelLoadedFalse()
        {
            await _engine.LoadModelAsync("nonexistent1.onnx");
            await _engine.LoadModelAsync("nonexistent2.onnx");
            Assert.That(_engine.IsModelLoaded, Is.False);
        }

        [Test]
        public async Task LoadModelAsync_NonExistentPath_DoesNotThrow()
        {
            Assert.DoesNotThrowAsync(async () =>
                await _engine.LoadModelAsync("nonexistent/best.onnx"));
            Assert.That(_engine.IsModelLoaded, Is.False);
        }

        [Test]
        public async Task LoadModelAsync_InvalidFile_DoesNotThrow()
        {
            var fakePath = CreateFakeOnnxFile();
            Assert.DoesNotThrowAsync(async () =>
                await _engine.LoadModelAsync(fakePath));
            Assert.That(_engine.IsModelLoaded, Is.False);
        }

        // --- RunInferenceAsync argument validation ---

        [Test]
        public void RunInferenceAsync_ModelNotLoaded_ThrowsInvalidOperationException()
        {
            Assert.ThrowsAsync<InvalidOperationException>(async () =>
                await _engine.RunInferenceAsync(
                    CreateDummyImage(), 100, 100, DefaultOptions()));
        }

        [Test]
        public void RunInferenceAsync_NullImageData_ThrowsArgumentNullException()
        {
            Assert.ThrowsAsync<ArgumentNullException>(async () =>
                await _engine.RunInferenceAsync(
                    null!, 100, 100, DefaultOptions()));
        }

        [Test]
        public void RunInferenceAsync_EmptyImageData_ThrowsArgumentNullException()
        {
            Assert.ThrowsAsync<ArgumentNullException>(async () =>
                await _engine.RunInferenceAsync(
                    Array.Empty<byte>(), 100, 100, DefaultOptions()));
        }

        [Test]
        public void RunInferenceAsync_ZeroWidth_ThrowsArgumentOutOfRangeException()
        {
            Assert.ThrowsAsync<ArgumentOutOfRangeException>(async () =>
                await _engine.RunInferenceAsync(
                    CreateDummyImage(), 0, 100, DefaultOptions()));
        }

        [Test]
        public void RunInferenceAsync_NegativeWidth_ThrowsArgumentOutOfRangeException()
        {
            Assert.ThrowsAsync<ArgumentOutOfRangeException>(async () =>
                await _engine.RunInferenceAsync(
                    CreateDummyImage(), -1, 100, DefaultOptions()));
        }

        [Test]
        public void RunInferenceAsync_ZeroHeight_ThrowsArgumentOutOfRangeException()
        {
            Assert.ThrowsAsync<ArgumentOutOfRangeException>(async () =>
                await _engine.RunInferenceAsync(
                    CreateDummyImage(), 100, 0, DefaultOptions()));
        }

        [Test]
        public void RunInferenceAsync_NegativeHeight_ThrowsArgumentOutOfRangeException()
        {
            Assert.ThrowsAsync<ArgumentOutOfRangeException>(async () =>
                await _engine.RunInferenceAsync(
                    CreateDummyImage(), 100, -1, DefaultOptions()));
        }

        [Test]
        public void RunInferenceAsync_CancelledToken_ThrowsOperationCanceledException()
        {
            var cts = new CancellationTokenSource();
            cts.Cancel();

            Assert.ThrowsAsync<OperationCanceledException>(async () =>
                await _engine.RunInferenceAsync(
                    CreateDummyImage(), 100, 100, DefaultOptions(), cts.Token));
        }

        [Test]
        public void RunInferenceAsync_NullOptions_ThrowsArgumentNullException()
        {
            Assert.ThrowsAsync<ArgumentNullException>(async () =>
                await _engine.RunInferenceAsync(
                    CreateDummyImage(), 100, 100, null!));
        }

        // --- IsModelLoaded state transitions ---

        [Test]
        public async Task IsModelLoaded_AfterFailedLoad_RemainingFalse()
        {
            await _engine.LoadModelAsync("bad_path.onnx");
            Assert.That(_engine.IsModelLoaded, Is.False);
        }

        [Test]
        public void IsModelLoaded_AfterDispose_DoesNotThrow()
        {
            _engine.Dispose();
            Assert.DoesNotThrow(() => { var _ = _engine.IsModelLoaded; });
        }

        // --- Dispose ---

        [Test]
        public void Dispose_CalledTwice_DoesNotThrow()
        {
            Assert.DoesNotThrow(() =>
            {
                _engine.Dispose();
                _engine.Dispose();
            });
        }

        // --- InferenceOptions edge cases (validation via engine) ---

        [Test]
        public void RunInferenceAsync_NullOptions_ModelNotLoaded_ThrowsArgumentNullException()
        {
            Assert.ThrowsAsync<ArgumentNullException>(async () =>
                await _engine.RunInferenceAsync(
                    CreateDummyImage(), 100, 100, null!));
        }

        [Test]
        public void RunInferenceAsync_LargeImage_ThrowsArgumentNullBeforeModelCheck()
        {
            Assert.ThrowsAsync<ArgumentNullException>(async () =>
                await _engine.RunInferenceAsync(
                    null!, 4096, 4096, DefaultOptions()));
        }
    }
}