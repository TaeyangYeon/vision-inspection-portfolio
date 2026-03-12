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
    public class NgImageSaverTests
    {
        private NgImageSaver _saver = null!;
        private string _tempDir = string.Empty;

        [SetUp]
        public void SetUp()
        {
            _saver = new NgImageSaver(NullLogger<NgImageSaver>.Instance);
            _tempDir = Path.Combine(
                Path.GetTempPath(),
                $"ng_test_{Guid.NewGuid()}"
            );
        }

        [TearDown]
        public void TearDown()
        {
            if (Directory.Exists(_tempDir))
                Directory.Delete(_tempDir, recursive: true);
        }

        private byte[] CreateDummyJpeg()
        {
            return new byte[100 * 100 * 3];
        }

        private DetectionResult CreateNgResult()
        {
            return new DetectionResult
            {
                Detections = new System.Collections.Generic.List<Detection>
                {
                    new Detection
                    {
                        ClassId = 0,
                        ClassName = "broken_large",
                        Confidence = 0.9f,
                        Box = new BoundingBox { X1=10, Y1=10, X2=50, Y2=50 }
                    }
                },
                InferenceTimeMs = 35.0,
                ImageWidth = 100,
                ImageHeight = 100,
            };
        }

        private DetectionResult CreateOkResult()
        {
            return new DetectionResult
            {
                Detections = new System.Collections.Generic.List<Detection>(),
                InferenceTimeMs = 35.0,
                ImageWidth = 100,
                ImageHeight = 100,
            };
        }

        [Test]
        public void Constructor_NullLogger_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(
                () => new NgImageSaver(null!)
            );
        }

        [Test]
        public void SaveAsync_NullImageData_ThrowsArgumentNullException()
        {
            Assert.ThrowsAsync<ArgumentNullException>(
                async () => await _saver.SaveAsync(
                    null!, 100, 100, CreateNgResult(), _tempDir)
            );
        }

        [Test]
        public void SaveAsync_EmptyImageData_ThrowsArgumentNullException()
        {
            Assert.ThrowsAsync<ArgumentNullException>(
                async () => await _saver.SaveAsync(
                    Array.Empty<byte>(), 100, 100, CreateNgResult(), _tempDir)
            );
        }

        [Test]
        public void SaveAsync_NullSavePath_ThrowsArgumentNullException()
        {
            Assert.ThrowsAsync<ArgumentNullException>(
                async () => await _saver.SaveAsync(
                    CreateDummyJpeg(), 100, 100, CreateNgResult(), null!)
            );
        }

        [Test]
        public async Task SaveAsync_OkResult_DoesNotSaveFile()
        {
            await _saver.SaveAsync(
                CreateDummyJpeg(), 100, 100, CreateOkResult(), _tempDir);

            bool hasFiles = Directory.Exists(_tempDir) &&
                Directory.GetFiles(_tempDir).Length > 0;
            Assert.That(hasFiles, Is.False);
        }

        [Test]
        public async Task SaveAsync_NgResult_CreatesDirectory()
        {
            await _saver.SaveAsync(
                CreateDummyJpeg(), 100, 100, CreateNgResult(), _tempDir);

            Assert.That(Directory.Exists(_tempDir), Is.True);
        }

        [Test]
        public async Task SaveAsync_NgResult_CreatesFile()
        {
            await _saver.SaveAsync(
                CreateDummyJpeg(), 100, 100, CreateNgResult(), _tempDir);

            var files = Directory.GetFiles(_tempDir);
            Assert.That(files.Length, Is.EqualTo(1));
        }

        [Test]
        public async Task SaveAsync_NgResult_FileNameContainsNG()
        {
            await _saver.SaveAsync(
                CreateDummyJpeg(), 100, 100, CreateNgResult(), _tempDir);

            var files = Directory.GetFiles(_tempDir);
            Assert.That(Path.GetFileName(files[0]),
                Does.StartWith("NG_"));
        }

        [Test]
        public async Task SaveAsync_NgResult_FileNameContainsDefectType()
        {
            await _saver.SaveAsync(
                CreateDummyJpeg(), 100, 100, CreateNgResult(), _tempDir);

            var files = Directory.GetFiles(_tempDir);
            Assert.That(Path.GetFileName(files[0]),
                Does.Contain("broken_large"));
        }

        [Test]
        public async Task SaveAsync_MultipleNgResults_CreatesMultipleFiles()
        {
            await _saver.SaveAsync(
                CreateDummyJpeg(), 100, 100, CreateNgResult(), _tempDir);
            await Task.Delay(10);
            await _saver.SaveAsync(
                CreateDummyJpeg(), 100, 100, CreateNgResult(), _tempDir);

            var files = Directory.GetFiles(_tempDir);
            Assert.That(files.Length, Is.EqualTo(2));
        }
    }
}