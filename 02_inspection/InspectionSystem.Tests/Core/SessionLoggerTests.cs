using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using NUnit.Framework;
using InspectionSystem.Core.Models;
using InspectionSystem.Core.Services;

namespace InspectionSystem.Tests.Core
{
    [TestFixture]
    public class SessionLoggerTests
    {
        private SessionLogger _logger = null!;

        [SetUp]
        public void SetUp()
        {
            _logger = new SessionLogger(NullLogger<SessionLogger>.Instance);
        }

        private InspectionRecord CreateRecord(bool isNg = true, double inferenceMs = 35.0)
        {
            return new InspectionRecord
            {
                Timestamp = DateTime.Now,
                ImagePath = "/test/image.jpg",
                IsNG = isNg,
                DefectCount = isNg ? 1 : 0,
                DefectTypes = isNg ? "broken_large" : string.Empty,
                InferenceTimeMs = inferenceMs,
                Confidence = 0.85f,
            };
        }

        [Test]
        public void Log_NullRecord_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() => _logger.Log(null!));
        }

        [Test]
        public void Log_ValidRecord_AddsToHistory()
        {
            var record = CreateRecord();
            _logger.Log(record);
            Assert.That(_logger.GetHistory().Count, Is.EqualTo(1));
        }

        [Test]
        public void Log_MultipleRecords_AllAddedToHistory()
        {
            _logger.Log(CreateRecord(isNg: true));
            _logger.Log(CreateRecord(isNg: false));
            _logger.Log(CreateRecord(isNg: true));
            Assert.That(_logger.GetHistory().Count, Is.EqualTo(3));
        }

        [Test]
        public void GetHistory_InitialState_ReturnsEmptyList()
        {
            Assert.That(_logger.GetHistory().Count, Is.EqualTo(0));
        }

        [Test]
        public void GetHistory_AfterLog_ReturnsCorrectRecord()
        {
            var record = CreateRecord(isNg: true);
            _logger.Log(record);
            var history = _logger.GetHistory();
            Assert.That(history[0].IsNG, Is.True);
            Assert.That(history[0].ImagePath, Is.EqualTo("/test/image.jpg"));
        }

        [Test]
        public void Clear_AfterLogging_EmptiesHistory()
        {
            _logger.Log(CreateRecord());
            _logger.Log(CreateRecord());
            _logger.Clear();
            Assert.That(_logger.GetHistory().Count, Is.EqualTo(0));
        }

        [Test]
        public void GetStats_EmptyHistory_ReturnsZeroStats()
        {
            var stats = _logger.GetStats();
            Assert.That(stats.TotalInspected, Is.EqualTo(0));
            Assert.That(stats.OkCount, Is.EqualTo(0));
            Assert.That(stats.NgCount, Is.EqualTo(0));
            Assert.That(stats.AverageInferenceMs, Is.EqualTo(0));
        }

        [Test]
        public void GetStats_MixedRecords_ReturnsCorrectCounts()
        {
            _logger.Log(CreateRecord(isNg: true, inferenceMs: 30.0));
            _logger.Log(CreateRecord(isNg: false, inferenceMs: 40.0));
            _logger.Log(CreateRecord(isNg: true, inferenceMs: 50.0));

            var stats = _logger.GetStats();
            Assert.That(stats.TotalInspected, Is.EqualTo(3));
            Assert.That(stats.NgCount, Is.EqualTo(2));
            Assert.That(stats.OkCount, Is.EqualTo(1));
            Assert.That(stats.AverageInferenceMs,
                Is.EqualTo(40.0).Within(0.001));
        }

        [Test]
        public void GetStats_NgRate_IsCalculatedCorrectly()
        {
            _logger.Log(CreateRecord(isNg: true));
            _logger.Log(CreateRecord(isNg: false));

            var stats = _logger.GetStats();
            Assert.That(stats.NgRate, Is.EqualTo(50.0).Within(0.001));
        }

        [Test]
        public void GetStats_EstimatedFps_IsCalculatedCorrectly()
        {
            _logger.Log(CreateRecord(inferenceMs: 40.0));
            var stats = _logger.GetStats();
            Assert.That(stats.EstimatedFps, Is.EqualTo(25.0).Within(0.001));
        }

        [Test]
        public async Task ExportToCsvAsync_ValidPath_CreatesFile()
        {
            _logger.Log(CreateRecord(isNg: true));
            _logger.Log(CreateRecord(isNg: false));

            string path = Path.Combine(Path.GetTempPath(), $"test_export_{Guid.NewGuid()}.csv");
            try
            {
                await _logger.ExportToCsvAsync(path);
                Assert.That(File.Exists(path), Is.True);
                string content = await File.ReadAllTextAsync(path);
                Assert.That(content, Does.Contain("Timestamp"));
                Assert.That(content, Does.Contain("True"));
                Assert.That(content, Does.Contain("False"));
            }
            finally
            {
                if (File.Exists(path)) File.Delete(path);
            }
        }

        [Test]
        public async Task ExportToCsvAsync_EmptyHistory_CreatesFileWithHeaderOnly()
        {
            string path = Path.Combine(Path.GetTempPath(), $"test_empty_{Guid.NewGuid()}.csv");
            try
            {
                await _logger.ExportToCsvAsync(path);
                Assert.That(File.Exists(path), Is.True);
                string content = await File.ReadAllTextAsync(path);
                string[] lines = content.Split('\n',
                    StringSplitOptions.RemoveEmptyEntries);
                Assert.That(lines.Length, Is.EqualTo(1));
            }
            finally
            {
                if (File.Exists(path)) File.Delete(path);
            }
        }

        [Test]
        public async Task ExportToCsvAsync_WithRecords_CreatesFile()
        {
            _logger.Log(new InspectionRecord
            {
                Timestamp = DateTime.Now,
                ImagePath = "test.jpg",
                IsNG = true,
                DefectCount = 1,
                InferenceTimeMs = 35.0,
                Confidence = 0.85f,
            });

            var csvPath = System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                $"test_{System.Guid.NewGuid()}.csv"
            );

            await _logger.ExportToCsvAsync(csvPath);

            Assert.That(System.IO.File.Exists(csvPath), Is.True);
            var content = await System.IO.File.ReadAllTextAsync(csvPath);
            Assert.That(content, Does.Contain("Timestamp"));
            Assert.That(content, Does.Contain("test.jpg"));

            System.IO.File.Delete(csvPath);
        }

        [Test]
        public void ExportToCsvAsync_NullPath_ThrowsArgumentNullException()
        {
            Assert.ThrowsAsync<ArgumentNullException>(
                async () => await _logger.ExportToCsvAsync(null!)
            );
        }
    }
}