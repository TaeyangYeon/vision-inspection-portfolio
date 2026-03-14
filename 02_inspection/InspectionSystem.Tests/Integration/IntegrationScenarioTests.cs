using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using NUnit.Framework;
using InspectionSystem.Core.Interfaces;
using InspectionSystem.Core.Models;
using InspectionSystem.Core.Services;
using InspectionSystem.UI.DependencyInjection;

namespace InspectionSystem.Tests.Integration
{
    [TestFixture]
    public class IntegrationScenarioTests
    {
        private ServiceProvider _provider = null!;
        private string _tempDir = string.Empty;

        [SetUp]
        public void SetUp()
        {
            var services = new ServiceCollection();
            services.AddLogging(b => b.AddConsole().SetMinimumLevel(LogLevel.Warning));
            services.AddInspectionServices();

            _tempDir = Path.Combine(Path.GetTempPath(), $"integration_{Guid.NewGuid()}");
            Directory.CreateDirectory(_tempDir);

            _provider = services.BuildServiceProvider();
        }

        [TearDown]
        public void TearDown()
        {
            _provider.Dispose();
            if (Directory.Exists(_tempDir))
                Directory.Delete(_tempDir, recursive: true);
        }

        private byte[] CreateDummyRgbImage(int width = 100, int height = 100)
        {
            var data = new byte[width * height * 3];
            var rng = new Random(42);
            rng.NextBytes(data);
            return data;
        }

        // --- Scenario 1: SessionLogger full workflow ---

        [Test]
        public async Task Scenario_SessionLogger_LogAndExportCsv()
        {
            var logger = _provider.GetRequiredService<ISessionLogger>();

            logger.Log(new InspectionRecord
            {
                Timestamp = DateTime.Now,
                ImagePath = "test_ok.jpg",
                IsNG = false,
                DefectCount = 0,
                InferenceTimeMs = 30.0,
                Confidence = 0f,
            });

            logger.Log(new InspectionRecord
            {
                Timestamp = DateTime.Now,
                ImagePath = "test_ng.jpg",
                IsNG = true,
                DefectCount = 2,
                InferenceTimeMs = 35.0,
                Confidence = 0.85f,
            });

            var stats = logger.GetStats();
            Assert.That(stats.TotalInspected, Is.EqualTo(2));
            Assert.That(stats.NgCount, Is.EqualTo(1));
            Assert.That(stats.OkCount, Is.EqualTo(1));

            var csvPath = Path.Combine(_tempDir, "session.csv");
            await logger.ExportToCsvAsync(csvPath);

            Assert.That(File.Exists(csvPath), Is.True);
            var content = await File.ReadAllTextAsync(csvPath);
            Assert.That(content, Does.Contain("test_ok.jpg"));
            Assert.That(content, Does.Contain("test_ng.jpg"));
            Assert.That(content, Does.Contain("Timestamp"));
        }

        // --- Scenario 2: SettingsService save and reload ---

        [Test]
        public void Scenario_SettingsService_SaveAndReload()
        {
            var settings = _provider.GetRequiredService<ISettingsService>();

            var newSettings = new AppSettings
            {
                ConfidenceThreshold = 0.6f,
                IouThreshold = 0.3f,
                ModelPath = "models/custom.onnx",
                SavePath = _tempDir,
                AutoSaveNg = false,
                EnableGradCam = false,
            };

            settings.Save(newSettings);

            var loaded = settings.Current;
            Assert.That(loaded.ConfidenceThreshold,
                Is.EqualTo(0.6f).Within(0.001f));
            Assert.That(loaded.IouThreshold,
                Is.EqualTo(0.3f).Within(0.001f));
            Assert.That(loaded.ModelPath, Is.EqualTo("models/custom.onnx"));
            Assert.That(loaded.AutoSaveNg, Is.False);
        }

        // --- Scenario 3: NgImageSaver NG workflow ---

        [Test]
        public async Task Scenario_NgImageSaver_SavesNgSkipsOk()
        {
            var saver = _provider.GetRequiredService<NgImageSaver>();
            var image = CreateDummyRgbImage(100, 100);

            var ngResult = new DetectionResult
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

            var okResult = new DetectionResult
            {
                Detections = new System.Collections.Generic.List<Detection>(),
                InferenceTimeMs = 30.0,
                ImageWidth = 100,
                ImageHeight = 100,
            };

            await saver.SaveAsync(image, 100, 100, ngResult, _tempDir);
            await saver.SaveAsync(image, 100, 100, okResult, _tempDir);

            var files = Directory.GetFiles(_tempDir);
            Assert.That(files.Length, Is.EqualTo(1));
            Assert.That(Path.GetFileName(files[0]), Does.StartWith("NG_"));
        }

        // --- Scenario 4: ImageProcessor preprocess pipeline ---

        [Test]
        public void Scenario_ImageProcessor_PreprocessPipeline()
        {
            var processor = _provider.GetRequiredService<IImageProcessor>();
            var image = CreateDummyRgbImage(640, 480);

            var processed = processor.Preprocess(image, 640, 480);

            Assert.That(processed, Is.Not.Null);
            Assert.That(processed.Tensor, Has.Length.EqualTo(3 * 640 * 640));
            Assert.That(processed.Scale, Is.GreaterThan(0f));

            foreach (var v in processed.Tensor)
            {
                Assert.That(v, Is.GreaterThanOrEqualTo(0f));
                Assert.That(v, Is.LessThanOrEqualTo(1f));
            }
        }

        // --- Scenario 5: InferenceEngine not loaded guard ---

        [Test]
        public void Scenario_InferenceEngine_GuardsBeforeLoad()
        {
            var engine = _provider.GetRequiredService<IInferenceEngine>();
            Assert.That(engine.IsModelLoaded, Is.False);

            Assert.ThrowsAsync<InvalidOperationException>(async () =>
                await engine.RunInferenceAsync(
                    CreateDummyRgbImage(), 100, 100,
                    new InferenceOptions
                    {
                        ConfidenceThreshold = 0.25f,
                        IouThreshold = 0.45f,
                        ImageSize = 640,
                    })
            );
        }

        // --- Scenario 6: SessionLogger clear ---

        [Test]
        public void Scenario_SessionLogger_ClearResetsStats()
        {
            var logger = _provider.GetRequiredService<ISessionLogger>();

            logger.Log(new InspectionRecord
            {
                Timestamp = DateTime.Now,
                ImagePath = "test.jpg",
                IsNG = true,
                DefectCount = 1,
                InferenceTimeMs = 35.0,
                Confidence = 0.8f,
            });

            Assert.That(logger.GetStats().TotalInspected, Is.EqualTo(1));

            logger.Clear();

            Assert.That(logger.GetStats().TotalInspected, Is.EqualTo(0));
            Assert.That(logger.GetHistory().Count, Is.EqualTo(0));
        }

        // --- Scenario 7: CSV empty session export ---

        [Test]
        public async Task Scenario_SessionLogger_ExportEmptySession()
        {
            var logger = _provider.GetRequiredService<ISessionLogger>();
            var csvPath = Path.Combine(_tempDir, "empty.csv");

            await logger.ExportToCsvAsync(csvPath);

            Assert.That(File.Exists(csvPath), Is.True);
            var content = await File.ReadAllTextAsync(csvPath);
            Assert.That(content, Does.Contain("Timestamp"));
            var lines = content.Split('\n',
                StringSplitOptions.RemoveEmptyEntries);
            Assert.That(lines.Length, Is.EqualTo(1));
        }

        // --- Scenario 8: DrawDetections with multiple detections ---

        [Test]
        public void Scenario_ImageProcessor_DrawMultipleDetections()
        {
            var processor = _provider.GetRequiredService<IImageProcessor>();
            var image = CreateDummyRgbImage(640, 640);

            var result = new DetectionResult
            {
                Detections = new System.Collections.Generic.List<Detection>
                {
                    new Detection
                    {
                        ClassId = 0,
                        ClassName = "broken_large",
                        Confidence = 0.9f,
                        Box = new BoundingBox { X1=10, Y1=10, X2=100, Y2=100 }
                    },
                    new Detection
                    {
                        ClassId = 1,
                        ClassName = "broken_small",
                        Confidence = 0.75f,
                        Box = new BoundingBox { X1=200, Y1=200, X2=300, Y2=300 }
                    },
                    new Detection
                    {
                        ClassId = 2,
                        ClassName = "contamination",
                        Confidence = 0.6f,
                        Box = new BoundingBox { X1=400, Y1=400, X2=500, Y2=500 }
                    },
                },
                ImageWidth = 640,
                ImageHeight = 640,
            };

            var output = processor.DrawDetections(
                image, 640, 640, result,
                new DrawOptions
                {
                    ShowLabel = true,
                    ShowConfidence = true,
                    LineThickness = 2,
                    FontScale = 0.5,
                }
            );

            Assert.That(output, Is.Not.Null);
            Assert.That(output, Has.Length.EqualTo(640 * 640 * 3));
        }
    }
}