using System;
using System.IO;
using Microsoft.Extensions.Logging.Abstractions;
using NUnit.Framework;
using InspectionSystem.Core.Models;
using InspectionSystem.Core.Services;

namespace InspectionSystem.Tests.Core
{
    [TestFixture]
    public class SettingsServiceTests
    {
        private string _tempPath = string.Empty;

        [SetUp]
        public void SetUp()
        {
            _tempPath = Path.Combine(
                Path.GetTempPath(),
                $"test_settings_{Guid.NewGuid()}.json"
            );
        }

        [TearDown]
        public void TearDown()
        {
            if (File.Exists(_tempPath))
                File.Delete(_tempPath);
        }

        [Test]
        public void Constructor_NullLogger_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(
                () => new SettingsService(null!)
            );
        }

        [Test]
        public void Load_NoFileExists_ReturnsDefaultSettings()
        {
            var service = new SettingsService(
                NullLogger<SettingsService>.Instance, _tempPath
            );
            var settings = service.Load();
            Assert.That(settings, Is.Not.Null);
            Assert.That(settings.ConfidenceThreshold, Is.EqualTo(0.25f).Within(0.001f));
            Assert.That(settings.IouThreshold, Is.EqualTo(0.45f).Within(0.001f));
            Assert.That(settings.ImageSize, Is.EqualTo(640));
        }

        [Test]
        public void Current_AfterInit_ReturnsDefaultSettings()
        {
            var service = new SettingsService(
                NullLogger<SettingsService>.Instance, _tempPath
            );
            Assert.That(service.Current, Is.Not.Null);
            Assert.That(service.Current.ImageSize, Is.EqualTo(640));
        }

        [Test]
        public void Save_NullSettings_ThrowsArgumentNullException()
        {
            var service = new SettingsService(
                NullLogger<SettingsService>.Instance, _tempPath
            );
            Assert.Throws<ArgumentNullException>(() => service.Save(null!));
        }

        [Test]
        public void Save_ValidSettings_CreatesFile()
        {
            var service = new SettingsService(
                NullLogger<SettingsService>.Instance, _tempPath
            );
            var settings = new AppSettings { ConfidenceThreshold = 0.5f };
            service.Save(settings);
            Assert.That(File.Exists(_tempPath), Is.True);
        }

        [Test]
        public void Save_ThenLoad_ReturnsSameSettings()
        {
            var service = new SettingsService(
                NullLogger<SettingsService>.Instance, _tempPath
            );
            var original = new AppSettings
            {
                ConfidenceThreshold = 0.6f,
                IouThreshold = 0.3f,
                ImageSize = 640,
                AutoSaveNg = false,
                Theme = "Light",
            };
            service.Save(original);

            var loaded = service.Load();
            Assert.That(loaded.ConfidenceThreshold,
                Is.EqualTo(0.6f).Within(0.001f));
            Assert.That(loaded.IouThreshold,
                Is.EqualTo(0.3f).Within(0.001f));
            Assert.That(loaded.AutoSaveNg, Is.False);
            Assert.That(loaded.Theme, Is.EqualTo("Light"));
        }

        [Test]
        public void Save_UpdatesCurrent()
        {
            var service = new SettingsService(
                NullLogger<SettingsService>.Instance, _tempPath
            );
            var newSettings = new AppSettings { ConfidenceThreshold = 0.9f };
            service.Save(newSettings);
            Assert.That(service.Current.ConfidenceThreshold,
                Is.EqualTo(0.9f).Within(0.001f));
        }

        [Test]
        public void Load_CorruptedFile_ReturnsDefaultSettings()
        {
            File.WriteAllText(_tempPath, "{ invalid json }}}");
            var service = new SettingsService(
                NullLogger<SettingsService>.Instance, _tempPath
            );
            var settings = service.Load();
            Assert.That(settings, Is.Not.Null);
            Assert.That(settings.ImageSize, Is.EqualTo(640));
        }
    }
}