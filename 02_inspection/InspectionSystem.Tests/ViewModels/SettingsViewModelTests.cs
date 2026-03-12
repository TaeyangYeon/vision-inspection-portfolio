using Moq;
using NUnit.Framework;
using InspectionSystem.Core.Interfaces;
using InspectionSystem.Core.Models;
using InspectionSystem.UI.ViewModels;

namespace InspectionSystem.Tests.ViewModels
{
    [TestFixture]
    public class SettingsViewModelTests
    {
        private Mock<ISettingsService> _settingsServiceMock = null!;
        private SettingsViewModel _viewModel = null!;

        [SetUp]
        public void SetUp()
        {
            _settingsServiceMock = new Mock<ISettingsService>();
            _settingsServiceMock
                .Setup(s => s.Current)
                .Returns(new AppSettings
                {
                    ModelPath = "models/best.onnx",
                    ConfidenceThreshold = 0.25f,
                    IouThreshold = 0.45f,
                    ImageSize = 640,
                    SavePath = "outputs/ng_images",
                    AutoSaveNg = true,
                    GradCamApiUrl = "http://localhost:8000",
                    EnableGradCam = true,
                });

            _viewModel = new SettingsViewModel(_settingsServiceMock.Object);
        }

        [Test]
        public void Constructor_LoadsSettingsFromService()
        {
            Assert.That(_viewModel.ModelPath, Is.EqualTo("models/best.onnx"));
            Assert.That(_viewModel.ConfidenceThreshold,
                Is.EqualTo(0.25f).Within(0.001f));
            Assert.That(_viewModel.IouThreshold,
                Is.EqualTo(0.45f).Within(0.001f));
            Assert.That(_viewModel.ImageSize, Is.EqualTo(640));
        }

        [Test]
        public void SaveSettingsCommand_CallsSaveOnService()
        {
            _viewModel.SaveSettingsCommand.Execute(null);

            _settingsServiceMock.Verify(
                s => s.Save(It.IsAny<AppSettings>()),
                Times.Once
            );
        }

        [Test]
        public void SaveSettingsCommand_UpdatesStatusMessage()
        {
            _viewModel.SaveSettingsCommand.Execute(null);
            Assert.That(_viewModel.StatusMessage, Is.Not.Empty);
        }

        [Test]
        public void SaveSettingsCommand_SavesCorrectValues()
        {
            _viewModel.ConfidenceThreshold = 0.6f;
            _viewModel.IouThreshold = 0.3f;
            _viewModel.ModelPath = "models/new_model.onnx";

            _viewModel.SaveSettingsCommand.Execute(null);

            _settingsServiceMock.Verify(s => s.Save(
                It.Is<AppSettings>(a =>
                    a.ConfidenceThreshold == 0.6f &&
                    a.IouThreshold == 0.3f &&
                    a.ModelPath == "models/new_model.onnx"
                )), Times.Once
            );
        }

        [Test]
        public void ResetDefaultsCommand_ResetsToDefaultValues()
        {
            _viewModel.ConfidenceThreshold = 0.9f;
            _viewModel.IouThreshold = 0.9f;

            _viewModel.ResetDefaultsCommand.Execute(null);

            var defaults = new AppSettings();
            Assert.That(_viewModel.ConfidenceThreshold,
                Is.EqualTo(defaults.ConfidenceThreshold).Within(0.001f));
            Assert.That(_viewModel.IouThreshold,
                Is.EqualTo(defaults.IouThreshold).Within(0.001f));
        }

        [Test]
        public void ResetDefaultsCommand_UpdatesStatusMessage()
        {
            _viewModel.ResetDefaultsCommand.Execute(null);
            Assert.That(_viewModel.StatusMessage, Is.Not.Empty);
        }

        [Test]
        public void ConfidenceThreshold_SetValue_PropertyChangedRaised()
        {
            bool raised = false;
            _viewModel.PropertyChanged += (s, e) =>
            {
                if (e.PropertyName == nameof(_viewModel.ConfidenceThreshold))
                    raised = true;
            };

            _viewModel.ConfidenceThreshold = 0.7f;
            Assert.That(raised, Is.True);
        }

        [Test]
        public void AutoSaveNg_DefaultValue_IsTrue()
        {
            Assert.That(_viewModel.AutoSaveNg, Is.True);
        }

        [Test]
        public void EnableGradCam_DefaultValue_IsTrue()
        {
            Assert.That(_viewModel.EnableGradCam, Is.True);
        }
    }
}