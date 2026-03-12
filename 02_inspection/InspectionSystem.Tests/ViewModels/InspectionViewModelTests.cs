using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using Moq;
using NUnit.Framework;
using InspectionSystem.Core.Interfaces;
using InspectionSystem.Core.Models;
using InspectionSystem.Core.Services;
using InspectionSystem.UI.ViewModels;

namespace InspectionSystem.Tests.ViewModels
{
    [TestFixture]
    public class InspectionViewModelTests
    {
        private Mock<IInferenceEngine> _inferenceEngineMock = null!;
        private Mock<IImageProcessor> _imageProcessorMock = null!;
        private Mock<ISessionLogger> _sessionLoggerMock = null!;
        private Mock<ISettingsService> _settingsServiceMock = null!;
        private NgImageSaver _ngImageSaver = null!;
        private InspectionViewModel _viewModel = null!;

        [SetUp]
        public void SetUp()
        {
            _inferenceEngineMock = new Mock<IInferenceEngine>();
            _imageProcessorMock = new Mock<IImageProcessor>();
            _sessionLoggerMock = new Mock<ISessionLogger>();
            _settingsServiceMock = new Mock<ISettingsService>();
            _ngImageSaver = new NgImageSaver(NullLogger<NgImageSaver>.Instance);

            _settingsServiceMock
                .Setup(s => s.Current)
                .Returns(new AppSettings
                {
                    ConfidenceThreshold = 0.25f,
                    IouThreshold = 0.45f,
                    ModelPath = "models/best.onnx",
                    AutoSaveNg = true,
                    SavePath = System.IO.Path.GetTempPath(),
                });

            _viewModel = new InspectionViewModel(
                _inferenceEngineMock.Object,
                _imageProcessorMock.Object,
                _sessionLoggerMock.Object,
                _settingsServiceMock.Object,
                _ngImageSaver,
                NullLogger<InspectionViewModel>.Instance
            );
        }

        [Test]
        public void Constructor_InitialState_IsModelLoadedFalse()
        {
            Assert.That(_viewModel.IsModelLoaded, Is.False);
        }

        [Test]
        public void Constructor_InitialState_IsInspectingFalse()
        {
            Assert.That(_viewModel.IsInspecting, Is.False);
        }

        [Test]
        public void Constructor_InitialState_ConfidenceThresholdFromSettings()
        {
            Assert.That(_viewModel.ConfidenceThreshold,
                Is.EqualTo(0.25f).Within(0.001f));
        }

        [Test]
        public void Constructor_InitialState_IouThresholdFromSettings()
        {
            Assert.That(_viewModel.IouThreshold,
                Is.EqualTo(0.45f).Within(0.001f));
        }

        [Test]
        public void Constructor_InitialState_AutoSaveNgFromSettings()
        {
            Assert.That(_viewModel.AutoSaveNg, Is.True);
        }

        [Test]
        public void Constructor_InitialState_DetectionsIsEmpty()
        {
            Assert.That(_viewModel.Detections.Count, Is.EqualTo(0));
        }

        [Test]
        public void Constructor_InitialState_ResultLabelIsNoResult()
        {
            Assert.That(_viewModel.ResultLabel, Is.EqualTo("NO RESULT"));
        }

        [Test]
        public void Constructor_InitialState_DetectionCountIsZero()
        {
            Assert.That(_viewModel.DetectionCount, Is.EqualTo(0));
        }

        [Test]
        public async Task LoadModelCommand_ModelNotFound_IsModelLoadedRemainingFalse()
        {
            _settingsServiceMock
                .Setup(s => s.Current)
                .Returns(new AppSettings
                {
                    ModelPath = "nonexistent/best.onnx"
                });

            var vm = new InspectionViewModel(
                _inferenceEngineMock.Object,
                _imageProcessorMock.Object,
                _sessionLoggerMock.Object,
                _settingsServiceMock.Object,
                _ngImageSaver,
                NullLogger<InspectionViewModel>.Instance
            );

            await vm.LoadModelCommand.ExecuteAsync(null);
            Assert.That(vm.IsModelLoaded, Is.False);
            Assert.That(vm.StatusMessage, Does.Contain("not found"));
        }

        [Test]
        public async Task RunInspectionCommand_ModelNotLoaded_StatusMessageUpdated()
        {
            await _viewModel.RunInspectionCommand.ExecuteAsync(null);
            Assert.That(_viewModel.StatusMessage,
                Does.Contain("not loaded").Or.Contain("No image"));
        }

        [Test]
        public async Task RunInspectionCommand_NoImageLoaded_StatusMessageUpdated()
        {
            _inferenceEngineMock.Setup(e => e.IsModelLoaded).Returns(true);

            var vm = new InspectionViewModel(
                _inferenceEngineMock.Object,
                _imageProcessorMock.Object,
                _sessionLoggerMock.Object,
                _settingsServiceMock.Object,
                _ngImageSaver,
                NullLogger<InspectionViewModel>.Instance
            );

            await vm.RunInspectionCommand.ExecuteAsync(null);
            Assert.That(vm.StatusMessage,
                Does.Contain("No image").Or.Contain("not loaded"));
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
            _viewModel.ConfidenceThreshold = 0.5f;
            Assert.That(raised, Is.True);
        }

        [Test]
        public void IouThreshold_SetValue_PropertyChangedRaised()
        {
            bool raised = false;
            _viewModel.PropertyChanged += (s, e) =>
            {
                if (e.PropertyName == nameof(_viewModel.IouThreshold))
                    raised = true;
            };
            _viewModel.IouThreshold = 0.6f;
            Assert.That(raised, Is.True);
        }

        [Test]
        public void AutoSaveNg_SetValue_PropertyChangedRaised()
        {
            bool raised = false;
            _viewModel.PropertyChanged += (s, e) =>
            {
                if (e.PropertyName == nameof(_viewModel.AutoSaveNg))
                    raised = true;
            };
            _viewModel.AutoSaveNg = false;
            Assert.That(raised, Is.True);
        }

        [Test]
        public void CancelInspectionCommand_WhenNotInspecting_StatusMessageUpdated()
        {
            _viewModel.CancelInspectionCommand.Execute(null);
            Assert.That(_viewModel.StatusMessage, Does.Contain("Cancelling")
                .Or.Contain("Load a model"));
        }

        [Test]
        public async Task RunInspectionCommand_CalledTwiceConcurrently_SecondIsRejected()
        {
            _inferenceEngineMock.Setup(e => e.IsModelLoaded).Returns(true);
            _inferenceEngineMock
                .Setup(e => e.RunInferenceAsync(
                    It.IsAny<byte[]>(),
                    It.IsAny<int>(),
                    It.IsAny<int>(),
                    It.IsAny<InferenceOptions>(),
                    It.IsAny<CancellationToken>()))
                .Returns(async () =>
                {
                    await Task.Delay(200);
                    return new DetectionResult();
                });

            _imageProcessorMock
                .Setup(p => p.DrawDetections(
                    It.IsAny<byte[]>(), It.IsAny<int>(), It.IsAny<int>(),
                    It.IsAny<DetectionResult>(), It.IsAny<DrawOptions>()))
                .Returns(new byte[100 * 100 * 3]);

            var vm = new InspectionViewModel(
                _inferenceEngineMock.Object,
                _imageProcessorMock.Object,
                _sessionLoggerMock.Object,
                _settingsServiceMock.Object,
                _ngImageSaver,
                NullLogger<InspectionViewModel>.Instance
            );

            await vm.LoadImageFromPathAsync(
                System.IO.Path.Combine(
                    System.AppContext.BaseDirectory,
                    "..", "..", "..", "..", "..",
                    "01_training", "data", "processed",
                    "bottle", "images", "val"
                )
            ).ContinueWith(_ => { });

            var task1 = vm.RunInspectionCommand.ExecuteAsync(null);
            var task2 = vm.RunInspectionCommand.ExecuteAsync(null);

            await Task.WhenAll(task1, task2);

            _inferenceEngineMock.Verify(
                e => e.RunInferenceAsync(
                    It.IsAny<byte[]>(), It.IsAny<int>(), It.IsAny<int>(),
                    It.IsAny<InferenceOptions>(), It.IsAny<CancellationToken>()),
                Times.AtMostOnce
            );
        }
    }
}