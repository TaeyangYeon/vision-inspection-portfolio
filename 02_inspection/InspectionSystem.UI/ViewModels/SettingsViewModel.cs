using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InspectionSystem.Core.Interfaces;
using InspectionSystem.Core.Models;

namespace InspectionSystem.UI.ViewModels
{
    public partial class SettingsViewModel : ObservableObject
    {
        private readonly ISettingsService _settingsService;

        [ObservableProperty]
        private string _modelPath = string.Empty;

        [ObservableProperty]
        private float _confidenceThreshold = 0.25f;

        [ObservableProperty]
        private float _iouThreshold = 0.45f;

        [ObservableProperty]
        private int _imageSize = 640;

        [ObservableProperty]
        private string _savePath = string.Empty;

        [ObservableProperty]
        private bool _autoSaveNg = true;

        [ObservableProperty]
        private string _gradCamApiUrl = string.Empty;

        [ObservableProperty]
        private bool _enableGradCam = true;

        [ObservableProperty]
        private string _statusMessage = string.Empty;

        public SettingsViewModel(ISettingsService settingsService)
        {
            _settingsService = settingsService;
            LoadFromSettings();
        }

        private void LoadFromSettings()
        {
            var s = _settingsService.Current;
            ModelPath = s.ModelPath;
            ConfidenceThreshold = s.ConfidenceThreshold;
            IouThreshold = s.IouThreshold;
            ImageSize = s.ImageSize;
            SavePath = s.SavePath;
            AutoSaveNg = s.AutoSaveNg;
            GradCamApiUrl = s.GradCamApiUrl;
            EnableGradCam = s.EnableGradCam;
        }

        [RelayCommand]
        private void SaveSettings()
        {
            var settings = new AppSettings
            {
                ModelPath = ModelPath,
                ConfidenceThreshold = ConfidenceThreshold,
                IouThreshold = IouThreshold,
                ImageSize = ImageSize,
                SavePath = SavePath,
                AutoSaveNg = AutoSaveNg,
                GradCamApiUrl = GradCamApiUrl,
                EnableGradCam = EnableGradCam,
            };
            _settingsService.Save(settings);
            StatusMessage = "Settings saved successfully.";
        }

        [RelayCommand]
        private void ResetDefaults()
        {
            var defaults = new AppSettings();
            ModelPath = defaults.ModelPath;
            ConfidenceThreshold = defaults.ConfidenceThreshold;
            IouThreshold = defaults.IouThreshold;
            ImageSize = defaults.ImageSize;
            SavePath = defaults.SavePath;
            AutoSaveNg = defaults.AutoSaveNg;
            GradCamApiUrl = defaults.GradCamApiUrl;
            EnableGradCam = defaults.EnableGradCam;
            StatusMessage = "Defaults restored. Click Save to apply.";
        }
    }
}