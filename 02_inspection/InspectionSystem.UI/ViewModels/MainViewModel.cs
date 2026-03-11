using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InspectionSystem.Core.Interfaces;
using InspectionSystem.Core.Models;

namespace InspectionSystem.UI.ViewModels
{
    public partial class MainViewModel : ObservableObject
    {
        private readonly ISettingsService _settingsService;
        private readonly ISessionLogger _sessionLogger;

        [ObservableProperty]
        private string _selectedPage = "Inspection";

        [ObservableProperty]
        private string _appTitle = "Vision Inspection System";

        [ObservableProperty]
        private SessionStats _sessionStats = new();

        [ObservableProperty]
        private bool _isModelLoaded = false;

        [ObservableProperty]
        private string _modelStatus = "Model not loaded";

        public ObservableCollection<string> Pages { get; } = new()
        {
            "Inspection",
            "GradCAM",
            "Settings",
        };

        public MainViewModel(
            ISettingsService settingsService,
            ISessionLogger sessionLogger)
        {
            _settingsService = settingsService;
            _sessionLogger = sessionLogger;
            RefreshStats();
        }

        [RelayCommand]
        private void RefreshStats()
        {
            SessionStats = _sessionLogger.GetStats();
        }

        [RelayCommand]
        private void ClearSession()
        {
            _sessionLogger.Clear();
            RefreshStats();
        }

        public AppSettings CurrentSettings => _settingsService.Current;
    }
}