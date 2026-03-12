using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InspectionSystem.Core.Interfaces;
using InspectionSystem.Core.Models;
using InspectionSystem.UI.Views;

namespace InspectionSystem.UI.ViewModels
{
    public partial class MainViewModel : ObservableObject
    {
        private readonly ISettingsService _settingsService;
        private readonly ISessionLogger _sessionLogger;
        private readonly InspectionViewModel _inspectionViewModel;
        private readonly GradCamViewModel _gradCamViewModel;
        private readonly SettingsViewModel _settingsViewModel;

        [ObservableProperty]
        private object? _currentPageView;

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
            ISessionLogger sessionLogger,
            InspectionViewModel inspectionViewModel,
            GradCamViewModel gradCamViewModel,
            SettingsViewModel settingsViewModel)
        {
            _settingsService = settingsService;
            _sessionLogger = sessionLogger;
            _inspectionViewModel = inspectionViewModel;
            _gradCamViewModel = gradCamViewModel;
            _settingsViewModel = settingsViewModel;
            RefreshStats();
            NavigateToPage("Inspection");
        }

        partial void OnSelectedPageChanged(string value)
        {
            NavigateToPage(value);
        }

        private void NavigateToPage(string page)
        {
            CurrentPageView = page switch
            {
                "Inspection" => new InspectionView { DataContext = _inspectionViewModel },
                "GradCAM" => new GradCamView { DataContext = _gradCamViewModel },
                "Settings" => new SettingsView { DataContext = _settingsViewModel },
                _ => null
            };
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