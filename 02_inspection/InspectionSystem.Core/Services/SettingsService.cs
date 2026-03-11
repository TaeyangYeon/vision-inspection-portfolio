using System;
using System.IO;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using InspectionSystem.Core.Interfaces;
using InspectionSystem.Core.Models;

namespace InspectionSystem.Core.Services
{
    public class SettingsService : ISettingsService
    {
        private readonly ILogger<SettingsService> _logger;
        private readonly string _settingsPath;
        private AppSettings _current;

        private static readonly JsonSerializerOptions JsonOptions = new()
        {
            WriteIndented = true,
        };

        public AppSettings Current => _current;

        public SettingsService(ILogger<SettingsService> logger, string settingsPath = "appsettings.json")
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _settingsPath = settingsPath ?? throw new ArgumentNullException(nameof(settingsPath));
            _current = Load();
        }

        public AppSettings Load()
        {
            if (!File.Exists(_settingsPath))
            {
                _logger.LogInformation("Settings file not found. Using defaults: {Path}", _settingsPath);
                _current = new AppSettings();
                return _current;
            }

            try
            {
                string json = File.ReadAllText(_settingsPath);
                var settings = JsonSerializer.Deserialize<AppSettings>(json, JsonOptions);
                _current = settings ?? new AppSettings();
                _logger.LogInformation("Settings loaded from {Path}", _settingsPath);
                return _current;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to load settings from {Path}. Using defaults.", _settingsPath);
                _current = new AppSettings();
                return _current;
            }
        }

        public void Save(AppSettings settings)
        {
            if (settings == null)
                throw new ArgumentNullException(nameof(settings));

            var dir = Path.GetDirectoryName(_settingsPath);
            if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
                Directory.CreateDirectory(dir);

            string json = JsonSerializer.Serialize(settings, JsonOptions);
            File.WriteAllText(_settingsPath, json);
            _current = settings;
            _logger.LogInformation("Settings saved to {Path}", _settingsPath);
        }
    }
}