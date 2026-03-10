using InspectionSystem.Core.Models;

namespace InspectionSystem.Core.Interfaces
{
    public interface ISettingsService
    {
        AppSettings Load();
        void Save(AppSettings settings);
        AppSettings Current { get; }
    }
}