using System.Collections.Generic;
using System.Threading.Tasks;
using InspectionSystem.Core.Models;

namespace InspectionSystem.Core.Interfaces
{
    public interface ISessionLogger
    {
        void Log(InspectionRecord record);
        IReadOnlyList<InspectionRecord> GetHistory();
        void Clear();
        Task ExportToCsvAsync(string filePath);
        SessionStats GetStats();
    }
}