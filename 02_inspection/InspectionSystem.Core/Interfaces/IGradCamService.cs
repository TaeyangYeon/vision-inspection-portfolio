using System.Threading;
using System.Threading.Tasks;
using InspectionSystem.Core.Models;

namespace InspectionSystem.Core.Interfaces
{
    public interface IGradCamService
    {
        bool IsAvailable { get; }
        Task<GradCamResult> GenerateAsync(byte[] imageData, int width, int height, int targetClass, float alpha = 0.5f, CancellationToken cancellationToken = default);
    }
}