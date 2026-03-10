using System.Threading;
using System.Threading.Tasks;
using InspectionSystem.Core.Models;

namespace InspectionSystem.Core.Interfaces
{
    public interface IInferenceEngine
    {
        bool IsModelLoaded { get; }
        Task LoadModelAsync(string modelPath, CancellationToken cancellationToken = default);
        Task<DetectionResult> RunInferenceAsync(byte[] imageData, int width, int height, InferenceOptions options, CancellationToken cancellationToken = default);
        void Dispose();
    }
}