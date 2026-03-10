using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using InspectionSystem.Core.Interfaces;
using InspectionSystem.Core.Models;

namespace InspectionSystem.Core.Services
{
    public class OnnxInferenceEngine : IInferenceEngine, IDisposable
    {
        private readonly ILogger<OnnxInferenceEngine> _logger;
        private InferenceSession? _session;
        private bool _disposed = false;

        public bool IsModelLoaded => _session != null;

        public OnnxInferenceEngine(ILogger<OnnxInferenceEngine> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        public Task LoadModelAsync(string modelPath, CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrWhiteSpace(modelPath))
                throw new ArgumentNullException(nameof(modelPath));

            if (!System.IO.File.Exists(modelPath))
                throw new System.IO.FileNotFoundException($"Model not found: {modelPath}");

            _session?.Dispose();
            var options = new SessionOptions();
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            _session = new InferenceSession(modelPath, options);
            _logger.LogInformation("Model loaded: {ModelPath}", modelPath);

            return Task.CompletedTask;
        }

        public Task<DetectionResult> RunInferenceAsync(
            byte[] imageData,
            int width,
            int height,
            InferenceOptions options,
            CancellationToken cancellationToken = default)
        {
            if (_session == null)
                throw new InvalidOperationException("Model is not loaded. Call LoadModelAsync first.");

            if (imageData == null || imageData.Length == 0)
                throw new ArgumentNullException(nameof(imageData));

            if (options == null)
                throw new ArgumentNullException(nameof(options));

            cancellationToken.ThrowIfCancellationRequested();

            var stopwatch = Stopwatch.StartNew();

            int imgSize = options.ImageSize;
            float scale = (float)imgSize / Math.Max(width, height);
            int newW = (int)(width * scale);
            int newH = (int)(height * scale);

            float[] tensor = PrepareTensor(imageData, width, height, imgSize, newW, newH);

            var inputName = _session.InputMetadata.Keys.First();
            var dimensions = new int[] { 1, 3, imgSize, imgSize };
            var inputTensor = new DenseTensor<float>(tensor, dimensions);
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            };

            using var outputs = _session.Run(inputs);
            var outputTensor = outputs.First().AsTensor<float>();
            var outputArray = outputTensor.ToArray();
            var outputShape = outputTensor.Dimensions.ToArray();

            stopwatch.Stop();

            var detections = ParseDetections(
                outputArray, outputShape, options, scale, width, height
            );

            var result = new DetectionResult
            {
                Detections = detections,
                InferenceTimeMs = stopwatch.Elapsed.TotalMilliseconds,
                ImageWidth = width,
                ImageHeight = height,
            };

            _logger.LogDebug(
                "Inference complete: {Count} detections in {Ms:F1}ms",
                detections.Count, result.InferenceTimeMs
            );

            return Task.FromResult(result);
        }

        private float[] PrepareTensor(
            byte[] imageData, int width, int height,
            int imgSize, int newW, int newH)
        {
            float[] tensor = new float[3 * imgSize * imgSize];
            float fillVal = 114f / 255f;
            for (int i = 0; i < tensor.Length; i++)
                tensor[i] = fillVal;

            int channels = imageData.Length / (width * height);
            float scaleX = (float)newW / width;
            float scaleY = (float)newH / height;

            for (int y = 0; y < newH; y++)
            {
                int srcY = (int)(y / scaleY);
                srcY = Math.Clamp(srcY, 0, height - 1);

                for (int x = 0; x < newW; x++)
                {
                    int srcX = (int)(x / scaleX);
                    srcX = Math.Clamp(srcX, 0, width - 1);

                    int srcIdx = (srcY * width + srcX) * channels;

                    float r = imageData[srcIdx] / 255f;
                    float g = channels > 1 ? imageData[srcIdx + 1] / 255f : r;
                    float b = channels > 2 ? imageData[srcIdx + 2] / 255f : r;

                    tensor[0 * imgSize * imgSize + y * imgSize + x] = r;
                    tensor[1 * imgSize * imgSize + y * imgSize + x] = g;
                    tensor[2 * imgSize * imgSize + y * imgSize + x] = b;
                }
            }

            return tensor;
        }

        private List<Detection> ParseDetections(
            float[] output, int[] shape,
            InferenceOptions options,
            float scale, int origW, int origH)
        {
            int numClasses = shape[shape.Length - 2] - 4;
            int numPredictions = shape[shape.Length - 1];

            var rawDetections = new List<(BoundingBox box, float conf, int cls)>();

            for (int i = 0; i < numPredictions; i++)
            {
                float cx = output[0 * numPredictions + i];
                float cy = output[1 * numPredictions + i];
                float bw = output[2 * numPredictions + i];
                float bh = output[3 * numPredictions + i];

                float maxConf = 0f;
                int maxCls = 0;
                for (int c = 0; c < numClasses; c++)
                {
                    float conf = output[(4 + c) * numPredictions + i];
                    if (conf > maxConf)
                    {
                        maxConf = conf;
                        maxCls = c;
                    }
                }

                if (maxConf < options.ConfidenceThreshold)
                    continue;

                int x1 = (int)Math.Clamp((cx - bw / 2) / scale, 0, origW);
                int y1 = (int)Math.Clamp((cy - bh / 2) / scale, 0, origH);
                int x2 = (int)Math.Clamp((cx + bw / 2) / scale, 0, origW);
                int y2 = (int)Math.Clamp((cy + bh / 2) / scale, 0, origH);

                rawDetections.Add((
                    new BoundingBox { X1 = x1, Y1 = y1, X2 = x2, Y2 = y2 },
                    maxConf, maxCls
                ));
            }

            return ApplyNms(rawDetections, options.IouThreshold);
        }

        private List<Detection> ApplyNms(
            List<(BoundingBox box, float conf, int cls)> detections,
            float iouThreshold)
        {
            var sorted = detections.OrderByDescending(d => d.conf).ToList();
            var kept = new List<Detection>();
            var suppressed = new bool[sorted.Count];

            for (int i = 0; i < sorted.Count; i++)
            {
                if (suppressed[i]) continue;

                kept.Add(new Detection
                {
                    ClassId = sorted[i].cls,
                    Confidence = sorted[i].conf,
                    Box = sorted[i].box,
                });

                for (int j = i + 1; j < sorted.Count; j++)
                {
                    if (suppressed[j]) continue;
                    if (sorted[i].cls != sorted[j].cls) continue;
                    if (ComputeIou(sorted[i].box, sorted[j].box) > iouThreshold)
                        suppressed[j] = true;
                }
            }

            return kept;
        }

        private float ComputeIou(BoundingBox a, BoundingBox b)
        {
            int interX1 = Math.Max(a.X1, b.X1);
            int interY1 = Math.Max(a.Y1, b.Y1);
            int interX2 = Math.Min(a.X2, b.X2);
            int interY2 = Math.Min(a.Y2, b.Y2);

            int interW = Math.Max(0, interX2 - interX1);
            int interH = Math.Max(0, interY2 - interY1);
            float intersection = interW * interH;

            float areaA = a.Width * a.Height;
            float areaB = b.Width * b.Height;
            float union = areaA + areaB - intersection;

            return union > 0 ? intersection / union : 0f;
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _session?.Dispose();
                _disposed = true;
            }
        }
    }
}