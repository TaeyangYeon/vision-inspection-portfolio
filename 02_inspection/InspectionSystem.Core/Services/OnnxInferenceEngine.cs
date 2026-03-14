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
        private readonly IImageProcessor _imageProcessor;
        private InferenceSession? _session;
        private bool _disposed = false;

        public bool IsModelLoaded => _session != null;

        public OnnxInferenceEngine(ILogger<OnnxInferenceEngine> logger, IImageProcessor imageProcessor)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _imageProcessor = imageProcessor ?? throw new ArgumentNullException(nameof(imageProcessor));
        }

        public Task LoadModelAsync(string modelPath, CancellationToken cancellationToken = default)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(modelPath))
                {
                    _logger.LogWarning("Model path is null or empty");
                    return Task.CompletedTask;
                }

                if (!System.IO.File.Exists(modelPath))
                {
                    _logger.LogWarning("Model file not found: {ModelPath}", modelPath);
                    return Task.CompletedTask;
                }

                _session?.Dispose();
                var options = new SessionOptions();
                options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                _session = new InferenceSession(modelPath, options);
                _logger.LogInformation("Model loaded: {ModelPath}", modelPath);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to load model: {ModelPath}", modelPath);
                _session?.Dispose();
                _session = null;
            }

            return Task.CompletedTask;
        }

        public Task<DetectionResult> RunInferenceAsync(
            byte[] imageData,
            int width,
            int height,
            InferenceOptions options,
            CancellationToken cancellationToken = default)
        {
            if (imageData == null)
                throw new ArgumentNullException(nameof(imageData));

            if (imageData.Length == 0)
                throw new ArgumentNullException(nameof(imageData), "Image data cannot be empty.");

            if (width <= 0)
                throw new ArgumentOutOfRangeException(nameof(width), "Width must be greater than zero.");

            if (height <= 0)
                throw new ArgumentOutOfRangeException(nameof(height), "Height must be greater than zero.");

            if (options == null)
                throw new ArgumentNullException(nameof(options));

            cancellationToken.ThrowIfCancellationRequested();

            if (_session == null)
                throw new InvalidOperationException("Model is not loaded. Call LoadModelAsync first.");

            var stopwatch = Stopwatch.StartNew();

            try
            {
                int imgSize = options.ImageSize;

                var processedImage = _imageProcessor.Preprocess(imageData, width, height, imgSize);
                float[] tensor = processedImage.Tensor;
                float scale = processedImage.Scale;

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
            catch (OnnxRuntimeException ex)
            {
                _logger.LogError(ex, "ONNX Runtime error during inference");
                throw new InvalidOperationException("ONNX Runtime error during inference", ex);
            }
            catch (Exception ex) when (!(ex is ArgumentNullException || ex is ArgumentOutOfRangeException || ex is InvalidOperationException || ex is OperationCanceledException))
            {
                _logger.LogError(ex, "Unexpected error during inference");
                throw new InvalidOperationException("Unexpected error during inference", ex);
            }
        }


        private List<Detection> ParseDetections(
            float[] output, int[] shape,
            InferenceOptions options,
            float scale, int origW, int origH)
        {
            int numClasses = shape[shape.Length - 2] - 4;
            int numPredictions = shape[shape.Length - 1];

            _logger.LogDebug($"Shape: [{string.Join(", ", shape)}], Classes: {numClasses}, Predictions: {numPredictions}");
            
            float maxConfFound = 0f;
            int totalAboveThreshold = 0;

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

                if (maxConf > maxConfFound)
                    maxConfFound = maxConf;

                if (maxConf >= options.ConfidenceThreshold)
                    totalAboveThreshold++;

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

            _logger.LogDebug($"Max confidence found: {maxConfFound:F6}, Above threshold ({options.ConfidenceThreshold}): {totalAboveThreshold}, Raw detections: {rawDetections.Count}");

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