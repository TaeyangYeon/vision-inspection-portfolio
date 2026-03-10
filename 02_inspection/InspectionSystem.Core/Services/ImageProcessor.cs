using System;
using System.Collections.Generic;
using InspectionSystem.Core.Interfaces;
using InspectionSystem.Core.Models;
using OpenCvSharp;

namespace InspectionSystem.Core.Services
{
    public class ImageProcessor : IImageProcessor
    {
        private static readonly Scalar[] ClassColors = new[]
        {
            new Scalar(50, 200, 50),
            new Scalar(255, 50, 50),
            new Scalar(50, 50, 255),
            new Scalar(255, 200, 50),
            new Scalar(200, 50, 255),
        };

        public ProcessedImage Preprocess(byte[] imageData, int width, int height, int targetSize = 640)
        {
            if (imageData == null || imageData.Length == 0)
                throw new ArgumentNullException(nameof(imageData));

            float scale = (float)targetSize / Math.Max(width, height);
            int newW = (int)(width * scale);
            int newH = (int)(height * scale);

            using var src = Mat.FromPixelData(height, width, MatType.CV_8UC3, imageData);
            using var resized = new Mat();
            Cv2.Resize(src, resized, new Size(newW, newH));

            using var padded = new Mat(targetSize, targetSize, MatType.CV_8UC3, new Scalar(114, 114, 114));
            var roi = new Rect(0, 0, newW, newH);
            resized.CopyTo(padded[roi]);

            using var floatMat = new Mat();
            padded.ConvertTo(floatMat, MatType.CV_32FC3, 1.0 / 255.0);

            var channels = floatMat.Split();
            float[] tensor = new float[3 * targetSize * targetSize];
            int planeSize = targetSize * targetSize;

            for (int c = 0; c < 3; c++)
            {
                channels[c].GetArray(out float[] plane);
                Array.Copy(plane, 0, tensor, c * planeSize, planeSize);
                channels[c].Dispose();
            }

            return new ProcessedImage
            {
                Tensor = tensor,
                TensorWidth = targetSize,
                TensorHeight = targetSize,
                Scale = scale,
                OriginalWidth = width,
                OriginalHeight = height,
            };
        }

        public byte[] DrawDetections(
            byte[] imageData, int width, int height,
            DetectionResult result, DrawOptions options)
        {
            if (imageData == null || imageData.Length == 0)
                throw new ArgumentNullException(nameof(imageData));

            if (result == null)
                throw new ArgumentNullException(nameof(result));

            using var mat = Mat.FromPixelData(height, width, MatType.CV_8UC3, imageData);
            var output = mat.Clone();

            foreach (var det in result.Detections)
            {
                var color = ClassColors[det.ClassId % ClassColors.Length];
                var rect = new Rect(
                    det.Box.X1, det.Box.Y1,
                    det.Box.Width, det.Box.Height
                );

                Cv2.Rectangle(output, rect, color, options.LineThickness);

                if (options.ShowLabel)
                {
                    string label = options.ShowConfidence
                        ? $"{det.ClassName}: {det.Confidence:F2}"
                        : det.ClassName;

                    var textSize = Cv2.GetTextSize(
                        label, HersheyFonts.HersheySimplex,
                        options.FontScale, 2, out int baseline
                    );

                    var textBg = new Rect(
                        det.Box.X1, det.Box.Y1 - textSize.Height - 8,
                        textSize.Width, textSize.Height + 8
                    );

                    if (textBg.Y >= 0)
                        Cv2.Rectangle(output, textBg, color, -1);

                    Cv2.PutText(
                        output, label,
                        new Point(det.Box.X1, det.Box.Y1 - 4),
                        HersheyFonts.HersheySimplex,
                        options.FontScale,
                        new Scalar(255, 255, 255), 2
                    );
                }
            }

            output.GetArray(out Vec3b[] pixels);
            byte[] resultData = new byte[width * height * 3];
            for (int i = 0; i < pixels.Length; i++)
            {
                resultData[i * 3 + 0] = pixels[i].Item0;
                resultData[i * 3 + 1] = pixels[i].Item1;
                resultData[i * 3 + 2] = pixels[i].Item2;
            }
            return resultData;
        }

        public byte[] ResizeImage(
            byte[] imageData, int width, int height,
            int targetWidth, int targetHeight)
        {
            if (imageData == null || imageData.Length == 0)
                throw new ArgumentNullException(nameof(imageData));

            using var src = Mat.FromPixelData(height, width, MatType.CV_8UC3, imageData);
            using var resized = new Mat();
            Cv2.Resize(src, resized, new Size(targetWidth, targetHeight));
            resized.GetArray(out Vec3b[] pixels);
            byte[] result = new byte[targetWidth * targetHeight * 3];
            for (int i = 0; i < pixels.Length; i++)
            {
                result[i * 3 + 0] = pixels[i].Item0;
                result[i * 3 + 1] = pixels[i].Item1;
                result[i * 3 + 2] = pixels[i].Item2;
            }
            return result;
        }
    }
}