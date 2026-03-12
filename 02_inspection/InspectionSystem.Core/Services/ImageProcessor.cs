using System;
using System.Collections.Generic;
using System.Linq;
using OpenCvSharp;
using InspectionSystem.Core.Interfaces;
using InspectionSystem.Core.Models;

namespace InspectionSystem.Core.Services
{
    public class ImageProcessor : IImageProcessor
    {
        public ProcessedImage Preprocess(byte[] imageData, int width, int height, int targetSize = 640)
        {
            if (imageData == null || imageData.Length == 0)
                throw new ArgumentNullException(nameof(imageData));

            using var src = Mat.FromPixelData(height, width, MatType.CV_8UC3, imageData);

            // BGR to RGB
            using var rgb = new Mat();
            Cv2.CvtColor(src, rgb, ColorConversionCodes.BGR2RGB);

            int h = rgb.Rows;
            int w = rgb.Cols;

            // letterbox scale
            float scale = (float)targetSize / Math.Max(h, w);
            int newH = (int)(h * scale);
            int newW = (int)(w * scale);

            // resize
            using var resized = new Mat();
            Cv2.Resize(rgb, resized, new Size(newW, newH));

            // pad with 114 (top-left placement, matching Python implementation)
            using var padded = new Mat(targetSize, targetSize, MatType.CV_8UC3,
                new Scalar(114, 114, 114));
            using var roi = new Mat(padded, new Rect(0, 0, newW, newH));
            resized.CopyTo(roi);

            // to float32 NCHW
            float[] tensor = new float[3 * targetSize * targetSize];
            var paddedData = new Vec3b[targetSize * targetSize];
            padded.GetArray(out paddedData);

            for (int y = 0; y < targetSize; y++)
            {
                for (int x = 0; x < targetSize; x++)
                {
                    int idx = y * targetSize + x;
                    var pixel = paddedData[idx];
                    // NCHW: channel * H * W
                    tensor[0 * targetSize * targetSize + idx] = pixel.Item0 / 255.0f; // R
                    tensor[1 * targetSize * targetSize + idx] = pixel.Item1 / 255.0f; // G
                    tensor[2 * targetSize * targetSize + idx] = pixel.Item2 / 255.0f; // B
                }
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
            byte[] imageData,
            int width,
            int height,
            DetectionResult result,
            DrawOptions options)
        {
            if (imageData == null)
                throw new ArgumentNullException(nameof(imageData));
            if (result == null)
                throw new ArgumentNullException(nameof(result));

            using var mat = Mat.FromPixelData(height, width, MatType.CV_8UC3, imageData);
            using var draw = mat.Clone();

            foreach (var det in result.Detections)
            {
                var box = det.Box;
                var rect = new Rect(
                    (int)box.X1, (int)box.Y1,
                    (int)(box.X2 - box.X1), (int)(box.Y2 - box.Y1)
                );

                var color = det.ClassId switch
                {
                    0 => new Scalar(80, 160, 255),
                    1 => new Scalar(80, 255, 160),
                    _ => new Scalar(255, 160, 80),
                };

                Cv2.Rectangle(draw, rect, color,
                    options.LineThickness > 0 ? (int)options.LineThickness : 2);

                if (options.ShowLabel)
                {
                    string label = options.ShowConfidence
                        ? $"{det.ClassName} {det.Confidence:F2}"
                        : det.ClassName;

                    int baseline = 0;
                    var textSize = Cv2.GetTextSize(label,
                        HersheyFonts.HersheySimplex,
                        options.FontScale > 0 ? options.FontScale : 0.5,
                        1, out baseline);

                    var labelOrigin = new Point(
                        (int)box.X1,
                        Math.Max((int)box.Y1 - 4, textSize.Height + 4)
                    );

                    Cv2.PutText(draw, label, labelOrigin,
                        HersheyFonts.HersheySimplex,
                        options.FontScale > 0 ? options.FontScale : 0.5,
                        color, 1);
                }
            }

            var outBytes = new byte[width * height * 3];
            draw.GetArray(out Vec3b[] pixels);
            for (int i = 0; i < pixels.Length; i++)
            {
                outBytes[i * 3 + 0] = pixels[i].Item0;
                outBytes[i * 3 + 1] = pixels[i].Item1;
                outBytes[i * 3 + 2] = pixels[i].Item2;
            }
            return outBytes;
        }

        public byte[] ResizeImage(byte[] imageData, int width, int height, int targetWidth, int targetHeight)
        {
            if (imageData == null)
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

        public ProcessedImage ResizeWithPad(byte[] imageData, int width, int height, int targetSize = 640)
        {
            if (imageData == null || imageData.Length == 0)
                throw new ArgumentNullException(nameof(imageData));

            using var src = Mat.FromPixelData(height, width, MatType.CV_8UC3, imageData);

            int h = src.Rows;
            int w = src.Cols;
            float scale = (float)targetSize / Math.Max(h, w);
            int newH = (int)(h * scale);
            int newW = (int)(w * scale);

            using var resized = new Mat();
            Cv2.Resize(src, resized, new Size(newW, newH));

            using var padded = new Mat(targetSize, targetSize, MatType.CV_8UC3,
                new Scalar(114, 114, 114));
            using var roi = new Mat(padded, new Rect(0, 0, newW, newH));
            resized.CopyTo(roi);

            padded.GetArray(out Vec3b[] pixels);
            byte[] data = new byte[targetSize * targetSize * 3];
            for (int i = 0; i < pixels.Length; i++)
            {
                data[i * 3 + 0] = pixels[i].Item0;
                data[i * 3 + 1] = pixels[i].Item1;
                data[i * 3 + 2] = pixels[i].Item2;
            }

            return new ProcessedImage
            {
                Tensor = data.Select(b => b / 255.0f).ToArray(),
                TensorWidth = targetSize,
                TensorHeight = targetSize,
                Scale = scale,
                OriginalWidth = width,
                OriginalHeight = height,
            };
        }
    }
}