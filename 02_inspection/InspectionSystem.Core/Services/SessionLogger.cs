using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using InspectionSystem.Core.Interfaces;
using InspectionSystem.Core.Models;

namespace InspectionSystem.Core.Services
{
    public class SessionLogger : ISessionLogger
    {
        private readonly ILogger<SessionLogger> _logger;
        private readonly List<InspectionRecord> _history = new();
        private readonly object _lock = new();

        public SessionLogger(ILogger<SessionLogger> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        public void Log(InspectionRecord record)
        {
            if (record == null)
                throw new ArgumentNullException(nameof(record));

            lock (_lock)
            {
                _history.Add(record);
            }

            _logger.LogDebug(
                "Logged record: IsNG={IsNG}, DefectCount={Count}, Time={Ms:F1}ms",
                record.IsNG, record.DefectCount, record.InferenceTimeMs
            );
        }

        public IReadOnlyList<InspectionRecord> GetHistory()
        {
            lock (_lock)
            {
                return _history.AsReadOnly();
            }
        }

        public void Clear()
        {
            lock (_lock)
            {
                _history.Clear();
            }
        }

        public async Task ExportToCsvAsync(string filePath)
        {
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentNullException(nameof(filePath));

            List<InspectionRecord> snapshot;
            lock (_lock)
            {
                snapshot = new List<InspectionRecord>(_history);
            }

            var sb = new System.Text.StringBuilder();
            sb.AppendLine("Timestamp,ImagePath,IsNG,DefectCount,InferenceTimeMs,Confidence");

            foreach (var r in snapshot)
            {
                sb.AppendLine(
                    $"{r.Timestamp:yyyy-MM-dd HH:mm:ss.fff}," +
                    $"{EscapeCsv(r.ImagePath)}," +
                    $"{r.IsNG}," +
                    $"{r.DefectCount}," +
                    $"{r.InferenceTimeMs:F2}," +
                    $"{r.Confidence:F4}"
                );
            }

            var dir = System.IO.Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(dir))
                System.IO.Directory.CreateDirectory(dir);

            await System.IO.File.WriteAllTextAsync(filePath, sb.ToString());
        }

        private static string EscapeCsv(string value)
        {
            if (string.IsNullOrEmpty(value)) return string.Empty;
            if (value.Contains(',') || value.Contains('"') || value.Contains('\n'))
                return $"\"{value.Replace("\"", "\"\"")}\"";
            return value;
        }

        public SessionStats GetStats()
        {
            lock (_lock)
            {
                int total = _history.Count;
                int ng = _history.Count(r => r.IsNG);
                int ok = total - ng;
                double avgMs = total > 0
                    ? _history.Average(r => r.InferenceTimeMs)
                    : 0;

                return new SessionStats
                {
                    TotalInspected = total,
                    OkCount = ok,
                    NgCount = ng,
                    AverageInferenceMs = avgMs,
                };
            }
        }
    }
}