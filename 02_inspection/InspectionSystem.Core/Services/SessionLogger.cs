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
            _logger.LogInformation("Session history cleared.");
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

            var dir = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
                Directory.CreateDirectory(dir);

            var sb = new StringBuilder();
            sb.AppendLine("Timestamp,ImagePath,IsNG,DefectCount,DefectTypes,InferenceTimeMs,Confidence");

            foreach (var record in snapshot)
            {
                sb.AppendLine(string.Join(",",
                    record.Timestamp.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture),
                    EscapeCsv(record.ImagePath),
                    record.IsNG ? "NG" : "OK",
                    record.DefectCount,
                    EscapeCsv(record.DefectTypes),
                    record.InferenceTimeMs.ToString("F2", CultureInfo.InvariantCulture),
                    record.Confidence.ToString("F4", CultureInfo.InvariantCulture)
                ));
            }

            await File.WriteAllTextAsync(filePath, sb.ToString(), Encoding.UTF8);
            _logger.LogInformation("Exported {Count} records to {Path}", snapshot.Count, filePath);
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

        private static string EscapeCsv(string value)
        {
            if (string.IsNullOrEmpty(value)) return string.Empty;
            if (value.Contains(',') || value.Contains('"') || value.Contains('\n'))
                return $"\"{value.Replace("\"", "\"\"")}\"";
            return value;
        }
    }
}