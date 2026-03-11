using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using InspectionSystem.Core.Interfaces;
using InspectionSystem.Core.Services;

namespace InspectionSystem.UI.DependencyInjection
{
    public static class ServiceCollectionExtensions
    {
        public static IServiceCollection AddInspectionServices(
            this IServiceCollection services,
            string settingsPath = "appsettings.json")
        {
            services.AddLogging(builder =>
            {
                builder.AddConsole();
                builder.SetMinimumLevel(LogLevel.Debug);
            });

            services.AddSingleton<ISettingsService>(provider =>
            {
                var logger = provider.GetRequiredService<ILogger<SettingsService>>();
                return new SettingsService(logger, settingsPath);
            });

            services.AddSingleton<ISessionLogger>(provider =>
            {
                var logger = provider.GetRequiredService<ILogger<SessionLogger>>();
                return new SessionLogger(logger);
            });

            services.AddTransient<IInferenceEngine>(provider =>
            {
                var logger = provider.GetRequiredService<ILogger<OnnxInferenceEngine>>();
                return new OnnxInferenceEngine(logger);
            });

            services.AddTransient<IImageProcessor>(_ => new ImageProcessor());

            services.AddTransient<IGradCamService>(provider =>
            {
                var logger = provider.GetRequiredService<ILogger<GradCamService>>();
                var settings = provider.GetRequiredService<ISettingsService>();
                return new GradCamService(logger, settings.Current.GradCamApiUrl);
            });

            services.AddTransient<InspectionSystem.UI.ViewModels.InspectionViewModel>();
            services.AddTransient<InspectionSystem.UI.ViewModels.MainViewModel>();

            return services;
        }
    }
}