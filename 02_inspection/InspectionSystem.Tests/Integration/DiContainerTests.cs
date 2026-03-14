using System;
using Microsoft.Extensions.DependencyInjection;
using NUnit.Framework;
using InspectionSystem.Core.Interfaces;
using InspectionSystem.UI.DependencyInjection;

namespace InspectionSystem.Tests.Integration
{
    [TestFixture]
    public class DiContainerTests
    {
        private ServiceProvider _provider = null!;

        [SetUp]
        public void SetUp()
        {
            var services = new ServiceCollection();
            services.AddInspectionServices("test_settings.json");
            _provider = services.BuildServiceProvider();
        }

        [TearDown]
        public void TearDown()
        {
            _provider.Dispose();
        }

        [Test]
        public void Resolve_ISettingsService_ReturnsInstance()
        {
            var service = _provider.GetService<ISettingsService>();
            Assert.That(service, Is.Not.Null);
        }

        [Test]
        public void Resolve_ISessionLogger_ReturnsInstance()
        {
            var service = _provider.GetService<ISessionLogger>();
            Assert.That(service, Is.Not.Null);
        }

        [Test]
        public void Resolve_IInferenceEngine_ReturnsInstance()
        {
            var service = _provider.GetService<IInferenceEngine>();
            Assert.That(service, Is.Not.Null);
        }

        [Test]
        public void Resolve_IImageProcessor_ReturnsInstance()
        {
            var service = _provider.GetService<IImageProcessor>();
            Assert.That(service, Is.Not.Null);
        }

        [Test]
        public void Resolve_IGradCamService_ReturnsInstance()
        {
            var service = _provider.GetService<IGradCamService>();
            Assert.That(service, Is.Not.Null);
        }

        [Test]
        public void Resolve_ISettingsService_IsSingleton()
        {
            var a = _provider.GetService<ISettingsService>();
            var b = _provider.GetService<ISettingsService>();
            Assert.That(a, Is.SameAs(b));
        }

        [Test]
        public void Resolve_ISessionLogger_IsSingleton()
        {
            var a = _provider.GetService<ISessionLogger>();
            var b = _provider.GetService<ISessionLogger>();
            Assert.That(a, Is.SameAs(b));
        }

        [Test]
        public void Resolve_IInferenceEngine_IsTransient()
        {
            var a = _provider.GetService<IInferenceEngine>();
            var b = _provider.GetService<IInferenceEngine>();
            Assert.That(a, Is.Not.SameAs(b));
        }

        [Test]
        public void Container_ResolveBenchmarkViewModel_NotNull()
        {
            var vm = _provider.GetService<InspectionSystem.UI.ViewModels.BenchmarkViewModel>();
            Assert.That(vm, Is.Not.Null);
        }

        [Test]
        public void Container_ResolveNgImageSaver_NotNull()
        {
            var saver = _provider.GetService<InspectionSystem.Core.Services.NgImageSaver>();
            Assert.That(saver, Is.Not.Null);
        }

        [Test]
        public void Container_ResolveSettingsViewModel_NotNull()
        {
            var vm = _provider.GetService<InspectionSystem.UI.ViewModels.SettingsViewModel>();
            Assert.That(vm, Is.Not.Null);
        }

        [Test]
        public void Container_ResolveGradCamViewModel_NotNull()
        {
            var vm = _provider.GetService<InspectionSystem.UI.ViewModels.GradCamViewModel>();
            Assert.That(vm, Is.Not.Null);
        }

        [Test]
        public void Container_ResolveMainViewModel_AllDependenciesInjected()
        {
            var vm = _provider.GetService<InspectionSystem.UI.ViewModels.MainViewModel>();
            Assert.That(vm, Is.Not.Null);
        }
    }
}