using System;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using NUnit.Framework;
using InspectionSystem.Core.Services;

namespace InspectionSystem.Tests.Core
{
    [TestFixture]
    public class GradCamServiceTests
    {
        private GradCamService _service = null!;

        [SetUp]
        public void SetUp()
        {
            _service = new GradCamService(
                NullLogger<GradCamService>.Instance,
                "http://localhost:9999"
            );
        }

        [TearDown]
        public void TearDown()
        {
            _service.Dispose();
        }

        [Test]
        public void Constructor_NullLogger_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(
                () => new GradCamService(null!)
            );
        }

        [Test]
        public void Constructor_NullApiUrl_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(
                () => new GradCamService(NullLogger<GradCamService>.Instance, null!)
            );
        }

        [Test]
        public void IsAvailable_InitialState_ReturnsFalse()
        {
            Assert.That(_service.IsAvailable, Is.False);
        }

        [Test]
        public void GenerateAsync_NullImageData_ThrowsArgumentNullException()
        {
            Assert.ThrowsAsync<ArgumentNullException>(
                async () => await _service.GenerateAsync(null!, 640, 640, 0)
            );
        }

        [Test]
        public void GenerateAsync_EmptyImageData_ThrowsArgumentNullException()
        {
            Assert.ThrowsAsync<ArgumentNullException>(
                async () => await _service.GenerateAsync(Array.Empty<byte>(), 640, 640, 0)
            );
        }

        [Test]
        public async Task GenerateAsync_ApiNotAvailable_ReturnsFailureResult()
        {
            var imageData = new byte[640 * 640 * 3];
            var result = await _service.GenerateAsync(imageData, 640, 640, 0);

            Assert.That(result, Is.Not.Null);
            Assert.That(result.Success, Is.False);
            Assert.That(result.ErrorMessage, Is.Not.Empty);
        }

        [Test]
        public async Task GenerateAsync_ApiNotAvailable_SetsIsAvailableFalse()
        {
            var imageData = new byte[640 * 640 * 3];
            await _service.GenerateAsync(imageData, 640, 640, 0);
            Assert.That(_service.IsAvailable, Is.False);
        }
    }
}