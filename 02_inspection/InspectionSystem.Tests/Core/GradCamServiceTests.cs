using System;
using System.Net;
using System.Net.Http;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using Moq;
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

        private byte[] CreateDummyImage(int width = 64, int height = 64)
        {
            var data = new byte[width * height * 3];
            new Random(42).NextBytes(data);
            return data;
        }

        // --- Constructor ---

        [Test]
        public void Constructor_NullLogger_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() =>
                new GradCamService(null!, "http://localhost:8000"));
        }

        [Test]
        public void Constructor_NullApiUrl_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() =>
                new GradCamService(
                    NullLogger<GradCamService>.Instance, null!));
        }

        [Test]
        public void Constructor_EmptyApiUrl_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() =>
                new GradCamService(
                    NullLogger<GradCamService>.Instance, string.Empty));
        }

        [Test]
        public void Constructor_ValidArgs_DoesNotThrow()
        {
            Assert.DoesNotThrow(() =>
            {
                using var svc = new GradCamService(
                    NullLogger<GradCamService>.Instance,
                    "http://localhost:8000");
            });
        }

        // --- GenerateAsync argument validation ---

        [Test]
        public void GenerateAsync_NullImageData_ThrowsArgumentNullException()
        {
            Assert.ThrowsAsync<ArgumentNullException>(async () =>
                await _service.GenerateAsync(null!, 64, 64, 0, 0.5f));
        }

        [Test]
        public void GenerateAsync_EmptyImageData_ThrowsArgumentNullException()
        {
            Assert.ThrowsAsync<ArgumentNullException>(async () =>
                await _service.GenerateAsync(
                    Array.Empty<byte>(), 64, 64, 0, 0.5f));
        }

        [Test]
        public void GenerateAsync_ZeroWidth_ThrowsArgumentOutOfRangeException()
        {
            Assert.ThrowsAsync<ArgumentOutOfRangeException>(async () =>
                await _service.GenerateAsync(
                    CreateDummyImage(), 0, 64, 0, 0.5f));
        }

        [Test]
        public void GenerateAsync_ZeroHeight_ThrowsArgumentOutOfRangeException()
        {
            Assert.ThrowsAsync<ArgumentOutOfRangeException>(async () =>
                await _service.GenerateAsync(
                    CreateDummyImage(), 64, 0, 0, 0.5f));
        }

        [Test]
        public void GenerateAsync_NegativeWidth_ThrowsArgumentOutOfRangeException()
        {
            Assert.ThrowsAsync<ArgumentOutOfRangeException>(async () =>
                await _service.GenerateAsync(
                    CreateDummyImage(), -1, 64, 0, 0.5f));
        }

        [Test]
        public void GenerateAsync_NegativeHeight_ThrowsArgumentOutOfRangeException()
        {
            Assert.ThrowsAsync<ArgumentOutOfRangeException>(async () =>
                await _service.GenerateAsync(
                    CreateDummyImage(), 64, -1, 0, 0.5f));
        }

        // --- GenerateAsync server unreachable ---

        [Test]
        public async Task GenerateAsync_ServerUnreachable_ReturnsFailureResult()
        {
            var result = await _service.GenerateAsync(
                CreateDummyImage(), 64, 64, 0, 0.5f);

            Assert.That(result.Success, Is.False);
            Assert.That(result.ErrorMessage, Is.Not.Null.And.Not.Empty);
        }

        [Test]
        public async Task GenerateAsync_ServerUnreachable_DoesNotThrow()
        {
            Assert.DoesNotThrowAsync(async () =>
                await _service.GenerateAsync(
                    CreateDummyImage(), 64, 64, 0, 0.5f));
        }

        // --- Alpha boundary ---

        [Test]
        public async Task GenerateAsync_AlphaZero_ServerUnreachable_ReturnsFailure()
        {
            var result = await _service.GenerateAsync(
                CreateDummyImage(), 64, 64, 0, 0.0f);
            Assert.That(result.Success, Is.False);
        }

        [Test]
        public async Task GenerateAsync_AlphaOne_ServerUnreachable_ReturnsFailure()
        {
            var result = await _service.GenerateAsync(
                CreateDummyImage(), 64, 64, 0, 1.0f);
            Assert.That(result.Success, Is.False);
        }

        // --- Dispose ---

        [Test]
        public void Dispose_CalledTwice_DoesNotThrow()
        {
            Assert.DoesNotThrow(() =>
            {
                _service.Dispose();
                _service.Dispose();
            });
        }
    }
}