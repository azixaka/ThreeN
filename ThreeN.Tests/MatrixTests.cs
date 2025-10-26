using ThreeN.LinearAlgebra;
using Xunit;

namespace ThreeN.Tests;

public class MatrixTests
{
    [Fact]
    public void DotProduct_SimpleMultiplication_ReturnsCorrectResult()
    {
        // 2x2 * 2x2 = 2x2
        var a = new Matrix<float>(2, 2, new float[] { 1, 2, 3, 4 });
        var b = new Matrix<float>(2, 2, new float[] { 2, 0, 1, 2 });
        var result = new Matrix<float>(2, 2);

        MatrixExtensions.DotProduct(result, a, b);

        Assert.Equal(4, result[0, 0]);  // 1*2 + 2*1
        Assert.Equal(4, result[0, 1]);  // 1*0 + 2*2
        Assert.Equal(10, result[1, 0]); // 3*2 + 4*1
        Assert.Equal(8, result[1, 1]);  // 3*0 + 4*2
    }

    [Fact]
    public void DotProduct_NonSquareMatrices_ReturnsCorrectResult()
    {
        // 2x3 * 3x2 = 2x2
        var a = new Matrix<float>(2, 3, new float[] { 1, 2, 3, 4, 5, 6 });
        var b = new Matrix<float>(3, 2, new float[] { 1, 2, 3, 4, 5, 6 });
        var result = new Matrix<float>(2, 2);

        MatrixExtensions.DotProduct(result, a, b);

        Assert.Equal(22, result[0, 0]); // 1*1 + 2*3 + 3*5
        Assert.Equal(28, result[0, 1]); // 1*2 + 2*4 + 3*6
        Assert.Equal(49, result[1, 0]); // 4*1 + 5*3 + 6*5
        Assert.Equal(64, result[1, 1]); // 4*2 + 5*4 + 6*6
    }

    [Fact]
    public void Add_TwoMatrices_AddsElementWise()
    {
        var a = new Matrix<float>(2, 2, new float[] { 1, 2, 3, 4 });
        var b = new Matrix<float>(2, 2, new float[] { 5, 6, 7, 8 });

        a.Add(ref b);

        Assert.Equal(6, a[0, 0]);
        Assert.Equal(8, a[0, 1]);
        Assert.Equal(10, a[1, 0]);
        Assert.Equal(12, a[1, 1]);
    }

    [Fact]
    public void Fill_WithValue_FillsAllElements()
    {
        var matrix = new Matrix<float>(3, 2);

        MatrixExtensions.Fill(ref matrix, 5.5f);

        for (int i = 0; i < matrix.Rows; i++)
            for (int j = 0; j < matrix.Columns; j++)
                Assert.Equal(5.5f, matrix[i, j]);
    }

    [Fact]
    public void CopyRow_CopiesRowToDestination()
    {
        var source = new Matrix<float>(3, 2, new float[] { 1, 2, 3, 4, 5, 6 });
        var dest = new Matrix<float>(1, 2);

        source.CopyRow(dest, 1);

        Assert.Equal(3, dest[0, 0]);
        Assert.Equal(4, dest[0, 1]);
    }

    [Fact]
    public void Indexer_GetAndSet_WorksCorrectly()
    {
        var matrix = new Matrix<float>(2, 2);

        matrix[0, 0] = 1.5f;
        matrix[1, 1] = 2.5f;

        Assert.Equal(1.5f, matrix[0, 0]);
        Assert.Equal(2.5f, matrix[1, 1]);
    }

    [Fact]
    public void Randomise_FillsWithValuesInRange()
    {
        var matrix = new Matrix<float>(10, 10);

        MatrixExtensions.Randomise(ref matrix, -1f, 1f);

        // Verify all values are in range
        for (int i = 0; i < matrix.Rows; i++)
            for (int j = 0; j < matrix.Columns; j++)
            {
                Assert.True(matrix[i, j] >= -1f);
                Assert.True(matrix[i, j] < 1f);
            }
    }

    [Fact]
    public void StridedMatrix_AccessesCorrectElements()
    {
        var data = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        var matrix = new Matrix<float>(3, 2, startIndex: 0, stride: 3, data);

        Assert.Equal(1, matrix[0, 0]);
        Assert.Equal(2, matrix[0, 1]);
        Assert.Equal(4, matrix[1, 0]);
        Assert.Equal(5, matrix[1, 1]);
        Assert.Equal(7, matrix[2, 0]);
        Assert.Equal(8, matrix[2, 1]);
    }

    [Theory]
    [InlineData(10, 10, 10)]     // Small: uses SIMD_IJK path
    [InlineData(100, 100, 100)]  // Medium: uses SIMD_IJK path
    [InlineData(512, 512, 512)]  // Threshold: uses blocked path
    [InlineData(600, 600, 600)]  // Large: uses blocked path
    [InlineData(784, 784, 784)]  // MNIST size: uses blocked path
    public void DotProduct_CacheBlocked_MatchesScalar(int m, int n, int p)
    {
        // Arrange: Create random matrices A (m×n) and B (n×p)
        // Use seeded RNG for deterministic, non-flaky tests
        var a = new Matrix<float>(m, n);
        var b = new Matrix<float>(n, p);
        var resultOptimized = new Matrix<float>(m, p);
        var resultScalar = new Matrix<float>(m, p);

        RandomiseSeeded(ref a, -1f, 1f, seed: 12345);
        RandomiseSeeded(ref b, -1f, 1f, seed: 67890);

        // Act: Compute using both optimized (with blocking) and scalar paths
        MatrixExtensions.DotProduct(resultOptimized, a, b); // Uses adaptive dispatch
        DotProductScalarReference(resultScalar, a, b);      // Pure scalar for comparison

        // Assert: Results must match within floating-point precision
        // Different accumulation order causes tiny differences, so use absolute tolerance
        for (int i = 0; i < m; i++)
        for (int j = 0; j < p; j++)
        {
            float diff = Math.Abs(resultScalar[i, j] - resultOptimized[i, j]);
            Assert.True(diff < 0.001f,
                $"Position [{i},{j}]: Expected {resultScalar[i, j]}, Actual {resultOptimized[i, j]}, Diff {diff}");
        }
    }

    [Theory]
    [InlineData(13, 17, 19)]     // Non-aligned sizes (not multiples of vector size)
    [InlineData(65, 65, 65)]     // Just above block size (64)
    [InlineData(127, 129, 131)]  // Prime-ish sizes
    [InlineData(513, 513, 513)]  // Just above threshold
    public void DotProduct_NonAlignedSizes_MatchesScalar(int m, int n, int p)
    {
        // Test that non-aligned sizes (not multiples of vector size or block size) work correctly
        // Use seeded RNG for deterministic tests
        var a = new Matrix<float>(m, n);
        var b = new Matrix<float>(n, p);
        var resultOptimized = new Matrix<float>(m, p);
        var resultScalar = new Matrix<float>(m, p);

        RandomiseSeeded(ref a, -1f, 1f, seed: 11111);
        RandomiseSeeded(ref b, -1f, 1f, seed: 22222);

        MatrixExtensions.DotProduct(resultOptimized, a, b);
        DotProductScalarReference(resultScalar, a, b);

        // Use absolute tolerance for floating-point comparison
        for (int i = 0; i < m; i++)
        for (int j = 0; j < p; j++)
        {
            float diff = Math.Abs(resultScalar[i, j] - resultOptimized[i, j]);
            Assert.True(diff < 0.001f,
                $"Position [{i},{j}]: Expected {resultScalar[i, j]}, Actual {resultOptimized[i, j]}, Diff {diff}");
        }
    }

    [Theory]
    [InlineData(100, 784, 10)]   // Typical NN layer shape
    [InlineData(784, 128, 60000)] // MNIST-like (tall matrices)
    [InlineData(60000, 784, 10)]  // Batch × features
    public void DotProduct_RectangularMatrices_MatchesScalar(int m, int n, int p)
    {
        // Test rectangular (non-square) matrices common in neural networks
        // Use seeded RNG for deterministic tests
        var a = new Matrix<float>(m, n);
        var b = new Matrix<float>(n, p);
        var resultOptimized = new Matrix<float>(m, p);
        var resultScalar = new Matrix<float>(m, p);

        RandomiseSeeded(ref a, -1f, 1f, seed: 33333);
        RandomiseSeeded(ref b, -1f, 1f, seed: 44444);

        MatrixExtensions.DotProduct(resultOptimized, a, b);
        DotProductScalarReference(resultScalar, a, b);

        // Use absolute tolerance for floating-point comparison
        for (int i = 0; i < m; i++)
        for (int j = 0; j < p; j++)
        {
            float diff = Math.Abs(resultScalar[i, j] - resultOptimized[i, j]);
            Assert.True(diff < 0.001f,
                $"Position [{i},{j}]: Expected {resultScalar[i, j]}, Actual {resultOptimized[i, j]}, Diff {diff}");
        }
    }

    [Fact]
    public void DotProduct_LargeMatrix_CompletesWithoutError()
    {
        // Stress test: ensure large matrix multiplication doesn't crash or overflow
        var a = new Matrix<float>(1000, 1000);
        var b = new Matrix<float>(1000, 1000);
        var result = new Matrix<float>(1000, 1000);

        MatrixExtensions.Fill(ref a, 0.001f); // Small values to avoid overflow
        MatrixExtensions.Fill(ref b, 0.001f);

        // Should complete without throwing
        MatrixExtensions.DotProduct(result, a, b);

        // Verify result is reasonable
        // Each element = sum of 1000 products of 0.001 * 0.001 = 1000 * 0.000001 = 0.001
        Assert.Equal(0.001f, result[0, 0], precision: 3);
        Assert.Equal(0.001f, result[500, 500], precision: 3);
        Assert.Equal(0.001f, result[999, 999], precision: 3);
    }

    /// <summary>
    /// Reference scalar implementation for testing correctness.
    /// Pure ijk order with no optimizations.
    /// </summary>
    private static void DotProductScalarReference(Matrix<float> destination, Matrix<float> a, Matrix<float> b)
    {
        for (int i = 0; i < destination.Rows; i++)
        for (int j = 0; j < destination.Columns; j++)
        {
            float sum = 0f;
            for (int k = 0; k < a.Columns; k++)
            {
                sum += a[i, k] * b[k, j];
            }
            destination[i, j] = sum;
        }
    }

    /// <summary>
    /// Fills a matrix with random values using a seeded RNG for deterministic tests.
    /// </summary>
    private static void RandomiseSeeded(ref Matrix<float> matrix, float low, float high, int seed)
    {
        var rng = new Random(seed);
        for (int i = 0; i < matrix.Rows; i++)
        for (int j = 0; j < matrix.Columns; j++)
        {
            matrix[i, j] = rng.NextSingle() * (high - low) + low;
        }
    }
}
