using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using ThreeN.NeuralNetwork;

namespace ThreeN.LinearAlgebra;

public static class MatrixExtensions
{
    /// <summary>
    /// Computes matrix multiplication (C = A × B) with SIMD optimization for float matrices.
    /// Uses System.Numerics.Vector for SIMD parallelization when T is float.
    /// </summary>
    /// <typeparam name="T">The numeric type (float gets SIMD, others use scalar).</typeparam>
    /// <param name="destination">The result matrix C (must be A.Rows × B.Columns).</param>
    /// <param name="a">The left matrix A.</param>
    /// <param name="b">The right matrix B (must have A.Columns rows).</param>
    /// <remarks>
    /// Expected speedup: 5-10x for large matrices (>100×100).
    /// Zero allocations on hot path.
    /// </remarks>
    public static void DotProduct<T>(in Matrix<T> destination, in Matrix<T> a, in Matrix<T> b) where T : INumber<T>
    {
        if (a.Columns != b.Rows)
        {
            throw new ArgumentException($"{nameof(a)}'s columns count must be equal to {nameof(b)}'s rows count");
        }

        if (destination.Rows != a.Rows || destination.Columns != b.Columns)
        {
            throw new ArgumentException($"{nameof(destination)}'s rows count must be equal to {nameof(a)}'s rows count AND columns count must be equal to {nameof(b)}'s columns count");
        }

        // Use SIMD for float, fallback for others
        if (typeof(T) == typeof(float))
        {
            DotProductSIMD(
                Unsafe.As<Matrix<T>, Matrix<float>>(ref Unsafe.AsRef(in destination)),
                Unsafe.As<Matrix<T>, Matrix<float>>(ref Unsafe.AsRef(in a)),
                Unsafe.As<Matrix<T>, Matrix<float>>(ref Unsafe.AsRef(in b))
            );
            return;
        }

        // Fallback: scalar implementation for non-float types
        DotProductScalar(destination, a, b);
    }

    /// <summary>
    /// Scalar implementation of matrix multiplication (for testing/fallback).
    /// </summary>
    private static void DotProductScalar<T>(in Matrix<T> destination, in Matrix<T> a, in Matrix<T> b) where T : INumber<T>
    {
        var n = a.Columns;

        for (int i = 0; i < destination.Rows; i++)
            for (int j = 0; j < destination.Columns; j++)
            {
                destination[i, j] = T.Zero;

                for (int k = 0; k < n; k++)
                {
                    destination[i, j] += a[i, k] * b[k, j];
                }
            }
    }

    /// <summary>
    /// SIMD-optimized matrix multiplication for float matrices with adaptive strategy.
    /// Uses cache-friendly blocking for large matrices (>512×512).
    /// </summary>
    private static void DotProductSIMD(in Matrix<float> destination, in Matrix<float> a, in Matrix<float> b)
    {
        // Threshold for using cache blocking: matrices where blocking overhead is beneficial
        const int BLOCKING_THRESHOLD = 512;

        // For large matrices, use cache-blocked implementation
        if (a.Rows >= BLOCKING_THRESHOLD || a.Columns >= BLOCKING_THRESHOLD || b.Columns >= BLOCKING_THRESHOLD)
        {
            DotProductBlocked(destination, a, b);
        }
        else
        {
            // For small/medium matrices, use simple SIMD (ijk order)
            DotProductSIMD_IJK(destination, a, b);
        }
    }

    /// <summary>
    /// Simple SIMD matrix multiplication using ijk loop order.
    /// Good for small matrices where cache blocking overhead isn't beneficial.
    /// </summary>
    private static void DotProductSIMD_IJK(in Matrix<float> destination, in Matrix<float> a, in Matrix<float> b)
    {
        int n = a.Columns;
        int vectorSize = Vector<float>.Count;

        for (int i = 0; i < destination.Rows; i++)
        {
            for (int j = 0; j < destination.Columns; j++)
            {
                Vector<float> sum = Vector<float>.Zero;
                int k = 0;

                // Vectorized loop - process multiple elements at once
                for (; k <= n - vectorSize; k += vectorSize)
                {
                    // Load row from A (contiguous memory - efficient)
                    var va = new Vector<float>(a.Data, a.StartIndex + i * a.Stride + k);

                    // Load column from B (strided access - less efficient but necessary)
                    var vb = LoadColumnVector(b, k, j, vectorSize);

                    // Multiply and accumulate
                    sum += va * vb;
                }

                // Horizontal sum - reduce vector to scalar
                float result = 0f;
                for (int v = 0; v < vectorSize; v++)
                    result += sum[v];

                // Scalar remainder for non-aligned sizes
                for (; k < n; k++)
                    result += a[i, k] * b[k, j];

                destination[i, j] = result;
            }
        }
    }

    /// <summary>
    /// Cache-friendly matrix multiplication using blocking/tiling with SIMD optimization.
    /// Uses ikj loop order for sequential memory access and blocks that fit in L1 cache.
    /// Optimal for large matrices (>512×512). Expected 10-30x speedup vs ijk order.
    /// </summary>
    /// <remarks>
    /// Block size tuned for typical L1 cache (32KB): 64×64 floats = 16KB, leaves room for 3 blocks.
    /// Loop order (ikj): For each A[i,k], updates entire row of C with A[i,k] * B[k,:].
    /// This ensures B is accessed by rows (sequential) instead of columns (strided).
    /// </remarks>
    private static void DotProductBlocked(in Matrix<float> destination, in Matrix<float> a, in Matrix<float> b)
    {
        // Block size tuned for L1 cache (32KB typical)
        // 64×64 floats = 16KB, leaves room for 3 blocks (A, B, C)
        const int blockSize = 64;
        int vectorSize = Vector<float>.Count;

        // Initialize result to zero (required for accumulation in blocked algorithm)
        Fill(ref Unsafe.AsRef(in destination), 0f);

        // Outer loops: iterate over blocks
        for (int ii = 0; ii < destination.Rows; ii += blockSize)
        for (int kk = 0; kk < a.Columns; kk += blockSize)
        for (int jj = 0; jj < destination.Columns; jj += blockSize)
        {
            // Compute block boundaries (handle partial blocks at edges)
            int iMax = Math.Min(ii + blockSize, destination.Rows);
            int kMax = Math.Min(kk + blockSize, a.Columns);
            int jMax = Math.Min(jj + blockSize, destination.Columns);

            // Inner loops: multiply this block using ikj order (cache-friendly)
            for (int i = ii; i < iMax; i++)
            for (int k = kk; k < kMax; k++)
            {
                // Load A[i,k] once, reuse across entire row of B
                float aik = a[i, k];

                // Broadcast A[i,k] to all SIMD lanes
                var vAik = new Vector<float>(aik);

                int j = jj;

                // SIMD loop: process multiple columns of B at once
                // Accesses B[k,:] sequentially (cache-friendly!)
                for (; j <= jMax - vectorSize; j += vectorSize)
                {
                    // Load row k of B starting at column j (sequential access)
                    int bIndex = b.StartIndex + k * b.Stride + j;
                    var vB = new Vector<float>(b.Data, bIndex);

                    // Load current result row i of C starting at column j
                    int cIndex = destination.StartIndex + i * destination.Stride + j;
                    var vC = new Vector<float>(destination.Data, cIndex);

                    // C[i,j:j+vectorSize] += A[i,k] * B[k,j:j+vectorSize]
                    var vResult = vC + vAik * vB;
                    vResult.CopyTo(destination.Data, cIndex);
                }

                // Scalar remainder for non-aligned columns
                for (; j < jMax; j++)
                {
                    destination[i, j] += aik * b[k, j];
                }
            }
        }
    }

    /// <summary>
    /// Loads a column vector from matrix B for SIMD operations.
    /// Uses stackalloc for zero-heap-allocation.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> LoadColumnVector(in Matrix<float> matrix, int startRow, int col, int count)
    {
        Span<float> temp = stackalloc float[Vector<float>.Count];
        temp.Clear();

        for (int i = 0; i < count && startRow + i < matrix.Rows; i++)
            temp[i] = matrix[startRow + i, col];

        return new Vector<float>(temp);
    }

    /// <summary>
    /// Fills a matrix with random values uniformly distributed in the range [low, high).
    /// </summary>
    /// <param name="matrix">The matrix to randomize.</param>
    /// <param name="low">The inclusive lower bound of the random range.</param>
    /// <param name="high">The exclusive upper bound of the random range.</param>
    /// <remarks>
    /// Uses Random.Shared for thread-safe random number generation.
    /// Commonly used for weight initialization in neural networks.
    /// </remarks>
    public static void Randomise(ref Matrix<float> matrix, float low, float high)
    {
        for (int i = 0; i < matrix.Rows; i++)
            for (int j = 0; j < matrix.Columns; j++)
            {
                matrix[i, j] = Random.Shared.NextSingle() * (high - low) + low;
            }
    }

    /// <summary>
    /// Fills a matrix with a constant value using SIMD optimization for float matrices.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="matrix">The matrix to fill.</param>
    /// <param name="value">The value to fill with.</param>
    /// <remarks>
    /// Uses SIMD for contiguous float matrices. Expected speedup: 4-6x.
    /// </remarks>
    public static void Fill<T>(ref Matrix<T> matrix, T value) where T : INumber<T>
    {
        // SIMD path for float with contiguous memory
        if (typeof(T) == typeof(float) && matrix.Stride == matrix.Columns)
        {
            FillSIMD(
                ref Unsafe.As<Matrix<T>, Matrix<float>>(ref matrix),
                Unsafe.As<T, float>(ref value)
            );
            return;
        }

        // Fallback: scalar
        for (int i = 0; i < matrix.Rows; i++)
            for (int j = 0; j < matrix.Columns; j++)
            {
                matrix[i, j] = value;
            }
    }

    /// <summary>
    /// SIMD-optimized fill for contiguous float matrices.
    /// </summary>
    private static void FillSIMD(ref Matrix<float> matrix, float value)
    {
        var span = matrix.Data.AsSpan(matrix.StartIndex, matrix.Rows * matrix.Columns);
        var vec = new Vector<float>(value); // Broadcast value to all lanes
        int vectorSize = Vector<float>.Count;
        int i = 0;

        // Vectorized loop
        for (; i <= span.Length - vectorSize; i += vectorSize)
            vec.CopyTo(span.Slice(i));

        // Scalar remainder
        for (; i < span.Length; i++)
            span[i] = value;
    }

    /// <summary>
    /// Applies an activation function to all elements of a matrix in-place.
    /// </summary>
    /// <param name="matrix">The matrix to apply activation to (modified in-place).</param>
    /// <param name="activationType">The activation function to apply.</param>
    /// <remarks>
    /// Softmax activation operates on entire rows (converts each row to probability distribution).
    /// Other activations (Sigmoid, ReLU, Tanh, etc.) are applied element-wise.
    /// For Softmax, uses numerically stable implementation with max subtraction.
    /// </remarks>
    public static void Activate(ref Matrix<float> matrix, Activation activationType)
    {
        // Special case: Softmax operates on entire rows, not individual elements
        if (activationType == Activation.Softmax)
        {
            for (int i = 0; i < matrix.Rows; i++)
            {
                // Find max for numerical stability
                float max = matrix[i, 0];
                for (int j = 1; j < matrix.Columns; j++)
                    if (matrix[i, j] > max) max = matrix[i, j];

                // Compute exp(x - max) and sum
                float sum = 0f;
                for (int j = 0; j < matrix.Columns; j++)
                {
                    matrix[i, j] = (float)Math.Exp(matrix[i, j] - max);
                    sum += matrix[i, j];
                }

                // Normalize
                for (int j = 0; j < matrix.Columns; j++)
                    matrix[i, j] /= sum;
            }
            return;
        }

        // Element-wise activations
        for (int i = 0; i < matrix.Rows; i++)
            for (int j = 0; j < matrix.Columns; j++)
            {
                matrix[i, j] = ActivationFunctions.Activate(matrix[i, j], activationType);
            }
    }   
}
