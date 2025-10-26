using System.Numerics;
using System.Runtime.CompilerServices;

namespace ThreeN.LinearAlgebra;

/// <summary>
/// A generic matrix structure supporting strided memory layout for efficient submatrix operations.
/// </summary>
/// <typeparam name="T">The numeric type of matrix elements (must implement INumber&lt;T&gt;).</typeparam>
/// <remarks>
/// Supports SIMD optimizations for float matrices when memory is contiguous (Stride == Columns).
/// Uses row-major memory layout with optional stride for submatrix views without copying.
/// </remarks>
public readonly struct Matrix<T> where T : INumber<T>
{
    /// <summary>The number of rows in the matrix.</summary>
    public readonly int Rows;

    /// <summary>The number of columns in the matrix.</summary>
    public readonly int Columns;

    /// <summary>The number of elements between consecutive rows in the underlying data array.</summary>
    public readonly int Stride;

    /// <summary>The underlying data array storing matrix elements in row-major order.</summary>
    public readonly T[] Data;

    /// <summary>The starting index in the data array where this matrix begins.</summary>
    public readonly int StartIndex;

    /// <summary>
    /// Creates a new matrix with contiguous memory layout.
    /// </summary>
    /// <param name="rows">Number of rows (must be &gt; 0).</param>
    /// <param name="columns">Number of columns (must be &gt; 0).</param>
    /// <exception cref="ArgumentException">If rows or columns are &lt;= 0.</exception>
    public Matrix(int rows, int columns)
        : this(rows, columns, 0, columns, new T[rows * columns]) { }

    /// <summary>
    /// Creates a new matrix with custom stride (for submatrix views).
    /// </summary>
    /// <param name="rows">Number of rows (must be &gt; 0).</param>
    /// <param name="columns">Number of columns (must be &gt; 0).</param>
    /// <param name="startIndex">Starting index in the data array.</param>
    /// <param name="stride">Number of elements between consecutive rows (must be &gt;= columns).</param>
    /// <exception cref="ArgumentException">If rows, columns, or stride are invalid.</exception>
    /// <exception cref="ArgumentOutOfRangeException">If startIndex is out of bounds.</exception>
    public Matrix(int rows, int columns, int startIndex, int stride)
        : this(rows, columns, startIndex, stride, new T[rows * stride]) { }

    /// <summary>
    /// Creates a matrix from existing data with contiguous layout.
    /// </summary>
    /// <param name="rows">Number of rows (must be &gt; 0).</param>
    /// <param name="columns">Number of columns (must be &gt; 0).</param>
    /// <param name="data">Existing data array containing matrix elements.</param>
    /// <exception cref="ArgumentNullException">If data is null.</exception>
    /// <exception cref="ArgumentException">If rows or columns are invalid.</exception>
    /// <exception cref="ArgumentOutOfRangeException">If data array is too small.</exception>
    public Matrix(int rows, int columns, T[] data)
        : this(rows, columns, 0, columns, data) { }

    /// <summary>
    /// Creates a matrix from existing data with custom stride (primary constructor).
    /// </summary>
    /// <param name="rows">Number of rows (must be &gt; 0).</param>
    /// <param name="columns">Number of columns (must be &gt; 0).</param>
    /// <param name="startIndex">Starting index in the data array.</param>
    /// <param name="stride">Number of elements between consecutive rows (must be &gt;= columns).</param>
    /// <param name="data">Existing data array containing matrix elements.</param>
    /// <exception cref="ArgumentNullException">If data is null.</exception>
    /// <exception cref="ArgumentException">If rows, columns, or stride are invalid.</exception>
    /// <exception cref="ArgumentOutOfRangeException">If startIndex or dimensions don't fit in data array.</exception>
    /// <remarks>
    /// This constructor validates that the matrix dimensions fit within the data array bounds.
    /// Allows creating submatrix views without copying data by adjusting startIndex and stride.
    /// </remarks>
    public Matrix(int rows, int columns, int startIndex, int stride, T[] data)
    {
        Rows = rows;
        Columns = columns;
        Stride = stride;
        Data = data ?? throw new ArgumentNullException(nameof(data));
        StartIndex = startIndex;

        if (rows < 1)
            throw new ArgumentException("Rows must be greater than 0", nameof(rows));

        if (columns < 1)
            throw new ArgumentException("Columns must be greater than 0", nameof(columns));

        if (stride < columns)
            throw new ArgumentException("Stride must be greater than or equal to the number of columns", nameof(stride));

        if (startIndex < 0 || startIndex >= data.Length)
            throw new ArgumentOutOfRangeException(nameof(startIndex),
                $"StartIndex must be within data array bounds [0, {data.Length})");

        // Validate that matrix dimensions fit within data array
        int requiredLength = startIndex + (rows - 1) * stride + columns;
        if (requiredLength > data.Length)
            throw new ArgumentOutOfRangeException(nameof(data),
                $"Matrix dimensions require {requiredLength} elements, but data array has only {data.Length}");
    }

    /// <summary>
    /// Gets or sets the element at the specified row and column (zero-indexed).
    /// </summary>
    /// <param name="row">The zero-based row index.</param>
    /// <param name="column">The zero-based column index.</param>
    /// <returns>A reference to the element at [row, column].</returns>
    /// <exception cref="ArgumentOutOfRangeException">If row or column are out of bounds.</exception>
    public ref T this[int row, int column] => ref ElementAt(row, column);

    /// <summary>
    /// Gets a reference to the element at the specified row and column.
    /// </summary>
    /// <param name="row">The zero-based row index.</param>
    /// <param name="column">The zero-based column index.</param>
    /// <returns>A reference to the element at [row, column].</returns>
    /// <exception cref="ArgumentOutOfRangeException">If row or column are out of bounds.</exception>
    /// <remarks>
    /// Returns a reference allowing in-place modification without copying.
    /// Accounts for stride when computing the element's position in the data array.
    /// </remarks>
    public ref T ElementAt(int row, int column)
    {
        if (row >= Rows || column >= Columns)
            throw new ArgumentOutOfRangeException();

        return ref Data[StartIndex + row * Stride + column];
    }

    /// <summary>
    /// Adds another matrix to this matrix element-wise (this = this + matrix).
    /// Uses SIMD vectorization for float matrices when memory is contiguous.
    /// </summary>
    /// <param name="matrix">The matrix to add.</param>
    /// <exception cref="ArgumentException">If matrix dimensions don't match.</exception>
    /// <remarks>
    /// SIMD path requires: T is float AND Stride == Columns (contiguous memory).
    /// Expected speedup: 3-5x for contiguous float matrices.
    /// </remarks>
    public void Add(ref Matrix<T> matrix)
    {
        if (Rows != matrix.Rows || Columns != matrix.Columns)
            throw new ArgumentException("Matrices dimensions are different");

        // SIMD path: only for float with contiguous memory
        if (typeof(T) == typeof(float) && Stride == Columns && matrix.Stride == matrix.Columns)
        {
            ref var thisRef = ref Unsafe.AsRef(in this);
            AddSIMD(
                ref Unsafe.As<Matrix<T>, Matrix<float>>(ref thisRef),
                ref Unsafe.As<Matrix<T>, Matrix<float>>(ref matrix)
            );
            return;
        }

        // Fallback: scalar for non-float or strided matrices
        for (int i = 0; i < matrix.Rows; i++)
            for (int j = 0; j < matrix.Columns; j++)
            {
                ElementAt(i, j) += matrix.ElementAt(i, j);
            }
    }

    /// <summary>
    /// SIMD-optimized element-wise addition for contiguous float matrices.
    /// </summary>
    private static void AddSIMD(ref Matrix<float> dest, ref Matrix<float> src)
    {
        // Get contiguous spans (validated by caller)
        var destSpan = dest.Data.AsSpan(dest.StartIndex, dest.Rows * dest.Columns);
        var srcSpan = src.Data.AsSpan(src.StartIndex, src.Rows * src.Columns);

        int vectorSize = Vector<float>.Count;
        int i = 0;

        // Vectorized loop - process multiple elements at once
        for (; i <= destSpan.Length - vectorSize; i += vectorSize)
        {
            var vDest = new Vector<float>(destSpan.Slice(i));
            var vSrc = new Vector<float>(srcSpan.Slice(i));
            (vDest + vSrc).CopyTo(destSpan.Slice(i));
        }

        // Scalar remainder for non-aligned sizes
        for (; i < destSpan.Length; i++)
            destSpan[i] += srcSpan[i];
    }

    /// <summary>
    /// Copies a row from this matrix to the first row of the destination matrix.
    /// </summary>
    /// <param name="destinationMatrix">The destination matrix (must have same number of columns).</param>
    /// <param name="row">The zero-based row index to copy from this matrix.</param>
    /// <exception cref="ArgumentException">If destination matrix has different number of columns.</exception>
    /// <exception cref="ArgumentOutOfRangeException">If row index is out of bounds.</exception>
    /// <remarks>
    /// Commonly used to load input data into a neural network's input layer.
    /// Copies to row 0 of the destination matrix regardless of which row is specified as source.
    /// </remarks>
    public void CopyRow(Matrix<T> destinationMatrix, int row)
    {
        if (destinationMatrix.Columns != Columns)
            throw new ArgumentException("Matrices dimensions are different");

        if (row >= Rows)
            throw new ArgumentOutOfRangeException(nameof(row));

        for (int i = 0; i < destinationMatrix.Columns; i++)
        {
            destinationMatrix.ElementAt(0, i) = ElementAt(row, i);
        }
    }
}
