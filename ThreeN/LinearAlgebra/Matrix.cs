using System.Numerics;

namespace ThreeN.LinearAlgebra;

public readonly struct Matrix<T> where T : INumber<T>
{
    public readonly int Rows;
    public readonly int Columns;
    public readonly int Stride;
    public readonly T[] Data;
    public readonly int StartIndex;

    public Matrix(int rows, int columns)
        : this(rows, columns, 0, columns, new T[rows * columns]) { }

    public Matrix(int rows, int columns, int startIndex, int stride)
        : this(rows, columns, startIndex, stride, new T[rows * stride]) { }

    public Matrix(int rows, int columns, T[] data)
        : this(rows, columns, 0, columns, data) { }

    public Matrix(int rows, int columns, int startIndex, int stride, T[] data)
    {
        Rows = rows;
        Columns = columns;
        Stride = stride;
        Data = data ?? throw new ArgumentNullException(nameof(data));
        StartIndex = startIndex;

        if (stride < columns)
            throw new ArgumentException("Stride must be greater than or equal to the number of columns", nameof(stride));

        if (rows < 1)
            throw new ArgumentException("Rows must be greater than 0", nameof(rows));

        if (columns < 1)
            throw new ArgumentException("Columns must be greater than 0", nameof(columns));

        var expectedLength = rows * columns;
        if (expectedLength > data.Length)
            throw new ArgumentOutOfRangeException($"The number of elements in the data array is not consistent with the specified number of rows and columns.");
    }

    public ref T this[int row, int column] => ref ElementAt(row, column);

    public ref T ElementAt(int row, int column)
    {
        if (row >= Rows || column >= Columns)
            throw new ArgumentOutOfRangeException();

        return ref Data[StartIndex + row * Stride + column];
    }

    public void Add(ref Matrix<T> matrix)
    {
        if (Rows != matrix.Rows || Columns != matrix.Columns)
            throw new ArgumentException("Matrices dimensions are different");

        for (int i = 0; i < matrix.Rows; i++)
            for (int j = 0; j < matrix.Columns; j++)
            {
                ElementAt(i, j) += matrix.ElementAt(i, j);
            }
    }

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
