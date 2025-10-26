using ThreeN.LinearAlgebra;
using Xunit;

namespace ThreeN.Tests;

public class MatrixEdgeCaseTests
{
    [Fact]
    public void Constructor_ZeroRows_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => new Matrix<float>(0, 2));
    }

    [Fact]
    public void Constructor_ZeroColumns_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => new Matrix<float>(2, 0));
    }

    [Fact]
    public void Constructor_NegativeRows_ThrowsOverflowException()
    {
        Assert.Throws<OverflowException>(() => new Matrix<float>(-1, 2));
    }

    [Fact]
    public void Constructor_NullData_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => new Matrix<float>(2, 2, null!));
    }

    [Fact]
    public void Constructor_StrideLessThanColumns_ThrowsArgumentException()
    {
        var data = new float[10];
        Assert.Throws<ArgumentException>(() => new Matrix<float>(2, 5, 0, 3, data));
    }

    [Fact]
    public void Constructor_DataTooSmall_ThrowsArgumentOutOfRangeException()
    {
        var data = new float[5];
        Assert.Throws<ArgumentOutOfRangeException>(() => new Matrix<float>(3, 3, data));
    }

    [Fact]
    public void Constructor_StartIndexOutOfBounds_ThrowsArgumentOutOfRangeException()
    {
        var data = new float[10];
        Assert.Throws<ArgumentOutOfRangeException>(() => new Matrix<float>(2, 2, startIndex: 10, stride: 2, data));
    }

    [Fact]
    public void Indexer_OutOfBoundsRow_ThrowsArgumentOutOfRangeException()
    {
        var matrix = new Matrix<float>(2, 2);
        Assert.Throws<ArgumentOutOfRangeException>(() => matrix[2, 0]);
    }

    [Fact]
    public void Indexer_OutOfBoundsColumn_ThrowsArgumentOutOfRangeException()
    {
        var matrix = new Matrix<float>(2, 2);
        Assert.Throws<ArgumentOutOfRangeException>(() => matrix[0, 2]);
    }

    [Fact]
    public void DotProduct_DimensionMismatch_ThrowsArgumentException()
    {
        var a = new Matrix<float>(2, 3);
        var b = new Matrix<float>(2, 2); // b.Rows != a.Columns
        var result = new Matrix<float>(2, 2);

        Assert.Throws<ArgumentException>(() => MatrixExtensions.DotProduct(result, a, b));
    }

    [Fact]
    public void DotProduct_WrongResultDimensions_ThrowsArgumentException()
    {
        var a = new Matrix<float>(2, 3);
        var b = new Matrix<float>(3, 2);
        var result = new Matrix<float>(3, 3); // Should be 2x2

        Assert.Throws<ArgumentException>(() => MatrixExtensions.DotProduct(result, a, b));
    }

    [Fact]
    public void Add_DimensionMismatch_ThrowsArgumentException()
    {
        var a = new Matrix<float>(2, 2);
        var b = new Matrix<float>(2, 3);

        Assert.Throws<ArgumentException>(() => a.Add(ref b));
    }

    [Fact]
    public void CopyRow_DimensionMismatch_ThrowsArgumentException()
    {
        var source = new Matrix<float>(3, 2);
        var dest = new Matrix<float>(1, 3); // Different column count

        Assert.Throws<ArgumentException>(() => source.CopyRow(dest, 0));
    }

    [Fact]
    public void CopyRow_RowOutOfBounds_ThrowsArgumentOutOfRangeException()
    {
        var source = new Matrix<float>(3, 2);
        var dest = new Matrix<float>(1, 2);

        Assert.Throws<ArgumentOutOfRangeException>(() => source.CopyRow(dest, 3));
    }

    [Fact]
    public void StridedMatrix_WithValidStride_Works()
    {
        var data = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        var matrix = new Matrix<float>(3, 2, startIndex: 1, stride: 4, data);

        // Should access: data[1], data[2], data[5], data[6], data[9], data[10]
        Assert.Equal(2, matrix[0, 0]);
        Assert.Equal(3, matrix[0, 1]);
        Assert.Equal(6, matrix[1, 0]);
        Assert.Equal(7, matrix[1, 1]);
        Assert.Equal(10, matrix[2, 0]);
        Assert.Equal(11, matrix[2, 1]);
    }

    [Fact]
    public void FromArrayStrided_CreatesCorrectView()
    {
        var data = new float[] { 1, 2, 3, 4, 5, 6 };
        var matrix = Matrix<float>.FromArrayStrided(2, 2, data, startIndex: 0, stride: 3);

        Assert.Equal(1, matrix[0, 0]);
        Assert.Equal(2, matrix[0, 1]);
        Assert.Equal(4, matrix[1, 0]);
        Assert.Equal(5, matrix[1, 1]);
    }

    [Fact]
    public void SingleElementMatrix_Works()
    {
        var matrix = new Matrix<float>(1, 1);
        matrix[0, 0] = 42f;

        Assert.Equal(42f, matrix[0, 0]);
        Assert.Equal(1, matrix.Rows);
        Assert.Equal(1, matrix.Columns);
    }
}
