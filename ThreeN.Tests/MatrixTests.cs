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
}
