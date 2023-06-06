using System.Numerics;

namespace ThreeN.LinearAlgebra;

public static class MatrixExtensions
{
    public static void DotProduct<T>(in Matrix<T> destination, in Matrix<T> a, in Matrix<T> b) where T : INumber<T>
    {
        if (a.Columns != b.Rows)
        {
            // inner sizes must be equal
            // 1x(2 * 2)x3 - OK
            // 1x(3 * 2)x3 - NOT OK
            throw new ArgumentException($"{nameof(a)}'s columns count must be equal to {nameof(b)}'s rows count");
        }

        if (destination.Rows != a.Rows || destination.Columns != b.Columns)
        {
            // destination size must be outter size
            // 1]x2 * 2x[3 - OK if destination is 1x3
            throw new ArgumentException($"{nameof(destination)}'s rows count must be equal to {nameof(a)}'s rows count AND columns count must be equal to {nameof(b)}'s columns count");
        }

        var n = a.Columns;

        for (int i = 0; i < destination.Rows; i++)
            for (int j = 0; j < destination.Columns; j++)
            {
                destination.ElementAt(i, j) = default;

                for (int k = 0; k < n; k++)
                {
                    // i (k * k) j
                    destination.ElementAt(i, j) += a.ElementAt(i, k) * b.ElementAt(k, j);
                }
            }
    }

    public static void Randomise(ref Matrix<float> matrix, float low, float high)
    {
        for (int i = 0; i < matrix.Rows; i++)
            for (int j = 0; j < matrix.Columns; j++)
            {
                matrix.ElementAt(i, j) = Random.Shared.NextSingle() * (high - low) + low;
            }
    }

    public static void Fill<T>(ref Matrix<T> matrix, T value) where T : INumber<T>
    {
        for (int i = 0; i < matrix.Rows; i++)
            for (int j = 0; j < matrix.Columns; j++)
            {
                matrix.ElementAt(i, j) = value;
            }
    }

    public static void Activate(ref Matrix<float> matrix, Func<float, float> activation)
    {
        for (int i = 0; i < matrix.Rows; i++)
            for (int j = 0; j < matrix.Columns; j++)
            {
                matrix.ElementAt(i, j) = activation(matrix.ElementAt(i, j));
            }
    }   
}
