using ThreeN.LinearAlgebra;

namespace ThreeN.NeuralNetwork;

public static class NeuralNetworkSerialiser
{
    public static byte[] Serialise(NeuralNetwork nn)
    {
        using (var ms = new MemoryStream())
        using (var bw = new BinaryWriter(ms))
        {
            // Serialize matrices
            foreach (var matrixType in new[] { nn.Weights, nn.Biases, nn.Activations })
            {
                bw.Write(matrixType.Length);  // Write the count of matrices

                foreach (var matrix in matrixType)
                {
                    bw.Write(matrix.Rows);
                    bw.Write(matrix.Columns);

                    for (var r = 0; r < matrix.Rows; r++)
                    {
                        for (var c = 0; c < matrix.Columns; c++)
                        {
                            bw.Write(matrix[r, c]);
                        }
                    }
                }
            }

            // Serialize ActivationFunctions
            bw.Write(nn.ActivationFunctions.Length); // Write the count of ActivationFunctions
            foreach (var activationFunction in nn.ActivationFunctions)
            {
                bw.Write((byte)activationFunction);
            }

            bw.Flush();
            return ms.ToArray();
        }
    }

    public static NeuralNetwork Deserialise(byte[] bytes)
    {
        using (var ms = new MemoryStream(bytes))
        using (var br = new BinaryReader(ms))
        {
            var weights = new List<Matrix<float>>();
            var biases = new List<Matrix<float>>();
            var activations = new List<Matrix<float>>();

            foreach (var matrixList in new[] { weights, biases, activations })
            {
                int count = br.ReadInt32();  // Read the count of matrices
                for (int i = 0; i < count; i++)
                {
                    var rows = br.ReadInt32();
                    var cols = br.ReadInt32();

                    var data = new float[rows * cols];
                    for (var j = 0; j < data.Length; j++)
                    {
                        data[j] = br.ReadSingle();
                    }

                    matrixList.Add(new Matrix<float>(rows, cols, data));
                }
            }

            var activationFunctions = new List<Activation>();
            int afCount = br.ReadInt32();  // Read the count of ActivationFunctions
            for (int i = 0; i < afCount; i++)
            {
                activationFunctions.Add((Activation)br.ReadByte());
            }

            return new NeuralNetwork(
                weights.ToArray(),
                biases.ToArray(),
                activations.ToArray(),
                activationFunctions.ToArray());
        }
    }

}
