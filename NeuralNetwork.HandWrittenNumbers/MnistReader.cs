using ThreeN.LinearAlgebra;

namespace NeuralNetwork.HandWrittenNumbers;

public sealed class MnistReader
{
    private const int ImageMagicNumber = 2051;
    private const int LabelMagicNumber = 2049;

    public (Matrix<float> Images, Matrix<float> Labels) LoadData(string imageFile, string labelFile)
    {
        byte[] imageBytes = File.ReadAllBytes(imageFile);
        byte[] labelBytes = File.ReadAllBytes(labelFile);

        if (BitConverter.ToInt32(imageBytes.Take(4).Reverse().ToArray()) != ImageMagicNumber)
            throw new Exception("Invalid image file format");

        if (BitConverter.ToInt32(labelBytes.Take(4).Reverse().ToArray()) != LabelMagicNumber)
            throw new Exception("Invalid label file format");

        int numberOfImages = BitConverter.ToInt32(imageBytes.Skip(4).Take(4).Reverse().ToArray());
        int numberOfRows = BitConverter.ToInt32(imageBytes.Skip(8).Take(4).Reverse().ToArray());
        int numberOfColumns = BitConverter.ToInt32(imageBytes.Skip(12).Take(4).Reverse().ToArray());

        var inData = new Matrix<float>(numberOfImages, numberOfRows * numberOfColumns);
        for (int i = 0; i < inData.Data.Length; i++)
        {
            inData.Data[i] = imageBytes[16 + i] / 255f;  // Normalize the pixel values to be between 0 and 1
        }

        var outData = new Matrix<float>(numberOfImages, 10);
        for (int i = 0; i < numberOfImages; i++)
        {
            outData[i, labelBytes[8 + i]] = 1.0f;  // One-hot encoding
        }

        return (inData, outData);
    }
}
