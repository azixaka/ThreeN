using System.Drawing;

namespace NeuralNetwork.HandWrittenNumbers;

public sealed class MnistSplitter
{
    private const int ImageMagicNumber = 2051;
    private const int LabelMagicNumber = 2049;

    public void SaveDataAsFiles(string imageFile, string labelFile, string imagesDirectory, string labelsFilePath)
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

        if (!Directory.Exists(imagesDirectory))
            Directory.CreateDirectory(imagesDirectory);

        for (int i = 0; i < numberOfImages; i++)
        {
            Bitmap image = new Bitmap(numberOfColumns, numberOfRows);
            for (int y = 0; y < numberOfRows; y++)
            {
                for (int x = 0; x < numberOfColumns; x++)
                {
                    byte pixel = imageBytes[16 + i * numberOfRows * numberOfColumns + y * numberOfColumns + x];
                    Color color = Color.FromArgb(pixel, pixel, pixel); // Grayscale
                    image.SetPixel(x, y, color);
                }
            }

            image.Save(Path.Combine(imagesDirectory, $"{i}.bmp"));
        }

        using (var writer = new StreamWriter(labelsFilePath))
        {
            writer.WriteLine("sequence,digit");
            for (int i = 0; i < numberOfImages; i++)
            {
                int label = labelBytes[8 + i];
                writer.WriteLine($"{i},{label}");
            }
        }
    }
}
