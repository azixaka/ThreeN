namespace ThreeN.Activations;

public static class Activations
{
    public static float Sigmoid(float x)
    {
        return 1f / (1f + (float)Math.Exp(-x));
    }
}
