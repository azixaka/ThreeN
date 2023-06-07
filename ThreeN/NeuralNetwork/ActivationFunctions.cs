namespace ThreeN.NeuralNetwork;

public static class ActivationFunctions
{
    private static readonly float ReluParam = 0.01f;

    public static float Activate(float x, ActivationFunctionType type)
        => type switch
        {
            ActivationFunctionType.Sigmoid => Sigmoid(x),
            ActivationFunctionType.Relu => Relu(x),
            ActivationFunctionType.Tahn => Tahn(x),
            ActivationFunctionType.Sin => Sin(x),
            ActivationFunctionType.PassThrough => PassThrough(x),
            _ => throw new ArgumentException("Invalid activation function type", nameof(type))
        };

    public static float Derivative(float y, ActivationFunctionType type)
        => type switch
        {
            ActivationFunctionType.Sigmoid => SigmoidDerivative(y),
            ActivationFunctionType.Relu => ReluDerivative(y),
            ActivationFunctionType.Tahn => TahnDerivative(y),
            ActivationFunctionType.Sin => SinDerivative(y),
            ActivationFunctionType.PassThrough => PassThroughDerivative(y),
            _ => throw new ArgumentException("Invalid activation function type", nameof(type))
        };

    public static float Sigmoid(float x) 
        => 1f / (1f + (float)Math.Exp(-x));

    public static float SigmoidDerivative(float y)
        => y * (1f - y);

    public static float Relu(float x)
        => x > 0 ? x : x * ReluParam;

    public static float ReluDerivative(float y) =>
        y >= 0 ? 1 : ReluParam;
    
    public static float Tahn(float x)
    {
        var exp = (float)Math.Exp(x);
        var negExp = (float)Math.Exp(-x);
        return (exp - negExp) / (exp + negExp);
    }

    public static float TahnDerivative(float y)
        => 1f - y * y;

    public static float Sin(float x) =>
        (float)Math.Sin(x);

    public static float SinDerivative(float y) =>
        (float)Math.Cos(Math.Asin(y));

    public static float PassThrough(float x) => x;

    public static float PassThroughDerivative(float y) => 1f;
}
