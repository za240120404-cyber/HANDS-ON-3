import java.util.Scanner;

public class Main { // CAMBIADO: Ahora coincide con el nombre de tu archivo Main.java

    public static void main(String[] args) {
        // 1. DATASET HARDCODED (17 Chemical Experiments)
        // Estructura: {1, x1, x2} -> El 1 es para el intercepto (Beta 0)
        double[][] xData = {
            {1, 41.9, 29.1}, {1, 43.4, 29.3}, {1, 43.9, 29.5}, {1, 44.5, 29.7},
            {1, 47.3, 29.9}, {1, 47.5, 30.3}, {1, 47.9, 30.5}, {1, 50.2, 30.7},
            {1, 52.8, 30.8}, {1, 53.2, 30.9}, {1, 56.7, 31.5}, {1, 57.0, 31.7},
            {1, 63.5, 31.9}, {1, 65.3, 32.0}, {1, 71.1, 32.1}, {1, 77.0, 32.5},
            {1, 77.8, 32.9}
        };

        double[] yData = {
            251.3, 251.3, 248.3, 267.5, 273.0, 276.5, 270.3, 274.9, 
            285.0, 290.0, 297.0, 302.5, 304.5, 309.3, 321.7, 330.7, 349.0
        };

        // 2. CÁLCULO MATRICIAL: Beta = (X'X)^-1 * X'Y
        double[] betas = calculateBetas(xData, yData);

        // 3. OUTPUT: Imprimir Ecuación de Regresión
        System.out.println("=== Hands-on 3: Least Squared Regressor (MLR) ===");
        System.out.printf("Ecuacion: y = %.4f + (%.4f * x1) + (%.4f * x2)\n", 
                          betas[0], betas[1], betas[2]);
        System.out.println("-------------------------------------------");

        // 4. SIMULACIÓN REQUERIDA (5 valores automáticos)
        double[][] simulationInputs = {
            {40.0, 30.0}, {50.0, 31.0}, {60.0, 32.0}, {70.0, 33.0}, {80.0, 34.0}
        };

        System.out.println("Simulacion de 5 experimentos:");
        for (int i = 0; i < simulationInputs.length; i++) {
            double pred = betas[0] + (betas[1] * simulationInputs[i][0]) + (betas[2] * simulationInputs[i][1]);
            System.out.printf("Sim %d: x1=%.1f, x2=%.1f -> Yield: %.4f\n", 
                              (i + 1), simulationInputs[i][0], simulationInputs[i][1], pred);
        }

        // 5. INYECCIÓN POR TERMINAL (Opcional pero recomendado por tus specs)
        Scanner sn = new Scanner(System.in);
        System.out.print("\n¿Quieres predecir un valor personalizado? (s/n): ");
        if (sn.next().equalsIgnoreCase("s")) {
            System.out.print("Ingresa x1 (Elemento quimico 1): ");
            double userX1 = sn.nextDouble();
            System.out.print("Ingresa x2 (Elemento quimico 2): ");
            double userX2 = sn.nextDouble();
            double userPred = betas[0] + (betas[1] * userX1) + (betas[2] * userX2);
            System.out.printf("Prediccion personalizada -> Yield: %.4f\n", userPred);
        }
    }

    // --- MÉTODOS DE LÓGICA MATRICIAL ---
    // (Utiliza la Ecuación Normal para resolver el sistema)

    public static double[] calculateBetas(double[][] X, double[] Y) {
        double[][] XT = transpose(X);
        double[][] XTX = multiply(XT, X);
        double[][] XTX_inv = invert(XTX);
        
        double[][] YMat = new double[Y.length][1];
        for (int i = 0; i < Y.length; i++) YMat[i][0] = Y[i];
        
        double[][] XTY = multiply(XT, YMat);
        double[][] result = multiply(XTX_inv, XTY);
        
        double[] betas = new double[result.length];
        for (int i = 0; i < result.length; i++) betas[i] = result[i][0];
        return betas;
    }

    public static double[][] multiply(double[][] A, double[][] B) {
        int rowsA = A.length;
        int colsA = A[0].length;
        int colsB = B[0].length;
        double[][] C = new double[rowsA][colsB];
        for (int i = 0; i < rowsA; i++)
            for (int j = 0; j < colsB; j++)
                for (int k = 0; k < colsA; k++)
                    C[i][j] += A[i][k] * B[k][j];
        return C;
    }

    public static double[][] transpose(double[][] matrix) {
        int r = matrix.length;
        int c = matrix[0].length;
        double[][] t = new double[c][r];
        for (int i = 0; i < r; i++)
            for (int j = 0; j < c; j++)
                t[j][i] = matrix[i][j];
        return t;
    }

    public static double[][] invert(double[][] matrix) {
        int n = matrix.length;
        double[][] augmented = new double[n][2 * n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(matrix[i], 0, augmented[i], 0, n);
            augmented[i][i + n] = 1;
        }
        for (int i = 0; i < n; i++) {
            double pivot = augmented[i][i];
            for (int j = 0; j < 2 * n; j++) augmented[i][j] /= pivot;
            for (int k = 0; k < n; k++) {
                if (k != i) {
                    double factor = augmented[k][i];
                    for (int j = 0; j < 2 * n; j++) augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
        double[][] inverse = new double[n][n];
        for (int i = 0; i < n; i++)
            System.arraycopy(augmented[i], n, inverse[i], 0, n);
        return inverse;
    }
}
