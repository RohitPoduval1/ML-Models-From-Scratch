"""
An implementation of Simple Linear Regression
"""
import numpy as np
import matplotlib.pyplot as plt


def compute_coefficients(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Computes the slope (B1) and intercept (B0) using the least squares formula
    
    Args:
        x (numpy.ndarray) A 1D array of values for the independent/input/explanatory variable.
        y (numpy.ndarray) A 1D array of values for the dependent/target/response variable.

    Returns:
        A 2-tuple of the intercept and slope which are both floats
    """
    if len(x) != len(y):
        raise ValueError("Vectors x and y must be the same length")

    n = len(y)
    B1 = (n * np.sum(x @ y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2)
    B0 = (np.sum(y) - B1*np.sum(x)) / n

    return (B0, B1)


def predict(x: float, B0: float, B1: float) -> float:
    """
    Predict y values given input x using the regression equation y = B0 + B1*x

    Args:
        x  (float) The input to make the prediction using
        B0 (float) The intercept of the model
        B1 (float) The slope of the model

    Returns:
        The predicted value based on the given parameters for the given x
    """
    return B0 + B1*x


def scatterplot(x, y):
    """
    Create but don't show the scatterplot using the points from x and y

    Args:
        x The array of x values to plot
        y The array of associated y values to plot
    """
    if len(x) != len(y):
        raise ValueError("Vectors x and y must be the same length")
    plt.scatter(x, y, color="blue", marker="o")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Scatter Plot of Data Points")


def plot_regression_line(x, B0, B1):
    """
    Plot the regression line on the graph
    """
    y_pred = predict(x, B0, B1)
    plt.plot(x, y_pred, color="red", label="Regression Line")


# Generate test data
x1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y1 = np.array([2.1, 3.9, 6.0, 7.8, 10.2, 11.9, 14.2, 15.9, 18.1, 20.3])  # Approximately y = 2x + noise

x2 = np.linspace(1, 10, 50)
y2 = 2*x2 + np.random.normal(0, 5, 50)

B0, B1 = compute_coefficients(x2, y2)

scatterplot(x2, y2)
plot_regression_line(x2, B0, B1)
plt.show()
