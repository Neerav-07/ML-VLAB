
import numpy as np
import time


def predict(X, weights, bias):
    return np.dot(X, weights) + bias


def calculate_loss(X, y, weights, bias):
    predictions = predict(X, weights, bias)
    return np.mean((predictions - y) ** 2)


def gradient_descent(
    X,
    y,
    learning_rate=0.01,
    epochs=500,
    method="Batch",
    batch_size=32,
    tolerance=1e-7,
    random_state=42
):
    """
    Implements Batch, Stochastic and Mini-Batch
    Gradient Descent for linear regression.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n_samples, n_features = X.shape

    weights = np.zeros(n_features)
    bias = 0.0

    loss_history = []
    converged = False
    status = "Maximum iterations reached"

    rng = np.random.default_rng(random_state)

    start_time = time.perf_counter()

    previous_loss = calculate_loss(X, y, weights, bias)
    loss_history.append(previous_loss)

    stable_epochs = 0
    completed_epochs = 0

    for epoch in range(epochs):

        if method == "Batch":

            predictions = predict(X, weights, bias)
            errors = predictions - y

            dw = (2 / n_samples) * np.dot(X.T, errors)
            db = 2 * np.mean(errors)

            weights -= learning_rate * dw
            bias -= learning_rate * db

        elif method == "Stochastic":

            indices = rng.permutation(n_samples)

            for i in indices:
                xi = X[i]
                yi = y[i]

                error = np.dot(xi, weights) + bias - yi

                dw = 2 * xi * error
                db = 2 * error

                weights -= learning_rate * dw
                bias -= learning_rate * db

        elif method == "Mini-Batch":

            indices = rng.permutation(n_samples)

            for start in range(0, n_samples, batch_size):

                batch_indices = indices[
                    start:start + batch_size
                ]

                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                predictions = predict(
                    X_batch, weights, bias
                )

                errors = predictions - y_batch

                batch_count = len(batch_indices)

                dw = (2 / batch_count) * np.dot(
                    X_batch.T, errors
                )

                db = 2 * np.mean(errors)

                weights -= learning_rate * dw
                bias -= learning_rate * db

        else:
            raise ValueError("Invalid gradient descent method")

        current_loss = calculate_loss(
            X, y, weights, bias
        )

        completed_epochs = epoch + 1

        if not np.isfinite(current_loss) or current_loss > 1e12:
            status = "Diverged: Learning rate too high"
            break

        loss_history.append(current_loss)

        relative_change = abs(
            previous_loss - current_loss
        ) / max(abs(previous_loss), 1e-12)

        if relative_change < tolerance:
            stable_epochs += 1
        else:
            stable_epochs = 0

        if stable_epochs >= 5:
            converged = True
            status = "Converged"
            break

        previous_loss = current_loss

    execution_time = time.perf_counter() - start_time

    return {
        "weights": weights,
        "bias": bias,
        "loss_history": loss_history,
        "final_loss": loss_history[-1],
        "iterations": completed_epochs,
        "execution_time": execution_time,
        "converged": converged,
        "status": status
    }
    