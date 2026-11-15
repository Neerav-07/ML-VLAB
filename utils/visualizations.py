
import numpy as np
import plotly.graph_objects as go


def plot_loss_curve(loss_history):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(len(loss_history))),
            y=loss_history,
            mode="lines",
            name="Training Loss"
        )
    )

    fig.update_layout(
        title="Loss vs Iterations",
        xaxis_title="Iterations",
        yaxis_title="Mean Squared Error",
        template="plotly_white"
    )

    return fig


def plot_learning_rate_comparison(results):

    fig = go.Figure()

    for label, history in results.items():

        fig.add_trace(
            go.Scatter(
                x=list(range(len(history))),
                y=history,
                mode="lines",
                name=label
            )
        )

    fig.update_layout(
        title="Learning Rate Comparison",
        xaxis_title="Iterations",
        yaxis_title="Training Loss",
        template="plotly_white"
    )

    return fig


def plot_actual_vs_predicted(actual, predicted):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=actual,
            y=predicted,
            mode="markers",
            name="Predictions"
        )
    )

    minimum = min(np.min(actual), np.min(predicted))
    maximum = max(np.max(actual), np.max(predicted))

    fig.add_trace(
        go.Scatter(
            x=[minimum, maximum],
            y=[minimum, maximum],
            mode="lines",
            name="Ideal Prediction"
        )
    )

    fig.update_layout(
        title="Actual vs Predicted Values",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        template="plotly_white"
    )

    return fig


def plot_method_comparison(results):

    fig = go.Figure()

    for method, history in results.items():

        fig.add_trace(
            go.Scatter(
                x=list(range(len(history))),
                y=history,
                mode="lines",
                name=method
            )
        )

    fig.update_layout(
        title="Optimization Method Comparison",
        xaxis_title="Iterations",
        yaxis_title="Training Loss",
        template="plotly_white"
    )

    return fig
