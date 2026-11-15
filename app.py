
import streamlit as st
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from data_handler import (
    load_dataset,
    preprocess_data
)

from gradient_descent import (
    gradient_descent,
    predict
)

from utils.visualizations import (
    plot_loss_curve,
    plot_learning_rate_comparison,
    plot_actual_vs_predicted,
    plot_method_comparison
)


st.set_page_config(
    page_title="Gradient Descent Virtual Lab",
    page_icon="📈",
    layout="wide"
)


# ---------------- SIDEBAR ----------------

st.sidebar.title("Experiment Controls")

st.sidebar.subheader("1. Dataset")

dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ["Diabetes"]
)

st.sidebar.subheader("2. Optimization Method")

method = st.sidebar.selectbox(
    "Select Method",
    ["Batch", "Stochastic", "Mini-Batch"]
)

st.sidebar.subheader("3. Model Parameters")

learning_rate = st.sidebar.selectbox(
    "Learning Rate",
    [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
    index=2
)

epochs = st.sidebar.slider(
    "Number of Iterations",
    100, 3000, 1000, 100
)

batch_size = st.sidebar.slider(
    "Mini-Batch Size",
    8, 128, 32, 8
)

tolerance = st.sidebar.selectbox(
    "Convergence Tolerance",
    [1e-4, 1e-5, 1e-6, 1e-7],
    index=2
)

st.sidebar.caption(
    "Features and target are standardized before training."
)

run_button = st.sidebar.button(
    "Run Experiment",
    type="primary",
    use_container_width=True
)


# ---------------- DATA LOADING ----------------

@st.cache_data
def get_dataset():
    return load_dataset()


X, y = get_dataset()

(
    X_train,
    X_test,
    y_train,
    y_test,
    y_scaler,
    original_X,
    original_y
) = preprocess_data()


# ---------------- HEADER ----------------

st.title("Gradient Descent Virtual Laboratory")

st.markdown(
    "### Experiment: Analyze Convergence, Learning Rate "
    "Effects and Optimization Performance"
)

st.markdown(
    "Study how different gradient descent methods optimize "
    "a linear regression model using an interactive experiment."
)

tab1, tab2, tab3 = st.tabs([
    "Theory",
    "Experiment & Results",
    "Dataset Explorer"
])


# ---------------- THEORY TAB ----------------

with tab1:

    st.header("Aim")

    st.write(
        "To implement the Gradient Descent algorithm on a given "
        "dataset and analyze its convergence behavior, learning "
        "rate effects and optimization performance."
    )

    st.header("Learning Objectives")

    objectives = [
        "Understand the working of the Gradient Descent algorithm.",
        "Study the role of the learning rate in optimization.",
        "Analyze the convergence of the cost function.",
        "Compare Batch, Stochastic and Mini-Batch Gradient Descent.",
        "Evaluate model performance using regression metrics."
    ]

    for objective in objectives:
        st.write("• " + objective)

    st.header("Mathematical Formulation")

    st.latex(
        r"J(w,b)=\frac{1}{n}\sum_{i=1}^{n}"
        r"(\hat{y}_i-y_i)^2"
    )

    st.latex(
        r"w := w-\alpha\frac{\partial J}{\partial w}"
    )

    st.latex(
        r"b := b-\alpha\frac{\partial J}{\partial b}"
    )

    st.header("Evaluation Metrics")

    st.write("• MSE: Measures average squared prediction error.")
    st.write("• MAE: Measures average absolute prediction error.")
    st.write("• R² Score: Measures the explanatory performance of the model.")
    st.write("• Convergence: Measures whether optimization stabilizes.")
    st.write("• Execution Time: Measures computational runtime.")


# ---------------- EXPERIMENT TAB ----------------

with tab2:

    st.header("Experiment & Results")

    st.write(
        "Configure the experiment using the sidebar and click "
        "'Run Experiment'."
    )

    if run_button:

        result = gradient_descent(
            X_train,
            y_train,
            learning_rate=learning_rate,
            epochs=epochs,
            method=method,
            batch_size=batch_size,
            tolerance=tolerance
        )

        st.session_state["result"] = result
        st.session_state["config"] = {
            "method": method,
            "learning_rate": learning_rate
        }

    if "result" in st.session_state:

        result = st.session_state["result"]

        st.subheader("Optimization Results")

        if "Diverged" in result["status"]:
            st.error(result["status"])
        elif result["converged"]:
            st.success(result["status"])
        else:
            st.warning(result["status"])

        weights = result["weights"]
        bias = result["bias"]

        test_predictions_scaled = predict(
            X_test, weights, bias
        )

        test_predictions = y_scaler.inverse_transform(
            test_predictions_scaled.reshape(-1, 1)
        ).ravel()

        actual_values = y_scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).ravel()

        if np.all(np.isfinite(test_predictions)):

            mse = mean_squared_error(
                actual_values, test_predictions
            )

            mae = mean_absolute_error(
                actual_values, test_predictions
            )

            r2 = r2_score(
                actual_values, test_predictions
            )

        else:
            mse = mae = r2 = float("nan")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Final Training Loss",
            f"{result['final_loss']:.6f}"
        )

        col2.metric(
            "Test MSE",
            f"{mse:.2f}" if np.isfinite(mse) else "N/A"
        )

        col3.metric(
            "Test MAE",
            f"{mae:.2f}" if np.isfinite(mae) else "N/A"
        )

        col4.metric(
            "R² Score",
            f"{r2:.4f}" if np.isfinite(r2) else "N/A"
        )

        col5, col6 = st.columns(2)

        col5.metric(
            "Iterations",
            result["iterations"]
        )

        col6.metric(
            "Execution Time",
            f"{result['execution_time']:.4f} sec"
        )

        st.subheader("Loss vs Iterations")

        st.plotly_chart(
            plot_loss_curve(result["loss_history"]),
            use_container_width=True
        )

        st.subheader("Actual vs Predicted Values")

        if np.all(np.isfinite(test_predictions)):

            st.plotly_chart(
                plot_actual_vs_predicted(
                    actual_values,
                    test_predictions
                ),
                use_container_width=True
            )

        st.subheader("Learning Rate Analysis")

        learning_rates = [0.001, 0.01, 0.05, 0.1, 0.5]

        comparison_results = {}

        for lr in learning_rates:

            comparison = gradient_descent(
                X_train,
                y_train,
                learning_rate=lr,
                epochs=epochs,
                method="Batch",
                tolerance=tolerance
            )

            comparison_results[f"LR = {lr}"] = (
                comparison["loss_history"]
            )

        st.plotly_chart(
            plot_learning_rate_comparison(
                comparison_results
            ),
            use_container_width=True
        )

        st.subheader("Optimization Method Comparison")

        method_results = {}

        for selected_method in [
            "Batch",
            "Stochastic",
            "Mini-Batch"
        ]:

            comparison = gradient_descent(
                X_train,
                y_train,
                learning_rate=learning_rate,
                epochs=epochs,
                method=selected_method,
                batch_size=batch_size,
                tolerance=tolerance
            )

            method_results[selected_method] = (
                comparison["loss_history"]
            )

        st.plotly_chart(
            plot_method_comparison(method_results),
            use_container_width=True
        )

        st.subheader("Performance Comparison Table")

        table_data = []

        for name, history in method_results.items():

            table_data.append({
                "Method": name,
                "Final Training Loss": round(history[-1], 6),
                "Iterations": len(history) - 1
            })

        st.dataframe(
            pd.DataFrame(table_data),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Loss History CSV",
            pd.DataFrame({
                "Iteration": range(
                    len(result["loss_history"])
                ),
                "Loss": result["loss_history"]
            }).to_csv(index=False),
            file_name="gradient_descent_results.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "No experiment has been run yet. "
            "Configure the parameters and click Run Experiment."
        )


# ---------------- DATASET TAB ----------------

with tab3:

    st.header("Dataset Explorer")

    st.write("Dataset: Diabetes")

    col1, col2, col3 = st.columns(3)

    col1.metric("Samples", original_X.shape[0])
    col2.metric("Features", original_X.shape[1])
    col3.metric("Target", "Disease Progression")

    st.subheader("Dataset Preview")

    dataset = original_X.copy()
    dataset["Target"] = original_y

    st.dataframe(
        dataset.head(15),
        use_container_width=True
    )

    st.subheader("Feature Statistics")

    st.dataframe(
        original_X.describe(),
        use_container_width=True
    )

    st.subheader("Training and Testing Split")

    st.write("Training Samples:", len(X_train))
    st.write("Testing Samples:", len(X_test))

    st.subheader("Feature Names")

    st.write(list(original_X.columns))

    st.success(
        "Dataset loaded and preprocessing completed successfully."
    )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "ML Virtual Lab | Gradient Descent Experiment | "
    "Built using Python, NumPy, Pandas and Streamlit"
)
