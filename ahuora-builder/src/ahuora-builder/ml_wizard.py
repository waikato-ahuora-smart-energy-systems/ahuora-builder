from pydantic import BaseModel
from ahuora_builder_types.payloads.ml_request_schema import MLTrainRequestPayload, MLTrainingCompletionPayload
import pandas as pd
import json
import numpy as np
from io import StringIO
import contextlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from idaes.core.surrogate.pysmo_surrogate import PysmoRBFTrainer, PysmoSurrogate


class MLResult(BaseModel):
    surrogate_model: dict
    charts: list[dict]
    metrics: list[dict]
    test_inputs: dict
    test_outputs: dict
    task_id: int


def ml_generate(schema: MLTrainRequestPayload) -> MLResult:
    df = pd.DataFrame(schema.datapoints, columns=schema.columns)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    input_labels = schema.input_labels
    output_labels = schema.output_labels
            
    trainer = PysmoRBFTrainer(
        input_labels=input_labels, output_labels=output_labels, training_dataframe=train_df)
    trainer.config.basis_function = 'gaussian'

    # Train surrogate (calls PySMO through IDAES Python wrapper)
    stream = StringIO()
    with contextlib.redirect_stdout(stream):
        rbf_train = trainer.train_surrogate()


    # create callable surrogate model
    rbf_surr = PysmoSurrogate(rbf_train, input_labels,
                              output_labels, input_bounds=None)
    f = StringIO()
    rbf_surr.save(f)
    content = f.getvalue()
    json_data = json.loads(content)

    df_evaluate = rbf_surr.evaluate_surrogate(test_df)

    metrics = []

    charts = []

    for output_label in output_labels:
        charts.append(compute_chart(
            test_df[output_label], df_evaluate[output_label], output_label))
        metrics.append({
            "mean_squared_error": round(mean_squared_error(df_evaluate[output_label], test_df[output_label]), 4),
            "r2_score": round(r2_score(df_evaluate[output_label], test_df[output_label]), 4),
        })

    return MLResult(
        surrogate_model=json_data,
        charts=charts,
        metrics=metrics,
        test_inputs=test_df.to_dict(orient='index'),
        test_outputs=df_evaluate.to_dict(orient='index'),
        task_id=schema.task_id
    )


def compute_chart(test_data_df, eval_data_df, output_label):
    minn = round(np.min([np.min(test_data_df), np.min(eval_data_df)]), 4)
    maxx = round(np.max([np.max(test_data_df), np.max(eval_data_df)]), 4)
    qq_plot_data = compute_qq_coordinates(test_data_df, eval_data_df)

    return {
        "min": minn,
        "max": maxx,
        "qq_plot_data": qq_plot_data,
        "output_label": output_label
    }


def compute_qq_coordinates(test_data, eval_data):
    """Compute QQ plot coordinates for two datasets."""
    test_values = test_data.to_numpy().flatten()
    eval_values = eval_data.to_numpy().flatten()

    # Sort values
    test_values.sort()
    eval_values.sort()

    # Generate QQ plot data points
    quantiles = np.linspace(0, 1, len(test_values))
    test_quantiles = np.quantile(test_values, quantiles)
    eval_quantiles = np.quantile(eval_values, quantiles)

    # Prepare JSON response for frontend
    qq_data = [{"x": round(float(t), 4), "y": round(float(e), 4)}
               for t, e in zip(test_quantiles, eval_quantiles)]
    return json.dumps(qq_data)
