"""Training-scaled balanced L2 prediction and saved-probability summaries."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = MODEL_SEED = 42

def new_model(model_seed=MODEL_SEED):
    return make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=model_seed,
            solver="liblinear",
        ),
    )


def predict_model(train_x, train_y, test_x, model_seed=MODEL_SEED):
    if train_x.shape[1] == 0:
        return np.full(test_x.shape[0], np.mean(train_y))
    model = new_model(model_seed)
    model.fit(train_x, train_y)
    return model.predict_proba(test_x)[:, 1]


def evaluate(train, test, ytrain, ytest, ks, c=1):
    rows, predictions, cache = [], [], {}
    for k in ks:
        effective = min(k, train.shape[1])
        if effective not in cache:
            probability = np.full(len(ytest), ytrain.mean())
            if effective:
                model = new_model(SEED)
                model.set_params(logisticregression__C=c)
                model.fit(train[:, :effective], ytrain)
                probability = model.predict_proba(test[:, :effective])[:, 1]
            cache[effective] = probability
        probability = cache[effective]
        predicted = probability >= .5
        rows.append(dict(requested_k=k, effective_k=effective,
                         auroc=roc_auc_score(ytest, probability),
                         precision=precision_score(ytest, predicted, zero_division=0),
                         recall=recall_score(ytest, predicted, zero_division=0),
                         f1=f1_score(ytest, predicted, zero_division=0)))
        predictions.extend(dict(requested_k=k, test_position=i, y=int(ytest[i]),
                                probability=float(value)) for i, value in enumerate(probability))
    return pd.DataFrame(rows), pd.DataFrame(predictions)



def prediction_metrics(predictions):
    rows = []
    for (method, k), group in predictions.groupby(['method', 'requested_k']):
        if not np.isfinite(group.probability).all() or not group.probability.between(0, 1).all():
            raise ValueError('Invalid out-of-fold probabilities')
        y, probability = group.y.to_numpy(), group.probability.to_numpy()
        predicted = probability >= .5
        rows.append(dict(method=method, requested_k=k, auroc=roc_auc_score(y, probability),
            precision=precision_score(y, predicted, zero_division=0), recall=recall_score(y, predicted, zero_division=0),
            f1=f1_score(y, predicted, zero_division=0)))
    return pd.DataFrame(rows).sort_values(['method', 'requested_k']).reset_index(drop=True)



def point_metrics(y, probability):
    prediction = probability >= .5
    return dict(auroc=roc_auc_score(y, probability), precision=precision_score(y, prediction, zero_division=0),
        recall=recall_score(y, prediction, zero_division=0), f1=f1_score(y, prediction, zero_division=0))

