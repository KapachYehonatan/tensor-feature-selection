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


def predict_prefixes(train, test, ytrain, ytest, ks):
    """Fit each distinct prefix once; metrics are calculated from saved probabilities."""
    predictions, cache = [], {}
    for k in ks:
        effective = min(k, train.shape[1])
        if effective not in cache:
            probability = predict_model(train[:, :effective], ytrain, test[:, :effective])
            cache[effective] = probability
        probability = cache[effective]
        predictions.extend(dict(requested_k=k, test_position=i, y=int(ytest[i]),
                                probability=float(value)) for i, value in enumerate(probability))
    return pd.DataFrame(predictions)



def point_metrics(y, probability):
    prediction = probability >= .5
    return dict(auroc=roc_auc_score(y, probability), precision=precision_score(y, prediction, zero_division=0),
        recall=recall_score(y, prediction, zero_division=0), f1=f1_score(y, prediction, zero_division=0))


def summarize(folds, predictions, evaluation):
    """Incomplete CV pairs never enter mean-fold or pooled OOF comparisons."""
    columns = ['dataset', 'method', 'requested_k', 'auroc_cv_mean', 'auroc_fold_sd',
               'precision_cv_mean', 'recall_cv_mean', 'f1_cv_mean', 'auroc_pooled_oof',
               'precision_pooled_oof', 'recall_pooled_oof', 'f1_pooled_oof', 'effective_k_min', 'effective_k_max']
    rows = []
    for (dataset, method), group in folds.groupby(['dataset', 'method']):
        expected = {(f, k) for f in range(evaluation.folds) for k in evaluation.k}
        if group.duplicated(['fold', 'requested_k']).any():
            raise ValueError('Duplicate fold/K metrics')
        if set(map(tuple, group[['fold', 'requested_k']].to_numpy())) != expected:
            continue
        pred = predictions[predictions.dataset.eq(dataset) & predictions.method.eq(method)]
        n = int(group.participants.iloc[0])
        if not group.participants.eq(n).all():
            raise ValueError('Changed participant count')
        for k in evaluation.k:
            selected = pred[pred.requested_k.eq(k)].sort_values('sample_index')
            if not np.array_equal(selected.sample_index.to_numpy(), np.arange(n)):
                raise ValueError('Incomplete once-only OOF coverage')
            fold = group[group.requested_k.eq(k)]
            pooled = point_metrics(selected.y.to_numpy(), selected.probability.to_numpy())
            row = dict(dataset=dataset, method=method, requested_k=k, auroc_fold_sd=fold.auroc.std(),
                       effective_k_min=fold.effective_k.min(), effective_k_max=fold.effective_k.max())
            row.update({name+'_cv_mean': float(fold[name].mean()) for name in pooled})
            row.update({name+'_pooled_oof': value for name, value in pooled.items()})
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)

