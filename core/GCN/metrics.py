import numpy as np
from scipy.stats import beta


def binary_vector(values, name='labels'):
    values = np.asarray(values)
    if values.ndim != 1 or not np.isin(values, [0, 1]).all():
        raise ValueError(f'{name} must be a one-dimensional binary vector')
    return values.astype(np.uint8)


def probabilities_vector(values, count=None):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or (count is not None and len(values) != count):
        raise ValueError('Probability vector does not align with labels')
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Probabilities must be finite and in [0, 1]')
    return values


def binomial_interval(errors, count, confidence=0.95):
    if not 0 < confidence < 1 or not 0 <= errors <= count:
        raise ValueError('Invalid binomial counts or confidence level')
    if count == 0:
        return [None, None]
    alpha = 1 - confidence
    return [0.0 if errors == 0 else float(beta.ppf(alpha / 2, errors, count - errors + 1)),
            1.0 if errors == count else float(beta.ppf(1 - alpha / 2, errors + 1, count - errors))]


def upper_error_bound(errors, count, alpha=0.05):
    if not 0 < alpha < 1 or not 0 <= errors <= count:
        raise ValueError('Invalid binomial counts or alpha')
    if count == 0:
        return None
    return 1.0 if errors == count else float(beta.ppf(1 - alpha, errors + 1, count - errors))


def classification_metrics(labels, predictions, confidence=0.95):
    labels = binary_vector(labels)
    predictions = binary_vector(predictions, 'predictions')
    if labels.shape != predictions.shape:
        raise ValueError('Labels and predictions must align')
    n = len(labels)
    errors = int(np.count_nonzero(labels != predictions))
    return {'shots': n, 'incorrect': errors, 'accuracy': 1 - errors / n if n else None,
            'error_rate': errors / n if n else None,
            'error_rate_ci': binomial_interval(errors, n, confidence),
            'error_rate_upper': upper_error_bound(errors, n, 1 - confidence),
            'confidence': confidence, 'interval_method': 'Clopper-Pearson; independent shots',
            'positive_prevalence': float(labels.mean()) if n else None,
            'confusion': {f'{actual}{pred}': int(np.sum((labels == actual) & (predictions == pred)))
                          for actual in (0, 1) for pred in (0, 1)}}


def probability_metrics(labels, probabilities, bins=10, confidence=0.95):
    labels = binary_vector(labels)
    p = probabilities_vector(probabilities, len(labels))
    if len(labels) == 0 or bins < 1:
        raise ValueError('Probability metrics require nonempty records and positive bin count')
    result = classification_metrics(labels, p > 0.5, confidence)
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    result['bce'] = float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)))
    result['bce_clipping_epsilon'] = 1e-7
    result['brier'] = float(np.mean((p - labels) ** 2))
    from scipy.stats import rankdata
    positives = int(labels.sum())
    negatives = len(labels) - positives
    result['roc_auc'] = (float((rankdata(p)[labels == 1].sum() - positives * (positives + 1) / 2)
                                / (positives * negatives)) if positives and negatives else None)
    bins_index = np.minimum((p * bins).astype(int), bins - 1)
    reliability = []
    for index in range(bins):
        mask = bins_index == index
        reliability.append({'lower': index / bins, 'upper': (index + 1) / bins,
                            'shots': int(mask.sum()),
                            'mean_probability': float(p[mask].mean()) if mask.any() else None,
                            'positive_fraction': float(labels[mask].mean()) if mask.any() else None})
    result['reliability'] = reliability
    result['ece'] = float(sum(row['shots'] / len(labels) * abs(row['mean_probability'] - row['positive_fraction'])
                              for row in reliability if row['shots']))
    return result


def paired_difference(labels, first, second, repetitions=2000, seed=73, confidence=0.95):
    labels = binary_vector(labels)
    first, second = binary_vector(first), binary_vector(second)
    if first.shape != labels.shape or second.shape != labels.shape or not len(labels):
        raise ValueError('Paired predictions must match nonempty labels')
    if repetitions < 1:
        raise ValueError('Bootstrap repetitions must be positive')
    difference = (first != labels).astype(int) - (second != labels).astype(int)
    counts = np.array([np.sum(difference == value) for value in (-1, 0, 1)])
    draws = np.random.default_rng(seed).multinomial(len(labels), counts / len(labels), size=repetitions)
    differences = (draws[:, 2] - draws[:, 0]) / len(labels)
    alpha = (1 - confidence) / 2
    return {'error_rate_difference': float(difference.mean()),
            'ci': np.quantile(differences, [alpha, 1 - alpha]).tolist(),
            'first_only_wrong': int(counts[2]), 'second_only_wrong': int(counts[0]),
            'method': 'paired shot bootstrap', 'repetitions': repetitions, 'seed': seed}
