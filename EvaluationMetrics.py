
from __future__ import division
from numpy import log, sqrt
from pandas import DataFrame


def mse(y_hat, y):
    return ((y_hat - y) ** 2).mean()


def rmse(y_hat, y):
    return sqrt(mse(y_hat, y))


def bin_class_dev(p_hat, y, pos_cat=None, tiny=1e-32):
    if hasattr(y, 'cat'):
        y_bool = y == pos_cat
    else:
        y_bool = y.astype(bool)
    return - 2 * (y_bool * log(p_hat + tiny) + (1 - y_bool) * log(1 - p_hat + tiny)).mean()


def bin_classif_eval_hard_pred(hard_predictions, actuals, pos_cat=None):

    if hasattr(hard_predictions, 'cat'):
        hard_predictions_bool = hard_predictions == pos_cat
    else:
        hard_predictions_bool = hard_predictions.astype(bool)

    if hasattr(actuals, 'cat'):
        actuals_bool = actuals == pos_cat
    else:
        actuals_bool = actuals.astype(bool)

    opposite_hard_predictions_bool = ~ hard_predictions_bool
    opposite_actuals_bool = ~ actuals_bool

    nb_samples = len(actuals)
    nb_pos = sum(actuals_bool)
    nb_neg = sum(opposite_actuals_bool)
    nb_pred_pos = sum(hard_predictions_bool)
    nb_pred_neg = sum(opposite_hard_predictions_bool)
    nb_true_pos = sum(hard_predictions_bool & actuals_bool)
    nb_true_neg = sum(opposite_hard_predictions_bool & opposite_actuals_bool)
    nb_false_pos = sum(hard_predictions_bool & opposite_actuals_bool)
    nb_false_neg = sum(opposite_hard_predictions_bool & actuals_bool)

    accuracy = (nb_true_pos + nb_true_neg) / nb_samples
    recall = nb_true_pos / nb_pos
    specificity = nb_true_neg / nb_neg
    precision = nb_true_pos / nb_pred_pos
    f1_score = (2 * precision * recall) / (precision + recall)

    return dict(
        accuracy=accuracy,
        recall=recall,
        specificity=specificity,
        precision=precision,
        f1_score=f1_score)


def bin_classif_eval(predictions, actuals, pos_cat=None, thresholds=.5):

    if hasattr(predictions, 'cat') or (predictions.dtype in ('bool', 'int')):
        return bin_classif_eval_hard_pred(predictions, actuals, pos_cat=pos_cat)

    if isinstance(thresholds, (float, int)):
        hard_predictions = predictions >= thresholds
        metrics = bin_classif_eval_hard_pred(hard_predictions, actuals, pos_cat=pos_cat)
        metrics['deviance'] = bin_class_dev(predictions, actuals, pos_cat=pos_cat)
    else:
        metrics = DataFrame(dict(threshold=thresholds))
        metrics['accuracy'] = 0.
        metrics['recall'] = 0.
        metrics['specificity'] = 0.
        metrics['precision'] = 0.
        metrics['f1_score'] = 0.
        metrics['deviance'] = 0.
        for i in range(len(thresholds)):
            m = bin_classif_eval(predictions, actuals, pos_cat=pos_cat, thresholds=thresholds[i])
            metrics.ix[i, 'accuracy'] = m['accuracy']
            metrics.ix[i, 'recall'] = m['recall']
            metrics.ix[i, 'specificity'] = m['specificity']
            metrics.ix[i, 'precision'] = m['precision']
            metrics.ix[i, 'f1_score'] = m['f1_score']
            metrics.ix[i, 'deviance'] = m['deviance']

    return metrics
