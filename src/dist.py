import numpy as np

from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import scale


def chem_dist_mat(
    fullOdorSpace_normalized,
    scaleIndependently=True,
    distanceMetric="correlation",
    with_mean=True,
    with_std=False,
):
    if scaleIndependently:
        indScaledChem_df = scale(
            fullOdorSpace_normalized,
            axis=0,
            with_mean=with_mean,
            with_std=with_std,
        )
        cormat = cor_dist(indScaledChem_df.T, distanceMetric)
    else:
        cormat = cor_dist(fullOdorSpace_normalized.T, distanceMetric)
    return cormat


def neural_dist_mat(
    pseudopop, meanOrTrial="mean", distanceMetric="correlation", shuff=False
):
    # pseudopop is trials by cells by odors for combining either layer 2 or 3 neurons across 3 subjects.
    if meanOrTrial == "mean":
        pseudopop = pseudopop.mean(0)
        if shuff:
            cormat = cor_dist(
                np.apply_along_axis(np.random.permutation, axis=-1, arr=pseudopop),
                distanceMetric,
            )
        else:
            cormat = cor_dist(pseudopop, distanceMetric)

    elif meanOrTrial == "trial":
        ntrials, ncells, nodors = pseudopop.shape
        pseudopop = pseudopop.transpose(1, 2, 0).reshape(ncells, nodors * ntrials)
        cormat = cor_dist(pseudopop, distanceMetric)
    return cormat


def cor_dist(cellsByOdor, metric_="correlation"):
    return pairwise_distances(cellsByOdor.T, metric=metric_)


def prepare_scatter(neural_dict, chem_subset, dist_keys):
    neural_flat_dist = {
        d: squareform(cor_dist(neural_dict[d].mean(0)), checks=False) for d in dist_keys
    }
    chemistry = scale(chem_subset, with_std=True)
    chem_flat = squareform(cor_dist(chemistry.T))
    return neural_flat_dist, chem_flat
