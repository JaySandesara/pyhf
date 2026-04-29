"""Custom pyhf modifier: Gaussian Process interpolation, additive variant.

Additive sibling of ``gphistosys``. Uses ``op_code='addition'`` and the
additive ``code4p`` HistFactory interpolation as the GP prior, so that under
on-axis-only training nodes the modifier reproduces ``histosys`` exactly.

GP posterior mean for bin b (with histosys-consistent additive prior)
---------------------------------------------------------------------
    delta_b(a) = m_b(a) + k(a, A) K^{-1} (r_b - m_b(A))

where
    m_b(a)   = sum_i delta_i(a_i, b)             additive HistFactory prior
    delta_i  = code4p interpolation (pyhf.interpolators.code4p) for dimension i
    A        = anchor nodes  (N x d)
    r_b      = templates[:, b] - nominal_b       (additive deltas at anchors)
    k(a, A)  = [k(a, a_1), ..., k(a, a_N)]
    K        = K(A, A) + noise^2 I

The modifier output is summed into the per-sample yield by pyhf:
    yield = nominal + sum_modifiers delta_b(a)

Spec format is identical to ``gphistosys`` apart from the type string:

    {
        "name": "alpha",
        "type": "gphistosys_additive",
        "data": {
            "nodes":     [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0 , -1.0]],
            "templates": [...],
            "labels":    ["jet_energy", "b_tagging"]
        }
    }
"""

import logging

import numpy as np
from pyhf import get_backend, events
from pyhf.parameters import ParamViewer
from pyhf.interpolators.code4p import _slow_code4p, code4p
from pyhf.modifiers.gphistosys import (
    gphistosys_builder,
    _sq_exp_kernel_np,
    _find_axis_node,
)

log = logging.getLogger(__name__)


# Spec parsing and data collection are identical to gphistosys — reuse the
# builder class verbatim under a name that matches the modifier type.
class gphistosys_additive_builder(gphistosys_builder):
    pass


_code4p_summand = _slow_code4p.__new__(_slow_code4p).summand


def _additive_prior_at_nodes(anchor_alphas, nom, lo_templates, hi_templates):
    """Additive code4p prior m(X_j) = sum_i delta_i at each training node.

    Parameters
    ----------
    anchor_alphas : (N, d)
    nom           : (n_bins,)
    lo_templates  : list of d arrays, each (n_bins,) — raw yields at -1 per dim
    hi_templates  : list of d arrays, each (n_bins,) — raw yields at +1 per dim

    Returns
    -------
    prior : (N, n_bins)  — additive deltas (yield space)
    """
    N, d = anchor_alphas.shape
    n_bins = nom.shape[0]
    prior = np.zeros((N, n_bins))
    for dim in range(d):
        for j in range(N):
            a_i = anchor_alphas[j, dim]
            for b in range(n_bins):
                prior[j, b] += _code4p_summand(
                    lo_templates[dim][b], nom[b], hi_templates[dim][b], a_i
                )
    return prior


class gphistosys_additive_combined:
    """Combined modifier class for additive GP-based histogram interpolation.

    Mirrors the multiplicative ``gphistosys_combined`` but operates entirely
    in delta space:

      * ``op_code = 'addition'`` — pyhf sums per-modifier outputs into yields
      * anchor targets are ``templates - nom`` (not ratios)
      * the GP prior is the *sum* of per-dimension code4p deltas
      * the masked / satellite no-op value is 0 (not 1)
    """

    name    = 'gphistosys_additive'
    op_code = 'addition'

    def __init__(
        self,
        modifiers,
        pdfconfig,
        builder_data,
        length_scale=1.0,
        variance=1.0,
        noise=0.0,
        batch_size=None,
    ):
        self.batch_size = batch_size
        self.length_scale = float(length_scale)
        self.variance = float(variance)
        self.noise = float(noise)

        keys = [f'{mtype}/{m}' for m, mtype in modifiers]

        key_to_labels = {}
        for key in keys:
            spec_name = key.split('/')[1]
            key_to_labels[key] = builder_data[key].get('__labels__', [spec_name])

        all_labels = [lbl for key in keys for lbl in key_to_labels[key]]

        label_to_key = {lbl: key for key in keys for lbl in key_to_labels[key]}
        label_to_dim = {lbl: i for key in keys
                        for i, lbl in enumerate(key_to_labels[key])}

        self._effective_batch = self.batch_size or 1
        parfield_shape = (self._effective_batch, pdfconfig.npars)
        self.param_viewer = ParamViewer(
            parfield_shape, pdfconfig.par_map, all_labels
        )

        # ------------------------------------------------------------------
        # Precompute GP weights with additive HistFactory prior.
        # ------------------------------------------------------------------
        self._gp_meta    = []
        self._key_labels = []

        for bkey in keys:
            labels_for_key = key_to_labels[bkey]
            self._key_labels.append(labels_for_key)

            first_sample = pdfconfig.samples[0]
            anchor_alphas = np.array(
                builder_data[bkey][first_sample]['data']['nodes'], dtype=float
            )
            N, d = anchor_alphas.shape

            lo_node_idx, hi_node_idx = [], []
            for dim in range(d):
                hi_idx = _find_axis_node(anchor_alphas, dim, +1.0)
                lo_idx = _find_axis_node(anchor_alphas, dim, -1.0)
                if hi_idx is None or lo_idx is None:
                    raise ValueError(
                        f"gphistosys_additive requires axis-aligned nodes at +1 and -1 "
                        f"for dimension {dim} (labels: {labels_for_key}) to build the "
                        f"HistFactory prior."
                    )
                hi_node_idx.append(hi_idx)
                lo_node_idx.append(lo_idx)

            K_train = _sq_exp_kernel_np(
                anchor_alphas, anchor_alphas, self.length_scale, self.variance
            )
            K_train += (self.noise ** 2) * np.eye(N)
            K_train_inv = np.linalg.inv(K_train)

            weights_per_sample = []
            # histogramssets for code4p: (d, n_samples, 3, n_bins)
            histogramssets_list = [[] for _ in range(d)]

            for s in pdfconfig.samples:
                sdata = builder_data[bkey][s]['data']
                nom = np.array(sdata['nom_data'], dtype=float)
                templates = np.array(sdata['templates'], dtype=float)

                # Anchor targets in delta space.
                anchor_deltas = templates - nom[None, :]

                lo_templates = [templates[lo_node_idx[dim]] for dim in range(d)]
                hi_templates = [templates[hi_node_idx[dim]] for dim in range(d)]

                # code4p does not divide by nom internally, so no nom == 0
                # sanitization is needed in the histogramssets.
                for dim in range(d):
                    histogramssets_list[dim].append(
                        [lo_templates[dim], nom, hi_templates[dim]]
                    )

                prior_at_nodes = _additive_prior_at_nodes(
                    anchor_alphas, nom, lo_templates, hi_templates
                )
                weights_per_sample.append(
                    K_train_inv @ (anchor_deltas - prior_at_nodes)
                )

            histogramssets = np.array(histogramssets_list, dtype=float)

            self._gp_meta.append({
                'anchor_alphas': anchor_alphas,
                'weights': weights_per_sample,
                'histogramssets': histogramssets,
            })

        n_labels = len(all_labels)
        _access_field = np.zeros((n_labels, self._effective_batch), dtype=int)
        for pv_idx in range(n_labels):
            for b in range(self._effective_batch):
                sel = self.param_viewer.index_selection[pv_idx][b]
                _access_field[pv_idx, b] = int(sel[0])
        self._access_field = _access_field

        self._label_to_pv_idx = {lbl: i for i, lbl in enumerate(all_labels)}

        # Mask: primary label uses the real mask; satellite (non-primary dim)
        # slots stay all-False so the additive default (0) fills them.
        self._gphistosys_mask = []
        for lbl in all_labels:
            bkey = label_to_key[lbl]
            is_primary = (label_to_dim[lbl] == 0)
            if is_primary:
                self._gphistosys_mask.append(
                    [[builder_data[bkey][s]['data']['mask']] for s in pdfconfig.samples]
                )
            else:
                n_bins = len(builder_data[bkey][pdfconfig.samples[0]]['data']['mask'])
                self._gphistosys_mask.append(
                    [[([False] * n_bins)] for _ in pdfconfig.samples]
                )

        self._precompute()
        events.subscribe('tensorlib_changed')(self._precompute)

    # ------------------------------------------------------------------
    def _precompute(self):
        if not self.param_viewer.index_selection:
            return
        tensorlib, _ = get_backend()

        self.gphistosys_mask = tensorlib.astensor(
            self._gphistosys_mask, dtype='bool'
        )
        # Additive identity: masked / satellite bins contribute 0 delta.
        self.gphistosys_default = tensorlib.zeros(
            tensorlib.shape(self.gphistosys_mask)
        )
        self.access_field = tensorlib.astensor(self._access_field, dtype='int')

        self._anchor_alphas_t = [
            tensorlib.astensor(d['anchor_alphas']) for d in self._gp_meta
        ]
        self._weights_t = [
            [tensorlib.astensor(w) for w in d['weights']] for d in self._gp_meta
        ]
        # Vectorized code4p interpolator per GP — returns additive deltas
        # of shape (d, n_samples, n_alphasets, n_bins).
        self._code4p_interps = [
            code4p(d['histogramssets'], subscribe=False)
            for d in self._gp_meta
        ]

    # ------------------------------------------------------------------
    def _gp_delta(self, alpha, anchor_alphas, weights, code4p_interp, s_idx, tensorlib):
        """GP posterior-mean additive delta with code4p HistFactory prior.

        delta_b(a) = m_b(a) + k(a,X) K^{-1} (r_b - m_b(X))

        Parameters
        ----------
        alpha         : (d,)
        anchor_alphas : (N, d)
        weights       : (N, n_bins)  — precomputed K^{-1}(r - m(X))
        code4p_interp : pyhf.interpolators.code4p instance
        s_idx         : int — sample index into code4p's sample axis

        Returns
        -------
        delta : (n_bins,)  — additive correction to the nominal yield
        """
        d = tensorlib.shape(alpha)[0]

        alphasets = tensorlib.reshape(alpha, (d, 1))
        # code4p returns additive deltas: (d, n_samples, 1, n_bins)
        all_deltas = code4p_interp(alphasets)

        # Additive prior: m(a) = sum_i delta_i
        prior = 0.0
        for i in range(d):
            prior = prior + all_deltas[i, s_idx, 0]

        diff = anchor_alphas - alpha
        sq_dist = tensorlib.einsum('nd,nd->n', diff, diff)
        K_eval = self.variance * tensorlib.exp(
            -0.5 * sq_dist / (self.length_scale ** 2)
        )
        residual = tensorlib.einsum('n,nb->b', K_eval, weights)
        return prior + residual

    # ------------------------------------------------------------------
    def apply(self, pars):
        """Compute additive correction deltas.

        Returns
        -------
        tensor : shape (n_labels, n_samples, batch_or_1, n_global_bins)

        Only the primary label (dim 0) of each GP carries a non-zero delta.
        Satellite labels return zeros (additive identity).
        """
        if not self.param_viewer.index_selection:
            return

        tensorlib, _ = get_backend()
        flat_pars = tensorlib.reshape(pars, (-1,))

        n_samples = len(self._weights_t[0]) if self._gp_meta else 0

        gp_deltas = []
        for gp_idx, gp_labels in enumerate(self._key_labels):
            batch_deltas = []
            for b in range(self._effective_batch):
                alpha_components = [
                    tensorlib.gather(flat_pars,
                        tensorlib.astensor(
                            [self._access_field[self._label_to_pv_idx[lbl], b]],
                            dtype='int'
                        )
                    )[0]
                    for lbl in gp_labels
                ]
                alpha = tensorlib.stack(alpha_components)

                sample_deltas = [
                    self._gp_delta(
                        alpha,
                        self._anchor_alphas_t[gp_idx],
                        self._weights_t[gp_idx][s_idx],
                        self._code4p_interps[gp_idx],
                        s_idx,
                        tensorlib,
                    )
                    for s_idx in range(n_samples)
                ]
                batch_deltas.append(sample_deltas)
            gp_deltas.append(batch_deltas)

        # Build output tensor (n_labels, n_samples, batch, n_bins).
        results_by_label = []
        for lbl, pv_idx in self._label_to_pv_idx.items():
            gp_idx = next(
                i for i, labels in enumerate(self._key_labels) if lbl in labels
            )
            dim_idx = self._key_labels[gp_idx].index(lbl)

            if dim_idx == 0:
                batch_stack = tensorlib.stack(
                    [tensorlib.stack(gp_deltas[gp_idx][b])
                     for b in range(self._effective_batch)]
                )
                results_by_label.append(
                    tensorlib.einsum('bsn->sbn', batch_stack)
                )
            else:
                # Satellite: zeros (additive no-op) with the same shape.
                results_by_label.append(
                    tensorlib.zeros(
                        tensorlib.shape(results_by_label[
                            self._label_to_pv_idx[self._key_labels[gp_idx][0]]
                        ])
                    )
                )

        results = tensorlib.stack(results_by_label)
        results = tensorlib.where(
            self.gphistosys_mask, results, self.gphistosys_default
        )
        return results
