"""Custom pyhf modifier: Gaussian Process interpolation.

The modifier uses GP posterior-mean regression to interpolate yield corrections across an arbitrary-dimensional nuisance-parameter space.

GP posterior mean for bin b
---------------------------
    ratio_b(a) = 1 + k(a, A) K^{-1} (r_b - 1)

where
    A          = anchor nodes  (N x d)
    r_b        = templates[:, b] / nominal_b   (N-vector of ratios at anchors)
    K(a, A)    = [k(a, a_1), ..., k(a, a_N)]  (row-vector of kernel values)
    K          = K(A, A) + noise^2 I           (N x N training kernel matrix)

Squared-exponential (RBF) kernel:
    k(a, a') = variance * exp(-||a - a'||^2 / (2 * length_scale^2))

Multiplicative kappa:
    kappa_b(a) = 1.0 + K(a, A) K^{-1} (r_b - 1)

Spec format example
-------------------
{
    "name": "alpha",
    "type": "gphistosys",
    "data": {
        "nodes":     [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0 , -1.0]],
        "templates": [
            [t0_bin0, t0_bin1, ...],
            ...
        ],
        "labels": ["jet_energy", "b_tagging"]
    }
}

For d=1 the "labels" field is optional and defaults to [modifier_name]:
{
    "name": "alpha",
    "type": "gphistosys",
    "data": {
        "nodes":     [[-1.0], [0.0], [1.0]],
        "templates": [[...], [...], [...]]
    }
}

Typical sample structure
------------------------
{
    "name":  "background",
    "data":  [100.0, 150.0],
    "modifiers": [
        {"name": "alpha", "type": "gphistosys", "data": {...}},
        {"name": "mu",     "type": "normfactor",  "data": null}
    ]
}

Implementation detail
---------------------

The challenge here was that we do interpolation with a vector alpha = (alpha_1, ... alpha_d), but each alpha_i is an independent parameter. But pyhf's parameter system indexes parameters by ``name``.  A d-dimensional GP input vector therefore cannot be registered as a single named entity and still expose its individual dimensions to pull plots, fit tables, and profile scans.

The workaround proposed here is that each dimension of the GP input vector is exposed as an individually named, scalar constrained_by_normal parameter via the ``labels`` field in the spec.  For a d-dimensional GP the spec carries one modifier entry with d labels; pyhf registers d separate scalar parameters under those names.

    labels[0]  ->  pyhf parameter "alpha_1"    (dim 0, carries the kappa)
    labels[1]  ->  pyhf parameter "alpha_2"    (dim 1, returns one)
    ...
    labels[d]  ->  pyhf parameter "alpha_d"    (dim 1, returns one)

The combined class reassembles the d scalars put into the labels field into the full alpha vector before evaluating the GP kernel.

For d=1, `labels` defaults to ``[modifier_name]`` — existing 1D specs work without change.
"""

import logging

import numpy as np
import pyhf
from pyhf import get_backend, events
from pyhf.parameters import ParamViewer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sq_exp_kernel_np(A, B, length_scale, variance):
    """Squared-exponential kernel (numpy).  A: (N1, d), B: (N2, d) -> (N1, N2)."""
    diff = A[:, None, :] - B[None, :, :]
    sq_dist = np.sum(diff * diff, axis=-1)
    return variance * np.exp(-0.5 * sq_dist / (length_scale ** 2))


def _get_labels(modifier_name, modifier_data):
    """Return per-dimension parameter labels from spec, defaulting to [name]."""
    return list(modifier_data.get('labels', [modifier_name]))


def required_parset(sample_data, modifier_data):
    """Return a scalar constrained_by_normal parset for a single GP dimension."""
    return {
        'paramset_type': 'constrained_by_normal',
        'n_parameters': 1,
        'is_scalar': True,
        'inits': (0.0,),
        'bounds': ((-5.0, 5.0),),
        'fixed': False,
        'auxdata': (0.0,),
    }


class gphistosys_builder:
    """Builder class for collecting gphistosys modifier data."""

    is_shared = True

    def __init__(self, config):
        self.builder_data = {}
        self.config = config
        self.required_parsets = {}

    # ------------------------------------------------------------------
    def collect(self, thismod, nom):
        if thismod:
            nodes = thismod['data']['nodes']
            templates = np.array(thismod['data']['templates'], dtype=float)
            maskval = True
        else:
            nodes = None
            templates = None
            maskval = False
        return {
            'nodes': nodes,
            'templates': templates,
            'nom_data': list(nom),
            'mask': [maskval] * len(nom),
        }

    # ------------------------------------------------------------------
    def append(self, key, channel, sample, thismod, defined_samp):
        self.builder_data.setdefault(key, {}).setdefault(sample, {}).setdefault(
            'data',
            {
                'channel_templates': [],
                'nom_data': [],
                'mask': [],
            },
        )
        nom = (
            defined_samp['data']
            if defined_samp
            else [0.0] * self.config.channel_nbins[channel]
        )
        moddata = self.collect(thismod, nom)

        self.builder_data[key][sample]['data']['channel_templates'].append(
            moddata['templates']
        )
        self.builder_data[key][sample]['data']['nom_data'].append(moddata['nom_data'])
        self.builder_data[key][sample]['data']['mask'].append(moddata['mask'])

        if moddata['nodes'] is not None:
            self.builder_data[key].setdefault('__nodes__', moddata['nodes'])

        if thismod:
            # Derive per-dimension parameter labels from the spec.
            # Store them on the builder_data so the combined class can find them.
            labels = _get_labels(thismod['name'], thismod['data'])
            self.builder_data[key].setdefault('__labels__', labels)

            # Register one scalar constrained_by_normal parameter per label.
            for label in labels:
                self.required_parsets.setdefault(
                    label,
                    [
                        required_parset(
                            defined_samp['data'] if defined_samp else [],
                            thismod['data'],
                        )
                    ],
                )

    # ------------------------------------------------------------------
    def finalize(self):
        for key in list(self.builder_data.keys()):
            nodes = self.builder_data[key].get('__nodes__')
            if nodes is None:
                log.warning(
                    'gphistosys modifier %s has no anchor nodes; using dummy node.', key
                )
                nodes = [[0.0]]
            nodes_arr = np.array(nodes, dtype=float)  # (N, d)

            # Ensure the origin node (alpha=0) is present
            zero_node = np.zeros(nodes_arr.shape[1], dtype=float)
            insert_origin = not any(
                np.allclose(nodes_arr[i], zero_node) for i in range(len(nodes_arr))
            )
            if insert_origin:
                nodes_arr = np.vstack([zero_node[None, :], nodes_arr])
            N = len(nodes_arr)

            for sample_name, sample in self.builder_data[key].items():
                if sample_name.startswith('__'):
                    continue
                data = sample['data']

                nom_flat = np.concatenate(
                    [np.array(n, dtype=float) for n in data['nom_data']]
                )
                mask_flat = list(
                    np.concatenate([np.array(m) for m in data['mask']])
                )
                n_total_bins = len(nom_flat)

                full_templates = np.zeros((N, n_total_bins), dtype=float)
                ptr = 0
                # If origin was inserted, reserve row 0 for nom; user nodes start at 1
                row_offset = 1 if insert_origin else 0
                if insert_origin:
                    full_templates[0, :] = nom_flat
                for ct, nm in zip(data['channel_templates'], data['nom_data']):
                    n_bins_c = len(nm)
                    if ct is not None:
                        full_templates[row_offset:, ptr : ptr + n_bins_c] = ct
                    else:
                        nom_c = np.array(nm, dtype=float)
                        full_templates[row_offset:, ptr : ptr + n_bins_c] = nom_c[None, :]
                    ptr += n_bins_c

                sample['data'] = {
                    'nodes': nodes_arr,
                    'templates': full_templates,
                    'nom_data': nom_flat,
                    'mask': mask_flat,
                }

        return self.builder_data


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------


class gphistosys_combined:
    """Combined modifier class for GP-based histogram interpolation.

    The `labels` field in the spec maps each GP dimension to a named
    scalar pyhf parameter.  The combined class receives all labeled
    parameters (e.g. "jet_energy", "b_tagging"), maps them back to
    their source modifier via ``builder_data[key]['__labels__']``, assembles
    the full alpha vector, and evaluates the GP.

    Parameters
    ----------
    modifiers    : list of (spec_name, 'gphistosys') tuples — spec modifier names
    pdfconfig    : pyhf model config
    builder_data : output of gphistosys_builder.finalize()
    length_scale : GP kernel length-scale
    variance     : GP kernel output variance
    noise        : diagonal jitter added to K_train
    batch_size   : vectorised batch dimension, or None
    """

    name    = 'gphistosys'
    op_code = 'multiplication'

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

        # Derive the per-dimension label names from builder_data.__labels__,
        # which the builder stored when it processed the spec.
        keys = [f'{mtype}/{m}' for m, mtype in modifiers]

        # For each spec key, read its labels; build a flat ordered list.
        # e.g. key_to_labels["gphistosys/alpha"] = ["alpha1", "alpha2"]
        key_to_labels = {}
        for key in keys:
            spec_name = key.split('/')[1]
            key_to_labels[key] = builder_data[key].get('__labels__', [spec_name])

        # Flat list of all label parameter names in key order.
        all_labels = [lbl for key in keys for lbl in key_to_labels[key]]

        # Reverse maps for use in apply().
        label_to_key = {lbl: key for key in keys for lbl in key_to_labels[key]}
        label_to_dim = {lbl: i for key in keys
                        for i, lbl in enumerate(key_to_labels[key])}

        self._effective_batch = self.batch_size or 1
        parfield_shape = (self._effective_batch, pdfconfig.npars)
        self.param_viewer = ParamViewer(
            parfield_shape, pdfconfig.par_map, all_labels
        )

        # ------------------------------------------------------------------
        # Precompute GP weights (done once at build time).
        # ------------------------------------------------------------------
        self._gp_meta           = [] # one entry per unique GP (builder key)
        self._key_labels        = [] # ordered labels per GP

        for bkey in keys:
            labels_for_key = key_to_labels[bkey]
            self._key_labels.append(labels_for_key)

            first_sample = pdfconfig.samples[0]
            anchor_alphas = np.array(
                builder_data[bkey][first_sample]['data']['nodes'], dtype=float
            )  # (N, d)
            N = len(anchor_alphas)

            K_train = _sq_exp_kernel_np(
                anchor_alphas, anchor_alphas, self.length_scale, self.variance
            )
            K_train += (self.noise ** 2) * np.eye(N)
            K_train_inv = np.linalg.inv(K_train)

            weights_per_sample = []
            for s in pdfconfig.samples:
                sdata = builder_data[bkey][s]['data']
                nom = np.array(sdata['nom_data'], dtype=float)
                templates = np.array(sdata['templates'], dtype=float)
                with np.errstate(divide='ignore', invalid='ignore'):
                    anchor_ratios = np.where(
                        nom[None, :] != 0,
                        templates / nom[None, :],
                        np.ones_like(templates),
                    )
                weights_per_sample.append(K_train_inv @ (anchor_ratios - 1.0))

            self._gp_meta.append(
                {
                    'anchor_alphas': anchor_alphas,
                    'weights': weights_per_sample,
                }
            )

        # ------------------------------------------------------------------
        # access_field[label_pv_idx, b] = flat par index for that label, batch b.
        # Stored as a 2-D array; we index it per-label during apply().
        # ------------------------------------------------------------------
        n_labels = len(all_labels)
        _access_field = np.zeros((n_labels, self._effective_batch), dtype=int)
        for pv_idx in range(n_labels):
            for b in range(self._effective_batch):
                sel = self.param_viewer.index_selection[pv_idx][b]
                _access_field[pv_idx, b] = int(sel[0])
        self._access_field = _access_field

        # Map label -> pv_idx for fast lookup in apply().
        self._label_to_pv_idx = {lbl: i for i, lbl in enumerate(all_labels)}

        # Mask: one entry per incoming label (n_labels, n_samples, 1, n_bins).
        # Primary label uses the actual mask; satellites use all-False.
        self._gphistosys_mask = []
        for lbl in all_labels:
            bkey = label_to_key[lbl]
            is_primary = (label_to_dim[lbl] == 0)
            if is_primary:
                self._gphistosys_mask.append(
                    [[builder_data[bkey][s]['data']['mask']] for s in pdfconfig.samples]
                )
            else:
                # Satellite slots always output ones — use all-False mask.
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
        self.gphistosys_default = tensorlib.ones(
            tensorlib.shape(self.gphistosys_mask)
        )
        self.access_field = tensorlib.astensor(self._access_field, dtype='int')

        self._anchor_alphas_t = [
            tensorlib.astensor(d['anchor_alphas']) for d in self._gp_meta
        ]
        self._weights_t = [
            [tensorlib.astensor(w) for w in d['weights']] for d in self._gp_meta
        ]

    # ------------------------------------------------------------------
    def _gp_factor(self, alpha, anchor_alphas, weights, tensorlib):
        """GP posterior-mean multiplicative factor for (GP, sample, batch).

        Parameters
        ----------
        alpha        : (d,)
        anchor_alphas: (N, d)
        weights      : (N, n_bins)  — precomputed K^{-1}(r - 1)

        Returns
        -------
        factor : (n_bins,)
        """
        diff = anchor_alphas - alpha
        sq_dist = tensorlib.einsum('nd,nd->n', diff, diff)
        K_eval = self.variance * tensorlib.exp(
            -0.5 * sq_dist / (self.length_scale ** 2)
        )
        delta_ratio = tensorlib.einsum('n,nb->b', K_eval, weights)
        return 1.0 + delta_ratio

    # ------------------------------------------------------------------
    def apply(self, pars):
        """Compute multiplicative correction factors.

        Returns
        -------
        tensor : shape (n_labels, n_samples, batch_or_1, n_global_bins)

        Only the primary label (dim 0) of each GP carries a non-ones kappa.
        Satellite labels return ones.
        """
        if not self.param_viewer.index_selection:
            return

        tensorlib, _ = get_backend()
        flat_pars = tensorlib.reshape(pars, (-1,))

        n_samples = len(self._weights_t[0]) if self._gp_meta else 0

        # Precompute GP delta for each unique GP × batch × sample.
        # gp_deltas[gp_idx][b_idx][s_idx] = (n_bins,) tensor
        gp_deltas = []
        for gp_idx, gp_labels in enumerate(self._key_labels):
            batch_deltas = []
            for b in range(self._effective_batch):
                # Assemble d-dimensional alpha vector from individual scalars.
                alpha_components = [
                    tensorlib.gather(flat_pars,
                        tensorlib.astensor(
                            [self._access_field[self._label_to_pv_idx[lbl], b]],
                            dtype='int'
                        )
                    )[0]
                    for lbl in gp_labels
                ]
                alpha = tensorlib.stack(alpha_components)  # (d,)

                sample_deltas = [
                    self._gp_factor(
                        alpha,
                        self._anchor_alphas_t[gp_idx],
                        self._weights_t[gp_idx][s_idx],
                        tensorlib,
                    )
                    for s_idx in range(n_samples)
                ]
                batch_deltas.append(sample_deltas)
            gp_deltas.append(batch_deltas)

        # Build output tensor (n_labels, n_samples, batch, n_bins).
        # Primary label (dim 0) carries the GP delta; satellites carry zeros.
        results_by_label = []
        for lbl, pv_idx in self._label_to_pv_idx.items():
            gp_idx = next(
                i for i, labels in enumerate(self._key_labels) if lbl in labels
            )
            dim_idx = self._key_labels[gp_idx].index(lbl)

            if dim_idx == 0:
                # Primary: stack (batch, n_samples, n_bins) -> (n_samples, batch, n_bins)
                batch_stack = tensorlib.stack(
                    [tensorlib.stack(gp_deltas[gp_idx][b]) for b in range(self._effective_batch)]
                )  # (batch, n_samples, n_bins)
                results_by_label.append(
                    tensorlib.einsum('bsn->sbn', batch_stack)
                )  # (n_samples, batch, n_bins)
            else:
                # Satellite: ones with same shape.
                results_by_label.append(
                    tensorlib.ones(
                        tensorlib.shape(results_by_label[
                            self._label_to_pv_idx[self._key_labels[gp_idx][0]]
                        ])
                    )
                )

        # Stack -> (n_labels, n_samples, batch, n_bins)
        results = tensorlib.stack(results_by_label)
        results = tensorlib.where(
            self.gphistosys_mask, results, self.gphistosys_default
        )
        return results
