"""Custom pyhf modifier: Gaussian Process interpolation.

The modifier uses GP posterior-mean regression to interpolate yield
corrections across an arbitrary-dimensional nuisance-parameter space.

The alpha parameter is a d-dimensional vector whose coordinates index the
GP input space.  At each anchor node alpha_i the user supplies a template
(bin yields for this sample).  The GP posterior mean predicts the
per-bin yield ratio at any query alpha, and the resulting *additive* delta
w.r.t. the nominal is returned.

GP posterior mean for bin b
---------------------------
    ratio_b(alpha) = 1 + k(alpha, X) K^{-1} (r_b - 1)

where
    X          = anchor nodes  (N x d)
    r_b        = templates[:, b] / nominal_b   (N-vector of ratios at anchors)
    k(a, X)    = [k(a, x_1), ..., k(a, x_N)]  (row-vector of kernel values)
    K          = k(X, X) + noise^2 I           (N x N training kernel matrix)

The squared-exponential (RBF) kernel is used:
    k(a, a') = variance * exp(-||a - a'||^2 / (2 * length_scale^2))

The additive delta returned by apply() is
    delta_b(alpha) = nominal_b * (ratio_b(alpha) - 1)
                   = nominal_b * k(alpha, X) K^{-1} (r_b - 1)

Spec format
-----------
{
    "name": "alpha",
    "type": "gphistosys",
    "data": {
        "nodes":     [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        "templates": [
            [t0_bin0, t0_bin1, ...],
            [t1_bin0, t1_bin1, ...],
            [t2_bin0, t2_bin1, ...]
        ]
    }
}

Typical sample structure
------------------------
{
    "name":  "process",
    "data":  [5.0, 9.0, ...],
    "modifiers": [
        {"name": "alpha",  "type": "gphistosys", "data": {...}},
        {"name": "mu",     "type": "normfactor",  "data": null}
    ]
}
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
    diff = A[:, None, :] - B[None, :, :]       # (N1, N2, d)
    sq_dist = np.sum(diff * diff, axis=-1)      # (N1, N2)
    return variance * np.exp(-0.5 * sq_dist / (length_scale ** 2))


# ---------------------------------------------------------------------------
# required_parset
# ---------------------------------------------------------------------------


def required_parset(sample_data, modifier_data):
    """Return parameter-set specification for one gphistosys modifier."""
    d = len(modifier_data['nodes'][0])
    return {
        'paramset_type': 'constrained_by_normal',
        'n_parameters': d,
        'is_scalar': False,
        'inits': (0.0,) * d,
        'bounds': ((-5.0, 5.0),) * d,
        'fixed': False,
        'auxdata': (0.0,) * d,
    }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


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
            # templates shape: (N, n_channel_bins)
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
                'channel_templates': [],   # list (one per channel) of (N, n_bins_c) or None
                'nom_data': [],            # list (one per channel) of [float, ...]
                'mask': [],               # list (one per channel) of [bool, ...]
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

        # Store nodes at modifier level (identical across samples / channels)
        if moddata['nodes'] is not None:
            self.builder_data[key].setdefault('__nodes__', moddata['nodes'])

        if thismod:
            self.required_parsets.setdefault(
                thismod['name'],
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
            N = len(nodes_arr)

            for sample_name, sample in self.builder_data[key].items():
                if sample_name == '__nodes__':
                    continue
                data = sample['data']

                # Concatenate nom_data and mask across channels  → (n_global_bins,)
                nom_flat = np.concatenate(
                    [np.array(n, dtype=float) for n in data['nom_data']]
                )
                mask_flat = list(
                    np.concatenate([np.array(m) for m in data['mask']])
                )
                n_total_bins = len(nom_flat)

                # Build full templates (N, n_total_bins)
                # Channels where modifier is not applied get ratio=1 (templates=nom)
                full_templates = np.zeros((N, n_total_bins), dtype=float)
                ptr = 0
                for ct, nm in zip(data['channel_templates'], data['nom_data']):
                    n_bins_c = len(nm)
                    if ct is not None:
                        full_templates[:, ptr : ptr + n_bins_c] = ct
                    else:
                        # ratio = 1 at every node → templates equal nominal
                        nom_c = np.array(nm, dtype=float)
                        full_templates[:, ptr : ptr + n_bins_c] = nom_c[None, :]
                    ptr += n_bins_c

                sample['data'] = {
                    'nodes': nodes_arr,          # (N, d)
                    'templates': full_templates, # (N, n_total_bins)
                    'nom_data': nom_flat,        # (n_total_bins,)
                    'mask': mask_flat,           # (n_total_bins,) bool
                }

        return self.builder_data


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------


class gphistosys_combined:
    """Combined modifier class for GP-based histogram interpolation.

    Parameters
    ----------
    modifiers : list of (name, type) tuples
    pdfconfig : pyhf model config
    builder_data : output of gphistosys_builder.finalize()
    length_scale : GP kernel length-scale (scalar, same for all dimensions)
    variance : GP kernel output variance
    noise : observation noise added to the diagonal of K_train
    batch_size : vectorised batch dimension, or None
    """

    name = 'gphistosys'
    op_code = 'addition'

    def __init__(
        self,
        modifiers,
        pdfconfig,
        builder_data,
        length_scale=1.0,
        variance=1.0,
        noise=1e-6,
        batch_size=None,
    ):
        self.batch_size = batch_size
        self.length_scale = float(length_scale)
        self.variance = float(variance)
        self.noise = float(noise)

        keys = [f'{mtype}/{m}' for m, mtype in modifiers]
        gphistosys_mods = [m for m, _ in modifiers]

        # Always use 2-D parfield (batch_or_1, npars) so index_selection[m][b]
        # is always well-defined.
        self._effective_batch = self.batch_size or 1
        parfield_shape = (self._effective_batch, pdfconfig.npars)
        self.param_viewer = ParamViewer(
            parfield_shape, pdfconfig.par_map, gphistosys_mods
        )

        # ------------------------------------------------------------------
        # Precompute GP weights using numpy (done once at build time)
        # ------------------------------------------------------------------
        # _gp_meta[m_idx] = {
        #     'anchor_alphas': (N, d)  ndarray,
        #     'weights':       list[s] of (N, n_global_bins) ndarray,
        #     'nom_data':      list[s] of (n_global_bins,) ndarray,
        # }
        self._gp_meta = []
        for m in keys:
            # nodes are the same for all samples; grab from the first sample
            first_sample = pdfconfig.samples[0]
            anchor_alphas = np.array(
                builder_data[m][first_sample]['data']['nodes'], dtype=float
            )  # (N, d)
            N = len(anchor_alphas)

            K_train = _sq_exp_kernel_np(
                anchor_alphas, anchor_alphas, self.length_scale, self.variance
            )
            K_train += (self.noise ** 2) * np.eye(N)
            K_train_inv = np.linalg.inv(K_train)

            weights_per_sample = []
            nom_per_sample = []
            for s in pdfconfig.samples:
                sdata = builder_data[m][s]['data']
                nom = np.array(sdata['nom_data'], dtype=float)       # (n_bins,)
                templates = np.array(sdata['templates'], dtype=float) # (N, n_bins)
                with np.errstate(divide='ignore', invalid='ignore'):
                    anchor_ratios = np.where(
                        nom[None, :] != 0,
                        templates / nom[None, :],
                        np.ones_like(templates),
                    )  # (N, n_bins)
                # weights: K_inv @ (r - 1), shape (N, n_bins)
                weights = K_train_inv @ (anchor_ratios - 1.0)
                weights_per_sample.append(weights)
                nom_per_sample.append(nom)

            self._gp_meta.append(
                {
                    'anchor_alphas': anchor_alphas,
                    'weights': weights_per_sample,
                    'nom_data': nom_per_sample,
                }
            )

        # ------------------------------------------------------------------
        # Build access field for alpha extraction  (n_mods, batch_or_1, d)
        # ------------------------------------------------------------------
        # access_field[m, b, i] = flat index into reshape(pars, (-1,)) for
        # coordinate i of modifier m in batch item b.
        n_mods = len(gphistosys_mods)
        d_per_mod = [
            len(self.param_viewer.index_selection[m_idx][0])
            for m_idx in range(n_mods)
        ] if n_mods > 0 else []
        self._d_per_mod = d_per_mod
        max_d = max(d_per_mod) if d_per_mod else 1

        _access_field = np.zeros((n_mods, self._effective_batch, max_d), dtype=int)
        for m_idx in range(n_mods):
            for b in range(self._effective_batch):
                sel = self.param_viewer.index_selection[m_idx][b]
                _access_field[m_idx, b, : d_per_mod[m_idx]] = np.asarray(sel)
        self._access_field = _access_field  # keep as numpy for _precompute

        # Mask tensor: (n_mods, n_samples, 1_or_batch, n_global_bins)
        self._gphistosys_mask = [
            [[builder_data[m][s]['data']['mask']] for s in pdfconfig.samples]
            for m in keys
        ]

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
        self.gphistosys_default = tensorlib.zeros(
            tensorlib.shape(self.gphistosys_mask)
        )
        self.access_field = tensorlib.astensor(self._access_field, dtype='int')

        # Convert GP numpy arrays to current backend tensors
        self._anchor_alphas_t = [
            tensorlib.astensor(d['anchor_alphas'])
            for d in self._gp_meta
        ]  # list[m] of (N, d)

        self._weights_t = [
            [tensorlib.astensor(w) for w in d['weights']]
            for d in self._gp_meta
        ]  # list[m][s] of (N, n_global_bins)

        self._nom_data_t = [
            [tensorlib.astensor(n) for n in d['nom_data']]
            for d in self._gp_meta
        ]  # list[m][s] of (n_global_bins,)

    # ------------------------------------------------------------------
    def _gp_delta(self, alpha, anchor_alphas, weights, nom, tensorlib):
        """GP posterior-mean additive delta for one (modifier, sample) pair.

        Parameters
        ----------
        alpha        : (d,) — current nuisance parameter value
        anchor_alphas: (N, d) — fixed training nodes
        weights      : (N, n_bins) — precomputed K^{-1}(r - 1)
        nom          : (n_bins,) — nominal bin yields
        tensorlib    : active pyhf backend

        Returns
        -------
        delta : (n_bins,)
        """
        # diff[n, i] = anchor_alphas[n, i] - alpha[i],  shape (N, d)
        diff = anchor_alphas - alpha            # broadcast (N, d) - (d,)
        sq_dist = tensorlib.einsum('nd,nd->n', diff, diff)  # (N,)
        K_eval = self.variance * tensorlib.exp(
            -0.5 * sq_dist / (self.length_scale ** 2)
        )  # (N,)
        # delta_ratio = k^T K^{-1} (r - 1):  (n_bins,)
        delta_ratio = tensorlib.einsum('n,nb->b', K_eval, weights)
        return nom * delta_ratio  # (n_bins,)

    # ------------------------------------------------------------------
    def apply(self, pars):
        """Compute additive yield deltas for all modifiers and samples.

        Returns
        -------
        tensor : shape (n_modifiers, n_global_samples, n_alphas, n_global_bin)
        """
        if not self.param_viewer.index_selection:
            return

        tensorlib, _ = get_backend()

        n_mods = len(self._gp_meta)
        n_samples = len(self._weights_t[0]) if n_mods > 0 else 0
        effective_batch = self._effective_batch

        # Extract all alpha vectors at once:
        #   alpha_all = gather(flat_pars, access_field)
        #   shape: (n_mods, effective_batch, max_d)
        flat_pars = tensorlib.reshape(pars, (-1,))
        alpha_all = tensorlib.gather(flat_pars, self.access_field)
        # alpha_all[m, b, :d_m] = alpha for modifier m, batch item b

        # Build result tensor  (n_mods, n_samples, effective_batch, n_global_bins)
        results_mod = []
        for m_idx in range(n_mods):
            d_m = self._d_per_mod[m_idx]

            results_batch = []
            for b_idx in range(effective_batch):
                alpha = alpha_all[m_idx, b_idx, :d_m]  # (d_m,)

                results_sample = []
                for s_idx in range(n_samples):
                    delta = self._gp_delta(
                        alpha,
                        self._anchor_alphas_t[m_idx],
                        self._weights_t[m_idx][s_idx],
                        self._nom_data_t[m_idx][s_idx],
                        tensorlib,
                    )  # (n_global_bins,)
                    results_sample.append(delta)

                # (n_samples, n_global_bins)
                results_batch.append(tensorlib.stack(results_sample))

            # (effective_batch, n_samples, n_global_bins)
            stacked_batch = tensorlib.stack(results_batch)
            # → (n_samples, effective_batch, n_global_bins)
            results_mod.append(
                tensorlib.einsum('bsn->sbn', stacked_batch)
            )

        # (n_mods, n_samples, effective_batch, n_global_bins)
        results = tensorlib.stack(results_mod)

        results = tensorlib.where(
            self.gphistosys_mask, results, self.gphistosys_default
        )
        return results
