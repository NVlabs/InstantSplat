# from internal import math
import numpy as np
import torch


def searchsorted(a, v):
    """Find indices where v should be inserted into a to maintain order.

  Args:
    a: tensor, the sorted reference points that we are scanning to see where v
      should lie.
    v: tensor, the query points that we are pretending to insert into a. Does
      not need to be sorted. All but the last dimensions should match or expand
      to those of a, the last dimension can differ.

  Returns:
    (idx_lo, idx_hi), where a[idx_lo] <= v < a[idx_hi], unless v is out of the
    range [a[0], a[-1]] in which case idx_lo and idx_hi are both the first or
    last index of a.
  """
    i = torch.arange(a.shape[-1], device=a.device)
    v_ge_a = v[..., None, :] >= a[..., :, None]
    idx_lo = torch.max(torch.where(v_ge_a, i[..., :, None], i[..., :1, None]), -2).values
    idx_hi = torch.min(torch.where(~v_ge_a, i[..., :, None], i[..., -1:, None]), -2).values
    return idx_lo, idx_hi


def query(tq, t, y, outside_value=0):
    """Look up the values of the step function (t, y) at locations tq."""
    idx_lo, idx_hi = searchsorted(t, tq)
    yq = torch.where(idx_lo == idx_hi, torch.full_like(idx_hi, outside_value),
                     torch.take_along_dim(y, idx_lo, dim=-1))
    return yq


def inner_outer(t0, t1, y1):
    """Construct inner and outer measures on (t1, y1) for t0."""
    cy1 = torch.cat([torch.zeros_like(y1[..., :1]),
                     torch.cumsum(y1, dim=-1)],
                    dim=-1)
    idx_lo, idx_hi = searchsorted(t1, t0)

    cy1_lo = torch.take_along_dim(cy1, idx_lo, dim=-1)
    cy1_hi = torch.take_along_dim(cy1, idx_hi, dim=-1)

    y0_outer = cy1_hi[..., 1:] - cy1_lo[..., :-1]
    y0_inner = torch.where(idx_hi[..., :-1] <= idx_lo[..., 1:],
                           cy1_lo[..., 1:] - cy1_hi[..., :-1], torch.zeros_like(idx_lo[..., 1:]))
    return y0_inner, y0_outer


def lossfun_outer(t, w, t_env, w_env):
    """The proposal weight should be an upper envelope on the nerf weight."""
    eps = torch.finfo(t.dtype).eps
    # eps = 1e-3

    _, w_outer = inner_outer(t, t_env, w_env)
    # We assume w_inner <= w <= w_outer. We don't penalize w_inner because it's
    # more effective to pull w_outer up than it is to push w_inner down.
    # Scaled half-quadratic loss that gives a constant gradient at w_outer = 0.
    return (w - w_outer).clamp_min(0) ** 2 / (w + eps)


def weight_to_pdf(t, w):
    """Turn a vector of weights that sums to 1 into a PDF that integrates to 1."""
    eps = torch.finfo(t.dtype).eps
    return w / (t[..., 1:] - t[..., :-1]).clamp_min(eps)


def pdf_to_weight(t, p):
    """Turn a PDF that integrates to 1 into a vector of weights that sums to 1."""
    return p * (t[..., 1:] - t[..., :-1])


def max_dilate(t, w, dilation, domain=(-torch.inf, torch.inf)):
    """Dilate (via max-pooling) a non-negative step function."""
    t0 = t[..., :-1] - dilation
    t1 = t[..., 1:] + dilation
    t_dilate, _ = torch.sort(torch.cat([t, t0, t1], dim=-1), dim=-1)
    t_dilate = torch.clip(t_dilate, *domain)
    w_dilate = torch.max(
        torch.where(
            (t0[..., None, :] <= t_dilate[..., None])
            & (t1[..., None, :] > t_dilate[..., None]),
            w[..., None, :],
            torch.zeros_like(w[..., None, :]),
        ), dim=-1).values[..., :-1]
    return t_dilate, w_dilate


def max_dilate_weights(t,
                       w,
                       dilation,
                       domain=(-torch.inf, torch.inf),
                       renormalize=False):
    """Dilate (via max-pooling) a set of weights."""
    eps = torch.finfo(w.dtype).eps
    # eps = 1e-3

    p = weight_to_pdf(t, w)
    t_dilate, p_dilate = max_dilate(t, p, dilation, domain=domain)
    w_dilate = pdf_to_weight(t_dilate, p_dilate)
    if renormalize:
        w_dilate /= torch.sum(w_dilate, dim=-1, keepdim=True).clamp_min(eps)
    return t_dilate, w_dilate


def integrate_weights(w):
    """Compute the cumulative sum of w, assuming all weight vectors sum to 1.

  The output's size on the last dimension is one greater than that of the input,
  because we're computing the integral corresponding to the endpoints of a step
  function, not the integral of the interior/bin values.

  Args:
    w: Tensor, which will be integrated along the last axis. This is assumed to
      sum to 1 along the last axis, and this function will (silently) break if
      that is not the case.

  Returns:
    cw0: Tensor, the integral of w, where cw0[..., 0] = 0 and cw0[..., -1] = 1
  """
    cw = torch.cumsum(w[..., :-1], dim=-1).clamp_max(1)
    shape = cw.shape[:-1] + (1,)
    # Ensure that the CDF starts with exactly 0 and ends with exactly 1.
    cw0 = torch.cat([torch.zeros(shape, device=cw.device), cw,
                     torch.ones(shape, device=cw.device)], dim=-1)
    return cw0


def integrate_weights_np(w):
    """Compute the cumulative sum of w, assuming all weight vectors sum to 1.

  The output's size on the last dimension is one greater than that of the input,
  because we're computing the integral corresponding to the endpoints of a step
  function, not the integral of the interior/bin values.

  Args:
    w: Tensor, which will be integrated along the last axis. This is assumed to
      sum to 1 along the last axis, and this function will (silently) break if
      that is not the case.

  Returns:
    cw0: Tensor, the integral of w, where cw0[..., 0] = 0 and cw0[..., -1] = 1
  """
    cw = np.minimum(1, np.cumsum(w[..., :-1], axis=-1))
    shape = cw.shape[:-1] + (1,)
    # Ensure that the CDF starts with exactly 0 and ends with exactly 1.
    cw0 = np.concatenate([np.zeros(shape), cw,
                          np.ones(shape)], axis=-1)
    return cw0


def invert_cdf(u, t, w_logits):
    """Invert the CDF defined by (t, w) at the points specified by u in [0, 1)."""
    # Compute the PDF and CDF for each weight vector.
    w = torch.softmax(w_logits, dim=-1)
    cw = integrate_weights(w)
    # Interpolate into the inverse CDF.
    t_new = math.sorted_interp(u, cw, t)
    return t_new


def invert_cdf_np(u, t, w_logits):
    """Invert the CDF defined by (t, w) at the points specified by u in [0, 1)."""
    # Compute the PDF and CDF for each weight vector.
    w = np.exp(w_logits) / np.exp(w_logits).sum(axis=-1, keepdims=True)
    cw = integrate_weights_np(w)
    # Interpolate into the inverse CDF.
    interp_fn = np.interp
    t_new = interp_fn(u, cw, t)
    return t_new


def sample(rand,
           t,
           w_logits,
           num_samples,
           single_jitter=False,
           deterministic_center=False):
    """Piecewise-Constant PDF sampling from a step function.

  Args:
    rand: random number generator (or None for `linspace` sampling).
    t: [..., num_bins + 1], bin endpoint coordinates (must be sorted)
    w_logits: [..., num_bins], logits corresponding to bin weights
    num_samples: int, the number of samples.
    single_jitter: bool, if True, jitter every sample along each ray by the same
      amount in the inverse CDF. Otherwise, jitter each sample independently.
    deterministic_center: bool, if False, when `rand` is None return samples that
      linspace the entire PDF. If True, skip the front and back of the linspace
      so that the centers of each PDF interval are returned.

  Returns:
    t_samples: [batch_size, num_samples].
  """
    eps = torch.finfo(t.dtype).eps
    # eps = 1e-3

    device = t.device

    # Draw uniform samples.
    if not rand:
        if deterministic_center:
            pad = 1 / (2 * num_samples)
            u = torch.linspace(pad, 1. - pad - eps, num_samples, device=device)
        else:
            u = torch.linspace(0, 1. - eps, num_samples, device=device)
        u = torch.broadcast_to(u, t.shape[:-1] + (num_samples,))
    else:
        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u_max = eps + (1 - eps) / num_samples
        max_jitter = (1 - u_max) / (num_samples - 1) - eps
        d = 1 if single_jitter else num_samples
        u = torch.linspace(0, 1 - u_max, num_samples, device=device) + \
            torch.rand(t.shape[:-1] + (d,), device=device) * max_jitter

    return invert_cdf(u, t, w_logits)


def sample_np(rand,
              t,
              w_logits,
              num_samples,
              single_jitter=False,
              deterministic_center=False):
    """
    numpy version of sample()
  """
    eps = np.finfo(np.float32).eps

    # Draw uniform samples.
    if not rand:
        if deterministic_center:
            pad = 1 / (2 * num_samples)
            u = np.linspace(pad, 1. - pad - eps, num_samples)
        else:
            u = np.linspace(0, 1. - eps, num_samples)
        u = np.broadcast_to(u, t.shape[:-1] + (num_samples,))
    else:
        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u_max = eps + (1 - eps) / num_samples
        max_jitter = (1 - u_max) / (num_samples - 1) - eps
        d = 1 if single_jitter else num_samples
        u = np.linspace(0, 1 - u_max, num_samples) + \
            np.random.rand(*t.shape[:-1], d) * max_jitter

    return invert_cdf_np(u, t, w_logits)


def sample_intervals(rand,
                     t,
                     w_logits,
                     num_samples,
                     single_jitter=False,
                     domain=(-torch.inf, torch.inf)):
    """Sample *intervals* (rather than points) from a step function.

  Args:
    rand: random number generator (or None for `linspace` sampling).
    t: [..., num_bins + 1], bin endpoint coordinates (must be sorted)
    w_logits: [..., num_bins], logits corresponding to bin weights
    num_samples: int, the number of intervals to sample.
    single_jitter: bool, if True, jitter every sample along each ray by the same
      amount in the inverse CDF. Otherwise, jitter each sample independently.
    domain: (minval, maxval), the range of valid values for `t`.

  Returns:
    t_samples: [batch_size, num_samples].
  """
    if num_samples <= 1:
        raise ValueError(f'num_samples must be > 1, is {num_samples}.')

    # Sample a set of points from the step function.
    centers = sample(
        rand,
        t,
        w_logits,
        num_samples,
        single_jitter,
        deterministic_center=True)

    # The intervals we return will span the midpoints of each adjacent sample.
    mid = (centers[..., 1:] + centers[..., :-1]) / 2

    # Each first/last fencepost is the reflection of the first/last midpoint
    # around the first/last sampled center. We clamp to the limits of the input
    # domain, provided by the caller.
    minval, maxval = domain
    first = (2 * centers[..., :1] - mid[..., :1]).clamp_min(minval)
    last = (2 * centers[..., -1:] - mid[..., -1:]).clamp_max(maxval)

    t_samples = torch.cat([first, mid, last], dim=-1)
    return t_samples


def lossfun_distortion(t, w):
    """Compute iint w[i] w[j] |t[i] - t[j]| di dj."""
    # The loss incurred between all pairs of intervals.
    ut = (t[..., 1:] + t[..., :-1]) / 2
    dut = torch.abs(ut[..., :, None] - ut[..., None, :])
    loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, dim=-1), dim=-1)

    # The loss incurred within each individual interval with itself.
    loss_intra = torch.sum(w ** 2 * (t[..., 1:] - t[..., :-1]), dim=-1) / 3

    return loss_inter + loss_intra


def interval_distortion(t0_lo, t0_hi, t1_lo, t1_hi):
    """Compute mean(abs(x-y); x in [t0_lo, t0_hi], y in [t1_lo, t1_hi])."""
    # Distortion when the intervals do not overlap.
    d_disjoint = torch.abs((t1_lo + t1_hi) / 2 - (t0_lo + t0_hi) / 2)

    # Distortion when the intervals overlap.
    d_overlap = (2 *
                 (torch.minimum(t0_hi, t1_hi) ** 3 - torch.maximum(t0_lo, t1_lo) ** 3) +
                 3 * (t1_hi * t0_hi * torch.abs(t1_hi - t0_hi) +
                      t1_lo * t0_lo * torch.abs(t1_lo - t0_lo) + t1_hi * t0_lo *
                      (t0_lo - t1_hi) + t1_lo * t0_hi *
                      (t1_lo - t0_hi))) / (6 * (t0_hi - t0_lo) * (t1_hi - t1_lo))

    # Are the two intervals not overlapping?
    are_disjoint = (t0_lo > t1_hi) | (t1_lo > t0_hi)

    return torch.where(are_disjoint, d_disjoint, d_overlap)


def weighted_percentile(t, w, ps):
    """Compute the weighted percentiles of a step function. w's must sum to 1."""
    cw = integrate_weights(w)
    # We want to interpolate into the integrated weights according to `ps`.
    fn = lambda cw_i, t_i: math.sorted_interp(torch.tensor(ps, device=t.device) / 100, cw_i, t_i)
    # Vmap fn to an arbitrary number of leading dimensions.
    cw_mat = cw.reshape([-1, cw.shape[-1]])
    t_mat = t.reshape([-1, t.shape[-1]])
    wprctile_mat = fn(cw_mat, t_mat)  # TODO
    wprctile = wprctile_mat.reshape(cw.shape[:-1] + (len(ps),))
    return wprctile


def resample(t, tp, vp, use_avg=False):
    """Resample a step function defined by (tp, vp) into intervals t.

  Args:
    t: tensor with shape (..., n+1), the endpoints to resample into.
    tp: tensor with shape (..., m+1), the endpoints of the step function being
      resampled.
    vp: tensor with shape (..., m), the values of the step function being
      resampled.
    use_avg: bool, if False, return the sum of the step function for each
      interval in `t`. If True, return the average, weighted by the width of
      each interval in `t`.
    eps: float, a small value to prevent division by zero when use_avg=True.

  Returns:
    v: tensor with shape (..., n), the values of the resampled step function.
  """
    eps = torch.finfo(t.dtype).eps
    # eps = 1e-3

    if use_avg:
        wp = torch.diff(tp, dim=-1)
        v_numer = resample(t, tp, vp * wp, use_avg=False)
        v_denom = resample(t, tp, wp, use_avg=False)
        v = v_numer / v_denom.clamp_min(eps)
        return v

    acc = torch.cumsum(vp, dim=-1)
    acc0 = torch.cat([torch.zeros(acc.shape[:-1] + (1,), device=acc.device), acc], dim=-1)
    acc0_resampled = math.sorted_interp(t, tp, acc0)  # TODO
    v = torch.diff(acc0_resampled, dim=-1)
    return v


def resample_np(t, tp, vp, use_avg=False):
    """
    numpy version of resample
  """
    eps = np.finfo(t.dtype).eps
    if use_avg:
        wp = np.diff(tp, axis=-1)
        v_numer = resample_np(t, tp, vp * wp, use_avg=False)
        v_denom = resample_np(t, tp, wp, use_avg=False)
        v = v_numer / np.maximum(eps, v_denom)
        return v

    acc = np.cumsum(vp, axis=-1)
    acc0 = np.concatenate([np.zeros(acc.shape[:-1] + (1,)), acc], axis=-1)
    acc0_resampled = np.vectorize(np.interp, signature='(n),(m),(m)->(n)')(t, tp, acc0)
    v = np.diff(acc0_resampled, axis=-1)
    return v


def blur_stepfun(x, y, r):
    xr, xr_idx = torch.sort(torch.cat([x - r, x + r], dim=-1))
    y1 = (torch.cat([y, torch.zeros_like(y[..., :1])], dim=-1) -
          torch.cat([torch.zeros_like(y[..., :1]), y], dim=-1)) / (2 * r)
    y2 = torch.cat([y1, -y1], dim=-1).take_along_dim(xr_idx[..., :-1], dim=-1)
    yr = torch.cumsum((xr[..., 1:] - xr[..., :-1]) *
                      torch.cumsum(y2, dim=-1), dim=-1).clamp_min(0)
    yr = torch.cat([torch.zeros_like(yr[..., :1]), yr], dim=-1)
    return xr, yr