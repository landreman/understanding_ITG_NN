"""Periodic baselines, structure-destroying perturbations, and support warnings.

The operators in this module act on geometry tensors with axes
``(sample, z, channel)``.  Each public perturbation has an explicit validity
class: most S03 ladder entries are deliberately off manifold and diagnose the
trained function rather than the plasma.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

import numpy as np
import torch


class ValidityTag(StrEnum):
    """Registered intervention-validity taxonomy from PLAN.md."""

    EXACT_SYMMETRY = "exact_symmetry"
    OBSERVED_COMPARISON = "observed_comparison"
    PLAUSIBLY_LOCAL = "plausibly_local_not_guaranteed_physical"
    OFF_MANIFOLD = "deliberately_off_manifold_diagnostic"


@dataclass(frozen=True)
class PerturbationSpec:
    """Machine-readable description of one ladder endpoint or path dose."""

    name: str
    family: str
    validity: ValidityTag
    dose: float = 1.0
    replicate: int = 0
    parameter: str = ""
    member_scope: str = "top10"
    matched_control: str = ""


def interpolate_geometry(
    original: torch.Tensor, endpoint: torch.Tensor, dose: float
) -> torch.Tensor:
    """Linearly interpolate to an endpoint without changing tensor metadata."""

    if original.shape != endpoint.shape:
        raise ValueError("original and endpoint geometries must have equal shape")
    if not 0.0 <= float(dose) <= 1.0:
        raise ValueError("dose must be in [0, 1]")
    return torch.lerp(original, endpoint, float(dose))


def wrapped_window_mask(
    grid_size: int, *, start: int, length: int, device: torch.device | None = None
) -> torch.Tensor:
    """Return a hard periodic window; wrapping never truncates its support."""

    if grid_size < 1 or not 1 <= length <= grid_size:
        raise ValueError("length must be between 1 and grid_size")
    mask = torch.zeros(grid_size, dtype=torch.bool, device=device)
    indices = (torch.arange(length, device=device) + int(start)).remainder(grid_size)
    mask[indices] = True
    return mask


def member_window_lengths(
    receptive_field_positions: Sequence[int], *, grid_size: int = 96
) -> tuple[int, ...]:
    """Combine physical grid scales with member-specific receptive-field spans."""

    candidates = {2, 4, 8, 16, 32}
    candidates.update(min(grid_size, max(1, int(value))) for value in receptive_field_positions)
    return tuple(sorted(candidates))


def robust_constant_profile(reference: torch.Tensor) -> torch.Tensor:
    """Per-channel median constant profile; never defaults positive fields to zero."""

    if reference.ndim != 3:
        raise ValueError("reference must have shape (sample, z, channel)")
    channel_median = reference.reshape(-1, reference.shape[2]).median(dim=0).values
    return channel_median.reshape(1, 1, -1).expand(1, reference.shape[1], -1).clone()


def low_pass(geometry: torch.Tensor, maximum_frequency: int) -> torch.Tensor:
    """Keep DC through ``maximum_frequency`` on the periodic 96-point grid."""

    if geometry.ndim != 3:
        raise ValueError("geometry must have shape (sample, z, channel)")
    nyquist = geometry.shape[1] // 2
    if not 0 <= maximum_frequency <= nyquist:
        raise ValueError("maximum_frequency is outside the rFFT grid")
    spectrum = torch.fft.rfft(geometry, dim=1)
    spectrum[:, maximum_frequency + 1 :, :] = 0
    return torch.fft.irfft(spectrum, n=geometry.shape[1], dim=1)


class ReferenceBackgrounds:
    """Reference distributions used by later replacement/path methods.

    Matching is restricted to equilibrium class and uses robustly scaled
    ``(a/L_T, a/L_n)`` distance.  An optional source row ID is excluded, making
    the returned observed background a comparison rather than the input itself.
    """

    def __init__(
        self,
        geometry: torch.Tensor,
        gradients: np.ndarray,
        equilibrium_class: np.ndarray,
        row_ids: np.ndarray,
    ) -> None:
        values = np.asarray(gradients, dtype=np.float64)
        classes = np.asarray(equilibrium_class)
        rows = np.asarray(row_ids, dtype=np.int64)
        if geometry.ndim != 3 or values.shape != (len(geometry), 2):
            raise ValueError("background geometry/gradient shapes are incompatible")
        if len(classes) != len(geometry) or len(rows) != len(geometry):
            raise ValueError("background metadata lengths are incompatible")
        self.geometry = geometry.detach().cpu()
        self.gradients = values
        self.equilibrium_class = classes
        self.row_ids = rows
        self.gradient_center = np.median(values, axis=0)
        self.gradient_scale = np.maximum(
            np.subtract(*np.quantile(values, (0.75, 0.25), axis=0)) / 1.349,
            np.finfo(np.float64).eps,
        )

    def constant(self) -> torch.Tensor:
        return robust_constant_profile(self.geometry)

    def matched_indices(
        self,
        gradients: np.ndarray,
        equilibrium_class: np.ndarray,
        *,
        source_row_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        query = np.asarray(gradients, dtype=np.float64)
        classes = np.asarray(equilibrium_class)
        if query.shape != (len(classes), 2):
            raise ValueError("query gradient/class shapes are incompatible")
        source = None if source_row_ids is None else np.asarray(source_row_ids, dtype=np.int64)
        selected = np.empty(len(query), dtype=np.int64)
        scaled_reference = (self.gradients - self.gradient_center) / self.gradient_scale
        scaled_query = (query - self.gradient_center) / self.gradient_scale
        for index, (point, class_value) in enumerate(zip(scaled_query, classes)):
            candidates = np.flatnonzero(self.equilibrium_class == class_value)
            if source is not None:
                candidates = candidates[self.row_ids[candidates] != source[index]]
            if not len(candidates):
                raise ValueError(f"no eligible background for equilibrium class {class_value}")
            distance = np.sum(np.square(scaled_reference[candidates] - point), axis=1)
            selected[index] = candidates[np.argmin(distance)]
        return selected

    def matched_observed(
        self,
        gradients: np.ndarray,
        equilibrium_class: np.ndarray,
        *,
        source_row_ids: np.ndarray | None = None,
    ) -> torch.Tensor:
        indices = self.matched_indices(
            gradients, equilibrium_class, source_row_ids=source_row_ids
        )
        return self.geometry[torch.as_tensor(indices)]

    def medoid(self, equilibrium_class: int | None = None) -> torch.Tensor:
        """Return the observed profile nearest the robust geometry center."""

        if equilibrium_class is None:
            indices = np.arange(len(self.geometry))
        else:
            indices = np.flatnonzero(self.equilibrium_class == equilibrium_class)
        if not len(indices):
            raise ValueError("medoid subset is empty")
        values = self.geometry[torch.as_tensor(indices)].numpy().astype(np.float64)
        center = np.median(values, axis=0)
        channel_scale = np.subtract(
            *np.quantile(values, (0.75, 0.25), axis=(0, 1))
        ) / 1.349
        channel_scale = np.maximum(channel_scale, np.finfo(np.float64).eps)
        distance = np.mean(np.square((values - center) / channel_scale), axis=(1, 2))
        return self.geometry[int(indices[np.argmin(distance)])].unsqueeze(0)

    def conditional_channel_profile(
        self,
        channel: int,
        gradients: np.ndarray,
        equilibrium_class: np.ndarray,
        *,
        neighbours: int = 16,
        source_row_ids: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Median channel profile among class/gradient-matched observed rows."""

        if not 0 <= channel < self.geometry.shape[2] or neighbours < 1:
            raise ValueError("invalid channel or neighbour count")
        query = np.asarray(gradients, dtype=np.float64)
        classes = np.asarray(equilibrium_class)
        source = None if source_row_ids is None else np.asarray(source_row_ids)
        output = torch.empty((len(query), self.geometry.shape[1]), dtype=self.geometry.dtype)
        scaled_reference = (self.gradients - self.gradient_center) / self.gradient_scale
        scaled_query = (query - self.gradient_center) / self.gradient_scale
        for index, (point, class_value) in enumerate(zip(scaled_query, classes)):
            candidates = np.flatnonzero(self.equilibrium_class == class_value)
            if source is not None:
                candidates = candidates[self.row_ids[candidates] != source[index]]
            if not len(candidates):
                raise ValueError(
                    f"no eligible conditional profile for equilibrium class {class_value}"
                )
            distance = np.sum(np.square(scaled_reference[candidates] - point), axis=1)
            nearest = candidates[np.argsort(distance, kind="stable")[:neighbours]]
            output[index] = self.geometry[torch.as_tensor(nearest), :, channel].median(dim=0).values
        return output

    def low_pass_inputs(self, maximum_frequency: int) -> torch.Tensor:
        return low_pass(self.geometry, maximum_frequency)


def _paired_prefix(geometry: torch.Tensor, paired_halves: bool) -> torch.Tensor | None:
    """Return the unique half of a registered varied/fixed twin tensor."""

    if not paired_halves:
        return None
    if len(geometry) % 2:
        raise ValueError("paired geometry must contain two equal-size halves")
    count = len(geometry) // 2
    if not torch.equal(geometry[:count], geometry[count:]):
        raise ValueError("paired geometry halves must be bit-identical")
    return geometry[:count]


def joint_permutation(
    geometry: torch.Tensor, *, seed: int, paired_halves: bool = False
) -> torch.Tensor:
    """Permute pointwise seven-channel vectors jointly for every sample."""

    paired = _paired_prefix(geometry, paired_halves)
    if paired is not None:
        endpoint = joint_permutation(paired, seed=seed)
        return torch.cat((endpoint, endpoint))

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    output = torch.empty_like(geometry)
    for sample in range(len(geometry)):
        order = torch.randperm(geometry.shape[1], generator=generator)
        output[sample] = geometry[sample, order]
    return output


def block_permutation(
    geometry: torch.Tensor,
    block_length: int,
    *,
    seed: int,
    paired_halves: bool = False,
) -> torch.Tensor:
    """Jointly reorder contiguous blocks, excluding exact cyclic shifts.

    A cyclic rotation of the block labels is only a joint circular shift, which
    is an exact symmetry of the canonical S03 estimand.  Such orders are rejected
    so every sampled endpoint actually destroys block order.
    """

    paired = _paired_prefix(geometry, paired_halves)
    if paired is not None:
        endpoint = block_permutation(paired, block_length, seed=seed)
        return torch.cat((endpoint, endpoint))

    grid_size = geometry.shape[1]
    if block_length < 1 or grid_size % block_length:
        raise ValueError("block_length must divide the periodic grid")
    block_count = grid_size // block_length
    if block_count < 3:
        raise ValueError("at least three blocks are required for a non-cyclic order")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    output = torch.empty_like(geometry)
    for sample in range(len(geometry)):
        # Randomize the block origin as well as block order.  This prevents grid
        # index zero from becoming an artificial boundary on the cyclic domain;
        # rolling back makes every sampled partition a wrapped partition.
        origin = int(torch.randint(0, grid_size, (1,), generator=generator))
        shifted = torch.roll(geometry[sample], shifts=-origin, dims=0)
        reshaped = shifted.reshape(block_count, block_length, geometry.shape[2])
        while True:
            order = torch.randperm(block_count, generator=generator)
            # Every adjacent label differs by +1 modulo block_count exactly for
            # a cyclic rotation of the identity order.
            if not torch.all(torch.remainder(order[1:] - order[:-1], block_count) == 1):
                break
        permuted = reshaped[order].reshape(grid_size, geometry.shape[2])
        output[sample] = torch.roll(permuted, shifts=origin, dims=0)
    return output


def random_joint_shift(
    geometry: torch.Tensor, *, seed: int, paired_halves: bool = False
) -> torch.Tensor:
    """Apply one random circular shift to all channels of each sample."""

    paired = _paired_prefix(geometry, paired_halves)
    if paired is not None:
        endpoint = random_joint_shift(paired, seed=seed)
        return torch.cat((endpoint, endpoint))

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    shifts = torch.randint(0, geometry.shape[1], (len(geometry),), generator=generator)
    output = torch.empty_like(geometry)
    for sample, shift in enumerate(shifts.tolist()):
        output[sample] = torch.roll(geometry[sample], shifts=shift, dims=0)
    return output


def independent_channel_shifts(
    geometry: torch.Tensor, *, seed: int, paired_halves: bool = False
) -> torch.Tensor:
    """Independently rotate every sample/channel, destroying co-location."""

    paired = _paired_prefix(geometry, paired_halves)
    if paired is not None:
        endpoint = independent_channel_shifts(paired, seed=seed)
        return torch.cat((endpoint, endpoint))

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    shifts = torch.randint(
        0, geometry.shape[1], (len(geometry), geometry.shape[2]), generator=generator
    )
    output = torch.empty_like(geometry)
    for sample in range(len(geometry)):
        for channel in range(geometry.shape[2]):
            output[sample, :, channel] = torch.roll(
                geometry[sample, :, channel], shifts=int(shifts[sample, channel]), dims=0
            )
    return output


def phase_scramble(
    geometry: torch.Tensor,
    *,
    seed: int,
    independent_channels: bool,
    paired_halves: bool = False,
) -> torch.Tensor:
    """Scramble non-DC Fourier phase while preserving marginal amplitudes.

    Independent scrambling replaces each channel phase separately.  Common
    scrambling instead rotates every channel coefficient by the same random
    phase at each frequency, preserving all cross-channel phase differences.
    """

    paired = _paired_prefix(geometry, paired_halves)
    if paired is not None:
        endpoint = phase_scramble(
            paired, seed=seed, independent_channels=independent_channels
        )
        return torch.cat((endpoint, endpoint))

    spectrum = torch.fft.rfft(geometry, dim=1)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    channel_count = geometry.shape[2] if independent_channels else 1
    phases = 2 * torch.pi * torch.rand(
        (len(geometry), spectrum.shape[1], channel_count), generator=generator
    )
    phases[:, 0, :] = 0
    if geometry.shape[1] % 2 == 0:
        phases[:, -1, :] = 0
    if not independent_channels:
        phases = phases.expand(-1, -1, geometry.shape[2])
    rotations = torch.polar(torch.ones_like(phases), phases)
    if independent_channels:
        scrambled = spectrum.abs() * rotations
    else:
        scrambled = spectrum * rotations
    # DC and Nyquist are real; retaining their signed values preserves means and
    # the full marginal amplitude spectrum exactly.
    scrambled[:, 0, :] = spectrum[:, 0, :]
    if geometry.shape[1] % 2 == 0:
        scrambled[:, -1, :] = spectrum[:, -1, :]
    return torch.fft.irfft(scrambled, n=geometry.shape[1], dim=1)


def attenuate_fourier_band(
    geometry: torch.Tensor, *, minimum_frequency: int, maximum_frequency: int, dose: float
) -> torch.Tensor:
    """Attenuate one wrapped Fourier band without changing its phase."""

    if not 0 <= dose <= 1:
        raise ValueError("dose must be in [0, 1]")
    nyquist = geometry.shape[1] // 2
    if not 0 <= minimum_frequency <= maximum_frequency <= nyquist:
        raise ValueError("invalid Fourier band")
    spectrum = torch.fft.rfft(geometry, dim=1)
    spectrum[:, minimum_frequency : maximum_frequency + 1, :] *= 1.0 - float(dose)
    return torch.fft.irfft(spectrum, n=geometry.shape[1], dim=1)


def scale_non_dc_amplitude(geometry: torch.Tensor, *, factor: float) -> torch.Tensor:
    """Scale all non-DC Fourier amplitudes while preserving every phase."""

    if factor < 0:
        raise ValueError("amplitude factor must be nonnegative")
    spectrum = torch.fft.rfft(geometry, dim=1)
    spectrum[:, 1:, :] *= float(factor)
    return torch.fft.irfft(spectrum, n=geometry.shape[1], dim=1)


def replace_channel(
    geometry: torch.Tensor, channel: int, profile: torch.Tensor
) -> torch.Tensor:
    """Replace exactly one channel with a sample-specific or shared profile."""

    if not 0 <= channel < geometry.shape[2]:
        raise ValueError("channel is outside geometry")
    if profile.ndim == 1:
        profile = profile.unsqueeze(0)
    if profile.shape[1] != geometry.shape[1] or profile.shape[0] not in (1, len(geometry)):
        raise ValueError("replacement profile has incompatible shape")
    output = geometry.clone()
    output[:, :, channel] = profile.expand(len(geometry), -1)
    return output


@dataclass
class RobustPCASupport:
    """Robustly scaled PCA with held-out nearest-neighbour calibration.

    This is deliberately only a data-support warning.  The fit uses per-channel
    median/IQR scaling before an ordinary SVD.  Scores report reconstruction
    error and nearest-neighbour distance relative to a held-out calibration
    distribution; neither score proves equilibrium or GX validity.
    """

    channel_center: np.ndarray
    channel_scale: np.ndarray
    feature_center: np.ndarray
    components: np.ndarray
    fit_scores: np.ndarray
    calibration_reconstruction: np.ndarray
    calibration_nearest: np.ndarray

    @staticmethod
    def _canonicalize(
        values: np.ndarray,
        channel_center: np.ndarray,
        channel_scale: np.ndarray,
    ) -> np.ndarray:
        """Fix cyclic phase at the strongest joint standardized excursion.

        The canonical anchor makes exact joint shifts receive the same endpoint
        score.  It does not make interpolation paths to a shifted copy exact:
        intermediate linear mixtures can still leave observed support.  A joint
        seven-channel anchor remains defined when channel 0 is replaced by a
        constant; if every channel is constant, rolling is immaterial.
        """

        array = np.asarray(values, dtype=np.float64)
        output = np.empty_like(array)
        standardized = (array - channel_center) / channel_scale
        anchor_strength = np.sum(np.square(standardized), axis=2)
        anchors = np.argmax(anchor_strength, axis=1)
        for index, anchor in enumerate(anchors):
            output[index] = np.roll(array[index], shift=-int(anchor), axis=0)
        return output

    @staticmethod
    def _channel_statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        center = np.median(values, axis=(0, 1))
        q75, q25 = np.quantile(values, (0.75, 0.25), axis=(0, 1))
        scale = np.maximum((q75 - q25) / 1.349, np.finfo(np.float64).eps)
        return center, scale

    @classmethod
    def fit(
        cls,
        fit_geometry: np.ndarray,
        heldout_geometry: np.ndarray,
        *,
        components: int = 24,
    ) -> "RobustPCASupport":
        fit_raw = np.asarray(fit_geometry, dtype=np.float64)
        heldout_raw = np.asarray(heldout_geometry, dtype=np.float64)
        if (
            fit_raw.ndim != 3
            or heldout_raw.shape[1:] != fit_raw.shape[1:]
            or len(heldout_raw) < 1
        ):
            raise ValueError("support fit/heldout arrays have incompatible shapes")
        channel_center, channel_scale = cls._channel_statistics(fit_raw)
        fit = cls._canonicalize(fit_raw, channel_center, channel_scale)
        heldout = cls._canonicalize(heldout_raw, channel_center, channel_scale)
        scaled = ((fit - channel_center) / channel_scale).reshape(len(fit), -1)
        feature_center = np.median(scaled, axis=0)
        centered = scaled - feature_center
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        count = min(max(1, int(components)), len(right), len(fit) - 1)
        basis = right[:count]
        fit_scores = centered @ basis.T
        temporary = cls(
            channel_center=channel_center,
            channel_scale=channel_scale,
            feature_center=feature_center,
            components=basis,
            fit_scores=fit_scores,
            calibration_reconstruction=np.ones(1),
            calibration_nearest=np.ones(1),
        )
        reconstruction, nearest = temporary.raw_scores(heldout)
        temporary.calibration_reconstruction = reconstruction
        temporary.calibration_nearest = nearest
        return temporary

    def _standardized(self, geometry: np.ndarray) -> np.ndarray:
        values = self._canonicalize(
            np.asarray(geometry, dtype=np.float64),
            self.channel_center,
            self.channel_scale,
        )
        return ((values - self.channel_center) / self.channel_scale).reshape(len(values), -1) - self.feature_center

    @staticmethod
    def _nearest(query: np.ndarray, reference: np.ndarray, chunk: int = 256) -> np.ndarray:
        result = np.empty(len(query), dtype=np.float64)
        reference_norm = np.sum(np.square(reference), axis=1)
        for start in range(0, len(query), chunk):
            stop = min(start + chunk, len(query))
            values = query[start:stop]
            squared = (
                np.sum(np.square(values), axis=1)[:, None]
                + reference_norm[None, :]
                - 2 * values @ reference.T
            )
            result[start:stop] = np.sqrt(np.maximum(np.min(squared, axis=1), 0))
        return result

    def raw_scores(self, geometry: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        standardized = self._standardized(geometry)
        scores = standardized @ self.components.T
        reconstruction = standardized - scores @ self.components
        reconstruction_rms = np.sqrt(np.mean(np.square(reconstruction), axis=1))
        nearest = self._nearest(scores, self.fit_scores) / np.sqrt(self.components.shape[0])
        return reconstruction_rms, nearest

    @staticmethod
    def _percentile(values: np.ndarray, calibration: np.ndarray) -> np.ndarray:
        ordered = np.sort(np.asarray(calibration, dtype=np.float64))
        return np.searchsorted(ordered, values, side="right") / len(ordered)

    def score(self, geometry: np.ndarray) -> dict[str, np.ndarray]:
        reconstruction, nearest = self.raw_scores(geometry)
        reconstruction_percentile = self._percentile(
            reconstruction, self.calibration_reconstruction
        )
        nearest_percentile = self._percentile(nearest, self.calibration_nearest)
        # Both tails are warnings.  A heavily smoothed or collapsed geometry can
        # be anomalously *closer* to the PCA center than any observed held-out
        # row, which a conventional upper-tail outlier score would miss.
        reconstruction_warning = np.minimum(
            1.0, 2.0 * np.abs(reconstruction_percentile - 0.5)
        )
        nearest_warning = np.minimum(1.0, 2.0 * np.abs(nearest_percentile - 0.5))
        return {
            "reconstruction_rms": reconstruction,
            "nearest_distance": nearest,
            "reconstruction_percentile": reconstruction_percentile,
            "nearest_percentile": nearest_percentile,
            "warning_score": np.maximum(reconstruction_warning, nearest_warning),
        }
