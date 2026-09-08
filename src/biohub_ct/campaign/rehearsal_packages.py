"""Compiled source identities allowed for bounded private E0 rehearsals.

Selecting a package here establishes reviewed source identity only. It does not
grant quota, launch permission, quality admission, or competition submission.
"""

from types import MappingProxyType

from biohub_ct.campaign.kaggle_rehearsal import E0_R3_PACKAGE_IDENTITY, PackageIdentity

E0_R4_PACKAGE_IDENTITY = PackageIdentity(
    notebook_slug="clarkkitchen/biohub-e0-instrumented-reference",
    notebook_title="Biohub E0 Instrumented Reference",
    competition="biohub-cell-tracking-during-development",
    machine_shape="NvidiaTeslaT4",
    release_digest="41b2810b88edd3b61a4bed8b3f32d1a0df3454af463f20520473d96bb6244353",
    package_manifest_sha256="62bcfb18db088e217f9aeb417f9329b464d8f86ae316fff1680fdc4e666ab049",
    kernel_metadata_sha256="df623fb2924738f7d21612408ff0590c2d4ab021deca9d1fa615b36c19e61c95",
    artifact_lock_sha256="68312573350cc079561167e25304dd1a5b053a10abc3c7c8ea3869c64fbeb6ff",
    notebook_sha256="e7f4fbdf7c94bd475eb00ee00dfff9cfdfa3022aa35ebb959421cd50c3b41fb8",
    cli_normalized_notebook_sha256=(
        "630e6cca6e663888573f19f4efd97721177483fb59495d2cb249365f8d9e90fa"
    ),
    dataset_versions=E0_R3_PACKAGE_IDENTITY.dataset_versions,
)

REHEARSAL_PACKAGES = MappingProxyType({"r3": E0_R3_PACKAGE_IDENTITY, "r4": E0_R4_PACKAGE_IDENTITY})
DEFAULT_PACKAGE_DIRS = MappingProxyType(
    {
        "r3": "work/e0-reference/package-r3",
        "r4": "work/e0-reference/package-r4-title-fixed-a",
    }
)


def reviewed_package(generation: str) -> PackageIdentity:
    """Select a compiled identity, never a caller-provided hash override."""
    try:
        return REHEARSAL_PACKAGES[generation]
    except (KeyError, TypeError) as exc:
        raise ValueError("unknown reviewed rehearsal package generation") from exc
