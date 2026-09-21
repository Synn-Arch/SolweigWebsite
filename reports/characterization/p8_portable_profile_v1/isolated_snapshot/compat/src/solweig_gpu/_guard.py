"""Reject distributions that own the same package namespace."""
from importlib import metadata


def reject_upstream():
    try:
        version = metadata.version('solweig-gpu')
    except metadata.PackageNotFoundError:
        return
    raise ImportError(
        f'Conflicting upstream solweig-gpu {version} is installed alongside '
        'solweig-light-compat. Both distributions own solweig_gpu. '
        'Create a clean environment and install solweig-light plus '
        'solweig-light-compat there. If removing either overlapping distribution '
        'from this environment, reinstall the remaining distribution to restore '
        'its package files. Keep upstream reference environments separate.'
    )
