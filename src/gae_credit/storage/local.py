from pathlib import Path

from .base import ArtifactStore, safe_key


class LocalArtifactStore(ArtifactStore):
    """Filesystem implementation used for local mirrors and cloud-contract tests."""

    def __init__(self, root):
        self.root = Path(root).resolve()

    def path(self, key):
        path = self.root / safe_key(key)
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("Artifact key escapes store through a symlink")
        return path

    def write_bytes(self, key, data, *, overwrite=False):
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive create is also our inter-process lease primitive.
        with path.open("wb" if overwrite else "xb") as stream:
            stream.write(data)

    def read_bytes(self, key):
        return self.path(key).read_bytes()

    def exists(self, key):
        return self.path(key).exists()

    def delete(self, key):
        self.path(key).unlink(missing_ok=True)
