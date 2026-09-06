"""GCS implementation; credentials are supplied exclusively by ADC."""

from .base import ArtifactStore, safe_key


class GCSArtifactStore(ArtifactStore):
    def __init__(self, uri, *, project=None, client=None):
        if not uri.startswith("gs://"):
            raise ValueError("Expected gs://bucket/prefix")
        bucket, _, prefix = uri[5:].partition("/")
        if not bucket or not prefix:
            raise ValueError("A run-specific GCS prefix is required")
        self.prefix = safe_key(prefix.rstrip("/"))
        if client is None:
            from google.cloud import storage

            client = storage.Client(project=project)
        self.bucket = client.bucket(bucket)

    def blob(self, key):
        return self.bucket.blob(f"{self.prefix}/{safe_key(key)}")

    def write_bytes(self, key, data, *, overwrite=False):
        from google.api_core.exceptions import NotFound, PreconditionFailed

        blob = self.blob(key)
        generation = 0
        if overwrite:
            try:
                blob.reload()
                generation = int(blob.generation)
            except NotFound:
                pass
        try:
            # The SDK enables conditional retry when a generation precondition is present.
            blob.upload_from_string(data, if_generation_match=generation, checksum="crc32c")
        except PreconditionFailed as exc:
            raise FileExistsError(
                f"GCS artifact already exists or changed concurrently: {key}"
            ) from exc

    def read_bytes(self, key):
        return self.blob(key).download_as_bytes(checksum="crc32c")

    def exists(self, key):
        return self.blob(key).exists()

    def delete(self, key):
        from google.api_core.exceptions import NotFound

        blob = self.blob(key)
        try:
            blob.reload()
            blob.delete(if_generation_match=int(blob.generation))
        except NotFound:
            pass
