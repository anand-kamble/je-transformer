from __future__ import annotations

from typing import Optional

from google.cloud import secretmanager


def access_secret(project_id: str, secret_name: str, version: str = "latest") -> str:
    """
    Read a secret payload from Secret Manager.
    `secret_name` may be either a short name or a full resource path.
    """
    client = secretmanager.SecretManagerServiceClient()
    if "/" in secret_name:
        name = secret_name
    else:
        name = f"projects/{project_id}/secrets/{secret_name}/versions/{version}"
    response = client.access_secret_version(name=name)
    return response.payload.data.decode("utf-8")


def maybe_from_secret(explicit: Optional[str], project_id: Optional[str], secret_name: Optional[str]) -> Optional[str]:
    """
    If `secret_name` is provided, fetch from Secret Manager; otherwise return `explicit`.
    """
    if secret_name:
        if not project_id and not secret_name.startswith("projects/"):
            raise ValueError("project_id is required when using secret short names")
        # Accept full resource paths without project_id
        pid = project_id or ""
        return access_secret(pid, secret_name)
    return explicit

