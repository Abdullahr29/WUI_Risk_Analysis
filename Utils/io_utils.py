import os
from pathlib import Path
import toml
import subprocess

CONFIG_FILE_PATH = "../config.toml"

RCLONE = "rclone"                
REMOTE = "onedrive:Datasets/WUI_Imagery"

class RemoteIO:

    def __init__(self):
        pass

    @staticmethod
    def remote_exists(remote_rel: str) -> bool:
        """Check if a remote file exists (relative to REMOTE)."""
        # lsl returns non-zero if not found; suppress output
        res = subprocess.run(
            [RCLONE, "lsl", f"{REMOTE}/{remote_rel}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return res.returncode == 0

    @staticmethod
    def get_file(remote_rel: str, dest: str | Path) -> Path:
        """Download a single file from OneDrive -> local path/dir."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True) if dest.suffix else dest.mkdir(parents=True, exist_ok=True)
        # If `dest` is a directory, use copy; if it’s a file path, use copyto
        if dest.is_dir() or dest.suffix == "":
            cmd = [RCLONE, "copy", f"{REMOTE}/{remote_rel}", str(dest), "--checksum", "--progress"]
        else:
            cmd = [RCLONE, "copyto", f"{REMOTE}/{remote_rel}", str(dest), "--checksum", "--progress"]
        RemoteIO._run(cmd)
        return dest if dest.is_dir() else dest

    @staticmethod
    def put_file(src: str | Path, remote_rel_or_full: str | None = None) -> None:
        """
        Upload a single local file -> OneDrive.
        If `remote_rel_or_full` is None, uploads to REMOTE/ (same basename).
        You can pass either a relative path (joined to REMOTE) or a full remote like 'onedrive:foo/bar.tif'.
        """
        src = Path(src)
        assert src.is_file(), f"Not a file: {src}"
        remote = (f"{REMOTE}/{remote_rel_or_full}"
                if remote_rel_or_full and not remote_rel_or_full.startswith("onedrive:")
                else (remote_rel_or_full or f"{REMOTE}/{src.name}"))

        # copy (dir target) vs copyto (file target) – here we target an exact filename with copyto
        # Ensure parent remote dir exists (rclone will create it if needed, but mkdir is cheap)
        RemoteIO._run([RCLONE, "mkdir", str(Path(remote).parent.as_posix().replace("onedrive:", "onedrive:"))])
        RemoteIO._run([RCLONE, "copyto", str(src), remote, "--checksum", "--progress"])


    def _run(cmd: list[str]) -> None:
        """Run a command and raise a helpful error if it fails."""
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}") from e
