import os
from pathlib import Path
import subprocess

RCLONE = "rclone"                

class RemoteIO:

    @staticmethod
    def remote_exists(remote_rel: str) -> bool:
        """Return True if remote file exists."""
        res = subprocess.run(
            [RCLONE, "lsl", remote_rel],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return res.returncode == 0

    @staticmethod
    def get_file(remote_rel: str, dest: str | Path) -> Path:
        """
        Download remote file REMOTE/remote_rel → local dest (full file path).
        Always uses copyto. Assumes dest parent directory exists.
        """
        dest = Path(dest)
        assert dest.suffix, f"dest must be a full file path, got directory: {dest}"

        cmd = [RCLONE, "copyto",
               remote_rel,
               str(dest),
               "--checksum", "--progress"]

        RemoteIO._run(cmd)
        return dest

    @staticmethod
    def put_file(src: str | Path, remote_rel: str) -> None:
        """
        Upload local file → REMOTE/remote_rel (relative path).
        Always uses copyto.
        """
        src = Path(src)
        assert src.is_file(), f"Not a file: {src}"

        cmd = [RCLONE, "copyto",
               str(src),
               remote_rel,
               "--checksum", "--progress"]

        RemoteIO._run(cmd)

    def _run(cmd: list[str]) -> None:
        """Run a command and raise an error if it fails."""
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}") from e
