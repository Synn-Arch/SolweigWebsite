"""File locks and directory syncs that work on POSIX and on Windows.

POSIX keeps the exact calls used before (fcntl.flock, fsync on a directory
descriptor or a read-only handle, chmod 0o444). Windows has none of them:
locks use msvcrt byte-range locking on the first byte of the lock file,
directories cannot be fsynced (NTFS journals its metadata), a file is flushed
through a writable handle, and files are not made read-only because Windows
then refuses to replace or delete them.
"""
import errno
import os
import time

WINDOWS = os.name == "nt"

if WINDOWS:
    import msvcrt

    _BUSY = {errno.EACCES, getattr(errno, "EDEADLOCK", errno.EDEADLK)}

    def _locking(fd, mode):
        os.lseek(fd, 0, os.SEEK_SET)          # msvcrt locks from the current position
        msvcrt.locking(fd, mode, 1)

    def lock_exclusive(fd, *, blocking):
        """Lock fd's file; with blocking=False a held lock raises BlockingIOError."""
        while True:
            try:
                _locking(fd, msvcrt.LK_NBLCK)
                return
            except OSError as error:
                if error.errno not in _BUSY:
                    raise
                if not blocking:
                    raise BlockingIOError(error.errno, "lock is held elsewhere") from error
                time.sleep(0.05)

    def unlock(fd):
        _locking(fd, msvcrt.LK_UNLCK)

    def sync_directory(path):
        return None

    def sync_file(path):
        # Windows flushes a file only through a handle that may write to it.
        with open(path, "r+b") as stream:
            os.fsync(stream.fileno())

    def make_read_only(path):
        # A read-only attribute makes Windows refuse later replace/unlink/rmtree.
        return None
else:
    import fcntl

    def lock_exclusive(fd, *, blocking):
        """Lock fd's file; with blocking=False a held lock raises BlockingIOError."""
        fcntl.flock(fd, fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB)

    def unlock(fd):
        fcntl.flock(fd, fcntl.LOCK_UN)

    def sync_directory(path):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def sync_file(path):
        with open(path, "rb") as stream:
            os.fsync(stream.fileno())

    def make_read_only(path):
        os.chmod(path, 0o444)
