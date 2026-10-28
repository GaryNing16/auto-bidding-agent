"""Safely extract local ZIPs; delete an archive only after full verification.

No network, recursive deletion, or extraction via ZipFile.extractall. All files,
including __MACOSX resources, are checked and retained. Failed .part files are
left for diagnosis; existing user files are never replaced. This is a trusted
local workflow, not protection against a hostile concurrent filesystem actor.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import stat
import sys
import uuid
import zipfile
import zlib


CHUNK_BYTES = 4 * 1024 * 1024
MANIFEST_NAME = "extraction_manifest.json"
SCHEMA_VERSION = "auctionnet-safe-extraction-v1"
DEFAULT_RESERVE_BYTES = 256 * 1024 * 1024
PROGRESS_BYTES = 512 * 1024 * 1024
WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _within(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path escapes extraction root: {path}")
    return resolved


def _snapshot(path: Path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _hash_file(path: Path) -> tuple[str, int, int]:
    before = _snapshot(path)
    digest, size, crc = sha256(), 0, 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
            crc = zlib.crc32(block, crc)
            size += len(block)
    if _snapshot(path) != before:
        raise ValueError(f"file changed during verification: {path}")
    return digest.hexdigest(), size, crc & 0xFFFFFFFF


def _member_path(root: Path, info: zipfile.ZipInfo) -> tuple[Path, str]:
    name = info.orig_filename
    if (not name or "\x00" in name or "\\" in name or name.startswith("/")
            or PureWindowsPath(name).drive):
        raise ValueError(f"unsafe ZIP entry path: {name!r}")
    parts = name.rstrip("/").split("/")
    if any(not part or part in (".", "..") or ":" in part or part[-1:] in (".", " ")
           or part.split(".")[0].upper() in WINDOWS_RESERVED for part in parts):
        raise ValueError(f"unsafe ZIP entry path component: {name!r}")
    if any(any(ord(char) < 32 for char in part) for part in parts):
        raise ValueError(f"control character in ZIP path: {name!r}")
    mode = info.external_attr >> 16
    kind = stat.S_IFMT(mode)
    if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
        raise ValueError(f"nonregular/symlink ZIP entry is forbidden: {name!r}")
    if kind == stat.S_IFDIR and not info.is_dir():
        raise ValueError(f"inconsistent ZIP directory entry: {name!r}")
    if info.flag_bits & 1:
        raise ValueError(f"encrypted ZIP entry is unsupported: {name!r}")
    if info.is_dir() and (info.file_size != 0 or info.CRC != 0):
        raise ValueError(f"ZIP directory has unverified payload: {name!r}")
    relative = PurePosixPath(*parts).as_posix()
    target = root.joinpath(*parts)
    _within(root, target)
    # Reject symlink/junction/reparse ancestors even when their current targets
    # happen to be inside the root. Normal directories alone are accepted.
    for candidate in (target, *target.parents):
        if candidate == root:
            break
        if candidate.is_symlink() or (hasattr(candidate, "is_junction") and candidate.is_junction()):
            raise ValueError(f"symlink/junction in destination path: {candidate}")
    if relative.casefold() == MANIFEST_NAME.casefold():
        raise ValueError("ZIP member collides with the extraction manifest")
    return target, relative


def _inspect(root: Path, archive: Path, protected_archives: set[Path]) -> list[tuple[zipfile.ZipInfo, Path, str]]:
    entries, names = [], {}
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            target, relative = _member_path(root, info)
            normalized = relative.casefold()
            if normalized in names:
                raise ValueError(f"duplicate/case-colliding ZIP destination: {relative}")
            if target.resolve() in protected_archives:
                raise ValueError(f"ZIP entry collides with an input archive: {relative}")
            names[normalized] = info.is_dir()
            if target.exists() and (target.is_dir() != info.is_dir()):
                raise FileExistsError(f"existing destination has a conflicting type: {target}")
            entries.append((info, target, relative))
    for name in names:
        for parent in PurePosixPath(name).parents:
            if str(parent) != "." and names.get(str(parent)) is False:
                raise ValueError(f"ZIP file conflicts with a descendant path: {name}")
    return entries


def _write_manifest(root: Path, manifest: dict):
    final = _within(root, root / MANIFEST_NAME)
    temporary = _within(root, root / f".extraction_manifest.{uuid.uuid4().hex}.part")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, final)


def _stream_member(handle: zipfile.ZipFile, info: zipfile.ZipInfo, output=None) -> tuple[str, int, int]:
    digest, size, crc, next_progress = sha256(), 0, 0, PROGRESS_BYTES
    with handle.open(info) as source:
        for block in iter(lambda: source.read(CHUNK_BYTES), b""):
            size += len(block)
            if size > info.file_size:
                raise ValueError(f"expanded ZIP member exceeds declared size: {info.filename}")
            digest.update(block)
            crc = zlib.crc32(block, crc)
            if output is not None:
                output.write(block)
            if size >= next_progress:
                print(f"  {info.filename}: {size:,}/{info.file_size:,} bytes", flush=True)
                next_progress += PROGRESS_BYTES
    if size != info.file_size or (crc & 0xFFFFFFFF) != info.CRC:
        raise ValueError(f"ZIP member length/CRC mismatch: {info.filename}")
    return digest.hexdigest(), size, crc & 0xFFFFFFFF


def _publish_part(root: Path, partial: Path, target: Path):
    _within(root, partial)
    _within(root, target)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"destination appeared during extraction: {target}")
    if os.name == "nt":
        # Windows rename fails rather than replacing an existing destination.
        os.rename(partial, target)
    else:
        # POSIX rename replaces existing files. Atomic exclusive hard-link
        # publication supplies the same no-overwrite guarantee instead.
        os.link(partial, target)
        partial.unlink()


def extract_archives(root: str | Path, archives=None, *, delete_archives=False,
                     list_only=False, reserve_bytes=DEFAULT_RESERVE_BYTES) -> dict:
    """Verify all contents per archive, checkpoint, then optionally unlink ZIP.

    Existing extracted files are reused only after SHA256/size/CRC match the
    actual decompressed ZIP member. All published files are reread from disk.
    An interrupted run can reuse verified outputs while its ZIP still exists.
    Only ZIPs selected under root can be deleted; no recursive deletion occurs.
    """
    root = Path(root).resolve(strict=True)
    if not root.is_dir() or type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("root must be a directory and reserve_bytes a nonnegative integer")
    if archives is None:
        selected = sorted((p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".zip"), key=lambda p: p.name.casefold())
    else:
        selected = []
        for item in archives:
            given = Path(item)
            raw = given if given.is_absolute() else root / given
            if raw.is_symlink():
                raise ValueError(f"archive must not be a symlink: {raw}")
            selected.append(_within(root, raw))
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate archive selection")
    for archive in selected:
        if not archive.is_file() or archive.suffix.lower() != ".zip" or archive.is_symlink():
            raise ValueError(f"expected a regular ZIP file within root: {archive}")
    protected = set(selected)
    listings = []
    for archive in selected:
        entries = _inspect(root, archive, protected)
        listings.append({"archive": archive.relative_to(root).as_posix(), "archive_bytes": archive.stat().st_size,
                         "expanded_bytes": sum(info.file_size for info, _, _ in entries),
                         "members": [{"path": relative, "size": info.file_size, "crc32": f"{info.CRC:08x}",
                                      "directory": info.is_dir()} for info, _, relative in entries]})
    if list_only:
        return {"root": str(root), "list_only": True, "archives": listings,
                "expanded_bytes": sum(item["expanded_bytes"] for item in listings),
                "free_bytes": shutil.disk_usage(root).free,
                "notice": "Header/path review only; member data CRC and SHA256 have not been verified."}
    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_symlink():
        raise ValueError("extraction manifest must not be a symlink")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION or not isinstance(manifest.get("archives"), dict):
            raise ValueError("existing extraction manifest has an unsupported schema; refusing to replace")
    else:
        manifest = {"schema_version": SCHEMA_VERSION, "root": str(root), "created_utc": _now(), "archives": {}}
    if not selected:
        return {**manifest, "no_archives_found": True,
                "notice": "No ZIP selected; prior manifest records are retained, not freshly reverified."}
    for archive in selected:
        key = archive.relative_to(root).as_posix()
        state = {"archive": key, "status": "in_progress", "started_utc": _now(),
                 "archive_deleted": False, "members": [], "partial_files": []}
        manifest["archives"][key] = state
        try:
            print(f"Verifying archive SHA256: {key}", flush=True)
            archive_stat = _snapshot(archive)
            archive_sha, archive_bytes, _ = _hash_file(archive)
            state.update(archive_sha256=archive_sha, archive_bytes=archive_bytes)
            entries = _inspect(root, archive, protected)
            missing_bytes = sum(info.file_size for info, target, _ in entries if not info.is_dir() and not target.exists())
            free = shutil.disk_usage(root).free
            state["disk_check"] = {"free_bytes": free, "required_new_bytes": missing_bytes, "reserve_bytes": reserve_bytes}
            if free < missing_bytes + reserve_bytes:
                raise OSError(f"insufficient disk space for {key}: need {missing_bytes + reserve_bytes:,}, free {free:,}")
            _write_manifest(root, manifest)
            with zipfile.ZipFile(archive) as handle:
                for info, target, relative in entries:
                    # Resolve again directly before filesystem operations.
                    _member_path(root, info)
                    if info.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                        expected = _stream_member(handle, info)
                        reused = True
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        reused = target.exists()
                        if reused:
                            if target.stat().st_size != info.file_size:
                                raise FileExistsError(f"existing file size conflicts with ZIP: {target}")
                            actual = _hash_file(target)
                            expected = _stream_member(handle, info)
                            if actual != expected:
                                raise FileExistsError(f"existing file hash/CRC conflicts with ZIP: {target}")
                        else:
                            partial = _within(root, target.with_name(f".{target.name}.{uuid.uuid4().hex}.part"))
                            state["partial_files"].append(partial.relative_to(root).as_posix())
                            _write_manifest(root, manifest)
                            with partial.open("xb") as output:
                                expected = _stream_member(handle, info, output)
                                output.flush()
                                os.fsync(output.fileno())
                            _publish_part(root, partial, target)
                            state["partial_files"].remove(partial.relative_to(root).as_posix())
                            if _hash_file(target) != expected:
                                raise ValueError(f"published file hash/length/CRC mismatch: {target}")
                    state["members"].append({"path": relative, "directory": info.is_dir(), "size": expected[1],
                                             "sha256": expected[0], "crc32": f"{expected[2]:08x}", "reused": reused})
                    print(f"  Verified {'reused' if reused else 'extracted'}: {relative} ({expected[1]:,} bytes)", flush=True)
                    _write_manifest(root, manifest)
            if _snapshot(archive) != archive_stat:
                raise ValueError(f"archive changed during extraction: {key}")
            state.update(status="verified", verified_utc=_now())
            _write_manifest(root, manifest)
            if delete_archives:
                print(f"Rechecking archive before deletion: {key}", flush=True)
                if _hash_file(archive)[:2] != (archive_sha, archive_bytes) or _snapshot(archive) != archive_stat:
                    raise ValueError(f"archive changed before deletion: {key}")
                _within(root, archive)
                try:
                    archive.unlink()
                except OSError as exc:
                    state["deletion_error"] = {"type": type(exc).__name__, "message": str(exc)}
                    state["status"] = "verified_delete_failed"
                    _write_manifest(root, manifest)
                    raise
                state.update(archive_deleted=True, status="verified_and_deleted", deleted_utc=_now())
                _write_manifest(root, manifest)
                print(f"Deleted verified archive: {key}", flush=True)
        except Exception as exc:
            if state["status"] != "verified_delete_failed":
                state["status"] = "failed"
            state["error"] = {"type": type(exc).__name__, "message": str(exc)}
            state["updated_utc"] = _now()
            _write_manifest(root, manifest)
            raise
    return manifest


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/auto-bidding")
    parser.add_argument("--archive", action="append", help="One ZIP path relative to root; may be repeated")
    parser.add_argument("--list-only", action="store_true", help="Inspect headers/paths only; do not extract, hash data, or delete")
    parser.add_argument("--delete-archives", action="store_true", help="Delete each ZIP only after all members and source integrity verify")
    parser.add_argument("--reserve-bytes", type=int, default=DEFAULT_RESERVE_BYTES)
    args = parser.parse_args(argv)
    result = extract_archives(args.root, args.archive, delete_archives=args.delete_archives,
                              list_only=args.list_only, reserve_bytes=args.reserve_bytes)
    if args.list_only:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"root": str(Path(args.root).resolve()), "manifest": MANIFEST_NAME,
                          "archives": {key: value["status"] for key, value in result["archives"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
