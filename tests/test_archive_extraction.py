"""Small, offline ZIP fixtures only; never touches the downloaded large archives."""
from hashlib import sha256
import json
from pathlib import Path
import stat
from types import SimpleNamespace
import warnings
import zipfile

import pytest

from scripts import extract_auctionnet_archives as extraction


def make_zip(root, members=None, name="sample.zip"):
    archive = root / name
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as handle:
        for path, content in (members or [("period.csv", b"index,value\n0,10\n")]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                handle.writestr(path, content)
    return archive


def read_manifest(root):
    return json.loads((root / extraction.MANIFEST_NAME).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["../escape.csv", "safe/../../escape.csv", "/absolute.csv", "C:/drive.csv",
                                  "C:relative.csv", "\\\\server\\share.csv", "safe\\..\\escape.csv", "a//b.csv",
                                  "./local.csv", "a/./b.csv", "a:stream.csv", "CON.txt", "nul", "folder./x.csv",
                                  "folder /x.csv", "control\x01.csv"])
def test_unsafe_member_paths_are_rejected_before_writes(tmp_path, name):
    archive = make_zip(tmp_path, [(name, b"bad")])
    with pytest.raises(ValueError, match="unsafe|control"):
        extraction.extract_archives(tmp_path, delete_archives=True, reserve_bytes=0)
    assert archive.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sample.zip"]


@pytest.mark.parametrize("members", [
    [("same.csv", b"a"), ("same.csv", b"a")],
    [("A.csv", b"a"), ("a.csv", b"b")],
    [("parent", b"a"), ("parent/child.csv", b"b")],
    [("parent/child.csv", b"b"), ("parent", b"a")],
])
def test_duplicate_case_collision_and_file_ancestor_conflict(tmp_path, members):
    archive = make_zip(tmp_path, members)
    with pytest.raises(ValueError, match="duplicate|conflicts"):
        extraction.extract_archives(tmp_path, delete_archives=True, reserve_bytes=0)
    assert archive.exists()
    assert not (tmp_path / extraction.MANIFEST_NAME).exists()


def test_symlink_entry_rejected(tmp_path):
    link = zipfile.ZipInfo("link.csv")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    archive = make_zip(tmp_path, [(link, b"../outside.csv")])
    with pytest.raises(ValueError, match="symlink"):
        extraction.extract_archives(tmp_path, delete_archives=True, reserve_bytes=0)
    assert archive.exists()


@pytest.mark.parametrize("target", [extraction.MANIFEST_NAME, "sample.zip"])
def test_members_cannot_replace_manifest_or_input_archive(tmp_path, target):
    archive = make_zip(tmp_path, [(target, b"not allowed")])
    original = archive.read_bytes()
    with pytest.raises(ValueError, match="collides"):
        extraction.extract_archives(tmp_path, delete_archives=True, reserve_bytes=0)
    assert archive.read_bytes() == original


def test_list_only_reads_headers_without_hashing_writing_or_deleting(tmp_path, monkeypatch):
    archive = make_zip(tmp_path)
    monkeypatch.setattr(extraction, "_hash_file", lambda path: pytest.fail("list-only must not hash full data"))
    report = extraction.extract_archives(tmp_path, list_only=True, delete_archives=True)
    assert report["list_only"] is True
    assert report["archives"][0]["expanded_bytes"] == len(b"index,value\n0,10\n")
    assert "have not been verified" in report["notice"]
    assert archive.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sample.zip"]


def test_extracts_and_verifies_every_member_including_macos_resources_before_delete(tmp_path):
    members = [("period/", b""), ("period/data.csv", b"a,b\n1,2\n"),
               ("__MACOSX/._data.csv", b"resource metadata"), ("empty.txt", b"")]
    archive = make_zip(tmp_path, members)
    archive_digest = sha256(archive.read_bytes()).hexdigest()
    manifest = extraction.extract_archives(tmp_path, delete_archives=True, reserve_bytes=0)
    assert not archive.exists()
    record = manifest["archives"]["sample.zip"]
    assert record["status"] == "verified_and_deleted"
    assert record["archive_deleted"] is True
    assert record["archive_sha256"] == archive_digest
    assert len(record["members"]) == len(members)
    for member in record["members"]:
        path = tmp_path / member["path"]
        if member["directory"]:
            assert path.is_dir()
        else:
            assert sha256(path.read_bytes()).hexdigest() == member["sha256"]
            assert path.stat().st_size == member["size"]
            assert len(member["crc32"]) == 8
    assert (tmp_path / "__MACOSX" / "._data.csv").read_bytes() == b"resource metadata"
    assert read_manifest(tmp_path) == manifest
    assert not list(tmp_path.rglob("*.part"))


def test_default_preserves_archive_and_resume_reuses_verified_output(tmp_path):
    archive = make_zip(tmp_path)
    first = extraction.extract_archives(tmp_path, reserve_bytes=0)
    assert archive.exists()
    assert first["archives"]["sample.zip"]["status"] == "verified"
    target = tmp_path / "period.csv"
    before = target.stat().st_mtime_ns
    second = extraction.extract_archives(tmp_path, archives=["sample.zip"], delete_archives=True, reserve_bytes=0)
    assert second["archives"]["sample.zip"]["members"][0]["reused"] is True
    assert target.stat().st_mtime_ns == before
    assert not archive.exists()
    no_work = extraction.extract_archives(tmp_path, reserve_bytes=0)
    assert no_work["no_archives_found"] is True
    assert "not freshly reverified" in no_work["notice"]


@pytest.mark.parametrize("existing", [b"wrong", b"other\nvalue\n0000\n"])
def test_existing_file_conflict_is_never_overwritten_or_deleted(tmp_path, existing):
    archive = make_zip(tmp_path, [("period.csv", b"right")])
    target = tmp_path / "period.csv"
    target.write_bytes(existing)
    with pytest.raises(FileExistsError, match="conflicts"):
        extraction.extract_archives(tmp_path, delete_archives=True, reserve_bytes=0)
    assert archive.exists() and target.read_bytes() == existing
    record = read_manifest(tmp_path)["archives"]["sample.zip"]
    assert record["status"] == "failed" and record["archive_deleted"] is False


def test_crc_corruption_preserves_archive_and_does_not_publish_target(tmp_path):
    archive = make_zip(tmp_path, [("period.csv", b"length preserved corrupted member")])
    with zipfile.ZipFile(archive) as handle:
        info = handle.getinfo("period.csv")
        payload_offset = info.header_offset + 30 + len(info.filename.encode("utf-8")) + len(info.extra)
    content = bytearray(archive.read_bytes())
    content[payload_offset] ^= 0x01
    archive.write_bytes(content)
    with pytest.raises(zipfile.BadZipFile, match="CRC"):
        extraction.extract_archives(tmp_path, delete_archives=True, reserve_bytes=0)
    assert archive.exists()
    assert not (tmp_path / "period.csv").exists()
    record = read_manifest(tmp_path)["archives"]["sample.zip"]
    assert record["status"] == "failed" and record["archive_deleted"] is False
    assert len(record["partial_files"]) == 1
    assert (tmp_path / record["partial_files"][0]).exists()


def test_disk_shortage_fails_before_writing_members(tmp_path, monkeypatch):
    archive = make_zip(tmp_path)
    monkeypatch.setattr(extraction.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    with pytest.raises(OSError, match="insufficient disk space"):
        extraction.extract_archives(tmp_path, delete_archives=True, reserve_bytes=0)
    assert archive.exists() and not (tmp_path / "period.csv").exists()
    assert read_manifest(tmp_path)["archives"]["sample.zip"]["disk_check"]["required_new_bytes"] > 0


def test_delete_failure_is_recorded_and_resume_can_finish(tmp_path, monkeypatch):
    archive = make_zip(tmp_path)
    original_unlink = Path.unlink

    def deny_archive(path, *args, **kwargs):
        if path == archive:
            raise PermissionError("fixture: archive is locked")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", deny_archive)
        with pytest.raises(PermissionError, match="locked"):
            extraction.extract_archives(tmp_path, delete_archives=True, reserve_bytes=0)
    record = read_manifest(tmp_path)["archives"]["sample.zip"]
    assert record["status"] == "verified_delete_failed"
    assert record["deletion_error"]["type"] == "PermissionError"
    assert archive.exists() and (tmp_path / "period.csv").exists()
    recovered = extraction.extract_archives(tmp_path, delete_archives=True, reserve_bytes=0)
    assert recovered["archives"]["sample.zip"]["status"] == "verified_and_deleted"
    assert not archive.exists()


def test_archive_changed_after_verification_is_not_deleted(tmp_path, monkeypatch):
    archive = make_zip(tmp_path)
    real_hash = extraction._hash_file
    archive_calls = 0

    def changed(path):
        nonlocal archive_calls
        if path == archive:
            archive_calls += 1
            if archive_calls == 2:
                with path.open("ab") as handle:
                    handle.write(b"external change")
        return real_hash(path)

    monkeypatch.setattr(extraction, "_hash_file", changed)
    with pytest.raises(ValueError, match="archive changed before deletion"):
        extraction.extract_archives(tmp_path, delete_archives=True, reserve_bytes=0)
    assert archive.exists()
    assert read_manifest(tmp_path)["archives"]["sample.zip"]["archive_deleted"] is False


def test_archive_selection_cannot_escape_root(tmp_path):
    root = tmp_path / "inside"
    root.mkdir()
    archive = make_zip(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        extraction.extract_archives(root, archives=["../sample.zip"], delete_archives=True)
    assert archive.exists()


def test_per_archive_manifest_survives_later_conflict(tmp_path):
    first = make_zip(tmp_path, [("first.csv", b"correct")], name="a.zip")
    second = make_zip(tmp_path, [("second.csv", b"expected")], name="b.zip")
    (tmp_path / "second.csv").write_bytes(b"conflict")
    with pytest.raises(FileExistsError):
        extraction.extract_archives(tmp_path, delete_archives=True, reserve_bytes=0)
    manifest = read_manifest(tmp_path)
    assert not first.exists() and second.exists()
    assert manifest["archives"]["a.zip"]["status"] == "verified_and_deleted"
    assert manifest["archives"]["b.zip"]["status"] == "failed"
