from __future__ import annotations

from argparse import Namespace
import json

import pytest

from cv.pipeline.provenance import (
    AUTOMATIC_MODE,
    DIAGNOSTIC_MODE,
    ProvenanceError,
    assert_automatic_document,
    build_provenance,
)
from cv.pipeline.run_manifest import StageRun


def test_automatic_provenance_rejects_human_inputs(tmp_path) -> None:
    with pytest.raises(ProvenanceError, match="forbidden inputs"):
        build_provenance(
            root=tmp_path,
            mode=AUTOMATIC_MODE,
            human_inputs=[{"type": "court_anchor"}],
        )


def test_diagnostic_provenance_records_human_inputs(tmp_path) -> None:
    document = build_provenance(
        root=tmp_path,
        mode=DIAGNOSTIC_MODE,
        human_inputs=[{"type": "court_anchor"}],
    )

    assert document["human_inputs"] == [{"type": "court_anchor"}]


def test_diagnostic_provenance_hashes_human_files(tmp_path) -> None:
    with pytest.raises(ProvenanceError, match="lacks sha256"):
        build_provenance(
            root=tmp_path,
            mode=DIAGNOSTIC_MODE,
            human_inputs=[{"type": "labels", "path": "/tmp/labels.csv"}],
        )


def test_automatic_document_rejects_nested_reviewed_choices() -> None:
    with pytest.raises(ProvenanceError, match="play_clusters"):
        assert_automatic_document(
            {"matches": [{"id": "sample", "play_clusters": [1, 2]}]},
            context="manifest",
        )


def test_stage_run_emits_source_and_configuration_provenance(tmp_path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    stage = StageRun(tmp_path, "example", Namespace(video=str(video), threshold=0.5))

    manifest = json.loads(open(stage.finish()).read())

    source = manifest["provenance"]["source_videos"][0]
    assert source["path"] == video.name
    assert source["path_base"] == "unconfigured_external"
    assert manifest["provenance"]["configuration"]["threshold"] == 0.5
    assert manifest["config"]["video"] == {
        "path": video.name,
        "path_base": "unconfigured_external",
    }
    assert str(tmp_path) not in json.dumps(manifest)


def test_stage_run_rejects_manual_automatic_configuration(tmp_path) -> None:
    with pytest.raises(ProvenanceError, match="allow_manual_anchors"):
        StageRun(tmp_path, "example", Namespace(allow_manual_anchors=True))


def test_stage_run_rejects_label_path_in_automatic_mode(tmp_path) -> None:
    with pytest.raises(ProvenanceError, match="labels"):
        StageRun(tmp_path, "example", Namespace(labels="owner_truth.csv"))


def test_sha256_coalesces_concurrent_reads_and_keeps_digest(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import hashlib
    import threading
    from cv.pipeline import provenance

    path = tmp_path / "shared.bin"
    path.write_bytes(b"unchanged bytes" * 1000)
    import time

    time.sleep(1.01)  # Exercise eligible stable-file caching, not freshness bypass.
    calls = []
    original = provenance._read_sha256
    entered, release = threading.Event(), threading.Event()

    def controlled(path, identity):
        calls.append(path)
        entered.set()
        assert release.wait(5)
        return original(path, identity)

    monkeypatch.setattr(provenance, "_read_sha256", controlled)
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(provenance.file_sha256, path) for _ in range(12)]
        assert entered.wait(5)
        release.set()
        hashes = [f.result() for f in futures]
    assert hashes == [hashlib.sha256(path.read_bytes()).hexdigest()] * 12
    assert provenance.file_sha256(path) == hashes[0]
    assert len(calls) == 1


def test_sha256_invalidates_changed_bytes_restored_mtime_and_atomic_replace(tmp_path):
    import hashlib
    import os
    from cv.pipeline import provenance

    path = tmp_path / "changing.bin"
    path.write_bytes(b"first")
    first = provenance.file_sha256(path)
    stamp = path.stat()
    path.write_bytes(b"other")
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    second = provenance.file_sha256(path)
    assert second == hashlib.sha256(b"other").hexdigest() != first
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"third")
    os.utime(replacement, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    replacement.replace(path)
    assert provenance.file_sha256(path) == hashlib.sha256(b"third").hexdigest() != second


def test_sha256_refuses_mutation_during_read_and_next_call_retries(tmp_path, monkeypatch):
    import hashlib
    from cv.pipeline import provenance

    path = tmp_path / "mutating.bin"
    path.write_bytes(b"old")
    original = provenance._read_sha256

    def mutate(path, identity):
        digest = original(path, identity)
        path.write_bytes(b"new bytes")
        return digest

    monkeypatch.setattr(provenance, "_read_sha256", mutate)
    with pytest.raises(provenance.ProvenanceError, match="changed while hashing"):
        provenance.file_sha256(path)
    monkeypatch.setattr(provenance, "_read_sha256", original)
    assert provenance.file_sha256(path) == hashlib.sha256(b"new bytes").hexdigest()


def test_sha256_cache_is_bounded(tmp_path, monkeypatch):
    from collections import OrderedDict
    import time
    from cv.pipeline import provenance

    monkeypatch.setattr(provenance, "_HASH_CACHE_LIMIT", 2)
    monkeypatch.setattr(provenance, "_HASH_CACHE", OrderedDict())
    files = []
    for i in range(3):
        path = tmp_path / str(i)
        path.write_text(str(i))
        files.append(path)
    time.sleep(1.01)
    for path in files:
        provenance.file_sha256(path)
    assert len(provenance._HASH_CACHE) == 2


def test_sha256_fresh_file_does_not_reuse_coalesced_timestamp(tmp_path, monkeypatch):
    import hashlib
    from cv.pipeline import provenance

    path = tmp_path / "fresh.bin"
    path.write_bytes(b"first")
    identity = provenance._file_identity(path)
    monkeypatch.setattr(provenance, "_file_identity", lambda _: identity)
    monkeypatch.setattr(
        provenance, "_read_sha256", lambda p, _: hashlib.sha256(p.read_bytes()).hexdigest()
    )
    assert provenance.file_sha256(path) == hashlib.sha256(b"first").hexdigest()
    path.write_bytes(b"other")
    # Identical reported stat identity must not reuse a just-created file's digest.
    assert provenance.file_sha256(path) == hashlib.sha256(b"other").hexdigest()


@pytest.mark.skipif(not hasattr(__import__("os"), "fork"), reason="requires POSIX fork")
def test_sha256_fork_child_does_not_inherit_pending_parent_hash(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import multiprocessing
    import os
    import threading
    from cv.pipeline import provenance

    path = tmp_path / "forked.bin"
    path.write_bytes(b"parent and child read identical bytes")
    parent = os.getpid()
    entered, release = threading.Event(), threading.Event()
    original = provenance._read_sha256

    def controlled(path, identity):
        if os.getpid() == parent:
            entered.set()
            assert release.wait(10)
        return original(path, identity)

    monkeypatch.setattr(provenance, "_read_sha256", controlled)
    context = multiprocessing.get_context("fork")
    receive, send = context.Pipe(duplex=False)

    def child():
        send.send(provenance.file_sha256(path))
        send.close()

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(provenance.file_sha256, path)
        assert entered.wait(5)
        process = context.Process(target=child)
        try:
            process.start()
            assert receive.poll(5), "forked child waited on inherited parent hash"
            child_digest = receive.recv()
        finally:
            release.set()
            process.join(5)
            if process.is_alive():
                process.terminate()
                process.join(5)
            receive.close()
            send.close()
        assert process.exitcode == 0
        assert pending.result() == child_digest


def test_sha256_concurrent_export_link_requires_identical_second_read(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import hashlib
    import os
    import threading
    from cv.pipeline import provenance

    path = tmp_path / "native.jpg"
    path.write_bytes(b"original native pixels" * 100_000)
    original = provenance._read_sha256
    read_complete, linked = threading.Event(), threading.Event()
    calls = []

    def controlled(path, identity):
        digest = original(path, identity)
        calls.append(digest)
        if len(calls) == 1:
            read_complete.set()
            assert linked.wait(5)
        return digest

    monkeypatch.setattr(provenance, "_read_sha256", controlled)
    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(provenance.file_sha256, path)
        assert read_complete.wait(5)
        os.link(path, tmp_path / "viewer.jpg")
        linked.set()
        assert future.result(timeout=5) == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(calls) == 2
    assert path.stat().st_nlink == 2


def test_sha256_link_between_stat_and_open_is_reverified(tmp_path, monkeypatch):
    import hashlib
    import os
    from cv.pipeline import provenance

    path = tmp_path / "native.jpg"
    path.write_bytes(b"unchanged")
    original = provenance._read_sha256
    calls = []

    def controlled(path, identity):
        calls.append(identity)
        if len(calls) == 1:
            os.link(path, tmp_path / "viewer.jpg")
        return original(path, identity)

    monkeypatch.setattr(provenance, "_read_sha256", controlled)
    assert provenance.file_sha256(path) == hashlib.sha256(b"unchanged").hexdigest()
    assert len(calls) == 2


def test_sha256_cached_digest_link_race_rechecks_bytes(tmp_path, monkeypatch):
    import os
    from collections import OrderedDict
    from cv.pipeline import provenance

    path = tmp_path / "native.jpg"
    path.write_bytes(b"cached bytes")
    monkeypatch.setattr(provenance, "_HASH_CACHE", OrderedDict())
    monkeypatch.setattr(provenance, "_HASH_SETTLE_NS", 0)
    expected = provenance.file_sha256(path)
    original_stat = provenance._file_identity
    original_read = provenance._read_sha256
    identities, reads = [], []

    def stat_then_link(path):
        identity = original_stat(path)
        identities.append(identity)
        if len(identities) == 1:
            os.link(path, tmp_path / "viewer.jpg")
        return identity

    def count_reads(path, identity):
        reads.append(path)
        return original_read(path, identity)

    monkeypatch.setattr(provenance, "_file_identity", stat_then_link)
    monkeypatch.setattr(provenance, "_read_sha256", count_reads)
    assert provenance.file_sha256(path) == expected
    assert len(reads) == 1  # Cache hit still requires a complete byte verification.


@pytest.mark.parametrize("restore_mtime", [False, True])
def test_sha256_rejects_real_write_even_when_export_also_links(
    tmp_path, monkeypatch, restore_mtime
):
    import os
    from cv.pipeline import provenance

    path = tmp_path / "native.jpg"
    path.write_bytes(b"first")
    stamp = path.stat()
    original = provenance._read_sha256
    calls = []

    def mutate(path, identity):
        digest = original(path, identity)
        calls.append(digest)
        if len(calls) == 1:
            path.write_bytes(b"other")
            if restore_mtime:
                os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            os.link(path, tmp_path / "viewer.jpg")
        return digest

    monkeypatch.setattr(provenance, "_read_sha256", mutate)
    with pytest.raises(provenance.ProvenanceError, match="changed while hashing"):
        provenance.file_sha256(path)
    assert not provenance._HASH_PENDING


def test_sha256_rejects_replacement_even_with_equal_bytes_and_link_count_change(
    tmp_path, monkeypatch
):
    import os
    from cv.pipeline import provenance

    path = tmp_path / "native.jpg"
    path.write_bytes(b"same bytes")
    stamp = path.stat()
    replacement = tmp_path / "replacement.jpg"
    replacement.write_bytes(path.read_bytes())
    os.utime(replacement, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    os.link(replacement, tmp_path / "new_viewer.jpg")
    original = provenance._read_sha256

    def replace(path, identity):
        digest = original(path, identity)
        replacement.replace(path)
        return digest

    monkeypatch.setattr(provenance, "_read_sha256", replace)
    with pytest.raises(provenance.ProvenanceError, match="changed while hashing"):
        provenance.file_sha256(path)


def test_sha256_repeated_link_race_is_bounded(tmp_path, monkeypatch):
    import os
    from cv.pipeline import provenance

    path = tmp_path / "native.jpg"
    path.write_bytes(b"same bytes")
    original = provenance._read_sha256
    calls = []

    def link_every_read(path, identity):
        digest = original(path, identity)
        calls.append(digest)
        os.link(path, tmp_path / f"viewer_{len(calls)}.jpg")
        return digest

    monkeypatch.setattr(provenance, "_read_sha256", link_every_read)
    with pytest.raises(provenance.ProvenanceError, match="changed repeatedly"):
        provenance.file_sha256(path)
    assert len(calls) == 4
    assert not provenance._HASH_PENDING
