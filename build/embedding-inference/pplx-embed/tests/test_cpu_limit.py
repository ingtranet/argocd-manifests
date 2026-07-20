from server.embedder import cgroup_cpu_limit, effective_cpu_count


def _write(tmp_path, rel, text):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def test_cgroup_v2_quota(tmp_path):
    """cgroup v2: '<quota> <period>' — 400000/100000 = 4 CPU"""
    _write(tmp_path, "cpu.max", "400000 100000\n")
    assert cgroup_cpu_limit(str(tmp_path)) == 4.0


def test_cgroup_v2_unlimited(tmp_path):
    _write(tmp_path, "cpu.max", "max 100000\n")
    assert cgroup_cpu_limit(str(tmp_path)) is None


def test_cgroup_v1_quota(tmp_path):
    _write(tmp_path, "cpu/cpu.cfs_quota_us", "250000\n")
    _write(tmp_path, "cpu/cpu.cfs_period_us", "100000\n")
    assert cgroup_cpu_limit(str(tmp_path)) == 2.5


def test_cgroup_v1_unlimited(tmp_path):
    _write(tmp_path, "cpu/cpu.cfs_quota_us", "-1\n")
    _write(tmp_path, "cpu/cpu.cfs_period_us", "100000\n")
    assert cgroup_cpu_limit(str(tmp_path)) is None


def test_missing_cgroup_files(tmp_path):
    assert cgroup_cpu_limit(str(tmp_path)) is None


def test_effective_count_rounds_down_but_never_zero(tmp_path):
    _write(tmp_path, "cpu.max", "250000 100000\n")  # 2.5 CPU
    assert effective_cpu_count(str(tmp_path)) == 2

    _write(tmp_path, "cpu.max", "50000 100000\n")  # 0.5 CPU
    assert effective_cpu_count(str(tmp_path)) == 1


def test_effective_count_falls_back_to_host_when_unlimited(tmp_path, monkeypatch):
    _write(tmp_path, "cpu.max", "max 100000\n")
    monkeypatch.setattr("server.embedder.os.cpu_count", lambda: 16)
    assert effective_cpu_count(str(tmp_path)) == 16


def test_effective_count_never_exceeds_host(tmp_path, monkeypatch):
    """limit이 호스트보다 크게 설정돼도 호스트 코어 수를 넘기지 않는다."""
    _write(tmp_path, "cpu.max", "3200000 100000\n")  # 32 CPU
    monkeypatch.setattr("server.embedder.os.cpu_count", lambda: 8)
    assert effective_cpu_count(str(tmp_path)) == 8
