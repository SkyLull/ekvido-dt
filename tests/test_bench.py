from ekvido.cli import main


def test_bench_prints_stage_timings(capsys):
    assert main(["bench", "--device", "cpu", "--sizes", "0.05"]) == 0
    out = capsys.readouterr().out
    assert "mist" in out and "kazuki" in out and "colour-only" in out and "glow" in out
