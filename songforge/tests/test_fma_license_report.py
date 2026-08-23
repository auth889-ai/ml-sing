"""FMA license census: bucket classification and the three-row-header parser."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "fma_license_report", ROOT / "scripts" / "fma_license_report.py"
)
fma = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fma)


class TestClassify:
    def test_nc_beats_everything(self):
        assert fma.classify("Attribution-NonCommercial-ShareAlike 3.0") == "cc_by_nc"
        assert fma.classify("http://creativecommons.org/licenses/by-nc-nd/3.0/") == "cc_by_nc"

    def test_nd_without_nc(self):
        assert fma.classify("Attribution-NoDerivatives 4.0") == "cc_by_nd"

    def test_sa_without_nc(self):
        assert fma.classify("Attribution-ShareAlike 3.0 United States") == "cc_by_sa"

    def test_plain_by(self):
        assert fma.classify("Attribution 3.0 International") == "cc_by"
        assert fma.classify("http://creativecommons.org/licenses/by/4.0/") == "cc_by"

    def test_cc0_and_public_domain(self):
        assert fma.classify("CC0 1.0 Universal") == "cc0_public_domain"
        assert fma.classify("Public Domain Mark 1.0") == "cc0_public_domain"
        assert fma.classify("creativecommons.org/publicdomain/zero/1.0/") == "cc0_public_domain"

    def test_unknown_and_empty(self):
        assert fma.classify("FMA-Limited: Download Only") == "other_unknown"
        assert fma.classify("") == "other_unknown"


class TestEndToEnd:
    def test_census_on_synthetic_tracks_csv(self, tmp_path):
        header0 = ",track,track,track,track,artist"
        header1 = ",license,duration,bit_rate,genre_top,name"
        header2 = "track_id,,,,,"
        rows = [
            "1,Attribution 4.0,120,320000,Rock,A",
            "2,CC0 1.0 Universal,60,128000,Jazz,B",
            "3,Attribution-NonCommercial 3.0,240,320000,Rock,C",
            "4,Attribution-ShareAlike 3.0,180,192000,Folk,D",
        ]
        path = tmp_path / "tracks.csv"
        path.write_text("\n".join([header0, header1, header2, *rows]) + "\n", encoding="utf-8")
        out = tmp_path / "report.json"
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "fma_license_report.py"),
             "--tracks-csv", str(path), "--output", str(out)],
            check=True, capture_output=True,
        )
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["total_tracks"] == 4
        assert report["per_bucket"]["cc_by"]["tracks"] == 1
        assert report["per_bucket"]["cc0_public_domain"]["tracks"] == 1
        assert report["per_bucket"]["cc_by_nc"]["tracks"] == 1
        accepted = report["deployable_default"]
        assert accepted["tracks"] == 2
        assert accepted["hours"] == round(180 / 3600, 1)
        assert "Rock" in accepted["genre_top_distribution"]
        flagged = report["flagged_cc_by_sa"]
        assert flagged["tracks"] == 1
