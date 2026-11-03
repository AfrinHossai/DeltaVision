"""Guard repository documentation structure and local links."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_research_report_contains_required_sections_and_exposure_note() -> None:
    report = (ROOT / "docs/research_report.md").read_text(encoding="utf-8")
    required_sections = (
        "## Abstract",
        "## 1. Introduction",
        "## 2. Research Questions",
        "## 3. Related Work",
        "## 4. Methodology",
        "## 5. Experimental Setup",
        "## 6. Baselines",
        "## 7. Results",
        "## 8. Ablation Studies",
        "## 9. Failure Analysis",
        "## 10. Discussion",
        "## 11. Limitations and Threats to Validity",
        "## 12. Conclusion",
        "## References",
        "## Reproducibility Appendix",
    )

    assert all(section in report for section in required_sections)
    assert "previously inspected held-out" in report
    assert "not a blind test" in report
    assert "selective DINOv2 token computation" in report


def test_readme_embedded_local_images_exist() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    image_refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme)
    local_refs = [
        ref for ref in image_refs if not ref.startswith(("https://", "http://"))
    ]

    assert "docs/figures/architecture.png" in local_refs
    assert "docs/demo/deltavision_demo.gif" in local_refs
    assert all((ROOT / ref).is_file() for ref in local_refs)


def test_local_markdown_links_resolve() -> None:
    documents = (
        ROOT / "README.md",
        ROOT / "docs/research_report.md",
        ROOT / "docs/reproducibility.md",
    )
    for document in documents:
        links = re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", document.read_text())
        for link in links:
            if link.startswith(("https://", "http://", "mailto:")):
                continue
            local_path = link.split("#", maxsplit=1)[0]
            if local_path:
                assert (document.parent / local_path).resolve().is_file(), (
                    f"broken Markdown link in {document.relative_to(ROOT)}: {link}"
                )


def test_reproducibility_guide_matches_pinned_model_and_existing_scripts() -> None:
    guide = (ROOT / "docs/reproducibility.md").read_text(encoding="utf-8")
    config = json.loads(
        (ROOT / "configs/pretrained_baseline.json").read_text(encoding="utf-8")
    )
    required_scripts = (
        "build_sequences.py",
        "build_research_benchmark.py",
        "run_baseline.py",
        "run_stateful.py",
        "run_final_evaluation.py",
        "analyze_final_results.py",
        "evaluate_pretrained_caching.py",
        "ablate_pretrained_cache.py",
        "create_research_demo.py",
    )

    assert config["revision"] in guide
    assert config["model_id"] in guide
    assert all((ROOT / "scripts" / script).is_file() for script in required_scripts)
    assert all(script in guide for script in required_scripts)
