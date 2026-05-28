from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.qwen36_vl_quality import (
    build_cases,
    build_payload,
    quality_flags,
    write_markdown,
)


def test_build_cases_covers_text_single_and_multi_image(tmp_path: Path):
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    (fixture_dir / "coco_sample.jpg").write_bytes(b"jpg")
    multi = fixture_dir / "multi_image"
    multi.mkdir()
    (multi / "image_0.jpg").write_bytes(b"jpg0")
    (multi / "image_1.jpg").write_bytes(b"jpg1")

    cases = build_cases(fixture_dir)

    assert [case.case_id for case in cases] == [
        "text_baseline",
        "single_image_cats",
        "multi_image_kitchen_street",
    ]
    assert cases[0].image_paths == []
    assert len(cases[1].image_paths) == 1
    assert len(cases[2].image_paths) == 2


def test_quality_flags_reject_blank_and_repetitive_outputs():
    assert "blank_output" in quality_flags("single_image_cats", "        ")
    assert "repetitive_output" in quality_flags("single_image_cats", "////////////////////")
    assert "repetitive_output" in quality_flags("text_baseline", ",, as,as,as,as,as,as,as,")
    assert quality_flags("single_image_cats", "Two cats are lying on a pink couch.") == []


def test_build_payload_embeds_images_as_openai_parts(tmp_path: Path):
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"abc")
    payload = build_payload(
        model="qwen3_5_moe",
        prompt="Describe this image.",
        image_paths=[image_path],
        max_tokens=16,
        stream=False,
    )

    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Describe this image."}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_write_markdown_summarizes_failures(tmp_path: Path):
    records = [
        {
            "engine": "ironmlx",
            "case_id": "single_image_cats",
            "status_code": 200,
            "finish_reason": "length",
            "output_text": "        ",
            "quality_flags": ["blank_output"],
            "elapsed_ms": 10.0,
        }
    ]
    out = tmp_path / "report.md"
    write_markdown(records, out)

    text = out.read_text()
    assert "| ironmlx | single_image_cats | FAIL |" in text
    assert "blank_output" in text
