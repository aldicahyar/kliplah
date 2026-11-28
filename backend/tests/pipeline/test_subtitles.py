import pytest

from app.core.models import SubtitleLine, Word
from app.pipeline.subtitles import ass_color, build_ass, group_lines, load_presets, sanitize_ass_text
from tests.helpers import W


def texts(lines):
    return [l.text for l in lines]


def test_group_lines_max_words_and_relative_times():
    ws = W("satu dua tiga empat lima enam tujuh delapan sembilan", start=10.0)
    lines = group_lines(ws, clip_start=10.0, clip_end=20.0)
    assert [len(l.text.split()) for l in lines] == [4, 4, 1]
    assert lines[0].start == pytest.approx(0.0) and lines[0].id == 1
    assert lines[1].words[0].start == pytest.approx(1.2)  # word times relative too


def test_group_lines_breaks_on_comma_and_connector():
    lines = group_lines(W("aku setuju, tapi menurutku itu salah"), 0.0, 10.0)
    assert texts(lines) == ["aku setuju,", "tapi menurutku itu salah"]


def test_connector_needs_two_words_before_it():
    lines = group_lines(W("dan aku pergi"), 0.0, 10.0)
    assert texts(lines) == ["dan aku pergi"]


def test_group_lines_breaks_on_sentence_end_and_pause():
    a = W("oke. jadi gitu")
    b = W("lanjut ya", start=a[-1].end + 0.5)
    assert texts(group_lines(a + b, 0.0, 10.0)) == ["oke.", "jadi gitu", "lanjut ya"]


def test_group_lines_only_words_inside_clip():
    ws = W("sebelum klip mulai di sini", start=0.0)  # 0.0 .. 1.45
    lines = group_lines(ws, clip_start=0.55, clip_end=1.0)
    assert texts(lines) == ["mulai di"]


def test_group_lines_preserves_prob_and_approx():
    ws = [
        Word(w="aku", start=1.0, end=1.3, prob=0.45, approx=False),
        Word(w="pergi", start=1.3, end=1.6, prob=None, approx=True),
    ]
    lines = group_lines(ws, clip_start=0.5, clip_end=3.0)
    assert len(lines) == 1
    assert [(w.prob, w.approx) for w in lines[0].words] == [(0.45, False), (None, True)]


def test_ass_color():
    assert ass_color("#FFE14D") == "&H004DE1FF"


def test_sanitize():
    assert sanitize_ass_text("a {b} \\N c\nd") == "a b /N c d"


def test_ass_header_uses_canvas_resolution():
    a = build_ass([], "Hook", 3.0, "klasik", load_presets())
    assert "PlayResX: 1080" in a and "PlayResY: 1920" in a and "Montserrat ExtraBold" in a


def test_ass_sanitizes_braces_and_backslashes():
    line = SubtitleLine(id=1, start=0, end=1, text=r"halo {\b1} \N dunia", words=[])
    a = build_ass([line], 'kata "kunci" {x}', 3.0, "klasik", load_presets())
    events = [l for l in a.splitlines() if l.startswith("Dialogue")]
    assert events and all("{\\b1}" not in e and "{x}" not in e and "\\N dunia" not in e for e in events)


def test_hook_event_duration_and_wrap():
    a = build_ass([], "Gaji dua digit tapi tetap bokek terus?", 3.0, "klasik", load_presets())
    hook = [l for l in a.splitlines() if l.startswith("Dialogue") and ",Hook," in l]
    assert len(hook) == 1 and ",0:00:00.00,0:00:03.00,Hook," in hook[0]
    assert "\\N" in hook[0]  # wrapped at 18 chars


def test_klasik_one_event_per_line():
    line = SubtitleLine(id=1, start=1.0, end=2.0, text="aku setuju banget",
                        words=[Word(w="aku", start=1.0, end=1.3), Word(w="setuju", start=1.3, end=1.6),
                               Word(w="banget", start=1.6, end=2.0)])
    a = build_ass([line], "", 3.0, "klasik", load_presets())
    subs = [l for l in a.splitlines() if l.startswith("Dialogue") and ",Sub," in l]
    assert len(subs) == 1 and subs[0].endswith(",aku setuju banget")


def test_sorot_kata_one_event_per_word():
    line = SubtitleLine(id=1, start=1.0, end=2.0, text="aku setuju banget",
                        words=[Word(w="aku", start=1.0, end=1.3), Word(w="setuju", start=1.3, end=1.6),
                               Word(w="banget", start=1.6, end=2.0)])
    a = build_ass([line], "", 3.0, "sorot_kata", load_presets())
    subs = [l for l in a.splitlines() if l.startswith("Dialogue") and ",Sub," in l]
    assert len(subs) == 3
    assert all(s.count("{\\c&H004DE1FF&}") == 1 for s in subs)
    assert subs[1].endswith("aku {\\c&H004DE1FF&}setuju{\\r} banget")
    assert ",0:00:01.30,0:00:01.60,Sub," in subs[1]


def test_bilingual_ass_renders_original_above_translation():
    from app.pipeline.subtitles import ORIGINAL_FONT_SCALE

    assert ORIGINAL_FONT_SCALE == pytest.approx(0.72)
    presets = load_presets()
    presets = {
        **presets,
        "presets": {k: {**v, "size": 104} for k, v in presets["presets"].items()},
    }
    words = [Word(w="halo", start=0.5, end=1.0), Word(w="semua", start=1.0, end=2.0)]
    bilingual = SubtitleLine(id=1, start=0.5, end=2.0, text="halo semua", words=words, original="hello everyone")
    plain = SubtitleLine(id=1, start=0.5, end=2.0, text="halo semua", words=words)

    for preset in ("klasik", "sorot_kata"):
        ass = build_ass([bilingual], "", 3.0, preset, presets)
        subs = [l for l in ass.splitlines() if l.startswith("Dialogue") and ",Sub," in l]
        assert len(subs) == 1
        assert subs[0].endswith(r",{\fs75}hello everyone{\r}\Nhalo semua")
        assert build_ass([SubtitleLine(id=1, start=0.5, end=2.0, text="halo semua", words=words, original=None)],
                         "", 3.0, preset, presets) == build_ass([plain], "", 3.0, preset, presets)


def test_bilingual_ass_sanitizes_both_lines():
    presets = load_presets()
    presets = {
        **presets,
        "presets": {k: {**v, "size": 104} for k, v in presets["presets"].items()},
    }
    line = SubtitleLine(
        id=1,
        start=0.0,
        end=1.5,
        text="teks {\\b1} \\N \\akhir\nbaris",
        words=[Word(w="teks", start=0.0, end=1.5)],
        original="asli {tag} \\N \\awal\nsatu",
    )
    for preset in ("klasik", "sorot_kata"):
        ass = build_ass([line], "", 3.0, preset, presets)
        subs = [l for l in ass.splitlines() if l.startswith("Dialogue") and ",Sub," in l]
        assert len(subs) == 1
        payload = subs[0].split(",", 9)[9]
        assert payload == r"{\fs75}asli tag /N /awal satu{\r}\Nteks /b1 /N /akhir baris"
        assert "{\\b1}" not in payload and "{tag}" not in payload


@pytest.mark.parametrize("blank_orig", ["{}", "{   }", "   ", "{\n}"])
def test_bilingual_ass_falls_back_to_single_line_when_sanitized_original_blank(blank_orig):
    presets = load_presets()
    words = [Word(w="halo", start=0.0, end=0.5), Word(w="dunia", start=0.5, end=1.0)]
    line = SubtitleLine(id=1, start=0.0, end=1.0, text="halo dunia", words=words, original=blank_orig)

    ass_klasik = build_ass([line], "", 3.0, "klasik", presets)
    subs_klasik = [l for l in ass_klasik.splitlines() if l.startswith("Dialogue") and ",Sub," in l]
    assert len(subs_klasik) == 1
    assert subs_klasik[0].endswith(",halo dunia")
    assert r"\N" not in subs_klasik[0] and r"\fs" not in subs_klasik[0]

    ass_sorot = build_ass([line], "", 3.0, "sorot_kata", presets)
    subs_sorot = [l for l in ass_sorot.splitlines() if l.startswith("Dialogue") and ",Sub," in l]
    assert len(subs_sorot) == 2
    assert all(r"\N" not in s and r"\fs" not in s for s in subs_sorot)
