import base64
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tallymark.cli import expand, record, write  # noqa: E402
from tallymark.extract import Extractor, image_url  # noqa: E402
from tallymark.checks import check, parse_amount  # noqa: E402
from tallymark.schemas import Invoice, Receipt  # noqa: E402
from evals.cord import digits, item_f1  # noqa: E402


def fake_llm(content):
    seen = {}

    def create(**kw):
        seen.update(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), seen


def extractor(content):
    ex = Extractor(base_url="http://unused", model="fake")
    ex.llm, seen = fake_llm(content)
    return ex, seen


def test_image_url_is_a_small_jpeg():
    url = image_url(Image.new("RGB", (4000, 3000), "white"))
    assert url.startswith("data:image/jpeg;base64,")
    img = Image.open(__import__("io").BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert max(img.size) <= 1600


def test_constrained_request_sends_the_schema():
    ex, seen = extractor('{"merchant": "Cafe", "items": [{"name": "Latte", "price": "4.50"}], "total": "4.50"}')
    r = ex.extract(Image.new("RGB", (10, 10)), "receipt")
    assert r.valid and r.data.items[0].name == "Latte" and r.data.total == "4.50"
    assert seen["response_format"]["json_schema"]["schema"] == Receipt.model_json_schema()


def test_free_mode_parses_json_inside_prose_and_reports_bad_output():
    ex, _ = extractor('Sure! Here it is:\n{"total": "12.00"}\nHope that helps.')
    r = ex.extract(Image.new("RGB", (10, 10)), "receipt", constrained=False)
    assert r.valid and r.data.total == "12.00"
    ex, _ = extractor('{"items": "not a list"}')
    r = ex.extract(Image.new("RGB", (10, 10)), "receipt", constrained=False)
    assert not r.valid and r.error


def test_scoring_helpers():
    assert digits("Rp 60.000") == "60000"
    assert item_f1(["Ice Tea", "Nasi Goreng"], ["ICE TEA", "NASI GORENG SPC"]) == 1.0
    assert item_f1(["Coffee"], ["Tea", "Cake"]) == 0.0


def test_batch_export_to_csv_and_jsonl(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.pdf").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("skip me")
    files = list(expand([tmp_path]))
    assert [f.name for f in files] == ["a.jpg", "b.pdf"]

    ex, _ = extractor('{"merchant": "Cafe", "items": [{"name": "Tea", "price": "2.00"}], "total": "2.00"}')
    rec = record("a.jpg", "receipt", ex.extract(Image.new("RGB", (10, 10)), "receipt"))
    bad, _ = extractor('{"items": "nope"}')
    rec_bad = record("b.pdf", "receipt", bad.extract(Image.new("RGB", (10, 10)), "receipt", constrained=False))

    write([rec, rec_bad], tmp_path / "out.csv")
    rows = (tmp_path / "out.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("file,kind,valid") and "Cafe" in rows[1] and ",1," in rows[1] and "False" in rows[2]
    write([rec, rec_bad], tmp_path / "out.jsonl")
    lines = (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and __import__("json").loads(lines[0])["items"][0]["name"] == "Tea"


def test_parse_amount_handles_common_formats():
    assert str(parse_amount("1,234.56")) == "1234.56"
    assert str(parse_amount("1.234,56")) == "1234.56"
    assert str(parse_amount("Rp 60.000")) == "60000"
    assert str(parse_amount("$4.50")) == "4.50"
    assert str(parse_amount("-12")) == "-12"
    assert parse_amount("n/a") is None and parse_amount(None) is None


def test_parse_amount_reads_short_decimals():
    # thousands groups have three digits, so one or two digits after a single separator are decimals
    assert str(parse_amount("4.5")) == "4.5"
    assert str(parse_amount("12,5")) == "12.5"
    assert str(parse_amount("0.500")) == "0.500"
    assert str(parse_amount("1.234")) == "1234"
    assert str(parse_amount("1.234.567")) == "1234567"


def test_checks_flag_numbers_that_disagree():
    good = Receipt(items=[{"name": "Latte", "price": "4.50"}, {"name": "Bagel", "price": "3.00"}],
                   subtotal="7.50", tax="0.38", total="7.88")
    assert check(good) == []
    misread = good.model_copy(update={"total": "78.80"})
    assert any("total" in i for i in check(misread))
    missing_line = Receipt(items=[{"name": "Latte", "price": "4.50"}], subtotal="7.50", total="7.50")
    assert any("line items" in i for i in check(missing_line))
    inv = Invoice(lines=[{"description": "Hosting", "quantity": 3, "unit_price": "10.00", "amount": "33.00"}],
                  subtotal="33.00", total="33.00")
    assert any("Hosting" in i for i in check(inv))


def test_extraction_reports_needs_review():
    ex, _ = extractor('{"items": [{"name": "Tea", "price": "2.00"}], "total": "20.00"}')
    r = ex.extract(Image.new("RGB", (10, 10)), "receipt")
    assert r.valid and r.needs_review and r.issues
    ex, _ = extractor('{"items": [{"name": "Tea", "price": "2.00"}], "total": "2.00"}')
    r = ex.extract(Image.new("RGB", (10, 10)), "receipt")
    assert r.valid and not r.needs_review


def test_cli_runs_end_to_end(tmp_path, monkeypatch, capsys):
    import tallymark.cli as cli
    ex, _ = extractor('{"merchant": "Cafe", "items": [{"name": "Tea", "price": "2.00"}], "total": "20.00"}')
    monkeypatch.setattr(cli, "Extractor", lambda: ex)
    img = tmp_path / "r.png"
    Image.new("RGB", (10, 10)).save(img)
    monkeypatch.setattr("sys.argv", ["tallymark", str(img), "--out", str(tmp_path / "o.csv")])
    cli.main()
    text = (tmp_path / "o.csv").read_text(encoding="utf-8")
    assert "needs_review" in text.splitlines()[0] and "True" in text.splitlines()[1]


def test_cli_prints_review_flags(tmp_path, monkeypatch, capsys):
    import tallymark.cli as cli
    ex, _ = extractor('{"items": [{"name": "Tea", "price": "2.00"}], "total": "20.00"}')
    monkeypatch.setattr(cli, "Extractor", lambda: ex)
    img = tmp_path / "r.png"
    Image.new("RGB", (10, 10)).save(img)
    monkeypatch.setattr("sys.argv", ["tallymark", str(img)])
    cli.main()
    assert '"needs_review": true' in capsys.readouterr().out
