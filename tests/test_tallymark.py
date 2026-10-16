import base64
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tallymark.extract import Extractor, image_url  # noqa: E402
from tallymark.schemas import Receipt  # noqa: E402
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
