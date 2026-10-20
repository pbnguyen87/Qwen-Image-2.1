# Vietnamese text rendering test

Does Qwen-Image-2.1 render Vietnamese text inside the image?

Nothing in the model's documentation answers this. The card names one language,
English. The original Qwen-Image paper frames its achievement as "alphabetic
languages such as English" plus "logographic languages like Chinese". Vietnamese
is alphabetic, which is the easier of those two categories, but its stacked
diacritics are a separate problem that neither source mentions.

Two capabilities get conflated and must be measured apart:

| capability | how it works | expectation |
|---|---|---|
| understanding a Vietnamese prompt | the Qwen3-VL text encoder, which claims 100+ languages | likely fine |
| rendering Vietnamese text in pixels | the image transformer's learned glyph shapes | unknown |

This test covers the second. Run it:

```bash
python generate.py --prompt-file tests/vietnamese_text_render.txt \
    --turbo --offload --size 1024 --out-dir outputs/vi_text
```

## What to look at

Twelve prompts, ordered easiest to hardest, each asking for one short string on a
plain sign. The target is quoted in the prompt, so compare it against the pixels.

| # | target | probes |
|---|---|---|
| 1-2 | `CAFE`, `HANOI 2026` | ASCII control. If these fail, Vietnamese is not the problem |
| 3-4 | `Cà Phê`, `Bánh Mì` | one diacritic per vowel |
| 5 | `Phở Bò` | horn plus tone on one vowel |
| 6 | `Đà Nẵng` | `Đ`, a letter no other major language has |
| 7-9 | `Tiếng Việt`, `Nguyễn Huệ`, `Kiểm Duyệt` | two combining marks on one vowel, where Latin-trained renderers usually fail |
| 10 | `Ưu tiên kiểm tra sức khỏe` | the full repertoire in one line |
| 11 | `Cà Phê Specialty Coffee` | code-switching inside one sign |
| 12 | `Chào mừng quý khách đến Hà Nội` | whether length degrades the glyphs |

Record for each: the string rendered correctly, rendered with the base letters
right but diacritics wrong or missing, or unreadable. The failure pattern matters
more than the pass rate, because "base letters correct, diacritics dropped" and
"glyphs invented" imply different fixes.

`manifest.jsonl` in the output directory logs the prompt, seed and settings for
every image, so results stay reproducible.
