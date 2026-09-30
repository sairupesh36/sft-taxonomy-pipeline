# SFT Data — Language-wise Distribution

**Source:** `/projects/data/datasets/traces_team/sft-hf-datasets/full/` (52 shards, 991GB)

**Method:** fastText `lid.176.bin`, confidence threshold 0.5, non-language pre-filter

**Total rows classified:** 570,230,155


## Special buckets (not real languages)

| Bucket | Meaning | Rows | % of total |
|---|---|---:|---:|
| `no_natural_language` | Not natural language (numbers/symbols only) | 144,524,975 | 25.35% |
| `uncertain` | Uncertain (below confidence threshold) | 56,008,941 | 9.82% |

## Real language distribution (122 languages)

| # | Code | Language | Rows | % of total |
|---:|---|---|---:|---:|
| 1 | `en` | English | 357,236,674 | 62.65% |
| 2 | `ko` | Korean | 5,054,973 | 0.89% |
| 3 | `zh` | Chinese | 1,571,143 | 0.28% |
| 4 | `hi` | Hindi | 846,912 | 0.15% |
| 5 | `vi` | Vietnamese | 794,105 | 0.14% |
| 6 | `ja` | Japanese | 606,583 | 0.11% |
| 7 | `nl` | Dutch | 490,840 | 0.09% |
| 8 | `ru` | Russian | 383,791 | 0.07% |
| 9 | `kn` | Kannada | 361,847 | 0.06% |
| 10 | `fr` | French | 256,996 | 0.05% |
| 11 | `es` | Spanish | 224,531 | 0.04% |
| 12 | `de` | German | 217,075 | 0.04% |
| 13 | `pt` | Portuguese | 194,422 | 0.03% |
| 14 | `da` | Danish | 169,047 | 0.03% |
| 15 | `ar` | Arabic | 139,792 | 0.02% |
| 16 | `it` | Italian | 139,288 | 0.02% |
| 17 | `fa` | Persian | 95,680 | 0.02% |
| 18 | `pl` | Polish | 86,331 | 0.02% |
| 19 | `tr` | Turkish | 73,045 | 0.01% |
| 20 | `id` | Indonesian | 71,515 | 0.01% |
| 21 | `bn` | Bengali | 56,351 | 0.01% |
| 22 | `ro` | Romanian | 56,192 | 0.01% |
| 23 | `th` | Thai | 47,411 | 0.01% |
| 24 | `ta` | Tamil | 46,603 | 0.01% |
| 25 | `gu` | Gujarati | 44,206 | 0.01% |
| 26 | `fi` | Finnish | 42,646 | 0.01% |
| 27 | `he` | Hebrew | 40,191 | 0.01% |
| 28 | `te` | Telugu | 37,973 | 0.01% |
| 29 | `cs` | Czech | 28,426 | 0.00% |
| 30 | `ml` | Malayalam | 26,329 | 0.00% |
| 31 | `mr` | Marathi | 26,052 | 0.00% |
| 32 | `pa` | Punjabi | 25,296 | 0.00% |
| 33 | `el` | Greek | 15,255 | 0.00% |
| 34 | `ga` | Irish | 12,742 | 0.00% |
| 35 | `sv` | Swedish | 12,100 | 0.00% |
| 36 | `hu` | Hungarian | 11,804 | 0.00% |
| 37 | `bg` | Bulgarian | 11,709 | 0.00% |
| 38 | `km` | Khmer | 10,002 | 0.00% |
| 39 | `ca` | Catalan | 8,563 | 0.00% |
| 40 | `lt` | Lithuanian | 8,524 | 0.00% |
| 41 | `sk` | Slovak | 7,472 | 0.00% |
| 42 | `lv` | Latvian | 7,324 | 0.00% |
| 43 | `et` | Estonian | 7,025 | 0.00% |
| 44 | `sl` | Slovenian | 6,697 | 0.00% |
| 45 | `uk` | Ukrainian | 6,587 | 0.00% |
| 46 | `ceb` | Cebuano | 6,457 | 0.00% |
| 47 | `or` | or | 6,395 | 0.00% |
| 48 | `eo` | Esperanto | 5,903 | 0.00% |
| 49 | `mt` | Maltese | 5,388 | 0.00% |
| 50 | `ur` | Urdu | 5,340 | 0.00% |
| 51 | `yi` | Yiddish | 5,237 | 0.00% |
| 52 | `tl` | Tagalog | 4,713 | 0.00% |
| 53 | `gl` | Galician | 4,578 | 0.00% |
| 54 | `cy` | Welsh | 4,498 | 0.00% |
| 55 | `ne` | Nepali | 4,211 | 0.00% |
| 56 | `hr` | Croatian | 4,145 | 0.00% |
| 57 | `as` | as | 3,752 | 0.00% |
| 58 | `ms` | Malay | 3,341 | 0.00% |
| 59 | `si` | Sinhala | 3,311 | 0.00% |
| 60 | `mzn` | Mazanderani | 2,179 | 0.00% |
| 61 | `my` | Burmese | 1,482 | 0.00% |
| 62 | `no` | Norwegian | 967 | 0.00% |
| 63 | `sw` | Swahili | 776 | 0.00% |
| 64 | `sq` | Albanian | 770 | 0.00% |
| 65 | `eu` | Basque | 711 | 0.00% |
| 66 | `ku` | ku | 607 | 0.00% |
| 67 | `gom` | Konkani | 536 | 0.00% |
| 68 | `yo` | Yoruba | 461 | 0.00% |
| 69 | `sa` | Sanskrit | 454 | 0.00% |
| 70 | `sr` | Serbian | 208 | 0.00% |
| 71 | `az` | Azerbaijani | 205 | 0.00% |
| 72 | `uz` | Uzbek | 202 | 0.00% |
| 73 | `ps` | Pashto | 160 | 0.00% |
| 74 | `lo` | Lao | 118 | 0.00% |
| 75 | `af` | Afrikaans | 83 | 0.00% |
| 76 | `mg` | mg | 80 | 0.00% |
| 77 | `kk` | Kazakh | 78 | 0.00% |
| 78 | `mk` | Macedonian | 73 | 0.00% |
| 79 | `hy` | Armenian | 60 | 0.00% |
| 80 | `mn` | Mongolian | 53 | 0.00% |
| 81 | `arz` | arz | 50 | 0.00% |
| 82 | `be` | Belarusian | 50 | 0.00% |
| 83 | `ky` | Kyrgyz | 47 | 0.00% |
| 84 | `ka` | Georgian | 45 | 0.00% |
| 85 | `la` | Latin | 44 | 0.00% |
| 86 | `is` | Icelandic | 41 | 0.00% |
| 87 | `wuu` | Wu Chinese | 37 | 0.00% |
| 88 | `sh` | Serbo-Croatian | 32 | 0.00% |
| 89 | `fy` | Frisian | 26 | 0.00% |
| 90 | `ht` | ht | 24 | 0.00% |
| 91 | `lb` | Luxembourgish | 21 | 0.00% |
| 92 | `tt` | Tatar | 19 | 0.00% |
| 93 | `gn` | Guarani | 18 | 0.00% |
| 94 | `tk` | Turkmen | 18 | 0.00% |
| 95 | `bh` | Bihari | 17 | 0.00% |
| 96 | `new` | new | 17 | 0.00% |
| 97 | `jbo` | Lojban | 14 | 0.00% |
| 98 | `jv` | Javanese | 14 | 0.00% |
| 99 | `azb` | azb | 11 | 0.00% |
| 100 | `sd` | Sindhi | 11 | 0.00% |
| 101 | `su` | Sundanese | 10 | 0.00% |
| 102 | `ckb` | ckb | 9 | 0.00% |
| 103 | `nn` | Norwegian Nynorsk | 9 | 0.00% |
| 104 | `gd` | Scottish Gaelic | 8 | 0.00% |
| 105 | `am` | Amharic | 7 | 0.00% |
| 106 | `io` | Ido | 6 | 0.00% |
| 107 | `bo` | Tibetan | 5 | 0.00% |
| 108 | `ug` | Uyghur | 4 | 0.00% |
| 109 | `ba` | Bashkir | 3 | 0.00% |
| 110 | `br` | Breton | 3 | 0.00% |
| 111 | `tg` | Tajik | 3 | 0.00% |
| 112 | `ia` | Interlingua | 2 | 0.00% |
| 113 | `pnb` | Western Punjabi | 2 | 0.00% |
| 114 | `sah` | Sakha | 2 | 0.00% |
| 115 | `bs` | Bosnian | 1 | 0.00% |
| 116 | `ce` | Chechen | 1 | 0.00% |
| 117 | `ilo` | Ilocano | 1 | 0.00% |
| 118 | `krc` | Karachay-Balkar | 1 | 0.00% |
| 119 | `nds` | Low German | 1 | 0.00% |
| 120 | `oc` | Occitan | 1 | 0.00% |
| 121 | `os` | Ossetian | 1 | 0.00% |
| 122 | `war` | Waray | 1 | 0.00% |

## Summary

- English: 357,236,674 (62.65%)
- All other real languages combined: 12,459,565 (2.19%)
- Non-language / uncertain buckets combined: 200,533,916 (35.17%)
