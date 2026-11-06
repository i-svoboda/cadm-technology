# Controlled automation of decision making

These are the programs behind my PhD thesis «Методи і технології прийняття рішень в сфері кібербезпеки на основі генеративного штучного інтелекту» (Igor Sikorsky Kyiv Polytechnic Institute, 2026). The thesis calls the technology КАПР, and the English name is CADM.

Це програми до дисертації. Опис технології наведено в розділі 5, а її модулі перелічено в таблиці в репозиторії [dissertation-supplement](https://github.com/i-svoboda/dissertation-supplement).

| Folder | What it does | Thesis chapter |
|:--|:--|:--|
| `chapter2_scenarios` | checks generated scenario documents, compares them with the reference model, ranks scenarios and runs the sensitivity analysis | 2 |
| `chapter3_ahp_anp` | asks virtual experts through the Codex command line tool, then computes AHP and ANP priorities and the portfolio | 3 |
| `chapter4_stability` | runs the repeated questionnaires and estimates variance shares and the probabilities of ranks and pairs | 4 |
| `chapter4_scenario_check` | applies the stability method to the attack scenario variants and compares it with the analysts | 4, 5 |
| `chapter5_controls` | applies the technology to the protection measures of the distribution system operator | 5 |

## What is left out

The reference model, the generated scenarios and the records of model calls are not here, because they describe attack paths on critical infrastructure. Members of the defence council and the official opponents can get them through the department on request. The stored model answers are left out for the same reason, so these programs can't be run from this repository alone.

The code is as it was run, apart from the path to the Codex tool on my computer and one note in `chapter3_ahp_anp/protocol_data.py`. `unmatched_content.py` from chapter 2 is left out because it quotes generated labels. The separate analysis programs in the last row of the module table will come later, after review.

## Licence

MIT, see `LICENSE`.
