# 03 — Avaliação

Código: `src/textforge_ai/{evaluate,guards,baselines}.py`.

## Métricas

| Métrica | Significado |
|---|---|
| `valid_rate` | o pipeline previsto passa pelo parser real do TextForge |
| `exec_rate` | roda sem erro em todos os probes |
| `probe_accuracy` | **métrica principal**: acerta a saída de *todos* os probes e não é suspeito |
| `exact_match` | texto canônico idêntico ao gabarito (mais rígida: `tla; tua;` ≠ `tua;`) |
| `suspicious_rate` | marcado pelos guardas anti-trapaça |

`probe_accuracy` aceita pipelines diferentes do gabarito que façam a mesma coisa. É o
que importa para o usuário; `exact_match` serve para diagnóstico.

## Guardas (`guards.py`)

- **Independência da entrada**: executa nos probes do exemplo; se textos diferentes
  dão a mesma saída não vazia, o pipeline ignora a entrada. Usa os probes (e não
  textos aleatórios) porque um `klm tiger` correto esvazia qualquer texto sem "tiger"
  e geraria alarme falso. Esse caso foi encontrado nos testes e está coberto por
  `test_filter_then_constant_is_not_a_false_alarm`.
- **Literal vazado**: string de 12+ caracteres no pipeline que aparece no texto de
  entrada ou na saída esperada, mas não no pedido.

## Baselines (sanidade das métricas)

```bash
python -m textforge_ai baseline oracle   --data data/test.jsonl
python -m textforge_ai baseline identity --data data/test.jsonl
python -m textforge_ai baseline cheater  --data data/test.jsonl
```

Resultados em 2026-11-18 (982 exemplos de teste):

| | valid | exec | probe_acc | exact | suspicious |
|---|---|---|---|---|---|
| oracle | 1.00 | 1.00 | **1.00** | 1.00 | 0.00 |
| identity | 1.00 | 1.00 | **0.00** | 0.00 | 0.00 |
| cheater | 1.00 | 1.00 | **0.00** | 0.00 | **1.00** |

## Avaliando um modelo

O modelo grava um JSONL com `{"id": ..., "pipeline": ...}` por linha:

```bash
python -m textforge_ai eval --data data/test.jsonl --pred preds.jsonl --details report.jsonl
```

`report.jsonl` traz, por exemplo, o erro estruturado do engine (`stage`, `kind`,
`message`, `hint`). Essa é a matéria-prima do laço de autocorreção (fase 4).
