# 05 — Generalização: do "decorou os templates" para pedidos livres

## O problema (2026-11-24)

O treino completo deu 100% na validação, mas o modelo falhava em quase qualquer pedido
escrito de outro jeito. A validação vinha do mesmo gerador que o treino, então 100% ali
significava só "aprendeu os templates".

Medição honesta, com 80 pedidos escritos à mão (`ood.py`, comando `ood-eval`):

| categoria | modelo 1 (codet5-small) |
|---|---|
| paraphrase (40) | 78% |
| format (9) | 67% |
| compose (18) | 28% |
| literal (9) | 22% |
| semantic (4) | 0% |
| **total (80)** | **55%** (81% de pipelines válidos) |

## Duas causas, as duas nos dados

**1. Frases.** Cada instrução tinha 2–3 frases fixas e só 4 jeitos de encadear passos.
Nada de aspas simples, "the input", "index 2" ou "a pipeline that ...".

**2. Valores literais.** `ate`/`atb` só tinham visto 10 literais fixos (`" END"`, `"--"`, ...),
e os padrões só vinham de 64 palavras. O modelo decorou essas strings em vez de aprender a
*copiar* o valor do pedido. Os erros mostram isso: `mango` → `morango`, `laranja` →
`laranaja`, `Terry Lee` → `Terry LEAR`.

Mais épocas com os mesmos dados não resolveriam nenhuma das duas.

## O que mudou no gerador

- `phrasing.py`: 5–13 frases extras por instrução, com formas `index N` (0-based, como o
  TextForge) e `number N` (1-based); aspas duplas, simples, nenhuma, “curvas” e `crase`;
  nomes de valores ("a space", "a period"); sinônimos de "the text" (input, string,
  content, file); ~25 formas de encadear passos, listas e "a pipeline that ..." (com o verbo
  conjugado); ruído: minúsculas, MAIÚSCULAS, sem pontuação, espaço duplo, erros de digitação.
  O ruído nunca altera os valores entre aspas.
- `lexicon.py`: literais de ~150 palavras (inclusive em português e outras línguas), nomes
  ("Maria Silva"), frases, símbolos, datas, versões e pseudopalavras aleatórias
  ("zorbel"), que não dá para decorar.
- 25% dos pedidos continuam com as frases originais e 40% dos valores com os pools originais,
  para o modelo não esquecer o que já sabia.

## Re-treino

```bash
bash scripts/treinar.sh variado        # gera data_v2 (30k) e treina 1 época
tail -f train-v2.log
```

Começa do modelo atual (`runs/codet5-small/best`), não do zero, com `lr` 2e-4 e 1 época.
São ~27k exemplos → ~105k pares → **~6,5–7,5 h** no Ryzen 5 4600G. Salva em
`runs/codet5-v2/`, e o `./tfai` passa a usar esse modelo automaticamente. No fim, o script
roda o `ood-eval` e grava o resultado no mesmo log. `parar`/`continuar` funcionam como antes.

## Ressalvas honestas

- O conjunto OOD e o gerador novo foram escritos pela mesma pessoa (o Claude), com o OOD
  escrito **antes**. Um teste garante que nenhum pedido do OOD aparece no treino, mas parte
  do vocabulário se sobrepõe, então o OOD deve superestimar um pouco. O teste mais honesto
  continua sendo pedidos novos, escritos por você depois do treino.
- A categoria `semantic` ("twice", "a period") foi coberta só em parte (nomes de valores e
  contagens por extenso).
- Continua sendo mundo fechado: só as ~50 instruções do catálogo, só em inglês. Pedidos em
  português exigiriam templates em português.
- Se o OOD ficar bem abaixo de ~85% depois disso, o próximo passo é um modelo que já
  entende linguagem (ex.: Qwen2.5-Coder 0.5B–1.5B Instruct), com este dataset para LoRA.
