# 02 — Dataset sintético

Não existe dataset público de "pedido em inglês → pipeline TextForge", então ele é
gerado. Código: `src/textforge_ai/{catalog,texts,synth,dataset}.py`.

## Como um exemplo nasce

1. Sorteia de 1 a 4 instruções do catálogo (`catalog.py`, 48 instruções na v1), com
   argumentos. Instruções que "achatam" o texto (`jsnc`, `joinln`, `rmws`, `urle`...)
   só podem ser a última etapa (`TERMINAL` em `synth.py`).
2. Monta o pipeline com o **builder oficial** do `textforgers` e serializa com o
   próprio engine. A sintaxe nunca é reimplementada em Python.
3. Gera 6 textos em inglês ("probes") com bagunça realista: linhas vazias, linhas
   duplicadas, espaços nas pontas, maiúsculas. As palavras usadas como padrão são
   injetadas pelo menos duas vezes em cada texto.
4. Executa etapa por etapa. **Descarta** o exemplo se alguma etapa der erro ou não
   mudar nenhum probe (descrever uma etapa inútil ensinaria o modelo a inventar
   etapas), ou se a saída final for vazia.
5. Descreve as etapas em inglês com os templates do catálogo e um de quatro estilos
   de frase (`First, ..., then ..., and finally ...`, frases separadas, lista
   numerada, `..., then ...`).

Convenção importante: humanos contam a partir de 1 ("the second word"), o TextForge
a partir de 0 (`tucw 1`). A conversão está no gerador, então o modelo precisa aprendê-la.

## Formato (JSONL, um exemplo por linha)

```json
{
  "id": "s0-17",
  "prompt": "1. Change all occurrences of \"sings\" to \"happy\"\n2. Replace the first 2 occurrences of \"village\" with \"BOOK\"",
  "pipeline": "raw sings happy;\nrcw village BOOK 2;",
  "steps": ["raw", "rcw"],
  "probes": [{"input": "...", "output": "..."}]
}
```

## Divisão treino/validação/teste

`split_by_pipeline` garante que **um mesmo pipeline nunca aparece em dois conjuntos**.
Sem isso, o teste teria pipelines que o modelo decorou no treino (só a frase mudaria)
e a nota superestimaria a capacidade de generalizar.

## Comando

```bash
python -m textforge_ai gen-data --n 20000 --seed 0 --out data
```

Gera ~18k/950/980 exemplos em ~5 s. É determinístico pela seed, então `data/` não vai
para o git: basta regenerar.

## Lacunas conhecidas (v1)

- **Frases engessadas**: só as variações dos templates. Um modelo treinado só nisso vai
  estranhar pedidos escritos livremente. Plano: paráfrases (fase 5).
- **`htmle` quase nunca aparece** (4 exemplos em 18k): os textos não têm `<`, `>`, `&`,
  então a instrução quase sempre é inútil e é descartada. Corrigir colocando esses
  caracteres em alguns probes.
- **Padrões só com palavras simples**: os padrões são regex no TextForge, mas a v1 usa
  apenas palavras alfabéticas. Regex de verdade fica para depois.
- **Sem variáveis, blocos, `iter`, `onln`** etc. A v1 cobre só transformações diretas.
