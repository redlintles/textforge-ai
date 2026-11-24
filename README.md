# TextForge AI

IA que lê um pedido em inglês e escreve um pipeline [TextForge](https://crates.io/crates/textforge)
para transformar um texto, em vez de reescrever o texto diretamente.

```
"Replace every "apple" with "pear", then uppercase the text."
        │
        ▼  modelo
raw apple pear;
tua;
        │
        ▼  textforgers (engine real)
texto transformado
```

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # só para treinar/usar o modelo
pip install -e ".[dev,train]"
```

## Uso

```bash
python -m textforge_ai gen-data --n 20000 --out data          # dataset sintético
python -m textforge_ai baseline oracle --data data/test.jsonl  # sanidade das métricas
python -m textforge_ai eval --data data/test.jsonl --pred preds.jsonl
python -m textforge_ai run pipeline.tft entrada.txt            # aplica um pipeline
python -m textforge_ai train --data data --out runs/codet5-small  # treina (ver docs/04)
python -m textforge_ai solve --model runs/codet5-small/best --input texto.txt
python -m textforge_ai verify --model runs/codet5-small/best --input texto.txt --pipeline p.tft
pytest                                                          # testes
```

## Estrutura

```
src/textforge_ai/
  engine.py     wrapper do textforgers: parse, execução, erros estruturados, forma canônica
  catalog.py    instruções permitidas + templates em inglês
  texts.py      gerador de textos de entrada ("probes")
  synth.py      gera e verifica exemplos (pedido → pipeline)
  dataset.py    geração em lote, divisão por pipeline, JSONL
  guards.py     detectores de "trapaça" (apagar o texto e escrever a resposta)
  evaluate.py   métricas
  baselines.py  preditores de referência (oracle / identity / cheater)
  verify_data.py exemplos do verificador: mutações rotuladas por execução
  tasks.py      formato de texto das tarefas generate/verify
  model.py      carregar CodeT5 ou criar o modelo minúsculo de teste
  train.py      laço de treino multitarefa em PyTorch
  infer.py      gerar, executar, validar e escolher o melhor pipeline
  cli.py        linha de comando
tests/          pytest
docs/           abordagem, dataset, avaliação, decisões, diário
```

## Documentação

1. [Abordagem](docs/01-abordagem.md): por que a IA não vai só substituir o texto, e o roteiro
2. [Dataset](docs/02-dataset.md)
3. [Avaliação](docs/03-avaliacao.md)
4. [Treinamento](docs/04-treinamento.md): passo a passo para treinar no seu PC
- [Decisões](docs/decisoes.md) · [Diário](docs/diario.md)

Licença: GPL-3.0 (mesma do TextForge).
