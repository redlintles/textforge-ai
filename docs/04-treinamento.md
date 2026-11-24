# 04 — Treinamento

Código: `src/textforge_ai/{tasks,verify_data,model,train,infer}.py`.

## O que é treinado

Um único modelo, `Salesforce/codet5-small` (~60M parâmetros, pré-treinado em código),
aprende duas tarefas ao mesmo tempo, separadas pelo prefixo da entrada (D7):

| Tarefa | Entrada | Saída |
|---|---|---|
| gerar | `generate: <pedido>` | `<pipeline>` |
| verificar | `verify: <pedido> \|\| pipeline: ... \|\| input: ... \|\| output: ...` | `match` ou `mismatch: step N (mnemônico)` |

Para cada exemplo do dataset há 1 par de geração e até 3 de verificação (o pipeline
correto + 2 alterações rotuladas pela execução; ver `verify_data.py`). Com 18k exemplos
de treino, são ~68k pares por época.

## Passo a passo (no seu PC)

### 1. Ambiente

```bash
cd ~/Documentos/textforge_ai
python3 -m venv --clear .venv        # se falhar: sudo apt install python3-venv
source .venv/bin/activate

# PyTorch só para CPU: ~200 MB em vez de ~3 GB da versão CUDA
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,train]"

pytest                                # 135 testes, ~30 s
```

### 2. Dataset

```bash
python -m textforge_ai gen-data --n 20000 --out data
```

### 3. Treino rápido de validação (~20–30 min)

Antes de deixar horas rodando, confirme que tudo funciona com o modelo de verdade
(baixa ~250 MB do Hugging Face na primeira vez):

```bash
python -m textforge_ai train --data data --out runs/teste \
    --max-train 1000 --epochs 1 --eval-examples 100
```

O log mostra `loss` caindo e uma estimativa de tempo (`eta`). No fim de cada época
aparece uma linha `eval {...}` com as métricas na validação.

### 4. Treino completo (~3–5 h em CPU, estimativa)

```bash
nohup python -m textforge_ai train --data data --out runs/codet5-small --epochs 3 \
    > train.log 2>&1 &
tail -f train.log         # acompanhar; Ctrl+C só fecha o tail
```

- Para interromper com segurança: `kill -INT <pid>`. O modelo é salvo em
  `runs/codet5-small/interrupted`.
- Para continuar depois: `--model runs/codet5-small/last` (ou `interrupted`). O otimizador
  recomeça do zero, então use `--lr 1e-4`.
- Cada época salva `last/`; a melhor (por `gen_probe_accuracy`) fica em `best/`, com
  `train_info.json` (configuração + histórico de métricas).

### 5. Avaliar no conjunto de teste

```bash
python -m textforge_ai model-eval --model runs/codet5-small/best --data data/test.jsonl
```

### 6. Usar

```bash
# pedido digitado no terminal (pergunta "prompt>")
python -m textforge_ai solve --model runs/codet5-small/best --input texto.txt

# pedido em arquivo, salvando pipeline e resultado
python -m textforge_ai solve --model runs/codet5-small/best --input texto.txt \
    --prompt-file pedido.txt --save-pipeline saida.tft --save-output saida.txt

# validação exata, quando você sabe a saída esperada
python -m textforge_ai solve ... --expected esperado.txt

# validar um pipeline escrito por você
python -m textforge_ai verify --model runs/codet5-small/best --input texto.txt \
    --pipeline meu.tft --prompt "uppercase everything"
```

`solve` mostra em stderr todos os candidatos (pipeline, erro do engine, veredito,
suspeita de trapaça) e em stdout o melhor: pipeline, veredito e saída.

## Hiperparâmetros (padrões)

| | valor | por quê |
|---|---|---|
| `--epochs` | 3 | dataset sintético é repetitivo; mais épocas tendem a decorar templates |
| `--batch-size` | 16 | cabe folgado em 32 GB de RAM |
| `--lr` | 3e-4 | valor usual para fine-tuning de T5/CodeT5 pequeno |
| warmup | 5% | evita passos grandes no começo |
| `--threads` | todos os núcleos | |

## Como sabemos que o código de treino está certo

Daqui não dá para baixar o CodeT5, então os testes (`tests/test_model.py`) usam um T5
minúsculo com pesos aleatórios e tokenizador treinado na hora (`--tiny`). Eles verificam:

- **overfitting de 8 exemplos**: a geração vai de <100% para 100% de acerto. Se um modelo
  não consegue decorar 8 exemplos, o laço de treino tem bug;
- checkpoints `best/` e `last/` são salvos e recarregados;
- `solve` com saída esperada escolhe o candidato exato;
- o veredito do modelo é uma probabilidade válida;
- `solve`/`verify` funcionam pela CLI.

Experimento à parte (2026-11-18): com 24 exemplos, o modelo minúsculo acertou 70/72
verificações nas alterações vistas no treino e ~60% em alterações novas. Ele decora, mas
não generaliza, o que é esperado para ~0,1M parâmetros. Generalizar é o papel do CodeT5.

## Medição de tempo

Arquitetura idêntica à do `codet5-small` (60,5M parâmetros), lote 16, em 2 núcleos de CPU:
**2,8 s por passo**. Extrapolando para 6 núcleos/12 threads (Ryzen 5 4600G): ~1 s por
passo → ~4,3k passos por época → **~1–1,5 h por época**. É estimativa, não medição: o
primeiro treino curto (passo 3) dá o número real pelo `eta`.

## Lacunas conhecidas

- O verificador vê só os primeiros 400 caracteres da entrada e da saída.
- Não há laço de autocorreção ainda (fase 4): `solve` só escolhe entre os candidatos.
- Sem decodificação restrita (fase 5): o modelo pode gerar mnemônicos inexistentes;
  esses candidatos são descartados pelo parser.
