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

### 3. Treino rápido de validação (~15 min, medido)

Antes de deixar horas rodando, confirme que tudo funciona com o modelo de verdade
(baixa ~250 MB do Hugging Face na primeira vez):

```bash
python -m textforge_ai train --data data --out runs/teste \
    --max-train 1000 --epochs 1 --eval-examples 100
```

O log mostra `loss` caindo e uma estimativa de tempo (`eta`). No fim de cada época
aparece uma linha `eval {...}` com as métricas na validação.

### 4. Treino completo (~9 h com 2 épocas, ~14 h com 3; ver "Medição de tempo")

```bash
bash scripts/treinar.sh completo --epochs 2   # roda em segundo plano, log em train.log
tail -f train.log                              # acompanhar; Ctrl+C só fecha o tail
```

### Parar e retomar

O treino pode ser interrompido a qualquer momento sem perder o progresso:

```bash
bash scripts/treinar.sh parar       # termina o passo atual, salva e sai
bash scripts/treinar.sh continuar   # retoma exatamente de onde parou
```

- A cada 200 passos (~12 min) o estado completo é salvo em `runs/codet5-small/checkpoint/`:
  pesos, otimizador, agenda da taxa de aprendizado, estado do gerador aleatório e posição
  dentro da época. Se o PC desligar ou travar, perde-se no máximo esses ~12 min.
- `Ctrl+C`, `kill` e desligamento normal do sistema (SIGTERM) também salvam antes de sair.
  Um segundo `Ctrl+C` força a saída sem salvar.
- A gravação é atômica: se o PC cair *durante* a gravação, o checkpoint anterior continua
  intacto.
- Ao retomar, as configurações do treino (épocas, lr, lote, dados) vêm do checkpoint, então
  não dá para misturar sem querer.
- Retomar dá **o mesmo resultado** de nunca ter parado: o teste
  `test_resume_continues_exactly` compara os pesos finais dos dois casos.
- Cada época também salva `last/`; a melhor (por `gen_probe_accuracy`) fica em `best/`,
  com `train_info.json` (configuração + histórico de métricas).

### 5. Avaliar no conjunto de teste

```bash
python -m textforge_ai model-eval --model runs/codet5-small/best --data data/test.jsonl
```

### 6. Usar

```bash
./tfai texto.txt "Uppercase the text."     # um pedido; não precisa ativar a .venv
./tfai texto.txt                           # interativo: carrega o modelo uma vez, vários pedidos
./tfai texto.txt "Remove all whitespace." -o saida.txt -p saida.tft   # salva texto e pipeline
./tfai texto.txt "..." -e esperado.txt     # validação exata, quando você sabe a saída
./tfai texto.txt "..." -v                  # mostra todos os candidatos
echo "some text" | ./tfai - "Uppercase the text."
```

- `texto.txt` é **o texto a transformar**. O pedido vai separado, como argumento.
- O pedido precisa ser em **inglês, no estilo do dataset**, com valores literais entre
  aspas duplas: `Remove all whitespace. Append "banana" to the text. Prepend "orange" to
  the text.` Pedidos em português ou com frases muito diferentes dos templates fazem o
  modelo gerar lixo, e o `ask` avisa isso.
- Modelo padrão: `runs/codet5-small/best` (troque com `--model` ou a variável `TFAI_MODEL`).
- Para usar de qualquer pasta: `ln -s "$PWD/tfai" ~/.local/bin/tfai`.

Comandos longos equivalentes (`--model` agora é opcional em todos):

```bash
python -m textforge_ai solve --input texto.txt --prompt-file pedido.txt \
    --save-pipeline saida.tft --save-output saida.txt
python -m textforge_ai verify --input texto.txt --pipeline meu.tft --prompt "Uppercase the text."
```

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

Medido no PC do usuário (Ryzen 5 4600G, 12 threads, CPU) em 2026-11-23, treino de teste:
**250 passos em 15,3 min = 3,7 s por passo** (lote 16).

A estimativa anterior (~1 s/passo, extrapolada de 2 núcleos de outra máquina) estava
errada por um fator de ~3,7. Lição: medir antes de prometer tempo.

Treino completo: ~72k pares (1 de geração + até 3 de verificação por exemplo) →
~4,5k passos por época → **~4,6 h por época**:

| épocas | tempo aproximado |
|---|---|
| 1 | ~4,6 h |
| 2 | ~9 h |
| 3 | ~14 h |

## Primeiro resultado real (treino de teste, 2026-11-23)

1000 exemplos, 1 época, avaliado em 100 exemplos de validação:

| métrica | valor |
|---|---|
| `gen_probe_accuracy` | 0,56 |
| `gen_exact_match` | 0,54 |
| `gen_valid_rate` | 0,92 |
| `verify_accuracy` | 0,56 |

A geração já aprende bem com poucos dados. O verificador ainda está perto do acaso
(~0,5): é a tarefa mais difícil e teve só ~3k pares. Se continuar perto de 0,5 depois do
treino completo, o formato da tarefa de verificação precisa ser revisto.

## Resultado do treino completo (2026-11-24)

Dataset completo, 2 épocas, 9036 passos, ~466 min. Avaliado em 300 exemplos de validação
(900 pares de verificação) ao fim de cada época:

| métrica | teste (1k ex.) | época 1 | época 2 |
|---|---|---|---|
| `gen_probe_accuracy` | 0,56 | 1,00 | **1,00** |
| `gen_exact_match` | 0,54 | 0,983 | 0,987 |
| `gen_valid_rate` | 0,92 | 1,00 | 1,00 |
| `verify_accuracy` | 0,56 | 0,887 | **0,929** |

Leitura:

- **Gerador**: acerta todos os probes em todos os exemplos avaliados. A diferença entre
  `exact_match` e `probe_accuracy` são pipelines escritos de outro jeito mas equivalentes
  na execução, o que é desejável. Já estava em 100% na época 1.
- **Verificador**: saiu do acaso (0,56) para 0,93, e a época 2 ainda ajudou (0,89 → 0,93).
  O formato da tarefa funciona. Ainda erra ~1 em 14: o veredito é estimativa.
- **Ressalva principal**: validação e treino vêm do mesmo gerador sintético (mesmos
  templates de frase). 100% aqui mede "aprendeu os templates", não "entende inglês livre".

Primeiro uso real (`solve`) falhou por dois motivos, nenhum deles bug do modelo:
1. `cat "frase" >> input.txt` não escreve a frase (o `cat` trata como nome de arquivo), mas
   o `>>` cria o arquivo vazio. O texto de entrada estava vazio.
2. O pedido foi escrito em português. Nenhum dos 4 candidatos era um pipeline válido, e o
   `solve` só dizia "no candidate passed validation". Agora `ask`/`solve` explicam o motivo.

Próximos passos de avaliação:

1. `model-eval` no `data/test.jsonl` completo (~1000 exemplos).
2. Teste manual com pedidos em inglês fora das frases dos templates (paráfrases, ordem
   diferente, sinônimos). É aí que se mede generalização (fase 6).
3. Se a fase 6 incluir pedidos em português, o dataset precisa de templates em português.

## Lacunas conhecidas

- O verificador vê só os primeiros 400 caracteres da entrada e da saída.
- Não há laço de autocorreção ainda (fase 4): `solve` só escolhe entre os candidatos.
- Sem decodificação restrita (fase 5): o modelo pode gerar mnemônicos inexistentes;
  esses candidatos são descartados pelo parser.
- Generalização para pedidos fora dos templates ainda não medida.
