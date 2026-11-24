# Diário de desenvolvimento

## 2026-11-18 — Fundação (fases 0 e 1)

- Explorado o binding `textforgers==0.2.0` (import `textforge`). Descobertas:
  - extensões aceitas: `.tft`/`.textforge` (texto) e `.tfb` (bytecode); `.atp` é recusado;
  - o parser exige **uma instrução por linha**;
  - `TextForgeError` traz `stage`, `kind`, `message`, `hint`, ótimos para autocorreção;
  - padrões de `raw`/`rfw`/`klm`... são **regex**;
  - `get_text_transform_vec` devolve a serialização canônica.
- Criados: `engine.py`, `catalog.py` (48 instruções), `texts.py`, `synth.py`, `dataset.py`,
  `guards.py`, `evaluate.py`, `baselines.py`, `cli.py`.
- Bug encontrado pelos testes: o guarda de independência dava falso positivo em pipelines
  corretos com `klm` (filtrar linhas esvazia textos sem a palavra). Corrigido usando os
  probes do exemplo. Teste de regressão adicionado.
- 123 testes passando; `ruff` limpo.
- Baselines no teste: oracle 100%, identity 0%, cheater 0% (100% detectado).
- Próximo: fase 2, primeiro modelo.

## 2026-11-18 — Escopo ampliado: validação

- Requisito novo: o modelo também valida pipelines sobre uma entrada.
- Decidido (D7): validação exata quando há saída esperada; verificador aprendido
  multitarefa quando não há. Roteiro reordenado: dados do verificador vêm antes do treino.
- Decidido (D8): treino em CPU (RX 580 não tem suporte oficial ao ROCm).

## 2026-11-18 — Fases 2 e 3 (código de treino)

- `verify_data.py`: mutações (argumento, instrução vizinha, adicionar, remover, inverter)
  rotuladas por execução. Em 300 exemplos: ~43% match / 57% mismatch. Inverter etapas é
  equivalente em ~65% dos casos, e o rótulo por execução capturou isso.
- `tasks.py`, `model.py`, `train.py` (laço PyTorch explícito), `infer.py` (`Forge.solve`),
  comandos `train`, `model-eval`, `solve`, `verify`.
- Ambiente aqui não acessa o Hugging Face: testes usam T5 minúsculo (`--tiny`).
  Overfitting de 8 exemplos: 0% → 100%. Laço de treino validado.
- Verificador minúsculo: 70/72 nas alterações vistas, ~60% nas novas (decora, não
  generaliza; esperado).
- Tempo medido: 2,8 s/passo (arquitetura codet5-small, 2 núcleos). Estimativa no Ryzen:
  ~1–1,5 h por época.
- 135 testes passando.

## 2026-11-23 — Primeiro treino real: erro no tokenizador

- `scripts/treinar.sh` criado (ambiente + testes + dataset + treino `teste`/`completo`).
- No PC do usuário: 135 testes OK, dataset OK, mas o carregamento do tokenizador do CodeT5
  falhou no transformers 5.x. Corrigido fixando `<5` (D9). Ao rodar a suíte na 4.57.6 os
  testes acharam um segundo problema (`token_type_ids`), também corrigido.
- Limitação: daqui não dá para baixar o tokenizador real; a correção foi validada pela
  suíte com o modelo minúsculo, não com o CodeT5.

## 2026-11-23 — Treino de teste e retomada

- Treino de teste no PC do usuário: geração 56% (válidos 92%), verificador 56% (~acaso).
- Tempo real: 3,7 s/passo, ~3,7× pior que a estimativa. Treino completo de 3 épocas ≈ 14 h.
- Retomada reescrita: checkpoint completo a cada 200 passos (pesos + otimizador + scheduler +
  RNG + posição), gravação atômica, SIGINT/SIGTERM salvam antes de sair, `--resume`.
  Embaralhamento agora é por época a partir de uma cópia (antes era cumulativo e impedia
  retomada exata). Teste compara pesos de "parar e retomar" vs "sem parar": idênticos.
- Script: modos `continuar` e `parar`; `completo` aceita argumentos extras.
- `.gitignore`: `runs/` e `*.log`.
- Incidente no ambiente de desenvolvimento: um processo órfão de teste deixou a suíte em
  9 min; encerrado, voltou a ~19 s. 139 testes passando.

## 2026-11-24 — Treino completo e comando simples

- Treino completo (2 épocas, ~7,8 h): geração 100% nos probes, verificador 93%. Detalhes em
  docs/04.
- Primeiro uso falhou: entrada vazia (`cat "frase" >> input.txt`) e pedido em português.
- Novo comando `ask` + atalho `./tfai texto.txt "pedido"`: pedido posicional, modelo padrão
  `runs/codet5-small/best`, modo interativo que carrega o modelo uma vez, aviso de entrada
  vazia e explicação quando o modelo não gera nenhum pipeline válido. `--model` virou
  opcional em `solve`/`verify`/`model-eval`. 3 testes novos.

## 2026-11-24 (tarde) — O modelo só sabia os templates

- Pedidos livres falhavam. Novo teste com 80 pedidos escritos à mão (`ood.py`, `ood-eval`):
  **55%** (validação: 100%). Compor passos 28%, literais novos 22%.
- Causas: 2–3 frases fixas por instrução, e só 10 literais fixos para `ate`/`atb` (o modelo
  decorou strings em vez de copiar: `laranja` → `laranaja`, `mango` → `morango`).
- Gerador novo (`phrasing.py`, `lexicon.py`): muito mais frases, aspas, índices, conectores,
  ruído e valores. Re-treino a partir do modelo atual: `bash scripts/treinar.sh variado`.
  Detalhes em docs/05. 186 testes passando (sem os de modelo, que precisam de torch).

