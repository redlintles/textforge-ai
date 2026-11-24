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
