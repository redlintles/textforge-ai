# 01 — Abordagem: como fazer a IA escrever pipelines (e não reescrever o texto)

## O problema

Você dá à IA um texto e um pedido ("troque todo *apple* por *pear* e deixe em
maiúsculas"). A saída desejada é um **programa** TextForge:

```
raw apple pear;
tua;
```

O risco que você identificou é real e tem nome em ML: *shortcut learning* (o modelo
acha um atalho que satisfaz a métrica sem aprender a tarefa). Aqui o atalho seria
apagar o texto e escrever a resposta pronta como literal:

```
slt 0 0;
dlf;
ate 'PEAR PIE AND PEAR JUICE';
```

Esse pipeline é *válido* e dá a saída certa *para aquele texto*. Se o treino ou a
avaliação recompensarem só "a saída bateu", o modelo pode aprender isso.

## Por que acontece

Um modelo aprende a imitar os **alvos** que mostramos e a maximizar a **métrica** que
usamos. O atalho só aparece se uma dessas duas coisas permitir. Então a defesa é
desenhar alvo, entrada e métrica para que o atalho nunca compense.

## As três camadas de defesa

**1. O alvo é sempre um pipeline.** Todo exemplo de treino tem como resposta um
pipeline curto, gerado pelo builder oficial e serializado pelo próprio engine.
Nenhum exemplo tem como alvo "o texto reescrito". O modelo não tem de onde imitar o
atalho. (Teste: `test_targets_are_pipelines_not_text`.)

**2. Na v1, o modelo nem vê o texto.** A entrada do modelo é só o pedido em inglês.
Se ele não vê o texto, não consegue copiá-lo. O texto só é usado *depois*, para
executar e conferir o pipeline. A maioria das instruções (trocar palavras, mudar
caixa, apagar linhas) é totalmente determinada pelo pedido. Mais tarde, quando o
modelo precisar ver o texto (para pedidos como "capitalize o nome da cidade"), as
camadas 3 e os guardas continuam valendo.

**3. A avaliação usa vários textos diferentes ("probes").** Cada exemplo vem com 6
textos de entrada e as saídas esperadas. Uma previsão só conta como correta se
acertar **todos**. Um pipeline que escreve a resposta de um texto erra os outros
cinco. Além disso, `guards.py` marca como suspeito:

- pipeline **independente da entrada** (textos diferentes → mesma saída);
- **literal vazado** (string longa no pipeline copiada do texto ou da saída esperada,
  e que não estava no pedido).

Resultado medido no conjunto de teste (982 exemplos):

| Preditor | Acerto nos probes | Marcado como suspeito |
|---|---|---|
| `oracle` (gabarito) | 100% | 0% |
| `identity` (não faz nada) | 0% | 0% |
| `cheater` (apaga e escreve a resposta) | **0%** | **100%** |

Ou seja: a métrica já está "à prova de trapaça" antes de existir modelo.

## Validação: o pipeline faz o que o usuário pediu?

O sistema também precisa **validar** um pipeline (gerado por ele ou escrito pelo
usuário) sobre uma entrada. Existem dois casos, com garantias bem diferentes:

**A. O usuário dá um exemplo de saída esperada.** A validação é determinística:
executar e comparar, como `evaluate.py` já faz com os probes. Não precisa de modelo e
a resposta é exata.

**B. O usuário dá só o pedido e a entrada.** Não há gabarito, então alguém precisa
*julgar* se a saída atende ao pedido. Isso é um **verificador aprendido**: recebe
(pedido, pipeline, trecho da entrada, diferença antes/depois) e responde
`match`/`mismatch` + qual etapa parece errada. Ele erra às vezes, e isso precisa ser
dito ao usuário: é uma estimativa, não uma prova.

### Um modelo, duas tarefas

O mesmo `codet5-small` aprende as duas tarefas, diferenciadas por um prefixo
(*multitask*), o que economiza memória e tempo de treino em CPU:

```
generate: <pedido>                                   -> <pipeline>
verify: <pedido> | <pipeline> | <entrada> | <diff>   -> match | mismatch: <etapa>
```

### Como treinar o verificador sem dados humanos

Os exemplos negativos vêm de **mutações** do pipeline correto: trocar um argumento,
inverter a ordem, remover ou adicionar uma etapa, trocar por uma instrução "vizinha"
(`rfw`↔`rlw`, `tua`↔`tla`, `klm`↔`dlm`, `tls`↔`trs`). O rótulo vem da **execução**,
não da diferença no texto do pipeline: se a mutação produz a mesma saída em todos os
probes, ela é equivalente e conta como `match`. Assim o verificador aprende a julgar
o resultado, não a decorar a forma.

### O laço completo

```
pedido + entrada
   │
   ▼ generate (N candidatos, beam search)
   ▼ engine: executa cada candidato na entrada (erro -> descarta ou corrige)
   ▼ verify: pontua (pedido, pipeline, entrada, diff)
   ▼ escolhe o melhor; se nenhum passar, gera de novo usando o erro/veredito
pipeline + saída + veredito (com confiança)
```

O verificador vê o texto, mas isso não reabre a "trapaça": ele só *julga*, não
escreve pipelines. Quem escreve continua sendo o gerador, sob as mesmas defesas.

## Roteiro

| Fase | O quê | Status |
|---|---|---|
| 0 | Wrapper do engine, erros estruturados | feito |
| 1 | Gerador de dataset sintético + avaliação + guardas | feito |
| 2 | Dados do verificador: mutações rotuladas por execução | feito |
| 3 | Modelo multitarefa (`generate` + `verify`) com `codet5-small`, treino em CPU | código pronto e testado; falta o treino real |
| 4 | Laço gerar → executar → verificar → corrigir | escolha entre candidatos e validação exata feitas; correção pendente |
| 5 | Decodificação restrita: o modelo só pode gerar mnemônicos válidos | planejado |
| 6 | Pedidos mais livres (paráfrases) e acesso ao texto de entrada no gerador | planejado |

### Sobre a fase 2 (o modelo)

Opções, da mais simples à mais robusta:

- **Transformer pequeno treinado do zero** (PyTorch). Roda em CPU, ótimo para
  aprender como tudo funciona. Vai bem nos pedidos com o vocabulário dos templates,
  mal em frases que ele nunca viu.
- **Fine-tuning de um modelo pré-treinado pequeno** (ex.: `Salesforce/codet5-small`,
  ~60M parâmetros). Já "sabe inglês" e lida bem com código; generaliza muito melhor
  para pedidos escritos livremente. Precisa baixar os pesos do Hugging Face e, de
  preferência, uma GPU (em CPU funciona, mas o treino leva horas).

Atenção ao tokenizador: modelos T5 "clássicos" não têm `{`, `}` nem `<` no
vocabulário, o que quebraria variáveis (`{{n}}`) e regex. O CodeT5 usa BPE e não
tem esse problema.
