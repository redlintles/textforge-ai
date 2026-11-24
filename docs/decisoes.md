# Registro de decisões

Formato: contexto → decisão → consequência. Decisões novas vão no fim.

## D1 — O alvo de treino é sempre um pipeline (2026-11-18)
**Contexto**: risco de o modelo aprender a reescrever o texto em vez de programar.
**Decisão**: nenhum exemplo tem texto transformado como alvo; o alvo é o pipeline canônico.
**Consequência**: imitar os dados nunca leva ao atalho.

## D2 — Modelo v1 recebe só o pedido, não o texto (2026-11-18)
**Contexto**: se o modelo vê o texto, pode copiá-lo.
**Decisão**: na v1 a entrada é só o prompt; o texto é usado para executar e conferir.
**Consequência**: copiar fica estruturalmente impossível. Pedidos que dependem do
conteúdo do texto ficam para a fase 5.

## D3 — Correção medida em vários probes + guardas (2026-11-18)
**Decisão**: correto = acerta todos os 6 probes e não é suspeito (`guards.py`).
**Consequência**: o baseline `cheater` tira 0% e é 100% detectado.

## D4 — Sintaxe sempre via engine (2026-11-18)
**Decisão**: pipelines são construídos com o builder do `textforgers` e serializados com
`get_text_transform_vec`; predições são lidas com o parser real.
**Consequência**: quoting/escape nunca diverge do TextForge. Atualizar o binding atualiza tudo.

## D5 — Divisão por pipeline (2026-11-18)
**Decisão**: um pipeline pertence a um só conjunto (treino/val/teste).
**Consequência**: a nota de teste mede generalização, não memória.

## D6 — Código em inglês, documentação em português (2026-11-18)
**Decisão**: segue a convenção do TextForge para código/commits (inglês); docs do projeto
em português para quem está aprendendo.

## D7 — Validação: determinística quando há gabarito, aprendida quando não há (2026-11-18)
**Contexto**: o sistema deve validar se um pipeline atinge o pedido sobre uma entrada.
**Decisão**: com saída esperada, validar por execução e comparação (exato). Sem ela, um
verificador aprendido, treinado no mesmo modelo do gerador (multitarefa, prefixos
`generate:`/`verify:`). Negativos vêm de mutações do pipeline correto, rotulados por
execução (mutação equivalente = `match`).
**Consequência**: um só modelo para treinar em CPU. O veredito sem gabarito é uma
estimativa e deve ser apresentado com confiança, nunca como prova.

## D8 — Hardware: treino em CPU, GPU opcional (2026-11-18)
**Contexto**: Ryzen 5 4600G, 32 GB RAM, RX 580 (Polaris/gfx803, sem suporte oficial no ROCm).
**Decisão**: treinar `codet5-small` em CPU; código detecta GPU automaticamente; Kaggle
(GPU gratuita) como reserva; ROCm comunitário na RX 580 fora do caminho principal.

## D9 — Fixar `transformers<5` (2026-11-23)
**Contexto**: no primeiro treino real, `AutoTokenizer.from_pretrained("Salesforce/codet5-small")`
quebrou com `TypeError: Input must be a List[Union[str, AddedToken]]`. O transformers 5.x mudou
o carregamento de tokens especiais e não lê a configuração do tokenizador do CodeT5
(publicado na época da 4.x).
**Decisão**: `transformers>=4.40,<5` no extra `train`. Também passamos a pedir
`return_token_type_ids=False` ao tokenizador, porque na 4.x ele devolve esse campo e o T5 o
rejeita (achado pelos testes ao trocar de versão).
**Consequência**: suíte completa passa na 4.57.6. Revisitar quando o CodeT5 for compatível
com a 5.x ou se trocarmos de modelo.
