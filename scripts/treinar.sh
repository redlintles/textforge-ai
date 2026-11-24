#!/usr/bin/env bash
# Prepara o ambiente e treina o modelo.
#   bash scripts/treinar.sh teste             -> treino curto (~15 min) para validar tudo
#   bash scripts/treinar.sh completo [args]   -> treino completo em segundo plano
#                                                ex.: completo --epochs 2
#   bash scripts/treinar.sh variado [args]    -> fase 6: continua o modelo atual com o
#                                                dataset variado (data_v2), em segundo plano;
#                                                no fim roda o ood-eval (tudo em train-v2.log)
#   bash scripts/treinar.sh continuar         -> retoma o último treino (variado, se existir)
#   bash scripts/treinar.sh parar             -> para com segurança (salva checkpoint)
set -euo pipefail
cd "$(dirname "$0")/.."
MODO="${1:-teste}"
shift || true
OUT=runs/codet5-small
OUT2=runs/codet5-v2

if [ "$MODO" = "parar" ]; then
    # marca a parada para o ood-eval do modo variado não rodar sobre um treino incompleto
    [ -d "$OUT2" ] && touch "$OUT2/STOPPED"
    pkill -INT -f "^python[0-9.]* -m textforge_ai train" && echo "sinal enviado; o treino salva e encerra no fim do passo atual" \
        || echo "nenhum treino rodando"
    exit 0
fi

# 1. ambiente (recria a .venv se ela não tiver um Python funcionando)
if ! .venv/bin/python -c "import sys" 2>/dev/null; then
    python3 -m venv --clear .venv
fi
source .venv/bin/activate
if ! python -c "import torch" 2>/dev/null; then
    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
# sempre: aplica as versões fixadas no pyproject.toml (ex.: transformers<5)
pip install -q -e ".[dev,train]"

# 2. testes (pulados ao continuar, para retomar rápido)
[ "$MODO" = "continuar" ] || python -m pytest -q

# 3. dataset (determinístico; só gera se ainda não existir)
[ -f data/train.jsonl ] || python -m textforge_ai gen-data --n 20000 --out data

# 4. treino
if [ "$MODO" = "continuar" ] && [ -d "$OUT2/checkpoint" ]; then MODO=continuar-variado; fi
case "$MODO" in
    teste)
        python -m textforge_ai train --data data --out runs/teste \
            --max-train 1000 --epochs 1 --eval-examples 100 2>&1 | tee train-teste.log
        ;;
    completo)
        if [ -d "$OUT/checkpoint" ]; then
            echo "já existe um treino em $OUT. Use 'continuar', ou apague $OUT para recomeçar."
            exit 1
        fi
        nohup python -m textforge_ai train --data data --out "$OUT" --epochs 3 "$@" \
            >> train.log 2>&1 &
        echo "treino rodando (pid $!). Acompanhe: tail -f train.log"
        echo "parar com segurança: bash scripts/treinar.sh parar"
        ;;
    variado)
        [ -f data_v2/train.jsonl ] || python -m textforge_ai gen-data --n 30000 --out data_v2
        if [ -d "$OUT2/checkpoint" ]; then
            echo "já existe um treino em $OUT2. Use 'continuar', ou apague $OUT2 para recomeçar."
            exit 1
        fi
        rm -f "$OUT2/STOPPED"
        nohup sh -c "python -m textforge_ai train --data data_v2 --model $OUT/best --out $OUT2 \
            --epochs 1 --lr 2e-4 $* && [ ! -f $OUT2/STOPPED ] \
            && python -m textforge_ai ood-eval --model $OUT2/best" \
            >> train-v2.log 2>&1 &
        echo "treino variado rodando (pid $!). Acompanhe: tail -f train-v2.log"
        echo "parar com segurança: bash scripts/treinar.sh parar"
        ;;
    continuar-variado)
        rm -f "$OUT2/STOPPED"
        nohup sh -c "python -m textforge_ai train --resume --out $OUT2 $* \
            && [ ! -f $OUT2/STOPPED ] && python -m textforge_ai ood-eval --model $OUT2/best" \
            >> train-v2.log 2>&1 &
        echo "treino variado retomado (pid $!). Acompanhe: tail -f train-v2.log"
        ;;
    continuar)
        nohup python -m textforge_ai train --resume --out "$OUT" "$@" >> train.log 2>&1 &
        echo "treino retomado (pid $!). Acompanhe: tail -f train.log"
        ;;
    *) echo "uso: $0 [teste|completo|variado|continuar|parar]"; exit 1 ;;
esac
