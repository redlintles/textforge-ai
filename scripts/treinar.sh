#!/usr/bin/env bash
# Prepara o ambiente e treina o modelo.
#   ./scripts/treinar.sh teste     -> treino curto (~20-30 min) para validar tudo
#   ./scripts/treinar.sh completo  -> treino completo em segundo plano (~3-5 h, estimativa)
set -euo pipefail
cd "$(dirname "$0")/.."
MODO="${1:-teste}"

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

# 2. testes
python -m pytest -q

# 3. dataset (determinístico; só gera se ainda não existir)
[ -f data/train.jsonl ] || python -m textforge_ai gen-data --n 20000 --out data

# 4. treino
case "$MODO" in
    teste)
        python -m textforge_ai train --data data --out runs/teste \
            --max-train 1000 --epochs 1 --eval-examples 100 2>&1 | tee train-teste.log
        ;;
    completo)
        nohup python -m textforge_ai train --data data --out runs/codet5-small --epochs 3 \
            > train.log 2>&1 &
        echo "treino rodando (pid $!). Acompanhe com: tail -f train.log"
        echo "para parar com segurança: kill -INT $!"
        ;;
    *) echo "uso: $0 [teste|completo]"; exit 1 ;;
esac
