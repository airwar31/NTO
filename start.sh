#!/usr/bin/env bash
# ==============================================================================
# AstroClassify Studio - Unix Launch Script
# ==============================================================================

echo "====================================================================="
echo "  🌌 AstroClassify Studio - Space Object Classifier (NTO AI Olympiad)"
echo "====================================================================="

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

python3 web/app.py
