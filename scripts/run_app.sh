#!/bin/bash

BACKEND_DIR="$(pwd)"
FRONTEND_DIR="$(pwd)/app/frontend"
LOG_DIR="$(pwd)/logs"
PID_DIR="$(pwd)/pids"

echo BACKEND_DIR: $BACKEND_DIR
echo FRONTEND_DIR: $FRONTEND_DIR

BACKEND_PORT=8000
FRONTEND_PORT=5173

mkdir -p $LOG_DIR
mkdir -p $PID_DIR

BACKEND_PID_FILE="$PID_DIR/backend.pid"
FRONTEND_PID_FILE="$PID_DIR/frontend.pid"

start() {
    echo "=== Lancement du backend FastAPI ==="
    cd $BACKEND_DIR
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate base
    export PYTHONPATH=$(pwd)

python - <<'END'
try:
    import torch
    print(f"torch est installé, version: {torch.__version__}")
except ModuleNotFoundError:
    print("torch n'est pas installé dans l'environnement actuel.")
    import sys
    sys.exit(1)
END

    nohup $(which python) -m uvicorn app.backend.main:app --host 0.0.0.0 --port $BACKEND_PORT \
        > $LOG_DIR/backend.log 2>&1 &
    echo $! > $BACKEND_PID_FILE
    echo "Backend lancé (PID=$(cat $BACKEND_PID_FILE)) sur http://localhost:$BACKEND_PORT"

    echo "=== Nettoyage du frontend existant ==="

    fuser -k -s ${FRONTEND_PORT}/tcp 2>/dev/null || true

    echo "=== Lancement du frontend React ==="
    cd $FRONTEND_DIR
    nohup npm run dev > $LOG_DIR/frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > $FRONTEND_PID_FILE
    echo "Frontend lancé (PID=$(cat $FRONTEND_PID_FILE)) sur http://localhost:$FRONTEND_PORT"

    echo "Logs backend : $LOG_DIR/backend.log"
    echo "Logs frontend : $LOG_DIR/frontend.log"
}

stop() {
    echo "=== Arrêt des applications ==="
    if [ -f $BACKEND_PID_FILE ]; then
        kill $(cat $BACKEND_PID_FILE) && rm $BACKEND_PID_FILE
        echo "Backend arrêté"
    else
        echo "Aucun PID backend trouvé"
    fi

    if [ -f $FRONTEND_PID_FILE ]; then
        FRONT_PID=$(cat $FRONTEND_PID_FILE)
        pkill -P $FRONT_PID
        kill $FRONT_PID && rm $FRONTEND_PID_FILE
        fuser -k -s ${FRONTEND_PORT}/tcp 2>/dev/null || true
        echo "Frontend arrêté"
    else
        echo "Aucun PID frontend trouvé"
    fi
}

status() {
    echo "=== Statut des applications ==="
    if [ -f $BACKEND_PID_FILE ]; then
        ps -p $(cat $BACKEND_PID_FILE)
    else
        echo "Backend non lancé"
    fi

    if [ -f $FRONTEND_PID_FILE ]; then
        ps -p $(cat $FRONTEND_PID_FILE)
    else
        echo "Frontend non lancé"
    fi
}

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    status)
        status
        ;;
    restart)
        stop
        start
        ;;
    *)
        echo "Usage: $0 {start|stop|status|restart}"
        exit 1
        ;;
esac
