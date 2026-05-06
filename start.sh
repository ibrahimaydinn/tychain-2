#!/bin/bash
# Tychain — Start the local Flask development server
# Usage: bash start.sh   (from inside BIST_PROJECT folder)

PORT=8080
DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║   📈  Tychain — Local Development Server      ║"
echo "  ╚══════════════════════════════════════════════╝"
echo ""
echo "  URL:          http://localhost:$PORT"
echo ""
echo "  Press Ctrl+C to stop."
echo ""

# Kill any existing server on this port
lsof -ti :$PORT | xargs kill -9 2>/dev/null

# Start the Flask app
cd "$DIR" && python3 app.py
