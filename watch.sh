#!/bin/bash

WATCH_DIR="$(pwd)"  # Set your project directory

echo "🔍 Watching for changes in $WATCH_DIR..."

inotifywait -m -r -e modify,create,delete --format '%w%f' "$WATCH_DIR" | while read FILE
do
    echo "🛠️ Change detected in $FILE. Running make all..."
    make all
done
