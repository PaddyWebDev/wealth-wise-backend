#!/bin/bash

# Exit on error
set -e

# Optional: activate virtual environment (uncomment if needed)
source .venv/Scripts/activate

py models/lumpSum_based_recommendation.py
py models/sip_based_recommendation.py


# Set Flask environment variables
export FLASK_APP=app.py
export FLASK_ENV=development
export FLASK_DEBUG=1

echo "🔥 Starting Flask server with live reload..."
echo "Watching for code changes..."

# Run Flask with auto reload
flask run --reload --host=0.0.0.0 --port=5000

