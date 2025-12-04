#!/bin/bash
# EIS Analyzer - Startup Script

echo "Starting EIS Analyzer..."
echo "========================"
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "Streamlit is not installed."
    echo "Please install requirements:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Run the app
streamlit run app.py
