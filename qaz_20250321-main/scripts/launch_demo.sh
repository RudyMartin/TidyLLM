#!/bin/bash

echo "🎬 VectorQA Sage Demo Launcher"
echo "================================"
echo "Launching 4 Streamlit apps for demo..."
echo ""

# Change to src directory
cd src

# Launch all apps in background
echo "🚀 Launching apps on different ports..."

# Main App
streamlit run main.py --server.port 8501 --server.headless true &
echo "✅ Main App: http://localhost:8501"

# Enhanced QA Demo (Simplified)
streamlit run enhanced_qa_demo_simple.py --server.port 8502 --server.headless true &
echo "✅ Enhanced QA: http://localhost:8502"

# RAG Query Demo
streamlit run rag_query_demo.py --server.port 8503 --server.headless true &
echo "✅ RAG Query: http://localhost:8503"

# MCP Dashboard
streamlit run mcp_dashboard.py --server.port 8504 --server.headless true &
echo "✅ MCP Dashboard: http://localhost:8504"

echo ""
echo "🎉 All demo apps launched!"
echo ""
echo "📱 Demo Flow:"
echo "1. MCP Dashboard: http://localhost:8504 (System overview)"
echo "2. Enhanced QA:   http://localhost:8502 (Document processing)"
echo "3. RAG Query:     http://localhost:8503 (Intelligent search)"
echo "4. Main App:      http://localhost:8501 (Complete pipeline)"
echo ""
echo "⏹️  To stop: Press Ctrl+C or run 'pkill -f streamlit'"
echo ""

# Keep script running
wait
