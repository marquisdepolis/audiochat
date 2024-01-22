#!/bin/bash

# Run ollama with the mistral command
ollama run mistral

# Set the OLLAMA_HOST environment variable and run ollama serve
OLLAMA_HOST=0.0.0.0:11434 ollama serve
