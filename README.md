---
title: QNA RAG Database
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# QNA-RAG-Database System

A production-ready Question-Answering system using Retrieval-Augmented Generation (RAG) with document processing, vector storage, and intelligent query enhancement.

## Features

- **Document Processing**: PDF, DOCX, TXT support with OCR capabilities
- **Semantic Chunking**: Intelligent text segmentation preserving context
- **Vector Embeddings**: Jina embeddings v2 (512 dimensions) with caching
- **Vector Storage**: ChromaDB for efficient similarity search
- **RAG Pipeline**: Query enhancement, retrieval, reranking, and response generation

## Usage

1. Upload documents (PDF, DOCX, TXT)
2. Wait for processing to complete
3. Ask questions about your documents
4. Get intelligent responses with source citations

## Demo

This is a live demo of the QNA-RAG-Database system running on Hugging Face Spaces.