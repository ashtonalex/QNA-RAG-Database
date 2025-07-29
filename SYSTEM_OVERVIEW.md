# QNA-RAG-Database System Overview

## 🏗️ Clean Production-Ready Codebase

This repository contains a complete RAG (Retrieval-Augmented Generation) system with document processing, vector search, dual reranking, and AI response generation.

## 📁 Core System Structure

### Backend Services (`backend/app/services/`)
- **`enhanced_rag_pipeline.py`** - Main RAG pipeline with all components integrated
- **`rag_service.py`** - Core RAG service with query enhancement, retrieval, context building, and response generation
- **`cohere_reranker.py`** - Cohere API integration for semantic reranking
- **`crossencoder_reranker.py`** - Local CrossEncoder model for reranking
- **`vector_service.py`** - ChromaDB vector database operations
- **`document_processor.py`** - Document parsing and text extraction
- **`chunking_service.py`** - Text chunking with semantic and syntactic strategies
- **`metadata_extractor.py`** - Document metadata extraction
- **`ocr_service.py`** - OCR processing for image-based documents
- **`vector_pipeline_service.py`** - Document processing pipeline

### API Layer (`backend/app/api/`)
- **`rag_pipeline.py`** - FastAPI endpoints for RAG queries (REST + WebSocket)
- **`documents.py`** - Document upload and management endpoints

### Models (`backend/app/models/`)
- **`chunk_models.py`** - Data models for document chunks

### Frontend (`frontend/`)
- React-based web interface for document upload and chat

### Testing (`tests/`)
- **`test_system.py`** - Comprehensive system test covering end-to-end functionality

## 🔧 Key Features

### RAG Pipeline Components
1. **Query Enhancement** - DeepSeek V3 via OpenRouter API
2. **Vector Retrieval** - ChromaDB with Jina embeddings
3. **Dual Reranking** - Cohere + CrossEncoder score fusion
4. **Context Building** - Token-aware windowing with source attribution
5. **Response Generation** - DeepSeek V3 with streaming support

### Advanced Capabilities
- **Configurable Pipeline** - Flexible parameters for different use cases
- **Error Handling** - Graceful fallbacks and retry logic
- **Rate Limiting** - Concurrent request management
- **Memory Management** - Token limits and context truncation
- **Comprehensive Logging** - Full pipeline observability
- **Health Monitoring** - Component status checking

## 🚀 API Endpoints

### RAG Pipeline
- `POST /rag/query` - Process RAG query (batch mode)
- `WebSocket /rag/query/stream` - Process RAG query (streaming mode)
- `GET /rag/health` - Check pipeline health

### Document Management
- `POST /upload` - Upload documents for processing
- `GET /projects` - List available projects
- `DELETE /project/:id` - Delete project and documents

## 🔑 Environment Variables

Required in `.env` file:
```
OPENROUTER_API_KEY=your_openrouter_key
cohere_API_key=your_cohere_key
JINA_API_KEY=your_jina_key
REDIS_URL=your_redis_connection_string
```

## 🧪 Testing

Run the comprehensive system test:
```bash
python -m pytest tests/test_system.py -v
```

## 📊 System Performance

- **Query Processing**: ~2-5 seconds end-to-end
- **Document Upload**: Supports PDF, DOCX, TXT
- **Concurrent Users**: Configurable (default: 10)
- **Context Window**: Up to 4000 tokens (configurable)
- **Reranking**: Dual-model approach for optimal relevance

## 🛡️ Production Features

- **Fault Tolerance** - System continues with component failures
- **Input Validation** - Comprehensive parameter checking
- **Resource Management** - Memory and token limit enforcement
- **Security** - API key protection and input sanitization
- **Monitoring** - Health checks and detailed logging
- **Scalability** - Async processing and rate limiting

## 🎯 Ready for Deployment

This codebase is production-ready with:
- ✅ Complete functionality tested
- ✅ Error handling and fallbacks
- ✅ Performance optimization
- ✅ Security best practices
- ✅ Comprehensive logging
- ✅ Health monitoring
- ✅ Clean, maintainable code

The system can be deployed immediately to production environments.