"use client"

import type React from "react"

import { useState, useCallback, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Upload, File, Trash2, CheckCircle, AlertCircle, Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"

interface Document {
  id: string
  name: string
  size: number
  type: string
  status: "uploading" | "processing" | "completed" | "error"
  progress: number
  error?: string
}

interface DocumentUploadProps {
  documents: Document[]
  onDocumentsChange: (documents: Document[]) => void
}

export function DocumentUpload({ documents, onDocumentsChange }: DocumentUploadProps) {
  const [isDragActive, setIsDragActive] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files) return

      const acceptedTypes = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
      ]
      const maxSize = 10 * 1024 * 1024 // 10MB

      const validFiles = Array.from(files).filter((file) => {
        return acceptedTypes.includes(file.type) && file.size <= maxSize
      })

      const newDocuments = validFiles.map((file) => ({
        id: Math.random().toString(36).substr(2, 9),
        name: file.name,
        size: file.size,
        type: file.type,
        status: "uploading" as const,
        progress: 0,
      }))

      onDocumentsChange([...documents, ...newDocuments])

      // Simulate upload and processing
      newDocuments.forEach((doc) => {
        simulateUpload(doc.id)
      })
    },
    [documents, onDocumentsChange],
  )

  const simulateUpload = (docId: string) => {
    const updateProgress = (progress: number, status?: Document["status"]) => {
      onDocumentsChange((prev) =>
        prev.map((doc) => (doc.id === docId ? { ...doc, progress, ...(status && { status }) } : doc)),
      )
    }

    // Simulate upload progress
    let progress = 0
    const uploadInterval = setInterval(() => {
      progress += Math.random() * 20
      if (progress >= 100) {
        clearInterval(uploadInterval)
        updateProgress(100, "processing")

        // Simulate processing
        setTimeout(() => {
          updateProgress(100, "completed")
        }, 2000)
      } else {
        updateProgress(progress)
      }
    }, 200)
  }

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(false)
  }, [])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
      setIsDragActive(false)
      handleFiles(e.dataTransfer.files)
    },
    [handleFiles],
  )

  const handleFileInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      handleFiles(e.target.files)
    },
    [handleFiles],
  )

  const handleClick = () => {
    fileInputRef.current?.click()
  }

  const removeDocument = (docId: string) => {
    onDocumentsChange(documents.filter((doc) => doc.id !== docId))
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes"
    const k = 1024
    const sizes = ["Bytes", "KB", "MB", "GB"]
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Number.parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i]
  }

  const getStatusIcon = (status: Document["status"]) => {
    switch (status) {
      case "uploading":
      case "processing":
        return <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
      case "completed":
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case "error":
        return <AlertCircle className="h-4 w-4 text-red-500" />
    }
  }

  return (
    <div className="space-y-4">
      {/* Upload Area */}
      <Card
        className={cn(
          "border-2 border-dashed transition-colors cursor-pointer",
          isDragActive ? "border-blue-500 bg-blue-50 dark:bg-blue-950/20" : "border-muted-foreground/25",
        )}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={handleClick}
      >
        <CardContent className="p-6">
          <div className="text-center">
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".pdf,.docx,.txt"
              onChange={handleFileInputChange}
              className="hidden"
            />
            <Upload className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
            <p className="text-sm text-muted-foreground mb-2">Drag & drop files here, or click to select</p>
            <p className="text-xs text-muted-foreground">Supports PDF, DOCX, TXT (max 10MB)</p>
          </div>
        </CardContent>
      </Card>

      {/* Document List */}
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {documents.map((doc) => (
          <Card key={doc.id} className="p-3">
            <div className="flex items-center space-x-3">
              <File className="h-8 w-8 text-muted-foreground flex-shrink-0" />

              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between mb-1">
                  <p className="text-sm font-medium truncate">{doc.name}</p>
                  <div className="flex items-center space-x-2">
                    {getStatusIcon(doc.status)}
                    <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => removeDocument(doc.id)}>
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  </div>
                </div>

                <div className="flex items-center justify-between text-xs text-muted-foreground mb-2">
                  <span>{formatFileSize(doc.size)}</span>
                  <Badge variant="secondary" className="text-xs">
                    {doc.status}
                  </Badge>
                </div>

                {(doc.status === "uploading" || doc.status === "processing") && (
                  <Progress value={doc.progress} className="h-1" />
                )}
              </div>
            </div>
          </Card>
        ))}
      </div>

      {documents.length === 0 && (
        <div className="text-center py-8 text-muted-foreground">
          <File className="mx-auto h-12 w-12 mb-2 opacity-50" />
          <p className="text-sm">No documents uploaded yet</p>
        </div>
      )}
    </div>
  )
}
