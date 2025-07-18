"use client";

import type React from "react";
import type { SetStateAction } from "react";

import { useState, useCallback, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Upload,
  File,
  Trash2,
  CheckCircle,
  AlertCircle,
  Loader2,
} from "lucide-react";
import { cn } from "@/lib/utils";

export interface Document {
  id: string;
  name: string;
  size: number;
  type: string;
  status: "uploading" | "processing" | "completed" | "error";
  progress: number;
  error?: string;
}

export interface DocumentUploadProps {
  documents: Document[];
  onDocumentsChange: (
    documents: Document[] | ((prev: Document[]) => Document[])
  ) => void;
}

export function DocumentUpload({
  documents,
  onDocumentsChange,
}: DocumentUploadProps) {
  const [isDragActive, setIsDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Helper to upload a file to the backend
  const uploadFile = async (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    setUploading(true);
    try {
      const res = await fetch("/api/documents/upload", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error("Upload failed");
      const data = await res.json();
      return data.document_id as string;
    } finally {
      setUploading(false);
    }
  };

  // Poll backend for document status
  const pollStatus = (docId: string) => {
    let pollInterval: NodeJS.Timeout | null = null;
    const poll = async (): Promise<void> => {
      try {
        const res = await fetch(`/api/documents/${docId}/status`);
        if (!res.ok) throw new Error("Status check failed");
        const status = await res.json();
        onDocumentsChange((prev) =>
          prev.map((doc: Document) =>
            doc.id === docId
              ? {
                  ...doc,
                  status:
                    status.status === "done"
                      ? "completed"
                      : status.status === "error"
                      ? "error"
                      : status.status === "pending"
                      ? "uploading"
                      : "processing",
                  progress: status.progress ?? doc.progress,
                  error: status.error_message || undefined,
                }
              : doc
          )
        );
        if (status.status === "done" || status.status === "error") {
          if (pollInterval) clearInterval(pollInterval);
        }
      } catch (e) {
        onDocumentsChange((prev) =>
          prev.map((doc: Document) =>
            doc.id === docId
              ? { ...doc, status: "error", error: "Failed to poll status" }
              : doc
          )
        );
        if (pollInterval) clearInterval(pollInterval);
      }
    };
    poll();
    pollInterval = setInterval(poll, 1500);
  };

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files) return;

      const acceptedTypes = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
      ];
      const maxSize = 10 * 1024 * 1024; // 10MB

      const validFiles = Array.from(files).filter((file) => {
        return acceptedTypes.includes(file.type) && file.size <= maxSize;
      });

      validFiles.forEach(async (file) => {
        // Add to UI as uploading
        const tempId = Math.random().toString(36).substr(2, 9);
        onDocumentsChange((docs: Document[]) => [
          ...docs,
          {
            id: tempId,
            name: file.name,
            size: file.size,
            type: file.type,
            status: "uploading" as const,
            progress: 0,
          },
        ]);
        try {
          const docId = await uploadFile(file);
          // Replace temp doc with real docId
          onDocumentsChange((docs) =>
            docs.map((doc: Document) =>
              doc.id === tempId
                ? { ...doc, id: docId, status: "processing", progress: 0 }
                : doc
            )
          );
          pollStatus(docId);
        } catch (e: any) {
          onDocumentsChange((docs) =>
            docs.map((doc: Document) =>
              doc.id === tempId
                ? {
                    ...doc,
                    status: "error",
                    error: e?.message || "Upload failed",
                  }
                : doc
            )
          );
        }
      });
    },
    [onDocumentsChange]
  );

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragActive(false);
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles]
  );

  const handleFileInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      handleFiles(e.target.files);
    },
    [handleFiles]
  );

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const removeDocument = (docId: string) => {
    // Only allow removal if not processing
    const doc = documents.find((d: Document) => d.id === docId);
    if (doc && (doc.status === "processing" || doc.status === "uploading"))
      return;
    onDocumentsChange(documents.filter((doc: Document) => doc.id !== docId));
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return (
      Number.parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i]
    );
  };

  const getStatusIcon = (status: Document["status"]) => {
    switch (status) {
      case "uploading":
      case "processing":
        return <Loader2 className="h-4 w-4 animate-spin text-blue-500" />;
      case "completed":
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case "error":
        return <AlertCircle className="h-4 w-4 text-red-500" />;
    }
  };

  return (
    <div className="space-y-4">
      {/* Upload Area */}
      <Card
        className={cn(
          "border-2 border-dashed transition-colors cursor-pointer",
          isDragActive
            ? "border-blue-500 bg-blue-50 dark:bg-blue-950/20"
            : "border-muted-foreground/25"
        )}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={uploading ? undefined : handleClick}
        aria-disabled={uploading}
        tabIndex={uploading ? -1 : 0}
        style={uploading ? { pointerEvents: "none", opacity: 0.6 } : {}}
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
            <p className="text-sm text-muted-foreground mb-2">
              Drag & drop files here, or click to select
            </p>
            <p className="text-xs text-muted-foreground">
              Supports PDF, DOCX, TXT (max 10MB)
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Document List */}
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {documents.map((doc: Document) => (
          <Card key={doc.id} className="p-3">
            <div className="flex items-center space-x-3">
              <File className="h-8 w-8 text-muted-foreground flex-shrink-0" />

              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between mb-1">
                  <p className="text-sm font-medium truncate">{doc.name}</p>
                  <div className="flex items-center space-x-2">
                    {getStatusIcon(doc.status)}
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6"
                      onClick={() => removeDocument(doc.id)}
                      disabled={
                        doc.status === "processing" ||
                        doc.status === "uploading"
                      }
                    >
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

                {(doc.status === "uploading" ||
                  doc.status === "processing") && (
                  <Progress value={doc.progress} className="h-1" />
                )}
                {doc.status === "error" && doc.error && (
                  <div className="text-xs text-red-500 mt-1">{doc.error}</div>
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
  );
}
