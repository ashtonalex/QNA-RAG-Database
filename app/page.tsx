"use client";

import { useState } from "react";
import type { Document } from "@/components/ui/document-upload";
import type { Message, Citation } from "@/components/ui/chat-interface";
import { MainLayout } from "@/components/layout/main-layout";
import { DocumentUpload } from "@/components/ui/document-upload";
import { ChatInterface } from "@/components/ui/chat-interface";
import { SourceCitations } from "@/components/ui/source-citations";
import { ThemeProvider } from "@/components/theme-provider";

export default function RAGSystem() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [citations, setCitations] = useState<Citation[]>([]);

  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
      <MainLayout>
        <div className="flex h-screen bg-background">
          {/* Left Sidebar - Document Management */}
          <div className="w-80 border-r border-border bg-card/50 backdrop-blur-sm">
            <div className="p-4 border-b border-border">
              <h2 className="text-lg font-semibold text-foreground">
                Documents
              </h2>
            </div>
            <div className="p-4 space-y-4">
              <DocumentUpload
                documents={documents}
                onDocumentsChange={setDocuments}
              />
            </div>
          </div>

          {/* Main Chat Interface */}
          <div className="flex-1 flex flex-col">
            <ChatInterface
              messages={messages}
              onMessagesChange={setMessages}
              onCitationsChange={setCitations}
            />
          </div>

          {/* Right Panel - Citations */}
          <div className="w-80 border-l border-border bg-card/50 backdrop-blur-sm">
            <div className="p-4 border-b border-border">
              <h2 className="text-lg font-semibold text-foreground">Sources</h2>
            </div>
            <SourceCitations citations={citations} documents={documents} />
          </div>
        </div>
      </MainLayout>
    </ThemeProvider>
  );
}
