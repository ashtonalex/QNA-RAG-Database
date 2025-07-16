"use client"

import { useState } from "react"
import { MainLayout } from "@/components/layout/main-layout"
import { DocumentUpload } from "@/components/ui/document-upload"
import { ChatInterface } from "@/components/ui/chat-interface"
import { SourceCitations } from "@/components/ui/source-citations"
import { SettingsPanel } from "@/components/ui/settings-panel"
import { ThemeProvider } from "@/components/theme-provider"

export default function RAGSystem() {
  const [documents, setDocuments] = useState([])
  const [messages, setMessages] = useState([])
  const [settings, setSettings] = useState({
    model: "gpt-4",
    temperature: 0.7,
    maxTokens: 2000,
    chunkSize: 500,
  })
  const [citations, setCitations] = useState([])

  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
      <MainLayout>
        <div className="flex h-screen bg-background">
          {/* Left Sidebar - Document Management */}
          <div className="w-80 border-r border-border bg-card/50 backdrop-blur-sm">
            <div className="p-4 border-b border-border">
              <h2 className="text-lg font-semibold text-foreground">Documents</h2>
            </div>
            <div className="p-4 space-y-4">
              <DocumentUpload documents={documents} onDocumentsChange={setDocuments} />
              <SettingsPanel settings={settings} onSettingsChange={setSettings} />
            </div>
          </div>

          {/* Main Chat Interface */}
          <div className="flex-1 flex flex-col">
            <ChatInterface
              messages={messages}
              onMessagesChange={setMessages}
              settings={settings}
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
  )
}
