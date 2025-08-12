"use client";

import type React from "react";

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Send, User, Bot, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  timestamp: Date;
}

export interface Citation {
  id: string;
  document: string;
  page?: number;
  snippet: string;
}

interface ChatInterfaceProps {
  messages: Message[];
  onMessagesChange: (messages: Message[]) => void;
  onCitationsChange: (citations: Citation[]) => void;
}

export function ChatInterface({
  messages,
  onMessagesChange,
  onCitationsChange,
}: ChatInterfaceProps) {
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [streamingMessage, setStreamingMessage] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingMessage]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
      timestamp: new Date(),
    };

    onMessagesChange([...messages, userMessage]);
    setInput("");
    setIsLoading(true);

    // Call real RAG API
    await callRAGAPI(input.trim());
  };

  const callRAGAPI = async (query: string) => {
    try {
      const res = await fetch(`/api/rag/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      
      if (!res.ok) throw new Error('RAG API failed');
      const data = await res.json();
      
      const assistantMessage: Message = {
        id: Date.now().toString(),
        role: "assistant",
        content: data.answer || "No answer found.",
        citations: data.sources?.map((source: any, index: number) => ({
          id: index.toString(),
          document: source.document || "Unknown",
          page: source.page,
          snippet: source.text || "",
        })) || [],
        timestamp: new Date(),
      };
      
      onMessagesChange([...messages, {
        id: (Date.now() - 1).toString(),
        role: "user",
        content: query,
        timestamp: new Date(),
      }, assistantMessage]);
      
      onCitationsChange(assistantMessage.citations || []);
    } catch (error) {
      const errorMessage: Message = {
        id: Date.now().toString(),
        role: "assistant",
        content: "Sorry, I couldn't process your question. Please make sure you have uploaded documents first.",
        timestamp: new Date(),
      };
      
      onMessagesChange([...messages, {
        id: (Date.now() - 1).toString(),
        role: "user",
        content: query,
        timestamp: new Date(),
      }, errorMessage]);
    }
    
    setStreamingMessage("");
    setIsLoading(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      handleSubmit(e as any);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center py-12">
            <Bot className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium text-foreground mb-2">
              Welcome to the RAG Q&A System
            </h3>
            <p className="text-muted-foreground max-w-md mx-auto">
              Upload documents and ask questions to get AI-powered answers with
              source citations.
            </p>
          </div>
        )}

        {messages.map((message) => (
          <div
            key={message.id}
            className={cn(
              "flex items-start space-x-3",
              message.role === "user" ? "justify-end" : "justify-start"
            )}
          >
            {message.role === "assistant" && (
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center flex-shrink-0">
                <Bot className="h-4 w-4 text-white" />
              </div>
            )}

            <Card
              className={cn(
                "max-w-[85%] sm:max-w-[80%]",
                message.role === "user"
                  ? "bg-primary text-primary-foreground"
                  : "bg-card"
              )}
            >
              <CardContent className="p-3">
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  {message.content}
                </div>

                {message.citations && message.citations.length > 0 && (
                  <div className="flex flex-wrap gap-1 mt-3 pt-3 border-t border-border">
                    {message.citations.map((citation, index) => (
                      <Badge
                        key={citation.id}
                        variant="secondary"
                        className="text-xs cursor-pointer hover:bg-secondary/80"
                      >
                        [{index + 1}] {citation.document}
                      </Badge>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>

            {message.role === "user" && (
              <div className="w-8 h-8 bg-muted rounded-full flex items-center justify-center flex-shrink-0">
                <User className="h-4 w-4" />
              </div>
            )}
          </div>
        ))}

        {/* Streaming Message */}
        {streamingMessage && (
          <div className="flex items-start space-x-3">
            <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center flex-shrink-0">
              <Bot className="h-4 w-4 text-white" />
            </div>
            <Card className="max-w-[80%] bg-card">
              <CardContent className="p-3">
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  {streamingMessage}
                  <span className="inline-block w-2 h-4 bg-current animate-pulse ml-1" />
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-border bg-card/50 backdrop-blur-sm p-3 sm:p-4">
        <form onSubmit={handleSubmit} className="flex space-x-2">
          <div className="flex-1 relative">
            <Textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your documents..."
              className="min-h-[50px] sm:min-h-[60px] max-h-32 resize-none pr-12 text-sm sm:text-base"
              disabled={isLoading}
            />
            <div className="hidden sm:block absolute bottom-2 right-2 text-xs text-muted-foreground">
              Ctrl+Enter to send
            </div>
          </div>
          <Button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="self-end h-[50px] sm:h-auto"
            size="sm"
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </Button>
        </form>
      </div>
    </div>
  );
}
