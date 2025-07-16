"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { FileText, ChevronDown, ChevronRight, ExternalLink } from "lucide-react"

interface Citation {
  id: string
  document: string
  page?: number
  snippet: string
}

interface Document {
  id: string
  name: string
  size: number
  type: string
  status: string
}

interface SourceCitationsProps {
  citations: Citation[]
  documents: Document[]
}

export function SourceCitations({ citations, documents }: SourceCitationsProps) {
  const [expandedCitations, setExpandedCitations] = useState<Set<string>>(new Set())

  const toggleCitation = (citationId: string) => {
    const newExpanded = new Set(expandedCitations)
    if (newExpanded.has(citationId)) {
      newExpanded.delete(citationId)
    } else {
      newExpanded.add(citationId)
    }
    setExpandedCitations(newExpanded)
  }

  const getDocumentInfo = (documentName: string) => {
    return documents.find((doc) => doc.name === documentName)
  }

  if (citations.length === 0) {
    return (
      <div className="p-4">
        <div className="text-center py-8 text-muted-foreground">
          <FileText className="mx-auto h-12 w-12 mb-2 opacity-50" />
          <p className="text-sm">No sources cited yet</p>
          <p className="text-xs mt-1">Citations will appear here when you ask questions</p>
        </div>
      </div>
    )
  }

  return (
    <div className="p-4 space-y-3 max-h-full overflow-y-auto">
      {citations.map((citation, index) => {
        const isExpanded = expandedCitations.has(citation.id)
        const documentInfo = getDocumentInfo(citation.document)

        return (
          <Card key={citation.id} className="border border-border/50">
            <Collapsible>
              <CollapsibleTrigger asChild>
                <CardHeader
                  className="p-3 cursor-pointer hover:bg-muted/50 transition-colors"
                  onClick={() => toggleCitation(citation.id)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Badge variant="outline" className="text-xs">
                        {index + 1}
                      </Badge>
                      <FileText className="h-4 w-4 text-muted-foreground" />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate">{citation.document}</p>
                        {citation.page && <p className="text-xs text-muted-foreground">Page {citation.page}</p>}
                      </div>
                    </div>
                    {isExpanded ? (
                      <ChevronDown className="h-4 w-4 text-muted-foreground" />
                    ) : (
                      <ChevronRight className="h-4 w-4 text-muted-foreground" />
                    )}
                  </div>
                </CardHeader>
              </CollapsibleTrigger>

              <CollapsibleContent>
                <CardContent className="p-3 pt-0 border-t border-border/50">
                  <div className="space-y-3">
                    {/* Document Info */}
                    {documentInfo && (
                      <div className="flex items-center justify-between text-xs text-muted-foreground">
                        <span>Status: {documentInfo.status}</span>
                        <Badge variant="secondary" className="text-xs">
                          {documentInfo.type.split("/").pop()?.toUpperCase()}
                        </Badge>
                      </div>
                    )}

                    {/* Citation Snippet */}
                    <div className="bg-muted/30 rounded-md p-3">
                      <p className="text-sm text-foreground leading-relaxed">"{citation.snippet}"</p>
                    </div>

                    {/* Actions */}
                    <div className="flex space-x-2">
                      <Button variant="outline" size="sm" className="text-xs bg-transparent">
                        <ExternalLink className="h-3 w-3 mr-1" />
                        View in Document
                      </Button>
                      <Button variant="ghost" size="sm" className="text-xs">
                        Copy Quote
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </CollapsibleContent>
            </Collapsible>
          </Card>
        )
      })}

      {/* Summary */}
      <div className="mt-4 p-3 bg-muted/20 rounded-md">
        <p className="text-xs text-muted-foreground text-center">
          {citations.length} source{citations.length !== 1 ? "s" : ""} cited from{" "}
          {new Set(citations.map((c) => c.document)).size} document
          {new Set(citations.map((c) => c.document)).size !== 1 ? "s" : ""}
        </p>
      </div>
    </div>
  )
}
