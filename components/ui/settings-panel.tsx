"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Settings, ChevronDown, ChevronRight, RotateCcw } from "lucide-react"

interface SettingsConfig {
  model: string
  temperature: number
  maxTokens: number
  chunkSize: number
}

interface SettingsPanelProps {
  settings: SettingsConfig
  onSettingsChange: (settings: SettingsConfig) => void
}

export function SettingsPanel({ settings, onSettingsChange }: SettingsPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false)

  const updateSetting = (key: keyof SettingsConfig, value: any) => {
    onSettingsChange({
      ...settings,
      [key]: value,
    })
  }

  const resetToDefaults = () => {
    onSettingsChange({
      model: "gpt-4",
      temperature: 0.7,
      maxTokens: 2000,
      chunkSize: 500,
    })
  }

  return (
    <Card className="border border-border/50">
      <Collapsible open={isExpanded} onOpenChange={setIsExpanded}>
        <CollapsibleTrigger asChild>
          <CardHeader className="p-3 cursor-pointer hover:bg-muted/50 transition-colors">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Settings className="h-4 w-4 text-muted-foreground" />
                <CardTitle className="text-sm">Settings</CardTitle>
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
          <CardContent className="p-3 pt-0 space-y-4 border-t border-border/50">
            {/* Model Selection */}
            <div className="space-y-2">
              <Label htmlFor="model" className="text-xs font-medium">
                Model
              </Label>
              <Select value={settings.model} onValueChange={(value) => updateSetting("model", value)}>
                <SelectTrigger className="h-8 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="gpt-4">GPT-4</SelectItem>
                  <SelectItem value="gpt-4-turbo">GPT-4 Turbo</SelectItem>
                  <SelectItem value="gpt-3.5-turbo">GPT-3.5 Turbo</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Temperature */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs font-medium">Temperature</Label>
                <span className="text-xs text-muted-foreground">{settings.temperature}</span>
              </div>
              <Slider
                value={[settings.temperature]}
                onValueChange={([value]) => updateSetting("temperature", value)}
                max={2}
                min={0}
                step={0.1}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Focused</span>
                <span>Creative</span>
              </div>
            </div>

            {/* Max Tokens */}
            <div className="space-y-2">
              <Label htmlFor="maxTokens" className="text-xs font-medium">
                Max Tokens
              </Label>
              <Input
                id="maxTokens"
                type="number"
                value={settings.maxTokens}
                onChange={(e) => updateSetting("maxTokens", Number.parseInt(e.target.value))}
                min={100}
                max={4000}
                className="h-8 text-xs"
              />
            </div>

            {/* Chunk Size */}
            <div className="space-y-2">
              <Label htmlFor="chunkSize" className="text-xs font-medium">
                Chunk Size
              </Label>
              <Input
                id="chunkSize"
                type="number"
                value={settings.chunkSize}
                onChange={(e) => updateSetting("chunkSize", Number.parseInt(e.target.value))}
                min={100}
                max={1000}
                className="h-8 text-xs"
              />
              <p className="text-xs text-muted-foreground">Tokens per document chunk</p>
            </div>

            {/* Reset Button */}
            <Button variant="outline" size="sm" onClick={resetToDefaults} className="w-full text-xs bg-transparent">
              <RotateCcw className="h-3 w-3 mr-1" />
              Reset to Defaults
            </Button>
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  )
}
