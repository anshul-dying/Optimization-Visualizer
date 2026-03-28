import React from "react"
import { Badge } from "@/components/ui/badge"
import { Bug } from "lucide-react"
import { cn } from "@/lib/utils"
import { ExecutionState, OptimizerConfig } from "./types"

interface CodeViewProps {
  generateCode: () => string
  isDebugMode: boolean
  executionState: ExecutionState
  config: OptimizerConfig
}

export function CodeView({ generateCode, isDebugMode, executionState, config }: CodeViewProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <div className="md:col-span-2 bg-muted/20 rounded-lg overflow-hidden border border-border/50">
        <div className="bg-muted/50 px-4 py-2 border-b border-border/50 flex justify-between items-center">
          <span className="text-xs font-bold uppercase tracking-widest text-muted-foreground">Source Code</span>
          {isDebugMode && (
            <Badge variant="secondary" className="text-[10px] animate-pulse">
              Line {executionState.activeLine + 1}
            </Badge>
          )}
        </div>
        <div className="p-4 overflow-x-auto font-mono text-sm leading-relaxed">
          {generateCode()
            .split("\n")
            .map((line, idx) => (
              <div
                key={idx}
                className={cn(
                  "flex gap-4 px-2 rounded transition-colors duration-200",
                  executionState.activeLine === idx
                    ? "bg-primary/20 text-primary border-l-2 border-primary -ml-[2px]"
                    : idx > 0 && line.trim() === ""
                      ? "h-4"
                      : "hover:bg-muted/30"
                )}
              >
                <span className="w-6 text-right text-muted-foreground/50 select-none text-[10px] pt-1">{idx + 1}</span>
                <span className="whitespace-pre">{line}</span>
              </div>
            ))}
        </div>
      </div>

      <div className="bg-muted/20 rounded-lg overflow-hidden border border-border/50">
        <div className="bg-muted/50 px-4 py-2 border-b border-border/50 flex items-center gap-2">
          <Bug className="w-3 h-3 text-primary" />
          <span className="text-xs font-bold uppercase tracking-widest text-muted-foreground">Variables</span>
        </div>
        <div className="p-4 space-y-3 font-mono text-xs">
          {Object.entries(executionState.variables).length > 0 ? (
            Object.entries(executionState.variables).map(([key, val]) => (
              <div key={key} className="flex justify-between items-center border-b border-border/30 pb-1">
                <span className="text-primary">{key}</span>
                <span className="text-foreground font-bold">
                  {typeof val === "number" ? val.toFixed(4) : String(val)}
                </span>
              </div>
            ))
          ) : (
            <div className="text-center py-8 text-muted-foreground italic">
              {isDebugMode ? "Start animation to see values" : "Enable Debug Mode"}
            </div>
          )}

          {isDebugMode && (
            <div className="pt-4 mt-4 border-t border-primary/20">
              <p className="text-[10px] font-bold text-primary uppercase mb-2">Global Config</p>
              <div className="space-y-1 opacity-70">
                <div className="flex justify-between">
                  <span>learning_rate</span>
                  <span>{config.learningRate}</span>
                </div>
                {config.momentum && (
                  <div className="flex justify-between">
                    <span>momentum</span>
                    <span>{config.momentum}</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
