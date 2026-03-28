import React from "react"

interface CanvasViewProps {
  canvasRef: React.RefObject<HTMLCanvasElement>
  handleCanvasClick: (e: React.MouseEvent<HTMLCanvasElement>) => void
  isRunning: boolean
}

export function CanvasView({ canvasRef, handleCanvasClick, isRunning }: CanvasViewProps) {
  return (
    <div className="bg-muted/20 rounded-lg p-4 relative cursor-crosshair group">
      <canvas
        ref={canvasRef}
        onClick={handleCanvasClick}
        className="w-full h-auto border rounded bg-background/50"
        style={{ maxWidth: "100%", height: "400px" }}
      />
      {!isRunning && (
        <div className="absolute top-8 left-1/2 -translate-x-1/2 bg-primary/90 text-primary-foreground px-3 py-1 rounded-full text-xs font-medium opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap">
          Click anywhere to set starting point
        </div>
      )}
    </div>
  )
}
