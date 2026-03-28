import React, { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { RotateCcw, ZoomIn, ZoomOut, Move } from "lucide-react"
import { Point } from "./types"
import { cn } from "@/lib/utils"

interface CanvasViewProps {
  canvasRef: React.RefObject<HTMLCanvasElement>
  handleCanvasClick: (e: React.MouseEvent<HTMLCanvasElement>) => void
  isRunning: boolean
  isFullScreen?: boolean
  zoom: number
  setZoom: (val: number | ((v: number) => number)) => void
  panOffset: Point
  setPanOffset: (val: Point | ((v: Point) => Point)) => void
  resetView: () => void
}

export function CanvasView({ 
  canvasRef, 
  handleCanvasClick, 
  isRunning, 
  isFullScreen = false,
  zoom,
  setZoom,
  panOffset,
  setPanOffset,
  resetView
}: CanvasViewProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [isDragging, setIsDragging] = useState(false)
  const lastMousePos = useRef<Point>({ x: 0, y: 0 })

  // Use native event for wheel to ensure we can preventDefault (zoom instead of scroll)
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const handleWheelNative = (e: WheelEvent) => {
      e.preventDefault()
      const delta = e.deltaY > 0 ? 0.9 : 1.1
      setZoom(prev => {
        const next = prev * delta
        return Math.min(Math.max(next, 0.1), 20)
      })
    }

    container.addEventListener('wheel', handleWheelNative, { passive: false })
    return () => container.removeEventListener('wheel', handleWheelNative)
  }, [setZoom])

  const handleMouseDown = (e: React.MouseEvent) => {
    // Pan with Middle Mouse Button, or Left Click + Alt, or Left Click drag if NOT setting a point
    // Let's simplify: Middle button or Alt+Left always pans. 
    // Left click alone: if isRunning it pans, if !isRunning it sets point (onMouseUp if no movement)
    if (e.button === 1 || (e.button === 0 && e.altKey) || (e.button === 0 && isRunning)) {
      setIsDragging(true)
      lastMousePos.current = { x: e.clientX, y: e.clientY }
      e.preventDefault()
    }
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      const dx = e.clientX - lastMousePos.current.x
      const dy = e.clientY - lastMousePos.current.y
      setPanOffset(prev => ({ x: prev.x + dx, y: prev.y + dy }))
      lastMousePos.current = { x: e.clientX, y: e.clientY }
    }
  }

  const handleMouseUp = (e: React.MouseEvent) => {
    if (isDragging) {
      setIsDragging(false)
    }
  }

  const adjustZoom = (factor: number) => {
    setZoom(prev => Math.min(Math.max(prev * factor, 0.1), 20))
  }

  return (
    <div 
      ref={containerRef}
      className={cn(
        "bg-muted/20 rounded-lg p-4 relative group w-full h-full flex flex-col items-center justify-center overflow-hidden touch-none select-none",
        isDragging ? "cursor-grabbing" : "cursor-crosshair"
      )}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={() => setIsDragging(false)}
      onContextMenu={(e) => e.preventDefault()}
    >
      <div className="relative flex items-center justify-center pointer-events-none">
        <canvas
          ref={canvasRef}
          onClick={(e) => {
            // Only trigger click if we weren't dragging
            if (!isDragging) {
              handleCanvasClick(e)
            }
          }}
          className="w-auto h-auto border rounded bg-background/50 shadow-2xl pointer-events-auto"
          style={{ 
            maxWidth: "none", 
            maxHeight: isFullScreen ? "85vh" : "400px",
          }}
        />
        
        {/* View Controls Overlay */}
        <div className="absolute top-4 left-4 flex flex-col gap-2 pointer-events-auto">
          <div className="flex items-center gap-1 bg-background/90 backdrop-blur-md border border-border shadow-xl rounded-lg p-1.5">
            <Button 
              variant="ghost" 
              size="icon" 
              className="h-8 w-8 hover:bg-primary/10 hover:text-primary transition-colors" 
              onClick={() => adjustZoom(1.2)} 
              title="Zoom In"
            >
              <ZoomIn className="h-4 w-4" />
            </Button>
            
            <div className="text-[11px] font-mono font-black w-12 text-center select-none bg-muted/50 py-1 rounded">
              {Math.round(zoom * 100)}%
            </div>
            
            <Button 
              variant="ghost" 
              size="icon" 
              className="h-8 w-8 hover:bg-primary/10 hover:text-primary transition-colors" 
              onClick={() => adjustZoom(0.8)} 
              title="Zoom Out"
            >
              <ZoomOut className="h-4 w-4" />
            </Button>
            
            <div className="w-px h-5 bg-border mx-1" />
            
            <Button 
              variant="ghost" 
              size="icon" 
              className={cn("h-8 w-8 transition-all", (zoom !== 1 || panOffset.x !== 0 || panOffset.y !== 0) ? "text-primary animate-pulse" : "text-muted-foreground")} 
              onClick={resetView} 
              title="Reset View"
            >
              <RotateCcw className="h-4 w-4" />
            </Button>
          </div>
          
          <div className="flex flex-col gap-1">
            <div className="bg-background/80 backdrop-blur-sm px-2 py-1.5 rounded-md text-[10px] font-bold text-muted-foreground border border-border/40 flex items-center gap-3 shadow-sm">
              <div className="flex items-center gap-1.5">
                <kbd className="px-1.5 py-0.5 bg-muted border border-border shadow-inner rounded text-[9px]">Scroll</kbd>
                <span>Zoom</span>
              </div>
              <div className="w-px h-3 bg-border/50" />
              <div className="flex items-center gap-1.5">
                <kbd className="px-1.5 py-0.5 bg-muted border border-border shadow-inner rounded text-[9px]">Alt + Drag</kbd>
                <span>Pan</span>
              </div>
            </div>
          </div>
        </div>

        {isDragging && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="bg-primary/20 p-4 rounded-full animate-ping">
              <Move className="w-8 h-8 text-primary" />
            </div>
          </div>
        )}

        {!isRunning && !isDragging && (
          <div className="absolute bottom-8 left-1/2 -translate-x-1/2 bg-primary/90 text-primary-foreground px-4 py-2 rounded-full text-xs font-bold opacity-0 group-hover:opacity-100 transition-all duration-300 pointer-events-none whitespace-nowrap shadow-2xl border border-white/20 translate-y-2 group-hover:translate-y-0">
            Click to set starting point | Alt + Drag to pan
          </div>
        )}
      </div>
    </div>
  )
}
