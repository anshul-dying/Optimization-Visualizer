"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Maximize2, Minimize2, X, Brain, Bug, Play, Pause, RotateCcw, Settings } from "lucide-react"
import { useTheme } from "next-themes"

import { useOptimization } from "./use-optimization"
import { ConfigPanel } from "./config-panel"
import { CodeView } from "./code-view"
import { CanvasView } from "./canvas-view"
import { optimizerConfigs, objectiveFunctions } from "./constants"
import { OptimizerConfig } from "./types"
import { cn } from "@/lib/utils"

export function InteractivePlayground() {
  const { resolvedTheme } = useTheme()
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [selectedOptimizer, setSelectedOptimizer] = useState<string>("adam")
  const [selectedFunction, setSelectedFunction] = useState<keyof typeof objectiveFunctions>("quadratic")
  const [config, setConfig] = useState<OptimizerConfig>(optimizerConfigs.adam)
  const [isFullScreen, setIsFullScreen] = useState(false)

  const {
    isRunning,
    currentPoint,
    path,
    iteration,
    loss,
    isDebugMode,
    setIsDebugMode,
    stepDelay,
    setStepDelay,
    zoom,
    setZoom,
    panOffset,
    setPanOffset,
    resetView,
    executionState,
    toggleAnimation,
    reset,
    handleCanvasClick
  } = useOptimization(selectedOptimizer, selectedFunction, config)

  // Memoized drawing function to avoid stale closures and unnecessary work
  const drawVisualization = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Don't reset canvas.width/height here as it clears the canvas and state
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const scale = 50 * zoom
    const offsetX = (canvas.width / 2) + panOffset.x
    const offsetY = (canvas.height / 2) + panOffset.y

    // Draw contour lines - optimization: use dynamic resolution based on zoom
    const resolution = zoom > 4 ? 2 : zoom > 2 ? 4 : 8
    for (let i = 0; i < canvas.width; i += resolution) {
      for (let j = 0; j < canvas.height; j += resolution) {
        const x = (i - offsetX) / scale
        const y = (j - offsetY) / scale
        const value = objectiveFunctions[selectedFunction](x, y)
        const normalizedValue = Math.min(1, Math.log10(value + 1) / 2)
        
        let r, g, b
        if (normalizedValue < 0.5) {
          const t = normalizedValue * 2
          r = 15 + t * (99 - 15)
          g = 23 + t * (102 - 23)
          b = 42 + t * (241 - 42)
        } else {
          const t = (normalizedValue - 0.5) * 2
          r = 99 + t * (34 - 99)
          g = 102 + t * (211 - 102)
          b = 241 + t * (238 - 241)
        }
        
        const alpha = resolvedTheme === "dark" ? 0.4 + normalizedValue * 0.4 : 0.3 + normalizedValue * 0.3
        ctx.fillStyle = `rgba(${Math.floor(r)}, ${Math.floor(g)}, ${Math.floor(b)}, ${alpha})`
        ctx.fillRect(i, j, resolution, resolution)
        
        const logValue = Math.log(value + 1)
        if (Math.abs((logValue * 1.5) % 1) < 0.07) {
          ctx.fillStyle = resolvedTheme === "dark" ? "rgba(255, 255, 255, 0.15)" : "rgba(255, 255, 255, 0.3)"
          ctx.fillRect(i, j, resolution, resolution)
        }
      }
    }

    // Draw optimization path
    if (path.length > 1) {
      ctx.shadowBlur = 15
      ctx.shadowColor = "rgba(249, 115, 22, 0.4)"
      ctx.strokeStyle = "#f97316"
      ctx.lineWidth = 3
      ctx.lineJoin = "round"
      ctx.lineCap = "round"
      ctx.beginPath()

      for (let i = 0; i < path.length; i++) {
        const x = path[i].x * scale + offsetX
        const y = path[i].y * scale + offsetY
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      }
      ctx.stroke()
      ctx.shadowBlur = 0

      path.forEach((point, index) => {
        const x = point.x * scale + offsetX
        const y = point.y * scale + offsetY
        const progress = index / path.length
        const alpha = 0.2 + progress * 0.6
        const radius = 2 + progress * 2

        ctx.fillStyle = `rgba(249, 115, 22, ${alpha})`
        ctx.beginPath()
        ctx.arc(x, y, radius, 0, 2 * Math.PI)
        ctx.fill()
      })
    }

    // Current point
    const currentX = currentPoint.x * scale + offsetX
    const currentY = currentPoint.y * scale + offsetY
    ctx.shadowBlur = 15
    ctx.shadowColor = "rgba(239, 68, 68, 0.5)"
    ctx.fillStyle = "#ef4444"
    ctx.beginPath()
    ctx.arc(currentX, currentY, 6, 0, 2 * Math.PI)
    ctx.fill()
    ctx.shadowBlur = 0

    // Axes
    ctx.strokeStyle = resolvedTheme === "dark" ? "rgba(255, 255, 255, 0.3)" : "rgba(0, 0, 0, 0.3)"
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(0, offsetY); ctx.lineTo(canvas.width, offsetY)
    ctx.moveTo(offsetX, 0); ctx.lineTo(offsetX, canvas.height)
    ctx.stroke()
  }, [currentPoint, path, selectedFunction, resolvedTheme, zoom, panOffset])

  // Handle canvas sizing - only when fullScreen changes
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    canvas.width = isFullScreen ? 900 : 600
    canvas.height = isFullScreen ? 600 : 400
    drawVisualization()
  }, [isFullScreen, drawVisualization])

  // Redraw when state changes
  useEffect(() => {
    drawVisualization()
  }, [drawVisualization])

  useEffect(() => {
    setConfig(optimizerConfigs[selectedOptimizer])
    reset()
  }, [selectedOptimizer, reset])

  const generateCode = () => {
    const optimizerCode = {
      gd: `# Gradient Descent
def gradient_descent(x, y, lr=${config.learningRate}):
    grad_x, grad_y = compute_gradient(x, y)
    x -= lr * grad_x
    y -= lr * grad_y
    return x, y`,

      sgd: `# Stochastic Gradient Descent
def sgd_step(x, y, lr=${config.learningRate}):
    grad_x, grad_y = compute_gradient(x, y)
    noise = np.random.normal(0, 0.1, 2)
    x -= lr * (grad_x + noise[0])
    y -= lr * (grad_y + noise[1])
    return x, y`,

      momentum: `# Momentum Optimizer
def momentum_step(x, y, velocity, lr=${config.learningRate}, momentum=${config.momentum}):
    grad_x, grad_y = compute_gradient(x, y)
    velocity_x = momentum * velocity_x - lr * grad_x
    velocity_y = momentum * velocity_y - lr * grad_y
    x += velocity_x
    y += velocity_y
    return x, y`,

      adam: `# Adam Optimizer
def adam_step(x, y, m, v, t, lr=${config.learningRate}, beta1=${config.beta1 || 0.9}, beta2=${config.beta2 || 0.999}):
    grad_x, grad_y = compute_gradient(x, y)
    m_x = beta1 * m_x + (1 - beta1) * grad_x
    m_y = beta1 * m_y + (1 - beta1) * grad_y
    v_x = beta2 * v_x + (1 - beta2) * grad_x**2
    v_y = beta2 * v_y + (1 - beta2) * grad_y**2
    m_hat_x = m_x / (1 - beta1**t)
    m_hat_y = m_y / (1 - beta1**t)
    v_hat_x = v_x / (1 - beta2**t)
    v_hat_y = v_y / (1 - beta2**t)
    x -= lr * m_hat_x / (np.sqrt(v_hat_x) + 1e-8)
    y -= lr * m_hat_y / (np.sqrt(v_hat_y) + 1e-8)
    return x, y`,

      rmsprop: `# RMSprop Optimizer
def rmsprop_step(x, y, g, lr=${config.learningRate}, beta=${config.momentum || 0.9}):
    grad_x, grad_y = compute_gradient(x, y)
    g_x = beta * g_x + (1 - beta) * grad_x**2
    g_y = beta * g_y + (1 - beta) * grad_y**2
    x -= lr * grad_x / (np.sqrt(g_x) + 1e-8)
    y -= lr * grad_y / (np.sqrt(g_y) + 1e-8)
    return x, y`,

      adagrad: `# AdaGrad Optimizer
def adagrad_step(x, y, g, lr=${config.learningRate}):
    grad_x, grad_y = compute_gradient(x, y)
    g_x += grad_x**2
    g_y += grad_y**2
    x -= lr * grad_x / (np.sqrt(g_x) + 1e-8)
    y -= lr * grad_y / (np.sqrt(g_y) + 1e-8)
    return x, y`,
    }
    return optimizerCode[selectedOptimizer as keyof typeof optimizerCode] || optimizerCode.gd
  }

  return (
    <section 
      id="playground" 
      className={cn(
        "py-20 px-4 bg-background transition-all duration-300", 
        isFullScreen && "fixed inset-0 z-[60] py-0 px-0 h-screen overflow-hidden"
      )}
    >
      <div className={cn("container mx-auto", isFullScreen && "max-w-none px-0 h-full flex flex-col")}>
        {!isFullScreen && (
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold mb-4 text-balance">Interactive Playground</h2>
            <p className="text-xl text-muted-foreground text-balance max-w-2xl mx-auto">
              Experiment with different optimizers and objective functions in real-time
            </p>
          </div>
        )}

        {isFullScreen && (
          <div className="bg-card border-b border-border p-3 flex items-center justify-between shadow-sm relative z-[70]">
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-2 mr-2">
                <div className="w-8 h-8 bg-primary rounded flex items-center justify-center">
                  <Brain className="w-5 h-5 text-primary-foreground" />
                </div>
                <h2 className="font-bold text-sm uppercase tracking-tighter">Optimizer Debugger</h2>
              </div>
              
              <div className="flex items-center gap-2 border-l pl-6 border-border">
                <label className="text-[10px] font-bold uppercase text-muted-foreground">Optimizer</label>
                <Select value={selectedOptimizer} onValueChange={setSelectedOptimizer}>
                  <SelectTrigger className="h-8 min-w-[140px] text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="z-[80]">
                    <SelectItem value="gd">Gradient Descent</SelectItem>
                    <SelectItem value="sgd">SGD</SelectItem>
                    <SelectItem value="momentum">Momentum</SelectItem>
                    <SelectItem value="adam">Adam</SelectItem>
                    <SelectItem value="rmsprop">RMSprop</SelectItem>
                    <SelectItem value="adagrad">AdaGrad</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="flex items-center gap-2">
                <label className="text-[10px] font-bold uppercase text-muted-foreground">Function</label>
                <Select value={selectedFunction} onValueChange={setSelectedFunction}>
                  <SelectTrigger className="h-8 min-w-[140px] text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="z-[80]">
                    <SelectItem value="quadratic">Quadratic</SelectItem>
                    <SelectItem value="rosenbrock">Rosenbrock</SelectItem>
                    <SelectItem value="himmelblau">Himmelblau</SelectItem>
                    <SelectItem value="beale">Beale</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="h-6 w-px bg-border" />

              <div className="flex items-center gap-2">
                <Button 
                  size="sm" 
                  variant={isRunning ? "destructive" : "default"} 
                  onClick={toggleAnimation}
                  className="h-8 px-4"
                >
                  {isRunning ? <Pause className="w-3 h-3 mr-2" /> : <Play className="w-3 h-3 mr-2" />}
                  {isRunning ? "Pause" : "Run"}
                </Button>
                <Button size="sm" variant="outline" onClick={reset} className="h-8 w-8 p-0">
                  <RotateCcw className="w-3 h-3" />
                </Button>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="flex items-center gap-4 bg-muted/50 px-3 py-1 rounded-md border border-border">
                <div className="flex flex-col items-center">
                  <span className="text-[9px] uppercase font-bold text-muted-foreground leading-none">Loss</span>
                  <span className="text-xs font-mono font-bold text-primary">{loss.toFixed(4)}</span>
                </div>
                <div className="w-px h-6 bg-border" />
                <div className="flex flex-col items-center">
                  <span className="text-[9px] uppercase font-bold text-muted-foreground leading-none">Iter</span>
                  <span className="text-xs font-mono font-bold">{iteration}</span>
                </div>
                <div className="w-px h-6 bg-border" />
                <div className="flex flex-col items-center min-w-[40px]">
                  <span className="text-[9px] uppercase font-bold text-muted-foreground leading-none">Zoom</span>
                  <span className="text-xs font-mono font-bold">{Math.round(zoom * 100)}%</span>
                </div>
              </div>
              
              {(zoom !== 1 || panOffset.x !== 0 || panOffset.y !== 0) && (
                <Button variant="outline" size="sm" onClick={resetView} className="h-8 text-[10px] uppercase font-bold">
                  <RotateCcw className="w-3 h-3 mr-1" /> Reset View
                </Button>
              )}
              
              <Button variant="ghost" size="sm" onClick={() => setIsFullScreen(false)} className="h-8 text-xs">
                <Minimize2 className="w-4 h-4 mr-2" />
                Exit
              </Button>
            </div>
          </div>
        )}

        <div className={cn("grid grid-cols-1 lg:grid-cols-3 gap-6", isFullScreen && "flex-1 grid-cols-1 lg:grid-cols-12 gap-0 overflow-hidden")}>
          {!isFullScreen ? (
            <>
              <ConfigPanel
                selectedOptimizer={selectedOptimizer}
                setSelectedOptimizer={setSelectedOptimizer}
                selectedFunction={selectedFunction}
                setSelectedFunction={setSelectedFunction}
                config={config}
                setConfig={setConfig}
                isDebugMode={isDebugMode}
                setIsDebugMode={setIsDebugMode}
                stepDelay={stepDelay}
                setStepDelay={setStepDelay}
                isRunning={isRunning}
                toggleAnimation={toggleAnimation}
                reset={reset}
                iteration={iteration}
                loss={loss}
                currentPoint={currentPoint}
              />

              <Card className="lg:col-span-2 shadow-none border-0">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle>Optimization Visualization</CardTitle>
                      <CardDescription>Watch the optimizer navigate the loss landscape in real-time</CardDescription>
                    </div>
                    <div className="flex items-center gap-2">
                      {isDebugMode && (
                        <Button variant="outline" size="sm" onClick={() => setIsFullScreen(true)}>
                          <Maximize2 className="w-4 h-4 mr-2" />
                          Debugger View
                        </Button>
                      )}
                      <Button variant="ghost" size="icon" onClick={() => setIsFullScreen(true)}>
                        <Maximize2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <Tabs defaultValue="visualization" className="w-full">
                    <TabsList className="grid w-full grid-cols-2">
                      <TabsTrigger value="visualization">Visualization</TabsTrigger>
                      <TabsTrigger value="code">Code</TabsTrigger>
                    </TabsList>

                    <TabsContent value="visualization" className="mt-4">
                      <CanvasView 
                        canvasRef={canvasRef} 
                        handleCanvasClick={handleCanvasClick} 
                        isRunning={isRunning} 
                        zoom={zoom}
                        setZoom={setZoom}
                        panOffset={panOffset}
                        setPanOffset={setPanOffset}
                        resetView={resetView}
                      />
                    </TabsContent>

                    <TabsContent value="code" className="mt-4">
                      <CodeView
                        generateCode={generateCode}
                        isDebugMode={isDebugMode}
                        executionState={executionState}
                        config={config}
                      />
                    </TabsContent>
                  </Tabs>
                </CardContent>
              </Card>
            </>
          ) : (
            <>
              {/* Full Screen Layout: 3 Columns (Config, Canvas, Code) */}
              <div className="lg:col-span-2 border-r border-border bg-muted/5 flex flex-col overflow-hidden">
                <div className="p-3 border-b bg-card">
                  <h3 className="text-[10px] font-bold uppercase text-muted-foreground flex items-center gap-2">
                    <Settings className="w-3 h-3" /> Parameters
                  </h3>
                </div>
                <div className="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-6">
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <label className="text-[11px] font-medium uppercase text-muted-foreground">Learning Rate</label>
                        <span className="text-xs font-mono">{config.learningRate}</span>
                      </div>
                      <Slider
                        value={[config.learningRate]}
                        onValueChange={([value]) => setConfig((prev) => ({ ...prev, learningRate: value }))}
                        min={0.001} max={0.1} step={0.001}
                      />
                    </div>

                    {selectedOptimizer === "momentum" && (
                      <div className="space-y-2">
                        <div className="flex justify-between items-center">
                          <label className="text-[11px] font-medium uppercase text-muted-foreground">Momentum</label>
                          <span className="text-xs font-mono">{config.momentum}</span>
                        </div>
                        <Slider
                          value={[config.momentum || 0.9]}
                          onValueChange={([value]) => setConfig((prev) => ({ ...prev, momentum: value }))}
                          min={0} max={0.99} step={0.01}
                        />
                      </div>
                    )}

                    <div className="pt-4 border-t">
                      <div className="flex items-center justify-between mb-4">
                        <Label className="text-[11px] font-bold uppercase text-muted-foreground flex items-center gap-2">
                          <Bug className="w-3 h-3" /> Debug Step (ms)
                        </Label>
                        <span className="text-xs font-mono">{stepDelay}ms</span>
                      </div>
                      <Slider
                        value={[stepDelay]}
                        onValueChange={([value]) => setStepDelay(value)}
                        min={50} max={1000} step={50}
                      />
                    </div>
                  </div>
                </div>
              </div>

              <div className="lg:col-span-7 bg-background flex flex-col h-full overflow-hidden">
                <div className="flex-1 relative flex items-center justify-center p-4">
                  <CanvasView 
                    canvasRef={canvasRef} 
                    handleCanvasClick={handleCanvasClick} 
                    isRunning={isRunning} 
                    isFullScreen={true} 
                    zoom={zoom}
                    setZoom={setZoom}
                    panOffset={panOffset}
                    setPanOffset={setPanOffset}
                    resetView={resetView}
                  />
                </div>
              </div>

              <div className="lg:col-span-3 border-l border-border bg-card flex flex-col h-full overflow-hidden">
                <div className="p-3 border-b bg-muted/20">
                  <h3 className="text-[10px] font-bold uppercase text-muted-foreground flex items-center gap-2">
                    <Bug className="w-3 h-3 text-primary" /> Trace & Inspector
                  </h3>
                </div>
                <div className="flex-1 overflow-y-auto custom-scrollbar">
                  <div className="p-2">
                    <CodeView
                      generateCode={generateCode}
                      isDebugMode={isDebugMode}
                      executionState={executionState}
                      config={config}
                      isStacked={true}
                    />
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </section>
  )
}
