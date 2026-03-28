"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Play, Pause, RotateCcw, Settings } from "lucide-react"
import { useTheme } from "next-themes"

interface OptimizerConfig {
  learningRate: number
  momentum?: number
  beta1?: number
  beta2?: number
  epsilon?: number
  batchSize?: number
}

interface Point {
  x: number
  y: number
}

const optimizerConfigs = {
  gd: { learningRate: 0.01 },
  sgd: { learningRate: 0.01 },
  momentum: { learningRate: 0.01, momentum: 0.9 },
  adam: { learningRate: 0.001, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 },
  rmsprop: { learningRate: 0.001, momentum: 0.9, epsilon: 1e-8 },
  adagrad: { learningRate: 0.01, epsilon: 1e-8 },
}

const objectiveFunctions = {
  quadratic: (x: number, y: number) => x * x + y * y,
  rosenbrock: (x: number, y: number) => (1 - x) ** 2 + 100 * (y - x ** 2) ** 2,
  himmelblau: (x: number, y: number) => (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2,
  beale: (x: number, y: number) =>
    (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2,
}

export function InteractivePlayground() {
  const { resolvedTheme } = useTheme()
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [selectedOptimizer, setSelectedOptimizer] = useState<keyof typeof optimizerConfigs>("adam")
  const [selectedFunction, setSelectedFunction] = useState<keyof typeof objectiveFunctions>("quadratic")
  const [config, setConfig] = useState<OptimizerConfig>(optimizerConfigs.adam)
  const [isRunning, setIsRunning] = useState(false)
  const [currentPoint, setCurrentPoint] = useState<Point>({ x: 2, y: 2 })
  const [path, setPath] = useState<Point[]>([])
  const [iteration, setIteration] = useState(0)
  const [loss, setLoss] = useState(0)
  const [showCode, setShowCode] = useState(false)

  // Animation state
  const isRunningRef = useRef(isRunning)
  const animationRef = useRef<number | null>(null)
  const currentPointRef = useRef<Point>(currentPoint)
  const iterationRef = useRef<number>(iteration)
  const momentumRef = useRef<Point>({ x: 0, y: 0 })
  const adamMRef = useRef<Point>({ x: 0, y: 0 })
  const adamVRef = useRef<Point>({ x: 0, y: 0 })
  const adagradGRef = useRef<Point>({ x: 0, y: 0 })
  const rmspropGRef = useRef<Point>({ x: 0, y: 0 })

  useEffect(() => {
    setConfig(optimizerConfigs[selectedOptimizer])
    reset()
  }, [selectedOptimizer])

  useEffect(() => {
    drawVisualization()
  }, [currentPoint, path, selectedFunction, resolvedTheme])

  useEffect(() => {
    isRunningRef.current = isRunning
  }, [isRunning])

  useEffect(() => {
    currentPointRef.current = currentPoint
  }, [currentPoint])

  useEffect(() => {
    iterationRef.current = iteration
  }, [iteration])

  const reset = () => {
    setIsRunning(false)
    isRunningRef.current = false
    setCurrentPoint({ x: 2, y: 2 })
    currentPointRef.current = { x: 2, y: 2 }
    setPath([{ x: 2, y: 2 }])
    setIteration(0)
    iterationRef.current = 0
    momentumRef.current = { x: 0, y: 0 }
    adamMRef.current = { x: 0, y: 0 }
    adamVRef.current = { x: 0, y: 0 }
    adagradGRef.current = { x: 0, y: 0 }
    rmspropGRef.current = { x: 0, y: 0 }
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current)
    }
  }

  const computeGradient = (x: number, y: number, func: keyof typeof objectiveFunctions) => {
    const h = 1e-5
    const fx = objectiveFunctions[func]

    const gradX = (fx(x + h, y) - fx(x - h, y)) / (2 * h)
    const gradY = (fx(x, y + h) - fx(x, y - h)) / (2 * h)

    return { x: gradX, y: gradY }
  }

  const optimizationStep = () => {
    const cp = currentPointRef.current
    const grad = computeGradient(cp.x, cp.y, selectedFunction)
    const newPoint = { ...cp }

    switch (selectedOptimizer) {
      case "gd":
        newPoint.x -= config.learningRate * grad.x
        newPoint.y -= config.learningRate * grad.y
        break

      case "sgd":
        const noiseX = (Math.random() - 0.5) * 0.1
        const noiseY = (Math.random() - 0.5) * 0.1
        newPoint.x -= config.learningRate * (grad.x + noiseX)
        newPoint.y -= config.learningRate * (grad.y + noiseY)
        break

      case "momentum":
        momentumRef.current.x = (config.momentum || 0.9) * momentumRef.current.x - config.learningRate * grad.x
        momentumRef.current.y = (config.momentum || 0.9) * momentumRef.current.y - config.learningRate * grad.y
        newPoint.x += momentumRef.current.x
        newPoint.y += momentumRef.current.y
        break

      case "adam":
        const beta1 = config.beta1 || 0.9
        const beta2 = config.beta2 || 0.999
        const eps = config.epsilon || 1e-8
        const t = iterationRef.current + 1

        adamMRef.current.x = beta1 * adamMRef.current.x + (1 - beta1) * grad.x
        adamMRef.current.y = beta1 * adamMRef.current.y + (1 - beta1) * grad.y

        adamVRef.current.x = beta2 * adamVRef.current.x + (1 - beta2) * grad.x * grad.x
        adamVRef.current.y = beta2 * adamVRef.current.y + (1 - beta2) * grad.y * grad.y

        const mHatX = adamMRef.current.x / (1 - Math.pow(beta1, t))
        const mHatY = adamMRef.current.y / (1 - Math.pow(beta1, t))
        const vHatX = adamVRef.current.x / (1 - Math.pow(beta2, t))
        const vHatY = adamVRef.current.y / (1 - Math.pow(beta2, t))

        newPoint.x -= (config.learningRate * mHatX) / (Math.sqrt(vHatX) + eps)
        newPoint.y -= (config.learningRate * mHatY) / (Math.sqrt(vHatY) + eps)
        break

      case "rmsprop":
        const betaDecay = config.momentum || 0.9
        const epsR = config.epsilon || 1e-8
        rmspropGRef.current.x = betaDecay * rmspropGRef.current.x + (1 - betaDecay) * grad.x * grad.x
        rmspropGRef.current.y = betaDecay * rmspropGRef.current.y + (1 - betaDecay) * grad.y * grad.y
        newPoint.x -= (config.learningRate * grad.x) / (Math.sqrt(rmspropGRef.current.x) + epsR)
        newPoint.y -= (config.learningRate * grad.y) / (Math.sqrt(rmspropGRef.current.y) + epsR)
        break

      case "adagrad":
        const epsA = config.epsilon || 1e-8
        adagradGRef.current.x += grad.x * grad.x
        adagradGRef.current.y += grad.y * grad.y
        newPoint.x -= (config.learningRate * grad.x) / (Math.sqrt(adagradGRef.current.x) + epsA)
        newPoint.y -= (config.learningRate * grad.y) / (Math.sqrt(adagradGRef.current.y) + epsA)
        break
    }

    currentPointRef.current = newPoint
    iterationRef.current = iterationRef.current + 1

    setCurrentPoint(newPoint)
    setPath((prev) => [...prev.slice(-100), newPoint])
    setIteration((prev) => prev + 1)
    setLoss(objectiveFunctions[selectedFunction](newPoint.x, newPoint.y))
  }

  const animate = () => {
    if (isRunningRef.current) {
      optimizationStep()
      animationRef.current = requestAnimationFrame(animate)
    }
  }

  const toggleAnimation = () => {
    if (isRunning) {
      setIsRunning(false)
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    } else {
      setIsRunning(true)
      animationRef.current = requestAnimationFrame(animate)
    }
  }

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isRunning) return
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    const x_px = (e.clientX - rect.left) * scaleX
    const y_px = (e.clientY - rect.top) * scaleY
    const scale = 50
    const offsetX = canvas.width / 2
    const offsetY = canvas.height / 2
    const dataX = (x_px - offsetX) / scale
    const dataY = (y_px - offsetY) / scale
    const newPoint = { x: dataX, y: dataY }
    setCurrentPoint(newPoint)
    currentPointRef.current = newPoint
    setPath([newPoint])
    setIteration(0)
    iterationRef.current = 0
    setLoss(objectiveFunctions[selectedFunction](dataX, dataY))
    momentumRef.current = { x: 0, y: 0 }
    adamMRef.current = { x: 0, y: 0 }
    adamVRef.current = { x: 0, y: 0 }
    adagradGRef.current = { x: 0, y: 0 }
    rmspropGRef.current = { x: 0, y: 0 }
  }

  const drawVisualization = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return
    canvas.width = 600
    canvas.height = 400
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    const scale = 50
    const offsetX = canvas.width / 2
    const offsetY = canvas.height / 2
    const resolution = 4
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
          ctx.fillStyle = resolvedTheme === "dark" ? "rgba(255, 255, 255, 0.2)" : "rgba(255, 255, 255, 0.4)"
          ctx.fillRect(i, j, resolution, resolution)
        }
      }
    }
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
    const currentX = currentPoint.x * scale + offsetX
    const currentY = currentPoint.y * scale + offsetY
    ctx.shadowBlur = 15
    ctx.shadowColor = "rgba(239, 68, 68, 0.5)"
    ctx.fillStyle = "#ef4444"
    ctx.beginPath()
    ctx.arc(currentX, currentY, 6, 0, 2 * Math.PI)
    ctx.fill()
    ctx.shadowBlur = 0
    ctx.strokeStyle = resolvedTheme === "dark" ? "rgba(255, 255, 255, 0.3)" : "rgba(0, 0, 0, 0.3)"
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(0, offsetY)
    ctx.lineTo(canvas.width, offsetY)
    ctx.moveTo(offsetX, 0)
    ctx.lineTo(offsetX, canvas.height)
    ctx.stroke()
  }

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
    # In practice, compute gradient on a single random sample
    grad_x, grad_y = compute_gradient(x, y)
    
    # Add noise to simulate stochastic nature
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
    <section id="playground" className="py-20 px-4 bg-background">
      <div className="container mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold mb-4 text-balance">Interactive Playground</h2>
          <p className="text-xl text-muted-foreground text-balance max-w-2xl mx-auto">
            Experiment with different optimizers and objective functions in real-time
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Card className="lg:col-span-1">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="w-5 h-5" />
                Configuration
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <label className="text-sm font-medium">Optimizer</label>
                <Select
                  value={selectedOptimizer}
                  onValueChange={(value) => setSelectedOptimizer(value as keyof typeof optimizerConfigs)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="gd">Gradient Descent</SelectItem>
                    <SelectItem value="sgd">SGD</SelectItem>
                    <SelectItem value="momentum">Momentum</SelectItem>
                    <SelectItem value="adam">Adam</SelectItem>
                    <SelectItem value="rmsprop">RMSprop</SelectItem>
                    <SelectItem value="adagrad">AdaGrad</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">Objective Function</label>
                <Select
                  value={selectedFunction}
                  onValueChange={(value) => setSelectedFunction(value as keyof typeof objectiveFunctions)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="quadratic">Quadratic</SelectItem>
                    <SelectItem value="rosenbrock">Rosenbrock</SelectItem>
                    <SelectItem value="himmelblau">Himmelblau</SelectItem>
                    <SelectItem value="beale">Beale</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <label className="text-sm font-medium">Learning Rate</label>
                  <Input 
                    type="number" 
                    className="w-20 h-8 text-xs"
                    step="0.0001"
                    value={config.learningRate}
                    onChange={(e) => setConfig(prev => ({ ...prev, learningRate: parseFloat(e.target.value) || 0 }))}
                  />
                </div>
                <Slider
                  value={[config.learningRate]}
                  onValueChange={([value]) => setConfig((prev) => ({ ...prev, learningRate: value }))}
                  min={0.001}
                  max={0.1}
                  step={0.001}
                />
              </div>

              {selectedOptimizer === "momentum" && (
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <label className="text-sm font-medium">Momentum</label>
                    <Input 
                      type="number" 
                      className="w-20 h-8 text-xs"
                      step="0.01"
                      value={config.momentum || 0.9}
                      onChange={(e) => setConfig(prev => ({ ...prev, momentum: parseFloat(e.target.value) || 0 }))}
                    />
                  </div>
                  <Slider
                    value={[config.momentum || 0.9]}
                    onValueChange={([value]) => setConfig((prev) => ({ ...prev, momentum: value }))}
                    min={0}
                    max={0.99}
                    step={0.01}
                  />
                </div>
              )}

              {selectedOptimizer === "adam" && (
                <>
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <label className="text-sm font-medium">Beta1</label>
                      <Input 
                        type="number" 
                        className="w-20 h-8 text-xs"
                        step="0.01"
                        value={config.beta1 || 0.9}
                        onChange={(e) => setConfig(prev => ({ ...prev, beta1: parseFloat(e.target.value) || 0 }))}
                      />
                    </div>
                    <Slider
                      value={[config.beta1 || 0.9]}
                      onValueChange={([value]) => setConfig((prev) => ({ ...prev, beta1: value }))}
                      min={0.1}
                      max={0.99}
                      step={0.01}
                    />
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <label className="text-sm font-medium">Beta2</label>
                      <Input 
                        type="number" 
                        className="w-20 h-8 text-xs"
                        step="0.001"
                        value={config.beta2 || 0.999}
                        onChange={(e) => setConfig(prev => ({ ...prev, beta2: parseFloat(e.target.value) || 0 }))}
                      />
                    </div>
                    <Slider
                      value={[config.beta2 || 0.999]}
                      onValueChange={([value]) => setConfig((prev) => ({ ...prev, beta2: value }))}
                      min={0.9}
                      max={0.999}
                      step={0.001}
                    />
                  </div>
                </>
              )}

              {selectedOptimizer === "rmsprop" && (
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <label className="text-sm font-medium">Beta (Decay Rate)</label>
                    <Input 
                      type="number" 
                      className="w-20 h-8 text-xs"
                      step="0.01"
                      value={config.momentum || 0.9}
                      onChange={(e) => setConfig(prev => ({ ...prev, momentum: parseFloat(e.target.value) || 0 }))}
                    />
                  </div>
                  <Slider
                    value={[config.momentum || 0.9]}
                    onValueChange={([value]) => setConfig((prev) => ({ ...prev, momentum: value }))}
                    min={0}
                    max={0.99}
                    step={0.01}
                  />
                </div>
              )}

              {selectedOptimizer === "adagrad" && (
                <div className="bg-muted/50 p-3 rounded-md">
                  <p className="text-xs text-muted-foreground">
                    AdaGrad has no extra hyperparameters besides learning rate. It accumulates square gradients automatically.
                  </p>
                </div>
              )}

              <div className="flex gap-2">
                <Button onClick={toggleAnimation} className="flex-1">
                  {isRunning ? <Pause className="w-4 h-4 mr-2" /> : <Play className="w-4 h-4 mr-2" />}
                  {isRunning ? "Pause" : "Start"}
                </Button>
                <Button onClick={reset} variant="outline">
                  <RotateCcw className="w-4 h-4" />
                </Button>
              </div>

              <div className="space-y-2 pt-4 border-t">
                <div className="flex justify-between text-sm">
                  <span>Iteration:</span>
                  <Badge variant="outline">{iteration}</Badge>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Loss:</span>
                  <Badge variant="outline">{loss.toFixed(6)}</Badge>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Position:</span>
                  <Badge variant="outline">
                    ({currentPoint.x.toFixed(3)}, {currentPoint.y.toFixed(3)})
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle>Optimization Visualization</CardTitle>
              <CardDescription>Watch the optimizer navigate the loss landscape in real-time</CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="visualization" className="w-full">
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="visualization">Visualization</TabsTrigger>
                  <TabsTrigger value="code">Code</TabsTrigger>
                </TabsList>

                <TabsContent value="visualization" className="mt-4">
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
                </TabsContent>

                <TabsContent value="code" className="mt-4">
                  <div className="bg-muted/20 rounded-lg p-4">
                    <pre className="text-sm overflow-x-auto">
                      <code>{generateCode()}</code>
                    </pre>
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  )
}
