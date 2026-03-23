"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
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
  const velocityRef = useRef<Point>({ x: 0, y: 0 })
  const momentumRef = useRef<Point>({ x: 0, y: 0 })
  const adamMRef = useRef<Point>({ x: 0, y: 0 })
  const adamVRef = useRef<Point>({ x: 0, y: 0 })

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

  // Keep refs for currentPoint and iteration in sync so the rAF loop
  // reads the latest values instead of stale closures.
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
    velocityRef.current = { x: 0, y: 0 }
    momentumRef.current = { x: 0, y: 0 }
    adamMRef.current = { x: 0, y: 0 }
    adamVRef.current = { x: 0, y: 0 }
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
    // Read latest values from refs to avoid stale closure issues when
    // this function is invoked from the rAF callback.
    const cp = currentPointRef.current
    const grad = computeGradient(cp.x, cp.y, selectedFunction)
    const newPoint = { ...cp }

    switch (selectedOptimizer) {
      case "gd":
        newPoint.x -= config.learningRate * grad.x
        newPoint.y -= config.learningRate * grad.y
        break

      case "sgd":
        // Add noise for SGD
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
    }

    // Update refs first so subsequent rAF calls see the latest values.
    currentPointRef.current = newPoint
    iterationRef.current = iterationRef.current + 1

    // Update React state so UI reflects the changes
    setCurrentPoint(newPoint)
    setPath((prev) => [...prev.slice(-100), newPoint]) // Keep last 100 points
    setIteration((prev) => prev + 1)
    setLoss(objectiveFunctions[selectedFunction](newPoint.x, newPoint.y))
  }

  const animate = () => {
    if (isRunningRef.current) {
      optimizationStep()
      console.log("Next Step..")
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
      console.log("Setting Running true");
      animationRef.current = requestAnimationFrame(animate)
    }
  }

  const drawVisualization = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    canvas.width = 600
    canvas.height = 400

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw contour lines
    const scale = 50
    const offsetX = canvas.width / 2
    const offsetY = canvas.height / 2

    // Create contour map
    for (let i = 0; i < canvas.width; i += 4) {
      for (let j = 0; j < canvas.height; j += 4) {
        const x = (i - offsetX) / scale
        const y = (j - offsetY) / scale
        const value = objectiveFunctions[selectedFunction](x, y)
        const intensity = Math.min(255, Math.max(0, 255 - Math.log(value + 1) * 30))
        ctx.fillStyle = `rgba(99, 102, 241, ${0.1 + ((255 - intensity) / 255) * 0.3})`
        ctx.fillRect(i, j, 4, 4)
      }
    }

    // Draw optimization path
    if (path.length > 1) {
      ctx.strokeStyle = "#f97316"
      ctx.lineWidth = 2
      ctx.beginPath()

      for (let i = 0; i < path.length; i++) {
        const x = path[i].x * scale + offsetX
        const y = path[i].y * scale + offsetY

        if (i === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      }
      ctx.stroke()

      // Draw path points
      path.forEach((point, index) => {
        const x = point.x * scale + offsetX
        const y = point.y * scale + offsetY
        const alpha = Math.max(0.2, index / path.length)

        ctx.fillStyle = `rgba(249, 115, 22, ${alpha})`
        ctx.beginPath()
        ctx.arc(x, y, 3, 0, 2 * Math.PI)
        ctx.fill()
      })
    }

    // Draw current point
    const currentX = currentPoint.x * scale + offsetX
    const currentY = currentPoint.y * scale + offsetY
    ctx.fillStyle = "#ef4444"
    ctx.beginPath()
    ctx.arc(currentX, currentY, 6, 0, 2 * Math.PI)
    ctx.fill()

    // Draw axes
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

      adam: `# Adam Optimizer
def adam_step(x, y, m, v, t, lr=${config.learningRate}, beta1=${config.beta1}, beta2=${config.beta2}):
    grad_x, grad_y = compute_gradient(x, y)
    
    m_x = beta1 * m_x + (1 - beta1) * grad_x
    m_y = beta1 * m_y + (1 - beta1) * grad_y
    
    v_x = beta2 * v_x + (1 - beta2) * grad_x**2
    v_y = beta2 * v_y + (1 - beta2) * grad_y**2
    
    m_hat_x = m_x / (1 - beta1**t)
    m_hat_y = m_y / (1 - beta1**t)
    v_hat_x = v_x / (1 - beta2**t)
    v_hat_y = v_y / (1 - beta2**t)
    
    x -= lr * m_hat_x / (sqrt(v_hat_x) + 1e-8)
    y -= lr * m_hat_y / (sqrt(v_hat_y) + 1e-8)
    
    return x, y, m, v`,
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
          {/* Controls */}
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
                <label className="text-sm font-medium">Learning Rate: {config.learningRate}</label>
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
                  <label className="text-sm font-medium">Momentum: {config.momentum}</label>
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
                    <label className="text-sm font-medium">Beta1: {config.beta1}</label>
                    <Slider
                      value={[config.beta1 || 0.9]}
                      onValueChange={([value]) => setConfig((prev) => ({ ...prev, beta1: value }))}
                      min={0.1}
                      max={0.99}
                      step={0.01}
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Beta2: {config.beta2}</label>
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

          {/* Visualization */}
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
                  <div className="bg-muted/20 rounded-lg p-4">
                    <canvas
                      ref={canvasRef}
                      className="w-full h-auto border rounded"
                      style={{ maxWidth: "100%", height: "400px" }}
                    />
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
