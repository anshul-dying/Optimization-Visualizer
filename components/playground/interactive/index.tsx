"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useTheme } from "next-themes"

import { useOptimization } from "./use-optimization"
import { ConfigPanel } from "./config-panel"
import { CodeView } from "./code-view"
import { CanvasView } from "./canvas-view"
import { optimizerConfigs, objectiveFunctions } from "./constants"
import { OptimizerConfig } from "./types"

export function InteractivePlayground() {
  const { resolvedTheme } = useTheme()
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [selectedOptimizer, setSelectedOptimizer] = useState<string>("adam")
  const [selectedFunction, setSelectedFunction] = useState<keyof typeof objectiveFunctions>("quadratic")
  const [config, setConfig] = useState<OptimizerConfig>(optimizerConfigs.adam)

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
    executionState,
    toggleAnimation,
    reset,
    handleCanvasClick
  } = useOptimization(selectedOptimizer, selectedFunction, config)

  useEffect(() => {
    setConfig(optimizerConfigs[selectedOptimizer])
    reset()
  }, [selectedOptimizer, reset])

  useEffect(() => {
    drawVisualization()
  }, [currentPoint, path, selectedFunction, resolvedTheme])

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

    // Draw contour lines
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
    <section id="playground" className="py-20 px-4 bg-background">
      <div className="container mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold mb-4 text-balance">Interactive Playground</h2>
          <p className="text-xl text-muted-foreground text-balance max-w-2xl mx-auto">
            Experiment with different optimizers and objective functions in real-time
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
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
                  <CanvasView canvasRef={canvasRef} handleCanvasClick={handleCanvasClick} isRunning={isRunning} />
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
        </div>
      </div>
    </section>
  )
}
