"use client"

import { useEffect, useRef, useState } from "react"
import { useTheme } from "next-themes"

interface OptimizerVisualizationProps {
  type: string
}

export function OptimizerVisualization({ type }: OptimizerVisualizationProps) {
  const { resolvedTheme } = useTheme()
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isAnimating, setIsAnimating] = useState(false)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Set canvas size
    canvas.width = 280
    canvas.height = 160

    let animationId: number
    let step = 0

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Draw background grid
      ctx.strokeStyle = resolvedTheme === "dark" ? "rgba(255, 255, 255, 0.1)" : "rgba(0, 0, 0, 0.1)"
      ctx.lineWidth = 1
      for (let i = 0; i < canvas.width; i += 20) {
        ctx.beginPath()
        ctx.moveTo(i, 0)
        ctx.lineTo(i, canvas.height)
        ctx.stroke()
      }
      for (let i = 0; i < canvas.height; i += 20) {
        ctx.beginPath()
        ctx.moveTo(0, i)
        ctx.lineTo(canvas.width, i)
        ctx.stroke()
      }

      // Draw optimization path based on type
      drawOptimizerPath(ctx, type, step)

      step += 0.02
      if (step > 2 * Math.PI) step = 0

      animationId = requestAnimationFrame(animate)
    }

    if (isAnimating) {
      animate()
    } else {
      // Draw static version
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      drawOptimizerPath(ctx, type, 0)
    }

    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId)
      }
    }
  }, [type, isAnimating, resolvedTheme])

  const drawOptimizerPath = (ctx: CanvasRenderingContext2D, optimizerType: string, step: number) => {
    const centerX = ctx.canvas.width / 2
    const centerY = ctx.canvas.height / 2

    switch (optimizerType) {
      case "gd":
        drawGradientDescent(ctx, centerX, centerY, step)
        break
      case "sgd":
        drawStochasticGD(ctx, centerX, centerY, step)
        break
      case "mini-batch":
        drawMiniBatchSGD(ctx, centerX, centerY, step)
        break
      case "adam":
        drawAdam(ctx, centerX, centerY, step)
        break
      case "analytical":
        drawAnalytical(ctx, centerX, centerY, step)
        break
      case "qp":
        drawQuadraticProgramming(ctx, centerX, centerY, step)
        break
    }
  }

  const drawGradientDescent = (ctx: CanvasRenderingContext2D, cx: number, cy: number, step: number) => {
    // Smooth descent path
    ctx.strokeStyle = "#3b82f6"
    ctx.lineWidth = 2
    ctx.beginPath()
    for (let i = 0; i < 100; i++) {
      const x = cx - 80 + i * 1.6
      const y = cy + 40 - i * 0.8 + Math.sin(step + i * 0.1) * 2
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.stroke()

    // Current position
    const currentX = cx - 80 + ((step * 20) % 160)
    const currentY = cy + 40 - ((step * 20) % 160) * 0.5
    ctx.fillStyle = "#3b82f6"
    ctx.beginPath()
    ctx.arc(currentX, currentY, 4, 0, 2 * Math.PI)
    ctx.fill()
  }

  const drawStochasticGD = (ctx: CanvasRenderingContext2D, cx: number, cy: number, step: number) => {
    // Noisy path
    ctx.strokeStyle = "#10b981"
    ctx.lineWidth = 2
    ctx.beginPath()
    for (let i = 0; i < 50; i++) {
      const x = cx - 60 + i * 2.4 + Math.sin(step + i * 0.3) * 8
      const y = cy + 30 - i * 1.2 + Math.cos(step + i * 0.2) * 6
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.stroke()

    // Current position with noise
    const currentX = cx + Math.sin(step * 3) * 20
    const currentY = cy + Math.cos(step * 2) * 15
    ctx.fillStyle = "#10b981"
    ctx.beginPath()
    ctx.arc(currentX, currentY, 3, 0, 2 * Math.PI)
    ctx.fill()
  }

  const drawMiniBatchSGD = (ctx: CanvasRenderingContext2D, cx: number, cy: number, step: number) => {
    // Balanced path
    ctx.strokeStyle = "#8b5cf6"
    ctx.lineWidth = 2
    ctx.beginPath()
    for (let i = 0; i < 60; i++) {
      const x = cx - 70 + i * 2.3
      const y = cy + 35 - i * 1.1 + Math.sin(step + i * 0.2) * 3
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.stroke()

    // Batch points
    for (let i = 0; i < 3; i++) {
      const batchX = cx + Math.sin(step + i * 2) * 15
      const batchY = cy + Math.cos(step + i * 2) * 10
      ctx.fillStyle = "#8b5cf6"
      ctx.beginPath()
      ctx.arc(batchX, batchY, 2, 0, 2 * Math.PI)
      ctx.fill()
    }
  }

  const drawAdam = (ctx: CanvasRenderingContext2D, cx: number, cy: number, step: number) => {
    // Adaptive path with momentum
    ctx.strokeStyle = "#f97316"
    ctx.lineWidth = 2
    ctx.beginPath()
    for (let i = 0; i < 80; i++) {
      const momentum = Math.exp(-i * 0.05)
      const x = cx - 90 + i * 2.25
      const y = cy + 45 - i * 1.1 * momentum + Math.sin(step + i * 0.15) * momentum * 4
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.stroke()

    // Adaptive learning rate visualization
    const adaptiveX = cx + Math.sin(step * 2) * 25
    const adaptiveY = cy + Math.cos(step * 1.5) * 20
    ctx.fillStyle = "#f97316"
    ctx.beginPath()
    ctx.arc(adaptiveX, adaptiveY, 4 + Math.sin(step * 4) * 2, 0, 2 * Math.PI)
    ctx.fill()
  }

  const drawAnalytical = (ctx: CanvasRenderingContext2D, cx: number, cy: number, step: number) => {
    // Direct line to solution
    ctx.strokeStyle = "#ec4899"
    ctx.lineWidth = 3
    ctx.setLineDash([5, 5])
    ctx.beginPath()
    ctx.moveTo(cx - 80, cy + 40)
    ctx.lineTo(cx + 80, cy - 40)
    ctx.stroke()
    ctx.setLineDash([])

    // Solution point
    ctx.fillStyle = "#ec4899"
    ctx.beginPath()
    ctx.arc(cx + 80, cy - 40, 6, 0, 2 * Math.PI)
    ctx.fill()

    // Matrix visualization
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        const x = cx - 40 + i * 20
        const y = cy - 20 + j * 15
        ctx.fillStyle = `rgba(236, 72, 153, ${0.3 + Math.sin(step + i + j) * 0.2})`
        ctx.fillRect(x, y, 15, 10)
      }
    }
  }

  const drawQuadraticProgramming = (ctx: CanvasRenderingContext2D, cx: number, cy: number, step: number) => {
    // Constrained optimization path
    ctx.strokeStyle = "#6366f1"
    ctx.lineWidth = 2

    // Draw constraints
    ctx.setLineDash([3, 3])
    ctx.strokeStyle = "rgba(99, 102, 241, 0.5)"
    ctx.beginPath()
    ctx.moveTo(cx - 60, cy + 30)
    ctx.lineTo(cx + 60, cy - 30)
    ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(cx - 30, cy + 50)
    ctx.lineTo(cx + 30, cy - 50)
    ctx.stroke()
    ctx.setLineDash([])

    // Feasible region path
    ctx.strokeStyle = "#6366f1"
    ctx.beginPath()
    for (let i = 0; i < 40; i++) {
      const t = i / 40
      const x = cx - 50 + t * 100
      const y = cy + 25 - t * 50 + Math.sin(step + t * Math.PI) * 5
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.stroke()

    // Optimal point
    const optX = cx + Math.sin(step) * 10
    const optY = cy + Math.cos(step) * 8
    ctx.fillStyle = "#6366f1"
    ctx.beginPath()
    ctx.arc(optX, optY, 5, 0, 2 * Math.PI)
    ctx.fill()
  }

  return (
    <div
      className="relative bg-card/50 rounded-lg p-2 cursor-pointer transition-all hover:bg-card/70"
      onClick={() => setIsAnimating(!isAnimating)}
    >
      <canvas ref={canvasRef} className="w-full h-auto rounded" style={{ maxWidth: "100%", height: "auto" }} />
      <div className="absolute top-2 right-2 text-xs text-muted-foreground">
        {isAnimating ? "Click to pause" : "Click to animate"}
      </div>
    </div>
  )
}
