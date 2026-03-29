"use client"

import { useState, useEffect, useRef } from "react"
import { useTheme } from "next-themes"

import { useMLLogic } from "./use-ml-logic"
import { MLConfigPanel } from "./ml-config-panel"
import { MLVisualizer } from "./ml-visualizer"
import { MLMetricsPanel } from "./ml-metrics-panel"
import { MLCharts } from "./ml-charts"
import { MLConfig } from "./types"

export function MLPlayground() {
  const { resolvedTheme } = useTheme()
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [config, setConfig] = useState<MLConfig>({
    algorithm: "linear",
    learningRate: 0.01,
    epochs: 100,
    batchSize: 32,
    regularization: 0.01,
    momentum: 0.9,
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
    kernelType: "rbf",
    C: 1.0,
    gamma: 0.1,
    hiddenLayers: [64, 32],
    activation: "relu",
  })

  const [dataset, setDataset] = useState<string>("linear")
  const [extremeValues, setExtremeValues] = useState(false)
  const [noiseLevel, setNoiseLevel] = useState(0.1)
  const [useRealAlgorithms, setUseRealAlgorithms] = useState(false)
  const [boundaryData, setBoundaryData] = useState<number[][] | null>(null)
  
  const SERVERURL = process.env.NEXT_PUBLIC_SERVER_URL || 'http://localhost:8000'

  const {
    dataPoints, isTraining, currentEpoch, trainingMetrics, model,
    testAccuracy, confusionMatrix, apiStatus, animationBoundary,
    gridBounds, supportVectors,
    simulateTraining, stopTraining, resetTraining
  } = useMLLogic(config, dataset, extremeValues, noiseLevel, useRealAlgorithms, SERVERURL)

  // Convert 1D animationBoundary into 2D grid
  useEffect(() => {
    if (animationBoundary && gridBounds) {
      const res = gridBounds.resolution
      const grid: number[][] = []
      for (let i = 0; i < res; i++) {
        grid[i] = animationBoundary.slice(i * res, (i + 1) * res)
      }
      setBoundaryData(grid)
    } else if (model?.boundaryData) {
      setBoundaryData(model.boundaryData)
    } else if (!model?.trained) {
      setBoundaryData(null)
    }
  }, [animationBoundary, gridBounds, model])

  // Fetch boundary when no snapshots (fallback)
  useEffect(() => {
    const fetchBoundary = async () => {
      if (!model?.trained || ['linear', 'logistic'].includes(config.algorithm)) return
      if (animationBoundary || model?.boundaryData || isTraining) return
      if (!useRealAlgorithms || apiStatus !== 'available') return

      const resolution = 50
      const scale = extremeValues ? 0.05 : 20
      const offsetX = 250, offsetY = 200
      const gridSize = 500 / resolution
      const points = []
      for (let i = 0; i < resolution; i++) {
        for (let j = 0; j < resolution; j++) {
          points.push([
            (i * gridSize + gridSize / 2 - offsetX) / scale,
            -(j * gridSize + gridSize / 2 - offsetY) / scale
          ])
        }
      }
      try {
        const res = await fetch(`${SERVERURL}/predict`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ X: points })
        })
        const data = await res.json()
        const grid: number[][] = []
        for (let i = 0; i < resolution; i++) grid[i] = data.predictions.slice(i * resolution, (i + 1) * resolution)
        setBoundaryData(grid)
      } catch (e) { console.error("Boundary fetch failed", e) }
    }
    fetchBoundary()
  }, [model, useRealAlgorithms, config.algorithm, extremeValues, SERVERURL, apiStatus, animationBoundary, isTraining])

  // ==================== CANVAS RENDERING ====================
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    canvas.width = 500
    canvas.height = 400
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const scale = extremeValues ? 0.05 : 20
    const offsetX = canvas.width / 2
    const offsetY = canvas.height / 2
    const isDark = resolvedTheme === "dark"

    // Helper: Draw a line defined by w1*x + w2*y + b = threshold
    const drawLine = (w1: number, w2: number, b: number, threshold: number, color: string, lineWidth: number, dash: number[]) => {
      if (Math.abs(w2) < 1e-10 && Math.abs(w1) < 1e-10) return
      ctx.save()
      ctx.strokeStyle = color
      ctx.lineWidth = lineWidth
      ctx.setLineDash(dash)
      ctx.beginPath()

      if (Math.abs(w2) > 1e-10) {
        const x1s = -offsetX / scale, x1e = (canvas.width - offsetX) / scale
        const y1 = (threshold - b - w1 * x1s) / w2
        const y2 = (threshold - b - w1 * x1e) / w2
        ctx.moveTo(x1s * scale + offsetX, offsetY - y1 * scale)
        ctx.lineTo(x1e * scale + offsetX, offsetY - y2 * scale)
      } else {
        const xVal = (threshold - b) / w1
        ctx.moveTo(xVal * scale + offsetX, 0)
        ctx.lineTo(xVal * scale + offsetX, canvas.height)
      }
      ctx.stroke()
      ctx.restore()
    }

    // Helper: Draw soft classification regions
    const drawSoftRegions = (w1: number, w2: number, b: number, threshold: number) => {
      ctx.save()
      ctx.globalAlpha = isDark ? 0.12 : 0.06
      const step = 5
      for (let x = 0; x < canvas.width; x += step) {
        for (let y = 0; y < canvas.height; y += step) {
          const dataX = (x - offsetX) / scale
          const dataY = -(y - offsetY) / scale
          const val = w1 * dataX + w2 * dataY + b
          ctx.fillStyle = val > threshold ? "#ef4444" : "#3b82f6"
          ctx.fillRect(x, y, step, step)
        }
      }
      ctx.restore()
    }

    // ===== Draw Decision Boundary =====
    const hasWeights = model?.weights?.length >= 2
    const hasTraining = trainingMetrics.length > 0

    if (hasTraining && model?.trained) {
      const algo = config.algorithm

      if (hasWeights && ['linear', 'logistic', 'svm'].includes(algo)) {
        const { weights, bias } = model
        const w1 = weights[0], w2 = weights[1]
        const threshold = algo === "linear" ? 0.5 : 0

        // Draw soft classification regions behind the line
        drawSoftRegions(w1, w2, bias, threshold)

        if (algo === 'svm') {
          // ===== SVM Classroom Style: H line + margin lines + M label =====
          const norm_w = Math.sqrt(w1 * w1 + w2 * w2)

          // Margin lines (dashed) at ±1/||w|| from hyperplane
          ctx.shadowColor = "transparent"
          drawLine(w1, w2, bias, 1, isDark ? "rgba(168,85,247,0.5)" : "rgba(124,58,237,0.4)", 2, [8, 5])
          drawLine(w1, w2, bias, -1, isDark ? "rgba(168,85,247,0.5)" : "rgba(124,58,237,0.4)", 2, [8, 5])

          // Main hyperplane H (solid, glowing)
          ctx.shadowColor = "#8b5cf6"
          ctx.shadowBlur = 10
          drawLine(w1, w2, bias, 0, "#8b5cf6", 3, [])
          ctx.shadowBlur = 0

          // Draw "H" label near the main line
          if (Math.abs(w2) > 1e-10) {
            const labelX = canvas.width - 40
            const labelDataX = (labelX - offsetX) / scale
            const labelDataY = (0 - bias - w1 * labelDataX) / w2
            const labelCanvasY = offsetY - labelDataY * scale
            if (labelCanvasY > 20 && labelCanvasY < canvas.height - 20) {
              ctx.save()
              ctx.font = "bold 14px sans-serif"
              ctx.fillStyle = "#8b5cf6"
              ctx.fillText("H", labelX + 4, labelCanvasY - 8)
              ctx.restore()
            }
          }

          // Draw "M" label showing margin width
          if (norm_w > 0.01 && Math.abs(w2) > 1e-10) {
            const midX = canvas.width * 0.15
            const midDataX = (midX - offsetX) / scale
            const midY_pos = (1 - bias - w1 * midDataX) / w2
            const midY_neg = (-1 - bias - w1 * midDataX) / w2
            const cy1 = offsetY - midY_pos * scale
            const cy2 = offsetY - midY_neg * scale
            if (cy1 > 0 && cy2 > 0 && cy1 < canvas.height && cy2 < canvas.height) {
              ctx.save()
              ctx.strokeStyle = isDark ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)"
              ctx.lineWidth = 1
              ctx.setLineDash([])
              // Arrow between margin lines
              ctx.beginPath(); ctx.moveTo(midX, cy1); ctx.lineTo(midX, cy2); ctx.stroke()
              // Arrow heads
              const arrowSize = 5
              ctx.beginPath(); ctx.moveTo(midX - arrowSize, cy1 + arrowSize); ctx.lineTo(midX, cy1); ctx.lineTo(midX + arrowSize, cy1 + arrowSize); ctx.stroke()
              ctx.beginPath(); ctx.moveTo(midX - arrowSize, cy2 - arrowSize); ctx.lineTo(midX, cy2); ctx.lineTo(midX + arrowSize, cy2 - arrowSize); ctx.stroke()
              ctx.font = "bold 13px sans-serif"
              ctx.fillStyle = isDark ? "rgba(255,255,255,0.7)" : "rgba(0,0,0,0.7)"
              ctx.fillText("M", midX + 6, (cy1 + cy2) / 2 + 4)
              ctx.restore()
            }
          }
        } else {
          // Linear / Logistic: decision line with glow
          ctx.shadowColor = "#8b5cf6"
          ctx.shadowBlur = 8
          drawLine(w1, w2, bias, threshold, "#8b5cf6", 3, [])
          ctx.shadowBlur = 0
        }

      } else if (boundaryData && boundaryData.length > 0 && boundaryData[0]) {
        // ===== Non-linear algorithms: colored grid regions =====
        const resolution = boundaryData.length
        const colRes = boundaryData[0].length

        if (gridBounds) {
          const { x_min, x_max, y_min, y_max } = gridBounds
          const cellW = (x_max - x_min) / resolution
          const cellH = (y_max - y_min) / colRes
          ctx.save()
          ctx.globalAlpha = isDark ? 0.35 : 0.2
          for (let i = 0; i < resolution; i++) {
            for (let j = 0; j < colRes; j++) {
              const dataX = x_min + i * cellW
              const dataY = y_max - j * cellH
              ctx.fillStyle = boundaryData[i][j] === 1 ? "#ef4444" : "#3b82f6"
              ctx.fillRect(dataX * scale + offsetX, offsetY - dataY * scale, cellW * scale + 1, cellH * scale + 1)
            }
          }
          ctx.restore()
        } else {
          const gx = canvas.width / resolution, gy = canvas.height / colRes
          ctx.save()
          ctx.globalAlpha = isDark ? 0.35 : 0.2
          for (let i = 0; i < resolution; i++) {
            for (let j = 0; j < colRes; j++) {
              ctx.fillStyle = boundaryData[i][j] === 1 ? "#ef4444" : "#3b82f6"
              ctx.fillRect(i * gx, j * gy, gx + 0.5, gy + 0.5)
            }
          }
          ctx.restore()
        }
      }
    }

    // ===== Draw Data Points =====
    dataPoints.forEach((point, idx) => {
      const x = point.x * scale + offsetX
      const y = offsetY - point.y * scale
      if (x < -5 || x > canvas.width + 5 || y < -5 || y > canvas.height + 5) return

      const isSV = supportVectors.includes(idx)

      if (isSV) {
        // Support Vector: open circle with glow (classroom style)
        ctx.save()
        ctx.shadowColor = point.label === 1 ? "#ef4444" : "#3b82f6"
        ctx.shadowBlur = 10
        ctx.strokeStyle = point.label === 1 ? "#ef4444" : "#3b82f6"
        ctx.lineWidth = 2.5
        ctx.beginPath()
        ctx.arc(x, y, 7, 0, 2 * Math.PI)
        ctx.stroke()
        // Small filled center dot
        ctx.fillStyle = point.label === 1 ? "#ef4444" : "#3b82f6"
        ctx.beginPath()
        ctx.arc(x, y, 2.5, 0, 2 * Math.PI)
        ctx.fill()
        ctx.restore()
      } else {
        // Regular data point
        ctx.fillStyle = point.label === 1 ? "#ef4444" : "#3b82f6"
        ctx.beginPath()
        ctx.arc(x, y, 3, 0, 2 * Math.PI)
        ctx.fill()
      }
    })

    // ===== Draw Axes =====
    ctx.strokeStyle = isDark ? "rgba(255,255,255,0.15)" : "rgba(0,0,0,0.15)"
    ctx.lineWidth = 1
    ctx.setLineDash([])
    ctx.beginPath()
    ctx.moveTo(0, offsetY); ctx.lineTo(canvas.width, offsetY)
    ctx.moveTo(offsetX, 0); ctx.lineTo(offsetX, canvas.height)
    ctx.stroke()

    // ===== Training Progress Overlay =====
    if (isTraining && trainingMetrics.length > 0) {
      ctx.save()
      // Epoch badge
      ctx.fillStyle = isDark ? "rgba(0,0,0,0.6)" : "rgba(255,255,255,0.7)"
      ctx.fillRect(8, 8, 120, 28)
      ctx.strokeStyle = "#8b5cf6"
      ctx.lineWidth = 1
      ctx.strokeRect(8, 8, 120, 28)
      ctx.fillStyle = isDark ? "#e2e8f0" : "#1e293b"
      ctx.font = "bold 13px 'Inter', sans-serif"
      ctx.fillText(`Epoch: ${currentEpoch}`, 16, 27)
      // Progress bar
      const progress = currentEpoch / config.epochs
      ctx.fillStyle = "#8b5cf680"
      ctx.fillRect(8, 38, 120 * progress, 3)
      ctx.restore()
    }

  }, [dataPoints, trainingMetrics, currentEpoch, config, model, resolvedTheme, boundaryData, extremeValues, isTraining, supportVectors, gridBounds])

  return (
    <div className="py-20 px-4 bg-background">
      <div className="container mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold mb-4 text-balance">ML Algorithm Playground</h1>
          <p className="text-xl text-muted-foreground text-balance max-w-3xl mx-auto">
            Test machine learning algorithms with extreme values, analyze performance, and visualize decision boundaries
          </p>
        </div>
        <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
          <MLConfigPanel
            config={config} setConfig={setConfig}
            dataset={dataset} setDataset={setDataset}
            extremeValues={extremeValues} setExtremeValues={setExtremeValues}
            useRealAlgorithms={useRealAlgorithms} setUseRealAlgorithms={setUseRealAlgorithms}
            noiseLevel={noiseLevel} setNoiseLevel={setNoiseLevel}
            apiStatus={apiStatus} isTraining={isTraining} currentEpoch={currentEpoch}
            simulateTraining={simulateTraining} stopTraining={stopTraining} resetTraining={resetTraining}
          />
          <MLVisualizer
            canvasRef={canvasRef} config={config}
            isTraining={isTraining} apiStatus={apiStatus} dataPoints={dataPoints}
          />
          <MLMetricsPanel
            trainingMetrics={trainingMetrics} testAccuracy={testAccuracy}
            confusionMatrix={confusionMatrix} algorithm={config.algorithm}
          />
        </div>
        <MLCharts trainingMetrics={trainingMetrics} />
      </div>
    </div>
  )
}
