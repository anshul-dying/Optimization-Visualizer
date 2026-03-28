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
    dataPoints,
    isTraining,
    currentEpoch,
    trainingMetrics,
    model,
    testAccuracy,
    confusionMatrix,
    apiStatus,
    simulateTraining,
    stopTraining,
    resetTraining
  } = useMLLogic(config, dataset, extremeValues, noiseLevel, useRealAlgorithms, SERVERURL)

  useEffect(() => {
    const updateBoundary = async () => {
      if (!model || !useRealAlgorithms || ['linear', 'logistic'].includes(config.algorithm)) {
        setBoundaryData(null)
        return
      }

      const resolution = 50
      const gridSize = 500 / resolution
      const scale = extremeValues ? 0.05 : 20
      const offsetX = 250
      const offsetY = 200

      const points = []
      for (let i = 0; i < resolution; i++) {
        for (let j = 0; j < resolution; j++) {
          const canvasX = i * gridSize
          const canvasY = j * gridSize
          const dataX = (canvasX - offsetX) / scale
          const dataY = -(canvasY - offsetY) / scale
          points.push([dataX, dataY])
        }
      }

      try {
        const res = await fetch(`${SERVERURL}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ X: points })
        })
        const data = await res.json()
        const grid: number[][] = []
        for (let i = 0; i < resolution; i++) {
          grid[i] = data.predictions.slice(i * resolution, (i + 1) * resolution)
        }
        setBoundaryData(grid)
      } catch (e) {
        console.error("Boundary update failed", e)
      }
    }
    updateBoundary()
  }, [model, useRealAlgorithms, config.algorithm, extremeValues, SERVERURL])

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

    if (trainingMetrics.length > 0 && model && model.weights && model.weights.length >= 2) {
      const { weights, bias } = model
      if (config.algorithm === "linear" || config.algorithm === "logistic") {
        const w1 = weights[0], w2 = weights[1]
        const threshold = config.algorithm === "linear" ? 0.5 : 0
        if (Math.abs(w2) > 1e-10) {
          ctx.strokeStyle = "#8b5cf6"
          ctx.lineWidth = 3
          ctx.beginPath()
          const x1_start = -offsetX / scale, x1_end = (canvas.width - offsetX) / scale
          const x2_start = (threshold - bias - w1 * x1_start) / w2
          const x2_end = (threshold - bias - w1 * x1_end) / w2
          ctx.moveTo(x1_start * scale + offsetX, offsetY - x2_start * scale)
          ctx.lineTo(x1_end * scale + offsetX, offsetY - x2_end * scale)
          ctx.stroke()
        }
      } else if (boundaryData) {
        const resolution = 50, gridSize = canvas.width / resolution
        ctx.globalAlpha = resolvedTheme === "dark" ? 0.35 : 0.2
        for (let i = 0; i < resolution; i++) {
          for (let j = 0; j < resolution; j++) {
            ctx.fillStyle = boundaryData[i][j] === 1 ? "#ef4444" : "#3b82f6"
            ctx.fillRect(i * gridSize, j * gridSize, gridSize + 0.5, gridSize + 0.5)
          }
        }
        ctx.globalAlpha = 1.0
      }
    }

    dataPoints.forEach((point) => {
      const x = point.x * scale + offsetX
      const y = offsetY - point.y * scale
      if (x >= 0 && x <= canvas.width && y >= 0 && y <= canvas.height) {
        ctx.fillStyle = point.label === 1 ? "#ef4444" : "#3b82f6"
        ctx.beginPath(); ctx.arc(x, y, 3, 0, 2 * Math.PI); ctx.fill()
      }
    })

    ctx.strokeStyle = resolvedTheme === "dark" ? "rgba(255, 255, 255, 0.2)" : "rgba(0, 0, 0, 0.2)"
    ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(0, offsetY); ctx.lineTo(canvas.width, offsetY)
    ctx.moveTo(offsetX, 0); ctx.lineTo(offsetX, canvas.height); ctx.stroke()
  }, [dataPoints, trainingMetrics, currentEpoch, config.algorithm, model, resolvedTheme, boundaryData, extremeValues])

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
