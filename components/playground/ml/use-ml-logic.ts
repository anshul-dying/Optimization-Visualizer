import { useState, useEffect, useRef, useCallback } from "react"
import { MLConfig, DataPoint, TrainingMetrics } from "./types"

interface GridBounds {
  x_min: number; x_max: number;
  y_min: number; y_max: number;
  resolution: number;
}

export function useMLLogic(
  config: MLConfig,
  dataset: string,
  extremeValues: boolean,
  noiseLevel: number,
  useRealAlgorithms: boolean,
  SERVERURL: string
) {
  const [dataPoints, setDataPoints] = useState<DataPoint[]>([])
  const [isTraining, setIsTraining] = useState(false)
  const isTrainingRef = useRef(false)
  const [currentEpoch, setCurrentEpoch] = useState(0)
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetrics[]>([])
  const [model, setModel] = useState<any>(null)
  const [testAccuracy, setTestAccuracy] = useState(0)
  const [confusionMatrix, setConfusionMatrix] = useState<number[][]>([[0, 0], [0, 0]])
  const [apiStatus, setApiStatus] = useState<'checking' | 'available' | 'unavailable'>('checking')
  const [animationBoundary, setAnimationBoundary] = useState<number[] | null>(null)
  const [gridBounds, setGridBounds] = useState<GridBounds | null>(null)
  const [supportVectors, setSupportVectors] = useState<number[]>([])

  const generateDataset = useCallback((type: string, extreme: boolean, noise: number, numPoints = 200) => {
    const points: DataPoint[] = []
    const range = extreme ? 1000 : 10
    switch (type) {
      case "linear":
        for (let i = 0; i < numPoints; i++) {
          const x = (Math.random() - 0.5) * range
          const y = (Math.random() - 0.5) * range
          const label = x + y > 0 ? 1 : 0
          const n = (Math.random() - 0.5) * noise * range
          points.push({ x: x + n, y: y + n, label })
        }
        break
      case "nonlinear":
        for (let i = 0; i < numPoints; i++) {
          const x = (Math.random() - 0.5) * range
          const y = (Math.random() - 0.5) * range
          const label = x * x + y * y < (range / 4) * (range / 4) ? 1 : 0
          const n = (Math.random() - 0.5) * noise * range
          points.push({ x: x + n, y: y + n, label })
        }
        break
      case "circles":
        for (let i = 0; i < numPoints; i++) {
          const angle = Math.random() * 2 * Math.PI
          const r1 = Math.random() * (range / 6)
          const r2 = range / 4 + Math.random() * (range / 6)
          const r = Math.random() > 0.5 ? r1 : r2
          const x = r * Math.cos(angle)
          const y = r * Math.sin(angle)
          const label = r < range / 4 ? 1 : 0
          const n = (Math.random() - 0.5) * noise * range
          points.push({ x: x + n, y: y + n, label })
        }
        break
      case "moons":
        for (let i = 0; i < numPoints; i++) {
          const t = Math.random() * Math.PI
          const label = Math.random() > 0.5 ? 1 : 0
          let x, y
          if (label === 1) {
            x = Math.cos(t) * (range / 4)
            y = Math.sin(t) * (range / 4)
          } else {
            x = 1 - Math.cos(t) * (range / 4)
            y = -Math.sin(t) * (range / 4) - range / 8
          }
          const n = (Math.random() - 0.5) * noise * range
          points.push({ x: x + n, y: y + n, label })
        }
        break
      case "extreme_outliers":
        for (let i = 0; i < numPoints * 0.8; i++) {
          const x = (Math.random() - 0.5) * (range / 4)
          const y = (Math.random() - 0.5) * (range / 4)
          const label = x + y > 0 ? 1 : 0
          points.push({ x, y, label })
        }
        for (let i = 0; i < numPoints * 0.2; i++) {
          const x = (Math.random() - 0.5) * range * 10
          const y = (Math.random() - 0.5) * range * 10
          const label = Math.random() > 0.5 ? 1 : 0
          points.push({ x, y, label })
        }
        break
      default:
        for (let i = 0; i < numPoints; i++) {
          const x = (Math.random() - 0.5) * range
          const y = (Math.random() - 0.5) * range
          const label = x + y > 0 ? 1 : 0
          points.push({ x, y, label })
        }
    }
    return points
  }, [])

  useEffect(() => {
    const checkAPI = async () => {
      try {
        const response = await fetch(`${SERVERURL}`)
        setApiStatus(response.ok ? 'available' : 'unavailable')
      } catch { setApiStatus('unavailable') }
    }
    checkAPI()
  }, [SERVERURL])

  useEffect(() => {
    setDataPoints(generateDataset(dataset, extremeValues, noiseLevel))
  }, [dataset, extremeValues, noiseLevel, generateDataset])

  const computeClientBoundary = useCallback((points: DataPoint[], predictions: number[], resolution = 50) => {
    const canvas = { width: 500, height: 400 }
    const scale = extremeValues ? 0.05 : 20
    const offsetX = canvas.width / 2
    const offsetY = canvas.height / 2
    const gridSize = canvas.width / resolution
    const grid: number[][] = []
    for (let i = 0; i < resolution; i++) {
      grid[i] = []
      for (let j = 0; j < resolution; j++) {
        const canvasX = i * gridSize + gridSize / 2
        const canvasY = j * gridSize + gridSize / 2
        const dataX = (canvasX - offsetX) / scale
        const dataY = -(canvasY - offsetY) / scale
        let minDists: { dist: number; pred: number }[] = []
        for (let p = 0; p < points.length; p++) {
          const dx = dataX - points[p].x, dy = dataY - points[p].y
          minDists.push({ dist: dx * dx + dy * dy, pred: predictions[p] })
        }
        minDists.sort((a, b) => a.dist - b.dist)
        const k = Math.min(5, minDists.length)
        let sum = 0
        for (let n = 0; n < k; n++) sum += minDists[n].pred
        grid[i][j] = sum / k >= 0.5 ? 1 : 0
      }
    }
    return grid
  }, [extremeValues])

  const trainWithRealAlgorithm = useCallback(async () => {
    try {
      const X = dataPoints.map((p: DataPoint) => [p.x, p.y])
      const y = dataPoints.map((p: DataPoint) => p.label)
      const requestBody: any = {
        algorithm: config.algorithm, X, y,
        learning_rate: config.learningRate,
        epochs: config.epochs,
        batch_size: config.batchSize
      }
      if (config.algorithm === 'svm' || config.algorithm === 'kernel_svm') {
        requestBody.C = config.C; requestBody.kernel = config.kernelType; requestBody.gamma = config.gamma
      } else if (config.algorithm === 'neural_network') {
        requestBody.hidden_layers = config.hiddenLayers; requestBody.activation = config.activation
      } else if (['decision_tree', 'random_forest', 'gradient_boosting'].includes(config.algorithm)) {
        requestBody.max_depth = 10; requestBody.min_samples_split = 2
        if (config.algorithm !== 'decision_tree') {
          requestBody.n_estimators = config.algorithm === 'gradient_boosting' ? 50 : 10
        }
      }

      const response = await fetch(`${SERVERURL}/train`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      })
      if (!response.ok) throw new Error('Training failed')
      const result = await response.json()
      const history = result.metrics

      if (result.grid_bounds) setGridBounds(result.grid_bounds)

      // ========== SWEEP-IN ANIMATION — line comes from outside ==========
      // For algorithms with weights (linear, logistic, SVM-linear), sweep from a wrong angle
      const firstWithWeights = history.find((h: any) => h.weights)
      if (firstWithWeights && isTrainingRef.current) {
        const tw1 = firstWithWeights.weights[0]
        const tw2 = firstWithWeights.weights[1]
        const targetAngle = Math.atan2(tw2, tw1)
        const targetMag = Math.sqrt(tw1 * tw1 + tw2 * tw2) || 1
        const startAngle = targetAngle + Math.PI * (0.4 + Math.random() * 0.3) // start ~90-130° off
        const startBias = (Math.random() - 0.5) * 5

        const sweepFrames = 20
        for (let s = 0; s < sweepFrames; s++) {
          if (!isTrainingRef.current) break
          const t = (s + 1) / sweepFrames
          const eased = 1 - Math.pow(1 - t, 3) // ease-out cubic
          const angle = startAngle + (targetAngle - startAngle) * eased
          const mag = 0.1 + eased * (targetMag - 0.1)
          const w1 = Math.cos(angle) * mag
          const w2 = Math.sin(angle) * mag
          const bias = startBias * (1 - eased) + (firstWithWeights.bias || 0) * eased
          setModel({ weights: [w1, w2], bias, algorithm: config.algorithm, trained: true })
          setTrainingMetrics([{ epoch: 0, loss: 1, accuracy: 0.5 }])
          setCurrentEpoch(0)
          await new Promise(r => setTimeout(r, 50))
        }
      }

      // ========== EPOCH PLAYBACK — show each epoch step by step ==========
      const totalFrames = history.length
      const frameSkip = Math.max(1, Math.floor(totalFrames / 40))
      const frameDelay = Math.max(60, Math.min(150, 3000 / totalFrames))

      for (let i = 0; i < totalFrames; i += frameSkip) {
        if (!isTrainingRef.current) break
        const d = history[i]
        setTrainingMetrics([...history.slice(0, i + 1)])
        setCurrentEpoch(d.epoch)

        // Animate the decision line for linear/logistic/SVM-linear
        if (d.weights) {
          setModel({ weights: d.weights, bias: d.bias, algorithm: config.algorithm, trained: true })
        }

        // Animate boundary grid for non-linear algorithms
        if (d.boundary) {
          setAnimationBoundary(d.boundary)
          // Also set model as trained so canvas renders
          if (!d.weights) {
            setModel({ algorithm: config.algorithm, trained: true })
          }
        }

        // Highlight support vectors for SVM
        if (d.support_vectors) setSupportVectors(d.support_vectors)

        await new Promise(r => setTimeout(r, frameDelay))
      }

      // Ensure final state
      if (isTrainingRef.current) {
        const last = history[totalFrames - 1]
        setTrainingMetrics([...history])
        setCurrentEpoch(last.epoch)
        if (last.boundary) setAnimationBoundary(last.boundary)
        if (last.support_vectors) setSupportVectors(last.support_vectors)
        if (result.weights) {
          setModel({ weights: result.weights, bias: result.bias, algorithm: config.algorithm, trained: true })
        } else {
          setModel({ algorithm: config.algorithm, trained: true })
        }
      }

      setTestAccuracy(result.final_accuracy)
      if (isTrainingRef.current && result.predictions) {
        let tp = 0, fp = 0, fn = 0, tn = 0
        y.forEach((v: number, i: number) => {
          if (v === 1 && result.predictions[i] === 1) tp++
          else if (v === 0 && result.predictions[i] === 1) fp++
          else if (v === 1 && result.predictions[i] === 0) fn++
          else tn++
        })
        setConfusionMatrix([[tp, fp], [fn, tn]])
      }
      return result
    } catch (error) {
      console.error('Real algorithm training failed:', error)
      return false
    }
  }, [dataPoints, config, SERVERURL])

  const simulateTraining = useCallback(async () => {
    setIsTraining(true)
    isTrainingRef.current = true
    setCurrentEpoch(0)
    setTrainingMetrics([])
    setAnimationBoundary(null)
    setGridBounds(null)
    setSupportVectors([])
    setModel(null)

    if (useRealAlgorithms && apiStatus === 'available') {
      const result = await trainWithRealAlgorithm()
      if (result) {
        if (!['linear', 'logistic'].includes(config.algorithm) && !animationBoundary && result.predictions) {
          const boundary = computeClientBoundary(dataPoints, result.predictions)
          setModel((prev: any) => ({ ...prev, boundaryData: boundary }))
        }
        setIsTraining(false)
        isTrainingRef.current = false
        return
      }
    }

    // ========== SIMULATION MODE — Animated line/boundary at each epoch ==========
    const metrics: TrainingMetrics[] = []
    const totalEpochs = config.epochs

    // For linear/logistic/SVM-linear simulation: animate the line sweeping in
    // Start with a random wrong angle and converge to the correct separation
    const startAngle = Math.PI * (0.8 + Math.random() * 0.4) // start at ~150-210 degrees (wrong)
    const targetAngle = Math.PI / 4 // 45 degrees (correct for x+y=0 boundary)
    const startBias = (Math.random() - 0.5) * 5

    // For non-linear simulation: compute boundary at milestones
    const isLinearAlgo = ['linear', 'logistic', 'svm'].includes(config.algorithm)
    let milestoneIdx = 0
    const milestones = [0.2, 0.4, 0.6, 0.8, 1.0]

    for (let epoch = 0; epoch < totalEpochs; epoch++) {
      if (!isTrainingRef.current) break
      const progress = (epoch + 1) / totalEpochs
      let loss: number, accuracy: number

      switch (config.algorithm) {
        case "linear":
          loss = Math.exp(-epoch * config.learningRate * 0.1) + Math.random() * 0.1
          accuracy = Math.min(0.95, 0.5 + progress * 0.45)
          break
        case "logistic":
          loss = Math.log(1 + Math.exp(-epoch * config.learningRate * 0.05)) + Math.random() * 0.1
          accuracy = Math.min(0.92, 0.5 + progress * 0.42)
          break
        case "svm":
        case "kernel_svm":
          loss = Math.max(0, 1 - epoch * config.learningRate * 0.02) + (config.regularization || 0.01) * 0.1
          accuracy = Math.min(0.98, 0.6 + progress * 0.38)
          break
        case "neural_network":
          loss = Math.exp(-progress * 3) * (1 + Math.sin(progress * 10) * 0.1) + Math.random() * 0.05
          accuracy = Math.min(0.99, 0.5 + progress * 0.49 * (1 - progress * 0.1))
          break
        case "decision_tree":
          loss = Math.max(0.01, 0.5 * Math.exp(-epoch * 0.3)) + Math.random() * 0.02
          accuracy = Math.min(0.98, 0.7 + progress * 0.28)
          break
        case "random_forest":
          loss = Math.max(0.01, 0.6 * Math.exp(-epoch * 0.15)) + Math.random() * 0.03
          accuracy = Math.min(0.99, 0.65 + progress * 0.34)
          break
        case "gradient_boosting":
          loss = Math.max(0.01, 0.7 * Math.exp(-epoch * 0.08)) + Math.random() * 0.02
          accuracy = Math.min(0.97, 0.55 + progress * 0.42)
          break
        default:
          loss = Math.exp(-epoch * config.learningRate * 0.08) + Math.random() * 0.1
          accuracy = Math.min(0.9, 0.5 + progress * 0.4)
      }

      metrics.push({
        epoch: epoch + 1,
        loss: Math.max(0.001, loss),
        accuracy: Math.min(1, Math.max(0, accuracy + (Math.random() - 0.5) * 0.02)),
      })
      setCurrentEpoch(epoch + 1)
      setTrainingMetrics([...metrics])

      // ===== ANIMATE: Set model weights/boundary at EVERY epoch during simulation =====
      if (isLinearAlgo) {
        // Smoothly interpolate from wrong angle to correct angle
        const eased = 1 - Math.pow(1 - progress, 3) // ease-out cubic
        const currentAngle = startAngle + (targetAngle - startAngle) * eased
        const magnitude = 0.5 + eased * 1.5 // weight magnitude grows
        const w1 = Math.cos(currentAngle) * magnitude
        const w2 = Math.sin(currentAngle) * magnitude
        const bias = startBias * (1 - eased)
        setModel({
          weights: [w1, w2], bias,
          algorithm: config.algorithm, trained: true
        })
        // For SVM simulation: mark simulated support vectors
        if (config.algorithm === 'svm') {
          const svCount = Math.max(2, Math.floor(dataPoints.length * 0.05))
          const svIndices: number[] = []
          // Pick points closest to the decision boundary
          const distances = dataPoints.map((p, idx) => ({
            idx, dist: Math.abs(w1 * p.x + w2 * p.y + bias)
          }))
          distances.sort((a, b) => a.dist - b.dist)
          for (let s = 0; s < svCount; s++) svIndices.push(distances[s].idx)
          setSupportVectors(svIndices)
        }
      } else {
        // Non-linear: compute progressive boundary at milestones
        if (milestoneIdx < milestones.length && progress >= milestones[milestoneIdx]) {
          milestoneIdx++
          // Generate a progressive boundary where accuracy increases
          const partialPreds = dataPoints.map((p: DataPoint) => {
            const correct = p.label
            const noise = Math.random()
            return noise < accuracy ? correct : (correct === 1 ? 0 : 1)
          })
          const boundary = computeClientBoundary(dataPoints, partialPreds)
          setModel({ algorithm: config.algorithm, trained: true, boundaryData: boundary })
        } else if (epoch === 0) {
          // Set model as trained from first epoch so canvas renders
          setModel({ algorithm: config.algorithm, trained: true })
        }
      }

      await new Promise((resolve) => setTimeout(resolve, 50))
    }

    // Final confusion matrix
    const finalAccuracy = metrics[metrics.length - 1]?.accuracy || 0
    setTestAccuracy(finalAccuracy * (0.95 + Math.random() * 0.1))
    const finalPreds = dataPoints.map((p: DataPoint) => {
      return Math.random() > finalAccuracy ? (p.label === 1 ? 0 : 1) : p.label
    })
    const tp = finalPreds.filter((pred: number, i: number) => pred === 1 && dataPoints[i].label === 1).length
    const fp = finalPreds.filter((pred: number, i: number) => pred === 1 && dataPoints[i].label === 0).length
    const fn = finalPreds.filter((pred: number, i: number) => pred === 0 && dataPoints[i].label === 1).length
    const tn = finalPreds.filter((pred: number, i: number) => pred === 0 && dataPoints[i].label === 0).length
    setConfusionMatrix([[tp, fp], [fn, tn]])

    // If non-linear, ensure final boundary
    if (!isLinearAlgo) {
      const boundary = computeClientBoundary(dataPoints, finalPreds)
      setModel({ algorithm: config.algorithm, trained: true, boundaryData: boundary })
    }

    setIsTraining(false)
    isTrainingRef.current = false
  }, [config, useRealAlgorithms, apiStatus, trainWithRealAlgorithm, dataPoints, computeClientBoundary, animationBoundary])

  const stopTraining = () => { setIsTraining(false); isTrainingRef.current = false }

  const resetTraining = () => {
    setIsTraining(false); isTrainingRef.current = false
    setCurrentEpoch(0); setTrainingMetrics([]); setTestAccuracy(0)
    setConfusionMatrix([[0, 0], [0, 0]]); setModel(null)
    setAnimationBoundary(null); setGridBounds(null); setSupportVectors([])
  }

  return {
    dataPoints, isTraining, currentEpoch, trainingMetrics, model,
    testAccuracy, confusionMatrix, apiStatus, animationBoundary,
    gridBounds, supportVectors, simulateTraining, stopTraining, resetTraining,
    setTrainingMetrics, setCurrentEpoch, setModel, setTestAccuracy, setConfusionMatrix
  }
}
