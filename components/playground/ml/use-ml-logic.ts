import { useState, useEffect, useRef, useCallback } from "react"
import { MLConfig, DataPoint, TrainingMetrics } from "./types"

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
      } catch (error) {
        setApiStatus('unavailable')
      }
    }
    checkAPI()
  }, [SERVERURL])

  useEffect(() => {
    setDataPoints(generateDataset(dataset, extremeValues, noiseLevel))
  }, [dataset, extremeValues, noiseLevel, generateDataset])

  const trainWithRealAlgorithm = useCallback(async () => {
    try {
      const X = dataPoints.map(p => [p.x, p.y])
      const y = dataPoints.map(p => p.label)
      const requestBody: any = {
        algorithm: config.algorithm,
        X, y,
        learning_rate: config.learningRate,
        epochs: config.epochs,
        batch_size: config.batchSize
      }

      if (config.algorithm === 'svm' || config.algorithm === 'kernel_svm') {
        requestBody.C = config.C
        requestBody.kernel = config.kernelType
        requestBody.gamma = config.gamma
      } else if (config.algorithm === 'neural_network') {
        requestBody.hidden_layers = config.hiddenLayers
        requestBody.activation = config.activation
      } else if (['decision_tree', 'random_forest', 'gradient_boosting'].includes(config.algorithm)) {
        requestBody.max_depth = 10
        requestBody.min_samples_split = 2
        if (config.algorithm === 'random_forest' || config.algorithm === 'gradient_boosting') {
          requestBody.n_estimators = config.algorithm === 'gradient_boosting' ? 50 : 10
        }
      }

      const response = await fetch(`${SERVERURL}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      })

      if (!response.ok) throw new Error('Training failed')
      const result = await response.json()
      const history = result.metrics

      for (let i = 0; i < history.length; i += Math.max(1, Math.floor(history.length / 50))) {
        if (!isTrainingRef.current) break
        const d = history[i]
        setTrainingMetrics(prev => [...history.slice(0, i + 1)])
        setCurrentEpoch(d.epoch)
        if (d.weights) {
          setModel({ weights: d.weights, bias: d.bias, algorithm: config.algorithm })
        }
        await new Promise(r => setTimeout(r, 20))
      }

      setTestAccuracy(result.final_accuracy)
      if (isTrainingRef.current && result.predictions) {
        let tp = 0, fp = 0, fn = 0, tn = 0
        y.forEach((v, i) => {
          if (v === 1 && result.predictions[i] === 1) tp++
          else if (v === 0 && result.predictions[i] === 1) fp++
          else if (v === 1 && result.predictions[i] === 0) fn++
          else if (v === 0 && result.predictions[i] === 0) tn++
        })
        setConfusionMatrix([[tp, fp], [fn, tn]])
      }
      return true
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

    if (useRealAlgorithms && apiStatus === 'available') {
      const success = await trainWithRealAlgorithm()
      if (success) {
        setIsTraining(false)
        isTrainingRef.current = false
        return
      }
    }

    const metrics: TrainingMetrics[] = []
    for (let epoch = 0; epoch < config.epochs; epoch++) {
      if (!isTrainingRef.current) break
      let loss, accuracy
      switch (config.algorithm) {
        case "linear":
          loss = Math.exp(-epoch * config.learningRate * 0.1) + Math.random() * 0.1
          accuracy = Math.min(0.95, 0.5 + (epoch / config.epochs) * 0.45)
          break
        case "logistic":
          loss = Math.log(1 + Math.exp(-epoch * config.learningRate * 0.05)) + Math.random() * 0.1
          accuracy = Math.min(0.92, 0.5 + (epoch / config.epochs) * 0.42)
          break
        case "svm":
          loss = Math.max(0, 1 - epoch * config.learningRate * 0.02) + config.regularization * 0.1
          accuracy = Math.min(0.98, 0.6 + (epoch / config.epochs) * 0.38)
          break
        case "neural_network":
          const progress = epoch / config.epochs
          loss = Math.exp(-progress * 3) * (1 + Math.sin(progress * 10) * 0.1) + Math.random() * 0.05
          accuracy = Math.min(0.99, 0.5 + progress * 0.49 * (1 - progress * 0.1))
          break
        default:
          loss = Math.exp(-epoch * config.learningRate * 0.08) + Math.random() * 0.1
          accuracy = Math.min(0.9, 0.5 + (epoch / config.epochs) * 0.4)
      }
      metrics.push({
        epoch: epoch + 1,
        loss: Math.max(0.001, loss),
        accuracy: Math.min(1, Math.max(0, accuracy + (Math.random() - 0.5) * 0.02)),
      })
      setCurrentEpoch(epoch + 1)
      setTrainingMetrics([...metrics])
      await new Promise((resolve) => setTimeout(resolve, 50))
    }

    const finalAccuracy = metrics[metrics.length - 1]?.accuracy || 0
    setTestAccuracy(finalAccuracy * (0.95 + Math.random() * 0.1))
    const tp = Math.floor(finalAccuracy * 50)
    const fn = Math.floor((1 - finalAccuracy) * 50)
    const fp = Math.floor((1 - finalAccuracy) * 30)
    const tn = Math.floor(finalAccuracy * 30)
    setConfusionMatrix([[tp, fp], [fn, tn]])
    setIsTraining(false)
    isTrainingRef.current = false
  }, [config, useRealAlgorithms, apiStatus, trainWithRealAlgorithm])

  const stopTraining = () => {
    setIsTraining(false)
    isTrainingRef.current = false
  }

  const resetTraining = () => {
    setIsTraining(false)
    isTrainingRef.current = false
    setCurrentEpoch(0)
    setTrainingMetrics([])
    setTestAccuracy(0)
    setConfusionMatrix([[0, 0], [0, 0]])
    setModel(null)
  }

  return {
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
    resetTraining,
    setTrainingMetrics,
    setCurrentEpoch,
    setModel,
    setTestAccuracy,
    setConfusionMatrix
  }
}
