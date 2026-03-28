"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Play, Pause, RotateCcw, Settings, Brain, TrendingUp, Target, Zap } from "lucide-react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts"
import { useTheme } from "next-themes"

interface MLConfig {
  algorithm: string
  learningRate: number
  epochs: number
  batchSize: number
  regularization: number
  momentum?: number
  beta1?: number
  beta2?: number
  epsilon?: number
  kernelType?: string
  C?: number
  gamma?: number
  hiddenLayers?: number[]
  activation?: string
}

interface DataPoint {
  x: number
  y: number
  label: number
}

interface TrainingMetrics {
  epoch: number
  loss: number
  accuracy: number
  valLoss?: number
  valAccuracy?: number
}

const algorithms = {
  linear: "Linear Regression",
  logistic: "Logistic Regression",
  svm: "Support Vector Machine",
  kernel_svm: "Kernel SVM",
  neural_network: "Neural Network",
  decision_tree: "Decision Tree",
  random_forest: "Random Forest",
  gradient_boosting: "Gradient Boosting",
}

const kernelTypes = {
  linear: "Linear",
  polynomial: "Polynomial",
  rbf: "RBF (Gaussian)",
  sigmoid: "Sigmoid",
}

const activationFunctions = {
  relu: "ReLU",
  sigmoid: "Sigmoid",
  tanh: "Tanh",
  leaky_relu: "Leaky ReLU",
}

const datasetTypes = {
  linear: "Linear Separable",
  nonlinear: "Non-linear",
  circles: "Concentric Circles",
  moons: "Half Moons",
  blobs: "Gaussian Blobs",
  extreme_outliers: "Extreme Outliers",
  high_noise: "High Noise",
  imbalanced: "Imbalanced Classes",
}

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
  const [dataPoints, setDataPoints] = useState<DataPoint[]>([])
  const [isTraining, setIsTraining] = useState(false)
  const isTrainingRef = useRef(false)
  const [currentEpoch, setCurrentEpoch] = useState(0)
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetrics[]>([])
  const [model, setModel] = useState<any>(null)
  const [extremeValues, setExtremeValues] = useState(false)
  const [noiseLevel, setNoiseLevel] = useState(0.1)
  const [testAccuracy, setTestAccuracy] = useState(0)
  const [confusionMatrix, setConfusionMatrix] = useState<number[][]>([
    [0, 0],
    [0, 0],
  ])
  const [boundaryData, setBoundaryData] = useState<number[][] | null>(null)
  const [useRealAlgorithms, setUseRealAlgorithms] = useState(false)
  const [apiStatus, setApiStatus] = useState<'checking' | 'available' | 'unavailable'>('checking')
  const SERVERURL = process.env.NEXT_PUBLIC_SERVER_URL || 'http://localhost:8000'

  // Generate synthetic datasets
  const generateDataset = (type: string, numPoints = 200) => {
    const points: DataPoint[] = []
    const range = extremeValues ? 1000 : 10

    switch (type) {
      case "linear":
        for (let i = 0; i < numPoints; i++) {
          const x = (Math.random() - 0.5) * range
          const y = (Math.random() - 0.5) * range
          const label = x + y > 0 ? 1 : 0
          const noise = (Math.random() - 0.5) * noiseLevel * range
          points.push({ x: x + noise, y: y + noise, label })
        }
        break

      case "nonlinear":
        for (let i = 0; i < numPoints; i++) {
          const x = (Math.random() - 0.5) * range
          const y = (Math.random() - 0.5) * range
          const label = x * x + y * y < (range / 4) * (range / 4) ? 1 : 0
          const noise = (Math.random() - 0.5) * noiseLevel * range
          points.push({ x: x + noise, y: y + noise, label })
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
          const noise = (Math.random() - 0.5) * noiseLevel * range
          points.push({ x: x + noise, y: y + noise, label })
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
          const noise = (Math.random() - 0.5) * noiseLevel * range
          points.push({ x: x + noise, y: y + noise, label })
        }
        break

      case "extreme_outliers":
        // Generate normal points
        for (let i = 0; i < numPoints * 0.8; i++) {
          const x = (Math.random() - 0.5) * (range / 4)
          const y = (Math.random() - 0.5) * (range / 4)
          const label = x + y > 0 ? 1 : 0
          points.push({ x, y, label })
        }
        // Add extreme outliers
        for (let i = 0; i < numPoints * 0.2; i++) {
          const x = (Math.random() - 0.5) * range * 10
          const y = (Math.random() - 0.5) * range * 10
          const label = Math.random() > 0.5 ? 1 : 0
          points.push({ x, y, label })
        }
        break

      default:
        // Default to linear
        for (let i = 0; i < numPoints; i++) {
          const x = (Math.random() - 0.5) * range
          const y = (Math.random() - 0.5) * range
          const label = x + y > 0 ? 1 : 0
          points.push({ x, y, label })
        }
    }

    return points
  }

  // Check API status
  useEffect(() => {
    const checkAPI = async () => {
      try {
        const response = await fetch(`${SERVERURL}`)
        if (response.ok) {
          setApiStatus('available')
        } else {
          setApiStatus('unavailable')
        }
      } catch (error) {
        setApiStatus('unavailable')
      }
    }
    checkAPI()
  }, [])

  // Train using real API
  const trainWithRealAlgorithm = async () => {
    try {
      // Prepare training data
      const X = dataPoints.map(p => [p.x, p.y])
      const y = dataPoints.map(p => p.label)

      const requestBody: any = {
        algorithm: config.algorithm,
        X,
        y,
        learning_rate: config.learningRate,
        epochs: config.epochs,
        batch_size: config.batchSize
      }

      // Add algorithm-specific parameters
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

      if (!response.ok) {
        throw new Error('Training failed')
      }

      const result = await response.json()
      
      const history = result.metrics
      // Animation playback for real algorithm results
      for (let i = 0; i < history.length; i += Math.max(1, Math.floor(history.length / 50))) {
        if (!isTrainingRef.current) break
        const d = history[i]
        setTrainingMetrics(prev => [...history.slice(0, i + 1)])
        setCurrentEpoch(d.epoch)
        if (d.weights) {
           setModel({
             weights: d.weights,
             bias: d.bias,
             algorithm: config.algorithm
           })
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
  }

  // Simulate ML algorithm training
  const simulateTraining = async () => {
    setIsTraining(true)
    isTrainingRef.current = true
    setCurrentEpoch(0)
    setTrainingMetrics([])

    // Try to use real algorithm first if enabled
    if (useRealAlgorithms && apiStatus === 'available') {
      const success = await trainWithRealAlgorithm()
      if (success) {
        setIsTraining(false)
        isTrainingRef.current = false
        return
      }
      // Fall back to simulation if real training fails
    }

    const metrics: TrainingMetrics[] = []

    for (let epoch = 0; epoch < config.epochs; epoch++) {
      if (!isTrainingRef.current) break
      
      // Simulate training step with different convergence patterns
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
          // More complex convergence with potential overfitting
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

      // Simulate training delay
      await new Promise((resolve) => setTimeout(resolve, 50))
    }

    // Calculate final test accuracy
    const finalAccuracy = metrics[metrics.length - 1]?.accuracy || 0
    setTestAccuracy(finalAccuracy * (0.95 + Math.random() * 0.1))

    // Generate confusion matrix
    const tp = Math.floor(finalAccuracy * 50)
    const fn = Math.floor((1 - finalAccuracy) * 50)
    const fp = Math.floor((1 - finalAccuracy) * 30)
    const tn = Math.floor(finalAccuracy * 30)
    setConfusionMatrix([
      [tp, fp],
      [fn, tn],
    ])

    setIsTraining(false)
    isTrainingRef.current = false
  }

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
    setConfusionMatrix([
      [0, 0],
      [0, 0],
    ])
    setModel(null)
    setBoundaryData(null)
  }

  const drawDecisionBoundary = () => {
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

    // Draw decision boundary using actual trained model
    if (trainingMetrics.length > 0 && model && model.weights && model.weights.length >= 2) {
      const weights = model.weights
      const bias = model.bias

      // For linear and logistic regression, draw the actual decision boundary
      // Decision boundary is where w1*x1 + w2*x2 + b = threshold
      // Solving for x2: x2 = (threshold - b - w1*x1) / w2
      if ((config.algorithm === "linear" || config.algorithm === "logistic")) {
        const w1 = weights[0]
        const w2 = weights[1]
        const threshold = config.algorithm === "linear" ? 0.5 : 0
        
        if (Math.abs(w2) > 1e-10) {
          ctx.strokeStyle = "#8b5cf6"
          ctx.lineWidth = 3
          ctx.beginPath()
          
          // Calculate boundary points across the canvas
          const x1_start = -offsetX / scale
          const x1_end = (canvas.width - offsetX) / scale
          
          const x2_start = (threshold - bias - w1 * x1_start) / w2
          const x2_end = (threshold - bias - w1 * x1_end) / w2
          
          const canvasX1 = x1_start * scale + offsetX
          const canvasY1 = offsetY - x2_start * scale
          const canvasX2 = x1_end * scale + offsetX
          const canvasY2 = offsetY - x2_end * scale
          
          ctx.moveTo(canvasX1, canvasY1)
          ctx.lineTo(canvasX2, canvasY2)
          ctx.stroke()
          
          // Draw confidence regions for classification
          if (config.algorithm === "logistic") {
            ctx.globalAlpha = resolvedTheme === "dark" ? 0.2 : 0.15
            ctx.fillStyle = "#ef4444" // Red (Class 1)
            ctx.fillRect(0, 0, canvas.width, canvasY1 + (canvasY2 - canvasY1) * 0.5)
            ctx.fillStyle = "#3b82f6" // Blue (Class 0)
            ctx.fillRect(0, canvasY1 + (canvasY2 - canvasY1) * 0.5, canvas.width, canvas.height)
            ctx.globalAlpha = 1.0
          }
        }
      } else if (boundaryData && boundaryData.length > 0) {
        const resolution = 50
        const gridSize = canvas.width / resolution
        
        // Use a softer, more sophisticated opacity
        ctx.globalAlpha = resolvedTheme === "dark" ? 0.35 : 0.2
        
        for (let i = 0; i < resolution; i++) {
          for (let j = 0; j < resolution; j++) {
            const prediction = boundaryData[i][j]
            
            // Refined Palette: Softer Red and Blue
            ctx.fillStyle = prediction === 1 ? "#ef4444" : "#3b82f6"
            
            // Add a very subtle "mesh" effect or slight overlap for smoothness
            ctx.fillRect(i * gridSize, j * gridSize, gridSize + 0.5, gridSize + 0.5)
          }
        }
        ctx.globalAlpha = 1.0
      }
    }

    // Draw data points
    dataPoints.forEach((point) => {
      const x = point.x * scale + offsetX
      const y = offsetY - point.y * scale

      if (x >= 0 && x <= canvas.width && y >= 0 && y <= canvas.height) {
        ctx.fillStyle = point.label === 1 ? "#ef4444" : "#3b82f6"
        ctx.beginPath()
        ctx.arc(x, y, 3, 0, 2 * Math.PI)
        ctx.fill()
      }
    })

    // Draw axes
    ctx.strokeStyle = resolvedTheme === "dark" ? "rgba(255, 255, 255, 0.2)" : "rgba(0, 0, 0, 0.2)"
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(0, offsetY)
    ctx.lineTo(canvas.width, offsetY)
    ctx.moveTo(offsetX, 0)
    ctx.lineTo(offsetX, canvas.height)
    ctx.stroke()
  }

  useEffect(() => {
    const points = generateDataset(dataset)
    setDataPoints(points)
  }, [dataset, extremeValues, noiseLevel])

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
        const preds = data.predictions
        
        const grid: number[][] = []
        for (let i = 0; i < resolution; i++) {
          grid[i] = preds.slice(i * resolution, (i + 1) * resolution)
        }
        setBoundaryData(grid)
      } catch (e) {
        console.error("Boundary update failed", e)
      }
    }

    updateBoundary()
  }, [model, useRealAlgorithms, config.algorithm, extremeValues])

  useEffect(() => {
    drawDecisionBoundary()
  }, [dataPoints, trainingMetrics, currentEpoch, config.algorithm, model, resolvedTheme, boundaryData])

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
          {/* Configuration Panel */}
          <Card className="xl:col-span-1">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="w-5 h-5" />
                Configuration
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label>Algorithm</Label>
                <Select
                  value={config.algorithm}
                  onValueChange={(value) => setConfig((prev) => ({ ...prev, algorithm: value }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {Object.entries(algorithms).map(([key, name]) => (
                      <SelectItem key={key} value={key}>
                        {name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Dataset Type</Label>
                <Select value={dataset} onValueChange={setDataset}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {Object.entries(datasetTypes).map(([key, name]) => (
                      <SelectItem key={key} value={key}>
                        {name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="flex items-center justify-between">
                <Label>Extreme Values</Label>
                <Switch checked={extremeValues} onCheckedChange={setExtremeValues} />
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>Use Real Algorithms</Label>
                  <Switch 
                    checked={useRealAlgorithms} 
                    onCheckedChange={setUseRealAlgorithms}
                    disabled={apiStatus !== 'available'}
                  />
                </div>
                <p className="text-xs text-muted-foreground">
                  {apiStatus === 'checking' && '⏳ Checking API...'}
                  {apiStatus === 'available' && '✅ API available - All algorithms ready'}
                  {apiStatus === 'unavailable' && '❌ API unavailable (run: python core/api.py)'}
                </p>
              </div>

              <div className="space-y-2">
                <Label>Learning Rate: {config.learningRate}</Label>
                <Slider
                  value={[config.learningRate]}
                  onValueChange={([value]) => setConfig((prev) => ({ ...prev, learningRate: value }))}
                  min={0.0001}
                  max={1.0}
                  step={0.0001}
                />
              </div>

              <div className="space-y-2">
                <Label>Epochs: {config.epochs}</Label>
                <Slider
                  value={[config.epochs]}
                  onValueChange={([value]) => setConfig((prev) => ({ ...prev, epochs: value }))}
                  min={10}
                  max={1000}
                  step={10}
                />
              </div>

              <div className="space-y-2">
                <Label>Noise Level: {noiseLevel}</Label>
                <Slider
                  value={[noiseLevel]}
                  onValueChange={([value]) => setNoiseLevel(value)}
                  min={0}
                  max={1}
                  step={0.01}
                />
              </div>

              {config.algorithm === "svm" && (
                <>
                  <div className="space-y-2">
                    <Label>Kernel Type</Label>
                    <Select
                      value={config.kernelType}
                      onValueChange={(value) => setConfig((prev) => ({ ...prev, kernelType: value }))}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {Object.entries(kernelTypes).map(([key, name]) => (
                          <SelectItem key={key} value={key}>
                            {name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label>C Parameter: {config.C}</Label>
                    <Slider
                      value={[config.C || 1]}
                      onValueChange={([value]) => setConfig((prev) => ({ ...prev, C: value }))}
                      min={0.1}
                      max={10}
                      step={0.1}
                    />
                  </div>
                </>
              )}

              {config.algorithm === "neural_network" && (
                <>
                  <div className="space-y-2">
                    <Label>Activation Function</Label>
                    <Select
                      value={config.activation}
                      onValueChange={(value) => setConfig((prev) => ({ ...prev, activation: value }))}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {Object.entries(activationFunctions).map(([key, name]) => (
                          <SelectItem key={key} value={key}>
                            {name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label>Hidden Layers</Label>
                    <Input
                      value={config.hiddenLayers?.join(", ") || "64, 32"}
                      onChange={(e) => {
                        const layers = e.target.value
                          .split(",")
                          .map((s) => Number.parseInt(s.trim()))
                          .filter((n) => !isNaN(n))
                        setConfig((prev) => ({ ...prev, hiddenLayers: layers }))
                      }}
                      placeholder="64, 32, 16"
                    />
                  </div>
                </>
              )}

              <div className="flex gap-2">
                <Button onClick={isTraining ? stopTraining : simulateTraining} className="flex-1">
                  {isTraining ? <Pause className="w-4 h-4 mr-2" /> : <Play className="w-4 h-4 mr-2" />}
                  {isTraining ? "Stop" : "Train"}
                </Button>
                <Button onClick={resetTraining} variant="outline">
                  <RotateCcw className="w-4 h-4" />
                </Button>
              </div>

              {isTraining && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Progress:</span>
                    <span>
                      {currentEpoch}/{config.epochs}
                    </span>
                  </div>
                  <div className="w-full bg-muted rounded-full h-2">
                    <div
                      className="bg-primary h-2 rounded-full transition-all duration-300"
                      style={{ width: `${(currentEpoch / config.epochs) * 100}%` }}
                    />
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Visualization Panel */}
          <Card className="xl:col-span-2 overflow-hidden">
            <CardHeader className="border-b bg-muted/10">
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  <Brain className="w-5 h-5 text-primary" />
                  Execution & Analysis
                </CardTitle>
                <div className="flex gap-2">
                  <Badge variant="outline" className="text-[10px] uppercase">{config.algorithm}</Badge>
                  <Badge variant="outline" className="text-[10px] uppercase bg-green-500/10 text-green-500 border-green-500/20">
                    {apiStatus === 'available' ? 'Backend: NumPy' : 'Simulation'}
                  </Badge>
                </div>
              </div>
            </CardHeader>
            <CardContent className="p-0">
              <Tabs defaultValue="boundary" className="w-full">
                <TabsList className="w-full justify-start rounded-none border-b bg-transparent h-12 px-4">
                  <TabsTrigger value="boundary" className="data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none h-full">Decision Boundary</TabsTrigger>
                  <TabsTrigger value="daa" className="data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none h-full">DAA Breakdown</TabsTrigger>
                </TabsList>
                
                <TabsContent value="boundary" className="p-6 m-0">
                  <div className="bg-muted/20 rounded-xl p-4 border border-border/50 shadow-inner relative group">
                    <canvas
                      ref={canvasRef}
                      className={`w-full h-auto border rounded-lg transition-colors ${resolvedTheme === 'dark' ? 'bg-black/40' : 'bg-white/40'}`}
                      style={{ maxWidth: "100%", height: "400px" }}
                    />
                    {isTraining && (
                      <div className="absolute inset-0 flex items-center justify-center bg-background/20 backdrop-blur-[1px] pointer-events-none rounded-lg">
                        <div className="bg-background/80 px-4 py-2 rounded-full border shadow-sm flex items-center gap-2">
                          <div className="w-2 h-2 bg-primary rounded-full animate-ping" />
                          <span className="text-xs font-medium">Model Optimizing...</span>
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="mt-6 grid grid-cols-2 gap-4">
                    <div className="p-4 rounded-xl border bg-blue-500/5 border-blue-500/10 text-center">
                      <div className="text-3xl font-bold text-blue-500">
                        {dataPoints.filter((p) => p.label === 0).length}
                      </div>
                      <div className="text-xs uppercase tracking-wider font-bold text-muted-foreground mt-1">Class 0 (Blue)</div>
                    </div>
                    <div className="p-4 rounded-xl border bg-red-500/5 border-red-500/10 text-center">
                      <div className="text-3xl font-bold text-red-500">
                        {dataPoints.filter((p) => p.label === 1).length}
                      </div>
                      <div className="text-xs uppercase tracking-wider font-bold text-muted-foreground mt-1">Class 1 (Red)</div>
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="daa" className="p-6 m-0">
                  <div className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="p-4 rounded-xl border bg-card">
                        <h4 className="text-base font-bold flex items-center gap-2 mb-3 text-primary">
                          <Target className="w-5 h-5" /> Optimization Goal
                        </h4>
                        <p className="text-sm text-muted-foreground leading-relaxed">
                          {config.algorithm === 'linear' ? 'Minimize Mean Squared Error (MSE) using Gradient Descent.' : 
                           config.algorithm === 'logistic' ? 'Minimize Binary Cross-Entropy (Log-Loss) using Iterative Weight Updates.' :
                           config.algorithm === 'svm' ? 'Maximize the margin between classes while minimizing hinge loss (Quadratic Programming).' :
                           'Global Optimization of an objective function via iterative step-wise improvements.'}
                        </p>
                      </div>
                      <div className="p-4 rounded-xl border bg-card">
                        <h4 className="text-base font-bold flex items-center gap-2 mb-3 text-primary">
                          <Zap className="w-5 h-5" /> Algorithmic Strategy
                        </h4>
                        <ul className="text-sm space-y-2 text-muted-foreground">
                          <li className="flex items-start gap-2">
                            <div className="w-1.5 h-1.5 rounded-full bg-primary mt-2" />
                            <span><strong>Initialization:</strong> Weight vectors set to {config.algorithm === 'neural_network' ? 'He/Xavier distribution' : 'zero or random'}.</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <div className="w-1.5 h-1.5 rounded-full bg-primary mt-2" />
                            <span><strong>Iteration:</strong> {config.epochs} epochs of {config.algorithm === 'gradient_boosting' ? 'residual fitting' : 'gradient computation'}.</span>
                          </li>
                        </ul>
                      </div>
                    </div>

                    <div className="p-6 rounded-xl border bg-muted/30">
                      <h4 className="text-base font-bold mb-6 flex items-center gap-2">
                        <TrendingUp className="w-5 h-5 text-primary" /> Algorithmic Training Loop
                      </h4>
                      <div className="space-y-6 font-sans">
                        <div className="flex gap-6 p-4 bg-background/50 rounded-lg border border-border/40">
                          <div className="flex-none w-8 h-8 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-bold border border-primary/20">1</div>
                          <div className="space-y-2">
                            <p className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Gradient Computation</p>
                            <div className="text-xl">
                              <math display="block">
                                <mi>g</mi>
                                <mo>&#x2190;</mo>
                                <mo>&#x2207;</mo>
                                <mi>J</mi>
                                <mo>(</mo>
                                <msub>
                                  <mi>θ</mi>
                                  <mi>t</mi>
                                </msub>
                                <mo>)</mo>
                                <mspace width="1em" />
                                <mtext className="text-sm text-muted-foreground font-sans">for batch B</mtext>
                              </math>
                            </div>
                          </div>
                        </div>
                        
                        <div className="flex gap-6 p-4 bg-background/50 rounded-lg border border-border/40">
                          <div className="flex-none w-8 h-8 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-bold border border-primary/20">2</div>
                          <div className="space-y-2">
                            <p className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Weight Update Rule</p>
                            <div className="text-xl">
                              <math display="block">
                                <msub>
                                  <mi>θ</mi>
                                  <mrow>
                                    <mi>t</mi>
                                    <mo>+</mo>
                                    <mn>1</mn>
                                  </mrow>
                                </msub>
                                <mo>&#x2190;</mo>
                                <msub>
                                  <mi>θ</mi>
                                  <mi>t</mi>
                                </msub>
                                <mo>&#x2212;</mo>
                                <mi>η</mi>
                                <mo>&#x22C5;</mo>
                                <mi>g</mi>
                              </math>
                            </div>
                          </div>
                        </div>

                        <div className="flex gap-6 p-4 bg-background/50 rounded-lg border border-border/40">
                          <div className="flex-none w-8 h-8 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-bold border border-primary/20">3</div>
                          <div className="space-y-2">
                            <p className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Convergence Check</p>
                            <div className="text-xl">
                              <math display="block">
                                <mo>||</mo>
                                <mo>&#x2207;</mo>
                                <mi>J</mi>
                                <mo>(</mo>
                                <mi>θ</mi>
                                <mo>)</mo>
                                <mo>||</mo>
                                <mo>&lt;</mo>
                                <mi>ε</mi>
                                <mspace width="1em" />
                                <mtext className="text-sm text-muted-foreground font-sans">return θ</mtext>
                              </math>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>

          {/* Metrics Panel */}
          <Card className="xl:col-span-1">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5" />
                Performance Metrics
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {trainingMetrics.length > 0 && (
                <>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Current Loss:</span>
                      <Badge variant="outline">{trainingMetrics[trainingMetrics.length - 1]?.loss.toFixed(4)}</Badge>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Current Accuracy:</span>
                      <Badge variant="outline">
                        {(trainingMetrics[trainingMetrics.length - 1]?.accuracy * 100).toFixed(1)}%
                      </Badge>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Test Accuracy:</span>
                      <Badge variant="outline">{(testAccuracy * 100).toFixed(1)}%</Badge>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label>Confusion Matrix</Label>
                    <div className="grid grid-cols-2 gap-1 text-xs">
                      <div className="bg-green-500/20 p-2 text-center rounded">TP: {confusionMatrix[0][0]}</div>
                      <div className="bg-red-500/20 p-2 text-center rounded">FP: {confusionMatrix[0][1]}</div>
                      <div className="bg-red-500/20 p-2 text-center rounded">FN: {confusionMatrix[1][0]}</div>
                      <div className="bg-green-500/20 p-2 text-center rounded">TN: {confusionMatrix[1][1]}</div>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="p-3 bg-primary/5 border border-primary/10 rounded-lg space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-[10px] uppercase font-bold text-muted-foreground tracking-wider">Time Complexity</span>
                        <Badge variant="secondary" className="font-mono text-[10px]">
                          {config.algorithm === 'linear' || config.algorithm === 'logistic' ? 'O(n·epochs)' : 
                           config.algorithm === 'svm' ? 'O(n²·epochs)' : 
                           config.algorithm === 'neural_network' ? 'O(n·L·epochs)' : 'O(n·log n)'}
                        </Badge>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-[10px] uppercase font-bold text-muted-foreground tracking-wider">Space Complexity</span>
                        <Badge variant="outline" className="font-mono text-[10px]">
                          {config.algorithm === 'neural_network' ? 'O(L·W)' : 'O(n·d)'}
                        </Badge>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <Label className="text-[10px] uppercase font-bold text-muted-foreground">DAA Characteristics</Label>
                      <div className="text-xs space-y-2">
                        <div className="flex justify-between p-2 bg-muted/30 rounded border border-border/50">
                          <span>Paradigm:</span>
                          <span className="font-semibold text-primary">
                            {['decision_tree', 'random_forest'].includes(config.algorithm) ? 'Greedy / Recursive' : 'Iterative Optimization'}
                          </span>
                        </div>
                        <div className="flex justify-between p-2 bg-muted/30 rounded border border-border/50">
                          <span>Convergence:</span>
                          <span className={trainingMetrics.length > 50 ? "text-green-500 font-semibold" : "text-yellow-500 font-semibold"}>
                            {config.algorithm === 'svm' ? 'Quadratic' : 'Linear / Sub-linear'}
                          </span>
                        </div>
                        <div className="flex justify-between p-2 bg-muted/30 rounded border border-border/50">
                          <span>Stability:</span>
                          <span className="text-green-500 font-semibold">Stable (Asymptotically)</span>
                        </div>
                      </div>
                    </div>

                    <div className="pt-2 border-t border-border/50">
                      <p className="text-[10px] text-muted-foreground italic leading-tight">
                        * Analysis based on the scratch implementation using NumPy-like operations.
                      </p>
                    </div>
                  </div>
                </>
              )}

              {trainingMetrics.length === 0 && (
                <div className="text-center text-muted-foreground py-8">
                  <Target className="w-12 h-12 mx-auto mb-2 opacity-50" />
                  <p>Start training to see metrics</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Training Progress Charts */}
        {trainingMetrics.length > 0 && (
          <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Loss Curve</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={trainingMetrics}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="epoch" stroke="var(--muted-foreground)" />
                      <YAxis stroke="var(--muted-foreground)" />
                      <Tooltip 
                        contentStyle={{
                          backgroundColor: "var(--card)",
                          border: "1px solid var(--border)",
                          borderRadius: "8px",
                        }}
                      />
                      <Legend />
                      <Line type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Accuracy Curve</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={trainingMetrics}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="epoch" stroke="var(--muted-foreground)" />
                      <YAxis domain={[0, 1]} stroke="var(--muted-foreground)" />
                      <Tooltip 
                        contentStyle={{
                          backgroundColor: "var(--card)",
                          border: "1px solid var(--border)",
                          borderRadius: "8px",
                        }}
                      />
                      <Legend />
                      <Line type="monotone" dataKey="accuracy" stroke="#22c55e" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  )
}
