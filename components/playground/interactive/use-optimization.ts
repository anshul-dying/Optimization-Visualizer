import { useState, useEffect, useRef, useCallback } from "react"
import { Point, OptimizerConfig, ExecutionState } from "./types"
import { optimizerConfigs, objectiveFunctions } from "./constants"

export function useOptimization(
  selectedOptimizer: string,
  selectedFunction: keyof typeof objectiveFunctions,
  config: OptimizerConfig
) {
  const [isRunning, setIsRunning] = useState(false)
  const [currentPoint, setCurrentPoint] = useState<Point>({ x: 2, y: 2 })
  const [path, setPath] = useState<Point[]>([{ x: 2, y: 2 }])
  const [iteration, setIteration] = useState(0)
  const [loss, setLoss] = useState(0)
  const [isDebugMode, setIsDebugMode] = useState(false)
  const [stepDelay, setStepDelay] = useState(200)
  const [zoom, setZoom] = useState(1)
  const [panOffset, setPanOffset] = useState<Point>({ x: 0, y: 0 })
  const [executionState, setExecutionState] = useState<ExecutionState>({
    activeLine: -1,
    variables: {},
  })

  // Refs for animation loop
  const isRunningRef = useRef(isRunning)
  const isDebugModeRef = useRef(isDebugMode)
  const activeLineRef = useRef(-1)
  const executionVariablesRef = useRef<Record<string, any>>({})
  const animationRef = useRef<number | null>(null)
  const currentPointRef = useRef<Point>(currentPoint)
  const iterationRef = useRef<number>(iteration)
  
  // Optimizer specific refs
  const momentumRef = useRef<Point>({ x: 0, y: 0 })
  const adamMRef = useRef<Point>({ x: 0, y: 0 })
  const adamVRef = useRef<Point>({ x: 0, y: 0 })
  const adagradGRef = useRef<Point>({ x: 0, y: 0 })
  const rmspropGRef = useRef<Point>({ x: 0, y: 0 })

  useEffect(() => {
    isRunningRef.current = isRunning
  }, [isRunning])

  useEffect(() => {
    isDebugModeRef.current = isDebugMode
    if (!isDebugMode) {
      setExecutionState({ activeLine: -1, variables: {} })
      activeLineRef.current = -1
      executionVariablesRef.current = {}
    }
  }, [isDebugMode])

  useEffect(() => {
    currentPointRef.current = currentPoint
  }, [currentPoint])

  useEffect(() => {
    iterationRef.current = iteration
  }, [iteration])

  const computeGradient = useCallback((x: number, y: number, func: keyof typeof objectiveFunctions) => {
    const h = 1e-5
    const fx = objectiveFunctions[func]
    const gradX = (fx(x + h, y) - fx(x - h, y)) / (2 * h)
    const gradY = (fx(x, y + h) - fx(x, y - h)) / (2 * h)
    return { x: gradX, y: gradY }
  }, [])

  const reset = useCallback(() => {
    setIsRunning(false)
    isRunningRef.current = false
    setCurrentPoint({ x: 2, y: 2 })
    currentPointRef.current = { x: 2, y: 2 }
    setPath([{ x: 2, y: 2 }])
    setIteration(0)
    iterationRef.current = 0
    setZoom(1)
    setPanOffset({ x: 0, y: 0 })
    setExecutionState({ activeLine: -1, variables: {} })
    activeLineRef.current = -1
    executionVariablesRef.current = {}
    momentumRef.current = { x: 0, y: 0 }
    adamMRef.current = { x: 0, y: 0 }
    adamVRef.current = { x: 0, y: 0 }
    adagradGRef.current = { x: 0, y: 0 }
    rmspropGRef.current = { x: 0, y: 0 }
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current)
    }
  }, [])

  const resetView = useCallback(() => {
    setZoom(1)
    setPanOffset({ x: 0, y: 0 })
  }, [])

  const optimizationStep = useCallback(() => {
    const cp = currentPointRef.current
    const newPoint = { ...cp }
    const isDebug = isDebugModeRef.current
    let activeLine = activeLineRef.current

    const updateExecution = (line: number, vars: Record<string, any>) => {
      activeLineRef.current = line
      executionVariablesRef.current = { ...executionVariablesRef.current, ...vars }
      setExecutionState({ activeLine: line, variables: { ...executionVariablesRef.current } })
    }

    if (!isDebug) {
      const grad = computeGradient(cp.x, cp.y, selectedFunction)
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
    } else {
      activeLine++
      const grad = computeGradient(cp.x, cp.y, selectedFunction)
      switch (selectedOptimizer) {
        case "gd":
          if (activeLine === 2) updateExecution(2, { grad_x: grad.x, grad_y: grad.y })
          else if (activeLine === 3) {
            newPoint.x -= config.learningRate * grad.x
            updateExecution(3, { x: newPoint.x })
          } else if (activeLine === 4) {
            newPoint.y -= config.learningRate * grad.y
            updateExecution(4, { y: newPoint.y })
          } else if (activeLine >= 5) {
            activeLine = 1
            updateExecution(1, {})
            return
          } else {
            updateExecution(activeLine, {})
          }
          break
        case "sgd":
          if (activeLine === 2) updateExecution(2, { grad_x: grad.x, grad_y: grad.y })
          else if (activeLine === 3) {
            const noise = [(Math.random() - 0.5) * 0.1, (Math.random() - 0.5) * 0.1]
            updateExecution(3, { noise_x: noise[0], noise_y: noise[1] })
          } else if (activeLine === 4) {
            const noiseX = executionVariablesRef.current.noise_x || 0
            newPoint.x -= config.learningRate * (grad.x + noiseX)
            updateExecution(4, { x: newPoint.x })
          } else if (activeLine === 5) {
            const noiseY = executionVariablesRef.current.noise_y || 0
            newPoint.y -= config.learningRate * (grad.y + noiseY)
            updateExecution(5, { y: newPoint.y })
          } else if (activeLine >= 6) {
            activeLine = 1
            updateExecution(1, {})
            return
          } else {
            updateExecution(activeLine, {})
          }
          break
        case "momentum":
          if (activeLine === 2) updateExecution(2, { grad_x: grad.x, grad_y: grad.y })
          else if (activeLine === 3) {
            momentumRef.current.x = (config.momentum || 0.9) * momentumRef.current.x - config.learningRate * grad.x
            updateExecution(3, { velocity_x: momentumRef.current.x })
          } else if (activeLine === 4) {
            momentumRef.current.y = (config.momentum || 0.9) * momentumRef.current.y - config.learningRate * grad.y
            updateExecution(4, { velocity_y: momentumRef.current.y })
          } else if (activeLine === 5) {
            newPoint.x += momentumRef.current.x
            updateExecution(5, { x: newPoint.x })
          } else if (activeLine === 6) {
            newPoint.y += momentumRef.current.y
            updateExecution(6, { y: newPoint.y })
          } else if (activeLine >= 7) {
            activeLine = 1
            updateExecution(1, {})
            return
          } else {
            updateExecution(activeLine, {})
          }
          break
        case "adam":
          const beta1 = config.beta1 || 0.9
          const beta2 = config.beta2 || 0.999
          const eps = config.epsilon || 1e-8
          const t = iterationRef.current + 1
          if (activeLine === 2) updateExecution(2, { grad_x: grad.x, grad_y: grad.y })
          else if (activeLine === 3) {
            adamMRef.current.x = beta1 * adamMRef.current.x + (1 - beta1) * grad.x
            updateExecution(3, { m_x: adamMRef.current.x })
          } else if (activeLine === 4) {
            adamMRef.current.y = beta1 * adamMRef.current.y + (1 - beta1) * grad.y
            updateExecution(4, { m_y: adamMRef.current.y })
          } else if (activeLine === 5) {
            adamVRef.current.x = beta2 * adamVRef.current.x + (1 - beta2) * grad.x * grad.x
            updateExecution(5, { v_x: adamVRef.current.x })
          } else if (activeLine === 6) {
            adamVRef.current.y = beta2 * adamVRef.current.y + (1 - beta2) * grad.y * grad.y
            updateExecution(6, { v_y: adamVRef.current.y })
          } else if (activeLine === 7) {
            const mHatX = adamMRef.current.x / (1 - Math.pow(beta1, t))
            updateExecution(7, { m_hat_x: mHatX })
          } else if (activeLine === 8) {
            const mHatY = adamMRef.current.y / (1 - Math.pow(beta1, t))
            updateExecution(8, { m_hat_y: mHatY })
          } else if (activeLine === 9) {
            const vHatX = adamVRef.current.x / (1 - Math.pow(beta2, t))
            updateExecution(9, { v_hat_x: vHatX })
          } else if (activeLine === 10) {
            const vHatY = adamVRef.current.y / (1 - Math.pow(beta2, t))
            updateExecution(10, { v_hat_y: vHatY })
          } else if (activeLine === 11) {
            const vars = executionVariablesRef.current
            newPoint.x -= (config.learningRate * vars.m_hat_x) / (Math.sqrt(vars.v_hat_x) + eps)
            updateExecution(11, { x: newPoint.x })
          } else if (activeLine === 12) {
            const vars = executionVariablesRef.current
            newPoint.y -= (config.learningRate * vars.m_hat_y) / (Math.sqrt(vars.v_hat_y) + eps)
            updateExecution(12, { y: newPoint.y })
          } else if (activeLine >= 13) {
            activeLine = 1
            updateExecution(1, {})
            return
          } else {
            updateExecution(activeLine, {})
          }
          break
        case "rmsprop":
          const bDecay = config.momentum || 0.9
          const eR = config.epsilon || 1e-8
          if (activeLine === 2) updateExecution(2, { grad_x: grad.x, grad_y: grad.y })
          else if (activeLine === 3) {
            rmspropGRef.current.x = bDecay * rmspropGRef.current.x + (1 - bDecay) * grad.x * grad.x
            updateExecution(3, { g_x: rmspropGRef.current.x })
          } else if (activeLine === 4) {
            rmspropGRef.current.y = bDecay * rmspropGRef.current.y + (1 - bDecay) * grad.y * grad.y
            updateExecution(4, { g_y: rmspropGRef.current.y })
          } else if (activeLine === 5) {
            newPoint.x -= (config.learningRate * grad.x) / (Math.sqrt(rmspropGRef.current.x) + eR)
            updateExecution(5, { x: newPoint.x })
          } else if (activeLine === 6) {
            newPoint.y -= (config.learningRate * grad.y) / (Math.sqrt(rmspropGRef.current.y) + eR)
            updateExecution(6, { y: newPoint.y })
          } else if (activeLine >= 7) {
            activeLine = 1
            updateExecution(1, {})
            return
          } else {
            updateExecution(activeLine, {})
          }
          break
        case "adagrad":
          const eA = config.epsilon || 1e-8
          if (activeLine === 2) updateExecution(2, { grad_x: grad.x, grad_y: grad.y })
          else if (activeLine === 3) {
            adagradGRef.current.x += grad.x * grad.x
            updateExecution(3, { g_x: adagradGRef.current.x })
          } else if (activeLine === 4) {
            adagradGRef.current.y += grad.y * grad.y
            updateExecution(4, { g_y: adagradGRef.current.y })
          } else if (activeLine === 5) {
            newPoint.x -= (config.learningRate * grad.x) / (Math.sqrt(adagradGRef.current.x) + eA)
            updateExecution(5, { x: newPoint.x })
          } else if (activeLine === 6) {
            newPoint.y -= (config.learningRate * grad.y) / (Math.sqrt(adagradGRef.current.y) + eA)
            updateExecution(6, { y: newPoint.y })
          } else if (activeLine >= 7) {
            activeLine = 1
            updateExecution(1, {})
            return
          } else {
            updateExecution(activeLine, {})
          }
          break
      }
    }

    currentPointRef.current = newPoint
    iterationRef.current = iterationRef.current + 1
    setCurrentPoint(newPoint)
    setPath((prev) => [...prev.slice(-100), newPoint])
    setIteration((prev) => prev + 1)
    setLoss(objectiveFunctions[selectedFunction](newPoint.x, newPoint.y))
  }, [selectedOptimizer, selectedFunction, config, computeGradient])

  const animate = useCallback((time: number) => {
    if (isRunningRef.current) {
      if (isDebugModeRef.current) {
        setTimeout(() => {
          optimizationStep()
          animationRef.current = requestAnimationFrame(animate)
        }, activeLineRef.current === -1 ? 0 : stepDelay) // No delay for first step
      } else {
        optimizationStep()
        animationRef.current = requestAnimationFrame(animate)
      }
    }
  }, [optimizationStep, stepDelay])

  const toggleAnimation = useCallback(() => {
    if (isRunning) {
      setIsRunning(false)
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    } else {
      if (isDebugModeRef.current) {
        setExecutionState({ activeLine: -1, variables: {} })
        activeLineRef.current = -1
        executionVariablesRef.current = {}
      }
      setIsRunning(true)
      animationRef.current = requestAnimationFrame(animate)
    }
  }, [isRunning, animate])

  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isRunning) return
    const canvas = e.currentTarget
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    const x_px = (e.clientX - rect.left) * scaleX
    const y_px = (e.clientY - rect.top) * scaleY
    
    const scale = 50 * zoom
    const offsetX = (canvas.width / 2) + panOffset.x
    const offsetY = (canvas.height / 2) + panOffset.y
    
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
  }, [isRunning, selectedFunction, zoom, panOffset])

  return {
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
  }
}
