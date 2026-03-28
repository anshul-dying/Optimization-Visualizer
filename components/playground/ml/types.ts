export interface MLConfig {
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

export interface DataPoint {
  x: number
  y: number
  label: number
}

export interface TrainingMetrics {
  epoch: number
  loss: number
  accuracy: number
  valLoss?: number
  valAccuracy?: number
}
