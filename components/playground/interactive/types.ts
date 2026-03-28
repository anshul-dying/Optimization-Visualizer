export interface OptimizerConfig {
  learningRate: number
  momentum?: number
  beta1?: number
  beta2?: number
  epsilon?: number
  batchSize?: number
}

export interface Point {
  x: number
  y: number
}

export interface ExecutionState {
  activeLine: number
  variables: Record<string, any>
}
