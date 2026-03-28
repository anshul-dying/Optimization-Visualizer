import { OptimizerConfig } from "./types"

export const optimizerConfigs: Record<string, OptimizerConfig> = {
  gd: { learningRate: 0.01 },
  sgd: { learningRate: 0.01 },
  momentum: { learningRate: 0.01, momentum: 0.9 },
  adam: { learningRate: 0.001, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 },
  rmsprop: { learningRate: 0.001, momentum: 0.9, epsilon: 1e-8 },
  adagrad: { learningRate: 0.01, epsilon: 1e-8 },
}

export const objectiveFunctions = {
  quadratic: (x: number, y: number) => x * x + y * y,
  rosenbrock: (x: number, y: number) => (1 - x) ** 2 + 100 * (y - x ** 2) ** 2,
  himmelblau: (x: number, y: number) => (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2,
  beale: (x: number, y: number) =>
    (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2,
}
