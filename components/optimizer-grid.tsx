"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { OptimizerVisualization } from "@/components/optimizer-visualization"
import { Play, BarChart3, Cpu, Zap, Target, TrendingUp, ExternalLink } from "lucide-react"

const optimizers = [
  {
    id: "gd",
    name: "Gradient Descent",
    description: "Classic batch gradient descent with fixed learning rate",
    complexity: "O(n)",
    convergence: "Linear",
    icon: TrendingUp,
    color: "from-blue-500 to-cyan-500",
    features: ["Deterministic", "Batch Processing", "Global Minimum"],
    formula: "θ = θ - α∇J(θ)",
    mathml: (
      <math>
        <mi>θ</mi><mo>&#x2190;</mo><mi>θ</mi><mo>&#x2212;</mo><mi>α</mi><mo>&#x2207;</mo><mi>J</mi><mo>(</mo><mi>θ</mi><mo>)</mo>
      </math>
    ),
    useCases: ["Convex optimization", "Large datasets", "Stable convergence"],
  },
  {
    id: "sgd",
    name: "Stochastic GD",
    description: "Single sample gradient updates with noise",
    complexity: "O(1)",
    convergence: "Sub-linear",
    icon: Zap,
    color: "from-green-500 to-emerald-500",
    features: ["Stochastic", "Fast Updates", "Noise Resilient"],
    formula: "θ = θ - α∇J(θ; x⁽ⁱ⁾, y⁽ⁱ⁾)",
    mathml: (
      <math>
        <mi>θ</mi><mo>&#x2190;</mo><mi>θ</mi><mo>&#x2212;</mo><mi>α</mi><mo>&#x2207;</mo><msub><mi>J</mi><mi>i</mi></msub><mo>(</mo><mi>θ</mi><mo>)</mo>
      </math>
    ),
    useCases: ["Online learning", "Large datasets", "Escape local minima"],
  },
  {
    id: "mini-batch",
    name: "Mini-Batch SGD",
    description: "Balanced approach with small batch processing",
    complexity: "O(k)",
    convergence: "Linear",
    icon: Target,
    color: "from-purple-500 to-violet-500",
    features: ["Balanced", "Vectorized", "Stable"],
    formula: "θ = θ - α∇J(θ; B)",
    mathml: (
      <math>
        <mi>θ</mi><mo>&#x2190;</mo><mi>θ</mi><mo>&#x2212;</mo><mi>α</mi><mo>&#x2207;</mo><msub><mi>J</mi><mi>B</mi></msub><mo>(</mo><mi>θ</mi><mo>)</mo>
      </math>
    ),
    useCases: ["Deep learning", "GPU optimization", "Balanced performance"],
  },
  {
    id: "adam",
    name: "Adam Optimizer",
    description: "Adaptive moment estimation with bias correction",
    complexity: "O(n)",
    convergence: "Fast",
    icon: Cpu,
    color: "from-orange-500 to-red-500",
    features: ["Adaptive", "Momentum", "Bias Correction"],
    formula: "θ = θ - α·m̂/(√v̂ + ε)",
    mathml: (
      <math>
        <mi>θ</mi><mo>&#x2190;</mo><mi>θ</mi><mo>&#x2212;</mo><mi>α</mi><mo>&#x22C5;</mo>
        <mfrac>
          <msub><mover><mi>m</mi><mo stretchy="false">^</mo></mover><mi>t</mi></msub>
          <mrow><msqrt><msub><mover><mi>v</mi><mo stretchy="false">^</mo></mover><mi>t</mi></msub></msqrt><mo>+</mo><mi>ε</mi></mrow>
        </mfrac>
      </math>
    ),
    useCases: ["Neural networks", "Sparse gradients", "Non-stationary objectives"],
  },
  {
    id: "analytical",
    name: "Analytical Solution",
    description: "Closed-form mathematical solution",
    complexity: "O(n³)",
    convergence: "Instant",
    icon: BarChart3,
    color: "from-pink-500 to-rose-500",
    features: ["Exact", "One-step", "Matrix Inverse"],
    formula: "θ = (XᵀX)⁻¹Xᵀy",
    mathml: (
      <math>
        <mi>θ</mi><mo>=</mo><msup><mrow><mo>(</mo><msup><mi>X</mi><mi>T</mi></msup><mi>X</mi><mo>)</mo></mrow><mrow><mo>&#x2212;</mo><mn>1</mn></mrow></msup><msup><mi>X</mi><mi>T</mi></msup><mi>y</mi>
      </math>
    ),
    useCases: ["Linear regression", "Small datasets", "Exact solutions"],
  },
  {
    id: "qp",
    name: "Quadratic Programming",
    description: "Constrained optimization with quadratic objective",
    complexity: "O(n³)",
    convergence: "Polynomial",
    icon: Target,
    color: "from-indigo-500 to-blue-500",
    features: ["Constrained", "Convex", "Interior Point"],
    formula: "min ½xᵀQx + cᵀx s.t. Ax ≤ b",
    mathml: (
      <math>
        <munder><mo>min</mo><mi>x</mi></munder><mfrac><mn>1</mn><mn>2</mn></mfrac><msup><mi>x</mi><mi>T</mi></msup><mi>Q</mi><mi>x</mi><mo>+</mo><msup><mi>c</mi><mi>T</mi></msup><mi>x</mi>
      </math>
    ),
    useCases: ["SVM", "Portfolio optimization", "Constrained problems"],
  },
]

export function OptimizerGrid() {
  const scrollToPlayground = () => {
    document.getElementById("playground")?.scrollIntoView({ behavior: "smooth" })
  }

  return (
    <section id="optimizers" className="py-20 px-4">
      <div className="container mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold mb-4 text-balance">Optimization Algorithms</h2>
          <p className="text-xl text-muted-foreground text-balance max-w-2xl mx-auto">
            Interactive visualizations of popular machine learning optimization techniques
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {optimizers.map((optimizer) => {
            const IconComponent = optimizer.icon
            return (
              <Card
                key={optimizer.id}
                className="group hover:shadow-lg transition-all duration-300 border-border/50 hover:border-primary/50 overflow-hidden"
              >
                <CardHeader>
                  <div className="flex items-center justify-between mb-2">
                    <div
                      className={`w-12 h-12 rounded-lg bg-gradient-to-br ${optimizer.color} flex items-center justify-center pulse-glow`}
                    >
                      <IconComponent className="w-6 h-6 text-white" />
                    </div>
                    <Badge variant="secondary" className="text-xs">
                      {optimizer.complexity}
                    </Badge>
                  </div>
                  <CardTitle className="text-xl">{optimizer.name}</CardTitle>
                  <CardDescription className="text-sm leading-relaxed">{optimizer.description}</CardDescription>
                </CardHeader>

                <CardContent className="space-y-4">
                  <OptimizerVisualization type={optimizer.id} />

                  <div className="bg-muted/30 rounded-lg p-4 border border-border/50">
                    <div className="text-xs font-bold uppercase tracking-widest text-muted-foreground mb-2">Update Rule / Objective:</div>
                    <div className="text-2xl text-primary py-1 overflow-x-auto">
                      {optimizer.mathml}
                    </div>
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Convergence:</span>
                      <span className="font-medium">{optimizer.convergence}</span>
                    </div>

                    <div className="flex flex-wrap gap-1">
                      {optimizer.features.map((feature) => (
                        <Badge key={feature} variant="outline" className="text-xs">
                          {feature}
                        </Badge>
                      ))}
                    </div>

                    <div className="pt-2 border-t border-border/50">
                      <div className="text-xs text-muted-foreground mb-2">Best for:</div>
                      <div className="space-y-1">
                        {optimizer.useCases.map((useCase, index) => (
                          <div key={index} className="text-xs text-muted-foreground flex items-center">
                            <div className="w-1 h-1 bg-primary rounded-full mr-2" />
                            {useCase}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  <Button
                    className="w-full group-hover:bg-primary group-hover:text-primary-foreground transition-colors"
                    onClick={scrollToPlayground}
                  >
                    <Play className="w-4 h-4 mr-2" />
                    Try in Playground
                    <ExternalLink className="w-3 h-3 ml-2" />
                  </Button>
                </CardContent>
              </Card>
            )
          })}
        </div>

        <div className="mt-16 text-center">
          <Card className="max-w-4xl mx-auto">
            <CardHeader>
              <CardTitle>Quick Comparison</CardTitle>
              <CardDescription>Choose the right optimizer for your specific use case</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center">
                  <div className="w-16 h-16 bg-gradient-to-br from-green-500 to-emerald-500 rounded-full flex items-center justify-center mx-auto mb-3">
                    <Zap className="w-8 h-8 text-white" />
                  </div>
                  <h3 className="font-semibold mb-2">Speed Priority</h3>
                  <p className="text-sm text-muted-foreground mb-3">
                    When you need fast iterations and can handle some noise
                  </p>
                  <Badge variant="outline">SGD, Mini-Batch SGD</Badge>
                </div>

                <div className="text-center">
                  <div className="w-16 h-16 bg-gradient-to-br from-orange-500 to-red-500 rounded-full flex items-center justify-center mx-auto mb-3">
                    <Target className="w-8 h-8 text-white" />
                  </div>
                  <h3 className="font-semibold mb-2">Accuracy Priority</h3>
                  <p className="text-sm text-muted-foreground mb-3">When you need stable, reliable convergence</p>
                  <Badge variant="outline">Adam, Gradient Descent</Badge>
                </div>

                <div className="text-center">
                  <div className="w-16 h-16 bg-gradient-to-br from-pink-500 to-rose-500 rounded-full flex items-center justify-center mx-auto mb-3">
                    <BarChart3 className="w-8 h-8 text-white" />
                  </div>
                  <h3 className="font-semibold mb-2">Exact Solutions</h3>
                  <p className="text-sm text-muted-foreground mb-3">When mathematical precision is required</p>
                  <Badge variant="outline">Analytical, QP</Badge>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  )
}
