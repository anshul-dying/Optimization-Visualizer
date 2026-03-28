import React from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Brain, Target, Zap, TrendingUp } from "lucide-react"
import { MLConfig } from "./types"

interface MLVisualizerProps {
  canvasRef: React.RefObject<HTMLCanvasElement>
  config: MLConfig
  isTraining: boolean
  apiStatus: string
  dataPoints: any[]
}

export function MLVisualizer({ canvasRef, config, isTraining, apiStatus, dataPoints }: MLVisualizerProps) {
  return (
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
                className="w-full h-auto border rounded-lg transition-colors bg-background/40"
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
                          <mi>g</mi><mo>&#x2190;</mo><mo>&#x2207;</mo><mi>J</mi><mo>(</mo><msub><mi>θ</mi><mi>t</mi></msub><mo>)</mo>
                          <mspace width="1em" /><mtext className="text-sm text-muted-foreground font-sans">for batch B</mtext>
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
                          <msub><mi>θ</mi><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>&#x2190;</mo><msub><mi>θ</mi><mi>t</mi></msub><mo>&#x2212;</mo><mi>η</mi><mo>&#x22C5;</mo><mi>g</mi>
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
                          <mo>||</mo><mo>&#x2207;</mo><mi>J</mi><mo>(</mo><mi>θ</mi><mo>)</mo><mo>||</mo><mo>&lt;</mo><mi>ε</mi>
                          <mspace width="1em" /><mtext className="text-sm text-muted-foreground font-sans">return θ</mtext>
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
  )
}
