"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ScatterChart,
  Scatter,
} from "recharts"
import { TrendingUp, Zap, Clock, Target, BarChart3, Activity } from "lucide-react"

const convergenceRates = [
  { iteration: 0, GD: 1.0, SGD: 1.0, Adam: 1.0, MiniBatch: 1.0, RMSprop: 1.0 },
  { iteration: 5, GD: 0.82, SGD: 0.71, Adam: 0.45, MiniBatch: 0.78, RMSprop: 0.65 },
  { iteration: 10, GD: 0.67, SGD: 0.52, Adam: 0.25, MiniBatch: 0.61, RMSprop: 0.42 },
  { iteration: 15, GD: 0.55, SGD: 0.38, Adam: 0.15, MiniBatch: 0.48, RMSprop: 0.28 },
  { iteration: 20, GD: 0.45, SGD: 0.29, Adam: 0.09, MiniBatch: 0.38, RMSprop: 0.19 },
  { iteration: 25, GD: 0.37, SGD: 0.23, Adam: 0.06, MiniBatch: 0.31, RMSprop: 0.13 },
  { iteration: 30, GD: 0.31, SGD: 0.19, Adam: 0.04, MiniBatch: 0.25, RMSprop: 0.09 },
]

const performanceMetrics = [
  { optimizer: "GD", speed: 60, stability: 95, memory: 40, convergence: 80, scalability: 30 },
  { optimizer: "SGD", speed: 95, stability: 45, memory: 95, convergence: 65, scalability: 90 },
  { optimizer: "Mini-Batch", speed: 80, stability: 85, memory: 75, convergence: 85, scalability: 85 },
  { optimizer: "Adam", speed: 85, stability: 80, memory: 35, convergence: 95, scalability: 70 },
  { optimizer: "RMSprop", speed: 80, stability: 75, memory: 40, convergence: 85, scalability: 75 },
]

const complexityAnalysis = [
  { algorithm: "Gradient Descent", timePerIteration: 0.1, memoryUsage: 100, parallelizability: 30 },
  { algorithm: "SGD", timePerIteration: 0.01, memoryUsage: 10, parallelizability: 95 },
  { algorithm: "Mini-Batch SGD", timePerIteration: 0.05, memoryUsage: 50, parallelizability: 85 },
  { algorithm: "Adam", timePerIteration: 0.12, memoryUsage: 200, parallelizability: 40 },
  { algorithm: "RMSprop", timePerIteration: 0.11, memoryUsage: 150, parallelizability: 45 },
]

export function AdvancedMetrics() {
  const [selectedMetric, setSelectedMetric] = useState("convergence")

  return (
    <section className="py-20 px-4 bg-muted/10">
      <div className="container mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold mb-4 text-balance">Advanced Performance Metrics</h2>
          <p className="text-xl text-muted-foreground text-balance max-w-2xl mx-auto">
            Deep dive into optimizer performance across multiple dimensions
          </p>
        </div>

        <Tabs defaultValue="convergence" className="w-full">
          <TabsList className="grid w-full grid-cols-4 mb-8">
            <TabsTrigger value="convergence" className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4" />
              Convergence
            </TabsTrigger>
            <TabsTrigger value="performance" className="flex items-center gap-2">
              <Activity className="w-4 h-4" />
              Performance
            </TabsTrigger>
            <TabsTrigger value="complexity" className="flex items-center gap-2">
              <BarChart3 className="w-4 h-4" />
              Complexity
            </TabsTrigger>
            <TabsTrigger value="realworld" className="flex items-center gap-2">
              <Target className="w-4 h-4" />
              Real-World
            </TabsTrigger>
          </TabsList>

          <TabsContent value="convergence" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Convergence Rate Analysis</CardTitle>
                <CardDescription>Loss reduction over iterations for different optimization algorithms</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={convergenceRates}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="iteration" stroke="var(--muted-foreground)" />
                      <YAxis stroke="var(--muted-foreground)" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "var(--card)",
                          border: "1px solid var(--border)",
                          borderRadius: "8px",
                        }}
                      />
                      <Line type="monotone" dataKey="GD" stroke="#3b82f6" strokeWidth={2} name="Gradient Descent" />
                      <Line type="monotone" dataKey="SGD" stroke="#10b981" strokeWidth={2} name="SGD" />
                      <Line type="monotone" dataKey="Adam" stroke="#f97316" strokeWidth={2} name="Adam" />
                      <Line
                        type="monotone"
                        dataKey="MiniBatch"
                        stroke="#8b5cf6"
                        strokeWidth={2}
                        name="Mini-Batch SGD"
                      />
                      <Line type="monotone" dataKey="RMSprop" stroke="#ec4899" strokeWidth={2} name="RMSprop" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="performance" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Multi-Dimensional Performance Radar</CardTitle>
                <CardDescription>Comparative analysis across key performance dimensions</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={performanceMetrics}>
                      <PolarGrid stroke="var(--border)" />
                      <PolarAngleAxis
                        dataKey="optimizer"
                        tick={{ fill: "var(--muted-foreground)", fontSize: 12 }}
                      />
                      <PolarRadiusAxis
                        angle={90}
                        domain={[0, 100]}
                        tick={{ fill: "var(--muted-foreground)", fontSize: 10 }}
                      />
                      <Radar name="Speed" dataKey="speed" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.1} />
                      <Radar name="Stability" dataKey="stability" stroke="#10b981" fill="#10b981" fillOpacity={0.1} />
                      <Radar
                        name="Memory Efficiency"
                        dataKey="memory"
                        stroke="#f97316"
                        fill="#f97316"
                        fillOpacity={0.1}
                      />
                      <Radar
                        name="Convergence"
                        dataKey="convergence"
                        stroke="#8b5cf6"
                        fill="#8b5cf6"
                        fillOpacity={0.1}
                      />
                      <Radar
                        name="Scalability"
                        dataKey="scalability"
                        stroke="#ec4899"
                        fill="#ec4899"
                        fillOpacity={0.1}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "var(--card)",
                          border: "1px solid var(--border)",
                          borderRadius: "8px",
                        }}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="complexity" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Computational Complexity Trade-offs</CardTitle>
                <CardDescription>Time vs Memory vs Parallelizability analysis</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart data={complexityAnalysis}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis
                        dataKey="timePerIteration"
                        stroke="var(--muted-foreground)"
                        name="Time per Iteration (s)"
                        label={{ value: "Time per Iteration (s)", position: "insideBottom", offset: -5 }}
                      />
                      <YAxis
                        dataKey="memoryUsage"
                        stroke="var(--muted-foreground)"
                        name="Memory Usage (MB)"
                        label={{ value: "Memory Usage (MB)", angle: -90, position: "insideLeft" }}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "var(--card)",
                          border: "1px solid var(--border)",
                          borderRadius: "8px",
                        }}
                        formatter={(value, name) => [value, name]}
                        labelFormatter={(label) => `Algorithm: ${label}`}
                      />
                      <Scatter name="Algorithms" dataKey="parallelizability" fill="#8b5cf6" />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="realworld" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Clock className="w-5 h-5" />
                    Training Time
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm">ImageNet (ResNet-50)</span>
                      <Badge variant="outline">24h</Badge>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">BERT-Base</span>
                      <Badge variant="outline">4 days</Badge>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">GPT-3 Scale</span>
                      <Badge variant="outline">Weeks</Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Zap className="w-5 h-5" />
                    GPU Utilization
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Adam</span>
                      <Badge variant="outline">85%</Badge>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">SGD</span>
                      <Badge variant="outline">92%</Badge>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Mini-Batch</span>
                      <Badge variant="outline">88%</Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Target className="w-5 h-5" />
                    Best Use Cases
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <div className="p-3 bg-muted/50 rounded-lg">
                      <h4 className="font-semibold text-sm mb-1">Computer Vision</h4>
                      <p className="text-xs text-muted-foreground">SGD with momentum</p>
                    </div>
                    <div className="p-3 bg-muted/50 rounded-lg">
                      <h4 className="font-semibold text-sm mb-1">NLP Transformers</h4>
                      <p className="text-xs text-muted-foreground">Adam or AdamW</p>
                    </div>
                    <div className="p-3 bg-muted/50 rounded-lg">
                      <h4 className="font-semibold text-sm mb-1">Reinforcement Learning</h4>
                      <p className="text-xs text-muted-foreground">RMSprop or Adam</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </section>
  )
}
