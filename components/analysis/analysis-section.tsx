"use client"

import React from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from "recharts"
import { Clock, Zap, Target, TrendingUp } from "lucide-react"

const convergenceData = [
  { iteration: 0, GD: 100, SGD: 100, Adam: 100, MiniBatch: 100 },
  { iteration: 10, GD: 85, SGD: 75, Adam: 60, MiniBatch: 80 },
  { iteration: 20, GD: 72, SGD: 55, Adam: 35, MiniBatch: 65 },
  { iteration: 30, GD: 61, SGD: 42, Adam: 20, MiniBatch: 52 },
  { iteration: 40, GD: 52, SGD: 35, Adam: 12, MiniBatch: 42 },
  { iteration: 50, GD: 45, SGD: 30, Adam: 8, MiniBatch: 35 },
]

const complexityData = [
  { algorithm: "GD", time: "O(n)", space: "O(n)", rate: "Linear" },
  { algorithm: "SGD", time: "O(1)", space: "O(1)", rate: "Sub-linear" },
  { algorithm: "Mini-Batch", time: "O(k)", space: "O(k)", rate: "Linear" },
  { algorithm: "Adam", time: "O(n)", space: "O(n)", rate: "Fast" },
  { algorithm: "Analytical", time: "O(n³)", space: "O(n²)", rate: "Instant" },
  { algorithm: "QP", time: "O(n³)", space: "O(n²)", rate: "Polynomial" },
]

const performanceData = [
  { metric: "Speed", GD: 60, SGD: 95, Adam: 85, MiniBatch: 75 },
  { metric: "Stability", GD: 90, SGD: 40, Adam: 80, MiniBatch: 85 },
  { metric: "Memory", GD: 30, SGD: 95, Adam: 30, MiniBatch: 70 },
  { metric: "Accuracy", GD: 85, SGD: 70, Adam: 90, MiniBatch: 80 },
]

export function AnalysisSection() {
  return (
    <section id="analysis" className="py-20 px-4 bg-muted/20">
      <div className="container mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-5xl font-bold mb-6 text-balance tracking-tight">Algorithm Analysis & Comparison</h2>
          <p className="text-2xl text-muted-foreground text-balance max-w-3xl mx-auto">
            Deep dive into the computational complexity and performance characteristics of each optimizer
          </p>
        </div>

        <Tabs defaultValue="convergence" className="w-full">
          <TabsList className="grid w-full grid-cols-4 mb-8">
            <TabsTrigger value="convergence">Convergence</TabsTrigger>
            <TabsTrigger value="complexity">Complexity</TabsTrigger>
            <TabsTrigger value="performance">Performance</TabsTrigger>
            <TabsTrigger value="daa">DAA Analysis</TabsTrigger>
          </TabsList>

          <TabsContent value="convergence" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="w-5 h-5" />
                  Convergence Comparison
                </CardTitle>
                <CardDescription>Loss reduction over iterations for different optimization algorithms</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                    <LineChart data={convergenceData}>
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
                      <Line type="monotone" dataKey="GD" stroke="#3b82f6" strokeWidth={2} />
                      <Line type="monotone" dataKey="SGD" stroke="#10b981" strokeWidth={2} />
                      <Line type="monotone" dataKey="Adam" stroke="#f97316" strokeWidth={2} />
                      <Line type="monotone" dataKey="MiniBatch" stroke="#8b5cf6" strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="complexity" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {complexityData.map((item) => (
                <Card key={item.algorithm} className="border-border/60 shadow-sm hover:shadow-md transition-shadow">
                  <CardHeader>
                    <CardTitle className="text-2xl font-bold text-primary">{item.algorithm}</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex justify-between items-center p-3 rounded-lg bg-muted/30 border border-border/50">
                      <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Time Complexity:</span>
                      <Badge variant="secondary" className="font-mono text-base px-3 py-1">
                        {item.time}
                      </Badge>
                    </div>
                    <div className="flex justify-between items-center p-3 rounded-lg bg-muted/30 border border-border/50">
                      <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Space Complexity:</span>
                      <Badge variant="outline" className="font-mono text-base px-3 py-1">
                        {item.space}
                      </Badge>
                    </div>
                    <div className="flex justify-between items-center p-3 rounded-lg bg-muted/30 border border-border/50">
                      <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Convergence:</span>
                      <Badge variant="secondary" className="text-base font-semibold px-3 py-1">{item.rate}</Badge>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="performance" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-2xl">
                  <Zap className="w-6 h-6 text-primary" />
                  Performance Metrics
                </CardTitle>
                <CardDescription className="text-base">Comparative analysis across key performance dimensions</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-96">
                  <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                    <BarChart data={performanceData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="metric" stroke="var(--muted-foreground)" />
                      <YAxis stroke="var(--muted-foreground)" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "var(--card)",
                          border: "1px solid var(--border)",
                          borderRadius: "8px",
                        }}
                      />
                      <Bar dataKey="GD" fill="#3b82f6" />
                      <Bar dataKey="SGD" fill="#10b981" />
                      <Bar dataKey="Adam" fill="#f97316" />
                      <Bar dataKey="MiniBatch" fill="#8b5cf6" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="daa" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-3 text-2xl">
                    <Clock className="w-6 h-6 text-primary" />
                    Time Complexity Analysis
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="space-y-4">
                    <div className="p-4 bg-muted/50 rounded-xl border border-border/50">
                      <h4 className="font-bold text-lg mb-2">Gradient Descent</h4>
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        Linear in dataset size. Each iteration processes all <span className="italic font-serif">n</span> samples.
                      </p>
                    </div>
                    <div className="p-4 bg-muted/50 rounded-xl border border-border/50">
                      <h4 className="font-bold text-lg mb-2">SGD</h4>
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        Constant time per iteration. Processes one sample at a time.
                      </p>
                    </div>
                    <div className="p-4 bg-muted/50 rounded-xl border border-border/50">
                      <h4 className="font-bold text-lg mb-2">Analytical</h4>
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        Cubic complexity due to matrix inversion operations.
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-3 text-2xl">
                    <Target className="w-6 h-6 text-primary" />
                    Convergence Theory
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="space-y-4">
                    <div className="p-4 bg-muted/50 rounded-xl border border-border/50">
                      <h4 className="font-bold text-lg mb-2">Linear Convergence</h4>
                      <div className="text-xl py-4 overflow-x-auto">
                        <math display="block" className="text-foreground">
                          <mo>||</mo>
                          <msub><mi>x</mi><mi>k</mi></msub>
                          <mo>&#x2212;</mo>
                          <msup><mi>x</mi><mo>*</mo></msup>
                          <mo>||</mo>
                          <mo>&#x2264;</mo>
                          <msup><mi>c</mi><mi>k</mi></msup>
                          <mo>||</mo>
                          <msub><mi>x</mi><mn>0</mn></msub>
                          <mo>&#x2212;</mo>
                          <msup><mi>x</mi><mo>*</mo></msup>
                          <mo>||</mo>
                        </math>
                      </div>
                      <span className="text-xs text-muted-foreground uppercase font-bold tracking-widest mt-2 block">where 0 &lt; c &lt; 1</span>
                    </div>
                    <div className="p-4 bg-muted/50 rounded-xl border border-border/50">
                      <h4 className="font-bold text-lg mb-2">Sub-linear Convergence</h4>
                      <div className="text-xl py-4 overflow-x-auto">
                        <math display="block" className="text-foreground">
                          <mo>||</mo>
                          <msub><mi>x</mi><mi>k</mi></msub>
                          <mo>&#x2212;</mo>
                          <msup><mi>x</mi><mo>*</mo></msup>
                          <mo>||</mo>
                          <mo>&#x2264;</mo>
                          <mfrac>
                            <mi>C</mi>
                            <msup><mi>k</mi><mi>α</mi></msup>
                          </mfrac>
                        </math>
                      </div>
                      <span className="text-xs text-muted-foreground uppercase font-bold tracking-widest mt-2 block">where α &gt; 0</span>
                    </div>
                    <div className="p-4 bg-muted/50 rounded-xl border border-border/50">
                      <h4 className="font-bold text-lg mb-2">Quadratic Convergence</h4>
                      <div className="text-xl py-4 overflow-x-auto">
                        <math display="block" className="text-foreground">
                          <mo>||</mo>
                          <msub><mi>x</mi><mrow><mi>k</mi><mo>+</mo><mn>1</mn></mrow></msub>
                          <mo>&#x2212;</mo>
                          <msup><mi>x</mi><mo>*</mo></msup>
                          <mo>||</mo>
                          <mo>&#x2264;</mo>
                          <mi>C</mi>
                          <msup>
                            <mrow>
                              <mo>||</mo>
                              <msub><mi>x</mi><mi>k</mi></msub>
                              <mo>&#x2212;</mo>
                              <msup><mi>x</mi><mo>*</mo></msup>
                              <mo>||</mo>
                            </mrow>
                            <mn>2</mn>
                          </msup>
                        </math>
                      </div>
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
