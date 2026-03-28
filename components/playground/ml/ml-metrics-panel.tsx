import React from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Label } from "@/components/ui/label"
import { TrendingUp, Target } from "lucide-react"
import { TrainingMetrics } from "./types"

interface MLMetricsPanelProps {
  trainingMetrics: TrainingMetrics[]
  testAccuracy: number
  confusionMatrix: number[][]
  algorithm: string
}

export function MLMetricsPanel({ trainingMetrics, testAccuracy, confusionMatrix, algorithm }: MLMetricsPanelProps) {
  const lastMetric = trainingMetrics[trainingMetrics.length - 1]

  return (
    <Card className="xl:col-span-1">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <TrendingUp className="w-5 h-5" />
          Performance Metrics
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {trainingMetrics.length > 0 ? (
          <>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Current Loss:</span>
                <Badge variant="outline">{lastMetric?.loss.toFixed(4)}</Badge>
              </div>
              <div className="flex justify-between text-sm">
                <span>Current Accuracy:</span>
                <Badge variant="outline">{(lastMetric?.accuracy * 100).toFixed(1)}%</Badge>
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
                    {algorithm === 'linear' || algorithm === 'logistic' ? 'O(n·epochs)' : 
                     algorithm === 'svm' ? 'O(n²·epochs)' : 
                     algorithm === 'neural_network' ? 'O(n·L·epochs)' : 'O(n·log n)'}
                  </Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-[10px] uppercase font-bold text-muted-foreground tracking-wider">Space Complexity</span>
                  <Badge variant="outline" className="font-mono text-[10px]">
                    {algorithm === 'neural_network' ? 'O(L·W)' : 'O(n·d)'}
                  </Badge>
                </div>
              </div>

              <div className="space-y-2">
                <Label className="text-[10px] uppercase font-bold text-muted-foreground">DAA Characteristics</Label>
                <div className="text-xs space-y-2">
                  <div className="flex justify-between p-2 bg-muted/30 rounded border border-border/50">
                    <span>Paradigm:</span>
                    <span className="font-semibold text-primary">
                      {['decision_tree', 'random_forest'].includes(algorithm) ? 'Greedy / Recursive' : 'Iterative Optimization'}
                    </span>
                  </div>
                  <div className="flex justify-between p-2 bg-muted/30 rounded border border-border/50">
                    <span>Convergence:</span>
                    <span className={trainingMetrics.length > 50 ? "text-green-500 font-semibold" : "text-yellow-500 font-semibold"}>
                      {algorithm === 'svm' ? 'Quadratic' : 'Linear / Sub-linear'}
                    </span>
                  </div>
                  <div className="flex justify-between p-2 bg-muted/30 rounded border border-border/50">
                    <span>Stability:</span>
                    <span className="text-green-500 font-semibold">Stable (Asymptotically)</span>
                  </div>
                </div>
              </div>
            </div>
          </>
        ) : (
          <div className="text-center text-muted-foreground py-8">
            <Target className="w-12 h-12 mx-auto mb-2 opacity-50" />
            <p>Start training to see metrics</p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
