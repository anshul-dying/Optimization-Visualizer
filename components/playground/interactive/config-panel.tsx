import React from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Settings, Bug, Play, Pause, RotateCcw } from "lucide-react"
import { OptimizerConfig } from "./types"
import { optimizerConfigs, objectiveFunctions } from "./constants"

interface ConfigPanelProps {
  selectedOptimizer: string
  setSelectedOptimizer: (val: string) => void
  selectedFunction: string
  setSelectedFunction: (val: any) => void
  config: OptimizerConfig
  setConfig: React.Dispatch<React.SetStateAction<OptimizerConfig>>
  isDebugMode: boolean
  setIsDebugMode: (val: boolean) => void
  stepDelay: number
  setStepDelay: (val: number) => void
  isRunning: boolean
  toggleAnimation: () => void
  reset: () => void
  iteration: number
  loss: number
  currentPoint: { x: number; y: number }
}

export function ConfigPanel({
  selectedOptimizer,
  setSelectedOptimizer,
  selectedFunction,
  setSelectedFunction,
  config,
  setConfig,
  isDebugMode,
  setIsDebugMode,
  stepDelay,
  setStepDelay,
  isRunning,
  toggleAnimation,
  reset,
  iteration,
  loss,
  currentPoint,
}: ConfigPanelProps) {
  return (
    <Card className="lg:col-span-1">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Settings className="w-5 h-5" />
          Configuration
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-2">
          <label className="text-sm font-medium">Optimizer</label>
          <Select value={selectedOptimizer} onValueChange={setSelectedOptimizer}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="gd">Gradient Descent</SelectItem>
              <SelectItem value="sgd">SGD</SelectItem>
              <SelectItem value="momentum">Momentum</SelectItem>
              <SelectItem value="adam">Adam</SelectItem>
              <SelectItem value="rmsprop">RMSprop</SelectItem>
              <SelectItem value="adagrad">AdaGrad</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">Objective Function</label>
          <Select value={selectedFunction} onValueChange={setSelectedFunction}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="quadratic">Quadratic</SelectItem>
              <SelectItem value="rosenbrock">Rosenbrock</SelectItem>
              <SelectItem value="himmelblau">Himmelblau</SelectItem>
              <SelectItem value="beale">Beale</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <label className="text-sm font-medium">Learning Rate</label>
            <Input
              type="number"
              className="w-20 h-8 text-xs"
              step="0.0001"
              value={config.learningRate}
              onChange={(e) => setConfig((prev) => ({ ...prev, learningRate: parseFloat(e.target.value) || 0 }))}
            />
          </div>
          <Slider
            value={[config.learningRate]}
            onValueChange={([value]) => setConfig((prev) => ({ ...prev, learningRate: value }))}
            min={0.001}
            max={0.1}
            step={0.001}
          />
        </div>

        {selectedOptimizer === "momentum" && (
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <label className="text-sm font-medium">Momentum</label>
              <Input
                type="number"
                className="w-20 h-8 text-xs"
                step="0.01"
                value={config.momentum || 0.9}
                onChange={(e) => setConfig((prev) => ({ ...prev, momentum: parseFloat(e.target.value) || 0 }))}
              />
            </div>
            <Slider
              value={[config.momentum || 0.9]}
              onValueChange={([value]) => setConfig((prev) => ({ ...prev, momentum: value }))}
              min={0}
              max={0.99}
              step={0.01}
            />
          </div>
        )}

        {selectedOptimizer === "adam" && (
          <>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <label className="text-sm font-medium">Beta1</label>
                <Input
                  type="number"
                  className="w-20 h-8 text-xs"
                  step="0.01"
                  value={config.beta1 || 0.9}
                  onChange={(e) => setConfig((prev) => ({ ...prev, beta1: parseFloat(e.target.value) || 0 }))}
                />
              </div>
              <Slider
                value={[config.beta1 || 0.9]}
                onValueChange={([value]) => setConfig((prev) => ({ ...prev, beta1: value }))}
                min={0.1}
                max={0.99}
                step={0.01}
              />
            </div>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <label className="text-sm font-medium">Beta2</label>
                <Input
                  type="number"
                  className="w-20 h-8 text-xs"
                  step="0.001"
                  value={config.beta2 || 0.999}
                  onChange={(e) => setConfig((prev) => ({ ...prev, beta2: parseFloat(e.target.value) || 0 }))}
                />
              </div>
              <Slider
                value={[config.beta2 || 0.999]}
                onValueChange={([value]) => setConfig((prev) => ({ ...prev, beta2: value }))}
                min={0.9}
                max={0.999}
                step={0.001}
              />
            </div>
          </>
        )}

        {selectedOptimizer === "rmsprop" && (
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <label className="text-sm font-medium">Beta (Decay Rate)</label>
              <Input
                type="number"
                className="w-20 h-8 text-xs"
                step="0.01"
                value={config.momentum || 0.9}
                onChange={(e) => setConfig((prev) => ({ ...prev, momentum: parseFloat(e.target.value) || 0 }))}
              />
            </div>
            <Slider
              value={[config.momentum || 0.9]}
              onValueChange={([value]) => setConfig((prev) => ({ ...prev, momentum: value }))}
              min={0}
              max={0.99}
              step={0.01}
            />
          </div>
        )}

        {selectedOptimizer === "adagrad" && (
          <div className="bg-muted/50 p-3 rounded-md">
            <p className="text-xs text-muted-foreground">
              AdaGrad has no extra hyperparameters besides learning rate. It accumulates square gradients automatically.
            </p>
          </div>
        )}

        <div className="pt-4 border-t space-y-4">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label className="text-sm font-medium flex items-center gap-2">
                <Bug className="w-4 h-4 text-primary" />
                Debug Mode
              </Label>
              <p className="text-[10px] text-muted-foreground leading-tight">Step through code line-by-line</p>
            </div>
            <Switch checked={isDebugMode} onCheckedChange={setIsDebugMode} />
          </div>

          {isDebugMode && (
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <label className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
                  Step Delay (ms)
                </label>
                <span className="text-[10px] font-mono">{stepDelay}ms</span>
              </div>
              <Slider
                value={[stepDelay]}
                onValueChange={([value]) => setStepDelay(value)}
                min={50}
                max={1000}
                step={50}
              />
            </div>
          )}
        </div>

        <div className="flex gap-2">
          <Button onClick={toggleAnimation} className="flex-1">
            {isRunning ? <Pause className="w-4 h-4 mr-2" /> : <Play className="w-4 h-4 mr-2" />}
            {isRunning ? "Pause" : "Start"}
          </Button>
          <Button onClick={reset} variant="outline">
            <RotateCcw className="w-4 h-4" />
          </Button>
        </div>

        <div className="space-y-2 pt-4 border-t">
          <div className="flex justify-between text-sm">
            <span>Iteration:</span>
            <Badge variant="outline">{iteration}</Badge>
          </div>
          <div className="flex justify-between text-sm">
            <span>Loss:</span>
            <Badge variant="outline">{loss.toFixed(6)}</Badge>
          </div>
          <div className="flex justify-between text-sm">
            <span>Position:</span>
            <Badge variant="outline">
              ({currentPoint.x.toFixed(3)}, {currentPoint.y.toFixed(3)})
            </Badge>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
