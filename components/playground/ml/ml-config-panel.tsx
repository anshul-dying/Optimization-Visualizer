import React from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Settings, Play, Pause, RotateCcw } from "lucide-react"
import { MLConfig } from "./types"
import { algorithms, datasetTypes, kernelTypes, activationFunctions } from "./constants"

interface MLConfigPanelProps {
  config: MLConfig
  setConfig: React.Dispatch<React.SetStateAction<MLConfig>>
  dataset: string
  setDataset: (val: string) => void
  extremeValues: boolean
  setExtremeValues: (val: boolean) => void
  useRealAlgorithms: boolean
  setUseRealAlgorithms: (val: boolean) => void
  noiseLevel: number
  setNoiseLevel: (val: number) => void
  apiStatus: string
  isTraining: boolean
  currentEpoch: number
  simulateTraining: () => void
  stopTraining: () => void
  resetTraining: () => void
}

export function MLConfigPanel({
  config,
  setConfig,
  dataset,
  setDataset,
  extremeValues,
  setExtremeValues,
  useRealAlgorithms,
  setUseRealAlgorithms,
  noiseLevel,
  setNoiseLevel,
  apiStatus,
  isTraining,
  currentEpoch,
  simulateTraining,
  stopTraining,
  resetTraining,
}: MLConfigPanelProps) {
  return (
    <Card className="xl:col-span-1">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Settings className="w-5 h-5" />
          Configuration
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-2">
          <Label>Algorithm</Label>
          <Select value={config.algorithm} onValueChange={(val) => setConfig((p) => ({ ...p, algorithm: val }))}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {Object.entries(algorithms).map(([key, name]) => (
                <SelectItem key={key} value={key}>{name}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label>Dataset Type</Label>
          <Select value={dataset} onValueChange={setDataset}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {Object.entries(datasetTypes).map(([key, name]) => (
                <SelectItem key={key} value={key}>{name}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex items-center justify-between">
          <Label>Extreme Values</Label>
          <Switch checked={extremeValues} onCheckedChange={setExtremeValues} />
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label>Use Real Algorithms</Label>
            <Switch checked={useRealAlgorithms} onCheckedChange={setUseRealAlgorithms} disabled={apiStatus !== 'available'} />
          </div>
          <p className="text-xs text-muted-foreground">
            {apiStatus === 'checking' && '⏳ Checking API...'}
            {apiStatus === 'available' && '✅ API available - All algorithms ready'}
            {apiStatus === 'unavailable' && '❌ API unavailable (run: python core/api.py)'}
          </p>
        </div>

        <div className="space-y-2">
          <Label>Learning Rate: {config.learningRate}</Label>
          <Slider
            value={[config.learningRate]}
            onValueChange={([val]) => setConfig((p) => ({ ...p, learningRate: val }))}
            min={0.0001} max={1.0} step={0.0001}
          />
        </div>

        <div className="space-y-2">
          <Label>Epochs: {config.epochs}</Label>
          <Slider
            value={[config.epochs]}
            onValueChange={([val]) => setConfig((p) => ({ ...p, epochs: val }))}
            min={10} max={1000} step={10}
          />
        </div>

        <div className="space-y-2">
          <Label>Noise Level: {noiseLevel}</Label>
          <Slider
            value={[noiseLevel]}
            onValueChange={([val]) => setNoiseLevel(val)}
            min={0} max={1} step={0.01}
          />
        </div>

        {config.algorithm === "svm" && (
          <>
            <div className="space-y-2">
              <Label>Kernel Type</Label>
              <Select value={config.kernelType} onValueChange={(val) => setConfig((p) => ({ ...p, kernelType: val }))}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {Object.entries(kernelTypes).map(([key, name]) => (
                    <SelectItem key={key} value={key}>{name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>C Parameter: {config.C}</Label>
              <Slider value={[config.C || 1]} onValueChange={([val]) => setConfig((p) => ({ ...p, C: val }))} min={0.1} max={10} step={0.1} />
            </div>
          </>
        )}

        {config.algorithm === "neural_network" && (
          <>
            <div className="space-y-2">
              <Label>Activation Function</Label>
              <Select value={config.activation} onValueChange={(val) => setConfig((p) => ({ ...p, activation: val }))}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {Object.entries(activationFunctions).map(([key, name]) => (
                    <SelectItem key={key} value={key}>{name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Hidden Layers</Label>
              <Input
                value={config.hiddenLayers?.join(", ") || "64, 32"}
                onChange={(e) => {
                  const layers = e.target.value.split(",").map((s) => Number.parseInt(s.trim())).filter((n) => !isNaN(n))
                  setConfig((p) => ({ ...p, hiddenLayers: layers }))
                }}
                placeholder="64, 32, 16"
              />
            </div>
          </>
        )}

        <div className="flex gap-2">
          <Button onClick={isTraining ? stopTraining : simulateTraining} className="flex-1">
            {isTraining ? <Pause className="w-4 h-4 mr-2" /> : <Play className="w-4 h-4 mr-2" />}
            {isTraining ? "Stop" : "Train"}
          </Button>
          <Button onClick={resetTraining} variant="outline"><RotateCcw className="w-4 h-4" /></Button>
        </div>

        {isTraining && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Progress:</span>
              <span>{currentEpoch}/{config.epochs}</span>
            </div>
            <div className="w-full bg-muted rounded-full h-2">
              <div className="bg-primary h-2 rounded-full transition-all duration-300" style={{ width: `${(currentEpoch / config.epochs) * 100}%` }} />
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
