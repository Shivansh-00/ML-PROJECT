"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Switch } from "@/components/ui/switch"
import { Slider } from "@/components/ui/slider"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
} from "recharts"
import { Brain, BarChart3, Play, Zap, Target, Shield, Layers, GitBranch, Activity, Cpu, Database } from "lucide-react"

interface AdvancedModelConfig {
  architecture: "attention_mlp" | "residual" | "bayesian" | "ensemble" | "transformer"
  uncertainty_method: "mc_dropout" | "deep_ensemble" | "bayesian" | "quantile"
  feature_engineering: {
    polynomial_degree: number
    include_interactions: boolean
    include_regimes: boolean
    include_technical: boolean
  }
  ensemble_config: {
    n_models: number
    weighting_method: "equal" | "performance" | "diversity" | "optimal"
    use_stacking: boolean
  }
}

interface ModelPerformance {
  model_name: string
  rmse: number
  mae: number
  r2: number
  mape: number
  uncertainty_score: number
  calibration_error: number
  training_time: number
}

interface UncertaintyMetrics {
  coverage_95: number
  coverage_90: number
  coverage_68: number
  sharpness: number
  calibration_error: number
  expected_shortfall_95: number
}

export default function AdvancedMLDashboard() {
  const [config, setConfig] = useState<AdvancedModelConfig>({
    architecture: "attention_mlp",
    uncertainty_method: "mc_dropout",
    feature_engineering: {
      polynomial_degree: 2,
      include_interactions: true,
      include_regimes: true,
      include_technical: false,
    },
    ensemble_config: {
      n_models: 5,
      weighting_method: "diversity",
      use_stacking: true,
    },
  })

  const [isTraining, setIsTraining] = useState(false)
  const [trainingProgress, setTrainingProgress] = useState(0)
  const [modelPerformance, setModelPerformance] = useState<ModelPerformance[]>([])
  const [uncertaintyMetrics, setUncertaintyMetrics] = useState<UncertaintyMetrics | null>(null)
  const [featureImportance, setFeatureImportance] = useState<any[]>([])
  const [ensembleWeights, setEnsembleWeights] = useState<any[]>([])
  const [hedgingResults, setHedgingResults] = useState<any>(null)
  const [calibrationData, setCalibrationData] = useState<any[]>([])

  // Mock training function
  const trainAdvancedModel = async () => {
    setIsTraining(true)
    setTrainingProgress(0)

    // Simulate training progress
    for (let i = 0; i <= 100; i += 5) {
      await new Promise((resolve) => setTimeout(resolve, 100))
      setTrainingProgress(i)
    }

    // Mock performance results
    const mockPerformance: ModelPerformance[] = [
      {
        model_name: "Attention MLP",
        rmse: 0.0234,
        mae: 0.0187,
        r2: 0.9987,
        mape: 1.23,
        uncertainty_score: 0.95,
        calibration_error: 0.012,
        training_time: 45.2,
      },
      {
        model_name: "Residual Network",
        rmse: 0.0267,
        mae: 0.0201,
        r2: 0.9982,
        mape: 1.45,
        uncertainty_score: 0.92,
        calibration_error: 0.018,
        training_time: 52.1,
      },
      {
        model_name: "Bayesian NN",
        rmse: 0.0298,
        mae: 0.0223,
        r2: 0.9978,
        mape: 1.67,
        uncertainty_score: 0.98,
        calibration_error: 0.008,
        training_time: 78.5,
      },
      {
        model_name: "Deep Ensemble",
        rmse: 0.0189,
        mae: 0.0145,
        r2: 0.9992,
        mape: 0.98,
        uncertainty_score: 0.97,
        calibration_error: 0.006,
        training_time: 234.7,
      },
      {
        model_name: "Transformer",
        rmse: 0.0156,
        mae: 0.0123,
        r2: 0.9995,
        mape: 0.87,
        uncertainty_score: 0.96,
        calibration_error: 0.009,
        training_time: 156.3,
      },
    ]

    setModelPerformance(mockPerformance)

    // Mock uncertainty metrics
    setUncertaintyMetrics({
      coverage_95: 0.947,
      coverage_90: 0.896,
      coverage_68: 0.682,
      sharpness: 0.234,
      calibration_error: 0.008,
      expected_shortfall_95: 0.156,
    })

    // Mock feature importance
    setFeatureImportance([
      { feature: "Volatility (σ)", importance: 0.28, category: "Market" },
      { feature: "Time to Maturity", importance: 0.24, category: "Time" },
      { feature: "Moneyness", importance: 0.22, category: "Strike" },
      { feature: "Vol-Time Interaction", importance: 0.15, category: "Interaction" },
      { feature: "Regime Indicator", importance: 0.08, category: "Regime" },
      { feature: "Technical RSI", importance: 0.03, category: "Technical" },
    ])

    // Mock ensemble weights
    setEnsembleWeights([
      { model: "Attention MLP", weight: 0.25, performance: 0.9987 },
      { model: "Residual Net", weight: 0.18, performance: 0.9982 },
      { model: "Bayesian NN", weight: 0.15, performance: 0.9978 },
      { model: "Deep Ensemble", weight: 0.32, performance: 0.9992 },
      { model: "Transformer", weight: 0.1, performance: 0.9995 },
    ])

    // Mock calibration data
    setCalibrationData([
      { expected: 0.1, observed: 0.098 },
      { expected: 0.2, observed: 0.195 },
      { expected: 0.3, observed: 0.305 },
      { expected: 0.4, observed: 0.398 },
      { expected: 0.5, observed: 0.502 },
      { expected: 0.6, observed: 0.595 },
      { expected: 0.7, observed: 0.708 },
      { expected: 0.8, observed: 0.795 },
      { expected: 0.9, observed: 0.897 },
    ])

    // Mock hedging results
    setHedgingResults({
      rl_reward: -0.0234,
      delta_reward: -0.0456,
      rl_variance: 0.0012,
      delta_variance: 0.0034,
      rl_transaction_cost: 0.0089,
      delta_transaction_cost: 0.0156,
    })

    setIsTraining(false)
  }

  const formatNumber = (num: number, decimals = 4) => num.toFixed(decimals)
  const formatPercent = (num: number) => `${(num * 100).toFixed(2)}%`

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-blue-100 p-4">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold text-gray-900 flex items-center justify-center gap-2">
            <Brain className="h-8 w-8 text-purple-600" />
            Advanced ML Option Pricing
          </h1>
          <p className="text-lg text-gray-600">
            State-of-the-art deep learning with uncertainty quantification and ensemble methods
          </p>
        </div>

        <Tabs defaultValue="architecture" className="space-y-6">
          <TabsList className="grid w-full grid-cols-6">
            <TabsTrigger value="architecture" className="flex items-center gap-2">
              <Layers className="h-4 w-4" />
              Architecture
            </TabsTrigger>
            <TabsTrigger value="uncertainty" className="flex items-center gap-2">
              <Shield className="h-4 w-4" />
              Uncertainty
            </TabsTrigger>
            <TabsTrigger value="ensemble" className="flex items-center gap-2">
              <GitBranch className="h-4 w-4" />
              Ensemble
            </TabsTrigger>
            <TabsTrigger value="features" className="flex items-center gap-2">
              <Database className="h-4 w-4" />
              Features
            </TabsTrigger>
            <TabsTrigger value="performance" className="flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Performance
            </TabsTrigger>
            <TabsTrigger value="hedging" className="flex items-center gap-2">
              <Target className="h-4 w-4" />
              RL Hedging
            </TabsTrigger>
          </TabsList>

          <TabsContent value="architecture" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Architecture Configuration */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Cpu className="h-5 w-5" />
                    Neural Architecture
                  </CardTitle>
                  <CardDescription>Configure advanced neural network architectures</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label>Architecture Type</Label>
                    <Select
                      value={config.architecture}
                      onValueChange={(value: any) => setConfig({ ...config, architecture: value })}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="attention_mlp">Attention MLP</SelectItem>
                        <SelectItem value="residual">Residual Network</SelectItem>
                        <SelectItem value="bayesian">Bayesian Neural Network</SelectItem>
                        <SelectItem value="ensemble">Deep Ensemble</SelectItem>
                        <SelectItem value="transformer">Transformer Encoder</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label>Uncertainty Method</Label>
                    <Select
                      value={config.uncertainty_method}
                      onValueChange={(value: any) => setConfig({ ...config, uncertainty_method: value })}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="mc_dropout">Monte Carlo Dropout</SelectItem>
                        <SelectItem value="deep_ensemble">Deep Ensemble</SelectItem>
                        <SelectItem value="bayesian">Bayesian Inference</SelectItem>
                        <SelectItem value="quantile">Quantile Regression</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <Button onClick={trainAdvancedModel} disabled={isTraining} className="w-full">
                    {isTraining ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                        Training... {trainingProgress}%
                      </>
                    ) : (
                      <>
                        <Play className="h-4 w-4 mr-2" />
                        Train Advanced Models
                      </>
                    )}
                  </Button>

                  {isTraining && (
                    <div className="space-y-2">
                      <Progress value={trainingProgress} className="h-2" />
                      <p className="text-sm text-gray-600 text-center">
                        Training epoch {Math.floor(trainingProgress / 2)}/50
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Architecture Visualization */}
              <Card>
                <CardHeader>
                  <CardTitle>Architecture Details</CardTitle>
                  <CardDescription>Current model architecture specifications</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {config.architecture === "attention_mlp" && (
                      <div className="space-y-3">
                        <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg">
                          <span className="font-medium">Input Layer</span>
                          <Badge>256 features</Badge>
                        </div>
                        <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                          <span className="font-medium">Dense + BatchNorm</span>
                          <Badge>256 → 128 → 64</Badge>
                        </div>
                        <div className="flex items-center justify-between p-3 bg-purple-50 rounded-lg">
                          <span className="font-medium">Multi-Head Attention</span>
                          <Badge>8 heads, 16 key_dim</Badge>
                        </div>
                        <div className="flex items-center justify-between p-3 bg-orange-50 rounded-lg">
                          <span className="font-medium">Output Layer</span>
                          <Badge>1 neuron (linear)</Badge>
                        </div>
                      </div>
                    )}

                    {config.architecture === "bayesian" && (
                      <div className="space-y-3">
                        <div className="flex items-center justify-between p-3 bg-red-50 rounded-lg">
                          <span className="font-medium">Variational Layers</span>
                          <Badge>256 → 128 → 64</Badge>
                        </div>
                        <div className="flex items-center justify-between p-3 bg-yellow-50 rounded-lg">
                          <span className="font-medium">Prior Distribution</span>
                          <Badge>MultivariateNormal</Badge>
                        </div>
                        <div className="flex items-center justify-between p-3 bg-indigo-50 rounded-lg">
                          <span className="font-medium">KL Weight</span>
                          <Badge>1/1000</Badge>
                        </div>
                      </div>
                    )}

                    {config.architecture === "transformer" && (
                      <div className="space-y-3">
                        <div className="flex items-center justify-between p-3 bg-cyan-50 rounded-lg">
                          <span className="font-medium">Positional Encoding</span>
                          <Badge>Learned</Badge>
                        </div>
                        <div className="flex items-center justify-between p-3 bg-teal-50 rounded-lg">
                          <span className="font-medium">Transformer Blocks</span>
                          <Badge>3 layers</Badge>
                        </div>
                        <div className="flex items-center justify-between p-3 bg-emerald-50 rounded-lg">
                          <span className="font-medium">Attention Heads</span>
                          <Badge>8 heads, 64 key_dim</Badge>
                        </div>
                      </div>
                    )}

                    <div className="pt-4 border-t">
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <p className="text-gray-600">Parameters</p>
                          <p className="font-semibold">~2.1M</p>
                        </div>
                        <div>
                          <p className="text-gray-600">Memory</p>
                          <p className="font-semibold">~45MB</p>
                        </div>
                        <div>
                          <p className="text-gray-600">Training Time</p>
                          <p className="font-semibold">~2.5 min</p>
                        </div>
                        <div>
                          <p className="text-gray-600">Inference</p>
                          <p className="font-semibold">~0.8ms</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="uncertainty" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Uncertainty Metrics */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Shield className="h-5 w-5" />
                    Uncertainty Quantification
                  </CardTitle>
                  <CardDescription>Model uncertainty and calibration metrics</CardDescription>
                </CardHeader>
                <CardContent>
                  {uncertaintyMetrics ? (
                    <div className="space-y-4">
                      <div className="grid grid-cols-2 gap-4">
                        <div className="text-center p-3 bg-green-50 rounded-lg">
                          <p className="text-sm text-green-600">95% Coverage</p>
                          <p className="text-lg font-bold text-green-700">
                            {formatPercent(uncertaintyMetrics.coverage_95)}
                          </p>
                        </div>
                        <div className="text-center p-3 bg-blue-50 rounded-lg">
                          <p className="text-sm text-blue-600">90% Coverage</p>
                          <p className="text-lg font-bold text-blue-700">
                            {formatPercent(uncertaintyMetrics.coverage_90)}
                          </p>
                        </div>
                        <div className="text-center p-3 bg-purple-50 rounded-lg">
                          <p className="text-sm text-purple-600">68% Coverage</p>
                          <p className="text-lg font-bold text-purple-700">
                            {formatPercent(uncertaintyMetrics.coverage_68)}
                          </p>
                        </div>
                        <div className="text-center p-3 bg-orange-50 rounded-lg">
                          <p className="text-sm text-orange-600">Sharpness</p>
                          <p className="text-lg font-bold text-orange-700">
                            {formatNumber(uncertaintyMetrics.sharpness, 3)}
                          </p>
                        </div>
                      </div>

                      <div className="space-y-2">
                        <div className="flex justify-between items-center">
                          <span className="text-sm font-medium">Calibration Error</span>
                          <span className="text-sm text-gray-600">
                            {formatNumber(uncertaintyMetrics.calibration_error, 3)}
                          </span>
                        </div>
                        <Progress value={(1 - uncertaintyMetrics.calibration_error) * 100} className="h-2" />
                      </div>

                      <div className="space-y-2">
                        <div className="flex justify-between items-center">
                          <span className="text-sm font-medium">Expected Shortfall (95%)</span>
                          <span className="text-sm text-gray-600">
                            {formatNumber(uncertaintyMetrics.expected_shortfall_95, 3)}
                          </span>
                        </div>
                        <Progress value={(1 - uncertaintyMetrics.expected_shortfall_95) * 100} className="h-2" />
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-8 text-gray-500">
                      <Shield className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p>Train models to see uncertainty metrics</p>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Calibration Plot */}
              <Card>
                <CardHeader>
                  <CardTitle>Calibration Analysis</CardTitle>
                  <CardDescription>Model calibration and reliability assessment</CardDescription>
                </CardHeader>
                <CardContent>
                  {calibrationData.length > 0 ? (
                    <div className="h-80">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={calibrationData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis
                            dataKey="expected"
                            label={{ value: "Expected Frequency", position: "insideBottom", offset: -5 }}
                          />
                          <YAxis label={{ value: "Observed Frequency", angle: -90, position: "insideLeft" }} />
                          <Tooltip />
                          <Legend />
                          <Line type="monotone" dataKey="observed" stroke="#8884d8" strokeWidth={2} name="Observed" />
                          <Line
                            type="monotone"
                            dataKey="expected"
                            stroke="#ff7300"
                            strokeDasharray="5 5"
                            name="Perfect Calibration"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <div className="text-center py-12 text-gray-500">
                      <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p>Calibration data will appear after training</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="ensemble" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Ensemble Configuration */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <GitBranch className="h-5 w-5" />
                    Ensemble Configuration
                  </CardTitle>
                  <CardDescription>Configure ensemble methods and weighting strategies</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label>Number of Models</Label>
                    <Slider
                      value={[config.ensemble_config.n_models]}
                      onValueChange={(value) =>
                        setConfig({
                          ...config,
                          ensemble_config: { ...config.ensemble_config, n_models: value[0] },
                        })
                      }
                      max={10}
                      min={3}
                      step={1}
                      className="w-full"
                    />
                    <p className="text-sm text-gray-600">{config.ensemble_config.n_models} models</p>
                  </div>

                  <div className="space-y-2">
                    <Label>Weighting Method</Label>
                    <Select
                      value={config.ensemble_config.weighting_method}
                      onValueChange={(value: any) =>
                        setConfig({
                          ...config,
                          ensemble_config: { ...config.ensemble_config, weighting_method: value },
                        })
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="equal">Equal Weights</SelectItem>
                        <SelectItem value="performance">Performance Based</SelectItem>
                        <SelectItem value="diversity">Diversity Weighted</SelectItem>
                        <SelectItem value="optimal">Optimal (Constrained)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="flex items-center space-x-2">
                    <Switch
                      id="stacking"
                      checked={config.ensemble_config.use_stacking}
                      onCheckedChange={(checked) =>
                        setConfig({
                          ...config,
                          ensemble_config: { ...config.ensemble_config, use_stacking: checked },
                        })
                      }
                    />
                    <Label htmlFor="stacking">Use Stacking Meta-Learner</Label>
                  </div>

                  <div className="p-3 bg-blue-50 rounded-lg">
                    <h4 className="font-medium text-blue-900 mb-2">Ensemble Strategy</h4>
                    <p className="text-sm text-blue-700">
                      {config.ensemble_config.use_stacking
                        ? "Using stacking with meta-learner for optimal combination"
                        : `Using ${config.ensemble_config.weighting_method} weighting for model combination`}
                    </p>
                  </div>
                </CardContent>
              </Card>

              {/* Ensemble Weights */}
              <Card>
                <CardHeader>
                  <CardTitle>Model Weights & Diversity</CardTitle>
                  <CardDescription>Individual model contributions to ensemble</CardDescription>
                </CardHeader>
                <CardContent>
                  {ensembleWeights.length > 0 ? (
                    <div className="space-y-4">
                      <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={ensembleWeights}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="model" angle={-45} textAnchor="end" height={80} />
                            <YAxis />
                            <Tooltip />
                            <Bar dataKey="weight" fill="#8884d8" name="Ensemble Weight" />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>

                      <div className="space-y-2">
                        <h4 className="font-medium">Diversity Metrics</h4>
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <p className="text-gray-600">Avg Correlation</p>
                            <p className="font-semibold">0.234</p>
                          </div>
                          <div>
                            <p className="text-gray-600">Diversity Index</p>
                            <p className="font-semibold">0.766</p>
                          </div>
                          <div>
                            <p className="text-gray-600">Disagreement</p>
                            <p className="font-semibold">0.0156</p>
                          </div>
                          <div>
                            <p className="text-gray-600">Q-Statistic</p>
                            <p className="font-semibold">0.123</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-12 text-gray-500">
                      <GitBranch className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p>Train ensemble to see weights and diversity</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="features" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Feature Engineering */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Database className="h-5 w-5" />
                    Feature Engineering
                  </CardTitle>
                  <CardDescription>Advanced feature engineering configuration</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label>Polynomial Degree</Label>
                    <Slider
                      value={[config.feature_engineering.polynomial_degree]}
                      onValueChange={(value) =>
                        setConfig({
                          ...config,
                          feature_engineering: { ...config.feature_engineering, polynomial_degree: value[0] },
                        })
                      }
                      max={4}
                      min={1}
                      step={1}
                      className="w-full"
                    />
                    <p className="text-sm text-gray-600">Degree {config.feature_engineering.polynomial_degree}</p>
                  </div>

                  <div className="space-y-3">
                    <div className="flex items-center space-x-2">
                      <Switch
                        id="interactions"
                        checked={config.feature_engineering.include_interactions}
                        onCheckedChange={(checked) =>
                          setConfig({
                            ...config,
                            feature_engineering: { ...config.feature_engineering, include_interactions: checked },
                          })
                        }
                      />
                      <Label htmlFor="interactions">Include Interaction Features</Label>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Switch
                        id="regimes"
                        checked={config.feature_engineering.include_regimes}
                        onCheckedChange={(checked) =>
                          setConfig({
                            ...config,
                            feature_engineering: { ...config.feature_engineering, include_regimes: checked },
                          })
                        }
                      />
                      <Label htmlFor="regimes">Include Market Regime Features</Label>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Switch
                        id="technical"
                        checked={config.feature_engineering.include_technical}
                        onCheckedChange={(checked) =>
                          setConfig({
                            ...config,
                            feature_engineering: { ...config.feature_engineering, include_technical: checked },
                          })
                        }
                      />
                      <Label htmlFor="technical">Include Technical Indicators</Label>
                    </div>
                  </div>

                  <div className="p-3 bg-green-50 rounded-lg">
                    <h4 className="font-medium text-green-900 mb-2">Feature Summary</h4>
                    <div className="text-sm text-green-700 space-y-1">
                      <p>• Base features: 8</p>
                      <p>• Market features: 25</p>
                      <p>• Polynomial features: {config.feature_engineering.polynomial_degree * 4}</p>
                      <p>• Interaction features: {config.feature_engineering.include_interactions ? 12 : 0}</p>
                      <p>• Regime features: {config.feature_engineering.include_regimes ? 15 : 0}</p>
                      <p>• Technical features: {config.feature_engineering.include_technical ? 8 : 0}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Feature Importance */}
              <Card>
                <CardHeader>
                  <CardTitle>Feature Importance Analysis</CardTitle>
                  <CardDescription>Most influential features for option pricing</CardDescription>
                </CardHeader>
                <CardContent>
                  {featureImportance.length > 0 ? (
                    <div className="space-y-4">
                      <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={featureImportance} layout="horizontal">
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis type="number" />
                            <YAxis dataKey="feature" type="category" width={120} />
                            <Tooltip />
                            <Bar dataKey="importance" fill="#8884d8" />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>

                      <div className="space-y-2">
                        <h4 className="font-medium">Feature Categories</h4>
                        <div className="flex flex-wrap gap-2">
                          {Array.from(new Set(featureImportance.map((f) => f.category))).map((category) => (
                            <Badge key={category} variant="outline">
                              {category}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-12 text-gray-500">
                      <Database className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p>Feature importance will appear after training</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="performance" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="h-5 w-5" />
                  Model Performance Comparison
                </CardTitle>
                <CardDescription>Comprehensive evaluation of all trained models</CardDescription>
              </CardHeader>
              <CardContent>
                {modelPerformance.length > 0 ? (
                  <div className="space-y-6">
                    {/* Performance Table */}
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b">
                            <th className="text-left p-2">Model</th>
                            <th className="text-right p-2">RMSE</th>
                            <th className="text-right p-2">MAE</th>
                            <th className="text-right p-2">R²</th>
                            <th className="text-right p-2">MAPE</th>
                            <th className="text-right p-2">Uncertainty</th>
                            <th className="text-right p-2">Calibration</th>
                            <th className="text-right p-2">Time (s)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {modelPerformance.map((model, index) => (
                            <tr key={index} className="border-b hover:bg-gray-50">
                              <td className="p-2 font-medium">{model.model_name}</td>
                              <td className="p-2 text-right">{formatNumber(model.rmse)}</td>
                              <td className="p-2 text-right">{formatNumber(model.mae)}</td>
                              <td className="p-2 text-right">{formatNumber(model.r2)}</td>
                              <td className="p-2 text-right">{formatNumber(model.mape, 2)}%</td>
                              <td className="p-2 text-right">{formatNumber(model.uncertainty_score, 2)}</td>
                              <td className="p-2 text-right">{formatNumber(model.calibration_error, 3)}</td>
                              <td className="p-2 text-right">{formatNumber(model.training_time, 1)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    {/* Performance Charts */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                      <div className="h-64">
                        <h4 className="font-medium mb-2">R² Score Comparison</h4>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={modelPerformance}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="model_name" angle={-45} textAnchor="end" height={80} />
                            <YAxis domain={[0.995, 1]} />
                            <Tooltip />
                            <Bar dataKey="r2" fill="#8884d8" />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>

                      <div className="h-64">
                        <h4 className="font-medium mb-2">RMSE vs Training Time</h4>
                        <ResponsiveContainer width="100%" height="100%">
                          <ScatterChart data={modelPerformance}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="training_time" name="Training Time (s)" />
                            <YAxis dataKey="rmse" name="RMSE" />
                            <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                            <Scatter dataKey="rmse" fill="#8884d8" />
                          </ScatterChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    {/* Best Model Highlight */}
                    <div className="p-4 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg">
                      <h4 className="font-medium text-green-900 mb-2">🏆 Best Performing Model</h4>
                      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
                        <div>
                          <p className="text-gray-600">Model</p>
                          <p className="font-semibold">Transformer</p>
                        </div>
                        <div>
                          <p className="text-gray-600">R² Score</p>
                          <p className="font-semibold">0.9995</p>
                        </div>
                        <div>
                          <p className="text-gray-600">RMSE</p>
                          <p className="font-semibold">0.0156</p>
                        </div>
                        <div>
                          <p className="text-gray-600">Uncertainty Score</p>
                          <p className="font-semibold">0.96</p>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12 text-gray-500">
                    <Activity className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>Train models to see performance comparison</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="hedging" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* RL Hedging Configuration */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Target className="h-5 w-5" />
                    Reinforcement Learning Hedging
                  </CardTitle>
                  <CardDescription>Advanced hedging strategies using deep RL</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <div className="p-3 bg-blue-50 rounded-lg">
                      <h4 className="font-medium text-blue-900 mb-2">PPO Agent Configuration</h4>
                      <div className="text-sm text-blue-700 space-y-1">
                        <p>• Actor-Critic Architecture</p>
                        <p>• Continuous Action Space</p>
                        <p>• Transaction Cost Aware</p>
                        <p>• Risk-Adjusted Rewards</p>
                      </div>
                    </div>

                    <div className="p-3 bg-purple-50 rounded-lg">
                      <h4 className="font-medium text-purple-900 mb-2">Environment Features</h4>
                      <div className="text-sm text-purple-700 space-y-1">
                        <p>• Geometric Brownian Motion</p>
                        <p>• Real-time Greeks Calculation</p>
                        <p>• Portfolio Value Tracking</p>
                        <p>• Dynamic Hedge Ratio Adjustment</p>
                      </div>
                    </div>

                    <Button className="w-full bg-transparent" variant="outline">
                      <Zap className="h-4 w-4 mr-2" />
                      Train RL Hedging Agent
                    </Button>
                  </div>
                </CardContent>
              </Card>

              {/* Hedging Results */}
              <Card>
                <CardHeader>
                  <CardTitle>Hedging Performance</CardTitle>
                  <CardDescription>RL agent vs traditional delta hedging</CardDescription>
                </CardHeader>
                <CardContent>
                  {hedgingResults ? (
                    <div className="space-y-4">
                      <div className="grid grid-cols-2 gap-4">
                        <div className="text-center p-3 bg-green-50 rounded-lg">
                          <p className="text-sm text-green-600">RL Agent Reward</p>
                          <p className="text-lg font-bold text-green-700">{formatNumber(hedgingResults.rl_reward)}</p>
                        </div>
                        <div className="text-center p-3 bg-blue-50 rounded-lg">
                          <p className="text-sm text-blue-600">Delta Hedging</p>
                          <p className="text-lg font-bold text-blue-700">{formatNumber(hedgingResults.delta_reward)}</p>
                        </div>
                      </div>

                      <div className="space-y-3">
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <span className="text-sm font-medium">PnL Variance Reduction</span>
                            <span className="text-sm text-gray-600">
                              {formatPercent(1 - hedgingResults.rl_variance / hedgingResults.delta_variance)}
                            </span>
                          </div>
                          <Progress
                            value={(1 - hedgingResults.rl_variance / hedgingResults.delta_variance) * 100}
                            className="h-2"
                          />
                        </div>

                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <span className="text-sm font-medium">Transaction Cost Reduction</span>
                            <span className="text-sm text-gray-600">
                              {formatPercent(
                                1 - hedgingResults.rl_transaction_cost / hedgingResults.delta_transaction_cost,
                              )}
                            </span>
                          </div>
                          <Progress
                            value={
                              (1 - hedgingResults.rl_transaction_cost / hedgingResults.delta_transaction_cost) * 100
                            }
                            className="h-2"
                          />
                        </div>
                      </div>

                      <div className="p-3 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg">
                        <h4 className="font-medium text-green-900 mb-2">🎯 RL Advantages</h4>
                        <div className="text-sm text-green-700 space-y-1">
                          <p>• 65% lower PnL variance</p>
                          <p>• 43% reduced transaction costs</p>
                          <p>• Adaptive to market conditions</p>
                          <p>• Risk-aware optimization</p>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-12 text-gray-500">
                      <Target className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p>Train RL agent to see hedging results</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
