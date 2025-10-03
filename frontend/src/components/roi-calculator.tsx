'use client'

import React, { useState, useEffect } from 'react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Separator } from '@/components/ui/separator'
import {
  Calculator,
  DollarSign,
  TrendingUp,
  Users,
  Zap,
  AlertTriangle,
} from 'lucide-react'

interface ROIInputs {
  teamSize: number
  projectCount: number
  avgTrainingHours: number
  platformTier: 'starter' | 'professional' | 'enterprise'
}

interface ROIResults {
  monthlyWastePerProject: number
  annualWastePerProject: number
  totalAnnualWaste: number
  platformCost: number
  annualSavings: number
  roi: number
  breakEvenMonths: number
  fiveYearValue: number
}

interface PlatformPricing {
  starter: number
  professional: number
  enterprise: number
}

const PLATFORM_PRICING: PlatformPricing = {
  starter: 25000,
  professional: 35000,
  enterprise: 50000,
}

const COST_PATTERNS = {
  small_batch_gpu_training: 600,
  memory_leak_training: 1800,
  inefficient_data_loading: 300,
  oversized_model_config: 900,
}

const MONTHLY_WASTE_PER_PROJECT = Object.values(COST_PATTERNS).reduce(
  (sum, cost) => sum + cost,
  0
) // $3,600

export function ROICalculator() {
  const [inputs, setInputs] = useState<ROIInputs>({
    teamSize: 15,
    projectCount: 8,
    avgTrainingHours: 8,
    platformTier: 'professional',
  })

  const [results, setResults] = useState<ROIResults | null>(null)

  useEffect(() => {
    calculateROI()
  }, [inputs])

  const calculateROI = () => {
    const { teamSize, projectCount, avgTrainingHours, platformTier } = inputs

    // Base calculations from ROI methodology
    const monthlyWastePerProject = MONTHLY_WASTE_PER_PROJECT
    const annualWastePerProject = monthlyWastePerProject * 12
    const totalAnnualWaste = annualWastePerProject * projectCount

    // Platform cost based on tier
    const platformCost = PLATFORM_PRICING[platformTier]

    // Conservative assumption: detect and fix 80% of waste
    const detectionRate = 0.8
    const annualSavings = totalAnnualWaste * detectionRate

    // ROI calculation
    const netSavings = annualSavings - platformCost
    const roi = (netSavings / platformCost) * 100

    // Break-even calculation
    const breakEvenMonths = platformCost / (annualSavings / 12)

    // 5-year value calculation
    const fiveYearSavings = annualSavings * 5
    const fiveYearCosts = platformCost * 5 // Assuming annual licensing
    const fiveYearValue = fiveYearSavings - fiveYearCosts

    setResults({
      monthlyWastePerProject,
      annualWastePerProject,
      totalAnnualWaste,
      platformCost,
      annualSavings,
      roi,
      breakEvenMonths,
      fiveYearValue,
    })
  }

  const handleInputChange = (field: keyof ROIInputs, value: string | number) => {
    setInputs(prev => ({
      ...prev,
      [field]: typeof value === 'string' ? parseInt(value) || 0 : value,
    }))
  }

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount)
  }

  const getROIBadge = (roi: number) => {
    if (roi > 1000) return <Badge className="bg-green-600">Exceptional ROI</Badge>
    if (roi > 500) return <Badge className="bg-green-500">Excellent ROI</Badge>
    if (roi > 200) return <Badge className="bg-blue-500">Great ROI</Badge>
    if (roi > 100) return <Badge className="bg-yellow-500">Good ROI</Badge>
    return <Badge variant="secondary">Needs Review</Badge>
  }

  return (
    <div className="w-full max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="text-center space-y-2">
        <div className="flex items-center justify-center gap-2">
          <Calculator className="h-8 w-8 text-blue-600" />
          <h1 className="text-3xl font-bold text-gray-900">ROI Calculator</h1>
        </div>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Calculate your return on investment with Attrahere's AI Governance Platform.
          See how much your team can save by eliminating ML infrastructure waste.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Panel */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="h-5 w-5" />
              Your Team Configuration
            </CardTitle>
            <CardDescription>
              Enter your team details to calculate potential savings
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="teamSize">Team Size (ML Engineers)</Label>
              <Input
                id="teamSize"
                type="number"
                value={inputs.teamSize}
                onChange={e => handleInputChange('teamSize', e.target.value)}
                min="1"
                max="100"
              />
              <p className="text-sm text-gray-500">
                Number of ML engineers on your team
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="projectCount">Active ML Projects</Label>
              <Input
                id="projectCount"
                type="number"
                value={inputs.projectCount}
                onChange={e => handleInputChange('projectCount', e.target.value)}
                min="1"
                max="50"
              />
              <p className="text-sm text-gray-500">
                Number of concurrent ML training projects
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="avgTrainingHours">Training Hours/Day</Label>
              <Input
                id="avgTrainingHours"
                type="number"
                value={inputs.avgTrainingHours}
                onChange={e => handleInputChange('avgTrainingHours', e.target.value)}
                min="1"
                max="24"
              />
              <p className="text-sm text-gray-500">
                Average GPU training hours per day
              </p>
            </div>

            <div className="space-y-2">
              <Label>Platform Tier</Label>
              <div className="grid grid-cols-1 gap-2">
                {Object.entries(PLATFORM_PRICING).map(([tier, price]) => (
                  <button
                    key={tier}
                    onClick={() => handleInputChange('platformTier', tier)}
                    className={`p-3 border rounded-lg text-left transition-colors ${
                      inputs.platformTier === tier
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="flex justify-between items-center">
                      <div>
                        <div className="font-medium capitalize">{tier}</div>
                        <div className="text-sm text-gray-500">
                          {tier === 'starter' && 'Up to 5 team members'}
                          {tier === 'professional' && 'Up to 20 team members'}
                          {tier === 'enterprise' && 'Unlimited team members'}
                        </div>
                      </div>
                      <div className="font-bold text-blue-600">
                        {formatCurrency(price)}/year
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Results Panel */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              ROI Analysis
            </CardTitle>
            <CardDescription>
              Your potential savings and return on investment
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {results && (
              <>
                {/* ROI Badge */}
                <div className="text-center">
                  <div className="text-4xl font-bold text-green-600 mb-2">
                    {results.roi.toFixed(0)}%
                  </div>
                  <div className="mb-3">{getROIBadge(results.roi)}</div>
                  <p className="text-sm text-gray-600">Annual Return on Investment</p>
                </div>

                <Separator />

                {/* Key Metrics */}
                <div className="grid grid-cols-1 gap-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">
                      Annual Waste (Current)
                    </span>
                    <span className="font-bold text-red-600">
                      {formatCurrency(results.totalAnnualWaste)}
                    </span>
                  </div>

                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Platform Cost</span>
                    <span className="font-bold text-blue-600">
                      {formatCurrency(results.platformCost)}
                    </span>
                  </div>

                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Annual Savings</span>
                    <span className="font-bold text-green-600">
                      {formatCurrency(results.annualSavings)}
                    </span>
                  </div>

                  <Separator />

                  <div className="flex justify-between items-center">
                    <span className="font-medium">Net Annual Benefit</span>
                    <span className="font-bold text-green-700 text-lg">
                      {formatCurrency(results.annualSavings - results.platformCost)}
                    </span>
                  </div>
                </div>

                {/* Break-even Timeline */}
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Break-even</span>
                    <span className="font-bold">
                      {results.breakEvenMonths.toFixed(1)} months
                    </span>
                  </div>
                  <Progress
                    value={Math.min((12 / results.breakEvenMonths) * 100, 100)}
                    className="h-2"
                  />
                  <p className="text-xs text-gray-500">
                    Time to recover platform investment
                  </p>
                </div>

                {/* 5-Year Value */}
                <div className="bg-green-50 p-4 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <Zap className="h-4 w-4 text-green-600" />
                    <span className="font-medium text-green-800">5-Year Value</span>
                  </div>
                  <div className="text-2xl font-bold text-green-700">
                    {formatCurrency(results.fiveYearValue)}
                  </div>
                  <p className="text-sm text-green-600">
                    Total value over 5 years after platform costs
                  </p>
                </div>
              </>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Cost Breakdown */}
      {results && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <DollarSign className="h-5 w-5" />
              Cost Waste Breakdown
            </CardTitle>
            <CardDescription>
              Monthly waste per project by inefficiency type
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {Object.entries(COST_PATTERNS).map(([pattern, cost]) => (
                <div key={pattern} className="p-4 border rounded-lg">
                  <div className="text-sm text-gray-600 mb-1">
                    {pattern
                      .split('_')
                      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                      .join(' ')}
                  </div>
                  <div className="text-lg font-bold text-red-600">
                    {formatCurrency(cost)}/mo
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-6 p-4 bg-yellow-50 rounded-lg">
              <div className="flex items-start gap-2">
                <AlertTriangle className="h-5 w-5 text-yellow-600 mt-0.5" />
                <div>
                  <div className="font-medium text-yellow-800">
                    Conservative Estimates
                  </div>
                  <p className="text-sm text-yellow-700 mt-1">
                    These calculations are based on conservative estimates and real AWS
                    GPU pricing. Actual savings may be higher depending on your specific
                    infrastructure usage patterns.
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* CTA Section */}
      <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-200">
        <CardContent className="p-6 text-center">
          <h3 className="text-xl font-bold text-gray-900 mb-2">
            Ready to eliminate ML infrastructure waste?
          </h3>
          <p className="text-gray-600 mb-4">
            Start your free trial and see exactly how much your team can save
          </p>
          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <Button size="lg" className="bg-blue-600 hover:bg-blue-700">
              Start Free Trial
            </Button>
            <Button variant="outline" size="lg">
              Schedule Demo
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
