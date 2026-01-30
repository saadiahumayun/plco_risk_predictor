import React, { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import axios from 'axios'
import type { PredictionResponse, PredictionRequest } from '../services/api'

// API URL - same as in api.ts
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1'

interface TestCase {
  test_id: string
  age: number
  race: string
  education_level: number
  marital_status: string
  occupation: number
  age_at_menarche: number
  age_at_first_birth: number
  number_of_live_births: number
  number_of_relatives_with_bc: number
  current_bmi: number
  bmi_at_20: number
  bmi_at_50: number
  personal_history_cancer: boolean
  benign_breast_disease: boolean
  hormone_therapy: boolean
  years_of_hormone_use: number
  pack_years_smoking: number
  birth_control_years: number
  aspirin_use: boolean
  ibuprofen_use: boolean
  family_history_cancer: boolean
  expected_risk_category: string
}

// Transform test case to API request format
function transformTestCaseToRequest(testCase: TestCase): PredictionRequest {
  return {
    patient_id: testCase.test_id,
    demographics: {
      age: Math.round(testCase.age),
      race: testCase.race,
      education: testCase.education_level?.toString(),
      marital_status: testCase.marital_status,
      occupation: testCase.occupation?.toString(),
    },
    reproductive_history: {
      age_at_menarche: Math.round(testCase.age_at_menarche),
      age_at_first_birth: testCase.age_at_first_birth ? Math.round(testCase.age_at_first_birth) : undefined,
      number_of_live_births: Math.round(testCase.number_of_live_births || 0),
      first_degree_bc: Math.round(testCase.number_of_relatives_with_bc || 0),
    },
    body_metrics: {
      current_bmi: testCase.current_bmi,
      bmi_at_age_20: testCase.bmi_at_20 || undefined,
      bmi_at_age_50: testCase.bmi_at_50 || undefined,
      height_cm: 165, // Default
      weight_kg: testCase.current_bmi * (1.65 * 1.65), // Calculate from BMI
    },
    medical_history: {
      personal_cancer_history: testCase.personal_history_cancer || false,
      benign_breast_disease: testCase.benign_breast_disease || false,
      breast_biopsies: testCase.benign_breast_disease ? 1 : 0,
      hormone_therapy_current: testCase.hormone_therapy || false,
      hormone_therapy_years: testCase.years_of_hormone_use ? Math.round(testCase.years_of_hormone_use) : undefined,
      aspirin_use: testCase.aspirin_use || false,
      ibuprofen_use: testCase.ibuprofen_use || false,
    },
    lifestyle: {
      smoking_status: testCase.pack_years_smoking > 0 ? 'former' : 'never',
      pack_years: testCase.pack_years_smoking || undefined,
      birth_control_years: testCase.birth_control_years || undefined,
    },
  }
}

// Direct API call for test cases
async function predictForTestCase(testCase: TestCase): Promise<PredictionResponse> {
  const request = transformTestCaseToRequest(testCase)
  const response = await axios.post<PredictionResponse>(`${API_BASE_URL}/predict`, request, {
    headers: { 'Content-Type': 'application/json' }
  })
  return response.data
}

interface PredictionResult {
  test_id: string
  expected: string
  predicted: string
  risk_score: number
  correct: boolean
  processing_time: number
}

const COLORS = {
  primary: '#6D2842',
  primaryLight: '#8B4A5E',
  success: '#10b981',
  warning: '#f59e0b',
  danger: '#ef4444',
  neutral: '#6b7280'
}

// Check if prediction is correct
// For "low" expected: both "low" and "moderate" are acceptable (probability < 0.5)
// For "high" expected: only "high" is acceptable
function isPredictionCorrect(expected: string, predicted: string): boolean {
  if (expected === 'low') {
    return predicted === 'low' || predicted === 'moderate'
  }
  if (expected === 'moderate') {
    return predicted === 'low' || predicted === 'moderate' || predicted === 'high'
  }
  // For "high" expected, only "high" is correct
  return expected === predicted
}

const ModelValidation: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview')
  const [testCases, setTestCases] = useState<TestCase[]>([])
  const [results, setResults] = useState<PredictionResult[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [errorMessage, setErrorMessage] = useState('')
  const [metrics, setMetrics] = useState({
    accuracy: 0,
    sensitivity: 0,
    specificity: 0,
    precision: 0,
    totalCases: 0,
    correctPredictions: 0,
    avgProcessingTime: 0
  })

  // Load test cases on mount
  useEffect(() => {
    loadTestCases()
  }, [])

  // Calculate metrics when results change
  useEffect(() => {
    if (results.length > 0) {
      calculateMetrics()
    }
  }, [results])

  const loadTestCases = async () => {
    setIsLoading(true)
    try {
      const response = await fetch('/test_cases.json')
      const data = await response.json()
      setTestCases(data)
    } catch (error) {
      console.error('Failed to load test cases:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const runValidation = async () => {
    setIsRunning(true)
    setResults([])
    setProgress(0)
    setErrorMessage('')
    
    const newResults: PredictionResult[] = []
    let consecutiveErrors = 0
    
    for (let i = 0; i < testCases.length; i++) {
      const testCase = testCases[i]
      const startTime = performance.now()
      
      try {
        const response = await predictForTestCase(testCase)
        const endTime = performance.now()
        consecutiveErrors = 0 // Reset on success
        
        newResults.push({
          test_id: testCase.test_id,
          expected: testCase.expected_risk_category,
          predicted: response.risk_category,
          risk_score: response.risk_score,
          correct: isPredictionCorrect(testCase.expected_risk_category, response.risk_category),
          processing_time: endTime - startTime
        })
      } catch (error: any) {
        consecutiveErrors++
        const endTime = performance.now()
        
        // Extract error message
        let errorMsg = 'Unknown error'
        if (axios.isAxiosError(error)) {
          if (error.response) {
            errorMsg = `API Error ${error.response.status}: ${JSON.stringify(error.response.data)}`
          } else if (error.request) {
            errorMsg = 'No response from server - is the backend running?'
          } else {
            errorMsg = error.message
          }
        } else if (error instanceof Error) {
          errorMsg = error.message
        }
        
        console.error(`Failed to predict for ${testCase.test_id}:`, errorMsg)
        
        // If we get 5 consecutive errors, show a message and stop
        if (consecutiveErrors >= 5 && i < 10) {
          setErrorMessage(`API connection failed: ${errorMsg}. Make sure the backend is running.`)
          break
        }
        
        newResults.push({
          test_id: testCase.test_id,
          expected: testCase.expected_risk_category,
          predicted: 'error',
          risk_score: 0,
          correct: false,
          processing_time: endTime - startTime
        })
      }
      
      setProgress(((i + 1) / testCases.length) * 100)
      setResults([...newResults])
    }
    
    setIsRunning(false)
  }

  const calculateMetrics = () => {
    const total = results.length
    const correct = results.filter(r => r.correct).length
    
    // Binary classification: high risk vs non-high risk (low/moderate)
    // For "low" expected: low or moderate predictions are acceptable
    // For "high" expected: only high predictions are acceptable
    const truePositives = results.filter(r => r.expected === 'high' && r.predicted === 'high').length
    const falseNegatives = results.filter(r => r.expected === 'high' && r.predicted !== 'high').length
    const trueNegatives = results.filter(r => r.expected !== 'high' && r.predicted !== 'high').length
    const falsePositives = results.filter(r => r.expected !== 'high' && r.predicted === 'high').length
    
    const sensitivity = truePositives / (truePositives + falseNegatives) || 0
    const specificity = trueNegatives / (trueNegatives + falsePositives) || 1
    const precision = truePositives / (truePositives + falsePositives) || 0
    
    const avgTime = results.reduce((acc, r) => acc + r.processing_time, 0) / total
    
    setMetrics({
      accuracy: correct / total,
      sensitivity,
      specificity,
      precision,
      totalCases: total,
      correctPredictions: correct,
      avgProcessingTime: avgTime
    })
  }

  const MetricCard = ({ title, value, subtitle, color = 'primary', isPercent = true }: {
    title: string
    value: number
    subtitle?: string
    color?: keyof typeof COLORS
    isPercent?: boolean
  }) => (
    <div className="bg-white rounded-lg shadow p-6 border-l-4" style={{ borderLeftColor: COLORS[color] }}>
      <div className="text-sm font-medium text-gray-500 uppercase tracking-wide">{title}</div>
      <div className="mt-2 flex items-baseline">
        <div className="text-3xl font-bold" style={{ color: COLORS[color] }}>
          {isPercent ? `${(value * 100).toFixed(1)}%` : value.toFixed(0)}
        </div>
      </div>
      {subtitle && <div className="mt-1 text-sm text-gray-600">{subtitle}</div>}
    </div>
  )

  // Prepare chart data
  const riskDistribution = results.reduce((acc, r) => {
    const category = r.predicted
    acc[category] = (acc[category] || 0) + 1
    return acc
  }, {} as Record<string, number>)

  const pieData = Object.entries(riskDistribution).map(([name, value]) => ({ name, value }))

  const processingTimeData = results.map(r => ({
    test_id: r.test_id.replace('SYNTH-', ''),
    time: r.processing_time
  }))

  // Calculate expected category distribution
  const expectedDistribution = testCases.reduce((acc, tc) => {
    acc[tc.expected_risk_category] = (acc[tc.expected_risk_category] || 0) + 1
    return acc
  }, {} as Record<string, number>)
  
  const expectedSummary = Object.entries(expectedDistribution)
    .map(([cat, count]) => `${count} ${cat}`)
    .join(', ') || 'Mixed'

  // Overview Tab
  const OverviewTab = () => (
    <div className="space-y-6">
      {/* Error Message */}
      {errorMessage && (
        <div className="rounded-lg p-4 bg-red-50 border border-red-200 text-red-800">
          <div className="flex items-center gap-2 font-semibold">
            <span>⚠️</span> Connection Error
          </div>
          <p className="mt-1 text-sm">{errorMessage}</p>
          <p className="mt-2 text-sm">
            Make sure the backend is running at <code className="bg-red-100 px-1 rounded">{API_BASE_URL}</code>
          </p>
        </div>
      )}

      {/* Status Banner */}
      <div className={`rounded-lg p-6 border ${
        results.length === 0 
          ? 'bg-gray-50 border-gray-200' 
          : metrics.accuracy >= 0.7 
            ? 'bg-green-50 border-green-200' 
            : 'bg-orange-50 border-orange-200'
      }`}>
        <h3 className="text-xl font-semibold text-gray-800 mb-2">
          {results.length === 0 ? 'Ready to Validate' : 'Validation Complete'}
        </h3>
        <p className="text-gray-700">
          {results.length === 0 
            ? `${testCases.length} test cases loaded and ready for validation against the live model.`
            : `Validated ${results.length} cases with ${metrics.correctPredictions} correct predictions (${(metrics.accuracy * 100).toFixed(1)}% accuracy).`
          }
        </p>
        {results.length > 0 && (
          <div className="mt-4 flex flex-wrap items-center gap-2 text-sm">
            <span className={`px-3 py-1 rounded-full font-medium ${
              metrics.accuracy >= 0.7 ? 'bg-green-100 text-green-800' : 'bg-orange-100 text-orange-800'
            }`}>
              {metrics.accuracy >= 0.7 ? '✓ Model Performing Well' : '⚠ Needs Review'}
            </span>
            <span className="bg-rose-100 text-rose-800 px-3 py-1 rounded-full font-medium">
              {testCases.length} Test Cases
            </span>
            <span className="bg-purple-100 text-purple-800 px-3 py-1 rounded-full font-medium">
              Avg {metrics.avgProcessingTime.toFixed(0)}ms Response
            </span>
          </div>
        )}
      </div>

      {/* Run Validation Button */}
      {results.length === 0 && (
        <button
          onClick={runValidation}
          disabled={isRunning || testCases.length === 0}
          className="w-full btn-primary py-4 text-lg flex items-center justify-center gap-3"
        >
          {isRunning ? (
            <>
              <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              Running Validation... {progress.toFixed(0)}%
            </>
          ) : (
            <>
              <span className="text-xl">🧪</span>
              Run Model Validation on {testCases.length} Cases
            </>
          )}
        </button>
      )}

      {/* Progress Bar */}
      {isRunning && (
        <div className="w-full bg-gray-200 rounded-full h-3">
          <div 
            className="bg-primary h-3 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
      )}

      {/* Metrics Grid */}
      {results.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            title="Overall Accuracy"
            value={metrics.accuracy}
            subtitle={`${metrics.correctPredictions}/${metrics.totalCases} correct`}
            color="primary"
          />
          <MetricCard
            title="Sensitivity"
            value={metrics.sensitivity}
            subtitle="High-risk detection rate"
            color="success"
          />
          <MetricCard
            title="Precision"
            value={metrics.precision}
            subtitle="Positive predictive value"
            color="warning"
          />
          <MetricCard
            title="Avg Response Time"
            value={metrics.avgProcessingTime}
            subtitle="milliseconds"
            color="neutral"
            isPercent={false}
          />
        </div>
      )}

      {/* Re-run Button */}
      {results.length > 0 && (
        <button
          onClick={runValidation}
          disabled={isRunning}
          className="btn-secondary flex items-center gap-2"
        >
          <span>🔄</span> Re-run Validation
        </button>
      )}
    </div>
  )

  // Results Tab
  const ResultsTab = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="p-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-800">Individual Test Results</h3>
          <p className="text-sm text-gray-600">Detailed results for each test case</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Test ID</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Expected</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Predicted</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Risk Score</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Time</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {results.map((result) => (
                <tr key={result.test_id} className={result.correct ? '' : 'bg-red-50'}>
                  <td className="px-4 py-3 text-sm font-mono">{result.test_id}</td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      result.expected === 'high' ? 'bg-red-100 text-red-800' :
                      result.expected === 'moderate' ? 'bg-orange-100 text-orange-800' :
                      'bg-green-100 text-green-800'
                    }`}>
                      {result.expected}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      result.predicted === 'high' ? 'bg-red-100 text-red-800' :
                      result.predicted === 'moderate' ? 'bg-orange-100 text-orange-800' :
                      result.predicted === 'error' ? 'bg-gray-100 text-gray-800' :
                      'bg-green-100 text-green-800'
                    }`}>
                      {result.predicted}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm">
                    {(result.risk_score * 100).toFixed(2)}%
                  </td>
                  <td className="px-4 py-3">
                    {result.correct ? (
                      <span className="text-green-600 font-medium">✓ Correct</span>
                    ) : (
                      <span className="text-red-600 font-medium">✗ Incorrect</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-500">
                    {result.processing_time.toFixed(0)}ms
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )

  // Charts Tab
  const ChartsTab = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Risk Distribution Pie Chart */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Predicted Risk Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={5}
                dataKey="value"
                label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
              >
                {pieData.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={
                      entry.name === 'high' ? COLORS.danger :
                      entry.name === 'moderate' ? COLORS.warning :
                      entry.name === 'low' ? COLORS.success :
                      COLORS.neutral
                    } 
                  />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Processing Time Chart */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Processing Time per Case</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={processingTimeData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="test_id" />
              <YAxis label={{ value: 'ms', angle: -90, position: 'insideLeft' }} />
              <Tooltip formatter={(value: number) => `${value.toFixed(0)}ms`} />
              <Bar dataKey="time" fill={COLORS.primary} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Risk Score Distribution */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Risk Score Distribution</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={results.map(r => ({ 
            id: r.test_id.replace('SYNTH-', ''), 
            score: r.risk_score * 100,
            correct: r.correct 
          }))}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="id" />
            <YAxis domain={[0, 100]} label={{ value: 'Risk %', angle: -90, position: 'insideLeft' }} />
            <Tooltip formatter={(value: number) => `${value.toFixed(2)}%`} />
            <Bar dataKey="score" fill={COLORS.primary}>
              {results.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.correct ? COLORS.success : COLORS.danger} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className="mt-4 flex items-center gap-4 text-sm">
          <span className="flex items-center gap-2">
            <div className="w-3 h-3 rounded" style={{ backgroundColor: COLORS.success }}></div>
            Correct Prediction
          </span>
          <span className="flex items-center gap-2">
            <div className="w-3 h-3 rounded" style={{ backgroundColor: COLORS.danger }}></div>
            Incorrect Prediction
          </span>
        </div>
      </div>
    </div>
  )

  // Test Cases Tab
  const TestCasesTab = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="p-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-800">Loaded Test Cases</h3>
          <p className="text-sm text-gray-600">{testCases.length} positive cases from synthetic dataset</p>
        </div>
        <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 sticky top-0">
              <tr>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">ID</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Age</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Race</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">BMI</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Family Hx</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Personal Hx</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Relatives BC</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">HRT</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Expected</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {testCases.map((tc) => (
                <tr key={tc.test_id} className="hover:bg-gray-50">
                  <td className="px-3 py-2 font-mono text-xs">{tc.test_id}</td>
                  <td className="px-3 py-2">{tc.age}</td>
                  <td className="px-3 py-2 capitalize">{tc.race}</td>
                  <td className="px-3 py-2">{tc.current_bmi.toFixed(1)}</td>
                  <td className="px-3 py-2">
                    {tc.family_history_cancer ? '✓' : '—'}
                  </td>
                  <td className="px-3 py-2">
                    {tc.personal_history_cancer ? '✓' : '—'}
                  </td>
                  <td className="px-3 py-2">{tc.number_of_relatives_with_bc}</td>
                  <td className="px-3 py-2">
                    {tc.hormone_therapy ? `✓ (${tc.years_of_hormone_use}y)` : '—'}
                  </td>
                  <td className="px-3 py-2">
                    <span className="px-2 py-1 rounded text-xs font-medium bg-red-100 text-red-800">
                      {tc.expected_risk_category}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Model Validation</h1>
        <p className="text-gray-600 mt-2">
          Test the breast cancer risk prediction model against validated positive cases
        </p>
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          <span className="ml-3 text-gray-600">Loading test cases...</span>
        </div>
      )}

      {!isLoading && (
        <>
          {/* Tab Navigation */}
          <div className="bg-white rounded-lg shadow mb-6">
            <div className="border-b border-gray-200">
              <nav className="flex -mb-px overflow-x-auto">
                {[
                  { id: 'overview', label: 'Overview', icon: '📊' },
                  { id: 'results', label: 'Results', icon: '📋', disabled: results.length === 0 },
                  { id: 'charts', label: 'Charts', icon: '📈', disabled: results.length === 0 },
                  { id: 'testcases', label: 'Test Cases', icon: '🧪' }
                ].map(tab => (
                  <button
                    key={tab.id}
                    onClick={() => !tab.disabled && setActiveTab(tab.id)}
                    disabled={tab.disabled}
                    className={`py-4 px-6 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
                      activeTab === tab.id
                        ? 'border-primary text-primary'
                        : tab.disabled
                          ? 'border-transparent text-gray-300 cursor-not-allowed'
                          : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }`}
                  >
                    <span className="mr-2">{tab.icon}</span>
                    {tab.label}
                  </button>
                ))}
              </nav>
            </div>
          </div>

          {/* Tab Content */}
          <div>
            {activeTab === 'overview' && <OverviewTab />}
            {activeTab === 'results' && <ResultsTab />}
            {activeTab === 'charts' && <ChartsTab />}
            {activeTab === 'testcases' && <TestCasesTab />}
          </div>

          {/* Data Info Footer */}
          <div className="mt-8 bg-gray-50 rounded-lg p-6 border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-800 mb-3">Validation Dataset Info</h3>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
              <div>
                <div className="text-gray-500">Dataset</div>
                <div className="font-semibold text-gray-800">Synthetic Test Cases</div>
              </div>
              <div>
                <div className="text-gray-500">Total Cases</div>
                <div className="font-semibold text-gray-800">{testCases.length}</div>
              </div>
              <div>
                <div className="text-gray-500">Expected Distribution</div>
                <div className="font-semibold text-gray-800">{expectedSummary}</div>
              </div>
              <div>
                <div className="text-gray-500">Purpose</div>
                <div className="font-semibold text-gray-800">Model Accuracy Testing</div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

export default ModelValidation

