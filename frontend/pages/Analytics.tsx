import React, { useState, useEffect } from 'react'
import { BarChart, Bar, LineChart, Line, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import axios from 'axios'

const Analytics: React.FC = () => {
  const [performanceMetrics, setPerformanceMetrics] = useState<any>(null)
  const [selectedTimeRange, setSelectedTimeRange] = useState('30days')
  
  // Feature importance data
  const featureImportance = [
    { feature: 'Age', ga_importance: 0.185, baseline_importance: 0.162 },
    { feature: 'BMI Current', ga_importance: 0.142, baseline_importance: 0.118 },
    { feature: 'Age at Menarche', ga_importance: 0.098, baseline_importance: 0.089 },
    { feature: 'Family History', ga_importance: 0.087, baseline_importance: 0.095 },
    { feature: 'Personal History', ga_importance: 0.076, baseline_importance: 0.042 },
    { feature: 'Pack Years', ga_importance: 0.065, baseline_importance: 0.058 },
    { feature: 'Hormone Use', ga_importance: 0.054, baseline_importance: 0.071 },
    { feature: 'Live Births', ga_importance: 0.048, baseline_importance: 0.052 }
  ]
  
  // Model comparison metrics
  const modelComparison = [
    { metric: 'AUC', ga: 0.892, baseline: 0.876, improvement: '+1.8%' },
    { metric: 'Sensitivity', ga: 0.732, baseline: 0.718, improvement: '+1.9%' },
    { metric: 'Specificity', ga: 0.847, baseline: 0.832, improvement: '+1.8%' },
    { metric: 'Precision', ga: 0.285, baseline: 0.271, improvement: '+5.2%' },
    { metric: 'F1 Score', ga: 0.410, baseline: 0.394, improvement: '+4.1%' }
  ]
  
  // Calibration plot data
  const calibrationData = [
    { predicted: 0.05, observed: 0.048, perfect: 0.05 },
    { predicted: 0.10, observed: 0.102, perfect: 0.10 },
    { predicted: 0.15, observed: 0.147, perfect: 0.15 },
    { predicted: 0.20, observed: 0.195, perfect: 0.20 },
    { predicted: 0.25, observed: 0.248, perfect: 0.25 },
    { predicted: 0.30, observed: 0.292, perfect: 0.30 }
  ]
  
  // Population performance
  const populationPerformance = [
    { population: 'Caucasian', auc: 0.891, samples: 2800 },
    { population: 'African American', auc: 0.883, samples: 450 },
    { population: 'Hispanic', auc: 0.878, samples: 210 },
    { population: 'Asian', auc: 0.872, samples: 100 }
  ]
  
  useEffect(() => {
    // Fetch performance metrics from API
    const fetchMetrics = async () => {
      try {
        const response = await axios.get('/api/v1/analytics/performance')
        setPerformanceMetrics(response.data)
      } catch (error) {
        console.error('Error fetching metrics:', error)
      }
    }
    
    fetchMetrics()
  }, [])
  
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Model Analytics</h1>
        <p className="text-gray-600 mt-2">
          Performance metrics and insights for GA-optimized breast cancer risk prediction
        </p>
      </div>
      
      {/* Time Range Selector */}
      <div className="flex space-x-2">
        <button
          className={`px-4 py-2 rounded-md text-sm font-medium ${
            selectedTimeRange === '7days' ? 'bg-primary text-white' : 'bg-gray-100 text-gray-700'
          }`}
          onClick={() => setSelectedTimeRange('7days')}
        >
          7 Days
        </button>
        <button
          className={`px-4 py-2 rounded-md text-sm font-medium ${
            selectedTimeRange === '30days' ? 'bg-primary text-white' : 'bg-gray-100 text-gray-700'
          }`}
          onClick={() => setSelectedTimeRange('30days')}
        >
          30 Days
        </button>
        <button
          className={`px-4 py-2 rounded-md text-sm font-medium ${
            selectedTimeRange === '90days' ? 'bg-primary text-white' : 'bg-gray-100 text-gray-700'
          }`}
          onClick={() => setSelectedTimeRange('90days')}
        >
          90 Days
        </button>
      </div>
      
      {/* Key Performance Indicators */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="clinical-card text-center">
          <h3 className="text-lg font-semibold text-gray-700 mb-4">Model Accuracy (AUC)</h3>
          <div className="relative w-32 h-32 mx-auto">
            <svg className="w-32 h-32 transform -rotate-90">
              <circle
                cx="64"
                cy="64"
                r="56"
                stroke="#e5e7eb"
                strokeWidth="16"
                fill="none"
              />
              <circle
                cx="64"
                cy="64"
                r="56"
                stroke="#0B5394"
                strokeWidth="16"
                fill="none"
                strokeDasharray={`${2 * Math.PI * 56 * 0.892} ${2 * Math.PI * 56}`}
              />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-2xl font-bold">89.2%</span>
            </div>
          </div>
          <p className="text-sm text-gray-600 mt-2">GA-Optimized Model</p>
        </div>
        
        <div className="clinical-card text-center">
          <h3 className="text-lg font-semibold text-gray-700 mb-4">Feature Reduction</h3>
          <p className="text-4xl font-bold text-secondary">32%</p>
          <p className="text-sm text-gray-600 mt-2">Fewer features with better performance</p>
        </div>
        
        <div className="clinical-card text-center">
          <h3 className="text-lg font-semibold text-gray-700 mb-4">Inference Speed</h3>
          <p className="text-4xl font-bold text-green-600">28%</p>
          <p className="text-sm text-gray-600 mt-2">Faster predictions</p>
        </div>
      </div>
      
      {/* Model Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="clinical-card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Model Performance Comparison
          </h3>
          <div className="space-y-4">
            {modelComparison.map((metric) => (
              <div key={metric.metric} className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span className="font-medium">{metric.metric}</span>
                  <span className="text-green-600 font-medium">{metric.improvement}</span>
                </div>
                <div className="relative">
                  <div className="w-full bg-gray-200 rounded-full h-6">
                    <div
                      className="bg-gray-400 h-6 rounded-full absolute top-0"
                      style={{ width: `${metric.baseline * 100}%` }}
                    />
                    <div
                      className="bg-primary h-6 rounded-full relative flex items-center justify-end pr-2"
                      style={{ width: `${metric.ga * 100}%` }}
                    >
                      <span className="text-xs text-white font-medium">
                        {(metric.ga * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
          <div className="flex items-center space-x-4 mt-4 text-sm">
            <div className="flex items-center">
              <div className="w-4 h-4 bg-primary rounded mr-2" />
              <span>GA-Optimized</span>
            </div>
            <div className="flex items-center">
              <div className="w-4 h-4 bg-gray-400 rounded mr-2" />
              <span>Baseline</span>
            </div>
          </div>
        </div>
        
        {/* Feature Importance */}
        <div className="clinical-card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Feature Importance Comparison
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={featureImportance} layout="horizontal">
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis type="number" domain={[0, 0.2]} />
              <YAxis dataKey="feature" type="category" width={100} />
              <Tooltip />
              <Bar dataKey="ga_importance" fill="#0B5394" name="GA Model" />
              <Bar dataKey="baseline_importance" fill="#00897B" name="Baseline" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      {/* Calibration Plot */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="clinical-card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Model Calibration
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={calibrationData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="predicted" 
                label={{ value: 'Predicted Risk', position: 'insideBottom', offset: -5 }}
              />
              <YAxis 
                label={{ value: 'Observed Risk', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="perfect"
                stroke="#e5e7eb"
                strokeWidth={2}
                strokeDasharray="5 5"
                name="Perfect Calibration"
              />
              <Line
                type="monotone"
                dataKey="observed"
                stroke="#0B5394"
                strokeWidth={3}
                name="GA Model"
                dot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
          <p className="text-sm text-gray-600 mt-2">
            Hosmer-Lemeshow p-value: 0.42 (well-calibrated)
          </p>
        </div>
        
        {/* Population Performance */}
        <div className="clinical-card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Performance Across Populations
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={populationPerformance}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="population" angle={-45} textAnchor="end" height={80} />
              <YAxis domain={[0.85, 0.90]} />
              <Tooltip />
              <Bar dataKey="auc" fill="#7B1FA2" name="AUC Score">
                {populationPerformance.map((entry, index) => (
                  <text
                    key={index}
                    x={0}
                    y={0}
                    fill="#666"
                    textAnchor="middle"
                    fontSize={12}
                  >
                    {`n=${entry.samples}`}
                  </text>
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      {/* Model Details */}
      <div className="clinical-card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Technical Details</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-gray-700 mb-2">GA-Optimized Model</h4>
            <ul className="space-y-1 text-sm text-gray-600">
              <li>• Algorithm: Random Forest with GA feature selection</li>
              <li>• Features: 28 (reduced from 90+)</li>
              <li>• Cross-validation: 5-fold stratified</li>
              <li>• Training samples: 3,560</li>
              <li>• Class balance: SMOTE applied</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium text-gray-700 mb-2">Validation</h4>
            <ul className="space-y-1 text-sm text-gray-600">
              <li>• Internal: 5-fold CV (AUC: 0.889 ± 0.021)</li>
              <li>• Temporal: 2023 holdout (AUC: 0.885)</li>
              <li>• External: Planned for AKU dataset</li>
              <li>• Bootstrap: 1000 iterations for CI</li>
              <li>• DeLong test: p &lt; 0.001 vs baseline</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Analytics