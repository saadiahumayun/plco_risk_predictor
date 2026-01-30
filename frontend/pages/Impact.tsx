import React from 'react'
import { motion } from 'framer-motion'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'

const Impact: React.FC = () => {
  // Key statistics
  const healthcareContext = [
    { label: 'Breast Cancer Incidence', value: 'Highest in South Asia', icon: '📊' },
    { label: 'Late-stage Detection', value: '90% of cases', icon: '⚠️' },
    { label: '5-year Survival Rate', value: '30-40%', comparison: 'vs 85%+ in developed countries', icon: '💔' },
    { label: 'Cost Differential', value: '3-4x higher', subtext: 'for late-stage treatment', icon: '💰' },
  ]

  const treatmentCosts = [
    { stage: 'Early Stage (I-II)', min: 0.3, max: 0.8, color: '#10b981' },
    { stage: 'Late Stage (III-IV)', min: 1.2, max: 3.5, color: '#ef4444' },
  ]

  const scenarioData = [
    { scenario: 'Conservative', improvement: '2%', cases: 200, savings: 180, color: '#6b7280' },
    { scenario: 'Moderate', improvement: '5%', cases: 500, savings: 450, color: '#f59e0b' },
    { scenario: 'Optimistic', improvement: '8%', cases: 800, savings: 720, color: '#10b981' },
  ]

  const totalBenefits = [
    { category: 'Early Detection Savings', conservative: 180, moderate: 450, optimistic: 720 },
    { category: 'Reduced False Positives', conservative: 2, moderate: 3, optimistic: 5 },
    { category: 'Operational Efficiency', conservative: 4, moderate: 6, optimistic: 8 },
    { category: 'Increased Throughput', conservative: 3, moderate: 5, optimistic: 8 },
  ]

  const implementationPhases = [
    { phase: 'Phase 1: Pilot', timeline: 'Months 1-6', activities: 'Deploy at AKU & SKMCH, Validate assumptions', investment: '15-25M', roi: 'Data validation' },
    { phase: 'Phase 2: Scale', timeline: 'Months 7-18', activities: 'Deploy across 10 hospitals, 100K+ screenings', investment: '50-100M', roi: '100-200M revenue' },
    { phase: 'Phase 3: National', timeline: 'Years 2-5', activities: 'National Health integration, 1M+ screenings', investment: '200-500M', roi: '1-2B revenue' },
  ]

  const pieData = [
    { name: 'Early Detection', value: 180, color: '#6D2842' },
    { name: 'False Positive Reduction', value: 2, color: '#8B4A5E' },
    { name: 'Efficiency Gains', value: 4, color: '#D4A574' },
    { name: 'Throughput', value: 3, color: '#B8860B' },
  ]

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-3xl font-bold text-gray-900">Impact Analysis</h1>
        <p className="text-gray-600 mt-2">
          AI-Enabled Early Breast Cancer Detection: Financial Impact for Pakistani Healthcare
        </p>
        <div className="mt-2 text-sm text-gray-500">
          Based on international research and adapted for Pakistan's healthcare context
        </div>
      </motion.div>

      {/* Key Value Proposition */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="bg-gradient-to-r from-primary to-primary-dark rounded-2xl p-8 text-white mb-8"
      >
        <div className="text-center">
          <div className="text-lg opacity-90 mb-2">Projected Annual Value Per Hospital</div>
          <div className="text-5xl font-bold mb-2">PKR 200-500 Million</div>
          <div className="text-lg opacity-90">Based on 10,000 annual screenings</div>
        </div>
      </motion.div>

      {/* Healthcare Context Cards */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="mb-8"
      >
        <h2 className="text-2xl font-semibold text-gray-800 mb-4">Pakistani Healthcare Context</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {healthcareContext.map((stat, index) => (
            <div key={index} className="bg-white rounded-xl shadow-md p-6 border-l-4 border-primary">
              <div className="text-3xl mb-2">{stat.icon}</div>
              <div className="text-sm text-gray-500 uppercase tracking-wide">{stat.label}</div>
              <div className="text-2xl font-bold text-gray-900 mt-1">{stat.value}</div>
              {stat.comparison && (
                <div className="text-sm text-red-600 mt-1">{stat.comparison}</div>
              )}
              {stat.subtext && (
                <div className="text-sm text-gray-600 mt-1">{stat.subtext}</div>
              )}
            </div>
          ))}
        </div>
      </motion.div>

      {/* Treatment Cost Comparison */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="bg-white rounded-xl shadow-md p-6 mb-8"
      >
        <h2 className="text-2xl font-semibold text-gray-800 mb-6">Treatment Cost Analysis (PKR Millions)</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={treatmentCosts} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 4]} unit="M" />
                <YAxis dataKey="stage" type="category" width={120} tick={{ fontSize: 12 }} />
                <Tooltip formatter={(value: number) => `PKR ${value}M`} />
                <Bar dataKey="max" fill="#ef4444" name="Maximum Cost" radius={[0, 4, 4, 0]} />
                <Bar dataKey="min" fill="#10b981" name="Minimum Cost" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="flex flex-col justify-center space-y-4">
            <div className="bg-green-50 rounded-lg p-4 border-l-4 border-green-500">
              <div className="text-sm text-green-700 font-medium">Early Stage (I-II)</div>
              <div className="text-2xl font-bold text-green-800">PKR 300K - 800K</div>
              <div className="text-sm text-green-600">Higher survival rates, fewer complications</div>
            </div>
            <div className="bg-red-50 rounded-lg p-4 border-l-4 border-red-500">
              <div className="text-sm text-red-700 font-medium">Late Stage (III-IV)</div>
              <div className="text-2xl font-bold text-red-800">PKR 1.2M - 3.5M</div>
              <div className="text-sm text-red-600">3-4x more expensive, lower survival rates</div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Scenario Analysis */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="bg-white rounded-xl shadow-md p-6 mb-8"
      >
        <h2 className="text-2xl font-semibold text-gray-800 mb-2">Financial Impact Scenarios</h2>
        <p className="text-gray-600 mb-6">Annual projections based on 10,000 screenings</p>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          {scenarioData.map((scenario, index) => (
            <div 
              key={index} 
              className={`rounded-xl p-6 ${
                scenario.scenario === 'Moderate' 
                  ? 'bg-gradient-to-br from-amber-50 to-amber-100 border-2 border-amber-300' 
                  : 'bg-gray-50'
              }`}
            >
              {scenario.scenario === 'Moderate' && (
                <div className="text-xs font-semibold text-amber-600 uppercase tracking-wide mb-2">Recommended</div>
              )}
              <div className="text-lg font-semibold text-gray-800">{scenario.scenario}</div>
              <div className="text-3xl font-bold mt-2" style={{ color: scenario.color }}>
                {scenario.improvement}
              </div>
              <div className="text-sm text-gray-500">improvement in early detection</div>
              <div className="mt-4 pt-4 border-t border-gray-200">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Additional Cases Detected</span>
                  <span className="font-semibold">{scenario.cases}</span>
                </div>
                <div className="flex justify-between text-sm mt-2">
                  <span className="text-gray-600">Annual Savings</span>
                  <span className="font-bold text-lg" style={{ color: scenario.color }}>PKR {scenario.savings}M</span>
                </div>
              </div>
            </div>
          ))}
        </div>

        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={scenarioData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="scenario" />
            <YAxis label={{ value: 'PKR Millions', angle: -90, position: 'insideLeft' }} />
            <Tooltip formatter={(value: number) => `PKR ${value}M`} />
            <Bar dataKey="savings" name="Annual Savings">
              {scenarioData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </motion.div>

      {/* Total Economic Impact */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="bg-white rounded-xl shadow-md p-6 mb-8"
      >
        <h2 className="text-2xl font-semibold text-gray-800 mb-6">Total Economic Impact Breakdown</h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b-2 border-gray-200">
                  <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600">Benefit Category</th>
                  <th className="text-right py-3 px-4 text-sm font-semibold text-gray-500">Conservative</th>
                  <th className="text-right py-3 px-4 text-sm font-semibold text-amber-600">Moderate</th>
                  <th className="text-right py-3 px-4 text-sm font-semibold text-green-600">Optimistic</th>
                </tr>
              </thead>
              <tbody>
                {totalBenefits.map((row, index) => (
                  <tr key={index} className="border-b border-gray-100">
                    <td className="py-3 px-4 text-sm font-medium text-gray-800">{row.category}</td>
                    <td className="py-3 px-4 text-sm text-right text-gray-600">PKR {row.conservative}M</td>
                    <td className="py-3 px-4 text-sm text-right text-amber-600 font-medium">PKR {row.moderate}M</td>
                    <td className="py-3 px-4 text-sm text-right text-green-600 font-medium">PKR {row.optimistic}M</td>
                  </tr>
                ))}
                <tr className="bg-gray-50 font-bold">
                  <td className="py-3 px-4 text-sm text-gray-800">TOTAL ANNUAL VALUE</td>
                  <td className="py-3 px-4 text-sm text-right text-gray-700">PKR 189M</td>
                  <td className="py-3 px-4 text-sm text-right text-amber-700">PKR 464M</td>
                  <td className="py-3 px-4 text-sm text-right text-green-700">PKR 741M</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-700 mb-4 text-center">Value Distribution (Conservative)</h3>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={90}
                  paddingAngle={3}
                  dataKey="value"
                  label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
                  labelLine={false}
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip formatter={(value: number) => `PKR ${value}M`} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </motion.div>

      {/* Implementation Roadmap */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="bg-white rounded-xl shadow-md p-6 mb-8"
      >
        <h2 className="text-2xl font-semibold text-gray-800 mb-6">Implementation Roadmap</h2>
        <div className="space-y-4">
          {implementationPhases.map((phase, index) => (
            <div key={index} className="flex items-start gap-4">
              <div className={`flex-shrink-0 w-12 h-12 rounded-full flex items-center justify-center text-white font-bold ${
                index === 0 ? 'bg-primary' : index === 1 ? 'bg-amber-500' : 'bg-green-500'
              }`}>
                {index + 1}
              </div>
              <div className="flex-grow bg-gray-50 rounded-lg p-4">
                <div className="flex flex-wrap items-center gap-4 mb-2">
                  <span className="font-semibold text-gray-800">{phase.phase}</span>
                  <span className="text-sm bg-white px-3 py-1 rounded-full text-gray-600 border">{phase.timeline}</span>
                </div>
                <p className="text-sm text-gray-600 mb-3">{phase.activities}</p>
                <div className="flex flex-wrap gap-4 text-sm">
                  <div>
                    <span className="text-gray-500">Investment: </span>
                    <span className="font-semibold text-gray-800">PKR {phase.investment}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Expected ROI: </span>
                    <span className="font-semibold text-green-600">{phase.roi}</span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Partner Institutions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
        className="bg-white rounded-xl shadow-md p-6 mb-8"
      >
        <h2 className="text-2xl font-semibold text-gray-800 mb-4">Target Partner Institutions</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[
            { name: 'Aga Khan University Hospital', location: 'Karachi', type: 'Private' },
            { name: 'Shaukat Khanum Memorial', location: 'Lahore', type: 'Non-profit' },
            { name: 'Combined Military Hospital', location: 'Rawalpindi', type: 'Public' },
            { name: 'Pakistan Health Research Council', location: 'Islamabad', type: 'Government' },
            { name: 'Pakistan Society of Radiology', location: 'National', type: 'Professional' },
            { name: 'Ministry of Health Services', location: 'Federal', type: 'Government' },
          ].map((partner, index) => (
            <div key={index} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
              <div className="font-semibold text-gray-800">{partner.name}</div>
              <div className="text-sm text-gray-500 mt-1">{partner.location}</div>
              <span className={`inline-block mt-2 text-xs px-2 py-1 rounded-full ${
                partner.type === 'Private' ? 'bg-blue-100 text-blue-700' :
                partner.type === 'Public' ? 'bg-green-100 text-green-700' :
                partner.type === 'Government' ? 'bg-purple-100 text-purple-700' :
                partner.type === 'Non-profit' ? 'bg-rose-100 text-rose-700' :
                'bg-gray-100 text-gray-700'
              }`}>
                {partner.type}
              </span>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Disclaimer */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
        className="bg-amber-50 border border-amber-200 rounded-xl p-6 mb-8"
      >
        <div className="flex items-start gap-3">
          <span className="text-2xl">⚠️</span>
          <div>
            <h3 className="font-semibold text-amber-800 mb-2">Important Disclaimer</h3>
            <p className="text-sm text-amber-700">
              This analysis presents a framework for financial impact assessment based on available international 
              literature and publicly accessible information. All financial projections are estimates pending 
              validation through pilot program implementation. Cost figures require validation through primary 
              research with Pakistani healthcare institutions.
            </p>
          </div>
        </div>
      </motion.div>

      {/* References */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.9 }}
        className="bg-gray-50 rounded-xl p-6"
      >
        <h2 className="text-xl font-semibold text-gray-800 mb-4">Key References</h2>
        <ul className="space-y-2 text-sm text-gray-600">
          <li>• McKinney et al. (2020) - "International evaluation of an AI system for breast cancer screening" - Nature</li>
          <li>• Rodriguez-Ruiz et al. (2019) - "Stand-alone artificial intelligence for breast cancer detection" - Radiology</li>
          <li>• GLOBOCAN 2020 - International Agency for Research on Cancer (IARC)</li>
          <li>• World Health Organization - Global Health Observatory data repository</li>
        </ul>
        <p className="text-xs text-gray-500 mt-4">Prepared: January 2026</p>
      </motion.div>
    </div>
  )
}

export default Impact
