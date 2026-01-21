import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

const Dashboard: React.FC = () => {
  const [metrics] = useState({
    totalAssessments: 1284,
    highRiskPatients: 89,
    averageRisk: 0.048,
    modelAccuracy: 0.892
  })
  
  // Mock data for charts
  const riskTrendData = [
    { month: 'Jan', assessments: 98, avgRisk: 0.045 },
    { month: 'Feb', assessments: 112, avgRisk: 0.048 },
    { month: 'Mar', assessments: 105, avgRisk: 0.046 },
    { month: 'Apr', assessments: 118, avgRisk: 0.049 },
    { month: 'May', assessments: 124, avgRisk: 0.047 },
    { month: 'Jun', assessments: 131, avgRisk: 0.048 },
  ]
  
  const riskDistribution = [
    { name: 'Low Risk', value: 842, color: '#E991B0' },
    { name: 'Moderate Risk', value: 353, color: '#9B59B6' },
    { name: 'High Risk', value: 89, color: '#F5CBA7' },
  ]
  
  const ageDistribution = [
    { age: '40-49', count: 312 },
    { age: '50-59', count: 456 },
    { age: '60-69', count: 389 },
    { age: '70+', count: 127 },
  ]
  
  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Clinical Dashboard</h1>
        <p className="text-gray-600 mt-2">
          Overview of breast cancer risk assessments and key metrics
        </p>
      </div>
      
      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Link
          to="/assessment"
          className="clinical-card hover:shadow-lg transition-shadow duration-200 border-l-4 border-primary"
        >
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-gray-900">New Assessment</h3>
              <p className="text-gray-600 mt-1">Perform a new risk evaluation</p>
            </div>
            <div className="text-4xl">🔬</div>
          </div>
        </Link>
        
        <Link
          to="/history"
          className="clinical-card hover:shadow-lg transition-shadow duration-200 border-l-4 border-secondary"
        >
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-gray-900">Recent Patients</h3>
              <p className="text-gray-600 mt-1">View assessment history</p>
            </div>
            <div className="text-4xl">📋</div>
          </div>
        </Link>
      </div>
      
      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="metric-card"
        >
          <p className="metric-label">Total Assessments</p>
          <p className="metric-value">{metrics.totalAssessments.toLocaleString()}</p>
          <p className="text-xs text-gray-500 mt-1">Last 6 months</p>
        </motion.div>
        
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.1 }}
          className="metric-card border-red-200"
        >
          <p className="metric-label text-red-700">High Risk Patients</p>
          <p className="metric-value text-red-700">{metrics.highRiskPatients}</p>
          <p className="text-xs text-red-600 mt-1">Requires follow-up</p>
        </motion.div>
        
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.2 }}
          className="metric-card"
        >
          <p className="metric-label">Average Risk Score</p>
          <p className="metric-value">{(metrics.averageRisk * 100).toFixed(1)}%</p>
          <p className="text-xs text-gray-500 mt-1">5-year risk</p>
        </motion.div>
        
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.3 }}
          className="metric-card border-green-200"
        >
          <p className="metric-label text-green-700">Model Accuracy</p>
          <p className="metric-value text-green-700">{(metrics.modelAccuracy * 100).toFixed(1)}%</p>
          <p className="text-xs text-green-600 mt-1">AUC Score</p>
        </motion.div>
      </div>
      
      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Risk Trend Chart */}
        <div className="clinical-card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Assessment Trends
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={riskTrendData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="month" />
              <YAxis yAxisId="left" />
              <YAxis yAxisId="right" orientation="right" />
              <Tooltip />
              <Legend />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="assessments"
                stroke="#0B5394"
                strokeWidth={2}
                name="Assessments"
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="avgRisk"
                stroke="#7B1FA2"
                strokeWidth={2}
                name="Avg Risk"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        
        {/* Risk Distribution */}
        <div className="clinical-card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Risk Distribution
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={riskDistribution}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                innerRadius={50}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {riskDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      {/* Age Distribution */}
      <div className="clinical-card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Age Distribution of Assessments
        </h3>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={ageDistribution}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="age" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#0B5394" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      
      {/* Recent Activity */}
      <div className="clinical-card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Activity</h3>
        <div className="space-y-3">
          <div className="flex items-center justify-between py-2 border-b border-gray-100">
            <div>
              <p className="font-medium">Patient ID: BC-2024-0142</p>
              <p className="text-sm text-gray-600">Risk: 12.3% - Moderate</p>
            </div>
            <span className="text-sm text-gray-500">2 hours ago</span>
          </div>
          <div className="flex items-center justify-between py-2 border-b border-gray-100">
            <div>
              <p className="font-medium">Patient ID: BC-2024-0141</p>
              <p className="text-sm text-gray-600">Risk: 3.2% - Low</p>
            </div>
            <span className="text-sm text-gray-500">4 hours ago</span>
          </div>
          <div className="flex items-center justify-between py-2">
            <div>
              <p className="font-medium">Patient ID: BC-2024-0140</p>
              <p className="text-sm text-red-600">Risk: 24.5% - High</p>
            </div>
            <span className="text-sm text-gray-500">Yesterday</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Dashboard