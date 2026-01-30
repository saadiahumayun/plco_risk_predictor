import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts'
import { getDashboardStats, DashboardStats } from '../services/api'
import { format } from 'date-fns'

const Dashboard: React.FC = () => {
  const [stats, setStats] = useState<DashboardStats | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadDashboardData()
  }, [])

  const loadDashboardData = async () => {
    setIsLoading(true)
    setError(null)
    try {
      const data = await getDashboardStats()
      setStats(data)
    } catch (err: any) {
      console.error('Failed to load dashboard data:', err)
      setError(err.message || 'Failed to load data')
    } finally {
      setIsLoading(false)
    }
  }

  // Prepare chart data
  const riskDistribution = stats ? [
    { name: 'Low Risk', value: stats.lowRiskCount, color: '#10b981' },
    { name: 'Moderate Risk', value: stats.moderateRiskCount, color: '#f59e0b' },
    { name: 'High Risk', value: stats.highRiskCount, color: '#ef4444' },
  ].filter(d => d.value > 0) : []

  const getRiskColor = (category: string) => {
    switch (category) {
      case 'high': return 'text-red-600'
      case 'moderate': return 'text-orange-600'
      default: return 'text-green-600'
    }
  }

  const getRiskBadge = (category: string) => {
    switch (category) {
      case 'high': return 'bg-red-100 text-red-800'
      case 'moderate': return 'bg-orange-100 text-orange-800'
      default: return 'bg-green-100 text-green-800'
    }
  }

  const formatTimeAgo = (dateStr: string) => {
    const date = new Date(dateStr)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMs / 3600000)
    const diffDays = Math.floor(diffMs / 86400000)

    if (diffMins < 60) return `${diffMins} min ago`
    if (diffHours < 24) return `${diffHours} hours ago`
    if (diffDays < 7) return `${diffDays} days ago`
    return format(date, 'MMM dd, yyyy')
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading dashboard...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Clinical Dashboard</h1>
          <p className="text-gray-600 mt-2">Overview of breast cancer risk assessments</p>
        </div>
        <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
          <p className="text-red-800 mb-4">{error}</p>
          <button onClick={loadDashboardData} className="btn-primary">
            Retry
          </button>
        </div>
      </div>
    )
  }

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
            <svg className="w-10 h-10 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
        </Link>
        
        <Link
          to="/history"
          className="clinical-card hover:shadow-lg transition-shadow duration-200 border-l-4 border-secondary"
        >
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-gray-900">Patient History</h3>
              <p className="text-gray-600 mt-1">View assessment history</p>
            </div>
            <svg className="w-10 h-10 text-secondary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
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
          <p className="metric-value">{stats?.totalAssessments || 0}</p>
          <p className="text-xs text-gray-500 mt-1">All time</p>
        </motion.div>
        
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.1 }}
          className="metric-card border-red-200"
        >
          <p className="metric-label text-red-700">High Risk Patients</p>
          <p className="metric-value text-red-700">{stats?.highRiskCount || 0}</p>
          <p className="text-xs text-red-600 mt-1">Requires follow-up</p>
        </motion.div>
        
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.2 }}
          className="metric-card border-orange-200"
        >
          <p className="metric-label text-orange-700">Moderate Risk</p>
          <p className="metric-value text-orange-700">{stats?.moderateRiskCount || 0}</p>
          <p className="text-xs text-orange-600 mt-1">Monitor closely</p>
        </motion.div>
        
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.3 }}
          className="metric-card border-green-200"
        >
          <p className="metric-label text-green-700">Average Risk Score</p>
          <p className="metric-value text-green-700">
            {stats?.averageRisk ? (stats.averageRisk * 100).toFixed(1) : 0}%
          </p>
          <p className="text-xs text-green-600 mt-1">5-year risk</p>
        </motion.div>
      </div>
      
      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Risk Distribution */}
        <div className="clinical-card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Risk Distribution
          </h3>
          {riskDistribution.length > 0 ? (
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
          ) : (
            <div className="flex items-center justify-center h-[300px] text-gray-500">
              No data available yet
            </div>
          )}
        </div>

        {/* Risk Score Distribution */}
        <div className="clinical-card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Risk Scores Overview
          </h3>
          {stats && stats.recentPredictions.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={stats.recentPredictions.slice(0, 10).map((p, i) => ({
                id: `#${i + 1}`,
                score: p.risk_score * 100,
                category: p.risk_category
              }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="id" />
                <YAxis domain={[0, 20]} label={{ value: 'Risk %', angle: -90, position: 'insideLeft' }} />
                <Tooltip formatter={(value: number) => `${value.toFixed(2)}%`} />
                <Bar dataKey="score" radius={[4, 4, 0, 0]}>
                  {stats.recentPredictions.slice(0, 10).map((p, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={p.risk_category === 'high' ? '#ef4444' : p.risk_category === 'moderate' ? '#f59e0b' : '#10b981'} 
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-[300px] text-gray-500">
              No data available yet
            </div>
          )}
        </div>
      </div>
      
      {/* Recent Activity */}
      <div className="clinical-card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Recent Assessments</h3>
          <Link to="/history" className="text-primary hover:text-primary-dark text-sm font-medium">
            View All →
          </Link>
        </div>
        {stats && stats.recentPredictions.length > 0 ? (
          <div className="space-y-3">
            {stats.recentPredictions.slice(0, 5).map((prediction, index) => (
              <div 
                key={prediction.prediction_id} 
                className="flex items-center justify-between py-3 border-b border-gray-100 last:border-0"
              >
                <div className="flex items-center gap-4">
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center text-white font-medium ${
                    prediction.risk_category === 'high' ? 'bg-red-500' :
                    prediction.risk_category === 'moderate' ? 'bg-orange-500' : 'bg-green-500'
                  }`}>
                    {(prediction.risk_score * 100).toFixed(0)}
                  </div>
                  <div>
                    <p className="font-medium text-gray-900">
                      {prediction.patient_id || `Assessment #${index + 1}`}
                    </p>
                    <p className={`text-sm ${getRiskColor(prediction.risk_category)}`}>
                      Risk: {(prediction.risk_score * 100).toFixed(1)}% - 
                      <span className={`ml-1 px-2 py-0.5 rounded-full text-xs ${getRiskBadge(prediction.risk_category)}`}>
                        {prediction.risk_category.charAt(0).toUpperCase() + prediction.risk_category.slice(1)}
                      </span>
                    </p>
                  </div>
                </div>
                <span className="text-sm text-gray-500">
                  {formatTimeAgo(prediction.created_at)}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <p>No assessments yet</p>
            <Link to="/assessment" className="text-primary hover:underline mt-2 inline-block">
              Create your first assessment →
            </Link>
          </div>
        )}
      </div>
    </div>
  )
}

export default Dashboard
