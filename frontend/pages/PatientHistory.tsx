import React, { useState, useEffect } from 'react'
import { format } from 'date-fns'
import { getPredictions, PredictionRecord } from '../services/api'

const PatientHistory: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('')
  const [filterCategory, setFilterCategory] = useState('all')
  const [predictions, setPredictions] = useState<PredictionRecord[]>([])
  const [totalCount, setTotalCount] = useState(0)
  const [currentPage, setCurrentPage] = useState(1)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  
  const pageSize = 10

  useEffect(() => {
    loadPredictions()
  }, [currentPage])

  const loadPredictions = async () => {
    setIsLoading(true)
    setError(null)
    try {
      const skip = (currentPage - 1) * pageSize
      const data = await getPredictions(skip, pageSize)
      setPredictions(data.predictions)
      setTotalCount(data.total)
    } catch (err: any) {
      console.error('Failed to load predictions:', err)
      setError(err.message || 'Failed to load data')
    } finally {
      setIsLoading(false)
    }
  }
  
  // Filter predictions based on search and category
  const filteredPredictions = predictions.filter(prediction => {
    const matchesSearch = !searchTerm || 
      (prediction.patient_id?.toLowerCase().includes(searchTerm.toLowerCase())) ||
      (prediction.prediction_id?.toLowerCase().includes(searchTerm.toLowerCase()))
    const matchesFilter = filterCategory === 'all' || prediction.risk_category === filterCategory
    return matchesSearch && matchesFilter
  })

  // Calculate stats from current page
  const stats = {
    total: totalCount,
    highRisk: predictions.filter(p => p.risk_category === 'high').length,
    moderateRisk: predictions.filter(p => p.risk_category === 'moderate').length,
    lowRisk: predictions.filter(p => p.risk_category === 'low').length,
    avgRisk: predictions.length > 0 
      ? predictions.reduce((sum, p) => sum + p.risk_score, 0) / predictions.length 
      : 0
  }
  
  const getRiskBadgeClasses = (category: string) => {
    switch (category) {
      case 'high':
        return 'bg-red-100 text-red-800 border-red-200'
      case 'moderate':
        return 'bg-orange-100 text-orange-800 border-orange-200'
      case 'low':
        return 'bg-green-100 text-green-800 border-green-200'
      default:
        return ''
    }
  }

  const totalPages = Math.ceil(totalCount / pageSize)

  const handleExport = () => {
    // Create CSV content
    const headers = ['Patient ID', 'Date', 'Risk Score', 'Risk Category', 'Model Version']
    const rows = predictions.map(p => [
      p.patient_id || p.prediction_id,
      p.created_at ? format(new Date(p.created_at), 'yyyy-MM-dd HH:mm') : '',
      (p.risk_score * 100).toFixed(2) + '%',
      p.risk_category,
      p.model_version || ''
    ])
    
    const csv = [headers, ...rows].map(row => row.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `predictions-export-${format(new Date(), 'yyyy-MM-dd')}.csv`
    a.click()
  }
  
  if (isLoading && predictions.length === 0) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading patient history...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Patient History</h1>
        <p className="text-gray-600 mt-2">
          View and manage previous breast cancer risk assessments
        </p>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800">{error}</p>
          <button onClick={loadPredictions} className="text-red-600 hover:text-red-800 text-sm mt-2">
            Retry
          </button>
        </div>
      )}
      
      {/* Search and Filter */}
      <div className="flex flex-col md:flex-row gap-4">
        <div className="flex-1">
          <label className="clinical-label">Search Patient ID</label>
          <input
            type="text"
            placeholder="Search by patient or prediction ID..."
            className="clinical-input"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        
        <div className="w-full md:w-48">
          <label className="clinical-label">Risk Category</label>
          <select
            className="clinical-input"
            value={filterCategory}
            onChange={(e) => setFilterCategory(e.target.value)}
          >
            <option value="all">All Categories</option>
            <option value="low">Low Risk</option>
            <option value="moderate">Moderate Risk</option>
            <option value="high">High Risk</option>
          </select>
        </div>
        
        <div className="flex items-end">
          <button onClick={handleExport} className="btn-primary flex items-center gap-2">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Export CSV
          </button>
        </div>
      </div>
      
      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="metric-card">
          <p className="metric-label">Total Assessments</p>
          <p className="metric-value">{stats.total}</p>
        </div>
        <div className="metric-card border-red-200">
          <p className="metric-label text-red-700">High Risk</p>
          <p className="metric-value text-red-600">{stats.highRisk}</p>
        </div>
        <div className="metric-card border-orange-200">
          <p className="metric-label text-orange-700">Moderate Risk</p>
          <p className="metric-value text-orange-600">{stats.moderateRisk}</p>
        </div>
        <div className="metric-card border-green-200">
          <p className="metric-label text-green-700">Average Risk</p>
          <p className="metric-value text-green-600">
            {(stats.avgRisk * 100).toFixed(1)}%
          </p>
        </div>
      </div>
      
      {/* Patient Table */}
      <div className="clinical-card overflow-hidden">
        <div className="overflow-x-auto">
          {filteredPredictions.length > 0 ? (
            <table className="clinical-table">
              <thead>
                <tr>
                  <th>Patient ID</th>
                  <th>Date</th>
                  <th>Risk Score</th>
                  <th>Category</th>
                  <th>Confidence</th>
                  <th>Model</th>
                  <th>Processing</th>
                </tr>
              </thead>
              <tbody>
                {filteredPredictions.map((prediction) => (
                  <tr key={prediction.prediction_id} className="hover:bg-gray-50">
                    <td className="font-medium">
                      {prediction.patient_id || prediction.prediction_id.slice(0, 12) + '...'}
                    </td>
                    <td>
                      {prediction.created_at 
                        ? format(new Date(prediction.created_at), 'MMM dd, yyyy HH:mm')
                        : '-'
                      }
                    </td>
                    <td className="font-semibold">
                      {(prediction.risk_score * 100).toFixed(2)}%
                    </td>
                    <td>
                      <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getRiskBadgeClasses(prediction.risk_category)}`}>
                        {prediction.risk_category.charAt(0).toUpperCase() + prediction.risk_category.slice(1)}
                      </span>
                    </td>
                    <td className="text-sm text-gray-600">
                      {prediction.confidence_interval 
                        ? `${(prediction.confidence_interval.lower * 100).toFixed(1)}% - ${(prediction.confidence_interval.upper * 100).toFixed(1)}%`
                        : '-'
                      }
                    </td>
                    <td className="text-sm">{prediction.model_version || 'GA-Optimized'}</td>
                    <td className="text-sm text-gray-500">
                      {prediction.processing_time_ms 
                        ? `${prediction.processing_time_ms.toFixed(0)}ms`
                        : '-'
                      }
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="text-center py-12 text-gray-500">
              {predictions.length === 0 ? (
                <>
                  <svg className="w-12 h-12 mx-auto text-gray-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <p>No assessments found</p>
                  <p className="text-sm mt-2">Start by creating a new risk assessment</p>
                </>
              ) : (
                <p>No results match your search criteria</p>
              )}
            </div>
          )}
        </div>
      </div>
      
      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex justify-between items-center">
          <p className="text-sm text-gray-600">
            Showing {((currentPage - 1) * pageSize) + 1} to {Math.min(currentPage * pageSize, totalCount)} of {totalCount} assessments
          </p>
          <div className="flex space-x-2">
            <button 
              onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className="px-3 py-1 border border-gray-300 rounded-md text-sm hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Previous
            </button>
            {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
              let pageNum = i + 1
              if (totalPages > 5 && currentPage > 3) {
                pageNum = currentPage - 2 + i
                if (pageNum > totalPages) pageNum = totalPages - 4 + i
              }
              return (
                <button
                  key={pageNum}
                  onClick={() => setCurrentPage(pageNum)}
                  className={`px-3 py-1 rounded-md text-sm ${
                    currentPage === pageNum 
                      ? 'bg-primary text-white' 
                      : 'border border-gray-300 hover:bg-gray-50'
                  }`}
                >
                  {pageNum}
                </button>
              )
            })}
            <button 
              onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
              className="px-3 py-1 border border-gray-300 rounded-md text-sm hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default PatientHistory
