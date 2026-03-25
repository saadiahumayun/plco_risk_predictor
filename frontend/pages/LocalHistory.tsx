import React, { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Link } from 'react-router-dom'
import {
  getAllAssessments,
  getAssessmentStats,
  deleteAssessment,
  OfflineAssessment,
  exportAllData,
  SyncStatus,
} from '../services/offlineDb'
import { useSyncStatus } from '../services/syncService'
import { format, formatDistanceToNow } from 'date-fns'

type FilterType = 'all' | 'pending' | 'synced' | 'failed'
type RiskFilter = 'all' | 'low' | 'moderate' | 'high'

const LocalHistory: React.FC = () => {
  const { isOnline, isSyncing, syncNow, pendingCount, lastSyncTime } = useSyncStatus()
  
  const [assessments, setAssessments] = useState<OfflineAssessment[]>([])
  const [stats, setStats] = useState<Awaited<ReturnType<typeof getAssessmentStats>> | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [syncFilter, setSyncFilter] = useState<FilterType>('all')
  const [riskFilter, setRiskFilter] = useState<RiskFilter>('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedAssessment, setSelectedAssessment] = useState<OfflineAssessment | null>(null)
  const [showExportModal, setShowExportModal] = useState(false)

  const loadData = useCallback(async () => {
    setIsLoading(true)
    try {
      const [allAssessments, assessmentStats] = await Promise.all([
        getAllAssessments(),
        getAssessmentStats()
      ])
      setAssessments(allAssessments)
      setStats(assessmentStats)
    } catch (error) {
      console.error('Failed to load assessments:', error)
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    loadData()
  }, [loadData])

  // Refresh after sync
  useEffect(() => {
    if (!isSyncing) {
      loadData()
    }
  }, [isSyncing, loadData])

  const filteredAssessments = assessments.filter(a => {
    // Sync status filter
    if (syncFilter !== 'all' && a.sync_status !== syncFilter) return false
    
    // Risk category filter
    if (riskFilter !== 'all' && a.prediction.risk_category !== riskFilter) return false
    
    // Search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      const matchesId = a.id.toLowerCase().includes(query)
      const matchesPatient = a.patient_name?.toLowerCase().includes(query)
      const matchesPatientId = a.patient_id.toLowerCase().includes(query)
      if (!matchesId && !matchesPatient && !matchesPatientId) return false
    }
    
    return true
  })

  const handleDelete = async (id: string) => {
    if (window.confirm('Delete this assessment? This cannot be undone.')) {
      await deleteAssessment(id)
      await loadData()
      setSelectedAssessment(null)
    }
  }

  const handleExport = async () => {
    try {
      const data = await exportAllData()
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `candetect-export-${format(new Date(), 'yyyy-MM-dd')}.json`
      a.click()
      URL.revokeObjectURL(url)
      setShowExportModal(false)
    } catch (error) {
      console.error('Export failed:', error)
      alert('Failed to export data')
    }
  }

  const getSyncStatusColor = (status: SyncStatus) => {
    switch (status) {
      case 'synced': return 'bg-emerald-500'
      case 'pending': return 'bg-amber-500'
      case 'syncing': return 'bg-blue-500 animate-pulse'
      case 'failed': return 'bg-red-500'
    }
  }

  const getSyncStatusText = (status: SyncStatus) => {
    switch (status) {
      case 'synced': return 'Synced'
      case 'pending': return 'Pending'
      case 'syncing': return 'Syncing...'
      case 'failed': return 'Failed'
    }
  }

  const getRiskColor = (category: string) => {
    switch (category) {
      case 'high': return 'text-red-600 bg-red-50'
      case 'moderate': return 'text-amber-600 bg-amber-50'
      default: return 'text-emerald-600 bg-emerald-50'
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 pb-20">
      {/* Header */}
      <div className="bg-white shadow-sm sticky top-0 z-10">
        <div className="px-4 py-4">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-xl font-bold text-gray-900">Assessment History</h1>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowExportModal(true)}
                className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg"
                title="Export Data"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                </svg>
              </button>
              <button
                onClick={syncNow}
                disabled={!isOnline || isSyncing || pendingCount === 0}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium ${
                  isOnline && pendingCount > 0
                    ? 'bg-primary text-white'
                    : 'bg-gray-100 text-gray-400'
                }`}
              >
                {isSyncing ? (
                  <>
                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Syncing...
                  </>
                ) : (
                  <>
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                    Sync ({pendingCount})
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Connection Status */}
          <div className={`flex items-center justify-between text-sm px-3 py-2 rounded-lg ${isOnline ? 'bg-emerald-50 text-emerald-700' : 'bg-amber-50 text-amber-700'}`}>
            <span className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${isOnline ? 'bg-emerald-500' : 'bg-amber-500 animate-pulse'}`}></span>
              {isOnline ? 'Online' : 'Offline'}
            </span>
            {lastSyncTime && (
              <span className="text-gray-500">
                Last sync: {formatDistanceToNow(lastSyncTime, { addSuffix: true })}
              </span>
            )}
          </div>
        </div>

        {/* Stats */}
        {stats && (
          <div className="grid grid-cols-4 gap-2 px-4 pb-4">
            <div className="bg-gray-50 rounded-lg p-3 text-center">
              <div className="text-2xl font-bold text-gray-900">{stats.total}</div>
              <div className="text-xs text-gray-500">Total</div>
            </div>
            <div className="bg-emerald-50 rounded-lg p-3 text-center">
              <div className="text-2xl font-bold text-emerald-600">{stats.byRiskCategory.low}</div>
              <div className="text-xs text-emerald-600">Low</div>
            </div>
            <div className="bg-amber-50 rounded-lg p-3 text-center">
              <div className="text-2xl font-bold text-amber-600">{stats.byRiskCategory.moderate}</div>
              <div className="text-xs text-amber-600">Moderate</div>
            </div>
            <div className="bg-red-50 rounded-lg p-3 text-center">
              <div className="text-2xl font-bold text-red-600">{stats.byRiskCategory.high}</div>
              <div className="text-xs text-red-600">High</div>
            </div>
          </div>
        )}

        {/* Search and Filters */}
        <div className="px-4 pb-4 space-y-3">
          <input
            type="text"
            placeholder="Search by name or ID..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-primary focus:border-primary"
          />
          
          <div className="flex gap-2 overflow-x-auto pb-1">
            {(['all', 'pending', 'synced', 'failed'] as FilterType[]).map((filter) => (
              <button
                key={filter}
                onClick={() => setSyncFilter(filter)}
                className={`px-3 py-1.5 rounded-full text-sm font-medium whitespace-nowrap ${
                  syncFilter === filter
                    ? 'bg-primary text-white'
                    : 'bg-gray-100 text-gray-600'
                }`}
              >
                {filter.charAt(0).toUpperCase() + filter.slice(1)}
              </button>
            ))}
          </div>

          <div className="flex gap-2 overflow-x-auto pb-1">
            {(['all', 'low', 'moderate', 'high'] as RiskFilter[]).map((filter) => (
              <button
                key={filter}
                onClick={() => setRiskFilter(filter)}
                className={`px-3 py-1.5 rounded-full text-sm font-medium whitespace-nowrap ${
                  riskFilter === filter
                    ? filter === 'low' ? 'bg-emerald-500 text-white' :
                      filter === 'moderate' ? 'bg-amber-500 text-white' :
                      filter === 'high' ? 'bg-red-500 text-white' :
                      'bg-primary text-white'
                    : 'bg-gray-100 text-gray-600'
                }`}
              >
                {filter === 'all' ? 'All Risks' : `${filter.charAt(0).toUpperCase() + filter.slice(1)} Risk`}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Assessment List */}
      <div className="px-4 py-4">
        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <svg className="animate-spin h-8 w-8 text-primary" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
          </div>
        ) : filteredAssessments.length === 0 ? (
          <div className="text-center py-12">
            <svg className="w-16 h-16 text-gray-300 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <p className="text-gray-500 mb-4">No assessments found</p>
            <Link
              to="/offline-assessment"
              className="inline-flex items-center gap-2 px-4 py-2 bg-primary text-white rounded-lg font-medium"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              New Assessment
            </Link>
          </div>
        ) : (
          <div className="space-y-3">
            {filteredAssessments.map((assessment) => (
              <motion.div
                key={assessment.id}
                layout
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-white rounded-xl p-4 shadow-sm"
                onClick={() => setSelectedAssessment(assessment)}
              >
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <div className="font-medium text-gray-900">
                      {assessment.patient_name || assessment.patient_id}
                    </div>
                    <div className="text-sm text-gray-500">
                      {format(new Date(assessment.created_at), 'MMM d, yyyy h:mm a')}
                    </div>
                  </div>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getRiskColor(assessment.prediction.risk_category)}`}>
                    {assessment.prediction.risk_category.toUpperCase()}
                  </span>
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="text-2xl font-bold text-gray-900">
                    {(assessment.prediction.risk_score * 100).toFixed(1)}%
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${getSyncStatusColor(assessment.sync_status)}`}></span>
                    <span className="text-sm text-gray-500">{getSyncStatusText(assessment.sync_status)}</span>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </div>

      {/* New Assessment FAB */}
      <Link
        to="/offline-assessment"
        className="fixed bottom-6 right-6 w-14 h-14 bg-primary text-white rounded-full shadow-lg flex items-center justify-center touch-manipulation"
      >
        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
        </svg>
      </Link>

      {/* Assessment Detail Modal */}
      <AnimatePresence>
        {selectedAssessment && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 z-50 flex items-end"
            onClick={() => setSelectedAssessment(null)}
          >
            <motion.div
              initial={{ y: '100%' }}
              animate={{ y: 0 }}
              exit={{ y: '100%' }}
              className="bg-white rounded-t-2xl w-full max-h-[80vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="sticky top-0 bg-white border-b px-4 py-3 flex items-center justify-between">
                <h2 className="text-lg font-bold">Assessment Details</h2>
                <button
                  onClick={() => setSelectedAssessment(null)}
                  className="p-2 hover:bg-gray-100 rounded-full"
                >
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              
              <div className="p-4 space-y-4">
                {/* Risk Score */}
                <div className={`rounded-xl p-4 text-center ${getRiskColor(selectedAssessment.prediction.risk_category)}`}>
                  <div className="text-4xl font-bold">
                    {(selectedAssessment.prediction.risk_score * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm font-medium">
                    {selectedAssessment.prediction.risk_category.toUpperCase()} RISK
                  </div>
                </div>

                {/* Details */}
                <div className="space-y-2">
                  <div className="flex justify-between py-2 border-b">
                    <span className="text-gray-600">Patient</span>
                    <span className="font-medium">{selectedAssessment.patient_name || selectedAssessment.patient_id}</span>
                  </div>
                  <div className="flex justify-between py-2 border-b">
                    <span className="text-gray-600">Date</span>
                    <span className="font-medium">{format(new Date(selectedAssessment.created_at), 'MMM d, yyyy h:mm a')}</span>
                  </div>
                  <div className="flex justify-between py-2 border-b">
                    <span className="text-gray-600">Age</span>
                    <span className="font-medium">{selectedAssessment.patient_data.age} years</span>
                  </div>
                  <div className="flex justify-between py-2 border-b">
                    <span className="text-gray-600">BMI</span>
                    <span className="font-medium">{selectedAssessment.patient_data.current_bmi}</span>
                  </div>
                  <div className="flex justify-between py-2 border-b">
                    <span className="text-gray-600">Family History</span>
                    <span className="font-medium">{selectedAssessment.patient_data.family_history_cancer ? 'Yes' : 'No'}</span>
                  </div>
                  <div className="flex justify-between py-2 border-b">
                    <span className="text-gray-600">Sync Status</span>
                    <span className="flex items-center gap-2">
                      <span className={`w-2 h-2 rounded-full ${getSyncStatusColor(selectedAssessment.sync_status)}`}></span>
                      {getSyncStatusText(selectedAssessment.sync_status)}
                    </span>
                  </div>
                  <div className="flex justify-between py-2 border-b">
                    <span className="text-gray-600">Assessment ID</span>
                    <span className="font-mono text-sm text-gray-500">{selectedAssessment.id}</span>
                  </div>
                </div>

                {/* Actions */}
                <div className="pt-4 space-y-3">
                  <button
                    onClick={() => handleDelete(selectedAssessment.id)}
                    className="w-full py-3 bg-red-50 text-red-600 rounded-xl font-medium"
                  >
                    Delete Assessment
                  </button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Export Modal */}
      <AnimatePresence>
        {showExportModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4"
            onClick={() => setShowExportModal(false)}
          >
            <motion.div
              initial={{ scale: 0.95 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.95 }}
              className="bg-white rounded-xl p-6 max-w-sm w-full"
              onClick={(e) => e.stopPropagation()}
            >
              <h2 className="text-lg font-bold mb-4">Export Data</h2>
              <p className="text-gray-600 mb-6">
                Download all {assessments.length} assessments as a JSON file for backup or reporting.
              </p>
              <div className="space-y-3">
                <button
                  onClick={handleExport}
                  className="w-full py-3 bg-primary text-white rounded-xl font-medium"
                >
                  Download JSON
                </button>
                <button
                  onClick={() => setShowExportModal(false)}
                  className="w-full py-3 bg-gray-100 text-gray-700 rounded-xl font-medium"
                >
                  Cancel
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default LocalHistory
