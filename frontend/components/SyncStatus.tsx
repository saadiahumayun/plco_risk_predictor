import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useSyncStatus } from '../services/syncService'
import { formatDistanceToNow } from 'date-fns'

interface SyncStatusProps {
  minimal?: boolean
}

const SyncStatus: React.FC<SyncStatusProps> = ({ minimal = false }) => {
  const {
    isOnline,
    isSyncing,
    pendingCount,
    failedCount,
    lastSyncTime,
    syncProgress,
    syncNow
  } = useSyncStatus()

  if (minimal) {
    return (
      <div className="flex items-center gap-2">
        <span className={`w-2 h-2 rounded-full ${isOnline ? 'bg-emerald-500' : 'bg-amber-500 animate-pulse'}`}></span>
        {pendingCount > 0 && (
          <span className="text-xs text-gray-500">{pendingCount} pending</span>
        )}
      </div>
    )
  }

  return (
    <div className="fixed bottom-0 left-0 right-0 z-40">
      <AnimatePresence>
        {/* Sync Progress Bar */}
        {isSyncing && syncProgress && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            className="bg-blue-600 text-white px-4 py-2"
          >
            <div className="max-w-7xl mx-auto flex items-center justify-between">
              <div className="flex items-center gap-3">
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                <span className="text-sm font-medium">
                  Syncing {syncProgress.synced} of {syncProgress.total}...
                </span>
              </div>
              <div className="w-24 h-1.5 bg-blue-400 rounded-full overflow-hidden">
                <div
                  className="h-full bg-white rounded-full transition-all duration-300"
                  style={{ width: `${(syncProgress.synced / syncProgress.total) * 100}%` }}
                />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Status Bar */}
      <div className={`px-4 py-2 ${isOnline ? 'bg-slate-800' : 'bg-amber-600'} text-white`}>
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          {/* Connection Status */}
          <div className="flex items-center gap-3">
            <span className={`w-2 h-2 rounded-full ${isOnline ? 'bg-emerald-400' : 'bg-white animate-pulse'}`}></span>
            <span className="text-sm font-medium">
              {isOnline ? 'Online' : 'Offline Mode'}
            </span>
            {lastSyncTime && isOnline && (
              <span className="text-sm text-slate-400 hidden sm:inline">
                Last sync: {formatDistanceToNow(lastSyncTime, { addSuffix: true })}
              </span>
            )}
          </div>

          {/* Pending/Actions */}
          <div className="flex items-center gap-3">
            {(pendingCount > 0 || failedCount > 0) && (
              <div className="flex items-center gap-2">
                {pendingCount > 0 && (
                  <span className="flex items-center gap-1 text-sm">
                    <span className="w-2 h-2 bg-amber-400 rounded-full"></span>
                    {pendingCount} pending
                  </span>
                )}
                {failedCount > 0 && (
                  <span className="flex items-center gap-1 text-sm text-red-300">
                    <span className="w-2 h-2 bg-red-400 rounded-full"></span>
                    {failedCount} failed
                  </span>
                )}
              </div>
            )}

            {isOnline && pendingCount > 0 && !isSyncing && (
              <button
                onClick={syncNow}
                className="flex items-center gap-2 px-3 py-1 bg-white/10 hover:bg-white/20 rounded-lg text-sm font-medium transition-colors"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                Sync Now
              </button>
            )}

            {!isOnline && (
              <span className="text-sm text-amber-100">
                Data saved locally
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default SyncStatus
