import {
  getPendingAssessments,
  getFailedAssessments,
  updateAssessmentSyncStatus,
  OfflineAssessment,
  getSetting,
  setSetting,
} from './offlineDb'
import { transformFormToRequest, predictRisk } from './api'

type SyncEventType = 'start' | 'progress' | 'complete' | 'error' | 'online' | 'offline'

interface SyncEvent {
  type: SyncEventType
  data?: {
    total?: number
    synced?: number
    failed?: number
    current?: string
    error?: string
  }
}

type SyncEventListener = (event: SyncEvent) => void

class SyncService {
  private listeners: SyncEventListener[] = []
  private isSyncing = false
  private isOnline = navigator.onLine
  private syncInterval: number | null = null
  private retryTimeouts: Map<string, number> = new Map()

  constructor() {
    this.setupNetworkListeners()
  }

  private setupNetworkListeners(): void {
    window.addEventListener('online', () => {
      this.isOnline = true
      this.emit({ type: 'online' })
      this.syncPending()
    })

    window.addEventListener('offline', () => {
      this.isOnline = false
      this.emit({ type: 'offline' })
    })
  }

  subscribe(listener: SyncEventListener): () => void {
    this.listeners.push(listener)
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener)
    }
  }

  private emit(event: SyncEvent): void {
    this.listeners.forEach(listener => listener(event))
  }

  getOnlineStatus(): boolean {
    return this.isOnline
  }

  getIsSyncing(): boolean {
    return this.isSyncing
  }

  async syncPending(): Promise<{
    synced: number
    failed: number
    total: number
  }> {
    if (!this.isOnline) {
      console.log('Offline, skipping sync')
      return { synced: 0, failed: 0, total: 0 }
    }

    if (this.isSyncing) {
      console.log('Sync already in progress')
      return { synced: 0, failed: 0, total: 0 }
    }

    this.isSyncing = true
    this.emit({ type: 'start' })

    const pending = await getPendingAssessments()
    const failed = await getFailedAssessments()
    const toSync = [...pending, ...failed.filter(a => a.sync_attempts < 5)]

    const results = { synced: 0, failed: 0, total: toSync.length }

    this.emit({
      type: 'progress',
      data: { total: results.total, synced: 0, failed: 0 }
    })

    for (const assessment of toSync) {
      try {
        await this.syncAssessment(assessment)
        results.synced++
      } catch (error) {
        results.failed++
        console.error(`Failed to sync assessment ${assessment.id}:`, error)
      }

      this.emit({
        type: 'progress',
        data: {
          total: results.total,
          synced: results.synced,
          failed: results.failed,
          current: assessment.id
        }
      })
    }

    this.isSyncing = false
    await setSetting('lastSyncTime', new Date().toISOString())

    this.emit({
      type: 'complete',
      data: results
    })

    return results
  }

  private async syncAssessment(assessment: OfflineAssessment): Promise<void> {
    await updateAssessmentSyncStatus(assessment.id, 'syncing')

    try {
      // Transform the stored patient data to API format
      const requestData = transformFormToRequest(assessment.patient_data)
      
      // Add metadata
      requestData.patient_id = assessment.patient_id
      
      // Send to server
      const response = await predictRisk(requestData)

      // Mark as synced
      await updateAssessmentSyncStatus(assessment.id, 'synced', {
        server_id: response.prediction_id || response.id
      })

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      
      await updateAssessmentSyncStatus(assessment.id, 'failed', {
        error: errorMessage
      })

      // Schedule retry with exponential backoff
      this.scheduleRetry(assessment.id)

      throw error
    }
  }

  private scheduleRetry(assessmentId: string): void {
    // Clear any existing retry
    const existingTimeout = this.retryTimeouts.get(assessmentId)
    if (existingTimeout) {
      window.clearTimeout(existingTimeout)
    }

    // Exponential backoff: 30s, 1m, 2m, 4m, 8m
    const attempts = this.retryTimeouts.size
    const delay = Math.min(30000 * Math.pow(2, attempts), 480000) // Max 8 minutes

    const timeout = window.setTimeout(() => {
      this.retryTimeouts.delete(assessmentId)
      if (this.isOnline) {
        this.syncPending()
      }
    }, delay)

    this.retryTimeouts.set(assessmentId, timeout)
  }

  startAutoSync(intervalMs: number = 60000): void {
    this.stopAutoSync()
    
    // Initial sync
    this.syncPending()

    // Set up interval
    this.syncInterval = window.setInterval(() => {
      if (this.isOnline && !this.isSyncing) {
        this.syncPending()
      }
    }, intervalMs)
  }

  stopAutoSync(): void {
    if (this.syncInterval) {
      window.clearInterval(this.syncInterval)
      this.syncInterval = null
    }
  }

  async getLastSyncTime(): Promise<Date | null> {
    const timeStr = await getSetting<string>('lastSyncTime')
    return timeStr ? new Date(timeStr) : null
  }

  async getSyncStatus(): Promise<{
    isOnline: boolean
    isSyncing: boolean
    pendingCount: number
    failedCount: number
    lastSyncTime: Date | null
  }> {
    const pending = await getPendingAssessments()
    const failed = await getFailedAssessments()
    const lastSyncTime = await this.getLastSyncTime()

    return {
      isOnline: this.isOnline,
      isSyncing: this.isSyncing,
      pendingCount: pending.length,
      failedCount: failed.length,
      lastSyncTime
    }
  }

  async forceSyncAll(): Promise<void> {
    // Reset failed assessments to pending and sync
    const failed = await getFailedAssessments()
    for (const assessment of failed) {
      await updateAssessmentSyncStatus(assessment.id, 'pending')
    }
    await this.syncPending()
  }
}

// Singleton instance
export const syncService = new SyncService()

// React hook for sync status
import { useState, useEffect, useCallback } from 'react'

export function useSyncStatus() {
  const [isOnline, setIsOnline] = useState(navigator.onLine)
  const [isSyncing, setIsSyncing] = useState(false)
  const [pendingCount, setPendingCount] = useState(0)
  const [failedCount, setFailedCount] = useState(0)
  const [lastSyncTime, setLastSyncTime] = useState<Date | null>(null)
  const [syncProgress, setSyncProgress] = useState<{ synced: number; total: number } | null>(null)

  const refreshStatus = useCallback(async () => {
    const status = await syncService.getSyncStatus()
    setIsOnline(status.isOnline)
    setIsSyncing(status.isSyncing)
    setPendingCount(status.pendingCount)
    setFailedCount(status.failedCount)
    setLastSyncTime(status.lastSyncTime)
  }, [])

  useEffect(() => {
    refreshStatus()

    const unsubscribe = syncService.subscribe((event) => {
      switch (event.type) {
        case 'online':
          setIsOnline(true)
          break
        case 'offline':
          setIsOnline(false)
          break
        case 'start':
          setIsSyncing(true)
          setSyncProgress({ synced: 0, total: 0 })
          break
        case 'progress':
          if (event.data) {
            setSyncProgress({
              synced: event.data.synced || 0,
              total: event.data.total || 0
            })
          }
          break
        case 'complete':
          setIsSyncing(false)
          setSyncProgress(null)
          refreshStatus()
          break
        case 'error':
          refreshStatus()
          break
      }
    })

    // Start auto-sync
    syncService.startAutoSync()

    return () => {
      unsubscribe()
    }
  }, [refreshStatus])

  const syncNow = useCallback(async () => {
    if (isOnline && !isSyncing) {
      await syncService.syncPending()
    }
  }, [isOnline, isSyncing])

  const forceSyncAll = useCallback(async () => {
    await syncService.forceSyncAll()
  }, [])

  return {
    isOnline,
    isSyncing,
    pendingCount,
    failedCount,
    lastSyncTime,
    syncProgress,
    syncNow,
    forceSyncAll,
    refreshStatus
  }
}
