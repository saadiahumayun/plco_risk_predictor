import { openDB, DBSchema, IDBPDatabase } from 'idb'

export type SyncStatus = 'pending' | 'syncing' | 'synced' | 'failed'

export interface PatientFormData {
  age: number
  race: string
  education_level: number
  marital_status: string
  occupation: number
  age_at_menarche: number
  age_at_first_birth?: number
  number_of_live_births: number
  number_of_relatives_with_bc: number
  birth_control_years?: number
  current_bmi: number
  bmi_at_20?: number
  bmi_at_50?: number
  pack_years_smoking?: number
  personal_history_cancer: boolean
  benign_breast_disease: boolean
  family_history_cancer: boolean
  hormone_therapy: boolean
  years_of_hormone_use?: number
  aspirin_use: boolean
  ibuprofen_use: boolean
}

export interface PredictionResult {
  risk_score: number
  risk_category: 'low' | 'moderate' | 'high'
  confidence: number
  computed_locally: boolean
}

export interface OfflineAssessment {
  id: string
  patient_id: string
  patient_name?: string
  patient_data: PatientFormData
  prediction: PredictionResult
  created_at: Date
  updated_at: Date
  sync_status: SyncStatus
  sync_error?: string
  sync_attempts: number
  server_id?: string
  lhw_id?: string
  location?: {
    latitude?: number
    longitude?: number
    village?: string
  }
}

export interface SyncQueueItem {
  id: string
  assessment_id: string
  created_at: Date
  attempts: number
  last_attempt?: Date
  error?: string
}

interface CANDetectDB extends DBSchema {
  assessments: {
    key: string
    value: OfflineAssessment
    indexes: {
      'by-sync-status': SyncStatus
      'by-created-at': Date
      'by-patient-id': string
    }
  }
  sync_queue: {
    key: string
    value: SyncQueueItem
    indexes: {
      'by-created-at': Date
    }
  }
  settings: {
    key: string
    value: {
      key: string
      value: unknown
    }
  }
}

const DB_NAME = 'candetect-offline'
const DB_VERSION = 1

let dbInstance: IDBPDatabase<CANDetectDB> | null = null

export async function getDB(): Promise<IDBPDatabase<CANDetectDB>> {
  if (dbInstance) return dbInstance

  dbInstance = await openDB<CANDetectDB>(DB_NAME, DB_VERSION, {
    upgrade(db) {
      // Assessments store
      if (!db.objectStoreNames.contains('assessments')) {
        const assessmentStore = db.createObjectStore('assessments', { keyPath: 'id' })
        assessmentStore.createIndex('by-sync-status', 'sync_status')
        assessmentStore.createIndex('by-created-at', 'created_at')
        assessmentStore.createIndex('by-patient-id', 'patient_id')
      }

      // Sync queue store
      if (!db.objectStoreNames.contains('sync_queue')) {
        const syncStore = db.createObjectStore('sync_queue', { keyPath: 'id' })
        syncStore.createIndex('by-created-at', 'created_at')
      }

      // Settings store
      if (!db.objectStoreNames.contains('settings')) {
        db.createObjectStore('settings', { keyPath: 'key' })
      }
    },
  })

  return dbInstance
}

export function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
}

export async function saveAssessment(
  patientData: PatientFormData,
  prediction: PredictionResult,
  options?: {
    patient_id?: string
    patient_name?: string
    lhw_id?: string
    location?: OfflineAssessment['location']
  }
): Promise<OfflineAssessment> {
  const db = await getDB()
  const now = new Date()

  const assessment: OfflineAssessment = {
    id: generateId(),
    patient_id: options?.patient_id || `PAT-${Date.now()}`,
    patient_name: options?.patient_name,
    patient_data: patientData,
    prediction,
    created_at: now,
    updated_at: now,
    sync_status: 'pending',
    sync_attempts: 0,
    lhw_id: options?.lhw_id,
    location: options?.location,
  }

  await db.put('assessments', assessment)
  
  // Add to sync queue
  await addToSyncQueue(assessment.id)

  return assessment
}

export async function getAssessment(id: string): Promise<OfflineAssessment | undefined> {
  const db = await getDB()
  return db.get('assessments', id)
}

export async function getAllAssessments(): Promise<OfflineAssessment[]> {
  const db = await getDB()
  const assessments = await db.getAll('assessments')
  return assessments.sort((a, b) => 
    new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
  )
}

export async function getAssessmentsByStatus(status: SyncStatus): Promise<OfflineAssessment[]> {
  const db = await getDB()
  return db.getAllFromIndex('assessments', 'by-sync-status', status)
}

export async function getPendingAssessments(): Promise<OfflineAssessment[]> {
  return getAssessmentsByStatus('pending')
}

export async function getFailedAssessments(): Promise<OfflineAssessment[]> {
  return getAssessmentsByStatus('failed')
}

export async function updateAssessmentSyncStatus(
  id: string,
  status: SyncStatus,
  options?: {
    server_id?: string
    error?: string
  }
): Promise<void> {
  const db = await getDB()
  const assessment = await db.get('assessments', id)
  
  if (!assessment) {
    throw new Error(`Assessment not found: ${id}`)
  }

  assessment.sync_status = status
  assessment.updated_at = new Date()
  assessment.sync_attempts += 1

  if (options?.server_id) {
    assessment.server_id = options.server_id
  }

  if (options?.error) {
    assessment.sync_error = options.error
  } else if (status === 'synced') {
    assessment.sync_error = undefined
  }

  await db.put('assessments', assessment)
}

export async function deleteAssessment(id: string): Promise<void> {
  const db = await getDB()
  await db.delete('assessments', id)
  await removeFromSyncQueue(id)
}

export async function getAssessmentCount(): Promise<{
  total: number
  pending: number
  synced: number
  failed: number
}> {
  const db = await getDB()
  const all = await db.getAll('assessments')
  
  return {
    total: all.length,
    pending: all.filter(a => a.sync_status === 'pending').length,
    synced: all.filter(a => a.sync_status === 'synced').length,
    failed: all.filter(a => a.sync_status === 'failed').length,
  }
}

export async function getAssessmentStats(): Promise<{
  total: number
  today: number
  thisWeek: number
  thisMonth: number
  byRiskCategory: { low: number; moderate: number; high: number }
  pendingSync: number
}> {
  const db = await getDB()
  const all = await db.getAll('assessments')
  
  const now = new Date()
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate())
  const weekStart = new Date(todayStart)
  weekStart.setDate(weekStart.getDate() - weekStart.getDay())
  const monthStart = new Date(now.getFullYear(), now.getMonth(), 1)

  return {
    total: all.length,
    today: all.filter(a => new Date(a.created_at) >= todayStart).length,
    thisWeek: all.filter(a => new Date(a.created_at) >= weekStart).length,
    thisMonth: all.filter(a => new Date(a.created_at) >= monthStart).length,
    byRiskCategory: {
      low: all.filter(a => a.prediction.risk_category === 'low').length,
      moderate: all.filter(a => a.prediction.risk_category === 'moderate').length,
      high: all.filter(a => a.prediction.risk_category === 'high').length,
    },
    pendingSync: all.filter(a => a.sync_status === 'pending' || a.sync_status === 'failed').length,
  }
}

// Sync Queue Management
async function addToSyncQueue(assessmentId: string): Promise<void> {
  const db = await getDB()
  const item: SyncQueueItem = {
    id: generateId(),
    assessment_id: assessmentId,
    created_at: new Date(),
    attempts: 0,
  }
  await db.put('sync_queue', item)
}

async function removeFromSyncQueue(assessmentId: string): Promise<void> {
  const db = await getDB()
  const all = await db.getAll('sync_queue')
  const item = all.find(i => i.assessment_id === assessmentId)
  if (item) {
    await db.delete('sync_queue', item.id)
  }
}

export async function getSyncQueue(): Promise<SyncQueueItem[]> {
  const db = await getDB()
  return db.getAllFromIndex('sync_queue', 'by-created-at')
}

export async function updateSyncQueueItem(
  id: string,
  updates: Partial<SyncQueueItem>
): Promise<void> {
  const db = await getDB()
  const item = await db.get('sync_queue', id)
  if (item) {
    Object.assign(item, updates)
    await db.put('sync_queue', item)
  }
}

export async function clearSyncedFromQueue(): Promise<void> {
  const synced = await getAssessmentsByStatus('synced')
  for (const assessment of synced) {
    await removeFromSyncQueue(assessment.id)
  }
}

// Settings Management
export async function getSetting<T>(key: string): Promise<T | undefined> {
  const db = await getDB()
  const setting = await db.get('settings', key)
  return setting?.value as T | undefined
}

export async function setSetting<T>(key: string, value: T): Promise<void> {
  const db = await getDB()
  await db.put('settings', { key, value })
}

// Export data for backup/reporting
export async function exportAllData(): Promise<{
  assessments: OfflineAssessment[]
  exportedAt: Date
  deviceInfo: string
}> {
  const assessments = await getAllAssessments()
  return {
    assessments,
    exportedAt: new Date(),
    deviceInfo: navigator.userAgent,
  }
}

// Clear all local data
export async function clearAllData(): Promise<void> {
  const db = await getDB()
  await db.clear('assessments')
  await db.clear('sync_queue')
}
