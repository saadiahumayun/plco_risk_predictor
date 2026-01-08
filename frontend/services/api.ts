// services/api.ts
import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
})

// Toggle between mock and real API
export const USE_MOCK_API = false // Backend is running - use real API

export interface PredictionRequest {
  patient_id?: string
  mr_number?: string  // Medical Record Number
  demographics: {
    age: number
    race: string
    education?: string
    marital_status?: string
    occupation?: string
  }
  reproductive_history: {
    age_at_menarche: number
    age_at_first_birth?: number
    number_of_live_births: number
    breastfeeding_months?: number
    first_degree_bc: number
  }
  body_metrics: {
    current_bmi: number
    bmi_at_age_20?: number
    bmi_at_age_50?: number
    height_cm: number
    weight_kg: number
  }
  medical_history: {
    personal_cancer_history: boolean
    benign_breast_disease: boolean
    breast_biopsies: number
    hormone_therapy_current: boolean
    hormone_therapy_years?: number
    aspirin_use?: boolean
    ibuprofen_use?: boolean
  }
  lifestyle?: {
    smoking_status: string
    pack_years?: number
    alcohol_drinks_per_week?: number
    physical_activity_hours?: number
    birth_control_ever?: boolean
    birth_control_years?: number
  }
}

export interface PredictionResponse {
  prediction_id: string
  risk_score: number
  risk_category: 'low' | 'moderate' | 'high'
  confidence_interval: { lower: number; upper: number }
  percentile: number
  relative_risk?: number
  feature_importance: Array<{
    feature: string
    importance: number
    value: number
    contribution: number
    description: string
  }>
  model_comparison: {
    ga_model: { risk_score: number; features_used: number; model_version: string }
    baseline_model: { risk_score: number; features_used: number; model_version: string }
    agreement: number
    recommended_model: string
  }
  recommendations?: Array<{
    type: string
    priority: string
    action: string
    rationale: string
    potential_impact?: string
  }>
  screening?: {
    recommendation: string
    frequency: string
    next_date?: string
    additional_imaging?: string[]
    rationale: string
  }
  model_version: string
  processing_time_ms: number
  timestamp: string
}

// Transform flat form data to backend format
export function transformFormToRequest(formData: any): PredictionRequest {
  return {
    patient_id: formData.mr_number || `PAT-${Date.now()}`,  // Use MR number if provided
    mr_number: formData.mr_number,  // Also send as separate field for database
    demographics: {
      age: Number(formData.age),
      race: formData.race,
      education: formData.education_level?.toString(),
      marital_status: formData.marital_status,
      occupation: formData.occupation?.toString(),
    },
    reproductive_history: {
      age_at_menarche: Number(formData.age_at_menarche),
      age_at_first_birth: formData.age_at_first_birth ? Number(formData.age_at_first_birth) : undefined,
      number_of_live_births: Number(formData.number_of_live_births || 0),
      first_degree_bc: Number(formData.number_of_relatives_with_bc || 0),
    },
    body_metrics: {
      current_bmi: Number(formData.current_bmi),
      bmi_at_age_20: formData.bmi_at_20 ? Number(formData.bmi_at_20) : undefined,
      bmi_at_age_50: formData.bmi_at_50 ? Number(formData.bmi_at_50) : undefined,
      height_cm: 165, // Default, should be added to form
      weight_kg: Number(formData.current_bmi) * (1.65 * 1.65), // Calculate from BMI
    },
    medical_history: {
      personal_cancer_history: Boolean(formData.personal_history_cancer),
      benign_breast_disease: Boolean(formData.benign_breast_disease),
      breast_biopsies: formData.benign_breast_disease ? 1 : 0,
      hormone_therapy_current: Boolean(formData.hormone_therapy),
      hormone_therapy_years: formData.years_of_hormone_use ? Number(formData.years_of_hormone_use) : undefined,
      aspirin_use: Boolean(formData.aspirin_use),
      ibuprofen_use: Boolean(formData.ibuprofen_use),
    },
    lifestyle: {
      smoking_status: formData.pack_years_smoking > 0 ? 'former' : 'never',
      pack_years: formData.pack_years_smoking ? Number(formData.pack_years_smoking) : undefined,
      birth_control_years: formData.birth_control_years ? Number(formData.birth_control_years) : undefined,
    },
  }
}

// Mock prediction response
export function getMockPrediction(formData: any): PredictionResponse {
  const riskScore = 0.183
  return {
    prediction_id: `PRED-${Date.now()}`,
    risk_score: riskScore,
    risk_category: riskScore > 0.2 ? 'high' : riskScore > 0.1 ? 'moderate' : 'low',
    confidence_interval: { lower: 0.156, upper: 0.210 },
    percentile: 85,
    relative_risk: 1.22,
    feature_importance: [
      {
        feature: 'family_history',
        importance: 0.15,
        value: Number(formData.number_of_relatives_with_bc || 0),
        contribution: 0.045,
        description: 'First-degree relative with breast cancer',
      },
      {
        feature: 'age',
        importance: 0.12,
        value: Number(formData.age),
        contribution: 0.032,
        description: `Age ${formData.age} years`,
      },
      {
        feature: 'bmi_current',
        importance: 0.08,
        value: Number(formData.current_bmi),
        contribution: 0.021,
        description: `BMI of ${formData.current_bmi}`,
      },
    ],
    model_comparison: {
      ga_model: { risk_score: 0.183, features_used: 28, model_version: '1.2' },
      baseline_model: { risk_score: 0.179, features_used: 92, model_version: '1.2' },
      agreement: 0.96,
      recommended_model: 'ga_model',
    },
    recommendations: [
      {
        type: 'lifestyle',
        priority: 'high',
        action: 'Maintain healthy BMI (<25)',
        rationale: 'Weight management can reduce risk',
        potential_impact: '2-3% risk reduction',
      },
      {
        type: 'lifestyle',
        priority: 'medium',
        action: 'Increase physical activity to 7+ hours/week',
        rationale: 'Regular exercise is protective',
        potential_impact: '1-2% risk reduction',
      },
    ],
    screening: {
      recommendation: 'Annual mammography',
      frequency: 'Yearly',
      next_date: new Date(Date.now() + 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
      additional_imaging: ['Ultrasound'],
      rationale: 'Moderate risk category warrants annual screening',
    },
    model_version: 'ga_model_v1.2',
    processing_time_ms: 145,
    timestamp: new Date().toISOString(),
  }
}

// API calls
export async function predictRisk(formData: any): Promise<PredictionResponse> {
  if (USE_MOCK_API) {
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 1500))
    return getMockPrediction(formData)
  }

  const request = transformFormToRequest(formData)
  const response = await api.post<PredictionResponse>('/predict', request)
  return response.data
}

export async function getModelInfo() {
  const response = await api.get('/models/info')
  return response.data
}

export async function getHealthStatus() {
  const response = await api.get('/health/detailed')
  return response.data
}

export default api

