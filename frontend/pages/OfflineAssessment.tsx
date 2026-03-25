import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { PatientFormData, PredictionResult, saveAssessment } from '../services/offlineDb'
import { predictLocally, localPredictor } from '../services/onnxInference'
import { useSyncStatus } from '../services/syncService'

interface FormStep {
  id: string
  title: string
  fields: FormField[]
}

interface FormField {
  name: keyof PatientFormData
  label: string
  type: 'number' | 'select' | 'checkbox'
  required?: boolean
  min?: number
  max?: number
  step?: number
  options?: { value: string | number; label: string }[]
  helpText?: string
}

const FORM_STEPS: FormStep[] = [
  {
    id: 'demographics',
    title: 'Patient Information',
    fields: [
      { 
        name: 'age', 
        label: 'Age (years)', 
        type: 'number', 
        required: true, 
        min: 40, 
        max: 80,
        helpText: 'Enter patient age between 40-80 years'
      },
      {
        name: 'education_level',
        label: 'Education Level',
        type: 'select',
        required: true,
        options: [
          { value: 1, label: 'Less than 8 years' },
          { value: 2, label: '8-11 years' },
          { value: 3, label: 'High school' },
          { value: 4, label: 'Some college' },
          { value: 5, label: 'College graduate' },
          { value: 6, label: 'Postgraduate' },
        ]
      },
      {
        name: 'marital_status',
        label: 'Marital Status',
        type: 'select',
        required: true,
        options: [
          { value: 'married', label: 'Married' },
          { value: 'single', label: 'Single' },
          { value: 'divorced', label: 'Divorced' },
          { value: 'widowed', label: 'Widowed' },
        ]
      },
    ]
  },
  {
    id: 'reproductive',
    title: 'Reproductive History',
    fields: [
      {
        name: 'age_at_menarche',
        label: 'Age at First Period',
        type: 'number',
        required: true,
        min: 8,
        max: 20,
        helpText: 'Age when menstruation started'
      },
      {
        name: 'number_of_live_births',
        label: 'Number of Children',
        type: 'number',
        required: true,
        min: 0,
        max: 15
      },
      {
        name: 'age_at_first_birth',
        label: 'Age at First Child (if any)',
        type: 'number',
        min: 10,
        max: 50,
        helpText: 'Leave empty if no children'
      },
      {
        name: 'number_of_relatives_with_bc',
        label: 'Relatives with Breast Cancer',
        type: 'number',
        required: true,
        min: 0,
        max: 5,
        helpText: 'Mother, sisters, daughters'
      },
    ]
  },
  {
    id: 'health',
    title: 'Health Information',
    fields: [
      {
        name: 'current_bmi',
        label: 'Current BMI',
        type: 'number',
        required: true,
        min: 15,
        max: 50,
        step: 0.1,
        helpText: 'Body Mass Index (kg/m²)'
      },
      {
        name: 'bmi_at_20',
        label: 'BMI at Age 20 (if known)',
        type: 'number',
        min: 15,
        max: 50,
        step: 0.1
      },
      {
        name: 'hormone_therapy',
        label: 'Currently using hormone therapy?',
        type: 'checkbox'
      },
      {
        name: 'years_of_hormone_use',
        label: 'Years of Hormone Use',
        type: 'number',
        min: 0,
        max: 30
      },
    ]
  },
  {
    id: 'history',
    title: 'Medical History',
    fields: [
      {
        name: 'personal_history_cancer',
        label: 'Personal history of any cancer?',
        type: 'checkbox'
      },
      {
        name: 'benign_breast_disease',
        label: 'History of benign breast disease?',
        type: 'checkbox'
      },
      {
        name: 'family_history_cancer',
        label: 'Family history of breast cancer?',
        type: 'checkbox'
      },
      {
        name: 'pack_years_smoking',
        label: 'Pack-Years of Smoking',
        type: 'number',
        min: 0,
        max: 100,
        helpText: '0 if never smoked'
      },
    ]
  },
]

const defaultFormData: PatientFormData = {
  age: 50,
  race: 'asian',
  education_level: 3,
  marital_status: 'married',
  occupation: 1,
  age_at_menarche: 13,
  number_of_live_births: 2,
  number_of_relatives_with_bc: 0,
  current_bmi: 24,
  personal_history_cancer: false,
  benign_breast_disease: false,
  family_history_cancer: false,
  hormone_therapy: false,
  aspirin_use: false,
  ibuprofen_use: false,
}

const OfflineAssessment: React.FC = () => {
  const navigate = useNavigate()
  const { isOnline, pendingCount } = useSyncStatus()
  
  const [currentStep, setCurrentStep] = useState(0)
  const [formData, setFormData] = useState<PatientFormData>(defaultFormData)
  const [patientName, setPatientName] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [savedId, setSavedId] = useState<string | null>(null)
  const [modelReady, setModelReady] = useState(false)

  useEffect(() => {
    localPredictor.initialize().then(() => {
      setModelReady(true)
    }).catch(console.error)
  }, [])

  const currentStepData = FORM_STEPS[currentStep]
  const isLastStep = currentStep === FORM_STEPS.length - 1
  const progress = ((currentStep + 1) / FORM_STEPS.length) * 100

  const handleFieldChange = (name: keyof PatientFormData, value: unknown) => {
    setFormData(prev => ({ ...prev, [name]: value }))
  }

  const nextStep = () => {
    if (currentStep < FORM_STEPS.length - 1) {
      setCurrentStep(prev => prev + 1)
    }
  }

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(prev => prev - 1)
    }
  }

  const handleSubmit = async () => {
    setIsSubmitting(true)
    try {
      // Run local prediction
      const prediction = await predictLocally(formData)
      setResult(prediction)

      // Save to local database
      const assessment = await saveAssessment(formData, prediction, {
        patient_name: patientName || undefined,
      })
      setSavedId(assessment.id)

    } catch (error) {
      console.error('Assessment failed:', error)
      alert('Failed to complete assessment. Please try again.')
    } finally {
      setIsSubmitting(false)
    }
  }

  const startNewAssessment = () => {
    setFormData(defaultFormData)
    setPatientName('')
    setResult(null)
    setSavedId(null)
    setCurrentStep(0)
  }

  const getRiskColor = (category: string) => {
    switch (category) {
      case 'high': return 'bg-red-500'
      case 'moderate': return 'bg-amber-500'
      default: return 'bg-emerald-500'
    }
  }

  const getRiskBgColor = (category: string) => {
    switch (category) {
      case 'high': return 'bg-red-50 border-red-200'
      case 'moderate': return 'bg-amber-50 border-amber-200'
      default: return 'bg-emerald-50 border-emerald-200'
    }
  }

  // Result Screen
  if (result) {
    return (
      <div className="min-h-screen bg-gray-50 p-4">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="max-w-lg mx-auto"
        >
          {/* Header */}
          <div className="text-center mb-6">
            <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-white ${getRiskColor(result.risk_category)} mb-4`}>
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Assessment Complete
            </div>
            <h1 className="text-2xl font-bold text-gray-900">Risk Assessment Result</h1>
            {patientName && <p className="text-gray-600 mt-1">{patientName}</p>}
          </div>

          {/* Risk Card */}
          <div className={`rounded-2xl border-2 p-6 mb-6 ${getRiskBgColor(result.risk_category)}`}>
            <div className="text-center">
              <div className="text-6xl font-bold text-gray-900 mb-2">
                {(result.risk_score * 100).toFixed(1)}%
              </div>
              <div className="text-xl font-semibold text-gray-700 mb-1">
                5-Year Risk Score
              </div>
              <div className={`inline-block px-4 py-2 rounded-full text-white font-bold text-lg ${getRiskColor(result.risk_category)}`}>
                {result.risk_category.toUpperCase()} RISK
              </div>
            </div>
          </div>

          {/* Details */}
          <div className="bg-white rounded-xl p-4 mb-6 space-y-3">
            <div className="flex justify-between items-center py-2 border-b">
              <span className="text-gray-600">Computed</span>
              <span className="font-medium flex items-center gap-2">
                {result.computed_locally ? (
                  <>
                    <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
                    On Device
                  </>
                ) : (
                  <>
                    <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                    Server
                  </>
                )}
              </span>
            </div>
            <div className="flex justify-between items-center py-2 border-b">
              <span className="text-gray-600">Confidence</span>
              <span className="font-medium">{(result.confidence * 100).toFixed(0)}%</span>
            </div>
            <div className="flex justify-between items-center py-2 border-b">
              <span className="text-gray-600">Sync Status</span>
              <span className="font-medium flex items-center gap-2">
                <span className="w-2 h-2 bg-amber-500 rounded-full animate-pulse"></span>
                Pending
              </span>
            </div>
            <div className="flex justify-between items-center py-2">
              <span className="text-gray-600">Assessment ID</span>
              <span className="font-mono text-sm text-gray-500">{savedId?.slice(0, 12)}...</span>
            </div>
          </div>

          {/* Recommendations based on risk */}
          <div className="bg-white rounded-xl p-4 mb-6">
            <h3 className="font-semibold text-gray-900 mb-3">Recommendations</h3>
            {result.risk_category === 'high' && (
              <ul className="space-y-2 text-sm text-gray-700">
                <li className="flex items-start gap-2">
                  <span className="text-red-500 mt-0.5">•</span>
                  Refer to specialist for further evaluation
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-red-500 mt-0.5">•</span>
                  Consider genetic counseling and testing
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-red-500 mt-0.5">•</span>
                  Discuss risk-reducing medications
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-red-500 mt-0.5">•</span>
                  Enhanced screening recommended
                </li>
              </ul>
            )}
            {result.risk_category === 'moderate' && (
              <ul className="space-y-2 text-sm text-gray-700">
                <li className="flex items-start gap-2">
                  <span className="text-amber-500 mt-0.5">•</span>
                  Annual mammogram recommended
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-amber-500 mt-0.5">•</span>
                  Regular breast self-examination
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-amber-500 mt-0.5">•</span>
                  Lifestyle modifications may help
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-amber-500 mt-0.5">•</span>
                  Follow up in 12 months
                </li>
              </ul>
            )}
            {result.risk_category === 'low' && (
              <ul className="space-y-2 text-sm text-gray-700">
                <li className="flex items-start gap-2">
                  <span className="text-emerald-500 mt-0.5">•</span>
                  Continue age-appropriate screening
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-emerald-500 mt-0.5">•</span>
                  Maintain healthy lifestyle
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-emerald-500 mt-0.5">•</span>
                  Regular check-ups as recommended
                </li>
              </ul>
            )}
          </div>

          {/* Actions */}
          <div className="space-y-3">
            <button
              onClick={startNewAssessment}
              className="w-full py-4 bg-primary text-white rounded-xl font-semibold text-lg touch-manipulation"
            >
              New Assessment
            </button>
            <button
              onClick={() => navigate('/local-history')}
              className="w-full py-4 bg-gray-100 text-gray-700 rounded-xl font-semibold text-lg touch-manipulation"
            >
              View All Assessments ({pendingCount} pending sync)
            </button>
          </div>
        </motion.div>
      </div>
    )
  }

  // Form Screen
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Offline Banner */}
      <div className={`px-4 py-2 text-center text-sm font-medium ${isOnline ? 'bg-emerald-500 text-white' : 'bg-amber-500 text-white'}`}>
        <span className="inline-flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${isOnline ? 'bg-white' : 'bg-white animate-pulse'}`}></span>
          {isOnline ? 'Online' : 'Offline Mode'} • {pendingCount} pending sync
        </span>
      </div>

      {/* Progress Bar */}
      <div className="bg-white px-4 py-3 shadow-sm">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700">
            Step {currentStep + 1} of {FORM_STEPS.length}
          </span>
          <span className="text-sm text-gray-500">{currentStepData.title}</span>
        </div>
        <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-primary rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>
      </div>

      {/* Patient Name (on first step) */}
      {currentStep === 0 && (
        <div className="px-4 pt-4">
          <div className="bg-white rounded-xl p-4 shadow-sm">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Patient Name (Optional)
            </label>
            <input
              type="text"
              value={patientName}
              onChange={(e) => setPatientName(e.target.value)}
              placeholder="Enter patient name for reference"
              className="w-full px-4 py-3 border border-gray-300 rounded-xl text-lg focus:ring-2 focus:ring-primary focus:border-primary"
            />
          </div>
        </div>
      )}

      {/* Form Fields */}
      <div className="px-4 py-4">
        <AnimatePresence mode="wait">
          <motion.div
            key={currentStep}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="space-y-4"
          >
            {currentStepData.fields.map((field) => (
              <div key={field.name} className="bg-white rounded-xl p-4 shadow-sm">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  {field.label}
                  {field.required && <span className="text-red-500 ml-1">*</span>}
                </label>
                
                {field.type === 'number' && (
                  <input
                    type="number"
                    value={formData[field.name] as number || ''}
                    onChange={(e) => handleFieldChange(field.name, e.target.value ? Number(e.target.value) : undefined)}
                    min={field.min}
                    max={field.max}
                    step={field.step || 1}
                    className="w-full px-4 py-4 border border-gray-300 rounded-xl text-xl font-medium focus:ring-2 focus:ring-primary focus:border-primary touch-manipulation"
                    inputMode="numeric"
                  />
                )}

                {field.type === 'select' && (
                  <select
                    value={formData[field.name] as string | number || ''}
                    onChange={(e) => handleFieldChange(field.name, e.target.value)}
                    className="w-full px-4 py-4 border border-gray-300 rounded-xl text-lg focus:ring-2 focus:ring-primary focus:border-primary appearance-none bg-white touch-manipulation"
                  >
                    <option value="">Select...</option>
                    {field.options?.map((opt) => (
                      <option key={opt.value} value={opt.value}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                )}

                {field.type === 'checkbox' && (
                  <button
                    type="button"
                    onClick={() => handleFieldChange(field.name, !formData[field.name])}
                    className={`w-full py-4 rounded-xl text-lg font-medium transition-colors touch-manipulation ${
                      formData[field.name]
                        ? 'bg-primary text-white'
                        : 'bg-gray-100 text-gray-700 border border-gray-300'
                    }`}
                  >
                    {formData[field.name] ? 'Yes' : 'No'}
                  </button>
                )}

                {field.helpText && (
                  <p className="mt-2 text-sm text-gray-500">{field.helpText}</p>
                )}
              </div>
            ))}
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Navigation Buttons */}
      <div className="fixed bottom-0 left-0 right-0 bg-white border-t p-4 safe-area-inset-bottom">
        <div className="flex gap-3 max-w-lg mx-auto">
          {currentStep > 0 && (
            <button
              onClick={prevStep}
              className="flex-1 py-4 bg-gray-100 text-gray-700 rounded-xl font-semibold text-lg touch-manipulation"
            >
              Back
            </button>
          )}
          
          {isLastStep ? (
            <button
              onClick={handleSubmit}
              disabled={isSubmitting || !modelReady}
              className="flex-1 py-4 bg-primary text-white rounded-xl font-semibold text-lg disabled:opacity-50 touch-manipulation"
            >
              {isSubmitting ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Calculating...
                </span>
              ) : (
                'Calculate Risk'
              )}
            </button>
          ) : (
            <button
              onClick={nextStep}
              className="flex-1 py-4 bg-primary text-white rounded-xl font-semibold text-lg touch-manipulation"
            >
              Next
            </button>
          )}
        </div>
      </div>

      {/* Bottom padding for fixed buttons */}
      <div className="h-24"></div>
    </div>
  )
}

export default OfflineAssessment
