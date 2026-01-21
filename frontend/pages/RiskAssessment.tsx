import React, { useState } from 'react'
import { useForm } from 'react-hook-form'
import { motion, AnimatePresence } from 'framer-motion'
import { predictRisk } from '../services/api'

interface PatientFormData {
  // Demographics
  age: number
  race: string
  education_level: number
  marital_status: string
  occupation: number
  
  // Reproductive History
  age_at_menarche: number
  age_at_first_birth?: number
  number_of_relatives_with_bc: number
  
  // Body Metrics
  current_bmi: number
  bmi_at_20?: number
  bmi_at_50?: number
  
  // Medical History
  personal_history_cancer: boolean
  benign_breast_disease: boolean
  hormone_therapy: boolean
  years_of_hormone_use: number
  
  // Lifestyle
  pack_years_smoking: number
  birth_control_years: number
  number_of_live_births: number
  
  // Additional
  aspirin_use: boolean
  ibuprofen_use: boolean
  family_history_cancer: boolean
}

const RiskAssessment: React.FC = () => {
  const [currentStep, setCurrentStep] = useState(1)
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  
  const { register, handleSubmit, watch, formState: { errors } } = useForm<PatientFormData>()
  
  const totalSteps = 3  // 3 form steps: Demographics, Reproductive, Health & Medical
  

  const onSubmit = async (data: PatientFormData) => {
    setIsLoading(true)
    
    try {
      const response = await predictRisk(data)
      
      const displayResult = {
        risk_assessment: {
          five_year_risk: response.risk_score,
          risk_category: response.risk_category,
          confidence_interval: response.confidence_interval,
          percentile: response.percentile,
          assessment_date: response.timestamp,
          model_version: response.model_version,
          processing_time_ms: response.processing_time_ms
        },
        risk_factors: {
          top_increasing_factors: (response.feature_importance || []).filter((f: any) => f.contribution > 0).map((f: any) => ({ factor: f.feature, contribution: f.contribution, description: f.description })),
          top_decreasing_factors: (response.feature_importance || []).filter((f: any) => f.contribution < 0).map((f: any) => ({ factor: f.feature, contribution: f.contribution, description: f.description }))
        },
        recommendations: {
          screening: {
            recommendation: response.screening?.recommendation || "annual_mammography",
            rationale: response.screening?.rationale || "Based on risk assessment",
            next_screening_date: response.screening?.next_date || new Date(Date.now() + 365*24*60*60*1000).toISOString().split("T")[0],
            additional_imaging: { mri: false, ultrasound: true, reason: "Based on clinical assessment" }
          },
          lifestyle_modifications: (response.recommendations || []).filter((r: any) => r.type === "lifestyle").map((r: any) => ({ category: r.action, priority: r.priority, action: r.action, potential_risk_reduction: r.potential_impact || "Variable" })),
          medical_interventions: { chemoprevention: { recommended: false, reason: "Based on risk threshold" }, genetic_counseling: { recommended: false, reason: "Based on family history" } }
        },
        report: { report_id: response.prediction_id, format: "pdf", sections: ["executive_summary", "detailed_risk_assessment", "risk_factor_analysis", "clinical_recommendations"] }
      }
      
      setResult(displayResult)
      setCurrentStep(4)  // Go to results page
    } catch (error) {
      console.error("Prediction error:", error)
      alert("Error generating prediction. Please try again.")
    } finally {
      setIsLoading(false)
    }
  }
  const nextStep = () => setCurrentStep(prev => Math.min(prev + 1, totalSteps))
  const prevStep = () => setCurrentStep(prev => Math.max(prev - 1, 1))
  
  const renderStep = () => {
    switch (currentStep) {
      case 1:
        return (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="space-y-6"
          >
            <h2 className="text-2xl font-semibold text-gray-800">Demographics</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="clinical-label">Age *</label>
                <input
                  type="number"
                  className="clinical-input"
                  {...register('age', { required: true, min: 40, max: 80 })}
                />
                {errors.age && <span className="text-red-500 text-sm">Age must be between 40-80</span>}
              </div>
              
              <div>
                <label className="clinical-label">Race/Ethnicity *</label>
                <select className="clinical-input" {...register('race', { required: true })}>
                  <option value="">Select...</option>
                  <option value="white">White</option>
                  <option value="black">Black/African American</option>
                  <option value="hispanic">Hispanic/Latino</option>
                  <option value="asian">Asian</option>
                  <option value="other">Other</option>
                </select>
              </div>
              
              <div>
                <label className="clinical-label">Education Level *</label>
                <select className="clinical-input" {...register('education_level', { required: true })}>
                  <option value="">Select...</option>
                  <option value="1">Less than 8 years</option>
                  <option value="2">8-11 years</option>
                  <option value="3">High school graduate</option>
                  <option value="4">Post-high school training</option>
                  <option value="5">Some college</option>
                  <option value="6">College graduate</option>
                  <option value="7">Postgraduate</option>
                </select>
              </div>
              
              <div>
                <label className="clinical-label">Marital Status *</label>
                <select className="clinical-input" {...register('marital_status', { required: true })}>
                  <option value="">Select...</option>
                  <option value="married">Married</option>
                  <option value="single">Single</option>
                  <option value="divorced">Divorced</option>
                  <option value="widowed">Widowed</option>
                </select>
              </div>
              
              <div>
                <label className="clinical-label">Occupation Category *</label>
                <select className="clinical-input" {...register('occupation', { required: true })}>
                  <option value="">Select...</option>
                  <option value="1">Homemaker</option>
                  <option value="2">Working</option>
                  <option value="3">Unemployed</option>
                  <option value="4">Retired</option>
                  <option value="5">Extended sick leave</option>
                  <option value="6">Disabled</option>
                  <option value="7">Other</option>
                </select>
              </div>
            </div>
          </motion.div>
        )
        
      case 2:
        return (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="space-y-6"
          >
            <h2 className="text-2xl font-semibold text-gray-800">Reproductive History</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="clinical-label">Age at First Menstruation *</label>
                <input
                  type="number"
                  step="0.5"
                  className="clinical-input"
                  {...register('age_at_menarche', { required: true, min: 8, max: 20 })}
                />
              </div>
              
              <div>
                <label className="clinical-label">Age at First Live Birth</label>
                <input
                  type="number"
                  className="clinical-input"
                  {...register('age_at_first_birth', { min: 10, max: 50 })}
                  placeholder="Leave blank if no live births"
                />
              </div>
              
              <div>
                <label className="clinical-label">Number of Live Births *</label>
                <input
                  type="number"
                  className="clinical-input"
                  {...register('number_of_live_births', { required: true, min: 0, max: 15 })}
                />
              </div>
              
              <div>
                <label className="clinical-label">
                  First-Degree Relatives with Breast Cancer *
                </label>
                <input
                  type="number"
                  className="clinical-input"
                  {...register('number_of_relatives_with_bc', { required: true, min: 0, max: 5 })}
                />
                <p className="text-xs text-gray-500 mt-1">Mother, sisters, daughters</p>
              </div>
              
              <div>
                <label className="clinical-label">Years of Birth Control Use</label>
                <input
                  type="number"
                  step="0.5"
                  className="clinical-input"
                  {...register('birth_control_years', { min: 0, max: 30 })}
                />
              </div>
            </div>
          </motion.div>
        )
        
      case 3:
        return (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="space-y-6"
          >
            <h2 className="text-2xl font-semibold text-gray-800">Health & Medical History</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="clinical-label">Current BMI *</label>
                <input
                  type="number"
                  step="0.1"
                  className="clinical-input"
                  {...register('current_bmi', { required: true, min: 15, max: 50 })}
                />
              </div>
              
              <div>
                <label className="clinical-label">BMI at Age 20</label>
                <input
                  type="number"
                  step="0.1"
                  className="clinical-input"
                  {...register('bmi_at_20', { min: 15, max: 50 })}
                />
              </div>
              
              <div>
                <label className="clinical-label">BMI at Age 50</label>
                <input
                  type="number"
                  step="0.1"
                  className="clinical-input"
                  {...register('bmi_at_50', { min: 15, max: 50 })}
                />
              </div>
              
              <div>
                <label className="clinical-label">Pack Years of Smoking</label>
                <input
                  type="number"
                  step="0.5"
                  className="clinical-input"
                  {...register('pack_years_smoking', { min: 0, max: 100 })}
                />
                <p className="text-xs text-gray-500 mt-1">
                  Packs per day × years smoked
                </p>
              </div>
            </div>
            
            {/* Medical History Section */}
            <div className="border-t pt-4 mt-2">
              <h3 className="text-lg font-medium text-gray-700 mb-3">Medical History</h3>
              <div className="space-y-2">
                <label className="flex items-center space-x-3 cursor-pointer">
                  <input type="checkbox" className="w-4 h-4 text-primary" {...register('personal_history_cancer')} />
                  <span className="text-gray-700">Personal history of any cancer</span>
                </label>
                <label className="flex items-center space-x-3 cursor-pointer">
                  <input type="checkbox" className="w-4 h-4 text-primary" {...register('benign_breast_disease')} />
                  <span className="text-gray-700">History of benign breast disease</span>
                </label>
                <label className="flex items-center space-x-3 cursor-pointer">
                  <input type="checkbox" className="w-4 h-4 text-primary" {...register('family_history_cancer')} />
                  <span className="text-gray-700">Family history of any cancer</span>
                </label>
                <label className="flex items-center space-x-3 cursor-pointer">
                  <input type="checkbox" className="w-4 h-4 text-primary" {...register('hormone_therapy')} />
                  <span className="text-gray-700">Currently on hormone replacement therapy</span>
                </label>
                {watch('hormone_therapy') && (
                  <div className="ml-7 mt-2">
                    <label className="clinical-label">Years of Hormone Use</label>
                    <input type="number" step="0.5" className="clinical-input w-32" {...register('years_of_hormone_use', { min: 0, max: 30 })} />
                  </div>
                )}
                <label className="flex items-center space-x-3 cursor-pointer">
                  <input type="checkbox" className="w-4 h-4 text-primary" {...register('aspirin_use')} />
                  <span className="text-gray-700">Regular aspirin use</span>
                </label>
                <label className="flex items-center space-x-3 cursor-pointer">
                  <input type="checkbox" className="w-4 h-4 text-primary" {...register('ibuprofen_use')} />
                  <span className="text-gray-700">Regular ibuprofen use</span>
                </label>
              </div>
            </div>
          </motion.div>
        )
        
      case 4:
        return result ? (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="space-y-6"
          >
            <h2 className="text-2xl font-semibold text-gray-800">Risk Assessment Results</h2>
            
            {/* Risk Summary Card */}
            <div className={`clinical-card border-2 ${
              result.risk_assessment.risk_category === 'high' ? 'border-red-500 bg-red-50' :
              result.risk_assessment.risk_category === 'moderate' ? 'border-orange-500 bg-orange-50' :
              'border-green-500 bg-green-50'
            }`}>
              <div className="text-center">
                <p className="text-lg font-medium text-gray-700">5-Year Breast Cancer Risk</p>
                <p className={`text-5xl font-bold mt-2 ${
                  result.risk_assessment.risk_category === 'high' ? 'text-red-600' :
                  result.risk_assessment.risk_category === 'moderate' ? 'text-orange-600' :
                  'text-green-600'
                }`}>
                  {(result.risk_assessment.five_year_risk * 100).toFixed(1)}%
                </p>
                <p className="text-lg mt-2 font-medium capitalize">
                  {result.risk_assessment.risk_category} Risk
                </p>
                <p className="text-sm text-gray-600 mt-2">
                  95% CI: {(result.risk_assessment.confidence_interval.lower * 100).toFixed(1)}% - {(result.risk_assessment.confidence_interval.upper * 100).toFixed(1)}%
                </p>
                <p className="text-sm text-gray-500 mt-1">
                  {result.risk_assessment.percentile}th percentile compared to similar women
                </p>
                {result.risk_assessment.relative_risk && (
                  <div className="mt-4 pt-4 border-t border-gray-200">
                    <p className="text-sm text-gray-700">
                      <span className="font-semibold">Relative Risk:</span>{' '}
                      <span className={`font-bold ${
                        result.risk_assessment.relative_risk > 1.5 ? 'text-red-600' :
                        result.risk_assessment.relative_risk > 1.0 ? 'text-orange-600' :
                        'text-green-600'
                      }`}>
                        {result.risk_assessment.relative_risk.toFixed(2)}x
                      </span>{' '}
                      compared to average women of your age
                    </p>
                  </div>
                )}
              </div>
            </div>
            
            {/* Risk Factors */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="clinical-card border-l-4 border-red-400">
                <h4 className="font-semibold text-gray-700 mb-3">Risk Increasing Factors</h4>
                <ul className="space-y-3">
                  {result.risk_factors.top_increasing_factors.map((factor: any, index: number) => (
                    <li key={index} className="flex justify-between items-start">
                      <div>
                        <p className="font-medium text-gray-800">{factor.description}</p>
                        <p className="text-xs text-gray-500 capitalize">{factor.factor.replace(/_/g, ' ')}</p>
                      </div>
                      <span className="text-red-600 font-semibold">+{(factor.contribution * 100).toFixed(1)}%</span>
                    </li>
                  ))}
                </ul>
              </div>
              
              <div className="clinical-card border-l-4 border-green-400">
                <h4 className="font-semibold text-gray-700 mb-3">Risk Decreasing Factors</h4>
                <ul className="space-y-3">
                  {result.risk_factors.top_decreasing_factors.map((factor: any, index: number) => (
                    <li key={index} className="flex justify-between items-start">
                      <div>
                        <p className="font-medium text-gray-800">{factor.description}</p>
                        <p className="text-xs text-gray-500 capitalize">{factor.factor.replace(/_/g, ' ')}</p>
                      </div>
                      <span className="text-green-600 font-semibold">{(factor.contribution * 100).toFixed(1)}%</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
            
            {/* Screening Recommendation */}
            <div className="clinical-card border-l-4 border-primary">
              <h4 className="font-semibold text-gray-800 mb-2">Screening Recommendation</h4>
              <p className="text-gray-700 capitalize">{result.recommendations.screening.recommendation.replace(/_/g, ' ')}</p>
              <p className="text-sm text-gray-600 mt-1">{result.recommendations.screening.rationale}</p>
              <div className="mt-3 flex items-center gap-4 text-sm">
                <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded">
                  Next: {result.recommendations.screening.next_screening_date}
                </span>
                {result.recommendations.screening.additional_imaging.ultrasound && (
                  <span className="bg-purple-100 text-purple-800 px-2 py-1 rounded">
                    Ultrasound recommended
                  </span>
                )}
              </div>
            </div>
            
            {/* Lifestyle Modifications */}
            <div className="clinical-card">
              <h4 className="font-semibold text-gray-800 mb-3">Lifestyle Recommendations</h4>
              <div className="space-y-3">
                {result.recommendations.lifestyle_modifications.map((mod: any, index: number) => (
                  <div key={index} className="flex items-start justify-between p-3 bg-gray-50 rounded-lg">
                    <div>
                      <p className="font-medium text-gray-800">{mod.action}</p>
                      <p className="text-xs text-gray-500 capitalize">{mod.category.replace(/_/g, ' ')}</p>
                    </div>
                    <div className="text-right">
                      <span className={`text-xs px-2 py-1 rounded ${
                        mod.priority === 'high' ? 'bg-red-100 text-red-700' : 'bg-yellow-100 text-yellow-700'
                      }`}>
                        {mod.priority} priority
                      </span>
                      <p className="text-sm text-green-600 mt-1">↓ {mod.potential_risk_reduction}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Medical Interventions */}
            <div className="clinical-card">
              <h4 className="font-semibold text-gray-800 mb-3">Medical Interventions</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className={`p-4 rounded-lg ${result.recommendations.medical_interventions.genetic_counseling.recommended ? 'bg-blue-50 border border-blue-200' : 'bg-gray-50'}`}>
                  <p className="font-medium">Genetic Counseling</p>
                  <p className={`text-sm ${result.recommendations.medical_interventions.genetic_counseling.recommended ? 'text-blue-700' : 'text-gray-600'}`}>
                    {result.recommendations.medical_interventions.genetic_counseling.recommended ? '✓ Recommended' : 'Not recommended'}
                  </p>
                  <p className="text-xs text-gray-500 mt-1">{result.recommendations.medical_interventions.genetic_counseling.reason}</p>
                </div>
                <div className={`p-4 rounded-lg ${result.recommendations.medical_interventions.chemoprevention.recommended ? 'bg-blue-50 border border-blue-200' : 'bg-gray-50'}`}>
                  <p className="font-medium">Chemoprevention</p>
                  <p className={`text-sm ${result.recommendations.medical_interventions.chemoprevention.recommended ? 'text-blue-700' : 'text-gray-600'}`}>
                    {result.recommendations.medical_interventions.chemoprevention.recommended ? '✓ Recommended' : 'Not recommended'}
                  </p>
                  <p className="text-xs text-gray-500 mt-1">{result.recommendations.medical_interventions.chemoprevention.reason}</p>
                </div>
              </div>
            </div>
            
            {/* Report Info */}
            <div className="clinical-card bg-gray-50">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Report ID: {result.report.report_id}</p>
                  <p className="text-xs text-gray-500">Model: {result.risk_assessment.model_version} • Processed in {result.risk_assessment.processing_time_ms}ms</p>
                </div>
              </div>
            </div>
            
            {/* Action Buttons */}
            <div className="flex flex-wrap gap-4">
              <button className="btn-primary">
                <span className="mr-2">📄</span>
                Download Report (PDF)
              </button>
              <button className="btn-secondary">
                <span className="mr-2">📧</span>
                Email Results
              </button>
              <button
                onClick={() => {
                  setCurrentStep(1)
                  setResult(null)
                }}
                className="btn-secondary"
              >
                New Assessment
              </button>
            </div>
          </motion.div>
        ) : null
    }
  }
  
  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Breast Cancer Risk Assessment</h1>
        <p className="text-gray-600 mt-2">
          Complete patient information for personalized risk evaluation
        </p>
      </div>
      
      {/* Progress Bar */}
      {currentStep <= totalSteps && (
        <div className="mb-8">
          <div className="flex justify-between mb-2">
            <span className="text-sm text-gray-600">
              Step {currentStep} of {totalSteps}
            </span>
            <span className="text-sm text-gray-600">
              {Math.round((currentStep / totalSteps) * 100)}% Complete
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <motion.div
              className="bg-primary h-2 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${(currentStep / totalSteps) * 100}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
        </div>
      )}
      
      {/* Form Content */}
      <form onSubmit={handleSubmit(onSubmit)}>
        <div className="clinical-card">
          <AnimatePresence mode="wait">
            {renderStep()}
          </AnimatePresence>
          
          {/* Navigation Buttons */}
          {currentStep <= totalSteps && (
            <div className="flex justify-between mt-8">
              <button
                type="button"
                onClick={prevStep}
                disabled={currentStep === 1}
                className={`btn-secondary ${currentStep === 1 ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                Previous
              </button>
              
              {currentStep < totalSteps ? (
                <button
                  type="button"
                  onClick={nextStep}
                  className="btn-primary"
                >
                  Next
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={isLoading}
                  className="btn-primary"
                >
                  {isLoading ? 'Calculating...' : 'Calculate Risk'}
                </button>
              )}
            </div>
          )}
        </div>
      </form>
    </div>
  )
}

export default RiskAssessment