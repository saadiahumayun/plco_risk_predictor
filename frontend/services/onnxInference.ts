import { PatientFormData, PredictionResult } from './offlineDb'

// ONNX Runtime types (loaded dynamically)
type OrtModule = typeof import('onnxruntime-web')
type InferenceSession = import('onnxruntime-web').InferenceSession

interface ScalerConfig {
  feature_names: string[]
  thresholds: {
    high: number
    moderate: number
  }
  feature_mapping: Record<string, {
    target: string
    type: string
    min?: number
    max?: number
    mapping?: Record<string, number>
    values?: number[]
  }>
  defaults: Record<string, number>
}

// Dynamic ONNX Runtime loader
let ortModule: OrtModule | null = null

async function loadOnnxRuntime(): Promise<OrtModule | null> {
  if (ortModule) return ortModule
  
  try {
    // Try to load from CDN
    const script = document.createElement('script')
    script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js'
    
    await new Promise<void>((resolve, reject) => {
      script.onload = () => resolve()
      script.onerror = () => reject(new Error('Failed to load ONNX Runtime'))
      document.head.appendChild(script)
    })
    
    // Access the global ort object
    ortModule = (window as unknown as { ort: OrtModule }).ort
    return ortModule
  } catch (error) {
    console.warn('ONNX Runtime not available:', error)
    return null
  }
}

class LocalPredictor {
  private session: InferenceSession | null = null
  private scalerConfig: ScalerConfig | null = null
  private isInitialized = false
  private initPromise: Promise<void> | null = null
  private ort: OrtModule | null = null

  async initialize(): Promise<void> {
    if (this.isInitialized) return
    if (this.initPromise) return this.initPromise

    this.initPromise = this._doInitialize()
    await this.initPromise
    this.isInitialized = true
  }

  private async _doInitialize(): Promise<void> {
    try {
      // Load scaler configuration first (always needed)
      const scalerResponse = await fetch('/models/scaler.json')
      if (!scalerResponse.ok) {
        throw new Error('Failed to load scaler configuration')
      }
      this.scalerConfig = await scalerResponse.json()

      // Try to load ONNX Runtime
      this.ort = await loadOnnxRuntime()
      
      if (this.ort) {
        // Configure ONNX Runtime for web
        this.ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/'
        
        // Try to load the ONNX model
        try {
          this.session = await this.ort.InferenceSession.create('/models/candetect.onnx', {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all',
          })
          console.log('ONNX model loaded successfully')
        } catch (modelError) {
          console.warn('ONNX model not available, using fallback prediction:', modelError)
          this.session = null
        }
      } else {
        console.warn('ONNX Runtime not available, using fallback prediction')
      }

      console.log('LocalPredictor initialized')
    } catch (error) {
      console.error('Failed to initialize LocalPredictor:', error)
      throw error
    }
  }

  async predict(formData: PatientFormData): Promise<PredictionResult> {
    await this.initialize()

    if (!this.scalerConfig) {
      throw new Error('Scaler configuration not loaded')
    }

    // Transform form data to feature vector
    const features = this.transformToFeatures(formData)

    let riskScore: number

    if (this.session) {
      // Run ONNX model inference
      riskScore = await this.runOnnxInference(features)
    } else {
      // Fallback: Use rule-based risk calculation
      riskScore = this.calculateFallbackRisk(formData)
    }

    // Categorize risk based on thresholds
    const riskCategory = this.categorizeRisk(riskScore)

    return {
      risk_score: riskScore,
      risk_category: riskCategory,
      confidence: this.session ? 0.89 : 0.7, // Lower confidence for fallback
      computed_locally: true,
    }
  }

  private transformToFeatures(formData: PatientFormData): Float32Array {
    const config = this.scalerConfig!
    const featureCount = config.feature_names.length
    const features = new Float32Array(featureCount)

    // Start with defaults
    for (let i = 0; i < featureCount; i++) {
      const featureName = config.feature_names[i]
      features[i] = config.defaults[featureName] ?? 0
    }

    // Map form data to features
    this.mapFormToFeatures(formData, features, config)

    return features
  }

  private mapFormToFeatures(
    formData: PatientFormData,
    features: Float32Array,
    config: ScalerConfig
  ): void {
    const featureNames = config.feature_names

    // Helper to set feature by name
    const setFeature = (name: string, value: number) => {
      const idx = featureNames.indexOf(name)
      if (idx !== -1) {
        features[idx] = value
      }
    }

    // Demographics
    setFeature('age', formData.age)
    setFeature('educat', formData.education_level)
    
    // Marital status mapping
    const maritalMap: Record<string, number> = {
      'married': 1, 'single': 2, 'divorced': 3, 'widowed': 4
    }
    setFeature('marital', maritalMap[formData.marital_status] ?? 1)

    // Reproductive history
    setFeature('fmenstr', this.categorizeAgeAtMenarche(formData.age_at_menarche))
    setFeature('prega', formData.number_of_live_births)
    setFeature('sisters', formData.number_of_relatives_with_bc)

    // Age at first birth (if applicable)
    if (formData.age_at_first_birth) {
      setFeature('bq_age', formData.age_at_first_birth)
    }

    // BMI
    setFeature('bmi_curr', formData.current_bmi)
    if (formData.bmi_at_20) {
      setFeature('bmi_20', formData.bmi_at_20)
    }

    // Medical history (binary flags)
    setFeature('fh_cancer', formData.family_history_cancer ? 1 : 0)
    setFeature('horm_f', formData.hormone_therapy ? 1 : 0)
    
    if (formData.hormone_therapy && formData.years_of_hormone_use) {
      setFeature('thorm', formData.years_of_hormone_use)
    }

    // Smoking
    if (formData.pack_years_smoking && formData.pack_years_smoking > 0) {
      setFeature('smoked_f', 1)
      setFeature('cig_years', Math.min(formData.pack_years_smoking, 60))
    }

    // Birth control
    if (formData.birth_control_years && formData.birth_control_years > 0) {
      setFeature('bcontr_f', 1)
    }

    // Study-related defaults (not from form, set to neutral values)
    setFeature('entryage_dhq', formData.age)
    setFeature('arm', 0)
  }

  private categorizeAgeAtMenarche(age: number): number {
    if (age < 11) return 1
    if (age < 12) return 2
    if (age < 13) return 3
    if (age < 14) return 4
    if (age < 15) return 5
    return 6 // 15+
  }

  private async runOnnxInference(features: Float32Array): Promise<number> {
    if (!this.session || !this.ort) {
      throw new Error('ONNX session not initialized')
    }

    try {
      // Create input tensor
      const inputTensor = new this.ort.Tensor('float32', features, [1, features.length])
      
      // Run inference
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const feeds = { input: inputTensor } as any
      const results = await this.session.run(feeds)

      // Get the output (probability of positive class)
      const output = results[Object.keys(results)[0]]
      const data = output.data as Float32Array

      // For binary classification, get probability of class 1
      // Output might be [batch, 2] for probabilities or [batch, 1] for single probability
      let probability: number
      if (data.length >= 2) {
        probability = data[1] // Class 1 probability
      } else {
        probability = data[0]
      }

      return Math.max(0, Math.min(1, probability))
    } catch (error) {
      console.error('ONNX inference failed:', error)
      throw error
    }
  }

  private calculateFallbackRisk(formData: PatientFormData): number {
    // Rule-based risk calculation when ONNX model is unavailable
    // Based on established breast cancer risk factors
    
    let riskScore = 0.01 // Base risk ~1%

    // Age factor (risk increases with age)
    if (formData.age >= 50) riskScore += 0.005
    if (formData.age >= 60) riskScore += 0.005
    if (formData.age >= 70) riskScore += 0.005

    // Family history (strong risk factor)
    if (formData.family_history_cancer) riskScore += 0.01
    if (formData.number_of_relatives_with_bc >= 1) riskScore += 0.01
    if (formData.number_of_relatives_with_bc >= 2) riskScore += 0.015

    // Personal history
    if (formData.personal_history_cancer) riskScore += 0.02
    if (formData.benign_breast_disease) riskScore += 0.01

    // Reproductive factors
    if (formData.age_at_menarche < 12) riskScore += 0.003
    if (formData.number_of_live_births === 0) riskScore += 0.005
    if (formData.age_at_first_birth && formData.age_at_first_birth > 30) riskScore += 0.005

    // Hormone therapy
    if (formData.hormone_therapy) {
      riskScore += 0.005
      if (formData.years_of_hormone_use && formData.years_of_hormone_use > 5) {
        riskScore += 0.005
      }
    }

    // BMI factor
    if (formData.current_bmi >= 30) riskScore += 0.003

    // Smoking
    if (formData.pack_years_smoking && formData.pack_years_smoking > 10) {
      riskScore += 0.002
    }

    // Cap at reasonable maximum
    return Math.min(riskScore, 0.15) // Max 15%
  }

  private categorizeRisk(riskScore: number): 'low' | 'moderate' | 'high' {
    const thresholds = this.scalerConfig?.thresholds || {
      high: 0.035,
      moderate: 0.018
    }

    if (riskScore >= thresholds.high) {
      return 'high'
    } else if (riskScore >= thresholds.moderate) {
      return 'moderate'
    }
    return 'low'
  }

  isReady(): boolean {
    return this.isInitialized
  }

  hasOnnxModel(): boolean {
    return this.session !== null
  }

  getModelInfo(): { type: string; features: number } {
    return {
      type: this.session ? 'ONNX Random Forest' : 'Rule-based Fallback',
      features: this.scalerConfig?.feature_names.length ?? 42
    }
  }
}

// Singleton instance
export const localPredictor = new LocalPredictor()

// Convenience function
export async function predictLocally(formData: PatientFormData): Promise<PredictionResult> {
  return localPredictor.predict(formData)
}
