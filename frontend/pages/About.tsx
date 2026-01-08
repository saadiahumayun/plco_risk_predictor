import React from 'react'
import { motion } from 'framer-motion'

const About: React.FC = () => {
  return (
    <div className="space-y-8 max-w-4xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h1 className="text-3xl font-bold text-gray-900">
          About the Breast Cancer Risk Predictor
        </h1>
        <p className="text-gray-600 mt-2 text-lg">
          A clinical decision support system using genetic algorithm-optimized machine learning
        </p>
      </motion.div>
      
      {/* Overview Section */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className="clinical-card"
      >
        <h2 className="text-2xl font-semibold text-gray-800 mb-4">Overview</h2>
        <p className="text-gray-700 leading-relaxed mb-4">
          This breast cancer risk prediction system represents a significant advancement in personalized 
          risk assessment. By leveraging genetic algorithm (GA) optimization for feature selection, 
          our model achieves superior predictive performance while using 32% fewer clinical variables 
          than traditional models.
        </p>
        <p className="text-gray-700 leading-relaxed">
          The system is designed specifically for clinical settings, providing healthcare professionals 
          with accurate, interpretable risk assessments that can inform screening decisions and 
          preventive interventions.
        </p>
      </motion.section>
      
      {/* Key Features */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="clinical-card"
      >
        <h2 className="text-2xl font-semibold text-gray-800 mb-4">Key Features</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-semibold text-gray-700 mb-2">🧬 GA-Optimized Selection</h3>
            <p className="text-gray-600 text-sm">
              Genetic algorithms identify the most predictive features from over 90 clinical variables, 
              reducing complexity while maintaining accuracy.
            </p>
          </div>
          <div>
            <h3 className="font-semibold text-gray-700 mb-2">📊 Superior Performance</h3>
            <p className="text-gray-600 text-sm">
              Achieves an AUC of 0.892, outperforming traditional models while using fewer inputs 
              for more efficient clinical assessment.
            </p>
          </div>
          <div>
            <h3 className="font-semibold text-gray-700 mb-2">🏥 Clinical Integration</h3>
            <p className="text-gray-600 text-sm">
              Designed for seamless integration into clinical workflows with intuitive interfaces 
              and actionable recommendations.
            </p>
          </div>
          <div>
            <h3 className="font-semibold text-gray-700 mb-2">🌍 Population-Specific</h3>
            <p className="text-gray-600 text-sm">
              Validated across diverse populations with plans for specific calibration for 
              South Asian populations.
            </p>
          </div>
        </div>
      </motion.section>
      
      {/* Methodology */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.3 }}
        className="clinical-card"
      >
        <h2 className="text-2xl font-semibold text-gray-800 mb-4">Methodology</h2>
        
        <div className="space-y-4">
          <div>
            <h3 className="font-semibold text-gray-700 mb-2">Feature Selection Process</h3>
            <p className="text-gray-600 mb-2">
              Our genetic algorithm approach optimizes feature selection through:
            </p>
            <ul className="list-disc list-inside text-gray-600 space-y-1 ml-4">
              <li>Population initialization with diverse feature subsets</li>
              <li>Fitness evaluation using cross-validated AUC scores</li>
              <li>Selection, crossover, and mutation operations</li>
              <li>Convergence to optimal feature subset</li>
            </ul>
          </div>
          
          <div>
            <h3 className="font-semibold text-gray-700 mb-2">Model Architecture</h3>
            <p className="text-gray-600">
              The prediction model uses a Random Forest classifier with:
            </p>
            <ul className="list-disc list-inside text-gray-600 space-y-1 ml-4">
              <li>100 decision trees with balanced class weights</li>
              <li>Maximum depth optimization through grid search</li>
              <li>Bootstrap aggregation for stability</li>
              <li>Calibration using isotonic regression</li>
            </ul>
          </div>
        </div>
      </motion.section>
      
      {/* Clinical Guidelines */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
        className="clinical-card"
      >
        <h2 className="text-2xl font-semibold text-gray-800 mb-4">Clinical Guidelines</h2>
        
        <div className="bg-blue-50 border-l-4 border-primary p-4 mb-4">
          <p className="text-sm text-gray-700">
            <strong>Important:</strong> This tool is designed to assist clinical decision-making, 
            not replace professional judgment. All risk assessments should be interpreted in the 
            context of individual patient history and current clinical guidelines.
          </p>
        </div>
        
        <div className="space-y-3">
          <div>
            <h4 className="font-medium text-gray-700">High Risk (≥20% 5-year risk)</h4>
            <p className="text-gray-600 text-sm">
              Annual mammography and MRI screening recommended. Consider genetic counseling and 
              risk-reducing interventions.
            </p>
          </div>
          <div>
            <h4 className="font-medium text-gray-700">Moderate Risk (10-20% 5-year risk)</h4>
            <p className="text-gray-600 text-sm">
              Annual mammography recommended. Discuss additional screening modalities based on 
              individual risk factors.
            </p>
          </div>
          <div>
            <h4 className="font-medium text-gray-700">Low Risk (&lt;10% 5-year risk)</h4>
            <p className="text-gray-600 text-sm">
              Follow standard screening guidelines based on age. Biennial mammography for women 
              50-74 years.
            </p>
          </div>
        </div>
      </motion.section>
      
      {/* Research & Development */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.5 }}
        className="clinical-card"
      >
        <h2 className="text-2xl font-semibold text-gray-800 mb-4">Research & Development</h2>
        <p className="text-gray-700 mb-4">
          This system is based on research conducted using the PLCO (Prostate, Lung, Colorectal, 
          and Ovarian) Cancer Screening Trial dataset, with ongoing validation studies planned 
          for diverse populations.
        </p>
        
        <div className="bg-gray-50 rounded-lg p-4">
          <h3 className="font-semibold text-gray-700 mb-2">Current Research Focus</h3>
          <ul className="space-y-2 text-sm text-gray-600">
            <li className="flex items-start">
              <span className="text-primary mr-2">•</span>
              <span>Population-specific validation for South Asian women at Aga Khan University</span>
            </li>
            <li className="flex items-start">
              <span className="text-primary mr-2">•</span>
              <span>Integration of genomic markers for enhanced prediction</span>
            </li>
            <li className="flex items-start">
              <span className="text-primary mr-2">•</span>
              <span>Development of multi-modal risk assessment incorporating imaging data</span>
            </li>
            <li className="flex items-start">
              <span className="text-primary mr-2">•</span>
              <span>Real-world clinical outcomes tracking and model refinement</span>
            </li>
          </ul>
        </div>
      </motion.section>
      
      {/* Contact & Support */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.6 }}
        className="clinical-card"
      >
        <h2 className="text-2xl font-semibold text-gray-800 mb-4">Contact & Support</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-semibold text-gray-700 mb-2">Technical Support</h3>
            <p className="text-gray-600 text-sm mb-2">
              For technical issues or questions about the system:
            </p>
            <p className="text-primary">support@breastcancerrisk.ai</p>
          </div>
          <div>
            <h3 className="font-semibold text-gray-700 mb-2">Research Collaboration</h3>
            <p className="text-gray-600 text-sm mb-2">
              For research partnerships and validation studies:
            </p>
            <p className="text-primary">research@breastcancerrisk.ai</p>
          </div>
        </div>
      </motion.section>
      
      {/* Version Info */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.7 }}
        className="text-center text-sm text-gray-500 pb-4"
      >
        <p>Version 1.0.0 | Last Updated: December 2025</p>
        <p>© 2026 Breast Cancer Risk Predictor. All rights reserved.</p>
      </motion.div>
    </div>
  )
}

export default About