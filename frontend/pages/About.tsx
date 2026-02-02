import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// SVG Icons
const DnaIcon = () => (
  <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
  </svg>
);

const TrendingUpIcon = () => (
  <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
  </svg>
);

const ActivityIcon = () => (
  <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
  </svg>
);

const UsersIcon = () => (
  <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
  </svg>
);

const CheckCircleIcon = ({ className = "w-6 h-6" }) => (
  <svg className={className} fill="currentColor" viewBox="0 0 20 20">
    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
  </svg>
);

const AlertTriangleIcon = ({ className = "w-6 h-6" }) => (
  <svg className={className} fill="currentColor" viewBox="0 0 20 20">
    <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
  </svg>
);

const ChevronRightIcon = ({ className = "w-8 h-8" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
  </svg>
);

const PlayIcon = ({ className = "w-4 h-4" }) => (
  <svg className={className} fill="currentColor" viewBox="0 0 20 20">
    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
  </svg>
);

const About = () => {
  const [activeFeature, setActiveFeature] = useState(0);
  const [showGAAnimation, setShowGAAnimation] = useState(false);
  const [gaGeneration, setGaGeneration] = useState(0);

  // GA Animation
  const runGAAnimation = () => {
    setShowGAAnimation(true);
    setGaGeneration(0);
    const interval = setInterval(() => {
      setGaGeneration(prev => {
        if (prev >= 50) {
          clearInterval(interval);
          setTimeout(() => setShowGAAnimation(false), 2000);
          return 50;
        }
        return prev + 1;
      });
    }, 50);
  };

  const features = [
    {
      icon: <DnaIcon />,
      title: "Genetic Algorithm-Optimized",
      description: "Uses evolutionary algorithms to identify the most predictive features from 90+ clinical variables",
      color: "blue",
      stats: { label: "Feature Reduction", value: "90 → 61" }
    },
    {
      icon: <TrendingUpIcon />,
      title: "Superior Performance",
      description: "Achieves 77.5% AUC with balanced sensitivity and specificity for clinical decision making",
      color: "green",
      stats: { label: "AUC Score", value: "0.775" }
    },
    {
      icon: <ActivityIcon />,
      title: "Clinical Integration",
      description: "Designed for real-world clinical workflows with actionable recommendations",
      color: "purple",
      stats: { label: "Processing Time", value: "<100ms" }
    },
    {
      icon: <UsersIcon />,
      title: "Population-Validated",
      description: "Tested on 4,451 women from diverse backgrounds in the PLCO trial",
      color: "orange",
      stats: { label: "Validation Size", value: "891 patients" }
    }
  ];

  const gaSteps = [
    { 
      step: 1, 
      title: "Initialize", 
      description: "Create random feature subsets",
      icon: "🎲"
    },
    { 
      step: 2, 
      title: "Evaluate", 
      description: "Test each subset's predictive power",
      icon: "📊"
    },
    { 
      step: 3, 
      title: "Select", 
      description: "Choose best-performing combinations",
      icon: "🎯"
    },
    { 
      step: 4, 
      title: "Crossover", 
      description: "Mix successful feature sets",
      icon: "🔀"
    },
    { 
      step: 5, 
      title: "Mutate", 
      description: "Introduce variations for exploration",
      icon: "🧬"
    },
    { 
      step: 6, 
      title: "Converge", 
      description: "Optimal features identified",
      icon: "✅"
    }
  ];

  const modelFlow = [
    { stage: "Input", items: ["Patient Data", "Medical History", "Demographics"], color: "bg-blue-100 border-blue-300" },
    { stage: "Process", items: ["Genetic Algorithm", "Feature Selection", "Random Forest"], color: "bg-purple-100 border-purple-300" },
    { stage: "Output", items: ["Risk Score", "Risk Category", "Recommendations"], color: "bg-green-100 border-green-300" }
  ];

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* Hero Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <h1 className="text-4xl font-bold text-gray-900 mb-3">
          About the Breast Cancer Risk Predictor
        </h1>
        <p className="text-gray-600 text-lg">
          AI-Powered Clinical Decision Support Using Genetic Algorithm Optimization
        </p>
      </motion.div>

      {/* Interactive Feature Cards */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <h2 className="text-2xl font-semibold text-gray-800 mb-6">Key Features</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {features.map((feature, idx) => (
            <motion.div
              key={idx}
              onHoverStart={() => setActiveFeature(idx)}
              className={`clinical-card cursor-pointer transition-all duration-300 ${
                activeFeature === idx ? 'ring-2 ring-primary shadow-lg scale-105' : ''
              }`}
            >
              <div className="flex items-start gap-4">
                <div className={`text-${feature.color}-600 bg-${feature.color}-50 p-3 rounded-lg`}>
                  {feature.icon}
                </div>
                <div className="flex-1">
                  <h3 className="font-semibold text-gray-800 mb-2">{feature.title}</h3>
                  <p className="text-gray-600 text-sm mb-3">{feature.description}</p>
                  <div className={`inline-block bg-${feature.color}-50 px-3 py-1 rounded-full`}>
                    <span className="text-xs font-semibold text-gray-700">
                      {feature.stats.label}: <span className={`text-${feature.color}-600`}>{feature.stats.value}</span>
                    </span>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.section>

      {/* Interactive GA Visualization */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="clinical-card bg-gradient-to-br from-blue-50 to-purple-50"
      >
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-semibold text-gray-800">How Genetic Algorithm Works</h2>
          <button
            onClick={runGAAnimation}
            disabled={showGAAnimation}
            className="btn-primary flex items-center gap-2"
          >
            <PlayIcon />
            {showGAAnimation ? 'Running...' : 'Run Animation'}
          </button>
        </div>

        {/* GA Process Steps */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
          {gaSteps.map((item, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ 
                opacity: showGAAnimation ? (gaGeneration >= (idx * 8) ? 1 : 0.3) : 1,
                scale: showGAAnimation ? (gaGeneration >= (idx * 8) ? 1 : 0.9) : 1
              }}
              className={`text-center p-4 rounded-lg border-2 transition-all ${
                showGAAnimation && gaGeneration >= (idx * 8)
                  ? 'bg-primary text-white border-primary'
                  : 'bg-white border-gray-200'
              }`}
            >
              <div className="text-3xl mb-2">{item.icon}</div>
              <div className="text-2xl font-bold mb-1">{item.step}</div>
              <div className="text-sm font-semibold mb-1">{item.title}</div>
              <div className="text-xs opacity-80">{item.description}</div>
            </motion.div>
          ))}
        </div>

        {/* GA Progress Visualization */}
        {showGAAnimation && (
          <div className="bg-white rounded-lg p-6 border-2 border-primary">
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-semibold text-gray-700">Generation: {gaGeneration}/50</span>
              <span className="text-sm font-semibold text-gray-700">
                Fitness: {(0.50 + (gaGeneration / 50) * 0.275).toFixed(3)}
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
              <motion.div
                className="bg-gradient-to-r from-blue-500 to-purple-600 h-4 rounded-full"
                initial={{ width: '0%' }}
                animate={{ width: `${(gaGeneration / 50) * 100}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>
            <div className="mt-3 grid grid-cols-3 gap-3 text-center text-sm">
              <div>
                <div className="text-gray-500">Features</div>
                <div className="font-bold text-gray-800">{90 - Math.floor((gaGeneration / 50) * 29)}</div>
              </div>
              <div>
                <div className="text-gray-500">Population</div>
                <div className="font-bold text-gray-800">100</div>
              </div>
              <div>
                <div className="text-gray-500">Best AUC</div>
                <div className="font-bold text-primary">{(0.50 + (gaGeneration / 50) * 0.275).toFixed(3)}</div>
              </div>
            </div>
          </div>
        )}
      </motion.section>

      {/* Model Flow Diagram */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="clinical-card"
      >
        <h2 className="text-2xl font-semibold text-gray-800 mb-6">Model Architecture</h2>
        
        <div className="flex items-center justify-between gap-4">
          {modelFlow.map((stage, idx) => (
            <React.Fragment key={idx}>
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.4 + idx * 0.1 }}
                className="flex-1"
              >
                <div className={`border-2 rounded-lg p-6 ${stage.color}`}>
                  <h3 className="font-bold text-gray-800 mb-4 text-center text-lg">{stage.stage}</h3>
                  <div className="space-y-2">
                    {stage.items.map((item, itemIdx) => (
                      <motion.div
                        key={itemIdx}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.5 + idx * 0.1 + itemIdx * 0.1 }}
                        className="bg-white rounded px-3 py-2 text-sm text-center font-medium text-gray-700"
                      >
                        {item}
                      </motion.div>
                    ))}
                  </div>
                </div>
              </motion.div>
              
              {idx < modelFlow.length - 1 && (
                <ChevronRightIcon className="text-gray-400 flex-shrink-0" />
              )}
            </React.Fragment>
          ))}
        </div>
      </motion.section>

      {/* Performance Metrics */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="clinical-card"
      >
        <h2 className="text-2xl font-semibold text-gray-800 mb-6">Performance Metrics</h2>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          {[
            { label: 'AUC-ROC', value: '73.05%', description: 'Discrimination ability', color: 'blue'},
            { label: 'Sensitivity', value: '37.03%', description: 'Catches 7/10 cases', color: 'green' },
            { label: 'Specificity', value: '68.1%', description: 'Correctly identifies negatives', color: 'purple' },
            { label: 'Precision', value: '46.44%', description: 'Positive predictive value', color: 'orange' }
          ].map((metric, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.5 + idx * 0.1 }}
              className={`text-center p-6 rounded-lg bg-${metric.color}-50 border-2 border-${metric.color}-200 hover:shadow-lg transition-shadow cursor-pointer`}
            >
              <div className="text-3xl mb-2">{metric.icon}</div>
              <div className={`text-4xl font-bold text-${metric.color}-600 mb-2`}>{metric.value}</div>
              <div className="font-semibold text-gray-800 mb-1">{metric.label}</div>
              <div className="text-xs text-gray-600">{metric.description}</div>
            </motion.div>
          ))}
        </div>
      </motion.section>

      {/* Clinical Guidelines with Visual Risk Scale */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="clinical-card"
      >
        <h2 className="text-2xl font-semibold text-gray-800 mb-6">Risk Categories & Guidelines</h2>
        
        <div className="relative">
          {/* Risk Scale Bar */}
          <div className="h-12 rounded-lg overflow-hidden mb-8 flex">
            <div className="flex-1 bg-green-400 flex items-center justify-center text-white font-semibold">
              Low Risk
            </div>
            <div className="flex-1 bg-yellow-400 flex items-center justify-center text-white font-semibold">
              Moderate
            </div>
            <div className="flex-1 bg-red-400 flex items-center justify-center text-white font-semibold">
              High Risk
            </div>
          </div>

          {/* Guidelines */}
          <div className="space-y-4">
            {[
              {
                level: 'Low Risk',
                threshold: '< 1.8%',
                color: 'green',
                icon: <CheckCircleIcon />,
                guideline: 'Standard screening guidelines. Biennial mammography for women 50-74 years.',
                actions: ['Routine screening', 'Healthy lifestyle', 'Regular checkups']
              },
              {
                level: 'Moderate Risk',
                threshold: '1.8% - 3.5%',
                color: 'yellow',
                icon: <AlertTriangleIcon />,
                guideline: 'Annual mammography recommended. Discuss additional screening modalities.',
                actions: ['Annual mammography', 'Risk factor management', 'Enhanced monitoring']
              },
              {
                level: 'High Risk',
                threshold: '> 3.5%',
                color: 'red',
                icon: <AlertTriangleIcon />,
                guideline: 'Intensive screening with MRI. Consider genetic counseling and chemoprevention.',
                actions: ['MRI + Mammography', 'Genetic counseling', 'Chemoprevention options']
              }
            ].map((risk, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.6 + idx * 0.1 }}
                className={`border-l-4 border-${risk.color}-500 bg-${risk.color}-50 p-4 rounded-r-lg`}
              >
                <div className="flex items-start gap-3">
                  <div className={`text-${risk.color}-600 flex-shrink-0 mt-1`}>
                    {risk.icon}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <h3 className={`font-bold text-${risk.color}-800`}>{risk.level}</h3>
                      <span className={`text-sm px-2 py-1 rounded bg-${risk.color}-200 text-${risk.color}-800 font-mono`}>
                        {risk.threshold}
                      </span>
                    </div>
                    <p className="text-gray-700 text-sm mb-3">{risk.guideline}</p>
                    <div className="flex flex-wrap gap-2">
                      {risk.actions.map((action, actionIdx) => (
                        <span
                          key={actionIdx}
                          className={`text-xs px-2 py-1 rounded-full bg-white border border-${risk.color}-300 text-${risk.color}-700`}
                        >
                          {action}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.section>

      {/* Research & Development */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="clinical-card bg-gradient-to-br from-purple-50 to-blue-50"
      >
        <h2 className="text-2xl font-semibold text-gray-800 mb-4">Research & Development</h2>
        <p className="text-gray-700 mb-6">
          Based on the PLCO Cancer Screening Trial with ongoing validation for diverse populations.
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[
            { title: 'South Asian Validation', status: 'In Progress', org: 'Aga Khan University' },
            { title: 'Genomic Integration', status: 'Planning', org: 'Multi-institutional' },
            { title: 'Multi-modal Assessment', status: 'Research', org: 'Imaging Centers' },
            { title: 'Clinical Outcomes Tracking', status: 'Active', org: 'Partner Hospitals' }
          ].map((project, idx) => (
            <div key={idx} className="bg-white rounded-lg p-4 border border-purple-200">
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-semibold text-gray-800">{project.title}</h3>
                <span className={`text-xs px-2 py-1 rounded-full ${
                  project.status === 'Active' || project.status === 'In Progress'
                    ? 'bg-green-100 text-green-700'
                    : 'bg-blue-100 text-blue-700'
                }`}>
                  {project.status}
                </span>
              </div>
              <p className="text-sm text-gray-600">{project.org}</p>
            </div>
          ))}
        </div>
      </motion.section>

      {/* Footer */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.7 }}
        className="text-center text-sm text-gray-500 pb-4 border-t pt-6"
      >
        <p className="mb-2">Version 1.0.0 | Last Updated: January 2026</p>
        <p>© 2026 Breast Cancer Risk Predictor. Built with ❤️ for better healthcare.</p>
      </motion.div>
    </div>
  );
};

export default About;