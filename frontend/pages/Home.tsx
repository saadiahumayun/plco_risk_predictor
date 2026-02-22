import React from 'react'
import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'

const Home: React.FC = () => {
  const features = [
    {
      icon: (
        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
        </svg>
      ),
      title: 'AI-Powered',
      description: 'Genetic Algorithm optimized ML model for accurate predictions'
    },
    {
      icon: (
        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      ),
      title: '5-Year Forecast',
      description: 'Predicts individual risk over a clinically relevant timeframe'
    },
    {
      icon: (
        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
        </svg>
      ),
      title: 'Evidence-Based',
      description: 'Built on validated international breast cancer research'
    },
    {
      icon: (
        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
        </svg>
      ),
      title: 'Clinical Guidance',
      description: 'Personalized screening recommendations for each patient'
    }
  ]

  const steps = [
    { number: '01', title: 'Enter Patient Data', description: 'Input demographics, medical history, and lifestyle factors' },
    { number: '02', title: 'AI Analysis', description: 'Our model analyzes validated risk factors' },
    { number: '03', title: 'Risk Assessment', description: 'Receive 5-year risk score with confidence intervals' },
    { number: '04', title: 'Recommendations', description: 'Get personalized screening and lifestyle guidance' }
  ]

  return (
    <div className="-mt-8 -mx-4 sm:-mx-6 lg:-mx-8">
      {/* Hero Section */}
      <section className="relative min-h-[90vh] flex items-center overflow-hidden bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
        {/* Animated background elements */}
        <div className="absolute inset-0 overflow-hidden">
          {/* Gradient orbs */}
          <motion.div
            animate={{ 
              x: [0, 30, 0],
              y: [0, -20, 0],
              scale: [1, 1.1, 1]
            }}
            transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
            className="absolute -top-40 -right-40 w-96 h-96 bg-gradient-to-br from-rose-500/30 to-pink-600/20 rounded-full blur-3xl"
          />
          <motion.div
            animate={{ 
              x: [0, -20, 0],
              y: [0, 30, 0],
              scale: [1, 1.15, 1]
            }}
            transition={{ duration: 10, repeat: Infinity, ease: "easeInOut" }}
            className="absolute top-1/2 -left-40 w-80 h-80 bg-gradient-to-tr from-primary/40 to-rose-400/20 rounded-full blur-3xl"
          />
          <motion.div
            animate={{ 
              x: [0, 15, 0],
              y: [0, 15, 0],
            }}
            transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
            className="absolute bottom-20 right-1/4 w-64 h-64 bg-gradient-to-tl from-amber-500/20 to-orange-400/10 rounded-full blur-3xl"
          />
          
          {/* Subtle grid pattern */}
          <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,.02)_1px,transparent_1px)] bg-[size:60px_60px]" />
        </div>

        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 w-full">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            {/* Left content */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
            >
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.2, duration: 0.5 }}
                className="inline-flex items-center gap-2 bg-white/10 backdrop-blur-md border border-white/10 px-4 py-2 rounded-full text-sm text-rose-200 mb-8"
              >
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                </span>
                AI-Powered Clinical Decision Support
              </motion.div>
              
              <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold text-white leading-[1.1] mb-6">
                <span className="block">Predict.</span>
                <span className="block">Prevent.</span>
                <span className="block bg-gradient-to-r from-rose-400 via-pink-400 to-rose-300 bg-clip-text text-transparent">
                  Protect.
                </span>
              </h1>
              
              <p className="text-xl text-slate-300 mb-10 leading-relaxed max-w-xl">
                CANDetect leverages advanced machine learning to assess individual breast cancer risk, 
                enabling <span className="text-white font-medium">early detection</span> and <span className="text-white font-medium">personalized care</span>.
              </p>
              
              <div className="flex flex-wrap gap-4">
                <Link
                  to="/assessment"
                  className="group relative inline-flex items-center gap-3 bg-gradient-to-r from-rose-500 to-pink-600 hover:from-rose-600 hover:to-pink-700 text-white px-8 py-4 rounded-xl font-semibold transition-all duration-300 shadow-lg shadow-rose-500/25 hover:shadow-xl hover:shadow-rose-500/30 hover:-translate-y-0.5"
                >
                  Start Assessment
                  <svg className="w-5 h-5 transition-transform group-hover:translate-x-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                </Link>
                <Link
                  to="/about"
                  className="inline-flex items-center gap-2 bg-white/5 hover:bg-white/10 backdrop-blur-sm text-white px-8 py-4 rounded-xl font-semibold transition-all duration-300 border border-white/10 hover:border-white/20"
                >
                  Learn More
                </Link>
              </div>

              {/* Trust indicators */}
              <div className="mt-12 pt-8 border-t border-white/10">
                <p className="text-sm text-slate-400 mb-4">Developed at</p>
                <div className="flex items-center gap-6">
                  <div className="text-white font-semibold">Institute of Business Administration, Karachi</div>
                </div>
              </div>
            </motion.div>

            {/* Right content - Stats cards */}
            <motion.div
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8, delay: 0.3 }}
              className="hidden lg:block"
            >
              <div className="relative">
                {/* Main stats grid */}
                <div className="grid grid-cols-2 gap-5">
                  {[
                    { value: '<90', label: 'Risk Factors', sublabel: 'Analyzed' },
                    { value: '5-Year', label: 'Risk Window', sublabel: 'Prediction' },
                    { value: '<75%', label: 'Model AUC', sublabel: 'Accuracy' },
                    { value: '<50ms', label: 'Response', sublabel: 'Time' }
                  ].map((stat, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.5, delay: 0.5 + index * 0.1 }}
                      whileHover={{ scale: 1.02, y: -2 }}
                      className="relative group"
                    >
                      <div className="absolute inset-0 bg-gradient-to-br from-rose-500/20 to-pink-600/10 rounded-2xl blur-xl group-hover:blur-2xl transition-all" />
                      <div className="relative bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl p-6 hover:border-rose-500/30 transition-colors">
                        <div className="text-4xl font-bold text-white mb-1">{stat.value}</div>
                        <div className="text-rose-300 font-medium">{stat.label}</div>
                        <div className="text-slate-400 text-sm">{stat.sublabel}</div>
                      </div>
                    </motion.div>
                  ))}
                </div>

              </div>
              
              {/* Validated badge - below stats */}
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1, duration: 0.5 }}
                className="mt-6 flex justify-center"
              >
                <div className="inline-flex items-center gap-2 bg-gradient-to-br from-emerald-500 to-emerald-600 text-white px-5 py-3 rounded-xl shadow-lg shadow-emerald-500/30">
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                  <span className="font-semibold">Clinically Validated</span>
                </div>
              </motion.div>
            </motion.div>
          </div>
        </div>

        {/* Bottom wave */}
        <div className="absolute bottom-0 left-0 right-0">
          <svg viewBox="0 0 1440 100" fill="none" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none" className="w-full h-20">
            <path d="M0 50L48 45C96 40 192 30 288 25C384 20 480 20 576 28C672 36 768 52 864 58C960 64 1056 60 1152 52C1248 44 1344 32 1392 26L1440 20V100H1392C1344 100 1248 100 1152 100C1056 100 960 100 864 100C768 100 672 100 576 100C480 100 384 100 288 100C192 100 96 100 48 100H0V50Z" fill="white"/>
          </svg>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <span className="inline-block text-primary font-semibold text-sm uppercase tracking-wider mb-3">
              Why CANDetect
            </span>
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
              Intelligent Risk Assessment
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Combining clinical expertise with cutting-edge AI for accurate, 
              personalized breast cancer risk evaluation.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                whileHover={{ y: -5 }}
                className="group relative"
              >
                <div className="absolute inset-0 bg-gradient-to-br from-rose-500/5 to-pink-500/5 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity" />
                <div className="relative bg-white border border-gray-100 rounded-2xl p-8 shadow-sm hover:shadow-lg hover:border-rose-100 transition-all">
                  <div className="w-14 h-14 bg-gradient-to-br from-rose-50 to-pink-50 rounded-xl flex items-center justify-center text-primary mb-5 group-hover:scale-110 transition-transform">
                    {feature.icon}
                  </div>
                  <h3 className="text-xl font-bold text-gray-900 mb-2">{feature.title}</h3>
                  <p className="text-gray-600 leading-relaxed">{feature.description}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Demo Section */}
      <section className="py-24 bg-gradient-to-b from-white to-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <span className="inline-block text-primary font-semibold text-sm uppercase tracking-wider mb-3">
              See It In Action
            </span>
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
              Product Demo
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Watch how CANDetect streamlines breast cancer risk assessment in clinical practice.
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 30, scale: 0.95 }}
            whileInView={{ opacity: 1, y: 0, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="relative"
          >
            {/* Browser frame */}
            <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl p-2 shadow-2xl shadow-slate-900/30">
              {/* Browser header */}
              <div className="flex items-center gap-2 px-4 py-3 bg-slate-700/50 rounded-t-xl">
                <div className="flex gap-1.5">
                  <div className="w-3 h-3 rounded-full bg-red-500"></div>
                  <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                  <div className="w-3 h-3 rounded-full bg-green-500"></div>
                </div>
                <div className="flex-1 mx-4">
                  <div className="bg-slate-600/50 rounded-lg px-4 py-1.5 text-slate-400 text-sm text-center">
                    candetect.app
                  </div>
                </div>
              </div>
              {/* GIF container */}
              <div className="rounded-b-xl overflow-hidden">
                <img 
                  src="/caandetect.gif" 
                  alt="CANDetect Application Demo" 
                  className="w-full h-auto"
                />
              </div>
            </div>

            {/* Decorative elements */}
            <div className="absolute -z-10 -top-8 -left-8 w-32 h-32 bg-rose-500/10 rounded-full blur-2xl"></div>
            <div className="absolute -z-10 -bottom-8 -right-8 w-40 h-40 bg-pink-500/10 rounded-full blur-2xl"></div>
          </motion.div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-24 bg-gradient-to-b from-gray-50 to-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <span className="inline-block text-primary font-semibold text-sm uppercase tracking-wider mb-3">
              Process
            </span>
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
              How It Works
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Get comprehensive risk assessment in four simple steps
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {steps.map((step, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.15 }}
                className="relative"
              >
                {/* Connector line */}
                {index < steps.length - 1 && (
                  <div className="hidden lg:block absolute top-12 left-[calc(50%+52px)] w-[calc(100%-90px)] h-[2px] bg-gradient-to-r from-rose-300 via-rose-200 to-rose-300" />
                )}
                
                <div className="text-center">
                  <div className="inline-flex items-center justify-center w-24 h-24 rounded-full bg-gradient-to-br from-rose-500 to-pink-600 text-white text-3xl font-bold mb-6 shadow-lg shadow-rose-500/25">
                    {step.number}
                  </div>
                  <h3 className="text-xl font-bold text-gray-900 mb-2">{step.title}</h3>
                  <p className="text-gray-600">{step.description}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Risk Categories Section */}
      <section className="py-24 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
            >
              <span className="inline-block text-primary font-semibold text-sm uppercase tracking-wider mb-3">
                Risk Stratification
              </span>
              <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
                Clinical Risk Categories
              </h2>
              <p className="text-xl text-gray-600 mb-8 leading-relaxed">
                Our model classifies patients into three clinically actionable risk categories, 
                aligned with established guidelines for breast cancer screening and prevention.
              </p>
              
              <div className="space-y-4">
                {[
                  { color: 'emerald', label: 'Low Risk', range: '< 1.8%', description: 'Standard age-appropriate screening' },
                  { color: 'amber', label: 'Moderate Risk', range: '1.8% - 3.5%', description: 'Enhanced surveillance recommended' },
                  { color: 'red', label: 'High Risk', range: '≥ 3.5%', description: 'Specialist referral, chemoprevention discussion' }
                ].map((category, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: index * 0.1 }}
                    className={`flex items-center gap-4 p-4 rounded-xl border-2 ${
                      category.color === 'emerald' ? 'border-emerald-200 bg-emerald-50/50' :
                      category.color === 'amber' ? 'border-amber-200 bg-amber-50/50' :
                      'border-red-200 bg-red-50/50'
                    }`}
                  >
                    <div className={`w-4 h-4 rounded-full ${
                      category.color === 'emerald' ? 'bg-emerald-500' :
                      category.color === 'amber' ? 'bg-amber-500' : 'bg-red-500'
                    }`} />
                    <div className="flex-1">
                      <div className="flex items-center gap-3">
                        <span className="font-bold text-gray-900">{category.label}</span>
                        <span className={`text-sm font-medium px-2 py-0.5 rounded ${
                          category.color === 'emerald' ? 'bg-emerald-100 text-emerald-700' :
                          category.color === 'amber' ? 'bg-amber-100 text-amber-700' :
                          'bg-red-100 text-red-700'
                        }`}>{category.range}</span>
                      </div>
                      <p className="text-gray-600 text-sm mt-1">{category.description}</p>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="relative"
            >
              <div className="bg-gradient-to-br from-slate-900 to-slate-800 rounded-3xl p-8 text-white">
                <h3 className="text-2xl font-bold mb-6 bg-gradient-to-r from-rose-300 via-pink-300 to-rose-200 bg-clip-text text-transparent">Validated Risk Factors</h3>
                
                <div className="space-y-6">
                  {[
                    { title: 'Demographics', items: ['Age', 'Race/Ethnicity', 'Education'] },
                    { title: 'Reproductive History', items: ['Age at menarche', 'Parity', 'Breastfeeding'] },
                    { title: 'Family History', items: ['First-degree relatives', 'Genetic predisposition'] },
                    { title: 'Medical History', items: ['Benign breast disease', 'Previous biopsies'] },
                    { title: 'Lifestyle Factors', items: ['BMI trajectory', 'Hormone therapy', 'Smoking'] }
                  ].map((group, index) => (
                    <div key={index}>
                      <h4 className="text-rose-400 font-semibold text-sm uppercase tracking-wider mb-2">
                        {group.title}
                      </h4>
                      <div className="flex flex-wrap gap-2">
                        {group.items.map((item, i) => (
                          <span key={i} className="bg-white/10 px-3 py-1 rounded-full text-sm">
                            {item}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 bg-gradient-to-br from-primary via-primary-dark to-[#4a1c2e] text-white relative overflow-hidden">
        <div className="absolute inset-0">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 50, repeat: Infinity, ease: "linear" }}
            className="absolute -top-1/2 -right-1/4 w-[800px] h-[800px] bg-gradient-to-br from-white/5 to-transparent rounded-full"
          />
        </div>
        
        <div className="relative max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl md:text-5xl font-bold mb-6 bg-gradient-to-r from-rose-300 via-pink-300 to-rose-200 bg-clip-text text-transparent">
              Ready to Get Started?
            </h2>
            <p className="text-xl text-rose-100 mb-10 max-w-2xl mx-auto">
              Perform a comprehensive breast cancer risk assessment in under 5 minutes. 
              Get personalized risk scores and clinical recommendations.
            </p>
            <div className="flex flex-wrap justify-center gap-4">
              <Link
                to="/assessment"
                className="group inline-flex items-center gap-3 bg-white text-primary hover:bg-rose-50 px-8 py-4 rounded-xl font-bold transition-all shadow-xl hover:shadow-2xl hover:-translate-y-1"
              >
                Begin Assessment
                <svg className="w-5 h-5 transition-transform group-hover:translate-x-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
              </Link>
              <Link
                to="/impact"
                className="inline-flex items-center gap-2 bg-white/10 hover:bg-white/20 backdrop-blur-sm text-white px-8 py-4 rounded-xl font-semibold transition-all border border-white/20"
              >
                View Impact Analysis
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Disclaimer */}
      <section className="py-6 bg-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <p className="text-center text-sm text-gray-500">
            <strong>Disclaimer:</strong> CANDetect is a clinical decision support tool and should not replace 
            professional medical advice. Always consult with qualified healthcare providers.
          </p>
        </div>
      </section>
    </div>
  )
}

export default Home
