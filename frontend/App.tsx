import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Home from './pages/Home'
import Dashboard from './pages/Dashboard'
import RiskAssessment from './pages/RiskAssessment'
import PatientHistory from './pages/PatientHistory'
import Analytics from './pages/Analytics'
import About from './pages/About'
import ModelValidation from './pages/ModelValidation'
import Impact from './pages/Impact'
import OfflineAssessment from './pages/OfflineAssessment'
import LocalHistory from './pages/LocalHistory'

const App: React.FC = () => {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/assessment" element={<RiskAssessment />} />
          <Route path="/history" element={<PatientHistory />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/validation" element={<ModelValidation />} />
          <Route path="/impact" element={<Impact />} />
          <Route path="/about" element={<About />} />
          {/* Offline-first PWA routes */}
          <Route path="/offline-assessment" element={<OfflineAssessment />} />
          <Route path="/local-history" element={<LocalHistory />} />
        </Routes>
      </Layout>
    </Router>
  )
}

export default App