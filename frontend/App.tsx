import React from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import RiskAssessment from './pages/RiskAssessment'
import PatientHistory from './pages/PatientHistory'
import Analytics from './pages/Analytics'
import About from './pages/About'
import ModelValidation from './pages/ModelValidation'

const App: React.FC = () => {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/assessment" element={<RiskAssessment />} />
          <Route path="/history" element={<PatientHistory />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/validation" element={<ModelValidation />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </Layout>
    </Router>
  )
}

export default App