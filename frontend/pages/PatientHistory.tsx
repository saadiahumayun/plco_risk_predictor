import React, { useState } from 'react'
import { format } from 'date-fns'

interface PatientRecord {
  id: string
  date: Date
  age: number
  risk_score: number
  risk_category: 'low' | 'moderate' | 'high'
  model_used: string
  follow_up: boolean
}

const PatientHistory: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('')
  const [filterCategory, setFilterCategory] = useState('all')
  
  // Mock data for demonstration
  const mockPatients: PatientRecord[] = [
    {
      id: 'BC-2024-0142',
      date: new Date('2024-01-14'),
      age: 52,
      risk_score: 0.123,
      risk_category: 'moderate',
      model_used: 'GA-Optimized',
      follow_up: true
    },
    {
      id: 'BC-2024-0141',
      date: new Date('2024-01-14'),
      age: 48,
      risk_score: 0.032,
      risk_category: 'low',
      model_used: 'GA-Optimized',
      follow_up: false
    },
    {
      id: 'BC-2024-0140',
      date: new Date('2024-01-13'),
      age: 61,
      risk_score: 0.245,
      risk_category: 'high',
      model_used: 'GA-Optimized',
      follow_up: true
    },
    {
      id: 'BC-2024-0139',
      date: new Date('2024-01-13'),
      age: 45,
      risk_score: 0.087,
      risk_category: 'moderate',
      model_used: 'Baseline',
      follow_up: true
    },
    {
      id: 'BC-2024-0138',
      date: new Date('2024-01-12'),
      age: 57,
      risk_score: 0.041,
      risk_category: 'low',
      model_used: 'GA-Optimized',
      follow_up: false
    }
  ]
  
  const filteredPatients = mockPatients.filter(patient => {
    const matchesSearch = patient.id.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesFilter = filterCategory === 'all' || patient.risk_category === filterCategory
    return matchesSearch && matchesFilter
  })
  
  const getRiskBadgeClasses = (category: string) => {
    switch (category) {
      case 'high':
        return 'bg-red-100 text-red-800 border-red-200'
      case 'moderate':
        return 'bg-orange-100 text-orange-800 border-orange-200'
      case 'low':
        return 'bg-green-100 text-green-800 border-green-200'
      default:
        return ''
    }
  }
  
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Patient History</h1>
        <p className="text-gray-600 mt-2">
          View and manage previous breast cancer risk assessments
        </p>
      </div>
      
      {/* Search and Filter */}
      <div className="flex flex-col md:flex-row gap-4">
        <div className="flex-1">
          <label className="clinical-label">Search Patient ID</label>
          <input
            type="text"
            placeholder="Search by patient ID..."
            className="clinical-input"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        
        <div className="w-full md:w-48">
          <label className="clinical-label">Risk Category</label>
          <select
            className="clinical-input"
            value={filterCategory}
            onChange={(e) => setFilterCategory(e.target.value)}
          >
            <option value="all">All Categories</option>
            <option value="low">Low Risk</option>
            <option value="moderate">Moderate Risk</option>
            <option value="high">High Risk</option>
          </select>
        </div>
        
        <div className="flex items-end">
          <button className="btn-primary">
            <span className="mr-2">📥</span>
            Export Data
          </button>
        </div>
      </div>
      
      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="metric-card">
          <p className="metric-label">Total Assessments</p>
          <p className="metric-value">{mockPatients.length}</p>
        </div>
        <div className="metric-card">
          <p className="metric-label">High Risk</p>
          <p className="metric-value text-red-600">
            {mockPatients.filter(p => p.risk_category === 'high').length}
          </p>
        </div>
        <div className="metric-card">
          <p className="metric-label">Require Follow-up</p>
          <p className="metric-value text-orange-600">
            {mockPatients.filter(p => p.follow_up).length}
          </p>
        </div>
        <div className="metric-card">
          <p className="metric-label">Average Risk</p>
          <p className="metric-value">
            {(mockPatients.reduce((sum, p) => sum + p.risk_score, 0) / mockPatients.length * 100).toFixed(1)}%
          </p>
        </div>
      </div>
      
      {/* Patient Table */}
      <div className="clinical-card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="clinical-table">
            <thead>
              <tr>
                <th>Patient ID</th>
                <th>Date</th>
                <th>Age</th>
                <th>Risk Score</th>
                <th>Category</th>
                <th>Model</th>
                <th>Follow-up</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredPatients.map((patient) => (
                <tr key={patient.id} className="hover:bg-gray-50">
                  <td className="font-medium">{patient.id}</td>
                  <td>{format(patient.date, 'MMM dd, yyyy')}</td>
                  <td>{patient.age}</td>
                  <td className="font-semibold">{(patient.risk_score * 100).toFixed(1)}%</td>
                  <td>
                    <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getRiskBadgeClasses(patient.risk_category)}`}>
                      {patient.risk_category.charAt(0).toUpperCase() + patient.risk_category.slice(1)}
                    </span>
                  </td>
                  <td>{patient.model_used}</td>
                  <td>
                    {patient.follow_up ? (
                      <span className="text-orange-600 font-medium">Required</span>
                    ) : (
                      <span className="text-gray-500">-</span>
                    )}
                  </td>
                  <td>
                    <div className="flex space-x-2">
                      <button className="text-primary hover:text-primary-dark">
                        View
                      </button>
                      <button className="text-gray-600 hover:text-gray-800">
                        Report
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      
      {/* Pagination */}
      <div className="flex justify-between items-center">
        <p className="text-sm text-gray-600">
          Showing {filteredPatients.length} of {mockPatients.length} patients
        </p>
        <div className="flex space-x-2">
          <button className="px-3 py-1 border border-gray-300 rounded-md text-sm hover:bg-gray-50">
            Previous
          </button>
          <button className="px-3 py-1 bg-primary text-white rounded-md text-sm">
            1
          </button>
          <button className="px-3 py-1 border border-gray-300 rounded-md text-sm hover:bg-gray-50">
            2
          </button>
          <button className="px-3 py-1 border border-gray-300 rounded-md text-sm hover:bg-gray-50">
            Next
          </button>
        </div>
      </div>
    </div>
  )
}

export default PatientHistory