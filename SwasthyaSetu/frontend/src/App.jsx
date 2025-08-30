import React from 'react';
import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { Stethoscope, Moon, Sun, User, LogOut, History, Download } from 'lucide-react'
import { Button } from './components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card'
import { Textarea } from './components/ui/textarea'
import { Badge } from './components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs'
import { Input } from './components/ui/input'
import { Label } from './components/ui/label'
import { Alert, AlertDescription } from './components/ui/alert'
import { Progress } from './components/ui/progress'
import { Separator } from './components/ui/separator'
import './App.css'

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Authentication Context
const AuthContext = React.createContext()

// Login Component
function LoginPage({ onLogin }) {
  const [isLogin, setIsLogin] = useState(true)
  const [formData, setFormData] = useState({
    username: '',
    password: '',
    email: ''
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')

    try {
      const endpoint = isLogin ? '/auth/login' : '/auth/register'
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      })

      const data = await response.json()

      if (response.ok) {
        if (isLogin) {
          localStorage.setItem('token', data.access_token)
          onLogin(data.access_token)
        } else {
          setIsLogin(true)
          setError('Registration successful! Please login.')
          // Clear form data after successful registration
          setFormData({ username: '', password: '', email: '' })
        }
      } else {
        setError(data.detail || 'An error occurred')
      }
    } catch (error) {
      setError('Network error. Please check your connection and try again.')
      console.error('Login/Register error:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center p-4">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          <div className="flex justify-center mb-4">
            <Stethoscope className="h-12 w-12 text-blue-600" />
          </div>
          <CardTitle className="text-2xl">Medical Summarizer</CardTitle>
          <CardDescription>
            {isLogin ? 'Sign in to your account' : 'Create a new account'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="username">Username</Label>
              <Input
                id="username"
                type="text"
                value={formData.username}
                onChange={(e) => setFormData({ ...formData, username: e.target.value })}
                required
                disabled={loading}
              />
            </div>
            
            {!isLogin && (
              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  value={formData.email}
                  onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                  required
                  disabled={loading}
                />
              </div>
            )}
            
            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                required
                disabled={loading}
              />
            </div>

            {error && (
              <Alert variant={error.includes('successful') ? 'default' : 'destructive'}>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? 'Please wait...' : (isLogin ? 'Sign In' : 'Sign Up')}
            </Button>

            <Button
              type="button"
              variant="ghost"
              className="w-full"
              onClick={() => {
                setIsLogin(!isLogin)
                setError('')
                setFormData({ username: '', password: '', email: '' })
              }}
              disabled={loading}
            >
              {isLogin ? "Don't have an account? Sign up" : "Already have an account? Sign in"}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  )
}

// Main Summarizer Component
function SummarizerPage({ token, onLogout }) {
  const [inputText, setInputText] = useState('')
  const [mode, setMode] = useState('both')
  const [summary, setSummary] = useState(null)
  const [loading, setLoading] = useState(false)
  const [darkMode, setDarkMode] = useState(() => {
    // Initialize dark mode from localStorage or system preference
    const saved = localStorage.getItem('darkMode')
    if (saved !== null) {
      return JSON.parse(saved)
    }
    return window.matchMedia('(prefers-color-scheme: dark)').matches
  })
  const [history, setHistory] = useState([])
  const [showHistory, setShowHistory] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    // Apply dark mode and save preference
    if (darkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
    localStorage.setItem('darkMode', JSON.stringify(darkMode))
  }, [darkMode])

  const handleSummarize = async () => {
    if (!inputText.trim()) return

    setLoading(true)
    setError('')
    try {
      const response = await fetch(`${API_BASE_URL}/summarize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({
          text: inputText,
          mode: mode,
        }),
      })

      const data = await response.json()
      if (response.ok) {
        setSummary(data)
        // Reload history to include the new summary
        loadHistory()
      } else {
        if (response.status === 401) {
          // Token expired, logout user
          onLogout()
          return
        }
        setError(data.detail || 'Failed to generate summary')
      }
    } catch (error) {
      console.error('Summarization error:', error)
      setError('Network error. Please check your connection and try again.')
    } finally {
      setLoading(false)
    }
  }

  const loadHistory = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/history`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      })
      
      if (response.ok) {
        const data = await response.json()
        setHistory(data.summaries || [])
      } else if (response.status === 401) {
        onLogout()
      }
    } catch (error) {
      console.error('Failed to load history:', error)
    }
  }

  useEffect(() => {
    loadHistory()
  }, [token]) // Add token as dependency

  const exportToPDF = () => {
    if (!summary) return

    const printContent = `Medical Summary Export
Generated on: ${new Date().toLocaleDateString()}

Original Text:
${inputText}

${summary?.patient_summary ? `Patient-Friendly Summary:
${summary.patient_summary}

` : ''}${summary?.clinician_summary ? `Clinical Summary:
${summary.clinician_summary}

` : ''}Faithfulness Score: ${summary?.faithfulness_score ? (summary.faithfulness_score * 100).toFixed(1) + '%' : 'N/A'}

---
Medical Disclaimer: This AI-generated content is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment.
`
    
    const blob = new Blob([printContent], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `medical-summary-${new Date().getTime()}.txt`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const handleHistoryItemClick = (historyItem) => {
    setInputText(historyItem.original_text)
    setSummary({
      patient_summary: historyItem.patient_summary,
      clinician_summary: historyItem.clinician_summary,
      faithfulness_score: historyItem.faithfulness_score,
      perspective_confidence: historyItem.perspective_confidence
    })
    setMode(historyItem.mode || 'both')
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      {/* Enhanced Header */}
      <header className="bg-gradient-to-r from-white to-blue-50 dark:from-gray-800 dark:to-blue-900/20 shadow-lg border-b border-blue-200 dark:border-blue-800 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-20">
            <div className="flex items-center space-x-4">
              <div className="p-3 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl shadow-lg">
                <Stethoscope className="h-8 w-8 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                  Medical Summarizer
                </h1>
                <p className="text-sm text-gray-600 dark:text-gray-300">
                  AI-Powered Dual Perspective Analysis
                </p>
              </div>
              <div className="flex gap-2">
                <Badge variant="secondary" className="bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-200">
                  Patient-Friendly
                </Badge>
                <Badge variant="secondary" className="bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-200">
                  Clinical Grade
                </Badge>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <Button
                variant="ghost"
                size="lg"
                onClick={() => setShowHistory(!showHistory)}
                className="hover:bg-blue-50 dark:hover:bg-blue-900/20 text-blue-700 dark:text-blue-300"
              >
                <History className="h-5 w-5 mr-2" />
                History
              </Button>
              
              <Button
                variant="ghost"
                size="lg"
                onClick={() => setDarkMode(!darkMode)}
                title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
                className="hover:bg-gray-100 dark:hover:bg-gray-700"
              >
                {darkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
              </Button>
              
              <Button
                variant="outline"
                size="lg"
                onClick={onLogout}
                className="border-red-300 text-red-700 hover:bg-red-50 dark:border-red-700 dark:text-red-300 dark:hover:bg-red-900/20"
              >
                <LogOut className="h-5 w-5 mr-2" />
                Logout
              </Button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Input Section */}
          <div className="lg:col-span-2 space-y-6">
            <Card className="border-2 border-gray-200 dark:border-gray-700 shadow-lg">
              <CardHeader className="bg-gradient-to-r from-gray-50 to-blue-50 dark:from-gray-800 dark:to-blue-900/20 rounded-t-lg">
                <CardTitle className="text-2xl text-gray-900 dark:text-gray-100 flex items-center gap-3">
                  <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-full">
                    <Stethoscope className="h-6 w-6 text-blue-700 dark:text-blue-300" />
                  </div>
                  Medical Text Input & Analysis
                </CardTitle>
                <CardDescription className="text-gray-600 dark:text-gray-300 text-lg">
                  Enter medical Q&A content, patient queries, clinical notes, or JSON/JSONL data for intelligent summarization
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6 p-6">
                {/* Enhanced Text Input */}
                <div className="space-y-3">
                  <Label className="text-lg font-semibold text-gray-700 dark:text-gray-300">
                    Medical Content Input
                  </Label>
                  <div className="relative">
                    <Textarea
                      placeholder="Example: What is diabetes and how should I manage my blood sugar levels? I'm a 45-year-old patient recently diagnosed with Type 2 diabetes...

Or paste JSON/JSONL medical data for enhanced processing..."
                      value={inputText}
                      onChange={(e) => setInputText(e.target.value)}
                      className="min-h-[250px] text-lg border-2 border-gray-200 dark:border-gray-700 focus:border-blue-500 dark:focus:border-blue-400 focus:ring-2 focus:ring-blue-200 dark:focus:ring-blue-800 transition-all duration-200"
                      disabled={loading}
                    />
                    <div className="absolute bottom-3 right-3 text-xs text-gray-400 dark:text-gray-500">
                      {inputText.length} characters
                    </div>
                  </div>
                </div>
                
                {/* Error Display */}
                {error && (
                  <Alert variant="destructive" className="border-red-300 bg-red-50 dark:bg-red-900/20">
                    <AlertDescription className="text-red-800 dark:text-red-200 font-medium">
                      {error}
                    </AlertDescription>
                  </Alert>
                )}
                
                {/* Enhanced Controls */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Summary Mode Selection */}
                  <div className="space-y-3">
                    <Label className="text-lg font-semibold text-gray-700 dark:text-gray-300">
                      Summary Perspective
                    </Label>
                    <Tabs value={mode} onValueChange={setMode} className="w-full">
                      <TabsList className="grid w-full grid-cols-3 h-12">
                        <TabsTrigger 
                          value="patient" 
                          disabled={loading}
                          className="data-[state=active]:bg-blue-100 data-[state=active]:text-blue-900 dark:data-[state=active]:bg-blue-900/30 dark:data-[state=active]:text-blue-100"
                        >
                          <User className="h-4 w-4 mr-2" />
                          Patient
                        </TabsTrigger>
                        <TabsTrigger 
                          value="clinician" 
                          disabled={loading}
                          className="data-[state=active]:bg-green-100 data-[state=active]:text-green-900 dark:data-[state=active]:bg-green-900/30 dark:data-[state=active]:text-green-100"
                        >
                          <Stethoscope className="h-4 w-4 mr-2" />
                          Clinician
                        </TabsTrigger>
                        <TabsTrigger 
                          value="both" 
                          disabled={loading}
                          className="data-[state=active]:bg-purple-100 data-[state=active]:text-purple-900 dark:data-[state=active]:bg-purple-900/30 dark:data-[state=active]:text-purple-100"
                        >
                          <div className="h-4 w-4 mr-2">🔄</div>
                          Both
                        </TabsTrigger>
                      </TabsList>
                    </Tabs>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {mode === 'patient' && 'Generate simplified, easy-to-understand summaries'}
                      {mode === 'clinician' && 'Create detailed, technical medical assessments'}
                      {mode === 'both' && 'Provide dual perspectives for comprehensive understanding'}
                    </p>
                  </div>
                  
                  {/* Generate Button */}
                  <div className="flex items-end">
                    <Button 
                      onClick={handleSummarize}
                      disabled={loading || !inputText.trim()}
                      className="w-full h-12 text-lg font-semibold bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white shadow-lg hover:shadow-xl transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {loading ? (
                        <div className="flex items-center gap-2">
                          <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                          Generating Summary...
                        </div>
                      ) : (
                        <div className="flex items-center gap-2">
                          <div className="w-5 h-5 bg-white rounded-full flex items-center justify-center">
                            <Stethoscope className="h-3 w-3 text-blue-600" />
                          </div>
                          Generate AI Summary
                        </div>
                      )}
                    </Button>
                  </div>
                </div>

                {/* Input Tips */}
                <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
                  <h4 className="font-semibold text-blue-900 dark:text-blue-100 mb-2 flex items-center gap-2">
                    <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                    Pro Tips for Better Results
                  </h4>
                  <ul className="text-sm text-blue-800 dark:text-blue-200 space-y-1">
                    <li>• Include specific medical conditions, symptoms, or questions</li>
                    <li>• For JSON data, paste the complete structure for enhanced processing</li>
                    <li>• Longer, more detailed inputs typically yield better summaries</li>
                    <li>• Use clear, descriptive language for optimal results</li>
                  </ul>
                </div>
              </CardContent>
            </Card>

            {/* Enhanced Summary Results */}
            {summary && (
              <Card className="border-2 border-blue-200 dark:border-blue-800 shadow-lg">
                <CardHeader className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-t-lg">
                  <div className="flex flex-row items-center justify-between">
                    <div>
                      <CardTitle className="text-2xl text-blue-900 dark:text-blue-100 flex items-center gap-2">
                        <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                        AI-Generated Medical Summaries
                      </CardTitle>
                      <CardDescription className="text-blue-700 dark:text-blue-300">
                        Dual-perspective analysis with enhanced accuracy metrics
                      </CardDescription>
                    </div>
                    <div className="flex gap-2">
                      <Button variant="outline" size="sm" onClick={exportToPDF} className="border-blue-300 text-blue-700 hover:bg-blue-50">
                        <Download className="h-4 w-4 mr-2" />
                        Export Summary
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="space-y-8 p-6">
                  {/* Patient Summary */}
                  {summary.patient_summary && (
                    <div className="space-y-4">
                      <div className="flex items-center space-x-3">
                        <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-full">
                          <User className="h-6 w-6 text-blue-700 dark:text-blue-300" />
                        </div>
                        <div>
                          <h3 className="font-bold text-xl text-blue-900 dark:text-blue-100">Patient-Friendly Summary</h3>
                          <div className="flex gap-2">
                            <Badge variant="secondary" className="bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-200">
                              Simplified Language
                            </Badge>
                            <Badge variant="outline" className="border-blue-300 text-blue-700">
                              Easy to Understand
                            </Badge>
                          </div>
                        </div>
                      </div>
                      <div className="bg-gradient-to-r from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-900/30 p-6 rounded-xl border border-blue-200 dark:border-blue-800">
                        <p className="text-gray-800 dark:text-gray-200 leading-relaxed text-lg font-medium">
                          {summary.patient_summary}
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Clinical Summary */}
                  {summary.clinician_summary && (
                    <div className="space-y-4">
                      <div className="flex items-center space-x-3">
                        <div className="p-2 bg-green-100 dark:bg-green-900/30 rounded-full">
                          <Stethoscope className="h-6 w-6 text-green-700 dark:text-green-300" />
                        </div>
                        <div>
                          <h3 className="font-bold text-xl text-green-900 dark:text-green-100">Clinical Assessment</h3>
                          <div className="flex gap-2">
                            <Badge variant="secondary" className="bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-200">
                              Medical Terminology
                            </Badge>
                            <Badge variant="outline" className="border-green-300 text-green-700">
                              Professional Grade
                            </Badge>
                          </div>
                        </div>
                      </div>
                      <div className="bg-gradient-to-r from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-900/30 p-6 rounded-xl border border-green-200 dark:border-green-800">
                        <p className="text-gray-800 dark:text-gray-200 leading-relaxed text-lg font-medium">
                          {summary.clinician_summary}
                        </p>
                      </div>
                    </div>
                  )}

                  <Separator className="my-8" />

                  {/* Enhanced Metrics Dashboard */}
                  <div className="bg-gray-50 dark:bg-gray-800/50 p-6 rounded-xl">
                    <h4 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4 flex items-center gap-2">
                      <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                      Quality Assessment Metrics
                    </h4>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {/* Faithfulness Score */}
                      <div className="space-y-3">
                        <div className="flex justify-between items-center">
                          <Label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                            Faithfulness Score
                          </Label>
                          <span className="text-lg font-bold text-blue-600 dark:text-blue-400">
                            {((summary.faithfulness_score || 0) * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="space-y-2">
                          <Progress 
                            value={(summary.faithfulness_score || 0) * 100} 
                            className="h-3 bg-gray-200 dark:bg-gray-700"
                          />
                          <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
                            <span>Low</span>
                            <span>Medium</span>
                            <span>High</span>
                          </div>
                        </div>
                        <p className="text-xs text-gray-600 dark:text-gray-400">
                          Measures how accurately the summary reflects the original content
                        </p>
                      </div>
                      
                      {/* Perspective Confidence */}
                      {summary.perspective_confidence && (
                        <div className="space-y-3">
                          <Label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                            Perspective Confidence
                          </Label>
                          <div className="space-y-3">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                <User className="h-4 w-4 text-blue-600" />
                                <span className="text-sm">Patient</span>
                              </div>
                              <div className="flex items-center gap-2">
                                <span className="text-sm font-medium">
                                  {(summary.perspective_confidence[1] * 100).toFixed(1)}%
                                </span>
                                <Progress 
                                  value={summary.perspective_confidence[1] * 100} 
                                  className="w-20 h-2 bg-blue-100 dark:bg-blue-900/30"
                                />
                              </div>
                            </div>
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                <Stethoscope className="h-4 w-4 text-green-600" />
                                <span className="text-sm">Clinical</span>
                              </div>
                              <div className="flex items-center gap-2">
                                <span className="text-sm font-medium">
                                  {(summary.perspective_confidence[2] * 100).toFixed(1)}%
                                </span>
                                <Progress 
                                  value={summary.perspective_confidence[2] * 100} 
                                  className="w-20 h-2 bg-green-100 dark:bg-green-900/30"
                                />
                              </div>
                            </div>
                          </div>
                          <p className="text-xs text-gray-600 dark:text-gray-400">
                            Confidence levels for each perspective generation
                          </p>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Medical Disclaimer */}
                  <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-4">
                    <div className="flex items-start gap-3">
                      <div className="w-5 h-5 bg-amber-500 rounded-full flex items-center justify-center text-white text-xs font-bold mt-0.5">
                        ⚠️
                      </div>
                      <div>
                        <h5 className="font-semibold text-amber-800 dark:text-amber-200 mb-1">
                          Medical Disclaimer
                        </h5>
                        <p className="text-sm text-amber-700 dark:text-amber-300">
                          This AI-generated summary is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.
                        </p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Enhanced Sidebar */}
          <div className="space-y-6">
            {/* How it Works */}
            <Card className="border-2 border-blue-200 dark:border-blue-800 shadow-lg">
              <CardHeader className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-t-lg">
                <CardTitle className="text-xl text-blue-900 dark:text-blue-100 flex items-center gap-2">
                  <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                  How It Works
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6 p-6">
                <div className="space-y-4">
                  <div className="flex items-start space-x-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                    <div className="p-2 bg-blue-100 dark:bg-blue-900/40 rounded-full">
                      <User className="h-5 w-5 text-blue-700 dark:text-blue-300" />
                    </div>
                    <div>
                      <h4 className="font-semibold text-blue-900 dark:text-blue-100">🏥 Patient Mode</h4>
                      <p className="text-sm text-blue-800 dark:text-blue-200">
                        Generates simplified, accessible summaries using everyday language and clear explanations
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-start space-x-3 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                    <div className="p-2 bg-green-100 dark:bg-green-900/40 rounded-full">
                      <Stethoscope className="h-5 w-5 text-green-700 dark:text-green-300" />
                    </div>
                    <div>
                      <h4 className="font-semibold text-green-900 dark:text-green-100">👨‍⚕️ Clinician Mode</h4>
                      <p className="text-sm text-green-800 dark:text-green-200">
                        Creates technical, detailed summaries with medical terminology and clinical insights
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-start space-x-3 p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                    <div className="p-2 bg-purple-100 dark:bg-purple-900/40 rounded-full">
                      <div className="h-5 w-5 text-purple-700 dark:text-purple-300 text-center">🔄</div>
                    </div>
                    <div>
                      <h4 className="font-semibold text-purple-900 dark:text-purple-100">🔄 Both Modes</h4>
                      <p className="text-sm text-purple-800 dark:text-purple-200">
                        Provides dual perspectives for comprehensive understanding and comparison
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Enhanced Model Features */}
            <Card className="border-2 border-green-200 dark:border-green-800 shadow-lg">
              <CardHeader className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-t-lg">
                <CardTitle className="text-xl text-green-900 dark:text-green-100 flex items-center gap-2">
                  <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                  AI Model Features
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4 p-6">
                <div className="grid grid-cols-1 gap-3">
                  <div className="flex items-center space-x-3 p-2 bg-green-50 dark:bg-green-900/20 rounded-lg">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-sm font-medium text-green-800 dark:text-green-200">Perspective-aware embeddings</span>
                  </div>
                  <div className="flex items-center space-x-3 p-2 bg-green-50 dark:bg-green-900/20 rounded-lg">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-sm font-medium text-green-800 dark:text-green-200">Enhanced JSON/JSONL processing</span>
                  </div>
                  <div className="flex items-center space-x-3 p-2 bg-green-50 dark:bg-green-900/20 rounded-lg">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-sm font-medium text-green-800 dark:text-green-200">Intelligent text extraction</span>
                  </div>
                  <div className="flex items-center space-x-3 p-2 bg-green-50 dark:bg-green-900/20 rounded-lg">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-sm font-medium text-green-800 dark:text-green-200">Advanced provenance tracking</span>
                  </div>
                  <div className="flex items-center space-x-3 p-2 bg-green-50 dark:bg-green-900/20 rounded-lg">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-sm font-medium text-green-800 dark:text-green-200">Faithfulness scoring</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Quick Stats */}
            <Card className="border-2 border-purple-200 dark:border-purple-800 shadow-lg">
              <CardHeader className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-t-lg">
                <CardTitle className="text-xl text-purple-900 dark:text-purple-100 flex items-center gap-2">
                  <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
                  Quick Stats
                </CardTitle>
              </CardHeader>
              <CardContent className="p-6">
                <div className="grid grid-cols-2 gap-4 text-center">
                  <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                    <div className="text-2xl font-bold text-purple-700 dark:text-purple-300">
                      {history.length}
                    </div>
                    <div className="text-xs text-purple-600 dark:text-purple-400">
                      Summaries Generated
                    </div>
                  </div>
                  <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                    <div className="text-2xl font-bold text-blue-700 dark:text-blue-300">
                      {summary ? 'Active' : 'Ready'}
                    </div>
                    <div className="text-xs text-blue-600 dark:text-blue-400">
                      Current Status
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Enhanced History Sidebar */}
            {showHistory && (
              <Card className="border-2 border-indigo-200 dark:border-indigo-800 shadow-lg">
                <CardHeader className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-t-lg">
                  <CardTitle className="text-xl text-indigo-900 dark:text-indigo-100 flex items-center gap-2">
                    <History className="h-5 w-5" />
                    Recent Summaries
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-4">
                  {history.length > 0 ? (
                    <div className="space-y-3 max-h-96 overflow-y-auto">
                      {history.slice(0, 10).map((item, index) => (
                        <div 
                          key={index} 
                          className="p-4 bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-lg cursor-pointer hover:from-indigo-100 hover:to-purple-100 dark:hover:from-indigo-900/30 dark:hover:to-purple-900/30 transition-all duration-200 border border-indigo-200 dark:border-indigo-800 hover:border-indigo-300 dark:hover:border-indigo-700"
                          onClick={() => handleHistoryItemClick(item)}
                        >
                          <div className="flex items-start justify-between mb-2">
                            <Badge variant="outline" className="text-xs border-indigo-300 text-indigo-700 dark:border-indigo-700 dark:text-indigo-300">
                              #{index + 1}
                            </Badge>
                            {item.mode && (
                              <Badge 
                                variant="secondary" 
                                className={`text-xs ${
                                  item.mode === 'patient' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-200' :
                                  item.mode === 'clinician' ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-200' :
                                  'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-200'
                                }`}
                              >
                                {item.mode}
                              </Badge>
                            )}
                          </div>
                          <p className="text-sm text-indigo-800 dark:text-indigo-200 font-medium leading-relaxed">
                            {item.original_text?.substring(0, 80)}...
                          </p>
                          <div className="flex items-center justify-between mt-3">
                            <p className="text-xs text-indigo-600 dark:text-indigo-400">
                              {new Date(item.created_at).toLocaleDateString()}
                            </p>
                            {item.faithfulness_score && (
                              <div className="flex items-center gap-1">
                                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                                <span className="text-xs text-green-600 dark:text-green-400 font-medium">
                                  {((item.faithfulness_score || 0) * 100).toFixed(0)}%
                                </span>
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <div className="w-16 h-16 bg-indigo-100 dark:bg-indigo-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
                        <History className="h-8 w-8 text-indigo-600 dark:text-indigo-400" />
                      </div>
                      <p className="text-indigo-600 dark:text-indigo-400 font-medium">No summaries yet</p>
                      <p className="text-xs text-indigo-500 dark:text-indigo-500 mt-1">
                        Generate your first summary to see it here
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>

      {/* Enhanced Footer */}
      <footer className="bg-gradient-to-r from-gray-50 to-blue-50 dark:from-gray-800 dark:to-blue-900/20 border-t border-blue-200 dark:border-blue-800 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Medical Disclaimer */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
                <div className="w-3 h-3 bg-amber-500 rounded-full"></div>
                Medical Disclaimer
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                This AI-generated content is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.
              </p>
            </div>
            
            {/* Features */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
                <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                Key Features
              </h3>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-2">
                <li>• Dual-perspective medical summaries</li>
                <li>• Enhanced JSON/JSONL processing</li>
                <li>• Intelligent text extraction</li>
                <li>• Quality assessment metrics</li>
                <li>• Professional-grade analysis</li>
              </ul>
            </div>
            
            {/* Contact & Support */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                About
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Medical Summarizer uses advanced AI to provide dual-perspective medical content analysis, helping both patients and healthcare professionals understand complex medical information.
              </p>
              <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-500">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                Powered by Perspective-Aware AI
              </div>
            </div>
          </div>
          
          <div className="border-t border-blue-200 dark:border-blue-800 mt-8 pt-6 text-center">
            <p className="text-sm text-gray-500 dark:text-gray-400">
              © 2025 Medical Summarizer. Built with ❤️ for better healthcare understanding.
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}

// Main App Component
function App() {
  const [token, setToken] = useState(localStorage.getItem('token'))

  const handleLogin = (newToken) => {
    setToken(newToken)
    localStorage.setItem('token', newToken)
  }

  const handleLogout = () => {
    setToken(null)
    localStorage.removeItem('token')
  }

  return (
    <Router
      future={{
        v7_startTransition: true,
        v7_relativeSplatPath: true
      }}
    >
      <div className="App">
        {!token ? (
          <LoginPage onLogin={handleLogin} />
        ) : (
          <SummarizerPage token={token} onLogout={handleLogout} />
        )}
      </div>
    </Router>
  )
}

export default App
