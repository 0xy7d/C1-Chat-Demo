import { useState } from 'react'
import { C1Chat, ThemeProvider } from '@thesysai/genui-sdk'
import "@crayonai/react-ui/styles/index.css"
import './App.css'

function App() {
  const [mode, setMode] = useState<'light' | 'dark'>('light')
  const apiUrl = import.meta.env.VITE_API_URL

  return (
    <div className="app-container">
      <button
        className="theme-toggle"
        onClick={() => setMode(mode === 'light' ? 'dark' : 'light')}
      >
        {mode === 'light' ? '🌙' : '☀️'}
      </button>

      <ThemeProvider mode={mode}>
        <C1Chat apiUrl={apiUrl} />
      </ThemeProvider>
    </div>
  )
}

export default App

