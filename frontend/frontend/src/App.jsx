import { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'
import TimelineView from './components/TimelineView'
import JobPill from './components/JobPill'
import PastRunCard from './components/PastRunCard'

const POLL_INTERVAL_MS = 3000
const POLL_INTERVAL_ERROR_MS = 5000

/* ────────────────────────────────────────────────────── */
/*  Helpers                                               */
/* ────────────────────────────────────────────────────── */
function SectionHeader({ title, count }) {
  return (
    <div className="section-header">
      <h2>{title}</h2>
      {count != null && <span className="count-badge">{count}</span>}
      <div className="divider" />
    </div>
  )
}

/* ────────────────────────────────────────────────────── */
/*  App                                                   */
/* ────────────────────────────────────────────────────── */
export default function App() {
  const [taskText, setTaskText]       = useState('')
  const [submitting, setSubmitting]   = useState(false)
  const [submitMsg, setSubmitMsg]     = useState('')

  const [activeRunId, setActiveRunId] = useState(null)
  const [runData, setRunData]         = useState(null)   // { run, jobs, events, logs }
  const [pollMsg, setPollMsg]         = useState('')

  const [pastRuns, setPastRuns]       = useState([])
  const [runsLoading, setRunsLoading] = useState(false)

  const [showLogs, setShowLogs]       = useState(false)

  const pollRef = useRef(null)

  /* ── Poll status ─────────────────────────────────── */
  const pollStatus = useCallback(async (runId) => {
    try {
      const res  = await fetch(`/status/${runId}`)
      const data = await res.json()

      if (data.error) {
        setPollMsg(`Error: ${data.error}`)
        return
      }

      setRunData(data)
      setPollMsg('')

      if (data.run.status !== 'completed') {
        pollRef.current = setTimeout(() => pollStatus(runId), POLL_INTERVAL_MS)
      } else {
        // Refresh past-runs list once pipeline finishes
        loadPastRuns()
      }
    } catch (err) {
      setPollMsg(`Polling error: ${err.message}`)
      pollRef.current = setTimeout(() => pollStatus(runId), POLL_INTERVAL_ERROR_MS)
    }
  }, [])

  /* ── Start polling ───────────────────────────────── */
  const startPolling = useCallback((runId) => {
    if (pollRef.current) clearTimeout(pollRef.current)
    setActiveRunId(runId)
    setRunData(null)
    setPollMsg('Loading run status…')
    pollStatus(runId)
  }, [pollStatus])

  /* ── Load past runs ──────────────────────────────── */
  const loadPastRuns = useCallback(async () => {
    setRunsLoading(true)
    try {
      const res  = await fetch('/runs')
      const data = await res.json()
      if (!data.error) setPastRuns(data)
    } catch (_) {
      // silently ignore
    } finally {
      setRunsLoading(false)
    }
  }, [])

  /* ── Submit task ─────────────────────────────────── */
  const handleSubmit = async () => {
    const text = taskText.trim()
    if (!text || submitting) return

    setSubmitting(true)
    setSubmitMsg('Dispatching workflow…')
    setRunData(null)
    if (pollRef.current) clearTimeout(pollRef.current)

    try {
      const res  = await fetch('/trigger', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_text: text }),
      })
      const data = await res.json()

      if (data.error) {
        setSubmitMsg(`Error: ${data.error}`)
        setSubmitting(false)
        return
      }

      setSubmitMsg('Workflow dispatched — locating run…')
      // Wait a few seconds for GitHub to register the run
      await new Promise(r => setTimeout(r, 4000))

      const runsRes  = await fetch('/runs')
      const runsData = await runsRes.json()
      if (!runsData.error && runsData.length > 0) {
        startPolling(runsData[0].id)
        setPastRuns(runsData)
      }
      setSubmitMsg('')
    } catch (err) {
      setSubmitMsg(`Network error: ${err.message}`)
    } finally {
      setSubmitting(false)
    }
  }

  /* ── Boot ────────────────────────────────────────── */
  useEffect(() => {
    loadPastRuns()
    return () => { if (pollRef.current) clearTimeout(pollRef.current) }
  }, [loadPastRuns])

  /* ── Derived ─────────────────────────────────────── */
  const isRunActive = runData?.run && runData.run.status !== 'completed'
  const runConclusion = runData?.run?.conclusion
  const runCls =
    runConclusion === 'success'  ? 'success' :
    runConclusion === 'failure'  ? 'failure' :
    isRunActive                  ? 'running' : 'neutral'

  return (
    <div className="app">
      {/* ── Header ─────────────────────────── */}
      <header className="app-header">
        <h1>Agentic AI <span>Pipeline</span></h1>
        <p>
          Submit a task - Dev Agent, Orchestrator, and Code Review Agent run inside
          GitHub Actions and stream live updates below.
        </p>
      </header>

      {/* ── Submit ─────────────────────────── */}
      <div className="glass submit-panel">
        <label htmlFor="task-input">Task description</label>
        <textarea
          id="task-input"
          rows={5}
          value={taskText}
          onChange={e => setTaskText(e.target.value)}
          placeholder="e.g. create a function for overdue list"
          onKeyDown={e => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleSubmit()
          }}
        />
        <div className="submit-row">
          <button
            id="submit-btn"
            className="btn-primary"
            onClick={handleSubmit}
            disabled={submitting || !taskText.trim()}
          >
            {submitting
              ? <><span className="spinner" style={{ width: 13, height: 13, borderWidth: 2 }} />Dispatching…</>
              : ' Submit Task'
            }
          </button>
          <kbd style={{ fontSize: 11, color: 'var(--text-muted)', userSelect: 'none' }}>⌘ Enter</kbd>
          {submitMsg && <span className="submit-status">{submitMsg}</span>}
        </div>
      </div>

      {/* ── Run status bar ─────────────────── */}
      {runData?.run && (
        <div className="run-status-bar">
          <span className="label">Run status</span>
          <span className={`tag ${runCls}`}>
            {isRunActive && <span className="live-dot" style={{ display: 'inline-block', marginRight: 4 }} />}
            {runData.run.status}
            {runData.run.conclusion && ` · ${runData.run.conclusion}`}
          </span>
          {pollMsg && <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>{pollMsg}</span>}
          <a
            className="gh-link"
            href={runData.run.html_url}
            target="_blank"
            rel="noreferrer"
          >
            View on GitHub ↗
          </a>
        </div>
      )}

      {/* ── Job pills ──────────────────────── */}
      {runData?.jobs?.length > 0 && (
        <div className="jobs-row">
          {runData.jobs.map(job => (
            <JobPill key={job.id} job={job} />
          ))}
        </div>
      )}

      {/* ── Timeline ───────────────────────── */}
      {(activeRunId || runData) && (
        <div className="timeline-wrap">
          <SectionHeader
            title="Agent Timeline"
            count={runData?.events?.length ?? 0}
          />
          <TimelineView
            events={runData?.events ?? []}
            isActive={!!isRunActive}
          />
        </div>
      )}

      {/* ── Full logs ──────────────────────── */}
      {runData?.logs?.length > 0 && (
        <div className="logs-panel">
          <SectionHeader title="Raw Logs" />
          <button
            id="toggle-logs-btn"
            className="btn-secondary"
            style={{ marginBottom: 14 }}
            onClick={() => setShowLogs(v => !v)}
          >
            {showLogs ? '▲ Hide raw logs' : '▼ Show raw logs'}
          </button>
          {showLogs && (
            <div>
              {runData.logs.map(job => (
                <div key={job.job_id} className="logs-job-block">
                  <h4>{job.job_name}</h4>
                  <pre className="logs-pre">{job.lines.join('\n')}</pre>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── Past runs ──────────────────────── */}
      <div className="past-runs">
        <SectionHeader title="Recent Runs" />
        {runsLoading && pastRuns.length === 0 && (
          <div className="timeline-empty" style={{ padding: 24 }}>Loading runs…</div>
        )}
        {!runsLoading && pastRuns.length === 0 && (
          <div className="timeline-empty" style={{ padding: 24 }}>No recent runs found.</div>
        )}
        {pastRuns.map(run => (
          <PastRunCard
            key={run.id}
            run={run}
            isActive={run.id === activeRunId}
            onLoadTimeline={startPolling}
          />
        ))}
      </div>
    </div>
  )
}
