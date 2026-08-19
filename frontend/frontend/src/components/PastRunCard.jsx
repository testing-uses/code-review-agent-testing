function formatDate(iso) {
  if (!iso) return '—'
  const d = new Date(iso)
  return d.toLocaleString(undefined, {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })
}

const conclusionClass = (run) => {
  if (run.conclusion === 'success')                         return 'success'
  if (run.conclusion === 'failure' || run.conclusion === 'cancelled') return 'failure'
  if (run.status === 'in_progress' || run.status === 'queued')        return 'running'
  return 'neutral'
}

const conclusionLabel = (run) => {
  if (run.conclusion) return run.conclusion
  return run.status
}

export default function PastRunCard({ run, onLoadTimeline, isActive }) {
  const cls = conclusionClass(run)
  return (
    <div className={`run-card ${isActive ? 'active' : ''}`}
         style={isActive ? { borderColor: 'rgba(139,92,246,0.4)', boxShadow: '0 0 0 1px rgba(139,92,246,0.15)' } : {}}>
      <div className="run-meta">
        <span className={`tag ${cls}`}>{conclusionLabel(run)}</span>
        <div className="run-date">{formatDate(run.created_at)}</div>
      </div>
      <div className="run-actions">
        <a href={run.html_url} target="_blank" rel="noreferrer">
          View on GitHub ↗
        </a>
        <button
          className="btn-secondary"
          style={{ padding: '5px 12px', fontSize: '12px' }}
          onClick={() => onLoadTimeline(run.id)}
        >
          {isActive ? '↺ Refresh' : 'Load timeline'}
        </button>
      </div>
    </div>
  )
}
