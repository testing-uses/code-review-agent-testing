import { useState } from 'react'

const EVENT_LABELS = {
  received_task:          'Task received',
  repository_scanned:     'Repository scanned',
  knowledge_base_built:   'Knowledge base built',
  knowledge_base_retrieved:'Context retrieved',
  graph_context_built:    'Graph context built',
  dcba_allocated:         'Token budget allocated',
  dev_agent_started:      'Dev Agent started',
  dev_agent_completed:    'Dev Agent completed',
  branch_created:         'Branch created',
  pull_request_created:   'Pull request created',
  code_review_started:    'Code Review Agent started',
  completed:              'Pipeline completed',
}

function eventTitle(event) {
  return EVENT_LABELS[event.event] || event.event?.replace(/_/g, ' ')
}

function dotClass(event, isLast, isActive) {
  if (isLast && isActive) return 'active'
  if (event.level === 'error')     return 'error'
  if (event.event === 'completed') return 'success'
  return 'neutral'
}

function formatTime(ts) {
  if (!ts) return ''
  const d = new Date(ts)
  return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function relativeTime(ts) {
  if (!ts) return ''
  const diff = Date.now() - new Date(ts).getTime()
  const secs = Math.floor(diff / 1000)
  if (secs < 5)   return 'just now'
  if (secs < 60)  return `${secs}s ago`
  const mins = Math.floor(secs / 60)
  if (mins < 60)  return `${mins}m ago`
  return `${Math.floor(mins / 60)}h ago`
}

/* Render JSON details as key–value rows, fall back to raw pre */
function EventDetails({ data }) {
  const entries = Object.entries(data)
  if (entries.length === 0) return null

  // If values are all primitives show key-value table, else raw JSON
  const allPrimitive = entries.every(([, v]) => typeof v !== 'object' || v === null)

  if (allPrimitive) {
    return (
      <div className="event-details">
        {entries.map(([k, v]) => (
          <div key={k} className="event-detail-row">
            <span className="key">{k}</span>
            <span className="val">{String(v)}</span>
          </div>
        ))}
      </div>
    )
  }

  return (
    <pre className="event-raw-pre">
      {JSON.stringify(data, null, 2)}
    </pre>
  )
}

function TimelineCard({ event, index, isLast, isActive }) {
  const [expanded, setExpanded] = useState(isLast && isActive)
  const { event: name, timestamp, level, ...details } = event
  const hasDetails = Object.keys(details).length > 0
  const dc = dotClass(event, isLast, isActive)

  return (
    <div
      className="timeline-item"
      style={{ animationDelay: `${index * 60}ms` }}
    >
      <div className={`timeline-dot ${dc}`} />
      <div
        className={`timeline-card ${expanded ? 'expanded' : ''}`}
        onClick={() => hasDetails && setExpanded(e => !e)}
        style={{ cursor: hasDetails ? 'pointer' : 'default' }}
      >
        <div className="timeline-card-header">
          <span
            className={`tag ${level === 'error' ? 'failure' : dc === 'success' ? 'success' : dc === 'active' ? 'running' : 'neutral'}`}
            style={{ fontSize: '10px', padding: '2px 7px' }}
          >
            {level || 'info'}
          </span>
          <span className="event-title">{eventTitle(event)}</span>
          {isLast && isActive && (
            <span className="live-indicator">
              <span className="live-dot" />
              live
            </span>
          )}
          {timestamp && (
            <time title={new Date(timestamp).toISOString()}>
              {formatTime(timestamp)}
              <span style={{ marginLeft: 4, opacity: 0.55, fontSize: '10px' }}>
                {relativeTime(timestamp)}
              </span>
            </time>
          )}
          {hasDetails && (
            <span className="chevron" aria-hidden="true">▾</span>
          )}
        </div>

        {expanded && hasDetails && (
          <div className="timeline-card-body">
            <EventDetails data={details} />
          </div>
        )}
      </div>
    </div>
  )
}

export default function TimelineView({ events, isActive }) {
  if (!events || events.length === 0) {
    return (
      <div className="timeline-empty">
        {isActive
          ? <><span className="live-indicator"><span className="live-dot" />Waiting for agent events…</span></>
          : 'No pipeline events found for this run.'}
      </div>
    )
  }

  return (
    <div className="timeline">
      {events.map((event, i) => (
        <TimelineCard
          key={`${event.event}-${event.timestamp ?? i}`}
          event={event}
          index={i}
          isLast={i === events.length - 1}
          isActive={isActive}
        />
      ))}
    </div>
  )
}
