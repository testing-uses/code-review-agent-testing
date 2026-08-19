import { useState, useMemo } from 'react'

const EVENT_CONFIG = {
  task_received: {
    title: 'Task Received by Orchestrator',
    agent: 'Orchestrator',
    code: 'ORCH',
    category: 'orchestrator',
    badgeClass: 'badge-orchestrator',
  },
  knowledge_base_checked: {
    title: 'Knowledge Base Graph Verification',
    agent: 'Knowledge Base',
    code: 'KB',
    category: 'kb',
    badgeClass: 'badge-kb',
  },
  knowledge_base_context_analyzed: {
    title: 'AST Symbol & Context Footprint Analyzed',
    agent: 'Knowledge Base',
    code: 'KB',
    category: 'kb',
    badgeClass: 'badge-kb',
  },
  dcba_allocated: {
    title: 'DCBA Phase 1: Complexity Softmax & Token Allocation',
    agent: 'DCBA Engine',
    code: 'DCBA',
    category: 'dcba',
    badgeClass: 'badge-dcba',
  },
  dev_agent_started: {
    title: 'Dev Agent Dispatched with Context MMR',
    agent: 'Dev Agent',
    code: 'DEV',
    category: 'dev',
    badgeClass: 'badge-dev',
  },
  dev_agent_finished: {
    title: 'Dev Agent Completed Synthesis & Patching',
    agent: 'Dev Agent',
    code: 'DEV',
    category: 'dev',
    badgeClass: 'badge-dev',
  },
  git_operations_started: {
    title: 'Git Operations & Branch Isolation',
    agent: 'GitOps Engine',
    code: 'GIT',
    category: 'git',
    badgeClass: 'badge-git',
  },
  branch_pushed: {
    title: 'Branch Pushed to Remote',
    agent: 'GitOps Engine',
    code: 'GIT',
    category: 'git',
    badgeClass: 'badge-git',
  },
  pull_request_created: {
    title: 'Pull Request Created on GitHub',
    agent: 'GitHub Bot',
    code: 'PR',
    category: 'git',
    badgeClass: 'badge-git',
  },
  dcba_review_budget_refined: {
    title: 'DCBA Phase 2: Diff-Measured Budget Refinement',
    agent: 'DCBA Engine',
    code: 'DCBA',
    category: 'dcba',
    badgeClass: 'badge-dcba',
  },
  code_review_started: {
    title: 'Code Review Agent Initialized',
    agent: 'Code Reviewer',
    code: 'REVIEW',
    category: 'review',
    badgeClass: 'badge-review',
  },
  review_files_identified: {
    title: 'Reviewable Source Files Filtered',
    agent: 'Code Reviewer',
    code: 'REVIEW',
    category: 'review',
    badgeClass: 'badge-review',
  },
  review_context_built: {
    title: 'Token-Budgeted Review Context Formatted',
    agent: 'Code Reviewer',
    code: 'REVIEW',
    category: 'review',
    badgeClass: 'badge-review',
  },
  review_context_ready: {
    title: 'Stage 1: Rubric Prompt Dispatched',
    agent: 'Code Reviewer',
    code: 'REVIEW',
    category: 'review',
    badgeClass: 'badge-review',
  },
  raw_findings_generated: {
    title: 'Stage 1: 8-Category Findings Synthesized',
    agent: 'Code Reviewer',
    code: 'REVIEW',
    category: 'review',
    badgeClass: 'badge-review',
  },
  findings_verified: {
    title: 'Stage 2: Skeptical Verifier Audited Findings',
    agent: 'Skeptical Verifier',
    code: 'VERIFY',
    category: 'review',
    badgeClass: 'badge-verifier',
  },
  breaking_change_detected: {
    title: 'Breaking API Change Detected in Downstream Usages',
    agent: 'AST Dependency Checker',
    code: 'WARN',
    category: 'review',
    badgeClass: 'badge-danger',
  },
  review_decision_made: {
    title: 'Hard-Gate Decision Engine Evaluated',
    agent: 'Decision Engine',
    code: 'GATE',
    category: 'decision',
    badgeClass: 'badge-decision',
  },
  review_posted_to_github: {
    title: 'Review Audit & Labels Posted to Pull Request',
    agent: 'GitHub Bot',
    code: 'PR',
    category: 'decision',
    badgeClass: 'badge-git',
  },
  code_review_finished: {
    title: 'Code Review Finished',
    agent: 'Code Reviewer',
    code: 'REVIEW',
    category: 'review',
    badgeClass: 'badge-review',
  },
  human_approval_required: {
    title: 'Human Gatekeeper Approval Required',
    agent: 'State Machine Gate',
    code: 'GATE',
    category: 'gate',
    badgeClass: 'badge-gate',
  },
  summary_generated: {
    title: 'Executive Multi-Agent Summary Generated',
    agent: 'Orchestrator',
    code: 'ORCH',
    category: 'orchestrator',
    badgeClass: 'badge-orchestrator',
  },
  summary_generation_failed: {
    title: 'Summary Generation Failed',
    agent: 'Orchestrator',
    code: 'WARN',
    category: 'orchestrator',
    badgeClass: 'badge-danger',
  },
  pipeline_finished: {
    title: 'Pipeline Lifecycle Execution Complete',
    agent: 'Orchestrator',
    code: 'DONE',
    category: 'orchestrator',
    badgeClass: 'badge-success',
  },
  workflow_failed: {
    title: 'Pipeline Halted / Step Failed',
    agent: 'State Machine',
    code: 'HALT',
    category: 'error',
    badgeClass: 'badge-danger',
  },
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

function SpecializedCardBody({ event, data }) {
  const { event: eventType } = event

  if (eventType === 'task_received') {
    return (
      <div className="event-specialized-block">
        <div className="task-preview-box">
          <span className="task-id-badge">ID: {data.task_id}</span>
          <p className="task-prompt-text">{data.task}</p>
        </div>
      </div>
    )
  }

  if (eventType === 'dcba_allocated') {
    return (
      <div className="event-specialized-block">
        <div className="token-split-row">
          <div className="token-split-item">
            <span className="token-label">Dev Agent</span>
            <span className="token-count">{data.dev_agent_tokens?.toLocaleString()} <span className="u">tokens</span></span>
          </div>
          <div className="token-split-item">
            <span className="token-label">Review Agent Initial</span>
            <span className="token-count">{data.code_review_agent_tokens_initial_estimate?.toLocaleString()} <span className="u">tokens</span></span>
          </div>
          <div className="token-split-item">
            <span className="token-label">Total Cap</span>
            <span className="token-count">{data.total_budget?.toLocaleString()} <span className="u">tokens</span></span>
          </div>
        </div>
      </div>
    )
  }

  if (eventType === 'dev_agent_finished') {
    return (
      <div className="event-specialized-block">
        <div className="files-changed-pills">
          <span className="pill-title">Files Changed:</span>
          {data.changed_files?.length > 0 ? (
            data.changed_files.map(f => <span key={f} className="file-pill">{f}</span>)
          ) : (
            <span className="file-pill none">None</span>
          )}
        </div>
        {data.usage && (
          <div className="usage-stats-chip">
            <span>Tokens: {data.usage.total_tokens?.toLocaleString() || 0}</span>
            <span>Latency: {Math.round(data.latency_ms || 0)}ms</span>
            <span>Status: <strong className="status-success">{data.status}</strong></span>
          </div>
        )}
      </div>
    )
  }

  if (eventType === 'pull_request_created') {
    return (
      <div className="event-specialized-block">
        <div className="pr-info-card">
          <span className="pr-number-tag">PR #{data.number}</span>
          <span className="pr-branch-tag">branch: {data.branch}</span>
        </div>
      </div>
    )
  }

  if (eventType === 'dcba_review_budget_refined') {
    return (
      <div className="event-specialized-block">
        <div className="budget-refine-chip">
          <span>Initial: {data.initial_estimate?.toLocaleString()} tokens</span>
          <span className="arrow">➔</span>
          <span className="refined">Refined: {data.refined_tokens?.toLocaleString()} tokens</span>
          <span className="badge-saved">Diff-measured savings</span>
        </div>
      </div>
    )
  }

  if (eventType === 'review_decision_made') {
    const actionClass = data.action === 'AUTO_APPROVE' ? 'action-approve' : data.action === 'HUMAN_REVIEW' ? 'action-human' : 'action-reject'
    return (
      <div className="event-specialized-block">
        <div className="decision-banner">
          <span className={`decision-action-pill ${actionClass}`}>{data.action}</span>
          <span className="decision-score">Score: <strong>{data.weighted_score}</strong>/100</span>
        </div>
        {data.reasons?.length > 0 && (
          <ul className="decision-reasons-list">
            {data.reasons.map((r, i) => <li key={i}>{r}</li>)}
          </ul>
        )}
      </div>
    )
  }

  if (eventType === 'findings_verified') {
    return (
      <div className="event-specialized-block">
        <div className="findings-summary-pills">
          <span className="f-pill submitted">Candidate Findings: {data.submitted}</span>
          <span className="f-pill confirmed">Confirmed: {data.confirmed}</span>
          {data.discarded > 0 && <span className="f-pill discarded">Discarded (filtered): {data.discarded}</span>}
        </div>
      </div>
    )
  }

  // Fallback: key-value table
  const entries = Object.entries(data)
  if (entries.length === 0) return null

  const allPrimitive = entries.every(([, v]) => typeof v !== 'object' || v === null)
  if (allPrimitive) {
    return (
      <div className="event-details-grid">
        {entries.map(([k, v]) => (
          <div key={k} className="detail-kv-row">
            <span className="k">{k}:</span>
            <span className="v">{String(v)}</span>
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
  const [expanded, setExpanded] = useState(isLast || index >= 0)
  const { event: eventName, timestamp, level, ...details } = event
  const config = EVENT_CONFIG[eventName] || {
    title: eventName.replace(/_/g, ' '),
    agent: 'Pipeline Agent',
    code: 'LOG',
    category: 'general',
    badgeClass: 'badge-neutral',
  }

  const isFailed = level === 'error' || eventName === 'workflow_failed'
  const isSuccess = eventName === 'pipeline_finished' || eventName === 'summary_generated'
  const dotState = isFailed ? 'error' : isSuccess ? 'success' : (isLast && isActive) ? 'active' : 'neutral'

  return (
    <div
      className="antigravity-timeline-card-wrapper"
      style={{ animationDelay: `${index * 40}ms` }}
    >
      <div className={`timeline-dot ${dotState}`} />
      <div className={`antigravity-timeline-card ${expanded ? 'expanded' : ''} ${dotState}`}>
        <div className="card-top-bar" onClick={() => setExpanded(e => !e)}>
          <div className="agent-badge-wrap">
            <span className={`agent-name-tag ${config.badgeClass}`}>{config.code || config.agent}</span>
          </div>
          
          <span className="event-main-title">{config.title}</span>

          {isLast && isActive && (
            <span className="live-pulse-badge">
              <span className="live-dot" />
              LIVE
            </span>
          )}

          {timestamp && (
            <time className="event-timestamp" title={new Date(timestamp).toISOString()}>
              {formatTime(timestamp)}
              <span className="rel-time">{relativeTime(timestamp)}</span>
            </time>
          )}

          <span className="chevron-toggle" aria-hidden="true">v</span>
        </div>

        {expanded && (
          <div className="card-content-body">
            <SpecializedCardBody event={event} data={details} />
          </div>
        )}
      </div>
    </div>
  )
}

export default function TimelineView({ events = [], isActive }) {
  const [filter, setFilter] = useState('ALL')

  const filteredEvents = useMemo(() => {
    if (filter === 'ALL') return events
    if (filter === 'ORCHESTRATOR') return events.filter(e => ['task_received', 'knowledge_base_checked', 'knowledge_base_context_analyzed', 'dcba_allocated', 'dcba_review_budget_refined', 'summary_generated', 'pipeline_finished', 'workflow_failed'].includes(e.event))
    if (filter === 'DEV') return events.filter(e => ['dev_agent_started', 'dev_agent_finished', 'git_operations_started', 'branch_pushed', 'pull_request_created'].includes(e.event))
    if (filter === 'REVIEW') return events.filter(e => ['code_review_started', 'review_files_identified', 'review_context_built', 'review_context_ready', 'raw_findings_generated', 'findings_verified', 'breaking_change_detected', 'review_decision_made', 'review_posted_to_github', 'code_review_finished', 'human_approval_required'].includes(e.event))
    return events
  }, [events, filter])

  if (!events || events.length === 0) {
    return (
      <div className="timeline-empty-antigravity">
        {isActive ? (
          <div className="waiting-animation">
            <span className="live-dot big" />
            <p>Orchestrator initialized -- listening for execution trace...</p>
          </div>
        ) : (
          <p>No execution timeline recorded for this task run.</p>
        )}
      </div>
    )
  }

  return (
    <div className="antigravity-timeline-root">
      <div className="timeline-controls-bar">
        <div className="filter-chips">
          <button className={`filter-chip ${filter === 'ALL' ? 'active' : ''}`} onClick={() => setFilter('ALL')}>
            All Steps ({events.length})
          </button>
          <button className={`filter-chip ${filter === 'ORCHESTRATOR' ? 'active' : ''}`} onClick={() => setFilter('ORCHESTRATOR')}>
            Orchestrator & DCBA
          </button>
          <button className={`filter-chip ${filter === 'DEV' ? 'active' : ''}`} onClick={() => setFilter('DEV')}>
            Dev Agent
          </button>
          <button className={`filter-chip ${filter === 'REVIEW' ? 'active' : ''}`} onClick={() => setFilter('REVIEW')}>
            Code Review & Verifier
          </button>
        </div>
      </div>

      <div className="timeline-items-stream">
        {filteredEvents.map((event, i) => (
          <TimelineCard
            key={`${event.event}-${event.timestamp ?? i}`}
            event={event}
            index={i}
            isLast={i === filteredEvents.length - 1}
            isActive={isActive}
          />
        ))}
      </div>
    </div>
  )
}
