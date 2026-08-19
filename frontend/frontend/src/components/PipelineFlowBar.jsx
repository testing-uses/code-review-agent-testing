import React from 'react'

const PIPELINE_PHASES = [
  { id: 'kb', label: '1. Task & KB', code: '01', events: ['task_received', 'knowledge_base_checked', 'knowledge_base_context_analyzed'] },
  { id: 'dcba', label: '2. DCBA Budget', code: '02', events: ['dcba_allocated'] },
  { id: 'dev', label: '3. Dev Agent', code: '03', events: ['dev_agent_started', 'dev_agent_finished'] },
  { id: 'git', label: '4. GitOps & PR', code: '04', events: ['git_operations_started', 'branch_pushed', 'pull_request_created'] },
  { id: 'review', label: '5. Code Review', code: '05', events: ['dcba_review_budget_refined', 'code_review_started', 'review_files_identified', 'review_context_built', 'review_context_ready', 'raw_findings_generated', 'findings_verified', 'breaking_change_detected'] },
  { id: 'gate', label: '6. Decision & Gate', code: '06', events: ['review_decision_made', 'review_posted_to_github', 'code_review_finished', 'human_approval_required', 'summary_generated', 'pipeline_finished'] },
]

export default function PipelineFlowBar({ events = [], isRunActive, conclusion }) {
  const eventNames = new Set(events.map(e => e.event))
  const failed = conclusion === 'failure' || events.some(e => e.event === 'workflow_failed')
  const completed = conclusion === 'success' || eventNames.has('pipeline_finished')

  const getPhaseStatus = (phase, index) => {
    const hasEvents = phase.events.some(ev => eventNames.has(ev))
    const isNext = !hasEvents && PIPELINE_PHASES.slice(0, index).every(p => p.events.some(ev => eventNames.has(ev)))

    if (failed && isNext) return 'failed'
    if (hasEvents) {
      // Check if subsequent phases are already underway or pipeline finished
      const subsequentHaveEvents = PIPELINE_PHASES.slice(index + 1).some(p => p.events.some(ev => eventNames.has(ev)))
      if (subsequentHaveEvents || completed) return 'completed'
      return isRunActive ? 'active' : 'completed'
    }
    if (isNext && isRunActive) return 'active'
    return 'pending'
  }

  return (
    <div className="pipeline-flow-container">
      <div className="flow-header">
        <span className="flow-title">Pipeline Execution Flow</span>
        <span className={`tag ${completed ? 'success' : failed ? 'failure' : isRunActive ? 'running' : 'neutral'}`}>
          {completed ? 'Pipeline Completed' : failed ? 'Pipeline Interrupted' : isRunActive ? 'Active Working Stream' : 'Ready'}
        </span>
      </div>

      <div className="flow-steps-track">
        {PIPELINE_PHASES.map((phase, idx) => {
          const status = getPhaseStatus(phase, idx)
          return (
            <React.Fragment key={phase.id}>
              <div className={`flow-step-node ${status}`}>
                <div className="step-badge">
                  <span className="step-code">{phase.code}</span>
                  {status === 'completed' && <span className="status-marker check">OK</span>}
                  {status === 'active' && <span className="status-marker pulse" />}
                  {status === 'failed' && <span className="status-marker fail">ERR</span>}
                </div>
                <div className="step-label">{phase.label}</div>
              </div>
              {idx < PIPELINE_PHASES.length - 1 && (
                <div className={`flow-connector-line ${status === 'completed' ? 'filled' : status === 'active' ? 'active-flow' : ''}`} />
              )}
            </React.Fragment>
          )
        })}
      </div>
    </div>
  )
}
