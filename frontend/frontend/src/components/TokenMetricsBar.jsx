import React from 'react'

export default function TokenMetricsBar({ events = [] }) {
  // Extract DCBA & usage metrics from events
  const dcbaAlloc = events.find(e => e.event === 'dcba_allocated')
  const devFinish = events.find(e => e.event === 'dev_agent_finished')
  const dcbaRefined = events.find(e => e.event === 'dcba_review_budget_refined')
  const reviewFinish = events.find(e => e.event === 'code_review_finished')

  const totalBudget = dcbaAlloc?.total_budget || 11000
  const devAlloc = dcbaAlloc?.dev_agent_tokens || 0
  const devActual = devFinish?.usage?.total_tokens || 0
  const reviewInitialAlloc = dcbaAlloc?.code_review_agent_tokens_initial_estimate || 0
  const reviewRefinedAlloc = dcbaRefined?.refined_tokens || reviewInitialAlloc
  const reviewActual = reviewFinish?.result?.usage?.total_tokens || 0

  const totalConsumed = devActual + reviewActual
  const savingsTokens = Math.max(0, (devAlloc + reviewRefinedAlloc) - totalConsumed)
  const savingsPct = (devAlloc + reviewRefinedAlloc) > 0 ? Math.round((savingsTokens / (devAlloc + reviewRefinedAlloc)) * 100) : 0

  if (!dcbaAlloc && !devFinish) return null

  return (
    <div className="token-metrics-dashboard">
      <div className="metrics-grid">
        <div className="metric-card">
          <span className="metric-title">TOTAL PIPELINE BUDGET</span>
          <span className="metric-val">{totalBudget.toLocaleString()} <span className="unit">tokens</span></span>
          <span className="metric-sub">Dynamic Softmax Cap</span>
        </div>

        <div className="metric-card">
          <span className="metric-title">DEV AGENT USAGE</span>
          <span className="metric-val accent">
            {devActual > 0 ? devActual.toLocaleString() : '—'} 
            <span className="unit">/ {devAlloc ? devAlloc.toLocaleString() : '—'}</span>
          </span>
          <span className="metric-sub">{devFinish?.latency_ms ? `${Math.round(devFinish.latency_ms)}ms latency` : 'Waiting for Dev completion'}</span>
        </div>

        <div className="metric-card">
          <span className="metric-title">CODE REVIEW USAGE</span>
          <span className="metric-val purple">
            {reviewActual > 0 ? reviewActual.toLocaleString() : '—'}
            <span className="unit">/ {reviewRefinedAlloc ? reviewRefinedAlloc.toLocaleString() : '—'}</span>
          </span>
          <span className="metric-sub">{dcbaRefined ? 'Phase-2 diff refined' : 'Initial estimate'}</span>
        </div>

        <div className="metric-card highlight">
          <span className="metric-title">TOKEN OPTIMIZATION</span>
          <span className="metric-val green">{savingsPct}% <span className="unit">saved</span></span>
          <span className="metric-sub">{savingsTokens > 0 ? `${savingsTokens.toLocaleString()} tokens conserved` : 'Adaptive budgeting'}</span>
        </div>
      </div>
    </div>
  )
}
