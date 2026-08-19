const STATUS_CLASS = {
  success:     'success',
  failure:     'failure',
  cancelled:   'failure',
  in_progress: 'in_progress',
  queued:      'queued',
  waiting:     'queued',
}

const STATUS_ICON = {
  success:     '✓',
  failure:     '✕',
  cancelled:   '✕',
  in_progress: null, // spinner
  queued:      '○',
  waiting:     '○',
}

export default function JobPill({ job }) {
  const cls  = STATUS_CLASS[job.conclusion ?? job.status] ?? 'neutral'
  const icon = STATUS_ICON[job.conclusion ?? job.status]
  const showSpinner = (job.status === 'in_progress' || job.status === 'queued') && !job.conclusion

  return (
    <span className={`job-pill ${cls}`}>
      {showSpinner
        ? <span className="spinner" aria-hidden="true" />
        : icon && <span aria-hidden="true">{icon}</span>
      }
      <span>{job.name}</span>
      {job.conclusion && <span style={{ opacity: 0.7 }}>· {job.conclusion}</span>}
    </span>
  )
}
