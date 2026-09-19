import { useEffect, useRef, useState } from 'react'
import { Badge, Empty, formatTime } from './common'

/**
 * Domain-specific minutes: pick a meeting context, generate, review/edit,
 * approve, export. All calls are passed in as callbacks (same convention as
 * ReviewItem/ApprovalGate) -- the actual `api.*` calls and refetch live in
 * MeetingDetail's `guard()`, exactly like every other tab.
 *
 * Export buttons stay disabled until the document is `approved`: the human
 * sign-off, not the template, is what makes a document eligible to leave the
 * system -- the same rule ExportTab already applies to action items.
 */

const DOMAINS = [
  { value: 'technology', label: 'Technology / IT' },
  { value: 'education', label: 'Education / Academic' },
  { value: 'healthcare', label: 'Healthcare' },
  { value: 'business', label: 'Business / Corporate' },
]

/** Every domain template renders its topics/decisions/actions sections under
 * these same three keys (see backend/app/services/minutes_templates.py), each
 * either an empty-state sentence or a list of "- " bullet lines -- counting
 * those lines describes exactly what is actually in the generated document,
 * nothing inferred beyond it. */
function countBullets(sections, key) {
  const section = sections.find((s) => s.key === key)
  if (!section) return 0
  return section.body.split('\n').filter((line) => line.trim().startsWith('-')).length
}

export default function DomainMinutes({
  generatedMinutes,
  busy,
  onGenerate,
  onEdit,
  onApprove,
  onExport,
}) {
  const [domain, setDomain] = useState('technology')
  const [activeId, setActiveId] = useState(generatedMinutes[0]?.id ?? null)
  const [draft, setDraft] = useState(generatedMinutes[0]?.sections ?? [])
  const prevCount = useRef(generatedMinutes.length)

  // Jump to the newest document whenever a new generation lands; otherwise
  // leave whatever the reviewer is currently looking at alone.
  useEffect(() => {
    if (generatedMinutes.length > prevCount.current) {
      setActiveId(generatedMinutes[0]?.id ?? null)
    } else if (!generatedMinutes.some((g) => g.id === activeId)) {
      setActiveId(generatedMinutes[0]?.id ?? null)
    }
    prevCount.current = generatedMinutes.length
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [generatedMinutes])

  const active = generatedMinutes.find((g) => g.id === activeId) || null

  useEffect(() => {
    setDraft(active ? active.sections : [])
  }, [active])

  const updateSection = (index, body) => {
    setDraft((prev) => prev.map((s, i) => (i === index ? { ...s, body } : s)))
  }

  const decisionCount = active ? countBullets(active.sections, 'decisions') : 0
  const actionCount = active ? countBullets(active.sections, 'actions') : 0
  const topicCount = active ? countBullets(active.sections, 'topics') : 0

  return (
    <>
      <div className="domain-picker">
        <h3 style={{ marginTop: 0 }}>Choose meeting type</h3>
        <p className="hint" style={{ marginTop: 0 }}>
          Select a meeting domain to organize the verified meeting information into a
          domain-specific minutes format.
        </p>
        <div className="row" style={{ flexWrap: 'wrap' }}>
          {DOMAINS.map((d) => (
            <button
              key={d.value}
              className={domain === d.value ? 'primary small' : 'small'}
              onClick={() => setDomain(d.value)}
            >
              {d.label}
            </button>
          ))}
        </div>
        <button
          className="primary"
          style={{ marginTop: 10 }}
          disabled={busy}
          onClick={() => onGenerate(domain)}
        >
          Generate Minutes
        </button>
        <p className="hint" style={{ marginTop: 8, marginBottom: 0 }}>
          Only approved and edited decisions/actions are included in the final minutes.
          Rejected and not-yet-reviewed items are kept for audit but left out here — review
          them in the <strong>Minutes</strong> tab first.
        </p>
      </div>

      {!active ? (
        <Empty>No minutes generated yet for this meeting. Choose a meeting type above.</Empty>
      ) : (
        <div style={{ marginTop: 18 }}>
          <div className="row" style={{ marginBottom: 4 }}>
            <strong style={{ flex: 1, fontSize: 16 }}>
              {DOMAINS.find((d) => d.value === active.domain)?.label || active.domain}
            </strong>
            <span className="muted">generated {formatTime(active.created_at)}</span>
          </div>
          <p className="row" style={{ marginTop: 0, marginBottom: 4, gap: 6 }}>
            <span className="muted">Status:</span>
            <Badge value={active.status} />
            {active.status === 'approved' && (
              <span style={{ color: 'var(--ok)', fontWeight: 700 }}>✓</span>
            )}
          </p>
          <p className="muted" style={{ marginTop: 0, marginBottom: 14 }}>
            {decisionCount} Decision{decisionCount === 1 ? '' : 's'} · {actionCount} Action
            Item{actionCount === 1 ? '' : 's'} · {topicCount} Topic{topicCount === 1 ? '' : 's'}
          </p>

          {draft.map((section, i) => (
            <div key={section.key} style={{ marginBottom: 14 }}>
              <label>{section.heading}</label>
              <textarea
                value={section.body}
                onChange={(e) => updateSection(i, e.target.value)}
                rows={Math.min(10, Math.max(3, section.body.split('\n').length))}
              />
            </div>
          ))}

          <div className="row">
            <button disabled={busy} onClick={() => onEdit(active.id, draft)}>
              Save edits
            </button>
            {active.status !== 'approved' && (
              <button className="ok" disabled={busy} onClick={() => onApprove(active.id)}>
                Approve
              </button>
            )}
          </div>

          <h3 style={{ marginTop: 22 }}>Export</h3>
          {active.status !== 'approved' && (
            <p className="muted" style={{ marginTop: 0 }}>
              Approve this document before it can be exported.
            </p>
          )}
          <div className="row">
            {['json', 'csv', 'markdown'].map((fmt) => (
              <button
                key={fmt}
                disabled={busy || active.status !== 'approved'}
                onClick={() => onExport(active.id, fmt)}
              >
                Request {fmt.toUpperCase()} export
              </button>
            ))}
          </div>

          {generatedMinutes.length > 1 && (
            <>
              <h3 style={{ marginTop: 22 }}>History</h3>
              <p className="hint" style={{ marginTop: 0 }}>
                Generating again — for this domain or a different one — creates a new version
                below rather than overwriting this one.
              </p>
              <ul className="list">
                {generatedMinutes.map((g) => (
                  <li
                    key={g.id}
                    className={`clickable ${g.id === activeId ? 'selected' : ''}`}
                    onClick={() => setActiveId(g.id)}
                  >
                    <div className="row">
                      <strong style={{ flex: 1 }}>
                        {DOMAINS.find((d) => d.value === g.domain)?.label || g.domain}
                      </strong>
                      <Badge value={g.status} />
                    </div>
                    <div className="muted">{formatTime(g.created_at)}</div>
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
    </>
  )
}
