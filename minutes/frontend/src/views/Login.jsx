import { useEffect, useState } from 'react'
import { api, setSession } from '../api'
import { Banner } from '../components/common'

export default function Login({ onSignedIn }) {
  const [mode, setMode] = useState('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [fullName, setFullName] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [demo, setDemo] = useState(null)
  const [captchaRequired, setCaptchaRequired] = useState(false)
  const [captcha, setCaptcha] = useState(null)
  const [captchaAnswer, setCaptchaAnswer] = useState('')

  // Only offer the demo button once the server confirms the seeded account
  // exists. Showing it on a fresh database would advertise a login that fails.
  useEffect(() => {
    api.demoAvailability().then(setDemo).catch(() => setDemo(null))
    // CAPTCHA_ENABLED defaults to false (see backend/app/config.py); when it
    // is, the challenge itself is fetched lazily, only once the register form
    // is actually shown (below), not on every page load.
    api.registrationPolicy().then((p) => setCaptchaRequired(!!p.captcha_required)).catch(() => {})
  }, [])

  const refreshCaptcha = () => {
    setCaptchaAnswer('')
    api.getCaptcha().then(setCaptcha).catch(() => setCaptcha(null))
  }

  useEffect(() => {
    if (mode === 'register' && captchaRequired && !captcha) refreshCaptcha()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, captchaRequired])

  const signInAsDemo = async () => {
    setError('')
    setBusy(true)
    try {
      const session = await api.login(demo.email, demo.password)
      setSession(session.access_token, session.user)
      onSignedIn(session.user)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const submit = async (e) => {
    e.preventDefault()
    setError('')
    setBusy(true)
    try {
      if (mode === 'register') {
        const payload = { email, password, full_name: fullName || null, role: 'reviewer' }
        if (captchaRequired) {
          payload.captcha_id = captcha?.captcha_id
          payload.captcha_answer = captchaAnswer
        }
        await api.register(payload)
      }
      const session = await api.login(email, password)
      setSession(session.access_token, session.user)
      onSignedIn(session.user)
    } catch (err) {
      setError(err.message)
      // A challenge is single-use (backend/app/security/captcha.py) -- any
      // failed attempt has consumed it, so get a fresh one for the retry.
      if (mode === 'register' && captchaRequired) refreshCaptcha()
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="login-wrap">
      <div className="card">
        <h2>Meeting Intelligence</h2>
        <p className="hint">Reviewer editor — sign in to review and approve meeting minutes.</p>

        <Banner kind="error">{error}</Banner>

        <form onSubmit={submit} className="stack">
          {mode === 'register' && (
            <div>
              <label>Full name</label>
              <input value={fullName} onChange={(e) => setFullName(e.target.value)} />
            </div>
          )}
          <div>
            <label>Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              autoComplete="username"
            />
          </div>
          <div>
            <label>Password {mode === 'register' && '(at least 10 characters)'}</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={mode === 'register' ? 10 : undefined}
              autoComplete={mode === 'register' ? 'new-password' : 'current-password'}
            />
          </div>
          {mode === 'register' && captchaRequired && (
            <div>
              <label>{captcha ? captcha.question : 'Loading challenge…'}</label>
              <input
                type="text"
                inputMode="numeric"
                value={captchaAnswer}
                onChange={(e) => setCaptchaAnswer(e.target.value)}
                required
                autoComplete="off"
              />
            </div>
          )}
          <button className="primary" type="submit" disabled={busy} style={{ width: '100%' }}>
            {busy ? 'Please wait…' : mode === 'login' ? 'Sign in' : 'Create account & sign in'}
          </button>
        </form>

        {demo?.available && mode === 'login' && (
          <>
            <div className="or-divider">or</div>
            <button
              type="button"
              disabled={busy}
              onClick={signInAsDemo}
              style={{ width: '100%' }}
            >
              Explore the demo →
            </button>
            <p className="muted" style={{ marginTop: 8, marginBottom: 0, textAlign: 'center' }}>
              Signs you in as <strong>{demo.email}</strong> ({demo.role}) with a
              sample meeting already waiting for review.
            </p>
          </>
        )}

        {demo && !demo.available && demo.reason?.includes('seed_demo') && (
          <p className="muted" style={{ marginTop: 14, marginBottom: 0 }}>
            No demo account yet — run <code>python scripts/seed_demo.py</code> to create one.
          </p>
        )}

        <p className="muted" style={{ marginTop: 14, marginBottom: 0 }}>
          {mode === 'login' ? 'No account yet? ' : 'Already registered? '}
          <button
            className="small"
            onClick={() => {
              setMode(mode === 'login' ? 'register' : 'login')
              setError('')
            }}
          >
            {mode === 'login' ? 'Register' : 'Sign in'}
          </button>
        </p>
      </div>
    </div>
  )
}
