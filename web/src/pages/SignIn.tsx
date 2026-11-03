// S1 Sign in (HANDOFF §1). Stub (Stage 18 PR 3c): shows which variant the guard chose, so it can be seen in a
// live run. The form and POST /api/session are PR 4.
export type SignInVariant = 'default' | 'expired' | 'unavailable'

const VARIANT_TEXT: Record<SignInVariant, string> = {
  default: 'default',
  expired: 'expired session',
  unavailable: 'sign-in service unavailable',
}

export default function SignIn({ variant }: { variant: SignInVariant }) {
  return (
    <main>
      <h1>Sign in</h1>
      <p>Stub page. Variant: {VARIANT_TEXT[variant]}.</p>
    </main>
  )
}
